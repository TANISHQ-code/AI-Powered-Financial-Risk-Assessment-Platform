import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample

from src.feature_engineering import prepare_model_dataset, get_model_feature_columns


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features),
        ],
        remainder='drop',
    )


def _build_classifier(model_type: str = 'logistic', n_estimators: int = 200):
    model_type = model_type.lower()
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
        )
    if model_type == 'xgboost':
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )
    if model_type == 'lightgbm':
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
    if model_type == 'logistic':
        return LogisticRegression(
            solver='liblinear',
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
        )
    raise ValueError(f"Unsupported model_type: {model_type}. Use 'logistic', 'random_forest', 'xgboost', or 'lightgbm'.")


def _split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def _balance_training_data(X: pd.DataFrame, y: pd.Series, method: str = 'oversample') -> tuple[pd.DataFrame, pd.Series]:
    if method != 'oversample':
        return X, y

    if y.nunique() <= 1:
        return X, y

    data = pd.concat([X, y], axis=1)
    target_name = y.name
    class_groups = [group for _, group in data.groupby(target_name)]
    max_count = max(group.shape[0] for group in class_groups)

    resampled_groups = []
    for _, group in data.groupby(target_name):
        if group.shape[0] == max_count:
            resampled_groups.append(group)
        else:
            resampled_groups.append(
                resample(group, replace=True, n_samples=max_count, random_state=42)
            )

    balanced = pd.concat(resampled_groups).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced.drop(columns=[target_name]), balanced[target_name]


def _evaluate_predictions(y_true: pd.Series, preds: pd.Series, probs: pd.Series) -> dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_true, probs),
        'confusion_matrix': confusion_matrix(y_true, preds).tolist(),
    }


def build_baseline_model(
    train_df: pd.DataFrame,
    target_col: str = 'TARGET',
    model_type: str = 'logistic',
    resample_method: str = 'oversample',
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
    X_train: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    y_test: pd.Series | None = None,
) -> tuple[Pipeline, dict[str, float], list[str]]:
    if target_col in train_df.columns:
        train_df = prepare_model_dataset(train_df, target_col=target_col)

    X = train_df.drop(columns=[target_col]) if target_col in train_df.columns else train_df.copy()
    y = train_df[target_col] if target_col in train_df.columns else pd.Series([], dtype='int64')

    if X_train is None or X_test is None or y_train is None or y_test is None:
        X_train, X_test, y_train, y_test = _split_dataset(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    preprocessor = _build_preprocessor(X)
    classifier = _build_classifier(model_type, n_estimators=n_estimators)

    model = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', classifier),
    ])

    X_train, y_train = _balance_training_data(X_train, y_train, method=resample_method)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        'model_type': model_type,
        'resample_method': resample_method,
        **_evaluate_predictions(y_test, preds, probs),
    }

    feature_columns = X.columns.tolist()
    return model, metrics, feature_columns


def compare_models(
    train_df: pd.DataFrame,
    target_col: str = 'TARGET',
    resample_method: str = 'oversample',
    model_types: list[str] | None = None,
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    if model_types is None:
        model_types = ['logistic', 'random_forest', 'xgboost']

    if target_col in train_df.columns:
        train_df = prepare_model_dataset(train_df, target_col=target_col)

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    X_train, X_test, y_train, y_test = _split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_bal, y_train_bal = _balance_training_data(X_train, y_train, method=resample_method)

    results: dict[str, dict[str, float]] = {}
    for model_type in model_types:
        _, metrics, _ = build_baseline_model(
            train_df,
            target_col=target_col,
            model_type=model_type,
            resample_method=resample_method,
            n_estimators=n_estimators,
            X_train=X_train_bal,
            X_test=X_test,
            y_train=y_train_bal,
            y_test=y_test,
        )
        results[model_type] = metrics
    return results


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return _evaluate_predictions(y_test, preds, probs)
