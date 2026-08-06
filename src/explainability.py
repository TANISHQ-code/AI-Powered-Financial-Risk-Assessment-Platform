import shap
import pandas as pd


def _get_transformed_feature_names(preprocessor) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def _get_feature_importances(model, feature_names: list[str]) -> list[tuple[str, float]]:
    classifier = model.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = abs(classifier.coef_[0])
    else:
        raise RuntimeError('Model does not expose feature importance or coefficients.')

    if len(feature_names) == len(importances):
        return list(zip(feature_names, importances.tolist()))

    if len(feature_names) < len(importances):
        importances = importances[: len(feature_names)]
    else:
        importances = list(importances) + [0.0] * (len(feature_names) - len(importances))

    return list(zip(feature_names, importances))


def compute_shap_values(model, X: pd.DataFrame):
    transformed_X = model.named_steps['preprocess'].transform(X)
    explainer = shap.TreeExplainer(
        model.named_steps['classifier'],
        feature_perturbation='tree_path_dependent',
    )
    shap_values = explainer(transformed_X)
    return explainer, shap_values


def explain_customer(model, X: pd.DataFrame, idx: int):
    preprocessor = model.named_steps['preprocess']
    try:
        explainer, shap_values = compute_shap_values(model, X)
        shap_row = shap_values[idx]
        contributions = sorted(
            zip(X.columns.tolist(), shap_row.values[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return contributions[:10]
    except Exception:
        feature_names = _get_transformed_feature_names(preprocessor)
        if not feature_names:
            raise
        contributions = _get_feature_importances(model, feature_names)
        contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        return contributions[:10]
