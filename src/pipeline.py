import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import HomeCreditDataLoader
from src.feature_engineering import engineer_features
from src.train_model import build_baseline_model, compare_models


def run_training_pipeline(data_dir: str | None = None, target_col: str = 'TARGET', model_type: str = 'random_forest') -> tuple[object, dict[str, float], list[str]]:
    loader = HomeCreditDataLoader(data_dir)
    missing = loader.validate_raw_files()
    if missing:
        raise FileNotFoundError(
            'Missing required raw dataset files: ' + ', '.join(missing)
        )

    train_df = loader.load('application_train.csv')
    train_df = engineer_features(train_df)

    model, metrics, features = build_baseline_model(
        train_df,
        target_col=target_col,
        model_type=model_type,
        resample_method='oversample',
        n_estimators=50,
    )
    return model, metrics, features


def compare_model_types(data_dir: str | None = None, target_col: str = 'TARGET', resample_method: str = 'oversample') -> dict[str, dict[str, float]]:
    loader = HomeCreditDataLoader(data_dir)
    missing = loader.validate_raw_files()
    if missing:
        raise FileNotFoundError(
            'Missing required raw dataset files: ' + ', '.join(missing)
        )

    train_df = loader.load('application_train.csv')
    train_df = engineer_features(train_df)

    return compare_models(
        train_df,
        target_col=target_col,
        resample_method=resample_method,
    )


if __name__ == '__main__':
    results = compare_model_types()
    print('Model comparison results:')
    for model_type, metrics in results.items():
        print(f'\n{model_type}')
        for name, value in metrics.items():
            print(f'- {name}: {value}')
