import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException
import pandas as pd

from src.model_store import save_model, load_model
from src.feature_engineering import engineer_features, get_model_feature_columns
from src.explainability import explain_customer

MODEL_FILENAME = "finrisk_model.joblib"
METRICS_FILENAME = "training_metrics.json"
METADATA_FILENAME = "model_metadata.json"


def _get_metadata_path() -> Path:
    return Path(__file__).resolve().parent.parent / METADATA_FILENAME


def save_metadata(metadata: dict[str, Any]) -> str:
    metadata_path = _get_metadata_path()
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return str(metadata_path)


def load_metadata() -> dict[str, Any]:
    metadata_path = _get_metadata_path()

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def train_and_save(model_type: str = "xgboost") -> dict[str, Any]:
    # Import training pipeline only when training is explicitly requested.
    from src.pipeline import run_training_pipeline

    model, metrics, feature_columns = run_training_pipeline(
        model_type=model_type
    )

    save_model(model, MODEL_FILENAME)

    save_metadata({
        "model_type": model_type,
        "feature_columns": feature_columns,
        "target_col": "TARGET",
    })

    output_path = (
        Path(__file__).resolve().parent.parent / METRICS_FILENAME
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_trained_model():
    try:
        return load_model(MODEL_FILENAME)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Trained model missing"
        ) from exc


def _prepare_input_dataframe(
    input_data: dict[str, Any],
    feature_columns: list[str]
) -> pd.DataFrame:

    raw_df = pd.DataFrame([input_data])
    raw_df = raw_df.copy()
    raw_df = engineer_features(raw_df)

    for col in feature_columns:
        if col not in raw_df.columns:
            raw_df[col] = 0

    return raw_df[feature_columns]


def predict_risk(input_data: dict[str, Any]) -> dict[str, Any]:
    model = load_trained_model()
    metadata = load_metadata()

    feature_columns = metadata.get(
        "feature_columns",
        get_model_feature_columns()
    )

    X = _prepare_input_dataframe(
        input_data,
        feature_columns
    )

    score = float(
        model.predict_proba(X)[:, 1][0]
    )

    return {
        "risk_score": score,
        "risk_category": (
            "HIGH"
            if score > 0.65
            else "MEDIUM"
            if score > 0.4
            else "LOW"
        ),
    }


def explain_risk(input_data: dict[str, Any]) -> dict[str, Any]:
    model = load_trained_model()
    metadata = load_metadata()

    feature_columns = metadata.get(
        "feature_columns",
        get_model_feature_columns()
    )

    X = _prepare_input_dataframe(
        input_data,
        feature_columns
    )

    try:
        contributions = explain_customer(
            model,
            X,
            idx=0
        )

        explanation = [
            {
                "feature": feature,
                "value": float(value)
            }
            for feature, value in contributions
        ]

    except Exception as exc:
        explanation = [
            {
                "feature": "explainability_error",
                "value": str(exc)
            }
        ]

    score = float(
        model.predict_proba(X)[:, 1][0]
    )

    return {
        "risk_score": score,
        "risk_category": (
            "HIGH"
            if score > 0.65
            else "MEDIUM"
            if score > 0.4
            else "LOW"
        ),
        "explanation": explanation,
    }