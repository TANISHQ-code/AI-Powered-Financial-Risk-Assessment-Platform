from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)


def save_model(model, filename: str) -> Path:
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    return path


def load_model(filename: str):
    path = MODEL_DIR / filename
    return joblib.load(path)
