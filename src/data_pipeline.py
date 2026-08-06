import pandas as pd
from pathlib import Path
from typing import Optional

RAW_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw'
PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
PROCESSED_DIR.mkdir(exist_ok=True)


class DataPipeline:
    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_DIR

    def load_csv(self, filename: str) -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing raw file: {path}')
        return pd.read_csv(path)

    def save_processed(self, df: pd.DataFrame, filename: str) -> Path:
        path = PROCESSED_DIR / filename
        df.to_parquet(path, index=False)
        return path

    def load_processed(self, filename: str) -> pd.DataFrame:
        path = PROCESSED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing processed file: {path}')
        return pd.read_parquet(path)

    def build_base_application_data(self) -> pd.DataFrame:
        application = self.load_csv('application_train.csv')
        application_test = self.load_csv('application_test.csv')
        if 'TARGET' in application_test.columns:
            application_test = application_test.drop(columns=['TARGET'])
        return application
