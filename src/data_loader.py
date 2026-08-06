from pathlib import Path
import pandas as pd


class HomeCreditDataLoader:
    """Load and validate raw Home Credit CSV files."""

    EXPECTED_FILES = [
        'application_train.csv',
        'application_test.csv',
        'bureau.csv',
        'bureau_balance.csv',
        'previous_application.csv',
        'installments_payments.csv',
        'POS_CASH_balance.csv',
        'credit_card_balance.csv',
    ]

    def __init__(self, data_dir: str | None = None) -> None:
        self.base_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / "raw"

    def load(self, filename: str) -> pd.DataFrame:
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return pd.read_csv(path)

    def load_all(self) -> dict[str, pd.DataFrame]:
        datasets: dict[str, pd.DataFrame] = {}
        for filename in self.EXPECTED_FILES:
            datasets[filename] = self.load(filename)
        return datasets

    def list_files(self) -> list[str]:
        return sorted([p.name for p in self.base_dir.glob("*.csv")])

    def describe_dataset(self, filename: str) -> dict:
        df = self.load(filename)
        return {
            "filename": filename,
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    def validate_raw_files(self) -> list[str]:
        missing = [f for f in self.EXPECTED_FILES if not (self.base_dir / f).exists()]
        return missing
