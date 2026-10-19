"""Loading, validating and describing CSV datasets."""
import io
from pathlib import Path

import pandas as pd

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 200_000


def _read(source) -> pd.DataFrame:
    try:
        df = pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc
    if df.shape[1] == 0 or len(df) == 0:
        raise ValueError("The CSV file contains no data rows.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"The CSV has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_csv_path(path) -> pd.DataFrame:
    return _read(path)


def load_csv_bytes(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    return _read(io.BytesIO(data))


def _kind(series: pd.Series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "text"


def column_overview(df: pd.DataFrame) -> str:
    """One-line schema used in the system prompt, e.g. 'name (text), salary (numeric)'."""
    return ", ".join(f"{c} ({_kind(df[c])})" for c in df.columns)


class DatasetStore:
    """In-memory registry of named datasets."""

    def __init__(self):
        self._data = {}

    def add(self, name: str, df: pd.DataFrame) -> None:
        self._data[name] = df

    def get(self, name: str) -> pd.DataFrame:
        if name not in self._data:
            raise KeyError(name)
        return self._data[name]

    def names(self):
        return sorted(self._data)

    def summary(self, name: str) -> dict:
        df = self.get(name)
        return {
            "name": name,
            "rows": len(df),
            "columns": [{"name": c, "dtype": _kind(df[c])} for c in df.columns],
        }


def load_samples(store: DatasetStore, data_dir) -> None:
    for path in sorted(Path(data_dir).glob("*.csv")):
        store.add(path.stem, load_csv_path(path))
