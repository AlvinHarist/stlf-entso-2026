"""
Data loading and validation utilities.

Provides functions to load the ENTSO-E / Open-Meteo dataset from CSV or
Parquet, parse timestamps, and validate that the resulting DataFrame has
a proper hourly time index with no gaps or duplicates.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────

def load_dataset(
    path: str | Path,
    timestamp_col: str = "utc_timestamp",
) -> pd.DataFrame:
    """Load a CSV or Parquet dataset and prepare it for time-series work.

    Steps performed:
    1. Read file (CSV or Parquet, detected by extension).
    2. Parse *timestamp_col* as UTC datetime and set it as index.
    3. Sort by timestamp (ascending / chronological).
    4. Remove duplicate timestamps (keep first).
    5. Validate that the index is monotonically increasing.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the data file.
    timestamp_col : str
        Name of the column containing timestamps.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by a ``DatetimeIndex`` in UTC.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *timestamp_col* is missing from the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # 1. Read ---------------------------------------------------------
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # 2. Parse timestamp -----------------------------------------------
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.set_index(timestamp_col)

    # 3. Sort chronologically ------------------------------------------
    df = df.sort_index()

    # 4. Remove duplicate timestamps -----------------------------------
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        print(f"[load_dataset] Removed {n_dups} duplicate timestamp(s).")
        df = df[~df.index.duplicated(keep="first")]

    # 5. Validate monotonic --------------------------------------------
    if not df.index.is_monotonic_increasing:
        raise RuntimeError(
            "Index is not monotonically increasing after sorting. "
            "This should not happen — please inspect the raw file."
        )

    return df


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────

def validate_hourly_index(df: pd.DataFrame) -> dict:
    """Validate that a DatetimeIndex follows an hourly frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``DatetimeIndex``.

    Returns
    -------
    dict
        Diagnostic report with keys:
        - ``start``:  first timestamp
        - ``end``:    last timestamp
        - ``n_rows``: number of rows
        - ``expected_rows``: number of rows if perfectly hourly
        - ``n_missing``: number of missing hourly slots
        - ``missing_timestamps``: list of missing timestamps (capped at 50)
        - ``n_duplicates``: number of duplicate timestamps
        - ``is_valid``: True if no gaps and no duplicates
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    start = df.index.min()
    end = df.index.max()

    expected_index = pd.date_range(start=start, end=end, freq="h")
    missing = expected_index.difference(df.index)
    duplicates = df.index.duplicated().sum()

    report = {
        "start": str(start),
        "end": str(end),
        "n_rows": len(df),
        "expected_rows": len(expected_index),
        "n_missing": len(missing),
        "missing_timestamps": [str(t) for t in missing[:50]],
        "n_duplicates": duplicates,
        "is_valid": len(missing) == 0 and duplicates == 0,
    }
    return report


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the dataset.

    Includes shape, dtypes, timestamp range, missing-value counts, and
    basic descriptive statistics.
    """
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape:  {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Index:  {df.index.min()}  -->  {df.index.max()}")
    print()

    print("Columns & dtypes:")
    for col in df.columns:
        print(f"  • {col:45s}  {df[col].dtype}")
    print()

    na = df.isna().sum()
    if na.sum() == 0:
        print("Missing values: NONE ✓")
    else:
        print("Missing values:")
        for col, count in na.items():
            if count > 0:
                print(f"  • {col}: {count}")
    print()

    print("Descriptive statistics:")
    print(df.describe().to_string())
    print("=" * 60)
