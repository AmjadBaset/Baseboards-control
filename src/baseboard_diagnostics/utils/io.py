"""
Input / Output helper functions.
"""

from pathlib import Path
import pandas as pd


def read_csv(path):
    """
    Read a CSV file into a pandas DataFrame.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def write_csv(df, path):
    """
    Save DataFrame as CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)


def ensure_datetime(df, column="timestamp"):
    """
    Convert timestamp column to datetime.
    """
    if column in df.columns:
        df[column] = pd.to_datetime(df[column])

    return df
