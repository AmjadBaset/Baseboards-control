"""
Inspect the raw OpenStudio/EnergyPlus CSV export.

This script is only for checking the raw file before building the full pipeline.
"""

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/openstudio/apartment_1970/raw_export.csv")


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH, nrows=20)

    print("Raw file:", RAW_PATH)
    print("Preview shape:", df.shape)
    print()
    print("Columns:")
    for i, col in enumerate(df.columns):
        print(f"{i:03d}: {col}")

    print()
    print("First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
