"""
Create 3-hour OpenStudio window features for the apartment_1970 baseline.
"""

from pathlib import Path

from baseboard_diagnostics.utils.io import read_csv, write_csv, ensure_datetime
from baseboard_diagnostics.features.window_features import create_window_features


INPUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_clean_area_new.csv"
)

OUTPUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_3h_window_features_area_tol_new.csv"
)


def main():
    df = read_csv(INPUT_PATH)
    df = ensure_datetime(df, column="timestamp")

    windows = create_window_features(
        df,
        window="3h",
        heating_active_fraction_threshold=0.25,
    )

    write_csv(windows, OUTPUT_PATH)

    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Shape:", windows.shape)
    print("Columns:")
    for col in windows.columns:
        print(" -", col)

    print()
    print("Preview:")
    print(windows.head())


if __name__ == "__main__":
    main()
