"""
Add timestep-level diagnostic features to the processed OpenStudio long file.

Input:
    data/processed/openstudio/apartment_1970/apartment_baseboard_zone_timeseries_long_new.csv

Output:
    data/processed/openstudio/apartment_1970/apartment_baseboard_zone_timeseries_clean_area_new.csv
"""

from pathlib import Path

from baseboard_diagnostics.utils.io import read_csv, write_csv, ensure_datetime
from baseboard_diagnostics.features.timestep_features import add_timestep_features


INPUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_long_new.csv"
)

OUTPUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_clean_area_new.csv"
)


def main():
    df = read_csv(INPUT_PATH)
    df = ensure_datetime(df, column="timestamp")

    df = add_timestep_features(
        df,
        comfort_tolerance=0.5,
        heating_delta_t_threshold=0.8,
        heating_power_threshold=20.0,
    )

    write_csv(df, OUTPUT_PATH)

    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Shape:", df.shape)
    print("Columns:")
    for col in df.columns:
        print(" -", col)

    print()
    print("Preview:")
    print(df.head())


if __name__ == "__main__":
    main()
