"""
Run raw OpenStudio processing for the apartment_1970 baseline case.
"""

from pathlib import Path

from baseboard_diagnostics.openstudio.process_openstudio_export import (
    process_openstudio_raw_export,
)


RAW_PATH = Path("data/raw/openstudio/apartment_1970/raw_export.csv")
OUTPUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_long_new.csv"
)


def main():
    df = process_openstudio_raw_export(
        raw_path=RAW_PATH,
        output_path=OUTPUT_PATH,
        case_id="apartment_1970",
        archetype="apartment",
        construction_year=1970,
        simulation_year=2023,
    )

    print("Saved:", OUTPUT_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())


if __name__ == "__main__":
    main()
