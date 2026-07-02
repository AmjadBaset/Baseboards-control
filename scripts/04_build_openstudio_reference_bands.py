"""
Build OpenStudio reference bands for the apartment_1970 baseline.
"""

from pathlib import Path

from baseboard_diagnostics.utils.io import read_csv, write_csv
from baseboard_diagnostics.bands.build_reference_bands import build_reference_bands


INPUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_3h_window_features_area_tol_new.csv"
)

OUTPUT_PATH = Path(
    "data/reference_bands/openstudio/apartment_1970/"
    "apartment_baseboard_reference_bands_area_normalized_new.csv"
)


def main():
    windows = read_csv(INPUT_PATH)

    bands = build_reference_bands(
        windows,
        heating_windows_only=True,
    )

    write_csv(bands, OUTPUT_PATH)

    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Shape:", bands.shape)
    print("Columns:")
    for col in bands.columns:
        print(" -", col)

    print()
    print("Preview:")
    print(bands.head())


if __name__ == "__main__":
    main()
