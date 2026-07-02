"""
Label OpenStudio apartment_1970 windows against the OpenStudio reference bands.

This is mainly a validation step. Since the windows are compared against bands
derived from the same healthy baseline, most labels should be NORMAL.
"""

from pathlib import Path

from baseboard_diagnostics.utils.io import read_csv, write_csv
from baseboard_diagnostics.diagnosis.feature_classification import (
    classify_features_against_bands,
    DEFAULT_FEATURES_TO_CLASSIFY,
)


WINDOWS_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_3h_window_features_area_tol_new.csv"
)

BANDS_PATH = Path(
    "data/reference_bands/openstudio/apartment_1970/"
    "apartment_baseboard_reference_bands_area_normalized_new.csv"
)

OUTPUT_PATH = Path(
    "data/diagnosis_outputs/openstudio/apartment_1970/"
    "apartment_baseboard_3h_windows_labeled_new.csv"
)


def main():
    windows = read_csv(WINDOWS_PATH)
    bands = read_csv(BANDS_PATH)

    # Label only valid heating windows for now.
    windows = windows[windows["valid_heating_window"] == True].copy()

    labeled = classify_features_against_bands(
        windows=windows,
        bands=bands,
    )

    write_csv(labeled, OUTPUT_PATH)

    print("Windows:", WINDOWS_PATH)
    print("Bands:", BANDS_PATH)
    print("Output:", OUTPUT_PATH)
    print("Shape:", labeled.shape)

    print()
    print("Label counts:")
    for feature in DEFAULT_FEATURES_TO_CLASSIFY:
        label_col = f"{feature}_label"
        print()
        print(label_col)
        print(labeled[label_col].value_counts(dropna=False))


if __name__ == "__main__":
    main()
