"""
Calibrate persistence thresholds from healthy OpenStudio diagnosed windows.
"""

from pathlib import Path

from baseboard_diagnostics.utils.io import read_csv, write_csv
from baseboard_diagnostics.diagnosis.persistence import (
    calculate_diagnosis_fractions,
    calibrate_persistence_thresholds,
)


INPUT_PATH = Path(
    "data/diagnosis_outputs/openstudio/apartment_1970/"
    "apartment_baseboard_3h_windows_diagnosed_new.csv"
)

FRACTIONS_OUTPUT_PATH = Path(
    "data/reference_bands/openstudio/apartment_1970/"
    "normal_reference_baseboard_diagnosis_fractions_new.csv"
)

THRESHOLDS_OUTPUT_PATH = Path(
    "data/reference_bands/openstudio/apartment_1970/"
    "diagnosis_persistence_thresholds_new.csv"
)


def main():
    diagnosed = read_csv(INPUT_PATH)

    fractions = calculate_diagnosis_fractions(diagnosed)
    thresholds = calibrate_persistence_thresholds(
        fractions,
        quantile=0.95,
        minimum_floor=0.02,
    )

    write_csv(fractions, FRACTIONS_OUTPUT_PATH)
    write_csv(thresholds, THRESHOLDS_OUTPUT_PATH)

    print("Input:", INPUT_PATH)
    print("Fractions output:", FRACTIONS_OUTPUT_PATH)
    print("Thresholds output:", THRESHOLDS_OUTPUT_PATH)

    print()
    print("Diagnosis fractions shape:", fractions.shape)
    print(fractions.head())

    print()
    print("Persistence thresholds:")
    print(thresholds)


if __name__ == "__main__":
    main()
