"""
Apply diagnostic rules to labeled OpenStudio apartment_1970 windows.

This is still a validation step on the healthy OpenStudio baseline.
The output will later be used to calibrate persistence thresholds.
"""

from pathlib import Path

from baseboard_diagnostics.utils.io import read_csv, write_csv
from baseboard_diagnostics.diagnosis.rules import apply_rules_to_labeled_windows


INPUT_PATH = Path(
    "data/diagnosis_outputs/openstudio/apartment_1970/"
    "apartment_baseboard_3h_windows_labeled_new.csv"
)

OUTPUT_PATH = Path(
    "data/diagnosis_outputs/openstudio/apartment_1970/"
    "apartment_baseboard_3h_windows_diagnosed_new.csv"
)


def main():
    labeled = read_csv(INPUT_PATH)

    diagnosed = apply_rules_to_labeled_windows(labeled)

    write_csv(diagnosed, OUTPUT_PATH)

    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Shape:", diagnosed.shape)

    print()
    print("Diagnosis text counts:")
    print(diagnosed["diagnosis_text"].value_counts().head(30))

    print()
    print("Diagnosis fractions:")
    diagnosis_cols = [
        col for col in diagnosed.columns
        if col.startswith("hydraulic_")
        or col.startswith("insufficient_")
        or col.startswith("possible_")
    ]

    for col in diagnosis_cols:
        print(f"{col}: {diagnosed[col].mean():.4f}")


if __name__ == "__main__":
    main()
