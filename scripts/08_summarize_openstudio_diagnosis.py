"""
Create final baseboard-level diagnosis summary for the OpenStudio baseline.

Since this is the healthy OpenStudio baseline, most/all baseboards should have
no persistent issue after applying calibrated thresholds.
"""

from pathlib import Path

from baseboard_diagnostics.utils.io import read_csv, write_csv
from baseboard_diagnostics.diagnosis.summary import summarize_persistent_diagnoses


DIAGNOSED_WINDOWS_PATH = Path(
    "data/diagnosis_outputs/openstudio/apartment_1970/"
    "apartment_baseboard_3h_windows_diagnosed_new.csv"
)

THRESHOLDS_PATH = Path(
    "data/reference_bands/openstudio/apartment_1970/"
    "diagnosis_persistence_thresholds_new.csv"
)

OUTPUT_PATH = Path(
    "data/diagnosis_outputs/openstudio/apartment_1970/"
    "apartment_baseboard_final_diagnosis_summary_new.csv"
)


def main():
    diagnosed = read_csv(DIAGNOSED_WINDOWS_PATH)
    thresholds = read_csv(THRESHOLDS_PATH)

    summary = summarize_persistent_diagnoses(
        diagnosed_windows=diagnosed,
        persistence_thresholds=thresholds,
        minimum_exceedance_margin=0.005,
    )

    write_csv(summary, OUTPUT_PATH)

    print("Diagnosed windows:", DIAGNOSED_WINDOWS_PATH)
    print("Thresholds:", THRESHOLDS_PATH)
    print("Output:", OUTPUT_PATH)
    print("Shape:", summary.shape)

    print()
    print("Persistent issue counts:")
    print(summary["has_persistent_issue"].value_counts())

    print()
    print("Dominant diagnosis counts:")
    print(summary["dominant_diagnosis"].value_counts())

    print()
    print("Preview:")
    print(summary[
        [
            "zone",
            "baseboard",
            "has_persistent_issue",
            "dominant_diagnosis",
            "dominant_fraction",
            "dominant_threshold",
            "dominant_exceedance",
        ]
    ])


if __name__ == "__main__":
    main()
