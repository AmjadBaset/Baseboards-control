"""
Legacy random-fault-case classifier using the same controller-signal-free
diagnostic logic as script 11.

This script is retained for the older random_fault_six_room case. It does not
use injected valve maximum-flow parameters or controller command signals as
diagnostic inputs. Measured valve position is classified against the
context-matched healthy normalized-flow-fraction band to obtain a relative
valve-effort label used only as supporting evidence.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from baseboard_diagnostics.features.timestep_features import add_timestep_features
from baseboard_diagnostics.features.window_features import create_window_features
from baseboard_diagnostics.diagnosis.feature_classification import classify_value
from baseboard_diagnostics.diagnosis.rules import apply_rules_to_labeled_windows
from baseboard_diagnostics.utils.io import write_csv


RAW_INPUT_PATH = Path(
    "data/raw/twin4build/random_fault_six_room/"
    "t4b_raw_timeseries.csv"
)

BANDS_PATH = Path(
    "data/reference_bands/openstudio/apartment_1970/"
    "apartment_baseboard_reference_bands_area_exposure_new.csv"
)

WINDOW_OUTPUT_PATH = Path(
    "data/processed/twin4build/random_fault_six_room/"
    "t4b_3h_window_features.csv"
)

LABELED_OUTPUT_PATH = Path(
    "data/processed/twin4build/random_fault_six_room/"
    "t4b_3h_windows_labeled_against_openstudio.csv"
)

DIAGNOSED_OUTPUT_PATH = Path(
    "data/processed/twin4build/random_fault_six_room/"
    "t4b_3h_windows_diagnosed_against_openstudio.csv"
)


WINDOW = "3h"
TIMESTEP_MINUTES = 15
SAMPLES_PER_WINDOW = int(
    pd.Timedelta(WINDOW)
    / pd.Timedelta(minutes=TIMESTEP_MINUTES)
)


NORMAL_FEATURES = [
    "comfort_violation_fraction",
    "overheating_fraction",
    "Q_density_mean",
    "m_dot_density_mean",
    "deltaT_water_mean",
    "flow_density_oscillation_index",
]


RELATIVE_VALVE_EFFORT_FEATURE = "valve_position_mean"
RELATIVE_VALVE_EFFORT_REFERENCE = "normalized_flow_fraction_mean"
RELATIVE_VALVE_EFFORT_LABEL = "relative_valve_effort_label"


MATCH_LEVELS = [
    [
        "construction_year",
        "zone_area_m2",
        "exposure_group",
        "T_out_bin",
        "period",
    ],
    [
        "construction_year",
        "exposure_group",
        "T_out_bin",
        "period",
    ],
    [
        "construction_year",
        "zone_area_m2",
        "T_out_bin",
        "period",
    ],
    [
        "construction_year",
        "T_out_bin",
        "period",
    ],
    [
        "construction_year",
        "T_out_bin",
    ],
    [
        "construction_year",
        "period",
    ],
    [
        "construction_year",
    ],
]


def add_t4b_window_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Intentionally do not derive m_dot_design from valve_max_flow because the
    # latter can be the injected restriction parameter.

    df = add_timestep_features(
        df,
        comfort_tolerance=0.5,
        heating_delta_t_threshold=0.8,
        heating_power_threshold=20.0,
    )

    windows = create_window_features(
        df,
        window=WINDOW,
        heating_active_fraction_threshold=0.25,
    )

    if "valve_position" not in df.columns:
        return windows

    extra_rows = []

    for keys, g in df.groupby(
        ["case_id", "zone", "baseboard"],
        dropna=False,
    ):
        case_id, zone, baseboard = keys
        g = g.sort_values(
            "timestamp"
        ).copy()

        g["window_id"] = (
            np.arange(len(g))
            // SAMPLES_PER_WINDOW
        )

        for _, w in g.groupby(
            "window_id"
        ):
            if len(w) != SAMPLES_PER_WINDOW:
                continue

            valve_position = pd.to_numeric(
                w["valve_position"],
                errors="coerce",
            )
            valve_diff = (
                valve_position
                .diff()
                .dropna()
            )

            extra_rows.append(
                {
                    "case_id": case_id,
                    "zone": zone,
                    "baseboard": baseboard,
                    "window_start": (
                        w["timestamp"].iloc[0]
                    ),
                    "valve_position_mean": (
                        valve_position.mean()
                    ),
                    "valve_position_max": (
                        valve_position.max()
                    ),
                    "valve_position_min": (
                        valve_position.min()
                    ),
                    "valve_position_range": (
                        valve_position.max()
                        - valve_position.min()
                    ),
                    "valve_position_std": (
                        valve_position.std()
                    ),
                    "valve_position_oscillation_index": (
                        valve_diff.abs().sum()
                    ),
                }
            )

    if not extra_rows:
        return windows

    extra = pd.DataFrame(
        extra_rows
    )

    windows["window_start"] = pd.to_datetime(
        windows["window_start"]
    )
    extra["window_start"] = pd.to_datetime(
        extra["window_start"]
    )

    return windows.merge(
        extra,
        on=[
            "case_id",
            "zone",
            "baseboard",
            "window_start",
        ],
        how="left",
    )


def summarize_bands_for_match_level(
    bands: pd.DataFrame,
    match_cols: list[str],
) -> pd.DataFrame:
    band_value_cols = [
        col
        for col in bands.columns
        if (
            col.endswith("_P10")
            or col.endswith("_P50")
            or col.endswith("_P90")
        )
    ]

    return (
        bands
        .groupby(
            match_cols,
            dropna=False,
        )[band_value_cols]
        .median()
        .reset_index()
    )


def classify_with_relaxed_bands(
    windows: pd.DataFrame,
    bands: pd.DataFrame,
) -> pd.DataFrame:
    remaining = windows.copy()
    remaining["_original_index"] = np.arange(
        len(remaining)
    )
    remaining["_match_level"] = pd.NA

    classified_parts = []

    for i, match_cols in enumerate(
        MATCH_LEVELS,
        start=1,
    ):
        if remaining.empty:
            break

        missing_window = [
            c
            for c in match_cols
            if c not in remaining.columns
        ]
        missing_bands = [
            c
            for c in match_cols
            if c not in bands.columns
        ]
        if missing_window or missing_bands:
            continue

        relaxed_bands = (
            summarize_bands_for_match_level(
                bands,
                match_cols,
            )
        )

        merged = remaining.merge(
            relaxed_bands,
            on=match_cols,
            how="left",
            suffixes=("", "_band"),
            indicator=True,
        )

        matched = merged[
            merged["_merge"] == "both"
        ].copy()

        unmatched = merged[
            merged["_merge"] == "left_only"
        ].copy()

        if not matched.empty:
            matched["_match_level"] = i
            matched["_match_columns"] = (
                " + ".join(match_cols)
            )
            classified_parts.append(
                matched.drop(
                    columns=["_merge"]
                )
            )

        original_cols = (
            list(windows.columns)
            + [
                "_original_index",
                "_match_level",
            ]
        )

        remaining = unmatched[
            original_cols
        ].copy()

    if remaining.empty:
        classified = pd.concat(
            classified_parts,
            ignore_index=True,
        )
    else:
        remaining["_match_level"] = "unmatched"
        remaining["_match_columns"] = "none"

        classified = pd.concat(
            classified_parts + [remaining],
            ignore_index=True,
        )

    for feature in NORMAL_FEATURES:
        p10_col = f"{feature}_P10"
        p90_col = f"{feature}_P90"
        label_col = f"{feature}_label"

        if (
            feature in classified.columns
            and p10_col in classified.columns
            and p90_col in classified.columns
        ):
            classified[label_col] = (
                classified.apply(
                    lambda row: classify_value(
                        row[feature],
                        row[p10_col],
                        row[p90_col],
                    ),
                    axis=1,
                )
            )
        else:
            classified[label_col] = "missing"

    # Context-relative valve effort. This does not use controller command and
    # does not impose an absolute low/high valve-opening threshold.
    valve_feature = RELATIVE_VALVE_EFFORT_FEATURE
    reference_feature = RELATIVE_VALVE_EFFORT_REFERENCE

    valve_p10_col = f"{reference_feature}_P10"
    valve_p90_col = f"{reference_feature}_P90"

    if (
        valve_feature in classified.columns
        and valve_p10_col in classified.columns
        and valve_p90_col in classified.columns
    ):
        classified[RELATIVE_VALVE_EFFORT_LABEL] = classified.apply(
            lambda row: classify_value(
                row[valve_feature],
                row[valve_p10_col],
                row[valve_p90_col],
            ),
            axis=1,
        )
    else:
        classified[RELATIVE_VALVE_EFFORT_LABEL] = "missing"

    return (
        classified
        .sort_values(
            "_original_index"
        )
        .drop(
            columns=["_original_index"]
        )
    )


def main():
    raw = pd.read_csv(
        RAW_INPUT_PATH
    )
    bands = pd.read_csv(
        BANDS_PATH
    )

    windows = add_t4b_window_features(
        raw
    )
    write_csv(
        windows,
        WINDOW_OUTPUT_PATH,
    )

    labeled = classify_with_relaxed_bands(
        windows,
        bands,
    )
    write_csv(
        labeled,
        LABELED_OUTPUT_PATH,
    )

    diagnosed = apply_rules_to_labeled_windows(
        labeled
    )
    write_csv(
        diagnosed,
        DIAGNOSED_OUTPUT_PATH,
    )

    print("Raw input:", RAW_INPUT_PATH)
    print("Bands:", BANDS_PATH)
    print("Window output:", WINDOW_OUTPUT_PATH)
    print("Labeled output:", LABELED_OUTPUT_PATH)
    print("Diagnosed output:", DIAGNOSED_OUTPUT_PATH)
    print()

    print("Match levels:")
    print(
        labeled["_match_columns"]
        .value_counts(dropna=False)
    )
    print()

    print("Relative valve effort labels:")
    if RELATIVE_VALVE_EFFORT_LABEL in labeled.columns:
        print(
            labeled
            .groupby("zone")[RELATIVE_VALVE_EFFORT_LABEL]
            .value_counts(dropna=False)
        )
    print()

    print("Diagnosis counts:")
    print(
        diagnosed
        .groupby("zone")["diagnosis_text"]
        .value_counts(dropna=False)
    )
    print()


if __name__ == "__main__":
    main()
