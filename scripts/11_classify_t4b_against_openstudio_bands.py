"""
Classify Twin4Build diagnostic windows against context-matched healthy bands.

The diagnostic pipeline intentionally does not use controller command signals.
Measured valve position is summarized only to characterize valve movement and
persistent limited-travel behaviour. Measured valve position is also
classified relative to the context-matched healthy normalized-flow-fraction
band to obtain a relative valve-effort label. This comparison is used only as
supporting evidence; no absolute valve-opening threshold is used.

Injected fault metadata and valve maximum-flow fault parameters are not used as
diagnostic inputs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseboard_diagnostics.features.timestep_features import add_timestep_features
from baseboard_diagnostics.features.window_features import create_window_features
from baseboard_diagnostics.diagnosis.feature_classification import classify_value
from baseboard_diagnostics.diagnosis.rules import apply_rules_to_labeled_windows
from baseboard_diagnostics.utils.io import write_csv


WINDOW = "3h"
TIMESTEP_MINUTES = 15
SAMPLES_PER_WINDOW = int(
    pd.Timedelta(WINDOW)
    / pd.Timedelta(minutes=TIMESTEP_MINUTES)
)


def paths(case_id: str) -> dict[str, Path]:
    base_raw = Path(f"data/raw/twin4build/{case_id}")
    base_processed = Path(f"data/processed/twin4build/{case_id}")

    return {
        "raw": base_raw / "t4b_raw_timeseries.csv",
        "bands": Path(
            "data/reference_bands/openstudio/apartment_1970/"
            "apartment_baseboard_reference_bands_area_exposure_new.csv"
        ),
        "features": base_processed / "t4b_3h_window_features.csv",
        "labeled": (
            base_processed
            / "t4b_3h_windows_labeled_against_openstudio.csv"
        ),
        "diagnosed": (
            base_processed
            / "t4b_3h_windows_diagnosed_against_openstudio.csv"
        ),
    }


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


FORBIDDEN_OPERATIONAL_COLUMNS = {
    "fault_type", "fault_severity", "fault_start", "fault_start_step",
    "restriction_factor", "fixed_valve_position", "fixed_valve_quantile",
    "fixed_valve_regime", "valve_upper_limit", "valve_lower_limit",
    "valve_limit_quantile", "valve_binding_fraction", "valve_binding_count",
    "valve_mean_absolute_position_change", "valve_post_fault_step_count",
    "valve_fault_activated", "controller_signal", "valve_max_flow",
    "healthy_valve_max_flow_kg_s",
}


def validate_blind_raw(raw: pd.DataFrame) -> None:
    forbidden = sorted(FORBIDDEN_OPERATIONAL_COLUMNS.intersection(raw.columns))
    if forbidden:
        raise ValueError(
            "Operational CSV contains injected-fault/controller information: "
            f"{forbidden}. Use blind simulation output only."
        )
    required = {"timestamp", "case_id", "zone", "baseboard"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Missing operational columns: {missing}")
    timestamps = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    if timestamps.isna().any():
        raise ValueError("Operational CSV has null timestamps.")
    keys = raw[["case_id", "zone", "baseboard"]].copy()
    keys["timestamp"] = timestamps
    if keys.duplicated().any():
        raise ValueError("Duplicate timestamps within a case/zone/baseboard.")
    # Windowing relies on 15-minute samples with no gaps. Otherwise its
    # position-based groups would silently span more than 3 hours.
    keys = keys.sort_values(["case_id", "zone", "baseboard", "timestamp"])
    delta = keys.groupby(["case_id", "zone", "baseboard"])["timestamp"].diff()
    bad = delta.notna() & delta.ne(pd.Timedelta(minutes=TIMESTEP_MINUTES))
    if bad.any():
        raise ValueError(
            f"Expected continuous {TIMESTEP_MINUTES}-minute samples. "
            f"Found {int(bad.sum())} missing/irregular steps."
        )


def add_t4b_window_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    validate_blind_raw(raw)
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Do NOT derive m_dot_design from valve_max_flow here.
    # In fault-injection cases valve_max_flow can be the parameter used to
    # create a restriction and therefore must not leak into diagnosis.

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

    # Valve-position features are optional. The remaining diagnostic rules
    # continue to work if valve position is unavailable, but specific valve
    # fault identification will then be less specific.
    if "valve_position" not in df.columns:
        return windows

    extra_rows = []

    for keys, g in df.groupby(
        ["case_id", "zone", "baseboard"],
        dropna=False,
    ):
        case_id, zone, baseboard = keys
        g = g.sort_values("timestamp").copy()
        g["window_id"] = (
            np.arange(len(g))
            // SAMPLES_PER_WINDOW
        )

        for _, w in g.groupby("window_id"):
            # Keep valve statistics aligned with the complete windows retained
            # by create_window_features().
            if len(w) != SAMPLES_PER_WINDOW:
                continue

            valve_position = pd.to_numeric(
                w["valve_position"],
                errors="coerce",
            )
            valve_diff = valve_position.diff().dropna()

            row = {
                "case_id": case_id,
                "zone": zone,
                "baseboard": baseboard,
                "window_start": w["timestamp"].iloc[0],
                "valve_position_mean": valve_position.mean(),
                "valve_position_max": valve_position.max(),
                "valve_position_min": valve_position.min(),
                "valve_position_range": (
                    valve_position.max()
                    - valve_position.min()
                ),
                "valve_position_std": valve_position.std(),
                "valve_position_oscillation_index": (
                    valve_diff.abs().sum()
                ),
            }
            extra_rows.append(row)

    if not extra_rows:
        return windows

    extra = pd.DataFrame(extra_rows)

    windows["window_start"] = pd.to_datetime(
        windows["window_start"]
    )
    extra["window_start"] = pd.to_datetime(
        extra["window_start"]
    )

    windows = windows.merge(
        extra,
        on=[
            "case_id",
            "zone",
            "baseboard",
            "window_start",
        ],
        how="left",
        validate="one_to_one",
    )
    if windows["valve_position_mean"].isna().any():
        raise ValueError("Valve-window alignment failed: unmatched window rows.")

    return windows


def summarize_bands_for_match_level(
    bands: pd.DataFrame,
    match_cols: list[str],
) -> pd.DataFrame:
    """
    Collapse healthy reference bands for a relaxed match level.

    If room-specific context is relaxed away, several healthy groups may match.
    P10/P50/P90 columns are aggregated using the median across those groups.
    """
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

        relaxed_bands = summarize_bands_for_match_level(
            bands,
            match_cols,
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
                matched.drop(columns=["_merge"])
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

    n_unmatched = int(classified["_match_level"].eq("unmatched").sum())
    if n_unmatched:
        raise ValueError(
            f"{n_unmatched} windows have no healthy-reference match. "
            "Expand reference coverage or document an explicit exclusion."
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
            classified[label_col] = classified.apply(
                lambda row: classify_value(
                    row[feature],
                    row[p10_col],
                    row[p90_col],
                ),
                axis=1,
            )
        else:
            classified[label_col] = "missing"

    # Context-relative valve effort:
    # compare measured valve position (0-1) with the healthy
    # normalized-flow-fraction P10/P90 band for the matched operating context.
    # This is a supporting effort proxy, not an absolute valve-opening rule.
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

    classified = (
        classified
        .sort_values("_original_index")
        .drop(columns=["_original_index"])
    )

    return classified


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        default="two_room_restricted_flow",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    p = paths(args.case_id)

    raw = pd.read_csv(p["raw"])
    validate_blind_raw(raw)
    bands = pd.read_csv(p["bands"])

    windows = add_t4b_window_features(raw)
    write_csv(
        windows,
        p["features"],
    )

    labeled = classify_with_relaxed_bands(
        windows,
        bands,
    )
    write_csv(
        labeled,
        p["labeled"],
    )

    diagnosed = apply_rules_to_labeled_windows(
        labeled
    )
    write_csv(
        diagnosed,
        p["diagnosed"],
    )

    print("Raw input:", p["raw"])
    print("Bands:", p["bands"])
    print("Window output:", p["features"])
    print("Labeled output:", p["labeled"])
    print("Diagnosed output:", p["diagnosed"])
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

    summary_cols = [
        "valve_position_mean",
        "valve_position_range",
        "normalized_flow_fraction_mean_P10",
        "normalized_flow_fraction_mean_P50",
        "normalized_flow_fraction_mean_P90",
        "Q_density_mean",
        "m_dot_density_mean",
        "deltaT_water_mean",
        "comfort_violation_fraction",
        "overheating_fraction",
    ]

    available_summary_cols = [
        c
        for c in summary_cols
        if c in diagnosed.columns
    ]

    print("Summary:")
    print(
        diagnosed
        .groupby("zone")[
            available_summary_cols
        ]
        .mean()
    )


if __name__ == "__main__":
    main()
