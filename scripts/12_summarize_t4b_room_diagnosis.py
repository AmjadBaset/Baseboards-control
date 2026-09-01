"""
Summarize window-level Twin4Build/OpenStudio diagnoses into one room-level table.

Important architecture:
- Diagnostic reasoning belongs in rules.py and script 11.
- This script only aggregates window-level outputs.
- The injected fault_type is kept only as ground-truth metadata for validation/reporting.
  It is never used to decide the diagnosis.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


NORMAL_DIAGNOSIS = "normal_or_no_rule_triggered"

# Ignore one-window / numerical artifacts when choosing the dominant room diagnosis.
MIN_DOMINANT_RULE_FRACTION = 0.01


FAMILY_BY_DIAGNOSIS = {
    # Restricted / hydraulic
    "hydraulic_delivery_deficit": "restricted_or_hydraulic",
    "hydraulic_abnormality_low_flow_high_deltaT": "restricted_or_hydraulic",
    "hydraulic_limitation_with_comfort_impact": "restricted_or_hydraulic",
    "hydraulic_delivery_deficit_with_controller_compensation": "restricted_or_hydraulic",
    "possible_compensated_hydraulic_restriction": "restricted_or_hydraulic",
    "compensated_hydraulic_restriction_low_flow": "restricted_or_hydraulic",
    "degraded_hydraulic_compensation": "restricted_or_hydraulic",
    "control_saturation_with_unmet_heat_demand": "restricted_or_hydraulic",
    "possible_flow_or_control_instability": "restricted_or_hydraulic",

    # Stuck valve behavior
    "possible_valve_leakage_or_stuck_open": "stuck_open_or_leakage",
    "possible_partially_closed_or_stuck_low_valve": "stuck_low_valve",

    # Heat output
    "insufficient_heat_output": "insufficient_heat_output",
    "insufficient_heat_output_despite_high_controller_effort": "insufficient_heat_output",
    "possible_emitter_undersizing_or_high_load": "insufficient_heat_output",

    # Oversupply
    "possible_oversupply_or_overheating": "oversupply_or_overheating",

    # Fallback from rules.py
    "comfort_violation_without_matched_diagnostic_rule": "comfort_violation_unclassified",
}


DIAGNOSIS_PRIORITY = [
    "hydraulic_limitation_with_comfort_impact",
    "possible_valve_leakage_or_stuck_open",
    "possible_partially_closed_or_stuck_low_valve",
    "hydraulic_delivery_deficit_with_controller_compensation",
    "control_saturation_with_unmet_heat_demand",
    "insufficient_heat_output_despite_high_controller_effort",
    "hydraulic_abnormality_low_flow_high_deltaT",
    "hydraulic_delivery_deficit",
    "insufficient_heat_output",
    "possible_emitter_undersizing_or_high_load",
    "possible_oversupply_or_overheating",
    "possible_compensated_hydraulic_restriction",
    "compensated_hydraulic_restriction_low_flow",
    "degraded_hydraulic_compensation",
    "possible_flow_or_control_instability",
    "comfort_violation_without_matched_diagnostic_rule",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        default="two_room_restricted_flow",
        help="Case id under data/raw/twin4build and data/processed/twin4build.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional diagnosed 3-hour window CSV path.",
    )
    parser.add_argument(
        "--raw",
        default=None,
        help="Optional raw Twin4Build time-series CSV path.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional room-level summary CSV path.",
    )
    return parser.parse_args()


def severity_from_comfort_violation(comfort_violation: float) -> str:
    """
    Room severity is based only on comfort violation fraction.

    Any persistent comfort violation above numerical noise is a warning.
    Large comfort violation is a fault.
    """
    if comfort_violation >= 0.20:
        return "fault"
    if comfort_violation > 0.001:
        return "warning"
    return "normal"


def parse_diagnosis_text(value) -> list[str]:
    """
    Convert diagnosis_text / diagnoses cell into a clean list of diagnosis strings.
    Handles:
    - normal_or_no_rule_triggered
    - semicolon-separated strings
    - comma-separated strings
    - Python-list-like strings from pandas CSV export
    """
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()

        if not text or text == NORMAL_DIAGNOSIS:
            return []

        raw_items = None

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    raw_items = parsed
            except (SyntaxError, ValueError):
                raw_items = None

        if raw_items is None:
            raw_items = []
            for semi_part in text.split(";"):
                raw_items.extend(semi_part.split(","))

    diagnoses = []
    for item in raw_items:
        d = str(item).strip().strip("'").strip('"')
        if d and d != NORMAL_DIAGNOSIS and d != "normal":
            diagnoses.append(d)

    return diagnoses


def diagnosis_fractions(g: pd.DataFrame) -> dict[str, float]:
    """
    Fraction of room windows in which each diagnosis appears.
    """
    counts: dict[str, int] = {}
    n = max(len(g), 1)

    source_col = "diagnosis_text"
    if source_col not in g.columns and "diagnoses" in g.columns:
        source_col = "diagnoses"

    for value in g[source_col].fillna(NORMAL_DIAGNOSIS):
        for diagnosis in set(parse_diagnosis_text(value)):
            counts[diagnosis] = counts.get(diagnosis, 0) + 1

    return {diagnosis: count / n for diagnosis, count in counts.items()}


def choose_main_diagnosis(fractions: dict[str, float], severity: str) -> tuple[str, float]:
    """
    Choose the dominant room-level diagnosis from script-11 window diagnoses.

    This is aggregation, not diagnostic inference:
    - use only diagnoses already emitted by rules.py/script 11
    - ignore tiny one-window artifacts
    - use priority only to break close/ambiguous ties
    """
    significant = {
        diagnosis: fraction
        for diagnosis, fraction in fractions.items()
        if fraction >= MIN_DOMINANT_RULE_FRACTION
    }

    if not significant:
        if severity in {"watch", "warning", "fault"}:
            return "comfort_violation_without_matched_diagnostic_rule", 0.0
        return NORMAL_DIAGNOSIS, 0.0

    priority_rank = {
        diagnosis: i for i, diagnosis in enumerate(DIAGNOSIS_PRIORITY)
    }

    def sort_key(item):
        diagnosis, fraction = item
        return (
            -fraction,
            priority_rank.get(diagnosis, len(priority_rank)),
            diagnosis,
        )

    diagnosis, fraction = sorted(significant.items(), key=sort_key)[0]
    return diagnosis, fraction


def family_from_diagnosis(main_diagnosis: str) -> str:
    if main_diagnosis in {NORMAL_DIAGNOSIS, "normal"}:
        return "normal"
    return FAMILY_BY_DIAGNOSIS.get(main_diagnosis, "comfort_violation_unclassified")


def read_raw_metadata(raw_path: Path) -> pd.DataFrame:
    """
    Read optional room metadata from the raw simulation file.

    fault_type is only ground-truth metadata for validation/reporting.
    It is not used in any diagnostic decision.
    """
    if not raw_path.exists():
        return pd.DataFrame(columns=["zone", "fault_type"])

    header = pd.read_csv(raw_path, nrows=0)
    possible_cols = [
        "zone",
        "fault_type",
        "level_name",
        "level_index",
        "plan_x",
        "plan_y",
        "plan_w",
        "plan_h",
        "layout_source",
        "building_width_m",
        "building_depth_m",
        "forced_valve_position_value",
    ]
    usecols = [c for c in possible_cols if c in header.columns]

    if "zone" not in usecols:
        return pd.DataFrame(columns=["zone", "fault_type"])

    raw = pd.read_csv(
        raw_path,
        usecols=usecols,
        engine="python",
        on_bad_lines="skip",
    )

    first_cols = [c for c in raw.columns if c != "zone"]
    meta = raw.groupby("zone", as_index=False)[first_cols].first()

    if "fault_type" not in meta.columns:
        meta["fault_type"] = "unknown"

    return meta


def summarize_room(zone: str, g: pd.DataFrame, raw_meta_row: dict | None) -> dict:
    n = len(g)

    comfort_violation = float(g["comfort_violation_fraction"].mean())
    severity = severity_from_comfort_violation(comfort_violation)

    fractions = diagnosis_fractions(g)
    main_diagnosis, main_diagnosis_fraction = choose_main_diagnosis(
        fractions,
        severity,
    )
    dominant_fault_family = family_from_diagnosis(main_diagnosis)

    if main_diagnosis == NORMAL_DIAGNOSIS:
        main_diagnosis_count = int(g["diagnosis_text"].eq(NORMAL_DIAGNOSIS).sum())
    else:
        main_diagnosis_count = int(round(main_diagnosis_fraction * n))

    normal_fraction = float(g["diagnosis_text"].eq(NORMAL_DIAGNOSIS).mean())
    abnormal_fraction = 1.0 - normal_fraction

    row = {
        "zone": zone,
        "fault_type": "unknown",
        "zone_area_m2": float(g["zone_area_m2"].iloc[0]) if "zone_area_m2" in g.columns else None,
        "exposure_group": g["exposure_group"].iloc[0] if "exposure_group" in g.columns else None,
        "n_windows": n,
        "normal_fraction": normal_fraction,
        "abnormal_fraction": abnormal_fraction,
        "dominant_fault_family": dominant_fault_family,
        "dominant_fault_fraction": main_diagnosis_fraction,
        "main_diagnosis": main_diagnosis,
        "main_diagnosis_count": main_diagnosis_count,
        "main_diagnosis_fraction": main_diagnosis_fraction,
        "severity": severity,
        "comfort_violation_fraction": comfort_violation,
    }

    numeric_summary_cols = [
        "T_zone_mean",
        "T_set_mean",
        "tracking_error_mean",
        "overheating_fraction",
        "underheating_mean",
        "overheating_mean",
        "Q_mean",
        "Q_max",
        "E_sum",
        "m_dot_mean",
        "m_dot_std",
        "deltaT_water_mean",
        "deltaT_water_std",
        "T_supply_mean",
        "T_return_mean",
        "heating_active_fraction",
        "Q_density_mean",
        "Q_density_max",
        "E_density_sum",
        "m_dot_density_mean",
        "m_dot_density_std",
        "flow_oscillation_index",
        "flow_density_oscillation_index",
        "normalized_flow_fraction_mean",
        "normalized_flow_fraction_std",
        "valve_position_mean",
        "valve_position_max",
        "valve_position_min",
        "valve_position_range",
        "valve_position_std",
        "valve_position_oscillation_index",
        "controller_signal_mean",
        "controller_signal_max",
    ]

    for col in numeric_summary_cols:
        if col in g.columns:
            row[col] = float(g[col].mean())

    # Add diagnosis probabilities as columns for transparent review.
    for diagnosis, fraction in sorted(fractions.items()):
        safe_name = "diagnosis_fraction__" + diagnosis
        row[safe_name] = fraction

    if raw_meta_row:
        for key, value in raw_meta_row.items():
            if key != "zone":
                row[key] = value

    return row


def main() -> None:
    args = parse_args()
    case_id = args.case_id

    in_path = Path(args.input) if args.input else Path(
        f"data/processed/twin4build/{case_id}/t4b_3h_windows_diagnosed_against_openstudio.csv"
    )
    raw_path = Path(args.raw) if args.raw else Path(
        f"data/raw/twin4build/{case_id}/t4b_raw_timeseries.csv"
    )
    out_path = Path(args.out) if args.out else Path(
        f"data/processed/twin4build/{case_id}/t4b_room_level_diagnosis_summary.csv"
    )

    df = pd.read_csv(in_path)

    required = [
        "zone",
        "diagnosis_text",
        "comfort_violation_fraction",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {in_path}: {missing}")

    raw_meta = read_raw_metadata(raw_path)
    raw_meta_by_zone = {
        row["zone"]: row.to_dict()
        for _, row in raw_meta.iterrows()
    }

    rows = []
    for zone, g in df.groupby("zone", dropna=False):
        rows.append(
            summarize_room(
                zone=zone,
                g=g.copy(),
                raw_meta_row=raw_meta_by_zone.get(zone),
            )
        )

    summary = pd.DataFrame(rows)

    severity_order = {
        "fault": 0,
        "warning": 1,
        "normal": 2,
    }
    summary["_severity_order"] = summary["severity"].map(severity_order).fillna(99)
    summary = summary.sort_values(
        ["_severity_order", "comfort_violation_fraction", "zone"],
        ascending=[True, False, True],
    ).drop(columns=["_severity_order"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print("Input:", in_path)
    print("Raw metadata:", raw_path)
    print("Output:", out_path)
    print()
    print(
        summary[
            [
                "zone",
                "fault_type",
                "dominant_fault_family",
                "main_diagnosis",
                "severity",
                "comfort_violation_fraction",
                "main_diagnosis_fraction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
