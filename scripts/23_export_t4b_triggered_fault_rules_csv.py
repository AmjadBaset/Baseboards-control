"""
Export the highest-probability triggered diagnostic rule per room.

Output CSV columns:
- room
- severity
- triggered_fault_rule

Architecture:
- script 11 / rules.py creates window-level diagnoses.
- script 12 creates room-level severity.
- this script only exports the highest-probability rule per room.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


NORMAL_DIAGNOSIS = "normal_or_no_rule_triggered"


LABELS = {
    "hydraulic_delivery_deficit": "Hydraulic delivery deficit",
    "hydraulic_abnormality_low_flow_high_deltaT": "Hydraulic abnormality: low flow / high ΔT",
    "hydraulic_limitation_with_comfort_impact": "Hydraulic limitation with comfort impact",
    "hydraulic_delivery_deficit_with_controller_compensation": "Hydraulic delivery deficit with controller compensation",
    "possible_compensated_hydraulic_restriction": "Possible compensated hydraulic restriction",
    "compensated_hydraulic_restriction_low_flow": "Compensated hydraulic restriction",
    "degraded_hydraulic_compensation": "Degraded hydraulic compensation",
    "control_saturation_with_unmet_heat_demand": "Control saturation with unmet heat demand",
    "possible_flow_or_control_instability": "Possible flow or control instability",
    "possible_valve_leakage_or_stuck_open": "Possible valve leakage or stuck open",
    "possible_partially_closed_or_stuck_low_valve": "Possible partially closed or stuck low valve",
    "insufficient_heat_output": "Insufficient heat output",
    "insufficient_heat_output_despite_high_controller_effort": "Insufficient heat output despite high controller effort",
    "possible_emitter_undersizing_or_high_load": "Possible emitter undersizing or high load",
    "possible_oversupply_or_overheating": "Possible oversupply or overheating",
    "comfort_violation_without_matched_diagnostic_rule": "Comfort violation without matched diagnostic rule",
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
        required=True,
        help="Case id under data/processed/twin4build/<case-id>.",
    )
    parser.add_argument(
        "--diagnosed",
        default=None,
        help="Optional path to t4b_3h_windows_diagnosed_against_openstudio.csv.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path to t4b_room_level_diagnosis_summary.csv.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path.",
    )
    parser.add_argument(
        "--include-normal",
        action="store_true",
        help="Include normal rooms. Default exports warning/fault rooms only.",
    )
    return parser.parse_args()


def readable_label(diagnosis: str) -> str:
    return LABELS.get(diagnosis, diagnosis.replace("_", " ").capitalize())


def parse_diagnoses(value) -> list[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()

        if not text or text in {NORMAL_DIAGNOSIS, "normal"}:
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
        diagnosis = str(item).strip().strip("'").strip('"')
        if diagnosis and diagnosis not in {NORMAL_DIAGNOSIS, "normal"}:
            diagnoses.append(diagnosis)

    return diagnoses


def highest_probability_diagnosis(g: pd.DataFrame) -> str:
    """
    Pick the most frequent diagnostic rule in this room.
    Ties are resolved by DIAGNOSIS_PRIORITY.
    """
    n_windows = max(len(g), 1)
    counts: dict[str, int] = {}

    for value in g["diagnosis_text"].fillna(NORMAL_DIAGNOSIS):
        # Count each rule at most once per window.
        for diagnosis in set(parse_diagnoses(value)):
            counts[diagnosis] = counts.get(diagnosis, 0) + 1

    if not counts:
        return "Normal"

    priority_rank = {
        diagnosis: i for i, diagnosis in enumerate(DIAGNOSIS_PRIORITY)
    }

    best_diagnosis, _ = sorted(
        counts.items(),
        key=lambda item: (
            -(item[1] / n_windows),
            priority_rank.get(item[0], len(priority_rank)),
            item[0],
        ),
    )[0]

    return readable_label(best_diagnosis)


def main() -> None:
    args = parse_args()
    case_id = args.case_id

    base = Path(f"data/processed/twin4build/{case_id}")

    diagnosed_path = Path(args.diagnosed) if args.diagnosed else (
        base / "t4b_3h_windows_diagnosed_against_openstudio.csv"
    )
    summary_path = Path(args.summary) if args.summary else (
        base / "t4b_room_level_diagnosis_summary.csv"
    )
    out_path = Path(args.out) if args.out else (
        Path("reports") / f"{case_id}_top_triggered_fault_rule.csv"
    )

    diagnosed = pd.read_csv(diagnosed_path)
    summary = pd.read_csv(summary_path)

    required_diagnosed = ["zone", "diagnosis_text"]
    required_summary = ["zone", "severity"]

    missing_diagnosed = [c for c in required_diagnosed if c not in diagnosed.columns]
    missing_summary = [c for c in required_summary if c not in summary.columns]

    if missing_diagnosed:
        raise ValueError(f"Missing columns in {diagnosed_path}: {missing_diagnosed}")

    if missing_summary:
        raise ValueError(f"Missing columns in {summary_path}: {missing_summary}")

    severity_by_zone = dict(zip(summary["zone"], summary["severity"]))

    rows = []

    for zone, g in diagnosed.groupby("zone", dropna=False):
        severity = str(severity_by_zone.get(zone, "unknown")).lower()

        if not args.include_normal and severity not in {"warning", "fault"}:
            continue

        rows.append(
            {
                "room": zone,
                "severity": severity.capitalize(),
                "triggered_fault_rule": highest_probability_diagnosis(g),
            }
        )

    out = pd.DataFrame(rows, columns=[
        "room",
        "severity",
        "triggered_fault_rule",
    ])

    if not out.empty:
        severity_order = {"Fault": 0, "Warning": 1, "Normal": 2, "Unknown": 3}
        out["_severity_order"] = out["severity"].map(severity_order).fillna(99)
        out = out.sort_values(
            ["_severity_order", "room"],
            ascending=[True, True],
        ).drop(columns=["_severity_order"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Diagnosed input:", diagnosed_path)
    print("Summary input:", summary_path)
    print("Output:", out_path)
    print()
    print(out.to_string(index=False) if not out.empty else "No warning/fault rooms found.")


if __name__ == "__main__":
    main()
