"""
Rule-based diagnosis from feature labels.

The rules use LOW / NORMAL / HIGH labels produced by feature_classification.py.
Each row represents one diagnostic window.
"""

from __future__ import annotations

import pandas as pd


def _has_label(row: pd.Series, label_col: str, expected: str) -> bool:
    """
    Safely check if a label column has a specific value.
    """
    if label_col not in row:
        return False

    return row[label_col] == expected


def _get_float(row: pd.Series, col: str, default=None):
    """
    Safely read a numeric feature from one diagnostic window.
    """
    if col not in row:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def apply_diagnostic_rules(row: pd.Series) -> list[str]:
    """
    Apply diagnostic rules to one labeled window.

    Diagnostic rules must use only observed features and labels.
    Ground-truth injected fault labels are not used here.
    """
    diagnoses = []

    q_low = _has_label(row, "Q_density_mean_label", "low")
    q_high = _has_label(row, "Q_density_mean_label", "high")
    q_normal = _has_label(row, "Q_density_mean_label", "normal")

    flow_low = _has_label(row, "m_dot_density_mean_label", "low")
    flow_high = _has_label(row, "m_dot_density_mean_label", "high")
    flow_normal = _has_label(row, "m_dot_density_mean_label", "normal")

    dt_high = _has_label(row, "deltaT_water_mean_label", "high")

    comfort_fraction = _get_float(row, "comfort_violation_fraction", 0.0)
    overheating_fraction = _get_float(row, "overheating_fraction", 0.0)
    underheating_mean = _get_float(row, "underheating_mean", 0.0)
    overheating_mean = _get_float(row, "overheating_mean", 0.0)

    comfort_high = (
        _has_label(row, "comfort_violation_fraction_label", "high")
        or comfort_fraction >= 0.05
    )

    overheating_high = (
        _has_label(row, "overheating_fraction_label", "high")
        or overheating_fraction >= 0.05
        or overheating_mean > 0.05
    )

    underheating_high = (
        comfort_high
        and not overheating_high
    ) or underheating_mean > 0.05

    oscillation_high = _has_label(
        row,
        "flow_density_oscillation_index_label",
        "high",
    )

    valve_effort_low = _has_label(row, "valve_effort_label", "low")
    valve_effort_high = _has_label(row, "valve_effort_label", "high")

    valve_mean = _get_float(row, "valve_position_mean", None)
    if valve_mean is None:
        valve_mean = _get_float(row, "controller_signal_mean", None)

    valve_std = _get_float(row, "valve_position_std", None)
    valve_range = _get_float(row, "valve_position_range", None)

    valve_high = (
        valve_effort_high
        or (valve_mean is not None and valve_mean >= 0.70)
    )

    valve_low = (
        valve_effort_low
        or (valve_mean is not None and valve_mean <= 0.25)
    )

    valve_fixed = (
        (valve_std is not None and valve_std <= 0.02)
        or (valve_range is not None and valve_range <= 0.05)
    )

    # 1. Stuck-open / leakage behavior.
    # Overheating plus an open/fixed valve should be treated as possible
    # stuck-open/leakage before falling back to generic oversupply.
    stuck_open_evidence = (
        overheating_high
        and (
            valve_effort_low
            or valve_high
            or (valve_fixed and valve_mean is not None and valve_mean >= 0.50)
        )
    )

    if stuck_open_evidence:
        diagnoses.append("possible_valve_leakage_or_stuck_open")

    # 2. Generic oversupply / overheating.
    # Use only when overheating exists but stuck-open/leakage evidence is weak.
    elif overheating_high:
        diagnoses.append("possible_oversupply_or_overheating")

    # 3. Hydraulic delivery deficit.
    # Do not classify overheated rooms as hydraulic restrictions.
    if (
        not overheating_high
        and flow_low
        and q_low
        and (underheating_high or valve_high)
    ):
        diagnoses.append("hydraulic_delivery_deficit")

    # 4. Low-flow / high-deltaT hydraulic abnormality.
    if (
        not overheating_high
        and flow_low
        and dt_high
        and (underheating_high or valve_high)
    ):
        diagnoses.append("hydraulic_abnormality_low_flow_high_deltaT")

    # 5. Hydraulic limitation with comfort impact.
    # High valve effort with underheating is a failed compensation pattern.
    if (
        not overheating_high
        and underheating_high
        and (
            flow_low
            or dt_high
            or valve_high
        )
    ):
        diagnoses.append("hydraulic_limitation_with_comfort_impact")

    # 6. Possible valve stuck low / partially closed.
    if (
        not overheating_high
        and underheating_high
        and (
            valve_low
            or (valve_fixed and valve_mean is not None and valve_mean <= 0.40)
        )
    ):
        diagnoses.append("possible_partially_closed_or_stuck_low_valve")

    # 7. Insufficient heat output.
    if not overheating_high and q_low and underheating_high:
        diagnoses.append("insufficient_heat_output")

    # 8. Possible emitter undersizing or high room load.
    if (
        not overheating_high
        and underheating_high
        and (q_normal or q_high)
        and (flow_normal or flow_high)
        and not valve_low
    ):
        diagnoses.append("possible_emitter_undersizing_or_high_load")

    # 9. Possible flow/control instability.
    if oscillation_high and not comfort_high:
        diagnoses.append("possible_flow_or_control_instability")

    # 10. Hydraulic delivery deficit with controller compensation.
    if not overheating_high and valve_high and flow_low and q_low:
        diagnoses.append("hydraulic_delivery_deficit_with_controller_compensation")

    # 11. Insufficient heat despite high controller effort.
    if not overheating_high and valve_high and underheating_high and q_low:
        diagnoses.append("insufficient_heat_output_despite_high_controller_effort")
        diagnoses.append("control_saturation_with_unmet_heat_demand")

    # 12. Compensated hydraulic restriction.
    # High valve effort, but comfort is still maintained.
    if not comfort_high and not overheating_high and valve_high and q_normal:
        diagnoses.append("possible_compensated_hydraulic_restriction")

    # 13. Stronger compensated restriction: high effort + low flow.
    if not comfort_high and not overheating_high and valve_high and flow_low and q_normal:
        diagnoses.append("compensated_hydraulic_restriction_low_flow")

    # 14. Degraded compensation: high effort + low flow + reduced heat,
    # but comfort has not yet failed.
    if not comfort_high and not overheating_high and valve_high and flow_low and q_low:
        diagnoses.append("degraded_hydraulic_compensation")

    # 15. Last fallback.
    # Comfort is violated but no physical diagnostic rule explains it.
    if comfort_high and not diagnoses:
        diagnoses.append("comfort_violation_without_matched_diagnostic_rule")

    return diagnoses


def apply_rules_to_labeled_windows(labeled_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Apply diagnostic rules to all labeled windows.

    Adds:
        diagnoses
        has_diagnosis
        one binary flag column per diagnosis
    """

    df = labeled_windows.copy()

    df["diagnoses"] = df.apply(apply_diagnostic_rules, axis=1)
    df["has_diagnosis"] = df["diagnoses"].apply(lambda x: len(x) > 0)

    all_diagnoses = sorted({d for ds in df["diagnoses"] for d in ds})

    for diagnosis in all_diagnoses:
        df[diagnosis] = df["diagnoses"].apply(lambda ds: diagnosis in ds)

    df["diagnosis_text"] = df["diagnoses"].apply(
        lambda ds: "; ".join(ds) if ds else "normal_or_no_rule_triggered"
    )

    return df
