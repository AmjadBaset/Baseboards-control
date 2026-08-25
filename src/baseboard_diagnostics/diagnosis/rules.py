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


def apply_diagnostic_rules(row: pd.Series) -> list[str]:
    """
    Apply diagnostic rules to one labeled window.

    Returns a list of diagnosis strings. A window can have more than one
    diagnosis if several rule conditions are true.
    """

    diagnoses = []

    q_low = _has_label(row, "Q_density_mean_label", "low")
    q_high = _has_label(row, "Q_density_mean_label", "high")
    q_normal = _has_label(row, "Q_density_mean_label", "normal")

    flow_low = _has_label(row, "m_dot_density_mean_label", "low")
    flow_high = _has_label(row, "m_dot_density_mean_label", "high")
    flow_normal = _has_label(row, "m_dot_density_mean_label", "normal")

    dt_high = _has_label(row, "deltaT_water_mean_label", "high")
    comfort_high = _has_label(row, "comfort_violation_fraction_label", "high")
    overheating_high = _has_label(row, "overheating_fraction_label", "high")

    # Comfort violation can mean either underheating or overheating.
    # Heat-deficit rules should only use underheating-like violations.
    underheating_high = comfort_high and not overheating_high

    oscillation_high = _has_label(
        row,
        "flow_density_oscillation_index_label",
        "high",
    )

    # In real-building deployment, valve position is treated as a proxy
    # for observed control effort.
    valve_effort_low = _has_label(row, "valve_effort_label", "low")
    valve_effort_high = _has_label(row, "valve_effort_label", "high")

    # 1. Hydraulic delivery deficit
    # Low flow and low heat alone are not treated as a strong fault unless
    # they are accompanied by comfort impact or high controller effort.
    if flow_low and q_low and (underheating_high or valve_effort_high):
        diagnoses.append("hydraulic_delivery_deficit")

    # 2. Low-flow / high-deltaT hydraulic abnormality
    # Low flow with high water-side deltaT can occur during stable operation.
    # Treat it as a stronger hydraulic abnormality only if it affects comfort
    # or requires high controller/valve effort.
    if flow_low and dt_high and (underheating_high or valve_effort_high):
        diagnoses.append("hydraulic_abnormality_low_flow_high_deltaT")

    # 3. Hydraulic limitation with comfort impact
    if flow_low and dt_high and underheating_high:
        diagnoses.append("hydraulic_limitation_with_comfort_impact")

    # 4. Possible valve stuck at low / partially closed position
    # The valve/control effort remains low while the room is underheated and
    # heat delivery or flow is low. This indicates that the valve may be stuck
    # near a closed position rather than responding to the heat demand.
    if valve_effort_low and underheating_high:
        diagnoses.append("possible_partially_closed_or_stuck_low_valve")

    # 5. Insufficient heat output
    if q_low and underheating_high:
        diagnoses.append("insufficient_heat_output")

    # 6. Possible emitter undersizing or high room load
    # This should only be used when the controller/valve is not stuck low.
    if (
        underheating_high
        and (q_normal or q_high)
        and (flow_normal or flow_high)
        and not valve_effort_low
    ):
        diagnoses.append("possible_emitter_undersizing_or_high_load")

    # 6. Possible oversupply or overheating
    if overheating_high and q_high:
        diagnoses.append("possible_oversupply_or_overheating")

    # 7. Possible control/hydraulic oscillation
    # This is kept as a weak rule. It should later be fused with controller ID.
    if oscillation_high and not comfort_high:
        diagnoses.append("possible_flow_or_control_instability")

    # 8. Hydraulic delivery deficit with controller compensation
    if valve_effort_high and flow_low and q_low:
        diagnoses.append("hydraulic_delivery_deficit_with_controller_compensation")

    # 9. Insufficient heat despite high controller effort
    if valve_effort_high and underheating_high and q_low:
        diagnoses.append("insufficient_heat_output_despite_high_controller_effort")

    # 9b. Control saturation with unmet heat demand.
    # The observed valve/control effort is high, but the room remains underheated
    # and heat output is still low. This is a control-response inconsistency:
    # the controller is asking for heat but cannot restore comfort.
    if valve_effort_high and underheating_high and q_low:
        diagnoses.append("control_saturation_with_unmet_heat_demand")

    # 10. Early/hidden restriction: controller compensates while comfort and heat are still normal
    if valve_effort_high and q_normal and not comfort_high:
        diagnoses.append("possible_compensated_hydraulic_restriction")

    # 11. Stronger compensated restriction: high effort, low flow,
    # but heat delivery and comfort are still acceptable.
    if valve_effort_high and flow_low and q_normal and not comfort_high:
        diagnoses.append("compensated_hydraulic_restriction_low_flow")

    # 12. Degraded compensation: high effort and low flow with reduced heat,
    # but comfort has not yet failed.
    if valve_effort_high and flow_low and q_low and not comfort_high:
        diagnoses.append("degraded_hydraulic_compensation")

    # 13. Possible valve leakage or stuck-open behavior.
    # If the room overheats while observed control effort is low, the emitter
    # may be receiving heat when the controller is not demanding it.
    if valve_effort_low and overheating_high:
        diagnoses.append("possible_valve_leakage_or_stuck_open")

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
