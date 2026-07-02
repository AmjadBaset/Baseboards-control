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
    oscillation_high = _has_label(
        row,
        "flow_density_oscillation_index_label",
        "high",
    )

    # 1. Hydraulic delivery deficit
    if flow_low and q_low:
        diagnoses.append("hydraulic_delivery_deficit")

    # 2. Low-flow / high-deltaT hydraulic abnormality
    if flow_low and dt_high:
        diagnoses.append("hydraulic_abnormality_low_flow_high_deltaT")

    # 3. Hydraulic limitation with comfort impact
    if flow_low and dt_high and comfort_high:
        diagnoses.append("hydraulic_limitation_with_comfort_impact")

    # 4. Insufficient heat output
    if q_low and comfort_high:
        diagnoses.append("insufficient_heat_output")

    # 5. Possible emitter undersizing or high room load
    if comfort_high and (q_normal or q_high) and (flow_normal or flow_high):
        diagnoses.append("possible_emitter_undersizing_or_high_load")

    # 6. Possible oversupply or overheating
    if overheating_high and q_high:
        diagnoses.append("possible_oversupply_or_overheating")

    # 7. Possible control/hydraulic oscillation
    # This is kept as a weak rule. It should later be fused with controller ID.
    if oscillation_high and not comfort_high:
        diagnoses.append("possible_flow_or_control_instability")

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
