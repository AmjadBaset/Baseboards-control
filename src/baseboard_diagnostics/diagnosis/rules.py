"""
Rule-based diagnosis from context-labeled hydronic heating features.

The diagnostic logic does not require controller command signals. Each row
represents one diagnostic window. Window-level rules are followed by a temporal
valve-travel check that can identify persistent limited-opening or
limited-closing behaviour across multiple windows.

Ground-truth injected fault labels are never used by the diagnostic rules.
"""

from __future__ import annotations

import math

import pandas as pd


# ---------------------------------------------------------------------------
# Valve-behaviour tolerances
# ---------------------------------------------------------------------------
# These thresholds describe lack of valve movement, not whether an absolute
# valve opening is "low" or "high".
VALVE_FIXED_STD_THRESHOLD = 0.02
VALVE_FIXED_RANGE_THRESHOLD = 0.05

# Temporal limited-travel detection uses the normalized physical valve stroke
# endpoints (0 = closed, 1 = fully open). A 5% travel tolerance is used when
# checking whether repeated extrema form a persistent internal ceiling/floor.
VALVE_TRAVEL_TOLERANCE = 0.05
VALVE_ENDPOINT_TOLERANCE = 0.05
MIN_PERSISTENT_LIMIT_FRACTION = 0.01
MIN_PERSISTENT_LIMIT_WINDOWS = 3


def _has_label(row: pd.Series, label_col: str, expected: str) -> bool:
    """Safely check whether a label column has a specific value."""
    if label_col not in row:
        return False
    return row[label_col] == expected


def _get_float(row: pd.Series, col: str, default=None):
    """Safely read a numeric feature from one diagnostic window."""
    if col not in row:
        return default
    value = row[col]
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_mask(df: pd.DataFrame, col: str, label: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].eq(label).fillna(False)


def _numeric_series(
    df: pd.DataFrame,
    col: str,
    default: float = 0.0,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float(default), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default))


def _append_diagnosis(
    df: pd.DataFrame,
    indices,
    diagnosis: str,
) -> None:
    """Append a diagnosis to selected rows without introducing duplicates."""
    for idx in indices:
        current = list(df.at[idx, "diagnoses"])
        if diagnosis not in current:
            current.append(diagnosis)
            df.at[idx, "diagnoses"] = current


def _remove_diagnoses(
    df: pd.DataFrame,
    indices,
    diagnoses_to_remove: set[str],
) -> None:
    """Remove less-specific diagnoses from selected rows."""
    for idx in indices:
        current = list(df.at[idx, "diagnoses"])
        df.at[idx, "diagnoses"] = [
            d for d in current
            if d not in diagnoses_to_remove
        ]


def apply_diagnostic_rules(row: pd.Series) -> list[str]:
    """
    Apply physical diagnostic rules to one labeled window.

    Required evidence comes from thermal comfort, heat output, hydronic flow,
    water-side temperature difference, flow variability, measured valve
    position, and a context-relative valve-effort label.

    The relative valve-effort label is derived by comparing measured valve
    position with the P10/P90 healthy normalized-flow-fraction reference for the
    matched operating context. It is supporting evidence only; low/high effort
    alone is never diagnosed as a fault.
    """
    diagnoses: list[str] = []

    # Context-dependent thermal/hydraulic labels.
    q_low = _has_label(row, "Q_density_mean_label", "low")
    q_high = _has_label(row, "Q_density_mean_label", "high")
    q_normal = _has_label(row, "Q_density_mean_label", "normal")

    flow_low = _has_label(row, "m_dot_density_mean_label", "low")
    flow_high = _has_label(row, "m_dot_density_mean_label", "high")
    flow_normal = _has_label(row, "m_dot_density_mean_label", "normal")

    dt_high = _has_label(row, "deltaT_water_mean_label", "high")

    oscillation_high = _has_label(
        row,
        "flow_density_oscillation_index_label",
        "high",
    )

    # Context-relative valve effort. This is not an absolute valve-opening rule.
    valve_effort_low = _has_label(
        row,
        "relative_valve_effort_label",
        "low",
    )
    valve_effort_high = _has_label(
        row,
        "relative_valve_effort_label",
        "high",
    )

    # Comfort state.
    comfort_fraction = _get_float(
        row,
        "comfort_violation_fraction",
        0.0,
    )
    overheating_fraction = _get_float(
        row,
        "overheating_fraction",
        0.0,
    )
    underheating_mean = _get_float(
        row,
        "underheating_mean",
        0.0,
    )
    overheating_mean = _get_float(
        row,
        "overheating_mean",
        0.0,
    )

    comfort_high = (
        _has_label(
            row,
            "comfort_violation_fraction_label",
            "high",
        )
        or comfort_fraction >= 0.05
    )

    overheating_high = (
        _has_label(
            row,
            "overheating_fraction_label",
            "high",
        )
        or overheating_fraction >= 0.05
        or overheating_mean > 0.05
    )

    underheating_high = (
        (comfort_high and not overheating_high)
        or underheating_mean > 0.05
    )

    # Measured valve-position persistence. No absolute opening threshold is used.
    valve_std = _get_float(
        row,
        "valve_position_std",
        None,
    )
    valve_range = _get_float(
        row,
        "valve_position_range",
        None,
    )

    valve_fixed = (
        (
            valve_std is not None
            and valve_std <= VALVE_FIXED_STD_THRESHOLD
        )
        or (
            valve_range is not None
            and valve_range <= VALVE_FIXED_RANGE_THRESHOLD
        )
    )

    low_delivery = q_low or flow_low
    continued_delivery = (
        q_normal
        or q_high
        or flow_normal
        or flow_high
    )
    hydraulic_abnormality = (
        flow_low
        or dt_high
        or q_low
    )

    # ------------------------------------------------------------------
    # 1. Fixed stuck-open / leakage-like behaviour
    # ------------------------------------------------------------------
    # Overheating + a non-modulating valve + continued heat/flow delivery.
    stuck_open_evidence = (
        overheating_high
        and valve_fixed
        and continued_delivery
    )

    if stuck_open_evidence:
        diagnoses.append(
            "possible_valve_leakage_or_stuck_open"
        )

    # ------------------------------------------------------------------
    # 2. Excessive heating capacity / hydronic oversupply
    # ------------------------------------------------------------------
    # The room overheats although the valve is operating below the healthy
    # context range and heat/flow delivery continues. This is stronger evidence
    # of excessive available heating capacity than overheating alone.
    if (
        overheating_high
        and not stuck_open_evidence
        and valve_effort_low
        and continued_delivery
    ):
        diagnoses.append(
            "possible_excessive_heating_capacity_or_oversupply"
        )

    # Generic hydronic oversupply when overheating and delivery continue but
    # neither the fixed-valve nor low-effort capacity pattern is present.
    elif (
        overheating_high
        and not stuck_open_evidence
        and continued_delivery
    ):
        diagnoses.append(
            "possible_oversupply_or_overheating"
        )

    # Overheating without evidence of ongoing hydronic delivery.
    elif (
        overheating_high
        and not stuck_open_evidence
    ):
        diagnoses.append(
            "possible_nonhydronic_or_unexplained_overheating"
        )

    # ------------------------------------------------------------------
    # 3. Fixed stuck-low / partially closed behaviour
    # ------------------------------------------------------------------
    # Underheating + a non-modulating valve + insufficient flow/heat delivery.
    stuck_low_evidence = (
        not overheating_high
        and underheating_high
        and valve_fixed
        and low_delivery
    )

    if stuck_low_evidence:
        diagnoses.append(
            "possible_partially_closed_or_stuck_low_valve"
        )

    # ------------------------------------------------------------------
    # 4. Hydraulic restriction with comfort impact
    # ------------------------------------------------------------------
    # High context-relative valve effort is required so that ordinary healthy
    # fluctuations in flow or delta-T do not create a restriction diagnosis.
    if (
        not overheating_high
        and underheating_high
        and valve_effort_high
        and hydraulic_abnormality
        and not stuck_low_evidence
    ):
        diagnoses.append(
            "hydraulic_restriction_with_comfort_impact"
        )

    # ------------------------------------------------------------------
    # 5. Degraded hydraulic restriction
    # ------------------------------------------------------------------
    # Comfort is still maintained, but the room requires unusually high valve
    # effort and one or more delivery indicators have degraded.
    if (
        not comfort_high
        and not overheating_high
        and valve_effort_high
        and q_low
    ):
        diagnoses.append(
            "hydraulic_restriction_degraded"
        )

    # ------------------------------------------------------------------
    # 6. Compensated hydraulic restriction
    # ------------------------------------------------------------------
    # Comfort and heat output are maintained only with unusually high
    # context-relative valve effort. This captures compensated restrictions
    # such as the controlled restricted-flow room.
    if (
        not comfort_high
        and not overheating_high
        and valve_effort_high
        and q_normal
    ):
        diagnoses.append(
            "possible_compensated_hydraulic_restriction"
        )

    # ------------------------------------------------------------------
    # 7. Insufficient heat output without a stronger valve/hydraulic cause
    # ------------------------------------------------------------------
    if (
        not overheating_high
        and underheating_high
        and q_low
        and not valve_effort_high
        and not stuck_low_evidence
    ):
        diagnoses.append(
            "insufficient_heat_output"
        )

    # ------------------------------------------------------------------
    # 8. Possible emitter undersizing / unusually high room load
    # ------------------------------------------------------------------
    # Delivery appears normal/high and no hydraulic restriction signature is
    # present, yet the room remains underheated.
    if (
        not overheating_high
        and underheating_high
        and (q_normal or q_high)
        and (flow_normal or flow_high)
        and not flow_low
        and not dt_high
        and not stuck_low_evidence
    ):
        diagnoses.append(
            "possible_emitter_undersizing_or_high_load"
        )

    # ------------------------------------------------------------------
    # 9. Flow/control instability
    # ------------------------------------------------------------------
    if (
        oscillation_high
        and not comfort_high
    ):
        diagnoses.append(
            "possible_flow_or_control_instability"
        )

    # ------------------------------------------------------------------
    # 10. Last fallback
    # ------------------------------------------------------------------
    if comfort_high and not diagnoses:
        diagnoses.append(
            "comfort_violation_without_matched_diagnostic_rule"
        )

    return diagnoses


def _apply_persistent_valve_travel_rules(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add temporal limited-travel valve evidence.

    Two behaviours are checked across repeated diagnostic windows:

    - Limited opening / stuck-low-like behaviour:
      underheating + low delivery while a moving valve repeatedly reaches a
      similar maximum below the physical fully-open endpoint.

    - Limited closing / stuck-open-like behaviour:
      overheating + continued delivery while a moving valve repeatedly reaches
      a similar minimum above the physical fully-closed endpoint.

    This supplements the single-window fixed-valve rules without introducing an
    arbitrary absolute boundary such as 0.40 or 0.50 valve opening.
    """
    required_columns = {
        "valve_position_min",
        "valve_position_max",
        "valve_position_range",
    }
    if not required_columns.issubset(
        df.columns
    ):
        return df

    group_cols = [
        col
        for col in [
            "case_id",
            "zone",
            "baseboard",
        ]
        if col in df.columns
    ]
    if not group_cols:
        return df

    for _, g in df.groupby(
        group_cols,
        dropna=False,
    ):
        n_windows = len(g)
        required_count = max(
            MIN_PERSISTENT_LIMIT_WINDOWS,
            int(
                math.ceil(
                    MIN_PERSISTENT_LIMIT_FRACTION
                    * n_windows
                )
            ),
        )

        comfort_fraction = _numeric_series(
            g,
            "comfort_violation_fraction",
            0.0,
        )
        overheating_fraction = _numeric_series(
            g,
            "overheating_fraction",
            0.0,
        )
        underheating_mean = _numeric_series(
            g,
            "underheating_mean",
            0.0,
        )
        overheating_mean = _numeric_series(
            g,
            "overheating_mean",
            0.0,
        )

        comfort_high = (
            _label_mask(
                g,
                "comfort_violation_fraction_label",
                "high",
            )
            | (comfort_fraction >= 0.05)
        )

        overheating_high = (
            _label_mask(
                g,
                "overheating_fraction_label",
                "high",
            )
            | (overheating_fraction >= 0.05)
            | (overheating_mean > 0.05)
        )

        underheating_high = (
            (comfort_high & ~overheating_high)
            | (underheating_mean > 0.05)
        )

        q_low = _label_mask(
            g,
            "Q_density_mean_label",
            "low",
        )
        q_normal = _label_mask(
            g,
            "Q_density_mean_label",
            "normal",
        )
        q_high = _label_mask(
            g,
            "Q_density_mean_label",
            "high",
        )

        flow_low = _label_mask(
            g,
            "m_dot_density_mean_label",
            "low",
        )
        flow_normal = _label_mask(
            g,
            "m_dot_density_mean_label",
            "normal",
        )
        flow_high = _label_mask(
            g,
            "m_dot_density_mean_label",
            "high",
        )

        valve_range = _numeric_series(
            g,
            "valve_position_range",
            0.0,
        )

        # Limited-travel evidence is only needed where the valve actually moves.
        # Fixed/near-fixed cases are already handled window by window.
        moving_valve = (
            valve_range
            > VALVE_FIXED_RANGE_THRESHOLD
        )

        low_delivery = (
            q_low
            | flow_low
        )
        continued_delivery = (
            q_normal
            | q_high
            | flow_normal
            | flow_high
        )

        # --------------------------------------------------------------
        # Persistent upper ceiling -> limited opening / stuck-low
        # --------------------------------------------------------------
        low_candidates = (
            underheating_high
            & ~overheating_high
            & moving_valve
        )

        upper_values = pd.to_numeric(
            g.loc[
                low_candidates,
                "valve_position_max",
            ],
            errors="coerce",
        ).dropna()

        if len(upper_values) >= required_count:
            upper_center = float(
                upper_values.median()
            )
            upper_iqr = float(
                upper_values.quantile(0.75)
                - upper_values.quantile(0.25)
            )

            upper_support = upper_values[
                (
                    upper_values
                    - upper_center
                ).abs()
                <= VALVE_TRAVEL_TOLERANCE
            ]

            internal_upper_limit = (
                upper_center
                < (
                    1.0
                    - VALVE_ENDPOINT_TOLERANCE
                )
            )

            if (
                internal_upper_limit
                and upper_iqr
                <= VALVE_TRAVEL_TOLERANCE
                and len(upper_support)
                >= required_count
            ):
                _append_diagnosis(
                    df,
                    upper_support.index,
                    (
                        "possible_partially_closed_"
                        "or_stuck_low_valve"
                    ),
                )
                _remove_diagnoses(
                    df,
                    upper_support.index,
                    {
                        "hydraulic_restriction_with_comfort_impact",
                        "insufficient_heat_output",
                        "possible_emitter_undersizing_or_high_load",
                    },
                )

        # --------------------------------------------------------------
        # Persistent lower floor -> limited closing / stuck-open
        # --------------------------------------------------------------
        open_candidates = (
            overheating_high
            & continued_delivery
            & moving_valve
        )

        lower_values = pd.to_numeric(
            g.loc[
                open_candidates,
                "valve_position_min",
            ],
            errors="coerce",
        ).dropna()

        if len(lower_values) >= required_count:
            lower_center = float(
                lower_values.median()
            )
            lower_iqr = float(
                lower_values.quantile(0.75)
                - lower_values.quantile(0.25)
            )

            lower_support = lower_values[
                (
                    lower_values
                    - lower_center
                ).abs()
                <= VALVE_TRAVEL_TOLERANCE
            ]

            internal_lower_limit = (
                lower_center
                > VALVE_ENDPOINT_TOLERANCE
            )

            if (
                internal_lower_limit
                and lower_iqr
                <= VALVE_TRAVEL_TOLERANCE
                and len(lower_support)
                >= required_count
            ):
                _append_diagnosis(
                    df,
                    lower_support.index,
                    (
                        "possible_valve_leakage_"
                        "or_stuck_open"
                    ),
                )
                _remove_diagnoses(
                    df,
                    lower_support.index,
                    {
                        "possible_excessive_heating_capacity_or_oversupply",
                        "possible_oversupply_or_overheating",
                        "possible_nonhydronic_or_unexplained_overheating",
                    },
                )

    return df


def apply_rules_to_labeled_windows(
    labeled_windows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply window-level and temporal diagnostic rules.

    Adds:
        diagnoses
        has_diagnosis
        one binary flag column per diagnosis
        diagnosis_text
    """
    df = labeled_windows.copy()

    df["diagnoses"] = df.apply(
        apply_diagnostic_rules,
        axis=1,
    )

    # Add persistent upper/lower valve travel-limit evidence across windows.
    df = _apply_persistent_valve_travel_rules(
        df
    )

    df["has_diagnosis"] = df[
        "diagnoses"
    ].apply(
        lambda x: len(x) > 0
    )

    all_diagnoses = sorted(
        {
            diagnosis
            for diagnoses in df["diagnoses"]
            for diagnosis in diagnoses
        }
    )

    for diagnosis in all_diagnoses:
        df[diagnosis] = df[
            "diagnoses"
        ].apply(
            lambda ds, d=diagnosis: d in ds
        )

    df["diagnosis_text"] = df[
        "diagnoses"
    ].apply(
        lambda ds: (
            "; ".join(ds)
            if ds
            else "normal_or_no_rule_triggered"
        )
    )

    return df
