"""
Persistence-threshold calibration for diagnostic rules.

This module uses healthy reference diagnosis outputs to estimate how often each
diagnosis can appear under normal operation. These normal fractions are then
used as persistence thresholds.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_ID_COLUMNS = [
    "case_id",
    "archetype",
    "construction_year",
    "zone",
    "baseboard",
]


def get_diagnosis_columns(df: pd.DataFrame) -> list[str]:
    """
    Find binary diagnosis columns created by rules.py.
    """
    prefixes = (
        "hydraulic_",
        "insufficient_",
        "possible_",
    )

    return [
        col for col in df.columns
        if col.startswith(prefixes) and df[col].dropna().isin([True, False]).all()
    ]


def calculate_diagnosis_fractions(
    diagnosed_windows: pd.DataFrame,
    id_columns: list[str] | None = None,
    diagnosis_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calculate diagnosis fraction per baseboard/zone.
    """

    df = diagnosed_windows.copy()

    if id_columns is None:
        id_columns = DEFAULT_ID_COLUMNS

    if diagnosis_columns is None:
        diagnosis_columns = get_diagnosis_columns(df)

    missing_ids = [col for col in id_columns if col not in df.columns]
    if missing_ids:
        raise ValueError(f"Missing id columns: {missing_ids}")

    if not diagnosis_columns:
        raise ValueError("No diagnosis columns found.")

    fractions = (
        df.groupby(id_columns)[diagnosis_columns]
        .mean()
        .reset_index()
    )

    return fractions


def calibrate_persistence_thresholds(
    diagnosis_fractions: pd.DataFrame,
    diagnosis_columns: list[str] | None = None,
    quantile: float = 0.95,
    minimum_floor: float = 0.02,
) -> pd.DataFrame:
    """
    Calibrate persistence thresholds from healthy diagnosis fractions.

    Threshold = max(P95 across healthy baseboards, minimum_floor)
    """

    df = diagnosis_fractions.copy()

    if diagnosis_columns is None:
        diagnosis_columns = [
            col for col in df.columns
            if col.startswith(("hydraulic_", "insufficient_", "possible_"))
        ]

    rows = []

    for diagnosis in diagnosis_columns:
        p_value = float(df[diagnosis].quantile(quantile))
        threshold = max(p_value, minimum_floor)

        rows.append(
            {
                "diagnosis": diagnosis,
                "calibration_quantile": quantile,
                "normal_fraction_quantile": p_value,
                "minimum_floor": minimum_floor,
                "persistence_threshold": threshold,
            }
        )

    return pd.DataFrame(rows)
