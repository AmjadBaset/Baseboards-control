"""
Baseboard-level diagnostic summary.

This module aggregates window-level diagnoses into final persistent diagnoses
per zone/baseboard using calibrated persistence thresholds.
"""

from __future__ import annotations

import pandas as pd

from baseboard_diagnostics.diagnosis.persistence import get_diagnosis_columns


DEFAULT_ID_COLUMNS = [
    "case_id",
    "archetype",
    "construction_year",
    "zone",
    "baseboard",
]


def summarize_persistent_diagnoses(
    diagnosed_windows: pd.DataFrame,
    persistence_thresholds: pd.DataFrame,
    id_columns: list[str] | None = None,
    minimum_exceedance_margin: float = 0.005,
) -> pd.DataFrame:
    """
    Summarize persistent diagnoses per zone/baseboard.

    Parameters
    ----------
    diagnosed_windows:
        Window-level diagnosed DataFrame.

    persistence_thresholds:
        DataFrame with columns:
            diagnosis
            persistence_threshold

    id_columns:
        Columns defining one diagnosed unit.

    Returns
    -------
    DataFrame with one row per zone/baseboard and final diagnosis summary.
    """

    if id_columns is None:
        id_columns = DEFAULT_ID_COLUMNS

    diagnosis_columns = get_diagnosis_columns(diagnosed_windows)

    if not diagnosis_columns:
        raise ValueError("No diagnosis columns found in diagnosed_windows.")

    required_threshold_cols = ["diagnosis", "persistence_threshold"]
    missing_threshold_cols = [
        col for col in required_threshold_cols
        if col not in persistence_thresholds.columns
    ]
    if missing_threshold_cols:
        raise ValueError(
            f"Missing threshold columns: {missing_threshold_cols}"
        )

    threshold_map = dict(
        zip(
            persistence_thresholds["diagnosis"],
            persistence_thresholds["persistence_threshold"],
        )
    )

    rows = []

    grouped = diagnosed_windows.groupby(id_columns)

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(id_columns, keys))

        diagnosis_results = []

        for diagnosis in diagnosis_columns:
            fraction = float(group[diagnosis].mean())
            threshold = float(threshold_map.get(diagnosis, 1.0))
            exceedance = fraction - threshold
            persistent = exceedance > minimum_exceedance_margin

            base[f"{diagnosis}_fraction"] = fraction
            base[f"{diagnosis}_threshold"] = threshold
            base[f"{diagnosis}_exceedance"] = exceedance
            base[f"{diagnosis}_persistent"] = persistent

            diagnosis_results.append(
                {
                    "diagnosis": diagnosis,
                    "fraction": fraction,
                    "threshold": threshold,
                    "exceedance": exceedance,
                    "persistent": persistent,
                }
            )

        persistent_results = [
            item for item in diagnosis_results
            if item["persistent"]
        ]

        if persistent_results:
            dominant = max(
                persistent_results,
                key=lambda item: item["exceedance"],
            )
            base["has_persistent_issue"] = True
            base["dominant_diagnosis"] = dominant["diagnosis"]
            base["dominant_fraction"] = dominant["fraction"]
            base["dominant_threshold"] = dominant["threshold"]
            base["dominant_exceedance"] = dominant["exceedance"]
        else:
            base["has_persistent_issue"] = False
            base["dominant_diagnosis"] = "normal_or_no_persistent_issue"
            base["dominant_fraction"] = 0.0
            base["dominant_threshold"] = 0.0
            base["dominant_exceedance"] = 0.0

        rows.append(base)

    return pd.DataFrame(rows)
