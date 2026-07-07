"""
Build OpenStudio reference bands from window-level features.

Reference bands are calculated from healthy OpenStudio baseline windows.
For each operating context, P10/P50/P90 are calculated for selected features.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_GROUP_COLUMNS = [
    "case_id",
    "archetype",
    "construction_year",
    "zone_area_m2",
    "exposure_group",
    "T_out_bin",
    "period",
]


DEFAULT_FEATURES_FOR_BANDS = [
    "tracking_error_mean",
    "comfort_violation_fraction",
    "overheating_fraction",
    "underheating_mean",
    "overheating_mean",
    "Q_mean",
    "Q_max",
    "m_dot_mean",
    "m_dot_std",
    "deltaT_water_mean",
    "deltaT_water_std",
    "Q_density_mean",
    "Q_density_max",
    "E_density_sum",
    "m_dot_density_mean",
    "m_dot_density_std",
    "flow_oscillation_index",
    "flow_density_oscillation_index",
    "normalized_flow_fraction_mean",
    "normalized_flow_fraction_std",
]


def build_reference_bands(
    windows: pd.DataFrame,
    group_columns: list[str] | None = None,
    features: list[str] | None = None,
    heating_windows_only: bool = True,
) -> pd.DataFrame:
    """
    Build percentile reference bands.

    Parameters
    ----------
    windows:
        Window-level feature DataFrame.

    group_columns:
        Context columns used to group comparable operating conditions.

    features:
        Numeric features for which P10/P50/P90 should be calculated.

    heating_windows_only:
        If True, use only rows where valid_heating_window == True.

    Returns
    -------
    DataFrame with one row per context and P10/P50/P90 columns.
    """

    df = windows.copy()

    if group_columns is None:
        group_columns = DEFAULT_GROUP_COLUMNS

    if features is None:
        features = DEFAULT_FEATURES_FOR_BANDS

    missing_groups = [col for col in group_columns if col not in df.columns]
    if missing_groups:
        raise ValueError(f"Missing group columns: {missing_groups}")

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if heating_windows_only:
        if "valid_heating_window" not in df.columns:
            raise ValueError("valid_heating_window column is required.")
        df = df[df["valid_heating_window"] == True].copy()

    grouped = df.groupby(group_columns, dropna=False)

    parts = []

    for feature in features:
        quantiles = grouped[feature].quantile([0.10, 0.50, 0.90]).unstack()
        quantiles = quantiles.rename(
            columns={
                0.10: f"{feature}_P10",
                0.50: f"{feature}_P50",
                0.90: f"{feature}_P90",
            }
        )
        parts.append(quantiles)

    bands = pd.concat(parts, axis=1).reset_index()

    return bands
