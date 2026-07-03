"""
Feature classification against OpenStudio reference bands.

Each real/Twin4Build/OpenStudio window is compared with the matching
reference-band row and labeled as LOW, NORMAL, or HIGH.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_MATCH_COLUMNS = [
    "case_id",
    "archetype",
    "construction_year",
    "zone",
    "baseboard",
    "zone_area_m2",
    "T_out_bin",
    "period",
]


DEFAULT_FEATURES_TO_CLASSIFY = [
    "tracking_error_mean",
    "comfort_violation_fraction",
    "overheating_fraction",
    "Q_density_mean",
    "m_dot_density_mean",
    "deltaT_water_mean",
    "flow_density_oscillation_index",
]


def classify_value(value, p10, p90):
    """
    Classify one value against P10/P90 thresholds.
    """
    if pd.isna(value) or pd.isna(p10) or pd.isna(p90):
        return "missing"

    if value < p10:
        return "low"

    if value > p90:
        return "high"

    return "normal"


def classify_features_against_bands(
    windows: pd.DataFrame,
    bands: pd.DataFrame,
    match_columns: list[str] | None = None,
    features: list[str] | None = None,
    feature_band_mapping: dict[str, str] | None = None,
    label_name_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Merge window features with matching reference bands and classify features.

    Parameters
    ----------
    windows:
        Window-level feature DataFrame.

    bands:
        Reference-band DataFrame containing P10/P50/P90 columns.

    match_columns:
        Columns used to match each window with the correct reference context.

    features:
        Features to classify.

    Returns
    -------
    DataFrame with original window columns, matching band columns, and
    added label columns named '<feature>_label'.
    """

    if match_columns is None:
        match_columns = DEFAULT_MATCH_COLUMNS

    if features is None:
        features = DEFAULT_FEATURES_TO_CLASSIFY

    if feature_band_mapping is None:
        feature_band_mapping = {}

    if label_name_mapping is None:
        label_name_mapping = {}

    missing_window_cols = [col for col in match_columns if col not in windows.columns]
    if missing_window_cols:
        raise ValueError(f"Missing window match columns: {missing_window_cols}")

    missing_band_cols = [col for col in match_columns if col not in bands.columns]
    if missing_band_cols:
        raise ValueError(f"Missing band match columns: {missing_band_cols}")

    for feature in features:
        if feature not in windows.columns:
            raise ValueError(f"Missing window feature column: {feature}")

        band_feature = feature_band_mapping.get(feature, feature)

        p10_col = f"{band_feature}_P10"
        p90_col = f"{band_feature}_P90"

        if p10_col not in bands.columns:
            raise ValueError(f"Missing band column: {p10_col}")

        if p90_col not in bands.columns:
            raise ValueError(f"Missing band column: {p90_col}")

    merged = windows.merge(
        bands,
        on=match_columns,
        how="left",
        suffixes=("", "_band"),
    )

    for feature in features:
        band_feature = feature_band_mapping.get(feature, feature)

        p10_col = f"{band_feature}_P10"
        p90_col = f"{band_feature}_P90"
        label_col = label_name_mapping.get(feature, f"{feature}_label")

        merged[label_col] = merged.apply(
            lambda row: classify_value(
                row[feature],
                row[p10_col],
                row[p90_col],
            ),
            axis=1,
        )

    return merged
