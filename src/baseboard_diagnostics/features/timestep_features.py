"""
Timestep-level feature engineering for baseboard diagnostic data.

This module is used for both OpenStudio/EnergyPlus data and Twin4Build data
after both have been converted to a common column format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_timestep_features(
    df: pd.DataFrame,
    comfort_tolerance: float = 0.5,
    heating_delta_t_threshold: float = 0.8,
    heating_power_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    Add timestep-level diagnostic features.

    Expected input columns:
        T_zone
        T_set
        Q_bb
        m_dot
        T_supply
        T_return
        zone_area_m2

    Optional input column:
        m_dot_design

    Added columns:
        tracking_error
        underheating
        overheating
        comfort_violation
        deltaT_water
        heating_active
        Q_density
        m_dot_density
        normalized_flow_fraction, if m_dot_design exists
    """

    df = df.copy()

    required_cols = [
        "T_zone",
        "T_set",
        "Q_bb",
        "m_dot",
        "T_supply",
        "T_return",
        "zone_area_m2",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["tracking_error"] = df["T_set"] - df["T_zone"]

    df["underheating"] = np.maximum(
        df["T_set"] - df["T_zone"] - comfort_tolerance,
        0.0,
    )

    df["overheating"] = np.maximum(
        df["T_zone"] - df["T_set"] - comfort_tolerance,
        0.0,
    )

    df["comfort_violation"] = (
        (df["underheating"] > 0.0) | (df["overheating"] > 0.0)
    ).astype(int)

    df["overheating_flag"] = (df["overheating"] > 0.0).astype(int)
    df["underheating_flag"] = (df["underheating"] > 0.0).astype(int)

    df["deltaT_water"] = df["T_supply"] - df["T_return"]

    df["heating_active"] = (
        (df["deltaT_water"] > heating_delta_t_threshold)
        | (df["Q_bb"] > heating_power_threshold)
    ).astype(int)

    df["Q_density"] = df["Q_bb"] / df["zone_area_m2"]
    df["m_dot_density"] = df["m_dot"] / df["zone_area_m2"]

    if "m_dot_design" in df.columns:
        df["normalized_flow_fraction"] = (
            df["m_dot"] / df["m_dot_design"]
        ).replace([np.inf, -np.inf], np.nan)

        df["normalized_flow_fraction"] = df["normalized_flow_fraction"].clip(
            lower=0.0,
            upper=1.0,
        )

    return df
