"""
Window-level feature engineering for baseboard diagnostic data.

This module converts timestep-level data into fixed-length diagnostic windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _oscillation_index(series: pd.Series) -> float:
    """
    Simple oscillation index based on mean absolute timestep-to-timestep change.
    """
    values = series.dropna().to_numpy()

    if len(values) < 2:
        return np.nan

    return float(np.mean(np.abs(np.diff(values))))


def create_window_features(
    df: pd.DataFrame,
    window: str = "3h",
    heating_active_fraction_threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Create window-level features for each case/zone/baseboard.

    Expected timestep columns:
        timestamp
        case_id
        archetype
        construction_year
        zone
        baseboard
        zone_area_m2
        T_out
        T_out_bin
        period
        occupancy
        T_zone
        T_set
        tracking_error
        comfort_violation
        overheating_flag
        underheating
        overheating
        Q_bb
        E_bb
        m_dot
        T_supply
        T_return
        deltaT_water
        heating_active
        Q_density
        E_density
        m_dot_density
    """

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    required_cols = [
        "timestamp",
        "case_id",
        "archetype",
        "construction_year",
        "zone",
        "baseboard",
        "zone_area_m2",
        "T_out",
        "T_out_bin",
        "period",
        "occupancy",
        "T_zone",
        "T_set",
        "tracking_error",
        "comfort_violation",
        "overheating_flag",
        "underheating",
        "overheating",
        "Q_bb",
        "E_bb",
        "m_dot",
        "T_supply",
        "T_return",
        "deltaT_water",
        "heating_active",
        "Q_density",
        "E_density",
        "m_dot_density",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    group_cols = [
        "case_id",
        "archetype",
        "construction_year",
        "zone",
        "baseboard",
    ]

    rows = []

    expected_timesteps = int(pd.Timedelta(window) / pd.Timedelta(minutes=15))

    for keys, group in df.groupby(group_cols):
        group = group.sort_values("timestamp").reset_index(drop=True)

        # Create row-based windows starting from the first actual timestep.
        # For EnergyPlus 15-min annual data, this preserves the first window:
        # 00:15, 00:30, ..., 03:00.
        group["window_id"] = np.arange(len(group)) // expected_timesteps

        for _, w in group.groupby("window_id"):
            if len(w) != expected_timesteps:
                continue

            heating_active_fraction = w["heating_active"].mean()
            window_start = w["timestamp"].iloc[0]
            window_end = w["timestamp"].iloc[-1]

            row = {
                "window_start": window_start,
                "window_end": window_end,
                "case_id": keys[0],
                "archetype": keys[1],
                "construction_year": keys[2],
                "zone": keys[3],
                "baseboard": keys[4],
                "zone_area_m2": w["zone_area_m2"].iloc[0],
                "n_timesteps": len(w),
                "T_out_mean": w["T_out"].mean(),
                "T_out_bin": w["T_out_bin"].mode().iloc[0] if not w["T_out_bin"].mode().empty else np.nan,
                "period": w["period"].mode().iloc[0] if not w["period"].mode().empty else np.nan,
                "occupancy_mean": w["occupancy"].mean(),
                "T_zone_mean": w["T_zone"].mean(),
                "T_set_mean": w["T_set"].mean(),
                "tracking_error_mean": w["tracking_error"].mean(),
                "comfort_violation_fraction": w["comfort_violation"].mean(),
                "overheating_fraction": w["overheating_flag"].mean(),
                "underheating_mean": w["underheating"].mean(),
                "overheating_mean": w["overheating"].mean(),
                "Q_mean": w["Q_bb"].mean(),
                "Q_max": w["Q_bb"].max(),
                "E_sum": w["E_bb"].sum(),
                "m_dot_mean": w["m_dot"].mean(),
                "m_dot_std": w["m_dot"].std(),
                "deltaT_water_mean": w["deltaT_water"].mean(),
                "deltaT_water_std": w["deltaT_water"].std(),
                "T_supply_mean": w["T_supply"].mean(),
                "T_return_mean": w["T_return"].mean(),
                "heating_active_fraction": heating_active_fraction,
                "Q_density_mean": w["Q_density"].mean(),
                "Q_density_max": w["Q_density"].max(),
                "E_density_sum": w["E_density"].sum(),
                "m_dot_density_mean": w["m_dot_density"].mean(),
                "m_dot_density_std": w["m_dot_density"].std(),
                "flow_oscillation_index": _oscillation_index(w["m_dot"]),
                "flow_density_oscillation_index": _oscillation_index(w["m_dot_density"]),
            }

            if "normalized_flow_fraction" in w.columns:
                row["normalized_flow_fraction_mean"] = w["normalized_flow_fraction"].mean()
                row["normalized_flow_fraction_std"] = w["normalized_flow_fraction"].std()
                row["flow_fraction_oscillation_index"] = _oscillation_index(
                    w["normalized_flow_fraction"]
                )

            row["valid_heating_window"] = (
                heating_active_fraction > heating_active_fraction_threshold
            )

            rows.append(row)

    return pd.DataFrame(rows)
