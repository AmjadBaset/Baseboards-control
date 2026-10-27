#!/usr/bin/env python3
"""Inspect thermal input/output consistency in a dedicated Twin4Build physics trace.

This is a debugging aid, not a diagnostic rule or a change to the simulator.
The approximate air-node energy balance is a directional sanity check: it uses
reported instantaneous room and wall temperatures and cannot reconstruct the
full discrete-time trajectory without the model state-space history.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CP_AIR_J_KG_K = 1005.0
F_AIR_SOLAR = 0.05
Q_OCC_PERSON_W = 100.0
DT_SECONDS = 900.0


def analyze(df: pd.DataFrame) -> None:
    required = {
        "zone", "timestamp", "T_zone_C", "T_wall_C", "T_out_input_C",
        "Q_heater_reported_W", "heater_Q_delivered_W", "R_in_K_W",
        "C_air_J_K", "T_supply_air_C", "air_supply_flow_kg_s",
        "air_exhaust_flow_kg_s", "occupants", "solar_input",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for zone, g in df.groupby("zone", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["Q_wall_W"] = (g["T_wall_C"] - g["T_zone_C"]) / g["R_in_K_W"]
        g["Q_air_W"] = CP_AIR_J_KG_K * (
            g["air_supply_flow_kg_s"] * g["T_supply_air_C"]
            - g["air_exhaust_flow_kg_s"] * g["T_zone_C"]
        )
        g["Q_occupants_W"] = Q_OCC_PERSON_W * g["occupants"]
        g["Q_solar_W"] = F_AIR_SOLAR * g["solar_input"]
        g["Q_net_model_W"] = (
            g["Q_wall_W"] + g["Q_air_W"] + g["Q_occupants_W"]
            + g["Q_solar_W"] + g["heater_Q_delivered_W"]
        )
        g["approx_dT_next_C"] = DT_SECONDS * g["Q_net_model_W"] / g["C_air_J_K"]
        g["observed_dT_next_C"] = g["T_zone_C"].shift(-1) - g["T_zone_C"]
        g["heat_connection_delta_W"] = g["heater_Q_delivered_W"] - g["Q_heater_reported_W"]
        bad = (g["T_zone_C"] < -5) | (g["T_zone_C"] < g["T_out_input_C"] - 2)
        h_received = g["heater_Q_delivered_W"].to_numpy()
        h_reported = g["Q_heater_reported_W"].to_numpy()
        aligned = {
            "same_step": (h_received, h_reported),
            "room_receives_previous_step": (h_received[1:], h_reported[:-1]),
            "room_receives_next_step": (h_received[:-1], h_reported[1:]),
        }
        print(f"\n{zone}: min T_zone={g['T_zone_C'].min():.2f} C; "
              f"first unphysical={g.loc[bad, 'timestamp'].min() if bad.any() else 'none'}")
        for label, (a, b) in aligned.items():
            if len(a):
                print(f"  Heating input comparison [{label}]: "
                      f"median absolute difference={np.median(np.abs(a-b)):.3f} W, "
                      f"P95={np.quantile(np.abs(a-b), .95):.3f} W")
        if not bad.any():
            continue
        ix = int(np.flatnonzero(bad)[0])
        sample = g.iloc[max(0, ix - 3): min(len(g), ix + 6)]
        cols = ["timestamp", "T_out_input_C", "T_zone_C", "T_wall_C",
                "Q_heater_reported_W", "heater_Q_delivered_W", "Q_wall_W",
                "Q_air_W", "Q_net_model_W", "approx_dT_next_C", "observed_dT_next_C"]
        print(sample[cols].round(3).to_string(index=False))
        # Look for especially strong contradictions, which motivate a closer
        # inspection of the installed state-space discretizer and connections.
        contradictory = (
            (g["T_zone_C"] < g["T_out_input_C"] - 2)
            & (g["Q_net_model_W"] > 500)
            & (g["observed_dT_next_C"] < -0.1)
        )
        print("Windows with strongly positive estimated net input but falling T_zone:",
              int(contradictory.sum()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    args = p.parse_args()
    path = (Path("data/raw/twin4build") / args.case_id
            / "physics_trace_evaluation_only.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the physics-trace generator first.")
    analyze(pd.read_csv(path))


if __name__ == "__main__":
    main()
