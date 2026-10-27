#!/usr/bin/env python3
"""Strict, automated gross-physics/signal-quality gate for v5 Twin4Build runs.

This is NOT a proof of physical realism. Only measured operational signals and
simulation metadata are read; no fault ground truth is used. In particular,
T_zone < T_out at a given time can be legitimate during rapid outdoor warming.
Review-only findings do not invalidate the simulation automatically.

Return code 0: screening passed; 2: rejected, do not run/report FDD metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("timestamp", "zone", "T_zone", "T_out", "T_set", "Q_bb",
            "m_dot", "T_supply", "T_return", "valve_position")


def screen(case_dir: Path) -> dict:
    raw_path = case_dir / "t4b_raw_timeseries.csv"
    manifest_path = case_dir / "simulation_manifest.json"
    if not raw_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete simulation output in {case_dir}")
    raw = pd.read_csv(raw_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing operational columns: {missing}")
    if not len(raw):
        raise ValueError("Empty raw operational data")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")

    columns = [c for c in raw.columns if c not in ("timestamp", "zone", "baseboard", "exposure_group", "period")]
    # All known numerical sensor fields must be numeric and finite, but retain any
    # optional textual metadata introduced in later generator versions.
    sensor_cols = [c for c in REQUIRED if c not in ("timestamp", "zone")]
    for c in sensor_cols:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    errors: list[str] = []
    warnings: list[str] = []
    if raw["zone"].isna().any():
        errors.append("missing_zone")
    if raw.duplicated(["zone", "timestamp"]).any():
        errors.append("duplicate_zone_timestamps")
    if not np.isfinite(raw[sensor_cols].to_numpy(dtype="float64")).all():
        errors.append("nonfinite_required_sensors")

    timestep = int(manifest.get("timestep_seconds", 900))
    tstart = pd.Timestamp(manifest["start_time"])
    tend = pd.Timestamp(manifest["end_time"])
    expected_steps = int((tend - tstart).total_seconds() / timestep)
    expected_rooms = int(manifest["total_rooms"])
    if expected_steps <= 0 or expected_rooms <= 0:
        errors.append("invalid_manifest_dimensions")
    if raw["zone"].nunique() != expected_rooms:
        errors.append("room_count_mismatch")
    if len(raw) != expected_steps * expected_rooms:
        errors.append("row_count_mismatch")

    room_details = []
    for zone, grp in raw.groupby("zone", sort=True):
        g = grp.sort_values("timestamp").reset_index(drop=True)
        steps = g["timestamp"].diff().dt.total_seconds().dropna()
        if len(g) != expected_steps:
            errors.append(f"incomplete_room:{zone}")
        if len(steps) and not (steps == timestep).all():
            errors.append(f"nonuniform_timestep:{zone}")
        jump = g["T_zone"].diff().abs().gt(5.0).sum()
        negative_heat = g["Q_bb"].lt(-10.0).sum()
        negative_flow = g["m_dot"].lt(-1e-9).sum()
        negative_valve = g["valve_position"].lt(-1e-6).sum()
        high_valve = g["valve_position"].gt(1.000001).sum()
        gross_temp = ((g["T_zone"] < -30) | (g["T_zone"] > 55)).sum()
        # T_zone < outdoor-2 is a *warning*; thermal lag can make it legitimate.
        cold_ambient = (g["T_zone"] < g["T_out"] - 2).sum()
        if jump:
            errors.append(f"temp_jump_gt5C_per_step:{zone}:{int(jump)}")
        if negative_heat:
            errors.append(f"negative_radiator_power_below_minus10W:{zone}:{int(negative_heat)}")
        if negative_flow:
            errors.append(f"negative_water_flow:{zone}:{int(negative_flow)}")
        if negative_valve or high_valve:
            errors.append(f"valve_outside_0_1:{zone}")
        if gross_temp:
            errors.append(f"gross_room_temperature:{zone}:{int(gross_temp)}")
        if cold_ambient:
            warnings.append(f"below_current_outdoor_minus_2C:{zone}:{int(cold_ambient)}")
        room_details.append({
            "zone": zone, "n_steps": int(len(g)),
            "min_T_zone_C": float(g["T_zone"].min()),
            "max_T_zone_C": float(g["T_zone"].max()),
            "max_abs_delta_T_C_per_step": float(g["T_zone"].diff().abs().max()) if len(g)>1 else 0.,
            "below_current_outdoor_minus_2C_steps": int(cold_ambient),
            "negative_heater_below_minus10W_steps": int(negative_heat),
        })
    # Bypass of schedule-cache fix would be caught by the generator's required
    # INPUT VALIDATION PASSED runtime guard; the manifest alone is not proof.
    return {
        "case_id": case_dir.name,
        "screen_status": "PASS" if not errors else "FAIL",
        "n_rooms": int(raw["zone"].nunique()),
        "n_rows": int(len(raw)),
        "expected_rows": expected_steps * expected_rooms,
        "ventilation_mode": manifest.get("ventilation_mode"),
        "generator_sha256": manifest.get("generator_sha256"),
        "hard_errors": errors,
        "review_warnings": warnings,
        "room_details": room_details,
        "disclaimer": "Operational gross-physics gate; not full hydronic/energy-balance validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--root", default="data/raw/twin4build")
    args = parser.parse_args()
    directory = Path(args.root) / args.case_id
    result = screen(directory)
    output = directory / "v5_physics_screen.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PHYSICS SCREEN: {result['screen_status']} — {args.case_id}")
    print(f"  rooms={result['n_rooms']}; rows={result['n_rows']}/{result['expected_rows']}")
    for x in result["hard_errors"]:
        print(f"  FAIL: {x}")
    for x in result["review_warnings"][:12]:
        print(f"  REVIEW: {x}")
    if len(result["review_warnings"])>12:
        print(f"  ... plus {len(result['review_warnings'])-12} more review warnings")
    print(f"  saved: {output}")
    raise SystemExit(0 if result["screen_status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
