#!/usr/bin/env python3
"""
Window-length sensitivity analysis for the room-level hydronic FDD pipeline.

Compares 1 h, 3 h, and 6 h diagnostic windows while keeping:
- the same 15-min healthy timestep data,
- the same Twin4Build raw case,
- the same context-matching hierarchy,
- the same diagnostic rules,
- the same room-level aggregation logic.

Important:
Injected fault metadata (fault_type, valve_max_flow) is never passed to the
diagnostic pipeline. fault_type is retained only as validation ground truth.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from baseboard_diagnostics.bands.build_reference_bands import build_reference_bands
from baseboard_diagnostics.diagnosis.rules import apply_rules_to_labeled_windows
from baseboard_diagnostics.features.timestep_features import add_timestep_features
from baseboard_diagnostics.features.window_features import create_window_features


WINDOWS = ("1h", "3h", "6h")

DEFAULT_HEALTHY_TIMESTEP = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_clean_area_exposure_new.csv"
)

CONTROLLED_SIX_ROOM_TRUTH = {
    "ROOM_ROOM_1": "healthy",
    "ROOM_ROOM_2": "healthy",
    "ROOM_ROOM_3": "restricted_flow",
    "ROOM_ROOM_4": "healthy",
    "ROOM_ROOM_5": "stuck_low_valve",
    "ROOM_ROOM_6": "healthy",
}


def load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLASSIFIER = load_script_module(
    "t4b_classifier",
    Path("scripts/11_classify_t4b_against_openstudio_bands.py"),
)
SUMMARIZER = load_script_module(
    "t4b_summarizer",
    Path("scripts/12_summarize_t4b_room_diagnosis.py"),
)


def expected_timesteps(window: str) -> int:
    return int(pd.Timedelta(window) / pd.Timedelta(minutes=15))


def add_t4b_window_features(raw: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    Parameterized equivalent of script 11's T4B feature builder.

    fault_type and valve_max_flow are explicitly removed before diagnosis so
    injected fault information cannot leak into the diagnostic features.
    """
    df = raw.drop(columns=["fault_type", "valve_max_flow"], errors="ignore").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = add_timestep_features(
        df,
        comfort_tolerance=0.5,
        heating_delta_t_threshold=0.8,
        heating_power_threshold=20.0,
    )

    windows = create_window_features(
        df,
        window=window,
        heating_active_fraction_threshold=0.25,
    )

    n_steps = expected_timesteps(window)
    extra_rows = []

    for keys, g in df.groupby(["case_id", "zone", "baseboard"], dropna=False):
        case_id, zone, baseboard = keys
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["window_id"] = np.arange(len(g)) // n_steps

        for _, w in g.groupby("window_id"):
            if len(w) != n_steps:
                continue

            valve_position = pd.to_numeric(w["valve_position"], errors="coerce")
            valve_diff = valve_position.diff().dropna()

            extra_rows.append(
                {
                    "case_id": case_id,
                    "zone": zone,
                    "baseboard": baseboard,
                    "window_start": w["timestamp"].iloc[0],
                    "valve_position_mean": valve_position.mean(),
                    "valve_position_max": valve_position.max(),
                    "valve_position_min": valve_position.min(),
                    "valve_position_range": valve_position.max() - valve_position.min(),
                    "valve_position_std": valve_position.std(),
                    "valve_position_oscillation_index": valve_diff.abs().sum(),
                    "controller_signal_mean": pd.to_numeric(
                        w["controller_signal"], errors="coerce"
                    ).mean(),
                    "controller_signal_max": pd.to_numeric(
                        w["controller_signal"], errors="coerce"
                    ).max(),
                }
            )

    extra = pd.DataFrame(extra_rows)

    windows["window_start"] = pd.to_datetime(windows["window_start"])
    if not extra.empty:
        extra["window_start"] = pd.to_datetime(extra["window_start"])
        windows = windows.merge(
            extra,
            on=["case_id", "zone", "baseboard", "window_start"],
            how="left",
        )

    return windows


def ground_truth_by_zone(raw: pd.DataFrame) -> dict[str, str]:
    if "fault_type" in raw.columns:
        truth = (
            raw[["zone", "fault_type"]]
            .dropna(subset=["zone"])
            .groupby("zone", as_index=False)["fault_type"]
            .first()
        )
        result = {
            str(row["zone"]): str(row["fault_type"])
            for _, row in truth.iterrows()
        }
        if any(v.lower() not in {"unknown", "nan", ""} for v in result.values()):
            return result

    zones = set(raw["zone"].astype(str).unique())
    if zones.issubset(CONTROLLED_SIX_ROOM_TRUTH):
        return {z: CONTROLLED_SIX_ROOM_TRUTH[z] for z in zones}

    raise ValueError(
        "No usable fault_type ground truth was found, and the case does not "
        "match the known controlled six-room layout."
    )


def is_faulty_truth(fault_type: str) -> bool:
    return str(fault_type).strip().lower() not in {
        "healthy",
        "normal",
        "none",
        "no_fault",
        "no fault",
    }


def truth_family(fault_type: str) -> str:
    ft = str(fault_type).strip().lower()
    if not is_faulty_truth(ft):
        return "normal"
    if ft in {"restricted_flow", "severe_restricted_flow", "restriction"}:
        return "restricted_or_hydraulic"
    if ft in {
        "stuck_valve",
        "stuck_low_valve",
        "stuck_open_valve",
        "valve_leakage",
    }:
        return "stuck_valve"
    return ft


def predicted_family_for_evaluation(dominant_family: str) -> str:
    pf = str(dominant_family).strip()
    if pf == "normal":
        return "normal"
    if pf == "restricted_or_hydraulic":
        return "restricted_or_hydraulic"
    if pf in {
        "stuck_open_or_leakage",
        "stuck_low_valve",
        "oversupply_or_overheating",
    }:
        return "stuck_valve"
    return pf


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def binary_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    y_true = eval_df["true_fault"].astype(bool)
    y_pred = eval_df["predicted_fault"].astype(bool)

    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall) if (
        not np.isnan(precision)
        and not np.isnan(recall)
        and (precision + recall) > 0
    ) else np.nan
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "F1": f1,
        "accuracy": accuracy,
    }


def summarize_case(
    diagnosed: pd.DataFrame,
    truth: dict[str, str],
) -> pd.DataFrame:
    rows = []

    for zone, g in diagnosed.groupby("zone", dropna=False):
        zone = str(zone)
        raw_meta_row = {"zone": zone, "fault_type": truth.get(zone, "unknown")}

        summary = SUMMARIZER.summarize_room(
            zone=zone,
            g=g.copy(),
            raw_meta_row=raw_meta_row,
        )
        rows.append(summary)

    return pd.DataFrame(rows)


def run_one_window(
    healthy_timestep: pd.DataFrame,
    raw: pd.DataFrame,
    truth: dict[str, str],
    window: str,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    healthy_windows = create_window_features(
        healthy_timestep,
        window=window,
        heating_active_fraction_threshold=0.25,
    )

    bands = build_reference_bands(
        healthy_windows,
        heating_windows_only=True,
    )

    t4b_windows = add_t4b_window_features(raw, window=window)
    labeled = CLASSIFIER.classify_with_relaxed_bands(t4b_windows, bands)
    diagnosed = apply_rules_to_labeled_windows(labeled)
    room_summary = summarize_case(diagnosed, truth)

    eval_df = room_summary[
        [
            "zone",
            "dominant_fault_family",
            "main_diagnosis",
            "main_diagnosis_fraction",
            "severity",
            "comfort_violation_fraction",
            "normal_fraction",
            "abnormal_fraction",
            "n_windows",
        ]
    ].copy()

    eval_df["fault_type"] = eval_df["zone"].map(truth)
    eval_df["true_fault"] = eval_df["fault_type"].map(is_faulty_truth)
    eval_df["predicted_fault"] = eval_df["dominant_fault_family"].ne("normal")
    eval_df["true_family"] = eval_df["fault_type"].map(truth_family)
    eval_df["predicted_family"] = eval_df["dominant_fault_family"].map(
        predicted_family_for_evaluation
    )
    eval_df["family_correct"] = (
        eval_df["true_family"] == eval_df["predicted_family"]
    )
    eval_df["window"] = window
    eval_df["window_hours"] = pd.Timedelta(window).total_seconds() / 3600.0

    metrics = binary_metrics(eval_df)
    metrics["window"] = window
    metrics["window_hours"] = pd.Timedelta(window).total_seconds() / 3600.0
    metrics["n_rooms"] = len(eval_df)
    metrics["n_faulty_rooms"] = int(eval_df["true_fault"].sum())
    metrics["n_healthy_rooms"] = int((~eval_df["true_fault"]).sum())
    metrics["family_accuracy_all_rooms"] = float(eval_df["family_correct"].mean())

    faulty = eval_df[eval_df["true_fault"]]
    metrics["family_accuracy_faulty_rooms"] = (
        float(faulty["family_correct"].mean()) if not faulty.empty else np.nan
    )

    healthy = eval_df[~eval_df["true_fault"]]
    metrics["healthy_mean_abnormal_fraction"] = (
        float(healthy["abnormal_fraction"].mean()) if not healthy.empty else np.nan
    )
    metrics["faulty_mean_abnormal_fraction"] = (
        float(faulty["abnormal_fraction"].mean()) if not faulty.empty else np.nan
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    bands.to_csv(out_dir / f"healthy_reference_bands_{window}.csv", index=False)
    diagnosed.to_csv(out_dir / f"diagnosed_windows_{window}.csv", index=False)
    eval_df.to_csv(out_dir / f"room_evaluation_{window}.csv", index=False)

    return eval_df, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare 1 h, 3 h, and 6 h FDD window lengths."
    )
    p.add_argument(
        "--case-id",
        default="two_room_restricted_flow",
        help="Twin4Build case folder under data/raw/twin4build/.",
    )
    p.add_argument(
        "--healthy-timestep",
        default=str(DEFAULT_HEALTHY_TIMESTEP),
        help="Healthy 15-min timestep feature CSV used to rebuild reference bands.",
    )
    p.add_argument(
        "--raw",
        default=None,
        help="Optional raw Twin4Build CSV; overrides --case-id.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    healthy_path = Path(args.healthy_timestep)
    raw_path = (
        Path(args.raw)
        if args.raw
        else Path(f"data/raw/twin4build/{args.case_id}/t4b_raw_timeseries.csv")
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"data/processed/twin4build/{args.case_id}/window_sensitivity")
    )

    if not healthy_path.exists():
        raise FileNotFoundError(f"Healthy timestep file not found: {healthy_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Twin4Build raw file not found: {raw_path}")

    healthy = pd.read_csv(healthy_path)
    healthy["timestamp"] = pd.to_datetime(healthy["timestamp"])

    raw = pd.read_csv(raw_path)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    truth = ground_truth_by_zone(raw)

    all_rooms = []
    metric_rows = []

    for window in WINDOWS:
        print(f"\n=== Testing {window} window ===")
        room_eval, metrics = run_one_window(
            healthy_timestep=healthy,
            raw=raw,
            truth=truth,
            window=window,
            out_dir=out_dir,
        )
        all_rooms.append(room_eval)
        metric_rows.append(metrics)

        print(
            room_eval[
                [
                    "zone",
                    "fault_type",
                    "true_family",
                    "predicted_family",
                    "main_diagnosis",
                    "severity",
                    "abnormal_fraction",
                ]
            ].to_string(index=False)
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("window_hours")
    rooms_df = pd.concat(all_rooms, ignore_index=True)

    metrics_df.to_csv(out_dir / "window_sensitivity_metrics.csv", index=False)
    rooms_df.to_csv(out_dir / "window_sensitivity_room_results.csv", index=False)

    print("\n=== Window-length comparison ===")
    cols = [
        "window",
        "n_rooms",
        "TP",
        "TN",
        "FP",
        "FN",
        "precision",
        "recall",
        "specificity",
        "F1",
        "accuracy",
        "family_accuracy_faulty_rooms",
        "healthy_mean_abnormal_fraction",
        "faulty_mean_abnormal_fraction",
    ]
    print(metrics_df[cols].round(4).to_string(index=False))

    print(f"\nSaved sensitivity outputs to: {out_dir}")


if __name__ == "__main__":
    main()
