#!/usr/bin/env python3
"""
Run 1 h / 3 h / 6 h window-length sensitivity across a randomized-building batch.

Healthy reference bands and diagnosis-persistence thresholds are rebuilt
independently for every temporal scale. The script uses the current classifier,
current diagnosis rules, and current room-level summarizer from the repository.
Ground truth is loaded only after room diagnosis for evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from baseboard_diagnostics.bands.build_reference_bands import build_reference_bands
from baseboard_diagnostics.diagnosis.rules import apply_rules_to_labeled_windows
from baseboard_diagnostics.diagnosis.persistence import (
    calculate_diagnosis_fractions,
    calibrate_persistence_thresholds,
)
from baseboard_diagnostics.features.timestep_features import add_timestep_features
from baseboard_diagnostics.features.window_features import create_window_features


WINDOWS = ("1h", "3h", "6h")
DEFAULT_HEALTHY_TIMESTEP = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_clean_area_exposure_new.csv"
)


def load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLASSIFIER = load_script_module(
    "t4b_classifier_sensitivity",
    Path("scripts/11_classify_t4b_against_openstudio_bands.py"),
)
SUMMARIZER = load_script_module(
    "t4b_summarizer_sensitivity",
    Path("scripts/12_summarize_t4b_room_diagnosis.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--healthy-timestep", default=str(DEFAULT_HEALTHY_TIMESTEP))
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def expected_timesteps(window: str) -> int:
    return int(pd.Timedelta(window) / pd.Timedelta(minutes=15))


def add_window_features(raw: pd.DataFrame, window: str) -> pd.DataFrame:
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    forbidden = { "fault_type", "fault_severity", "fault_start", "fault_start_step", "restriction_factor", "fixed_valve_position", "fixed_valve_regime", "fixed_valve_quantile", "valve_upper_limit", "valve_lower_limit", "valve_limit_quantile", "valve_binding_fraction", "valve_binding_count", "valve_mean_absolute_position_change", "valve_post_fault_step_count", "valve_fault_activated", "controller_signal", "valve_max_flow", "healthy_valve_max_flow_kg_s" }.intersection(df.columns)
    if forbidden:
        raise ValueError(
            f"Ground truth/controller leak in window sensitivity input: "
            f"{sorted(forbidden)}"
        )

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

    if "valve_position" not in df.columns:
        return windows

    n_steps = expected_timesteps(window)
    rows = []

    for keys, g in df.groupby(
        ["case_id", "zone", "baseboard"],
        dropna=False,
    ):
        case_id, zone, baseboard = keys
        g = g.sort_values("timestamp").reset_index(drop=True)
        g["window_id"] = np.arange(len(g)) // n_steps

        for _, w in g.groupby("window_id"):
            if len(w) != n_steps:
                continue
            valve = pd.to_numeric(w["valve_position"], errors="coerce")
            valve_diff = valve.diff().dropna()
            rows.append(
                {
                    "case_id": case_id,
                    "zone": zone,
                    "baseboard": baseboard,
                    "window_start": w["timestamp"].iloc[0],
                    "valve_position_mean": valve.mean(),
                    "valve_position_max": valve.max(),
                    "valve_position_min": valve.min(),
                    "valve_position_range": valve.max() - valve.min(),
                    "valve_position_std": valve.std(),
                    "valve_position_oscillation_index": valve_diff.abs().sum(),
                }
            )

    if not rows:
        return windows

    extra = pd.DataFrame(rows)
    windows["window_start"] = pd.to_datetime(windows["window_start"])
    extra["window_start"] = pd.to_datetime(extra["window_start"])
    merged = windows.merge(
        extra,
        on=["case_id", "zone", "baseboard", "window_start"],
        how="left",
        validate="one_to_one",
    )
    if merged["valve_position_mean"].isna().any():
        raise ValueError("Sensitivity valve-window alignment failed.")
    return merged


def truth_eval_family(fault_type: str) -> str:
    ft = str(fault_type).strip().lower()
    if ft == "healthy":
        return "normal"
    if ft == "restricted_flow":
        return "restricted_or_hydraulic"
    if ft in {"fixed_valve", "limited_opening", "limited_closing"}:
        return "valve_fault"
    return ft


def predicted_eval_family(family: str) -> str:
    mapping = {
        "normal": "normal",
        "restricted_or_hydraulic": "restricted_or_hydraulic",
        "stuck_open_or_leakage": "valve_fault",
        "stuck_low_valve": "valve_fault",
        "insufficient_heat_output": "insufficient_heat_output",
        "oversupply_or_overheating": "oversupply_or_overheating",
        "comfort_violation_unclassified": "unclassified",
    }
    return mapping.get(str(family), str(family))


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else np.nan


def metrics(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["true_fault"].astype(bool)
    y_pred = df["predicted_fault"].astype(bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, len(df))
    f1 = (
        2 * precision * recall / (precision + recall)
        if not np.isnan(precision)
        and not np.isnan(recall)
        and precision + recall > 0
        else np.nan
    )
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


def main() -> None:
    args = parse_args()
    batch_manifest_path = Path(args.batch_manifest)
    out_dir = Path(args.out_dir) if args.out_dir else batch_manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = pd.read_csv(batch_manifest_path)
    if "status" in batch.columns:
        batch = batch[batch["status"].isin(["ok", "reused"])].copy()
    if batch.empty:
        raise ValueError("No completed cases found in batch manifest.")

    healthy = pd.read_csv(args.healthy_timestep)
    healthy["timestamp"] = pd.to_datetime(healthy["timestamp"])

    all_room_rows = []
    summary_rows = []
    threshold_rows = []

    for window in WINDOWS:
        print("\n" + "=" * 90)
        print(f"WINDOW = {window}")
        print("=" * 90)

        healthy_windows = create_window_features(
            healthy,
            window=window,
            heating_active_fraction_threshold=0.25,
        )
        bands = build_reference_bands(
            healthy_windows,
            heating_windows_only=True,
        )

        # Calibrate diagnosis persistence independently at this temporal scale.
        healthy_labeled = CLASSIFIER.classify_with_relaxed_bands(
            healthy_windows,
            bands,
        )
        healthy_diagnosed = apply_rules_to_labeled_windows(healthy_labeled)
        healthy_fractions = calculate_diagnosis_fractions(healthy_diagnosed)
        threshold_table = calibrate_persistence_thresholds(
            healthy_fractions,
            quantile=0.95,
            minimum_floor=0.02,
        )
        persistence_thresholds = {
            str(row["diagnosis"]): float(row["persistence_threshold"])
            for _, row in threshold_table.iterrows()
        }
        threshold_table = threshold_table.copy()
        threshold_table.insert(0, "window", window)
        threshold_rows.append(threshold_table)

        window_room_rows = []

        for _, case in batch.iterrows():
            case_id = str(case["case_id"])
            raw = pd.read_csv(case["operational_path"])
            truth = pd.read_csv(case["ground_truth_path"])
            truth_by_zone = truth.set_index("zone").to_dict("index")

            t4b_windows = add_window_features(raw, window)
            labeled = CLASSIFIER.classify_with_relaxed_bands(t4b_windows, bands)
            diagnosed = apply_rules_to_labeled_windows(labeled)

            for zone, g in diagnosed.groupby("zone", dropna=False):
                zone = str(zone)
                room = SUMMARIZER.summarize_room(
                    zone=zone,
                    g=g.copy(),
                    raw_meta_row=None,
                    persistence_thresholds=persistence_thresholds,
                    persistence_fallback=0.02,
                )
                gt = truth_by_zone[zone]
                room.update(
                    {
                        "window": window,
                        "case_id": case_id,
                        "seed": gt.get("seed"),
                        "fault_type": gt["fault_type"],
                        "fault_severity": gt.get("fault_severity", "none"),
                        "restriction_factor": gt.get("restriction_factor"),
                    }
                )
                room["true_fault"] = room["fault_type"] != "healthy"
                room["predicted_fault"] = room["dominant_fault_family"] != "normal"
                room["true_eval_family"] = truth_eval_family(room["fault_type"])
                room["predicted_eval_family"] = predicted_eval_family(
                    room["dominant_fault_family"]
                )
                room["family_correct"] = (
                    room["true_eval_family"] == room["predicted_eval_family"]
                )
                window_room_rows.append(room)

        window_rooms = pd.DataFrame(window_room_rows)
        all_room_rows.extend(window_room_rows)
        m = metrics(window_rooms)
        faulty = window_rooms[window_rooms["true_fault"]]
        healthy_rooms = window_rooms[~window_rooms["true_fault"]]
        summary_rows.append(
            {
                "window": window,
                "n_buildings": window_rooms["case_id"].nunique(),
                "n_rooms": len(window_rooms),
                **m,
                "family_accuracy_all_rooms": float(
                    window_rooms["family_correct"].mean()
                ),
                "faulty_family_accuracy": (
                    float(faulty["family_correct"].mean())
                    if len(faulty)
                    else np.nan
                ),
                "healthy_mean_abnormal_fraction": float(
                    healthy_rooms["abnormal_fraction"].mean()
                ),
                "faulty_mean_abnormal_fraction": float(
                    faulty["abnormal_fraction"].mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    rooms = pd.DataFrame(all_room_rows)

    thresholds = pd.concat(threshold_rows, ignore_index=True)

    summary.to_csv(out_dir / "window_sensitivity_batch_summary.csv", index=False)
    rooms.to_csv(out_dir / "window_sensitivity_batch_rooms.csv", index=False)
    thresholds.to_csv(
        out_dir / "window_sensitivity_persistence_thresholds.csv",
        index=False,
    )

    print("\n" + "=" * 90)
    print("BATCH WINDOW-SENSITIVITY SUMMARY")
    print("=" * 90)
    print(summary.round(4).to_string(index=False))
    print("\nSaved:")
    print(out_dir / "window_sensitivity_batch_summary.csv")
    print(out_dir / "window_sensitivity_batch_rooms.csv")
    print(out_dir / "window_sensitivity_persistence_thresholds.csv")


if __name__ == "__main__":
    main()
