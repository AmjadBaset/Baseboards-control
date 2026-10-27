#!/usr/bin/env python3
"""
Generate and diagnose one or many randomized Twin4Build buildings.

Examples
--------
One building (seed 1):
    python scripts/23_run_t4b_random_building_batch.py --n-buildings 1 --start-seed 1

Twenty reproducible buildings:
    python scripts/23_run_t4b_random_building_batch.py \
        --n-buildings 20 --start-seed 1 --batch-name paper_validation_v1

Selected seeds:
    python scripts/23_run_t4b_random_building_batch.py \
        --seeds 1,4,7-10 --batch-name selected_cases

The script calls, in order:
    t4b_ventilation_ablation_seed2001_v5.py
    11_classify_t4b_against_openstudio_bands.py
    12_summarize_t4b_room_diagnosis.py

It then calls 24_evaluate_t4b_random_building_batch.py unless --no-evaluate is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


GENERATOR = Path("scripts/t4b_ventilation_ablation_seed2001_v5.py")
PHYSICS_GATE = Path("scripts/28_screen_t4b_physics_v5.py")
CLASSIFIER = Path("scripts/11_classify_t4b_against_openstudio_bands.py")
SUMMARIZER = Path("scripts/12_summarize_t4b_room_diagnosis.py")
EVALUATOR = Path("scripts/24_evaluate_t4b_random_building_batch.py")

# Exactly these sources must remain unchanged when an existing diagnosis is
# reused. A matching simulation hash alone is not enough.
PIPELINE_FINGERPRINT_SOURCES = {
    "generator": GENERATOR,
    "physics_gate": PHYSICS_GATE,
    "classifier": CLASSIFIER,
    "summarizer": SUMMARIZER,
    "diagnostic_rules": Path("src/baseboard_diagnostics/diagnosis/rules.py"),
    "persistence_calibration": Path(
        "data/reference_bands/openstudio/apartment_1970/"
        "diagnosis_persistence_thresholds_new.csv"
    ),
    "healthy_reference_bands": Path(
        "data/reference_bands/openstudio/apartment_1970/"
        "apartment_baseboard_reference_bands_area_exposure_new.csv"
    ),
}


def pipeline_fingerprints() -> dict[str, str]:
    fingerprints = {}
    for name, path in PIPELINE_FINGERPRINT_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing freeze/provenance input {name}: {path}")
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints



def parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid seed range: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))

    unique = []
    seen = set()
    for seed in seeds:
        if seed not in seen:
            unique.append(seed)
            seen.add(seed)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible multi-building Twin4Build FDD batch."
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma/range specification, e.g. 1,4,7-10.",
    )
    parser.add_argument("--n-buildings", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--case-prefix", default="random_building_seed_")
    parser.add_argument("--batch-name", default=None)
    parser.add_argument("--fault-fraction", type=float, default=0.50)
    parser.add_argument("--ventilation-mode", choices=["none", "occupied"], default="none")
    parser.add_argument("--n-levels", type=int, default=None)
    parser.add_argument("--rooms-per-level", type=int, default=None)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-04-01")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse cases whose generated and diagnosed outputs already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to the single-building generator.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Generate cases only; do not run scripts 11 and 12.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Do not run the aggregate evaluator after the batch.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next seed if one case fails.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess:
    print("\n$ " + " ".join(cmd), flush=True)
    started = time.time()
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.write(result.stdout or "")
        f.write(f"\n[exit={result.returncode}, elapsed_s={elapsed:.2f}]\n")

    if result.stdout:
        print(result.stdout, end="")

    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")

    return result


def expected_paths(case_id: str) -> dict[str, Path]:
    raw_dir = Path(f"data/raw/twin4build/{case_id}")
    processed_dir = Path(f"data/processed/twin4build/{case_id}")
    return {
        "operational": raw_dir / "t4b_raw_timeseries.csv",
        "ground_truth": raw_dir / "ground_truth.csv",
        "building_manifest": raw_dir / "building_manifest.csv",
        "diagnosed": (
            processed_dir / "t4b_3h_windows_diagnosed_against_openstudio.csv"
        ),
        "room_summary": processed_dir / "t4b_room_level_diagnosis_summary.csv",
    }


def main() -> None:
    args = parse_args()

    if args.seeds:
        seeds = parse_seed_spec(args.seeds)
    else:
        if args.n_buildings < 1:
            raise ValueError("--n-buildings must be >= 1.")
        seeds = list(
            range(args.start_seed, args.start_seed + args.n_buildings)
        )

    if not seeds:
        raise ValueError("No seeds selected.")

    if args.batch_name:
        batch_name = args.batch_name
    elif len(seeds) == 1:
        batch_name = f"seed_{seeds[0]:04d}"
    else:
        batch_name = f"seeds_{min(seeds):04d}_{max(seeds):04d}"

    report_dir = Path("reports/random_building_validation") / batch_name
    log_dir = report_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    expected_fingerprints = pipeline_fingerprints() if not args.no_diagnostics else None

    print("Randomized-building validation batch")
    print(f"  batch: {batch_name}")
    print(f"  seeds: {seeds}")
    print(f"  fault fraction: {args.fault_fraction}")
    print(f"  ventilation mode: {args.ventilation_mode}")

    for i, seed in enumerate(seeds, start=1):
        case_id = f"{args.case_prefix}{seed:04d}"
        paths = expected_paths(case_id)
        log_path = log_dir / f"{case_id}.log"
        case_start = time.time()

        print("\n" + "=" * 90)
        print(f"CASE {i}/{len(seeds)}: {case_id} (seed={seed})")
        print("=" * 90)

        row = {
            "batch_name": batch_name,
            "case_id": case_id,
            "seed": seed,
            "status": "pending",
            "generation_status": "not_run",
            "physics_screen_status": "not_run",
            "classification_status": "not_run",
            "summary_status": "not_run",
            "operational_path": str(paths["operational"]),
            "ground_truth_path": str(paths["ground_truth"]),
            "building_manifest_path": str(paths["building_manifest"]),
            "diagnosed_path": str(paths["diagnosed"]),
            "room_summary_path": str(paths["room_summary"]),
            "log_path": str(log_path),
        }

        complete_existing = all(
            paths[key].exists()
            for key in (
                "operational",
                "ground_truth",
                "building_manifest",
                "diagnosed",
                "room_summary",
            )
        )

        if args.skip_existing and complete_existing:
            manifest_path = Path(paths["operational"]).parent / "simulation_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {}
            )
            current_sha = hashlib.sha256(GENERATOR.read_bytes()).hexdigest()
            provenance_path = Path(paths["operational"]).parent / "diagnostics_fingerprints.json"
            saved = (
                json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance_path.is_file() else {}
            )
            if (manifest.get("generator_sha256") != current_sha
                    or saved != expected_fingerprints):
                raise RuntimeError(
                    f"Refusing to reuse {case_id}: generator or diagnostic "
                    "fingerprints differ (or were not recorded). "
                    "Use fresh case IDs after any revision."
                )
            # Physically invalid results must never be silently reused.
            physics_result = run_command(
                [sys.executable, str(PHYSICS_GATE), "--case-id", case_id],
                log_path,
            )
            if physics_result.returncode != 0:
                raise RuntimeError(f"Refusing to reuse {case_id}: failed physics screening.")
            if manifest.get("ventilation_mode") != args.ventilation_mode:
                raise RuntimeError(f"Refusing to reuse {case_id}: ventilation mode differs.")
            print("Existing case has matching fingerprints and passed screening; reusing.")
            row.update(
                {
                    "status": "reused",
                    "generation_status": "reused",
                    "physics_screen_status": "reused_passed",
                    "classification_status": "reused",
                    "summary_status": "reused",
                    "elapsed_s": 0.0,
                }
            )
            rows.append(row)
            continue

        generator_cmd = [
            sys.executable,
            str(GENERATOR),
            "--seed",
            str(seed),
            "--case-id",
            case_id,
            "--fault-fraction",
            str(args.fault_fraction),
            "--ventilation-mode",
            args.ventilation_mode,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
        ]
        if args.n_levels is not None:
            generator_cmd += ["--n-levels", str(args.n_levels)]
        if args.rooms_per_level is not None:
            generator_cmd += ["--rooms-per-level", str(args.rooms_per_level)]
        if args.overwrite:
            generator_cmd.append("--overwrite")

        result = run_command(generator_cmd, log_path)
        row["generation_status"] = (
            "ok" if result.returncode == 0 else "failed"
        )

        if result.returncode != 0:
            row["status"] = "failed_generation"
            row["elapsed_s"] = time.time() - case_start
            rows.append(row)
            if args.continue_on_error:
                continue
            break

        physics_result = run_command(
            [sys.executable, str(PHYSICS_GATE), "--case-id", case_id],
            log_path,
        )
        row["physics_screen_status"] = (
            "passed" if physics_result.returncode == 0 else "failed"
        )
        if physics_result.returncode != 0:
            row["status"] = "failed_physics_screen"
            row["elapsed_s"] = time.time() - case_start
            rows.append(row)
            if args.continue_on_error:
                continue
            break

        if args.no_diagnostics:
            row["status"] = "generated_only"
            row["elapsed_s"] = time.time() - case_start
            rows.append(row)
            continue

        result = run_command(
            [
                sys.executable,
                str(CLASSIFIER),
                "--case-id",
                case_id,
            ],
            log_path,
        )
        row["classification_status"] = (
            "ok" if result.returncode == 0 else "failed"
        )

        if result.returncode != 0:
            row["status"] = "failed_classification"
            row["elapsed_s"] = time.time() - case_start
            rows.append(row)
            if args.continue_on_error:
                continue
            break

        result = run_command(
            [
                sys.executable,
                str(SUMMARIZER),
                "--case-id",
                case_id,
            ],
            log_path,
        )
        row["summary_status"] = "ok" if result.returncode == 0 else "failed"
        row["status"] = "ok" if result.returncode == 0 else "failed_summary"
        if result.returncode == 0:
            provenance_path = Path(paths["operational"]).parent / "diagnostics_fingerprints.json"
            provenance_path.write_text(
                json.dumps(expected_fingerprints, indent=2) + "\n",
                encoding="utf-8",
            )
        row["elapsed_s"] = time.time() - case_start
        rows.append(row)

        if result.returncode != 0 and not args.continue_on_error:
            break

    batch_manifest = pd.DataFrame(rows)
    batch_manifest_path = report_dir / "batch_manifest.csv"
    batch_manifest.to_csv(batch_manifest_path, index=False)
    # Always retain failure counts; evaluation on the valid subset is never
    # a substitute for reporting simulations rejected by screening.
    if len(batch_manifest):
        print("  attempted:", len(batch_manifest),
              "; completed and valid:", int(batch_manifest["status"].isin(["ok", "reused"]).sum()),
              "; rejected/error:", int((~batch_manifest["status"].isin(["ok", "reused"])).sum()))

    print("\n" + "=" * 90)
    print("BATCH STATUS")
    print("=" * 90)
    if not batch_manifest.empty:
        print(
            batch_manifest[
                [
                    "case_id",
                    "seed",
                    "status",
                    "generation_status",
                    "classification_status",
                    "summary_status",
                    "elapsed_s",
                ]
            ].to_string(index=False)
        )
    print(f"\nSaved batch manifest: {batch_manifest_path}")

    usable = batch_manifest[
        batch_manifest["status"].isin(["ok", "reused"])
    ]

    if (
        not args.no_evaluate
        and not args.no_diagnostics
        and not usable.empty
    ):
        result = run_command(
            [
                sys.executable,
                str(EVALUATOR),
                "--batch-manifest",
                str(batch_manifest_path),
                "--out-dir",
                str(report_dir),
            ],
            report_dir / "evaluation.log",
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
