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
    21_run_t4b_random_building_case.py
    11_classify_t4b_against_openstudio_bands.py
    12_summarize_t4b_room_diagnosis.py

It then calls 24_evaluate_t4b_random_building_batch.py unless --no-evaluate is used.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


GENERATOR = Path("scripts/21_run_t4b_random_building_case.py")
CLASSIFIER = Path("scripts/11_classify_t4b_against_openstudio_bands.py")
SUMMARIZER = Path("scripts/12_summarize_t4b_room_diagnosis.py")
EVALUATOR = Path("scripts/24_evaluate_t4b_random_building_batch.py")


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

    print("Randomized-building validation batch")
    print(f"  batch: {batch_name}")
    print(f"  seeds: {seeds}")
    print(f"  fault fraction: {args.fault_fraction}")

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
            print("Existing complete case found; reusing it.")
            row.update(
                {
                    "status": "reused",
                    "generation_status": "reused",
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
        row["elapsed_s"] = time.time() - case_start
        rows.append(row)

        if result.returncode != 0 and not args.continue_on_error:
            break

    batch_manifest = pd.DataFrame(rows)
    batch_manifest_path = report_dir / "batch_manifest.csv"
    batch_manifest.to_csv(batch_manifest_path, index=False)

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
