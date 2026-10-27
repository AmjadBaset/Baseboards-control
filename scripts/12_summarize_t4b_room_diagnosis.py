"""
Summarize window-level Twin4Build diagnoses into one room-level table.

Important architecture:
- Diagnostic reasoning belongs in rules.py and the classification script.
- This script aggregates window-level outputs only.
- Injected fault_type is validation/reporting metadata and is never used to
  choose a diagnosis.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


NORMAL_DIAGNOSIS = "normal_or_no_rule_triggered"

DEFAULT_PERSISTENCE_THRESHOLDS_PATH = Path(
    "data/reference_bands/openstudio/apartment_1970/"
    "diagnosis_persistence_thresholds_new.csv"
)

# Diagnoses absent from the healthy calibration receive this conservative floor.
DEFAULT_PERSISTENCE_FALLBACK = 0.02


# Pure presentation mapping. These diagnoses are already assigned by rules.py;
# aggregation must not infer compensation from comfort alone.
HYDRAULIC_OPERATIONAL_STATUS = {
    "possible_compensated_hydraulic_restriction": (
        "compensation_suspected_comfort_maintained"
    ),
    "hydraulic_restriction_degraded": (
        "delivery_degraded_comfort_maintained"
    ),
    "hydraulic_restriction_with_comfort_impact": "comfort_impacted",
    "possible_flow_or_control_instability": "flow_or_control_instability",
}

def hydraulic_operational_status_for_room(main_diagnosis: str, severity: str) -> str:
    """Report existing hydraulic signature while respecting whole-room impact.

    The window-level compensated/degraded diagnoses describe the evidence within
    selected windows. They DO NOT prove comfort was maintained across the full
    analysis period. Reuse the existing room severity criterion to avoid
    contradictory room-level labels, without adding or changing thresholds.
    """
    label = HYDRAULIC_OPERATIONAL_STATUS.get(
        main_diagnosis, "not_hydraulic_main_diagnosis"
    )
    if severity == "fault":
        if main_diagnosis == "possible_compensated_hydraulic_restriction":
            return "compensation_signature_with_room_comfort_impact"
        if main_diagnosis == "hydraulic_restriction_degraded":
            return "delivery_degraded_with_room_comfort_impact"
    return label


HYDRAULIC_EVIDENCE_COLUMNS = {
    "possible_compensated_hydraulic_restriction": (
        "hydraulic_compensation_evidence_fraction"
    ),
    "hydraulic_restriction_degraded": (
        "hydraulic_degraded_evidence_fraction"
    ),
    "hydraulic_restriction_with_comfort_impact": (
        "hydraulic_comfort_impact_evidence_fraction"
    ),
    "possible_flow_or_control_instability": (
        "hydraulic_instability_evidence_fraction"
    ),
}


FAMILY_BY_DIAGNOSIS = {
    # Valve faults
    "possible_valve_leakage_or_stuck_open": "stuck_open_or_leakage",
    "possible_partially_closed_or_stuck_low_valve": "stuck_low_valve",

    # Restricted / hydraulic
    "hydraulic_restriction_with_comfort_impact": "restricted_or_hydraulic",
    "hydraulic_restriction_degraded": "restricted_or_hydraulic",
    "possible_compensated_hydraulic_restriction": "restricted_or_hydraulic",
    "possible_flow_or_control_instability": "restricted_or_hydraulic",

    # Heat-output / capacity
    "insufficient_heat_output": "insufficient_heat_output",
    "possible_emitter_undersizing_or_high_load": "insufficient_heat_output",

    # Oversupply / overheating
    "possible_excessive_heating_capacity_or_oversupply": "oversupply_or_overheating",
    "possible_oversupply_or_overheating": "oversupply_or_overheating",
    "possible_nonhydronic_or_unexplained_overheating": "oversupply_or_overheating",

    # Fallback
    "comfort_violation_without_matched_diagnostic_rule": "comfort_violation_unclassified",
}


# Priority is used only to break ties after diagnosis persistence has been
# considered. More specific physical diagnoses therefore precede generic ones.
DIAGNOSIS_PRIORITY = [
    "possible_valve_leakage_or_stuck_open",
    "possible_partially_closed_or_stuck_low_valve",

    "hydraulic_restriction_with_comfort_impact",
    "hydraulic_restriction_degraded",
    "possible_compensated_hydraulic_restriction",

    "insufficient_heat_output",
    "possible_emitter_undersizing_or_high_load",

    "possible_excessive_heating_capacity_or_oversupply",
    "possible_oversupply_or_overheating",
    "possible_nonhydronic_or_unexplained_overheating",

    "possible_flow_or_control_instability",
    "comfort_violation_without_matched_diagnostic_rule",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        default="two_room_restricted_flow",
        help=(
            "Case id under data/raw/twin4build "
            "and data/processed/twin4build."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional diagnosed 3-hour window CSV path.",
    )
    parser.add_argument(
        "--raw",
        default=None,
        help="Optional raw Twin4Build time-series CSV path.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional room-level summary CSV path.",
    )
    parser.add_argument(
        "--persistence-thresholds",
        default=str(DEFAULT_PERSISTENCE_THRESHOLDS_PATH),
        help=(
            "Healthy-reference diagnosis-persistence calibration CSV. "
            "Diagnoses absent from this file use --persistence-fallback."
        ),
    )
    parser.add_argument(
        "--persistence-fallback",
        type=float,
        default=DEFAULT_PERSISTENCE_FALLBACK,
        help=(
            "Persistence threshold used for diagnoses absent from the "
            "healthy calibration CSV (default: 0.02)."
        ),
    )
    return parser.parse_args()


def severity_from_comfort_violation(
    comfort_violation: float,
) -> str:
    """
    Severity of a *persistently diagnosed* room based on comfort impact.

    Comfort by itself does not create a room-level diagnosis. A room with no
    diagnosis surviving persistence remains normal; the raw comfort-violation
    fraction is still retained in the output for review.
    """
    if comfort_violation >= 0.20:
        return "fault"
    return "warning"


def final_room_severity(
    comfort_violation: float,
    main_diagnosis: str,
) -> str:
    """
    Assign room severity only after a diagnosis survives persistence.
    """
    if main_diagnosis in {
        NORMAL_DIAGNOSIS,
        "normal",
        "normal_or_no_rule_triggered",
    }:
        return "normal"

    return severity_from_comfort_violation(comfort_violation)


def load_persistence_thresholds(
    path: Path,
) -> dict[str, float]:
    """Load diagnosis-specific healthy-operation persistence thresholds."""
    if not path.exists():
        raise FileNotFoundError(
            f"Persistence-threshold file not found: {path}"
        )

    df = pd.read_csv(path)
    required = {"diagnosis", "persistence_threshold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing persistence-threshold columns in {path}: "
            f"{sorted(missing)}"
        )

    thresholds = {}
    for _, row in df.iterrows():
        if pd.isna(row["diagnosis"]) or pd.isna(row["persistence_threshold"]):
            continue
        name = str(row["diagnosis"]).strip()
        value = float(row["persistence_threshold"])
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Invalid persistence threshold for {name}: {value}")
        if name in thresholds:
            raise ValueError(f"Duplicate persistence calibration diagnosis: {name}")
        thresholds[name] = value
    return thresholds

def parse_diagnosis_text(value) -> list[str]:
    """
    Convert diagnosis_text / diagnoses cells into a clean list of diagnoses.
    """
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()

        if not text or text == NORMAL_DIAGNOSIS:
            return []

        raw_items = None

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    raw_items = parsed
            except (SyntaxError, ValueError):
                raw_items = None

        if raw_items is None:
            raw_items = []
            for semi_part in text.split(";"):
                raw_items.extend(
                    semi_part.split(",")
                )

    diagnoses = []
    for item in raw_items:
        d = (
            str(item)
            .strip()
            .strip("'")
            .strip('"')
        )
        if (
            d
            and d != NORMAL_DIAGNOSIS
            and d != "normal"
        ):
            diagnoses.append(d)

    return diagnoses


def diagnosis_fractions(
    g: pd.DataFrame,
) -> dict[str, float]:
    """
    Fraction of room windows in which each diagnosis appears.
    """
    counts: dict[str, int] = {}
    n = max(len(g), 1)

    source_col = "diagnosis_text"
    if (
        source_col not in g.columns
        and "diagnoses" in g.columns
    ):
        source_col = "diagnoses"

    for value in g[source_col].fillna(
        NORMAL_DIAGNOSIS
    ):
        for diagnosis in set(
            parse_diagnosis_text(value)
        ):
            counts[diagnosis] = (
                counts.get(diagnosis, 0)
                + 1
            )

    return {
        diagnosis: count / n
        for diagnosis, count in counts.items()
    }


def choose_main_diagnosis(
    fractions: dict[str, float],
    persistence_thresholds: dict[str, float],
    fallback_threshold: float,
) -> tuple[str, float, float]:
    """
    Choose the dominant diagnosis that exceeds healthy-operation persistence.

    For each diagnosis d, persistence is significant only when its observed
    room fraction is strictly greater than its calibrated healthy threshold.
    Diagnoses absent from healthy calibration use the configured fallback.
    Diagnosis fraction is ranked first; priority is only a tie-breaker.
    """
    if not 0.0 <= fallback_threshold <= 1.0:
        raise ValueError("Persistence fallback must be in [0, 1].")
    significant: dict[str, tuple[float, float]] = {}

    for diagnosis, fraction in fractions.items():
        threshold = float(
            persistence_thresholds.get(
                diagnosis,
                fallback_threshold,
            )
        )
        if fraction > threshold:
            significant[diagnosis] = (fraction, threshold)

    if not significant:
        return NORMAL_DIAGNOSIS, 0.0, float("nan")

    priority_rank = {
        diagnosis: i
        for i, diagnosis in enumerate(DIAGNOSIS_PRIORITY)
    }

    def sort_key(item):
        diagnosis, values = item
        fraction, threshold = values
        return (
            -fraction,
            priority_rank.get(
                diagnosis,
                len(priority_rank),
            ),
            diagnosis,
        )

    diagnosis, (fraction, threshold) = sorted(
        significant.items(),
        key=sort_key,
    )[0]

    return diagnosis, fraction, threshold

def family_from_diagnosis(
    main_diagnosis: str,
) -> str:
    if main_diagnosis in {
        NORMAL_DIAGNOSIS,
        "normal",
    }:
        return "normal"

    return FAMILY_BY_DIAGNOSIS.get(
        main_diagnosis,
        "comfort_violation_unclassified",
    )


def read_raw_metadata(
    raw_path: Path,
) -> pd.DataFrame:
    """
    Read optional room metadata.

    fault_type and forced_valve_position_value are retained only as validation
    metadata; they do not enter diagnostic reasoning.
    """
    if not raw_path.exists():
        return pd.DataFrame(
            columns=["zone", "fault_type"]
        )

    header = pd.read_csv(
        raw_path,
        nrows=0,
    )

    possible_cols = [
        "zone",
        "fault_type",
        "level_name",
        "level_index",
        "plan_x",
        "plan_y",
        "plan_w",
        "plan_h",
        "layout_source",
        "building_width_m",
        "building_depth_m",
        "forced_valve_position_value",
    ]

    usecols = [
        c
        for c in possible_cols
        if c in header.columns
    ]

    if "zone" not in usecols:
        return pd.DataFrame(
            columns=["zone", "fault_type"]
        )

    raw = pd.read_csv(
        raw_path,
        usecols=usecols,
        engine="python",
        on_bad_lines="skip",
    )

    first_cols = [
        c
        for c in raw.columns
        if c != "zone"
    ]

    meta = (
        raw
        .groupby("zone", as_index=False)[first_cols]
        .first()
    )

    if "fault_type" not in meta.columns:
        meta["fault_type"] = "unknown"

    return meta


def summarize_room(
    zone: str,
    g: pd.DataFrame,
    raw_meta_row: dict | None,
    persistence_thresholds: dict[str, float],
    persistence_fallback: float,
) -> dict:
    n = len(g)

    comfort_violation = float(
        g["comfort_violation_fraction"].mean()
    )
    fractions = diagnosis_fractions(g)

    (
        main_diagnosis,
        main_diagnosis_fraction,
        main_diagnosis_threshold,
    ) = choose_main_diagnosis(
        fractions=fractions,
        persistence_thresholds=persistence_thresholds,
        fallback_threshold=persistence_fallback,
    )

    severity = final_room_severity(
        comfort_violation,
        main_diagnosis,
    )

    dominant_fault_family = (
        family_from_diagnosis(
            main_diagnosis
        )
    )

    if main_diagnosis == NORMAL_DIAGNOSIS:
        main_diagnosis_count = int(
            g["diagnosis_text"]
            .eq(NORMAL_DIAGNOSIS)
            .sum()
        )
    else:
        main_diagnosis_count = int(
            round(
                main_diagnosis_fraction
                * n
            )
        )

    normal_fraction = float(
        g["diagnosis_text"]
        .eq(NORMAL_DIAGNOSIS)
        .mean()
    )
    abnormal_fraction = (
        1.0 - normal_fraction
    )

    row = {
        "zone": zone,
        "fault_type": "unknown",
        "zone_area_m2": (
            float(g["zone_area_m2"].iloc[0])
            if "zone_area_m2" in g.columns
            else None
        ),
        "exposure_group": (
            g["exposure_group"].iloc[0]
            if "exposure_group" in g.columns
            else None
        ),
        "n_windows": n,
        "normal_fraction": normal_fraction,
        "abnormal_fraction": abnormal_fraction,
        "dominant_fault_family": (
            dominant_fault_family
        ),
        # Readable hydraulic subtype: derives ONLY from the existing
        # persistence-selected main diagnosis, never from injected truth.
        "hydraulic_operational_status": hydraulic_operational_status_for_room(
            main_diagnosis, severity
        ),
        "dominant_fault_fraction": (
            main_diagnosis_fraction
        ),
        "main_diagnosis": main_diagnosis,
        "main_diagnosis_count": (
            main_diagnosis_count
        ),
        "main_diagnosis_fraction": (
            main_diagnosis_fraction
        ),
        "main_diagnosis_persistence_threshold": (
            main_diagnosis_threshold
        ),
        "main_diagnosis_persistence_excess": (
            main_diagnosis_fraction - main_diagnosis_threshold
            if pd.notna(main_diagnosis_threshold)
            else 0.0
        ),
        "severity": severity,
        "comfort_violation_fraction": (
            comfort_violation
        ),
    }

    numeric_summary_cols = [
        "T_zone_mean",
        "T_set_mean",
        "tracking_error_mean",
        "overheating_fraction",
        "underheating_mean",
        "overheating_mean",
        "Q_mean",
        "Q_max",
        "E_sum",
        "m_dot_mean",
        "m_dot_std",
        "deltaT_water_mean",
        "deltaT_water_std",
        "T_supply_mean",
        "T_return_mean",
        "heating_active_fraction",
        "Q_density_mean",
        "Q_density_max",
        "E_density_sum",
        "m_dot_density_mean",
        "m_dot_density_std",
        "flow_oscillation_index",
        "flow_density_oscillation_index",
        "valve_position_mean",
        "valve_position_max",
        "valve_position_min",
        "valve_position_range",
        "valve_position_std",
        "valve_position_oscillation_index",
    ]

    for col in numeric_summary_cols:
        if col in g.columns:
            row[col] = float(
                g[col].mean()
            )

    # Explicit, stable columns for ALL hydraulic signatures, including zeros
    # when the diagnosis never occurred. Multiple signatures can coexist,
    # while the operational status reflects the dominant main diagnosis only.
    for diagnosis, column in HYDRAULIC_EVIDENCE_COLUMNS.items():
        row[column] = float(fractions.get(diagnosis, 0.0))

    # Transparent room-level diagnosis probabilities.
    for diagnosis, fraction in sorted(
        fractions.items()
    ):
        safe_name = (
            "diagnosis_fraction__"
            + diagnosis
        )
        row[safe_name] = fraction

    if raw_meta_row:
        for key, value in raw_meta_row.items():
            if key != "zone":
                row[key] = value

    return row


def main() -> None:
    args = parse_args()
    case_id = args.case_id

    in_path = (
        Path(args.input)
        if args.input
        else Path(
            f"data/processed/twin4build/{case_id}/"
            "t4b_3h_windows_diagnosed_against_openstudio.csv"
        )
    )

    raw_path = (
        Path(args.raw)
        if args.raw
        else Path(
            f"data/raw/twin4build/{case_id}/"
            "t4b_raw_timeseries.csv"
        )
    )

    out_path = (
        Path(args.out)
        if args.out
        else Path(
            f"data/processed/twin4build/{case_id}/"
            "t4b_room_level_diagnosis_summary.csv"
        )
    )

    persistence_thresholds_path = Path(args.persistence_thresholds)
    persistence_thresholds = load_persistence_thresholds(
        persistence_thresholds_path
    )
    persistence_fallback = float(args.persistence_fallback)
    if not 0.0 <= persistence_fallback <= 1.0:
        raise ValueError("--persistence-fallback must be in [0, 1].")

    df = pd.read_csv(in_path)

    required = [
        "zone",
        "diagnosis_text",
        "comfort_violation_fraction",
    ]
    missing = [
        c
        for c in required
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required columns in "
            f"{in_path}: {missing}"
        )

    raw_meta = read_raw_metadata(
        raw_path
    )
    raw_meta_by_zone = {
        row["zone"]: row.to_dict()
        for _, row in raw_meta.iterrows()
    }

    rows = []
    for zone, g in df.groupby(
        "zone",
        dropna=False,
    ):
        rows.append(
            summarize_room(
                zone=zone,
                g=g.copy(),
                raw_meta_row=(
                    raw_meta_by_zone.get(zone)
                ),
                persistence_thresholds=persistence_thresholds,
                persistence_fallback=persistence_fallback,
            )
        )

    summary = pd.DataFrame(rows)

    severity_order = {
        "fault": 0,
        "warning": 1,
        "normal": 2,
    }

    summary["_severity_order"] = (
        summary["severity"]
        .map(severity_order)
        .fillna(99)
    )

    summary = (
        summary
        .sort_values(
            [
                "_severity_order",
                "comfort_violation_fraction",
                "zone",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop(
            columns=["_severity_order"]
        )
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(
        out_path,
        index=False,
    )

    print("Input:", in_path)
    print("Raw metadata:", raw_path)
    print("Persistence thresholds:", persistence_thresholds_path)
    print("Persistence fallback:", persistence_fallback)
    print("Output:", out_path)
    print()

    print(
        summary[
            [
                "zone",
                "fault_type",
                "dominant_fault_family",
                "main_diagnosis",
                "hydraulic_operational_status",
                "severity",
                "comfort_violation_fraction",
                "main_diagnosis_fraction",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
