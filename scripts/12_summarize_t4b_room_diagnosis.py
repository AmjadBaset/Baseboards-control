from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/twin4build/two_room_restricted_flow/"
    "t4b_3h_windows_diagnosed_against_openstudio.csv"
)

RAW_T4B_PATH = Path(
    "data/raw/twin4build/two_room_restricted_flow/"
    "t4b_raw_timeseries.csv"
)

OUTPUT_PATH = Path(
    "data/processed/twin4build/two_room_restricted_flow/"
    "t4b_room_level_diagnosis_summary.csv"
)


RESTRICTED_FLOW_DIAGNOSES = [
    "possible_compensated_hydraulic_restriction",
    "compensated_hydraulic_restriction_low_flow",
    "degraded_hydraulic_compensation",
    "hydraulic_delivery_deficit",
    "hydraulic_abnormality_low_flow_high_deltaT",
    "hydraulic_limitation_with_comfort_impact",
]

STUCK_OPEN_DIAGNOSES = [
    "possible_valve_leakage_or_stuck_open",
]

STUCK_LOW_DIAGNOSES = [
    "possible_partially_closed_or_stuck_low_valve",
]

HEAT_OUTPUT_DIAGNOSES = [
    "insufficient_heat_output",
    "insufficient_heat_output_despite_high_controller_effort",
    "possible_emitter_undersizing_or_high_load",
]

OVERSUPPLY_DIAGNOSES = [
    "possible_oversupply_or_overheating",
]


def has_diagnosis(text: str, diagnosis_names: list[str]) -> bool:
    if not isinstance(text, str):
        return False
    parts = {p.strip() for p in text.split(";")}
    return any(name in parts for name in diagnosis_names)


def severity_from_fraction(frac: float) -> str:
    if frac >= 0.50:
        return "fault"
    if frac >= 0.20:
        return "warning"
    if frac >= 0.05:
        return "watch"
    return "normal"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # The diagnosed window file may not always carry the ground-truth
    # simulation fault labels. If missing, recover them from the raw T4B file.
    if "fault_type" not in df.columns and RAW_T4B_PATH.exists():
        raw = pd.read_csv(RAW_T4B_PATH)
        meta_cols = [
            "zone",
            "fault_type",
            "zone_area_m2",
            "exposure_group",
        ]
        available_meta_cols = [c for c in meta_cols if c in raw.columns]
        meta = raw[available_meta_cols].drop_duplicates("zone")
        df = df.merge(meta, on="zone", how="left", suffixes=("", "_raw"))

        for col in ["zone_area_m2", "exposure_group"]:
            raw_col = f"{col}_raw"
            if raw_col in df.columns:
                if col not in df.columns:
                    df[col] = df[raw_col]
                else:
                    df[col] = df[col].fillna(df[raw_col])
                df = df.drop(columns=[raw_col])

    if "fault_type" not in df.columns:
        df["fault_type"] = "unknown"

    rows = []

    for keys, g in df.groupby(
        ["zone", "fault_type", "zone_area_m2", "exposure_group"],
        dropna=False,
    ):
        zone, fault_type, zone_area_m2, exposure_group = keys
        n = len(g)

        diagnosis_counts = g["diagnosis_text"].value_counts(dropna=False)
        main_diagnosis = diagnosis_counts.index[0]
        main_diagnosis_count = int(diagnosis_counts.iloc[0])
        main_diagnosis_fraction = main_diagnosis_count / n

        normal_fraction = (
            g["diagnosis_text"].eq("normal_or_no_rule_triggered").mean()
        )

        restricted_fraction = g["diagnosis_text"].apply(
            lambda x: has_diagnosis(x, RESTRICTED_FLOW_DIAGNOSES)
        ).mean()

        stuck_open_fraction = g["diagnosis_text"].apply(
            lambda x: has_diagnosis(x, STUCK_OPEN_DIAGNOSES)
        ).mean()

        stuck_low_fraction = g["diagnosis_text"].apply(
            lambda x: has_diagnosis(x, STUCK_LOW_DIAGNOSES)
        ).mean()

        heat_output_fraction = g["diagnosis_text"].apply(
            lambda x: has_diagnosis(x, HEAT_OUTPUT_DIAGNOSES)
        ).mean()

        oversupply_fraction = g["diagnosis_text"].apply(
            lambda x: has_diagnosis(x, OVERSUPPLY_DIAGNOSES)
        ).mean()

        abnormal_fraction = 1.0 - normal_fraction

        if stuck_low_fraction >= max(
            stuck_open_fraction,
            restricted_fraction,
            heat_output_fraction,
            oversupply_fraction,
        ):
            dominant_fault_family = "stuck_low_valve"
            dominant_fault_fraction = stuck_low_fraction
        elif stuck_open_fraction >= max(
            restricted_fraction, heat_output_fraction, oversupply_fraction
        ):
            dominant_fault_family = "stuck_open_or_leakage"
            dominant_fault_fraction = stuck_open_fraction
        elif restricted_fraction >= max(heat_output_fraction, oversupply_fraction):
            dominant_fault_family = "restricted_or_hydraulic"
            dominant_fault_fraction = restricted_fraction
        elif heat_output_fraction >= oversupply_fraction:
            dominant_fault_family = "insufficient_heat_output"
            dominant_fault_fraction = heat_output_fraction
        else:
            dominant_fault_family = "oversupply_or_overheating"
            dominant_fault_fraction = oversupply_fraction

        if abnormal_fraction < 0.05:
            dominant_fault_family = "normal"
            dominant_fault_fraction = abnormal_fraction

        comfort_violation = (
            g["comfort_violation_fraction"].mean()
            if "comfort_violation_fraction" in g.columns
            else 0.0
        )

        if dominant_fault_family == "normal":
            severity = "normal"
        elif dominant_fault_family == "restricted_or_hydraulic" and comfort_violation < 0.05:
            # Compensated hydraulic restriction: clear abnormal control/flow symptom,
            # but comfort is still maintained.
            severity = "warning"
        else:
            severity = severity_from_fraction(dominant_fault_fraction)

        row = {
            "zone": zone,
            "fault_type": fault_type,
            "zone_area_m2": zone_area_m2,
            "exposure_group": exposure_group,
            "n_windows": n,
            "main_diagnosis": main_diagnosis,
            "main_diagnosis_fraction": main_diagnosis_fraction,
            "normal_fraction": normal_fraction,
            "abnormal_fraction": abnormal_fraction,
            "dominant_fault_family": dominant_fault_family,
            "dominant_fault_fraction": dominant_fault_fraction,
            "severity": severity,
            "restricted_or_hydraulic_fraction": restricted_fraction,
            "stuck_open_or_leakage_fraction": stuck_open_fraction,
            "stuck_low_valve_fraction": stuck_low_fraction,
            "insufficient_heat_output_fraction": heat_output_fraction,
            "oversupply_or_overheating_fraction": oversupply_fraction,
        }

        optional_means = [
            "valve_position_mean",
            "normalized_flow_fraction_mean_P90",
            "m_dot_density_mean",
            "Q_density_mean",
            "comfort_violation_fraction",
            "overheating_fraction",
            "underheating_fraction",
        ]

        for col in optional_means:
            if col in g.columns:
                row[col] = g[col].mean()

        if "valve_effort_label" in g.columns:
            row["valve_effort_high_fraction"] = (
                g["valve_effort_label"].eq("high").mean()
            )
            row["valve_effort_low_fraction"] = (
                g["valve_effort_label"].eq("low").mean()
            )

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("zone")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    display_cols = [
        "zone",
        "fault_type",
        "dominant_fault_family",
        "dominant_fault_fraction",
        "severity",
        "normal_fraction",
        "main_diagnosis",
        "main_diagnosis_fraction",
    ]

    print(f"Saved: {OUTPUT_PATH}")
    print()
    print(summary[display_cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
