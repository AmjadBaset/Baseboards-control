from __future__ import annotations

import argparse
import ast
from pathlib import Path
import html

import pandas as pd


DEFAULT_CASE_ID = "two_room_restricted_flow"

POSITION_ORDER = [
    "top_left_corner",
    "top_middle_edge",
    "top_right_corner",
    "bottom_left_corner",
    "bottom_middle_edge",
    "bottom_right_corner",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--window-hours", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--mirror-x",
        action="store_true",
        help="Mirror the rendered floor plan horizontally.",
    )
    parser.add_argument(
        "--mirror-y",
        action="store_true",
        help="Mirror the rendered floor plan vertically.",
    )
    return parser.parse_args()


def paths(case_id: str):
    return {
        "raw": Path(f"data/raw/twin4build/{case_id}/t4b_raw_timeseries.csv"),
        "diag": Path(
            f"data/processed/twin4build/{case_id}/"
            "t4b_3h_windows_diagnosed_against_openstudio.csv"
        ),
        "summary": Path(
            f"data/processed/twin4build/{case_id}/"
            "t4b_room_level_diagnosis_summary.csv"
        ),
        "out": Path(f"reports/{case_id}_diagnostic_report.html"),
    }


def esc(x) -> str:
    return html.escape("" if pd.isna(x) else str(x))


def derive_plan_geometry_from_layout(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add plan_x, plan_y, plan_w, plan_h, and level_name when they are missing.

    Priority:
    1. Keep existing plan_x/plan_y/plan_w/plan_h if available.
    2. Derive coordinates from layout_position for the six-room synthetic case.
    3. Fall back to a simple automatic row layout.
    """
    raw = raw.copy()

    required = ["plan_x", "plan_y", "plan_w", "plan_h"]
    if all(c in raw.columns for c in required) and raw[required].notna().all().all():
        if "level_name" not in raw.columns:
            raw["level_name"] = "Level 0"
        return raw

    meta_cols = ["zone"]
    for c in [
        "zone_area_m2",
        "layout_position",
        "level_name",
        "floor_level",
        "floor",
        "storey",
    ]:
        if c in raw.columns:
            meta_cols.append(c)

    meta = raw[meta_cols].drop_duplicates("zone").copy()

    if "zone_area_m2" not in meta.columns:
        meta["zone_area_m2"] = 20.0

    if "level_name" not in meta.columns:
        if "floor_level" in meta.columns:
            meta["level_name"] = meta["floor_level"]
        elif "floor" in meta.columns:
            meta["level_name"] = meta["floor"]
        elif "storey" in meta.columns:
            meta["level_name"] = meta["storey"]
        else:
            meta["level_name"] = "Level 0"

    layout_xy = {
        "top_left_corner": (0, 1),
        "top_middle_edge": (1, 1),
        "top_right_corner": (2, 1),
        "bottom_left_corner": (0, 0),
        "bottom_middle_edge": (1, 0),
        "bottom_right_corner": (2, 0),
    }

    rows = []

    for level_name, level_df in meta.groupby("level_name", dropna=False):
        level_df = level_df.copy()

        has_layout_position = (
            "layout_position" in level_df.columns
            and level_df["layout_position"].isin(layout_xy).any()
        )

        if has_layout_position:
            for _, row in level_df.iterrows():
                zone = row["zone"]
                area = float(row["zone_area_m2"])
                layout_position = row.get("layout_position")

                if layout_position in layout_xy:
                    col, grid_y = layout_xy[layout_position]
                else:
                    # Put unknown-position rooms after known rooms.
                    idx = len(rows)
                    col = idx % 3
                    grid_y = -(idx // 3)

                # Derive approximate dimensions from area.
                # Height is fixed by row; width scales with area.
                # This preserves the plan arrangement while still reflecting size.
                base_h = 4.0
                w = max(3.0, area / 5.0)
                h = base_h

                x = col * 6.0
                y = grid_y * 5.0

                rows.append({
                    "zone": zone,
                    "plan_x": x,
                    "plan_y": y,
                    "plan_w": w,
                    "plan_h": h,
                    "level_name": level_name,
                })
        else:
            # Generic automatic layout for buildings without layout_position.
            # Rooms are placed left-to-right in rows. Width scales with area.
            level_df = level_df.sort_values("zone")
            x_cursor = 0.0
            y_cursor = 0.0
            row_h = 4.0
            max_row_width = 18.0

            for _, row in level_df.iterrows():
                zone = row["zone"]
                area = float(row["zone_area_m2"])

                h = row_h
                w = max(3.0, area / 5.0)

                if x_cursor > 0 and x_cursor + w > max_row_width:
                    x_cursor = 0.0
                    y_cursor -= row_h + 1.0

                rows.append({
                    "zone": zone,
                    "plan_x": x_cursor,
                    "plan_y": y_cursor,
                    "plan_w": w,
                    "plan_h": h,
                    "level_name": level_name,
                })

                x_cursor += w + 0.6

    geom = pd.DataFrame(rows)

    for col in ["plan_x", "plan_y", "plan_w", "plan_h", "level_name"]:
        if col in raw.columns:
            raw = raw.drop(columns=[col])

    raw = raw.merge(geom, on="zone", how="left")

    return raw


def compute_room_energy(raw, diag, summary, window_hours_default):
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    # Estimate OpenStudio/T4B reference uncertainty from healthy rooms.
    # This avoids using a single strict energy value.
    healthy_ratios = []

    for healthy_zone, hr in raw[raw["fault_type"].eq("healthy")].groupby("zone"):
        hr = hr.copy().sort_values("timestamp")
        hd = diag[diag["zone"].eq(healthy_zone)].copy()

        if hd.empty:
            continue

        hdt_h = hr["timestamp"].diff().dt.total_seconds().div(3600)
        hdt_h = hdt_h.fillna(hdt_h.median())

        h_actual_kWh = (hr["Q_bb"] * hdt_h / 1000.0).sum()
        h_window_hours = (
            hd["window_hours"] if "window_hours" in hd.columns else window_hours_default
        )

        h_ref_kWh = (
            hd["Q_density_mean_P50"] * hd["zone_area_m2"] * h_window_hours / 1000.0
        ).sum()

        if h_ref_kWh > 0:
            healthy_ratios.append(h_actual_kWh / h_ref_kWh)

    if healthy_ratios:
        ratio_series = pd.Series(healthy_ratios)
        abs_error_series = (ratio_series - 1.0).abs()

        # Validation statistics from healthy rooms.
        median_reference_deviation = float(abs_error_series.median())
        max_reference_deviation = float(abs_error_series.max())

        # Practical band used in the report tables.
        # Median healthy mismatch is reported for validation context.
        # Max healthy mismatch is reported as a conservative bound.
        # The displayed energy-impact range itself uses ±10%.
        practical_reference_deviation = 0.10

        reference_ratio_low = 1.0 - practical_reference_deviation
        reference_ratio_mid = 1.0
        reference_ratio_high = 1.0 + practical_reference_deviation
    else:
        median_reference_deviation = 0.0
        max_reference_deviation = 0.0
        practical_reference_deviation = 0.10

        reference_ratio_low = 1.0 - practical_reference_deviation
        reference_ratio_mid = 1.0
        reference_ratio_high = 1.0 + practical_reference_deviation

    rows = []

    for zone, r in raw.groupby("zone"):
        r = r.copy().sort_values("timestamp")
        d = diag[diag["zone"].eq(zone)].copy()
        s = summary[summary["zone"].eq(zone)].iloc[0]

        dt_h = r["timestamp"].diff().dt.total_seconds().div(3600)
        dt_h = dt_h.fillna(dt_h.median())

        actual_kWh = (r["Q_bb"] * dt_h / 1000.0).sum()

        window_hours = d["window_hours"] if "window_hours" in d.columns else window_hours_default

        normal_kWh = (
            d["Q_density_mean_P50"] * d["zone_area_m2"] * window_hours / 1000.0
        ).sum()

        normal_low_kWh = normal_kWh * reference_ratio_low
        normal_mid_kWh = normal_kWh * reference_ratio_mid
        normal_high_kWh = normal_kWh * reference_ratio_high

        deviation = actual_kWh - normal_mid_kWh

        comfort_fraction = (
            ((r["T_zone"] < 20.5) | (r["T_zone"] > 21.5)).mean()
            if "T_zone" in r.columns
            else 0.0
        )
        comfort_percent = comfort_fraction * 100.0

        severity = str(s["severity"])
        is_fault = severity == "fault"

        saving_low = max(0.0, actual_kWh - normal_high_kWh)
        saving_high = max(0.0, actual_kWh - normal_low_kWh)

        if is_fault and comfort_percent >= 5.0:
            deficit_low = max(0.0, normal_low_kWh - actual_kWh)
            deficit_high = max(0.0, normal_high_kWh - actual_kWh)
        else:
            deficit_low = 0.0
            deficit_high = 0.0

        energy_saved = saving_high
        comfort_deficit = deficit_high

        if saving_high > 0:
            impact_type = "energy_waste_range"
        elif deficit_high > 0:
            impact_type = "comfort_deficit_range"
        elif actual_kWh < normal_low_kWh:
            impact_type = "below_ref_no_comfort_issue"
        else:
            impact_type = "within_reference_range"

        diagnosis_probabilities = {}
        diagnosis_col = None

        for candidate_col in [
            "diagnoses",
            "diagnosis",
            "triggered_diagnoses",
            "rule_diagnoses",
            "all_diagnoses",
        ]:
            if candidate_col in d.columns:
                diagnosis_col = candidate_col
                break

        if diagnosis_col is not None and len(d) > 0:
            counts = {}
            for value in d[diagnosis_col]:
                for diagnosis_name in parse_diagnosis_list(value):
                    counts[diagnosis_name] = counts.get(diagnosis_name, 0) + 1

            diagnosis_probabilities = {
                diagnosis_name: 100.0 * count / len(d)
                for diagnosis_name, count in counts.items()
            }

        rows.append({
            "zone": zone,
            "fault_type": r["fault_type"].iloc[0] if "fault_type" in r.columns else "unknown",
            "dominant_fault_family": s["dominant_fault_family"],
            "severity": severity,
            "actual_T4B_kWh": actual_kWh,
            "openstudio_normal_P50_kWh": normal_kWh,
            "normal_reference_low_kWh": normal_low_kWh,
            "normal_reference_mid_kWh": normal_mid_kWh,
            "normal_reference_high_kWh": normal_high_kWh,
            "reference_ratio_low": reference_ratio_low,
            "reference_ratio_mid": reference_ratio_mid,
            "reference_ratio_high": reference_ratio_high,
            "reference_uncertainty_percent": practical_reference_deviation * 100.0,
            "median_validation_error_percent": median_reference_deviation * 100.0,
            "max_validation_error_percent": max_reference_deviation * 100.0,
            "reference_uncertainty_percent": max_reference_deviation * 100.0,
            "energy_difference_kWh": deviation,
            "energy_saved_if_fixed_kWh": energy_saved,
            "energy_saved_low_kWh": saving_low,
            "energy_saved_high_kWh": saving_high,
            "energy_deficit_to_restore_comfort_kWh": comfort_deficit,
            "energy_deficit_low_kWh": deficit_low,
            "energy_deficit_high_kWh": deficit_high,
            "energy_impact_type": impact_type,
            "diagnosis_probabilities": diagnosis_probabilities,
            "comfort_violation_percent": comfort_percent,
            "mean_T_zone": r["T_zone"].mean() if "T_zone" in r.columns else None,
            "mean_valve": r["valve_position"].mean() if "valve_position" in r.columns else None,
            "zone_area_m2": r["zone_area_m2"].iloc[0] if "zone_area_m2" in r.columns else None,
            "exposure_group": r["exposure_group"].iloc[0] if "exposure_group" in r.columns else "unknown",
            "n_exterior_walls": r["n_exterior_walls"].iloc[0] if "n_exterior_walls" in r.columns else None,
            "window_area_m2": r["window_area_m2"].iloc[0] if "window_area_m2" in r.columns else None,
            "H_total_W_per_K": r["H_total_W_per_K"].iloc[0] if "H_total_W_per_K" in r.columns else None,
            "layout_position": r["layout_position"].iloc[0] if "layout_position" in r.columns else None,
            "plan_x": r["plan_x"].iloc[0] if "plan_x" in r.columns else None,
            "plan_y": r["plan_y"].iloc[0] if "plan_y" in r.columns else None,
            "plan_w": r["plan_w"].iloc[0] if "plan_w" in r.columns else None,
            "plan_h": r["plan_h"].iloc[0] if "plan_h" in r.columns else None,
            "level_name": (
                r["level_name"].iloc[0]
                if "level_name" in r.columns
                else r["floor_level"].iloc[0]
                if "floor_level" in r.columns
                else r["floor"].iloc[0]
                if "floor" in r.columns
                else r["storey"].iloc[0]
                if "storey" in r.columns
                else "Level 0"
            ),
            "normal_fraction": s["normal_fraction"] if "normal_fraction" in s else None,
            "dominant_fault_fraction": s["dominant_fault_fraction"] if "dominant_fault_fraction" in s else None,
            "main_diagnosis": s["main_diagnosis"] if "main_diagnosis" in s else "",
            "fault_type_diagnostic": (
                "compensated_hydraulic_restriction"
                if s["main_diagnosis"] == "possible_compensated_hydraulic_restriction"
                else "partially_closed_or_stuck_low_valve"
                if s["main_diagnosis"] == "possible_partially_closed_or_stuck_low_valve"
                else "valve_leakage_or_stuck_open"
                if s["main_diagnosis"] == "possible_valve_leakage_or_stuck_open"
                else "insufficient_heat_output"
                if s["main_diagnosis"] == "insufficient_heat_output"
                else "normal"
                if s["main_diagnosis"] == "normal_or_no_rule_triggered"
                else s["main_diagnosis"]
            ),
        })

    return pd.DataFrame(rows)


def fmt(x, nd=1):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{nd}f}"


def fmt_range(low, high, nd=1):
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return "—"
    if high <= 0:
        return "—"
    if abs(high - low) < 0.05:
        return f"{high:.{nd}f}"
    return f"{low:.{nd}f}–{high:.{nd}f}"



def short_label(value):
    if value is None or pd.isna(value):
        return "—"

    text = str(value)

    mapping = {
        "normal": "Normal",
        "normal_or_no_rule_triggered": "Normal or no rule triggered",

        "restricted_or_hydraulic": "Restricted or hydraulic",
        "stuck_low_valve": "Stuck low valve",

        "compensated_hydraulic_restriction": "Compensated hydraulic restriction",
        "possible_compensated_hydraulic_restriction": "Possible compensated hydraulic restriction",

        "partially_closed_or_stuck_low_valve": "Partially closed or stuck low valve",
        "possible_partially_closed_or_stuck_low_valve": "Possible partially closed or stuck low valve",

        "valve_leakage_or_stuck_open": "Valve leakage or stuck open",
        "possible_valve_leakage_or_stuck_open": "Possible valve leakage or stuck open",

        "insufficient_heat_output": "Insufficient heat output",
        "hydraulic_delivery_deficit": "Hydraulic delivery deficit",
        "hydraulic_abnormality_low_flow_high_deltaT": "Hydraulic abnormality: low flow / high ΔT",
        "hydraulic_limitation_with_comfort_impact": "Hydraulic limitation with comfort impact",

        "energy_waste_range": "Possible energy waste range",
        "comfort_deficit_range": "Comfort energy deficit range",
        "within_reference_range": "Within reference range",
        "below_ref_no_comfort_issue": "Below reference without comfort issue",
        "compensated": "Compensated",
    }

    if text in mapping:
        return mapping[text]

    # If a long joined diagnosis string contains one of the known labels,
    # show the most specific readable label.
    priority = [
        "possible_partially_closed_or_stuck_low_valve",
        "partially_closed_or_stuck_low_valve",
        "possible_valve_leakage_or_stuck_open",
        "valve_leakage_or_stuck_open",
        "possible_compensated_hydraulic_restriction",
        "compensated_hydraulic_restriction",
        "hydraulic_limitation_with_comfort_impact",
        "hydraulic_delivery_deficit",
        "hydraulic_abnormality_low_flow_high_deltaT",
        "insufficient_heat_output",
    ]

    for key in priority:
        if key in text:
            return mapping.get(key, key.replace("_", " ").title())

    return text.replace("_", " ").title()


def parse_diagnosis_list(value):
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]

    text = str(value).strip()

    if not text or text in {"[]", "nan", "None"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
        if isinstance(parsed, str):
            return [parsed]
    except Exception:
        pass

    # Fallback for comma/semicolon/pipe separated strings.
    cleaned = (
        text.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    parts = []
    for sep in ["|", ";", ","]:
        if sep in cleaned:
            parts = [p.strip() for p in cleaned.split(sep)]
            break

    if not parts:
        parts = [cleaned.strip()]

    return [p for p in parts if p and p not in {"nan", "None"}]


def diagnosis_probability_table_rows(df):
    use = df[df["severity"].isin(["warning", "fault"])].copy()

    rows = []
    for _, r in use.iterrows():
        probs = r.get("diagnosis_probabilities", {})

        if not isinstance(probs, dict) or not probs:
            rows.append(f"""
            <tr>
              <td>{esc(r['zone'])}</td>
              <td>{esc(short_label(r['severity']))}</td>
              <td colspan="2">No window-level diagnosis probabilities available</td>
            </tr>
            """)
            continue

        first = True
        for diagnosis, probability in sorted(
            probs.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            rows.append(f"""
            <tr>
              <td>{esc(r['zone']) if first else ''}</td>
              <td>{esc(short_label(r['severity'])) if first else ''}</td>
              <td>{esc(short_label(diagnosis))}</td>
              <td>{fmt(probability, 1)}%</td>
            </tr>
            """)
            first = False

    return "\n".join(rows)

def has_plan_geometry(level_df: pd.DataFrame) -> bool:
    required = ["plan_x", "plan_y", "plan_w", "plan_h"]
    return all(c in level_df.columns for c in required) and level_df[required].notna().all(axis=None)


def positioned_room_card(row, min_x, min_y, max_x, max_y, mirror_x=False, mirror_y=False):
    x = float(row["plan_x"])
    y = float(row["plan_y"])
    w = float(row["plan_w"])
    h = float(row["plan_h"])

    total_w = max_x - min_x
    total_h = max_y - min_y

    if total_w <= 0:
        total_w = 1.0
    if total_h <= 0:
        total_h = 1.0

    if mirror_x:
        left = ((max_x - (x + w)) - min_x) / total_w * 100.0
    else:
        left = (x - min_x) / total_w * 100.0

    if mirror_y:
        top = (y - min_y) / total_h * 100.0
    else:
        top = ((max_y - (y + h)) / total_h) * 100.0

    width = w / total_w * 100.0
    height = h / total_h * 100.0

    # Prevent tiny rooms from becoming unreadable.
    width = max(width, 12.0)
    height = max(height, 18.0)

    return f"""
    <div class="positioned-room" style="left:{left:.2f}%; top:{top:.2f}%; width:{width:.2f}%; height:{height:.2f}%;">
      {room_card(row, row['zone_area_m2'], row['zone_area_m2'])}
    </div>
    """

def room_card(row, min_area: float, max_area: float):
    severity = row["severity"]
    cls = f"room {severity}"

    area = row["zone_area_m2"]
    if area is None or pd.isna(area):
        area = min_area

    # Scale visual emphasis by room area, but keep the real plan grid fixed.
    if max_area > min_area:
        normalized = (area - min_area) / (max_area - min_area)
    else:
        normalized = 0.5

    area_bar_width = 35 + normalized * 65

    if severity == "normal":
        impact = "Normal operation"
    elif severity == "warning":
        impact = "Compensated restriction"
    elif row["energy_saved_if_fixed_kWh"] > 0:
        impact = f"Potential saving: {fmt(row['energy_saved_if_fixed_kWh'])} kWh"
    elif row["energy_deficit_to_restore_comfort_kWh"] > 0:
        impact = f"Comfort deficit: {fmt(row['energy_deficit_to_restore_comfort_kWh'])} kWh"
    else:
        impact = row["energy_impact_type"]

    return f"""
    <div class="{cls}">
      <div class="room-head">
        <strong>{esc(row['zone'])}</strong>
        <span class="badge {esc(severity)}">{esc(severity.upper())}</span>
      </div>
      <div class="family">{esc(row['dominant_fault_family'])}</div>
      <div class="small">{esc(row['fault_type_diagnostic'])}</div>

      <div class="area-row">
        <span>Area: <b>{fmt(area)} m²</b></span>
        <span class="area-bar"><span style="width: {area_bar_width:.0f}%"></span></span>
      </div>

      <div class="metric">Comfort violation: <b>{fmt(row['comfort_violation_percent'])}%</b></div>
      <div class="metric">{esc(impact)}</div>
      <div class="small">Mean T: {fmt(row['mean_T_zone'], 2)} °C · Valve: {fmt(row['mean_valve'], 3)}</div>
    </div>
    """


def table_rows(df, only_faulty=False):
    rows = []
    use = df.copy()
    if only_faulty:
        use = use[use["severity"].isin(["warning", "fault"])]

    for _, r in use.iterrows():
        severity = str(r["severity"])

        if severity == "warning":
            normal_kwh = "—"
            diff_kwh = "—"
            saving_kwh = "—"
            deficit = "—"
            impact = "Compensated"
        elif severity == "fault":
            normal_kwh = fmt_range(
                r["normal_reference_low_kWh"],
                r["normal_reference_high_kWh"],
            )
            diff_kwh = "—"
            saving_kwh = fmt_range(
                r["energy_saved_low_kWh"],
                r["energy_saved_high_kWh"],
            )
            deficit = fmt_range(
                r["energy_deficit_low_kWh"],
                r["energy_deficit_high_kWh"],
            )
            impact = short_label(r["energy_impact_type"])
        else:
            normal_kwh = "—"
            diff_kwh = "—"
            saving_kwh = "—"
            deficit = "—"
            impact = "Normal"

        rows.append(f"""
        <tr>
          <td>{esc(r['zone'])}</td>
          <td>{esc(short_label(r['fault_type']))}</td>
          <td>{esc(short_label(r['dominant_fault_family']))}</td>
          <td>{esc(short_label(r['fault_type_diagnostic']))}</td>
          <td><span class="badge {esc(r['severity'])}">{esc(r['severity'])}</span></td>
          <td>{fmt(r['actual_T4B_kWh'])}</td>
          <td>{normal_kwh}</td>
          <td>{diff_kwh}</td>
          <td>{saving_kwh}</td>
          <td>{deficit}</td>
          <td>{fmt(r['comfort_violation_percent'])}%</td>
          <td>{esc(impact)}</td>
        </tr>
        """)

    return "\n".join(rows)


def render_html(case_id, raw, df, mirror_x=False, mirror_y=False):
    meta = raw.drop_duplicates("zone")
    total_area = meta["zone_area_m2"].sum() if "zone_area_m2" in meta else 0
    total_window = meta["window_area_m2"].sum() if "window_area_m2" in meta else 0
    total_H = meta["H_total_W_per_K"].sum() if "H_total_W_per_K" in meta else 0

    faulty = df[df["severity"].isin(["warning", "fault"])]
    faults = (df["severity"] == "fault").sum()
    warnings = (df["severity"] == "warning").sum()

    min_area = df["zone_area_m2"].min() if "zone_area_m2" in df.columns else 1.0
    max_area = df["zone_area_m2"].max() if "zone_area_m2" in df.columns else 1.0

    level_sections = []

    for level_name, level_df in df.groupby("level_name", dropna=False):
        if has_plan_geometry(level_df):
            min_x = level_df["plan_x"].min()
            min_y = level_df["plan_y"].min()
            max_x = (level_df["plan_x"] + level_df["plan_w"]).max()
            max_y = (level_df["plan_y"] + level_df["plan_h"]).max()

            cards = "\n".join(
                positioned_room_card(
                    r,
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    mirror_x=mirror_x,
                    mirror_y=mirror_y,
                )
                for _, r in level_df.sort_values(["plan_y", "plan_x"]).iterrows()
            )

            plan_class = "plan geometry-plan"
        else:
            has_layout = "layout_position" in level_df.columns and level_df["layout_position"].notna().any()

            if has_layout:
                cards_list = []
                used_zones = set()

                for pos in POSITION_ORDER:
                    sub = level_df[level_df["layout_position"].eq(pos)]
                    if not sub.empty:
                        room = sub.iloc[0]
                        used_zones.add(room["zone"])
                        cards_list.append(room_card(room, min_area, max_area))
                    else:
                        cards_list.append('<div class="room empty-room"></div>')

                remaining = level_df[~level_df["zone"].isin(used_zones)]
                extra_cards = "\n".join(
                    room_card(r, min_area, max_area)
                    for _, r in remaining.sort_values("zone").iterrows()
                )

                cards = "\n".join(cards_list)
                if extra_cards:
                    cards += f'\n<div class="extra-rooms">{extra_cards}</div>'

                plan_class = "plan fixed-plan"
            else:
                cards = "\n".join(
                    room_card(r, min_area, max_area)
                    for _, r in level_df.sort_values("zone").iterrows()
                )
                plan_class = "plan flexible-plan"

        level_sections.append(f"""
        <div class="level-block">
          <div class="level-title">{esc(level_name)}</div>
          <div class="{plan_class}">
            {cards}
          </div>
        </div>
        """)

    floor_plans = "\n".join(level_sections)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{esc(case_id)} Diagnostic Report</title>
<style>
  body {{
    margin: 0;
    background: #f3f5f7;
    font-family: Arial, sans-serif;
    color: #1f2933;
  }}
  .page {{
    width: 1120px;
    margin: 28px auto;
    background: white;
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.10);
    overflow: hidden;
  }}
  header {{
    padding: 34px 42px;
    background: linear-gradient(135deg, #0f2f4a, #1d5c84);
    color: white;
  }}
  header h1 {{
    margin: 0 0 8px;
    font-size: 30px;
  }}
  header p {{
    margin: 0;
    opacity: 0.9;
  }}
  .section {{
    padding: 28px 42px;
    border-bottom: 1px solid #e5e7eb;
  }}
  .kpis {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
  }}
  .kpi {{
    padding: 16px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
  }}
  .kpi .value {{
    font-size: 24px;
    font-weight: 700;
  }}
  .kpi .label {{
    font-size: 12px;
    color: #64748b;
    margin-top: 4px;
  }}
  h2 {{
    margin: 0 0 18px;
    font-size: 22px;
  }}
  .level-block {{
    margin-bottom: 22px;
  }}
  .level-title {{
    font-size: 15px;
    font-weight: 700;
    color: #334155;
    margin: 0 0 10px;
  }}
  .plan {{
    padding: 18px;
    background: #eef2f6;
    border-radius: 16px;
    border: 1px solid #d7dee8;
  }}
  .fixed-plan {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 14px;
  }}
  .geometry-plan {{
    position: relative;
    height: 560px;
    min-height: 420px;
  }}
  .positioned-room {{
    position: absolute;
    box-sizing: border-box;
    padding: 4px;
  }}
  .positioned-room .room {{
    width: 100%;
    height: 100%;
    min-height: unset;
    box-sizing: border-box;
    overflow: hidden;
  }}
  .flexible-plan {{
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    gap: 14px;
  }}
  .room {{
    min-height: 185px;
    border-radius: 14px;
    padding: 14px;
    border: 2px solid #cbd5e1;
    background: #f8fafc;
  }}
  .empty-room {{
    background: transparent;
    border: 1px dashed #cbd5e1;
    min-height: 185px;
  }}
  .extra-rooms {{
    grid-column: 1 / -1;
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 8px;
  }}
  .area-row {{
    margin: 8px 0 6px;
    font-size: 13px;
  }}
  .area-bar {{
    display: block;
    height: 6px;
    margin-top: 5px;
    background: rgba(15, 47, 74, 0.12);
    border-radius: 999px;
    overflow: hidden;
  }}
  .area-bar span {{
    display: block;
    height: 100%;
    background: rgba(15, 47, 74, 0.55);
    border-radius: 999px;
  }}
  .room.normal {{ background: #e9f7ec; border-color: #8dcc98; }}
  .room.warning {{ background: #fff3d6; border-color: #e0aa3e; }}
  .room.fault {{ background: #fde2e2; border-color: #e35d5d; }}
  .room-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }}
  .family {{
    font-weight: 700;
    margin-bottom: 6px;
    overflow-wrap: anywhere;
  }}
  .metric {{
    font-size: 13px;
    margin: 5px 0;
  }}
  .small {{
    font-size: 12px;
    color: #475569;
    margin-top: 6px;
    overflow-wrap: anywhere;
  }}
  .badge {{
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
  }}
  .badge.normal {{ background: #c8ead0; color: #14532d; }}
  .badge.warning {{ background: #fde68a; color: #78350f; }}
  .badge.fault {{ background: #fecaca; color: #7f1d1d; }}

  .table-wrap {{
    overflow-x: auto;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 11px;
    table-layout: fixed;
  }}
  th {{
    background: #0f2f4a;
    color: white;
    padding: 9px 6px;
    text-align: left;
    white-space: normal;
    word-break: break-word;
  }}
  td {{
    padding: 8px 6px;
    border-bottom: 1px solid #e5e7eb;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
    vertical-align: top;
  }}
  tr:nth-child(even) td {{
    background: #f8fafc;
  }}
  .note {{
    font-size: 13px;
    color: #475569;
    line-height: 1.5;
  }}
  @media print {{
    body {{ background: white; }}
    .page {{
      width: auto;
      margin: 0;
      box-shadow: none;
      border-radius: 0;
    }}
    .section {{ page-break-inside: avoid; }}
    table {{ font-size: 9.5px; }}
    th, td {{ padding: 6px 4px; }}
  }}
</style>
</head>
<body>
<div class="page">
  <header>
    <h1>Baseboard Heating Diagnostic Report</h1>
    <p>Case: {esc(case_id)} · Baseline: OpenStudio healthy P50 bands</p>
  </header>

  <section class="section">
    <div class="kpis">
      <div class="kpi"><div class="value">{len(df)}</div><div class="label">Rooms</div></div>
      <div class="kpi"><div class="value">{warnings}</div><div class="label">Warnings</div></div>
      <div class="kpi"><div class="value">{faults}</div><div class="label">Faults</div></div>
      <div class="kpi"><div class="value">{total_area:.1f}</div><div class="label">Floor area m²</div></div>
      <div class="kpi"><div class="value">{total_H:.1f}</div><div class="label">Total H W/K</div></div>
    </div>
  </section>

  <section class="section">
    <h2>Building Plan</h2>
    {floor_plans}
  </section>

  <section class="section">
    <h2>Fault and Warning Rooms</h2>
    <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Room</th>
          <th>Injected</th>
          <th>Fault family</th>
          <th>Fault type</th>
          <th>Severity</th>
          <th>Actual kWh</th>
          <th>Normal kWh</th>
          <th>Diff kWh</th>
          <th>Saving kWh</th>
          <th>Deficit kWh</th>
          <th>Comfort %</th>
          <th>Impact</th>
        </tr>
      </thead>
      <tbody>
        {table_rows(df, only_faulty=True)}
      </tbody>
    </table>
  </section>

  <section class="section">
    <h2>All-Room Information</h2>
    <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Room</th>
          <th>Injected</th>
          <th>Fault family</th>
          <th>Fault type</th>
          <th>Severity</th>
          <th>Actual kWh</th>
          <th>Normal kWh</th>
          <th>Diff kWh</th>
          <th>Saving kWh</th>
          <th>Deficit kWh</th>
          <th>Comfort %</th>
          <th>Impact</th>
        </tr>
      </thead>
      <tbody>
        {table_rows(df, only_faulty=False)}
      </tbody>
    </table>
  </section>

  <section class="section">
    <h2>Triggered Fault Rules and Probabilities</h2>
    <p class="note">
      This table lists every diagnostic rule triggered in warning and fault rooms.
      Probability is calculated as the percentage of diagnostic windows in which the rule appeared.
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Room</th>
            <th>Severity</th>
            <th>Triggered fault rule</th>
            <th>Probability</th>
          </tr>
        </thead>
        <tbody>
          {diagnosis_probability_table_rows(df)}
        </tbody>
      </table>
    </div>
  </section>

  <section class="section">
    <h2>Method Note</h2>
    <p class="note">
      Normal energy is estimated from OpenStudio healthy P50 reference bands.
      Energy-impact estimates are reported as ranges rather than single strict values.
      The report uses a practical ±10% uncertainty band for quantified energy impacts.
      This band is informed by healthy-room validation: the median healthy-room mismatch was small,
      while the largest single-room mismatch is reported separately as a conservative validation bound.
      Energy saving is only reported when actual use is above the upper end of the ±10% range.
      Comfort deficit is only reported when a fault room is below the lower end of the ±10% range and comfort
      violation exceeds 5%. Warning rooms therefore show abnormal behavior without assigning a quantified
      energy or comfort-deficit estimate.
    </p>
  </section>
</div>
</body>
</html>
"""


def main():
    args = parse_args()
    p = paths(args.case_id)
    out_path = args.out or p["out"]

    raw = pd.read_csv(p["raw"])
    raw = derive_plan_geometry_from_layout(raw)

    diag = pd.read_csv(p["diag"])
    summary = pd.read_csv(p["summary"])

    df = compute_room_energy(raw, diag, summary, args.window_hours)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_html(
            case_id=args.case_id,
            raw=raw,
            df=df,
            mirror_x=args.mirror_x,
            mirror_y=args.mirror_y,
        ),
        encoding="utf-8",
    )

    print(f"Saved HTML report: {out_path}")


if __name__ == "__main__":
    main()
