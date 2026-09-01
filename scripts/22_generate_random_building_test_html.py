from pathlib import Path
import argparse
import html

import pandas as pd


DEFAULT_CASE_ID = "random_building_fault_case"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument(
        "--raw",
        default=None,
        help="Optional raw T4B CSV path.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output HTML path.",
    )
    return parser.parse_args()


def esc(value):
    if value is None or pd.isna(value):
        return "—"
    return html.escape(str(value))


def fmt(value, nd=1):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{nd}f}"


def fault_label(value):
    value = str(value)
    mapping = {
        "healthy": "Healthy",
        "restricted_flow": "Restricted flow",
        "severe_restricted_flow": "Severe restricted flow",
        "stuck_low_valve": "Stuck-low valve",
        "stuck_mid_valve": "Stuck-mid valve",
        "stuck_open_valve": "Stuck-open valve",
    }
    return mapping.get(value, value.replace("_", " ").title())


def severity_for_fault(fault_type):
    if fault_type == "healthy":
        return "normal"
    return "fault"


def room_card(row, min_x, min_y, scale, pad):
    x = pad + (row["plan_x"] - min_x) * scale
    y = pad + (row["plan_y"] - min_y) * scale
    w = row["plan_w"] * scale
    h = row["plan_h"] * scale

    fault_type = row["fault_type"]
    severity = severity_for_fault(fault_type)

    extra = ""
    if fault_type == "restricted_flow":
        extra = f"""
        <div class="detail">Valve max-flow factor: <b>{fmt(row.get("valve_max_flow"), 3)}</b></div>
        """
    elif fault_type == "stuck_low_valve":
        extra = f"""
        <div class="detail">Forced valve position: <b>{fmt(row.get("forced_valve_position_value"), 3)}</b></div>
        """

    return f"""
    <div class="room-card {severity}"
         style="left:{x:.2f}px; top:{y:.2f}px; width:{w:.2f}px; height:{h:.2f}px;">
      <div class="room-header">
        <b>{esc(row["zone"])}</b>
        <span class="badge {severity}">{esc(fault_label(fault_type))}</span>
      </div>
      <div class="fault">{esc(fault_label(fault_type))}</div>
      <div class="detail">Area: <b>{fmt(row["zone_area_m2"], 1)} m²</b></div>
      <div class="detail">Exterior surfaces: <b>{int(row["n_exterior_walls"])}</b></div>
      <div class="detail">Mean T: <b>{fmt(row.get("T_zone"), 2)} °C</b></div>
      <div class="detail">Mean valve: <b>{fmt(row.get("valve_position"), 3)}</b></div>
      {extra}
    </div>
    """


def make_level_section(level_name, level_df):
    pad = 18
    max_plot_w = 980
    max_plot_h = 520

    min_x = level_df["plan_x"].min()
    min_y = level_df["plan_y"].min()
    max_x = (level_df["plan_x"] + level_df["plan_w"]).max()
    max_y = (level_df["plan_y"] + level_df["plan_h"]).max()

    width = max_x - min_x
    height = max_y - min_y

    scale = min(max_plot_w / width, max_plot_h / height)
    canvas_w = width * scale + 2 * pad
    canvas_h = height * scale + 2 * pad

    cards = "\n".join(
        room_card(row, min_x=min_x, min_y=min_y, scale=scale, pad=pad)
        for _, row in level_df.iterrows()
    )

    return f"""
    <section class="section">
      <h2>{esc(level_name)}</h2>
      <div class="plan-wrap">
        <div class="plan-canvas" style="width:{canvas_w:.1f}px; height:{canvas_h:.1f}px;">
          {cards}
        </div>
      </div>
    </section>
    """


def table_rows(meta):
    rows = []
    for _, r in meta.iterrows():
        rows.append(f"""
        <tr>
          <td>{esc(r["zone"])}</td>
          <td>{esc(r["level_name"])}</td>
          <td>{esc(fault_label(r["fault_type"]))}</td>
          <td>{fmt(r["zone_area_m2"], 1)}</td>
          <td>{int(r["n_exterior_walls"])}</td>
          <td>{fmt(r["plan_x"], 1)}</td>
          <td>{fmt(r["plan_y"], 1)}</td>
          <td>{fmt(r["plan_w"], 1)}</td>
          <td>{fmt(r["plan_h"], 1)}</td>
          <td>{fmt(r.get("valve_max_flow"), 3)}</td>
          <td>{fmt(r.get("forced_valve_position_value"), 3)}</td>
          <td>{fmt(r.get("T_zone"), 2)}</td>
          <td>{fmt(r.get("valve_position"), 3)}</td>
        </tr>
        """)
    return "\n".join(rows)


def main():
    args = parse_args()

    raw_path = (
        Path(args.raw)
        if args.raw
        else Path(f"data/raw/twin4build/{args.case_id}/t4b_raw_timeseries.csv")
    )

    out_path = (
        Path(args.out)
        if args.out
        else Path(f"reports/{args.case_id}_test_building_fault_injection.html")
    )

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    raw = pd.read_csv(raw_path)

    required = [
        "zone",
        "level_name",
        "fault_type",
        "zone_area_m2",
        "n_exterior_walls",
        "plan_x",
        "plan_y",
        "plan_w",
        "plan_h",
    ]

    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw file is missing required columns: {missing}")

    numeric_means = (
        raw.groupby("zone", as_index=False)[
            ["T_zone", "Q_bb", "m_dot", "valve_position"]
        ]
        .mean()
        .rename(
            columns={
                "T_zone": "T_zone",
                "Q_bb": "Q_bb",
                "m_dot": "m_dot",
                "valve_position": "valve_position",
            }
        )
    )

    meta_cols = [
        "zone",
        "case_id",
        "level_name",
        "level_index",
        "fault_type",
        "zone_area_m2",
        "n_exterior_walls",
        "plan_x",
        "plan_y",
        "plan_w",
        "plan_h",
        "building_width_m",
        "building_depth_m",
        "layout_source",
        "valve_max_flow",
        "forced_valve_position_value",
    ]

    meta_cols = [c for c in meta_cols if c in raw.columns]

    meta = (
        raw[meta_cols]
        .drop_duplicates("zone")
        .merge(numeric_means, on="zone", how="left")
        .sort_values(["level_name", "zone"])
    )

    total_rooms = len(meta)
    faulty_rooms = int((meta["fault_type"] != "healthy").sum())
    fault_fraction = faulty_rooms / total_rooms * 100.0 if total_rooms else 0.0

    building_w = meta["building_width_m"].dropna().iloc[0] if "building_width_m" in meta else None
    building_h = meta["building_depth_m"].dropna().iloc[0] if "building_depth_m" in meta else None

    level_sections = "\n".join(
        make_level_section(level_name, level_df)
        for level_name, level_df in meta.groupby("level_name", sort=True)
    )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{esc(args.case_id)} — Random Building Fault Injection</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #102033;
      background: #f4f7fb;
    }}

    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}

    h2 {{
      margin: 0 0 14px;
      font-size: 21px;
    }}

    .subtitle {{
      margin-bottom: 24px;
      color: #506070;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}

    .metric {{
      background: white;
      border: 1px solid #dce4ee;
      border-radius: 14px;
      padding: 14px;
    }}

    .metric .label {{
      color: #5d6b7a;
      font-size: 13px;
      margin-bottom: 5px;
    }}

    .metric .value {{
      font-weight: 800;
      font-size: 22px;
    }}

    .section {{
      background: white;
      border: 1px solid #dce4ee;
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 22px;
    }}

    .plan-wrap {{
      overflow-x: auto;
      background: #eef3f8;
      border: 1px solid #cfdae6;
      border-radius: 14px;
      padding: 10px;
    }}

    .plan-canvas {{
      position: relative;
      background: #eaf0f6;
      border-radius: 12px;
      min-width: 500px;
    }}

    .room-card {{
      position: absolute;
      box-sizing: border-box;
      border-radius: 12px;
      padding: 12px;
      overflow: hidden;
      border: 2px solid #7bc98a;
      background: #e9f8ec;
    }}

    .room-card.fault {{
      border-color: #ff5b5b;
      background: #ffe8e8;
    }}

    .room-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 8px;
      font-size: 15px;
      margin-bottom: 8px;
    }}

    .fault {{
      font-weight: 700;
      margin-bottom: 8px;
    }}

    .detail {{
      font-size: 12px;
      margin: 3px 0;
    }}

    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }}

    .badge.normal {{
      color: #036b25;
      background: #bff0c9;
    }}

    .badge.fault {{
      color: #a10000;
      background: #ffc9c9;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}

    th {{
      background: #0d3554;
      color: white;
      text-align: left;
      padding: 8px;
      white-space: nowrap;
    }}

    td {{
      border-bottom: 1px solid #dce4ee;
      padding: 8px;
      vertical-align: top;
    }}

    tr:nth-child(even) {{
      background: #f7f9fc;
    }}

    .note {{
      color: #536271;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
<div class="page">
  <h1>Random Building Fault Injection Test</h1>
  <div class="subtitle">
    Case: <b>{esc(args.case_id)}</b>. This report visualizes the generated Twin4Build test building and injected faults.
  </div>

  <div class="cards">
    <div class="metric">
      <div class="label">Levels</div>
      <div class="value">{meta["level_name"].nunique()}</div>
    </div>
    <div class="metric">
      <div class="label">Rooms</div>
      <div class="value">{total_rooms}</div>
    </div>
    <div class="metric">
      <div class="label">Faulty rooms</div>
      <div class="value">{faulty_rooms} / {total_rooms}</div>
    </div>
    <div class="metric">
      <div class="label">Fault fraction</div>
      <div class="value">{fault_fraction:.1f}%</div>
    </div>
  </div>

  <section class="section">
    <h2>Building Metadata</h2>
    <p class="note">
      Building rectangle: <b>{fmt(building_w, 1)} m × {fmt(building_h, 1)} m</b>.
      The floor plans are schematic rectangular layouts exported from the random Twin4Build test-case generator.
      They are used for diagnostic visualization, not for changing Twin4Build physics.
    </p>
  </section>

  {level_sections}

  <section class="section">
    <h2>Injected Fault Table</h2>
    <div style="overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th>Room</th>
            <th>Level</th>
            <th>Injected fault</th>
            <th>Area m²</th>
            <th>Exterior surfaces</th>
            <th>x</th>
            <th>y</th>
            <th>w</th>
            <th>h</th>
            <th>Valve max flow</th>
            <th>Forced valve</th>
            <th>Mean T °C</th>
            <th>Mean valve</th>
          </tr>
        </thead>
        <tbody>
          {table_rows(meta)}
        </tbody>
      </table>
    </div>
  </section>
</div>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
