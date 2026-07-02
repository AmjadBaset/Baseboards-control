"""
Process raw OpenStudio/EnergyPlus baseboard CSV exports.

This converts the wide EnergyPlus ReadVarsESO CSV format into a long
zone/baseboard time-series format.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from baseboard_diagnostics.openstudio.archetype_metadata import (
    get_apartment_1970_metadata,
)
from baseboard_diagnostics.utils.io import write_csv


WARMUP_ROWS = 672


def _find_column(columns, exact_name):
    """
    Return exact column name if found, otherwise raise clear error.
    """
    if exact_name not in columns:
        raise ValueError(f"Column not found: {exact_name}")
    return exact_name


def _parse_energyplus_datetime(date_time_series: pd.Series, year: int = 2023) -> pd.Series:
    """
    Parse EnergyPlus Date/Time strings like '01/01  00:15:00'.

    EnergyPlus can report '24:00:00'. This function handles that by converting
    it to 00:00:00 of the next day.
    """

    def parse_one(value):
        value = str(value).strip()
        value = re.sub(r"\s+", " ", value)

        # Expected: MM/DD HH:MM:SS
        date_part, time_part = value.split(" ")

        month, day = map(int, date_part.split("/"))
        hour, minute, second = map(int, time_part.split(":"))

        if hour == 24:
            base = pd.Timestamp(year=year, month=month, day=day)
            return base + pd.Timedelta(days=1)

        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
        )

    return date_time_series.apply(parse_one)


def process_openstudio_raw_export(
    raw_path: str | Path,
    output_path: str | Path,
    case_id: str = "apartment_1970",
    archetype: str = "apartment",
    construction_year: int = 1970,
    simulation_year: int = 2023,
    warmup_rows: int = WARMUP_ROWS,
) -> pd.DataFrame:
    """
    Process raw OpenStudio/EnergyPlus CSV into long baseboard time-series.

    Output columns:
        timestamp
        case_id
        archetype
        construction_year
        zone
        baseboard
        zone_area_m2
        T_out
        occupancy
        T_zone
        T_set
        Q_bb
        E_bb
        m_dot
        T_supply
        T_return
    """

    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    metadata = get_apartment_1970_metadata()
    zone_to_baseboard = metadata["zone_to_baseboard"]
    zone_area_m2 = metadata["zone_area_m2"]

    raw = pd.read_csv(raw_path)

    if warmup_rows > 0:
        raw = raw.iloc[warmup_rows:].copy().reset_index(drop=True)

    raw["timestamp"] = _parse_energyplus_datetime(raw["Date/Time"], year=simulation_year)

    records = []

    columns = raw.columns

    outdoor_col = _find_column(
        columns,
        "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",
    )

    for zone, baseboard in zone_to_baseboard.items():
        zone_outdoor_col = f"{zone}:Zone Outdoor Air Drybulb Temperature [C](TimeStep)"
        occupancy_col = f"{zone}:Zone People Occupant Count [](Hourly)"
        t_zone_col = f"{zone}:Zone Air Temperature [C](TimeStep)"
        t_set_col = f"{zone}:Zone Thermostat Heating Setpoint Temperature [C](TimeStep)"

        e_col = f"{baseboard}:Baseboard Total Heating Energy [J](TimeStep)"
        q_col = f"{baseboard}:Baseboard Total Heating Rate [W](TimeStep)"
        mdot_col = f"{baseboard}:Baseboard Hot Water Mass Flow Rate [kg/s](TimeStep)"
        tin_col = f"{baseboard}:Baseboard Water Inlet Temperature [C](TimeStep)"
        tout_col = f"{baseboard}:Baseboard Water Outlet Temperature [C](TimeStep)"

        needed_cols = [
            zone_outdoor_col,
            occupancy_col,
            t_zone_col,
            t_set_col,
            e_col,
            q_col,
            mdot_col,
            tin_col,
            tout_col,
        ]

        missing = [col for col in needed_cols if col not in columns]
        if missing:
            raise ValueError(f"Missing columns for {zone} / {baseboard}: {missing}")

        temp = pd.DataFrame(
            {
                "timestamp": raw["timestamp"],
                "case_id": case_id,
                "archetype": archetype,
                "construction_year": construction_year,
                "zone": zone,
                "baseboard": baseboard,
                "zone_area_m2": zone_area_m2[zone],
                "T_out": raw[outdoor_col],
                "T_out_zone": raw[zone_outdoor_col],
                "occupancy": raw[occupancy_col],
                "T_zone": raw[t_zone_col],
                "T_set": raw[t_set_col],
                "Q_bb": raw[q_col],
                "E_bb": raw[e_col],
                "m_dot": raw[mdot_col],
                "T_supply": raw[tin_col],
                "T_return": raw[tout_col],
            }
        )

        records.append(temp)

    long_df = pd.concat(records, ignore_index=True)

    # Occupancy is hourly in the raw export, so make sure gaps are filled
    # within each zone after long-format conversion.
    long_df = long_df.sort_values(["zone", "timestamp"]).reset_index(drop=True)
    long_df["occupancy"] = (
        long_df.groupby("zone")["occupancy"]
        .transform(lambda s: s.ffill().bfill())
    )

    write_csv(long_df, output_path)

    return long_df
