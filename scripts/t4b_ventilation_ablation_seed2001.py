#!/usr/bin/env python3
"""
Generate one reproducible randomized Twin4Build validation building.

Key design choices for blind FDD validation
-------------------------------------------
1. The same schematic floor plan is reused on every level.
2. The diagnostic time-series file contains only signals/context available to
   the FDD pipeline. Injected fault labels and parameters are written to a
   separate ground_truth.csv file.
3. The PID command is used internally by the simulation but is NOT exported to
   the diagnostic time-series file.
4. Faults begin after an initial healthy period and persist until the end of
   the simulation.
5. Hydraulic restrictions are injected downstream of an otherwise healthy
   valve, so the room controller can compensate by opening the valve further.
6. Valve faults include fixed position, limited opening, and limited closing;
   limited-travel severity is defined relative to each room's own pre-fault
   requested-valve distribution.
7. Heating-emitter capacity and healthy maximum water flow are sized per room
   from its heat-loss coefficient instead of using one 1800 W emitter for all
   rooms.
8. Ground/top/middle floors retain the same layout but receive physically
   different horizontal-envelope exposure.
9. Occupancy, solar gain, and ventilation schedules add realistic operating
   variation while remaining fully reproducible from the seed.

The output t4b_raw_timeseries.csv intentionally keeps the historical filename
expected by scripts/11 and 12, but it is now an OPERATIONAL-ONLY file. Ground
truth is stored separately and should be merged only during evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import datetime as dt
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import twin4build as tb
import twin4build.core as core
import twin4build.utils.types as tps


# -----------------------------------------------------------------------------
# Fixed study configuration
# -----------------------------------------------------------------------------

OPENSTUDIO_TOUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_clean_area_new.csv"
)

CONSTRUCTION_YEAR = 1970
NOMINAL_SUPPLY_TEMP_C = 60.0
NOMINAL_RETURN_TEMP_C = 45.0
NOMINAL_WATER_DELTA_T_K = NOMINAL_SUPPLY_TEMP_C - NOMINAL_RETURN_TEMP_C
WATER_CP_J_PER_KG_K = 4180.0
ROOM_SETPOINT_C = 21.0
SUPPLY_AIR_TEMP_C = 18.0
OUTDOOR_CO2_PPM = 420.0
TIMESTEP_SECONDS = 900

# The physical room areas are sampled independently of the schematic drawing.
# The same base-floor room receives the same physical area on every level.
ROOM_AREA_OPTIONS_M2 = [10.0, 12.5, 15.0, 20.0, 25.0, 30.0]

# Reference envelope values already used by the synthetic validation model.
U_VALUES_W_M2K = {
    "exterior_wall": 0.83,
    "roof": 0.64,
    "floor": 0.64,
    "window": 5.03,
}

ROOM_HEIGHT_M = 2.5
WALL_LENGTH_PER_25M2_ROOM_M = 5.0
REFERENCE_ROOM_AREA_M2 = 25.0
WINDOW_TO_FLOOR_AREA_RATIO = 0.10

# Emitter sizing: room design load multiplied by a modest sizing margin.
EMITTER_SIZING_MARGIN_RANGE = (1.10, 1.25)
DESIGN_OUTDOOR_QUANTILE = 0.01

# Valve-characteristic robustness. 1.0 preserves the previous near-linear case;
# lower values introduce moderate non-linearity in the Twin4Build valve model.
VALVE_AUTHORITY_RANGE = (0.70, 1.00)

# Faults occur only after an initial healthy period.
FAULT_ONSET_FRACTION_RANGE = (0.15, 0.35)

# Remaining hydraulic-flow capacity after a restriction is applied.
RESTRICTION_FACTOR_RANGES = {
    "mild": (0.75, 0.90),
    "moderate": (0.50, 0.75),
    "severe": (0.20, 0.50),
}

# Limited-travel faults are defined relative to each room's own healthy
# pre-fault valve-demand distribution. This avoids labeling an absolute travel
# limit as "severe" when the room would almost never request that position.
#
# limited_opening: upper limit sampled from a progressively lower healthy
# request quantile as severity increases.
# limited_closing: lower limit sampled from a progressively higher healthy
# request quantile as severity increases.
OPENING_LIMIT_QUANTILE_RANGES = {
    "mild": (0.80, 0.90),
    "moderate": (0.60, 0.75),
    "severe": (0.40, 0.55),
}

CLOSING_LIMIT_QUANTILE_RANGES = {
    "mild": (0.10, 0.20),
    "moderate": (0.25, 0.40),
    "severe": (0.45, 0.60),
}

# Fixed-valve positions. These are injection ranges, not diagnostic thresholds.
# A fixed actuator is sampled from either a lower or higher part of its travel.
# Fixed-valve positions, like travel limits, must be room-relative.
# A position of 0.4 can be an unusually HIGH opening for one room,
# even though an absolute 0.4 would previously be tagged as "low".
FIXED_VALVE_QUANTILE_RANGES = {
    ("low", "mild"): (0.35, 0.45),
    ("low", "moderate"): (0.15, 0.30),
    ("low", "severe"): (0.05, 0.15),
    ("high", "mild"): (0.55, 0.65),
    ("high", "moderate"): (0.70, 0.85),
    ("high", "severe"): (0.85, 0.95),
}

FAULT_MECHANISMS = [
    "restricted_flow",
    "fixed_valve",
    "limited_opening",
    "limited_closing",
]
FAULT_SEVERITIES = ["mild", "moderate", "severe"]


# -----------------------------------------------------------------------------
# Small Twin4Build helper systems
# -----------------------------------------------------------------------------

class ScalarMultiplierSystem(core.System, nn.Module):
    """Multiply a scalar input by a constant factor."""

    sp = []

    def __init__(self, multiplier: float, **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.multiplier = float(multiplier)
        self.input = {"inputSignal": tps.Scalar()}
        self.output = {"outputSignal": tps.Scalar(0)}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size, simulator):
        self.input["inputSignal"].initialize(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            simulator=simulator,
        )
        self.output["outputSignal"].initialize(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            simulator=simulator,
        )

    def do_step(self, secondTime, dateTime, step_size, stepIndex):
        value = self.input["inputSignal"].get() * self.multiplier
        self.output["outputSignal"].set(value, stepIndex)


class ValveFaultInjectorSystem(core.System, nn.Module):
    """Apply a valve-position fault after a specified simulation step.

    Limited-travel faults are resolved at fault onset from the room's own
    pre-fault requested-valve distribution. The requested controller signal is
    used internally only and is never exported to the FDD dataset.
    """

    sp = []

    def __init__(
        self,
        mode: str = "healthy",
        fault_start_step: int | None = None,
        fixed_position: float | None = None,
        fixed_position_quantile: float | None = None,
        upper_limit: float | None = None,
        lower_limit: float | None = None,
        limit_quantile: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.mode = str(mode)
        self.fault_start_step = (
            int(fault_start_step)
            if fault_start_step is not None
            else 10**12
        )
        self.fixed_position = fixed_position
        self.fixed_position_quantile = fixed_position_quantile
        if self.fixed_position_quantile is not None and not 0.0 <= self.fixed_position_quantile <= 1.0:
            raise ValueError("fixed_position_quantile must be in [0, 1].")
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.limit_quantile = limit_quantile

        if self.limit_quantile is not None and not 0.0 <= self.limit_quantile <= 1.0:
            raise ValueError("limit_quantile must be in [0, 1].")

        self._pre_fault_requested: list[float] = []
        self.binding_count = 0
        self.post_fault_step_count = 0
        self.absolute_position_change_sum = 0.0

        self.input = {"requestedValvePosition": tps.Scalar()}
        self.output = {"actualValvePosition": tps.Scalar(0)}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    @property
    def binding_fraction(self) -> float:
        if self.post_fault_step_count <= 0:
            return float("nan")
        return self.binding_count / self.post_fault_step_count

    @property
    def mean_absolute_position_change(self) -> float:
        if self.post_fault_step_count <= 0:
            return float("nan")
        return self.absolute_position_change_sum / self.post_fault_step_count

    def initialize(self, start_time, end_time, step_size, simulator):
        self.input["requestedValvePosition"].initialize(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            simulator=simulator,
        )
        self.output["actualValvePosition"].initialize(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            simulator=simulator,
        )

    @staticmethod
    def _constant_like(reference: torch.Tensor, value: float) -> torch.Tensor:
        return torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )

    @staticmethod
    def _scalar_float(value: torch.Tensor) -> float:
        return float(value.detach().cpu().reshape(-1)[0].item())

    def _resolve_from_healthy_request(
        self, requested_value: float, quantile: float | None
    ) -> float:
        if quantile is None:
            raise ValueError(f"{self.mode} requires a configured position quantile.")
        history = self._pre_fault_requested or [requested_value]
        limit = float(np.quantile(np.asarray(history, dtype=float), quantile))
        return float(np.clip(limit, 0.0, 1.0))

    def do_step(self, secondTime, dateTime, step_size, stepIndex):
        requested = torch.clamp(
            self.input["requestedValvePosition"].get(),
            0.0,
            1.0,
        )
        requested_value = self._scalar_float(requested)

        if stepIndex < self.fault_start_step:
            if self.mode in {"limited_opening", "limited_closing", "fixed_valve"}:
                self._pre_fault_requested.append(requested_value)
            actual = requested

        elif self.mode == "healthy":
            actual = requested

        else:
            self.post_fault_step_count += 1
            binding = False

            if self.mode == "fixed_valve":
                if self.fixed_position is None:
                    self.fixed_position = self._resolve_from_healthy_request(
                        requested_value, self.fixed_position_quantile
                    )
                actual = self._constant_like(requested, float(self.fixed_position))
                binding = abs(requested_value - float(self.fixed_position)) > 1e-9

            elif self.mode == "limited_opening":
                if self.upper_limit is None:
                    self.upper_limit = self._resolve_from_healthy_request(requested_value, self.limit_quantile)
                limit = self._constant_like(requested, float(self.upper_limit))
                actual = torch.minimum(requested, limit)
                binding = requested_value > float(self.upper_limit) + 1e-9

            elif self.mode == "limited_closing":
                if self.lower_limit is None:
                    self.lower_limit = self._resolve_from_healthy_request(requested_value, self.limit_quantile)
                limit = self._constant_like(requested, float(self.lower_limit))
                actual = torch.maximum(requested, limit)
                binding = requested_value < float(self.lower_limit) - 1e-9

            else:
                raise ValueError(f"Unsupported valve fault mode: {self.mode}")

            self.absolute_position_change_sum += abs(
                requested_value - self._scalar_float(actual)
            )
            if binding:
                self.binding_count += 1

        actual = torch.clamp(actual, 0.0, 1.0)
        self.output["actualValvePosition"].set(actual, stepIndex)


class HydraulicRestrictionSystem(core.System, nn.Module):
    """Apply a downstream hydraulic flow restriction after fault onset."""

    sp = []

    def __init__(
        self,
        restriction_factor: float = 1.0,
        fault_start_step: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.restriction_factor = float(restriction_factor)
        self.fault_start_step = (
            int(fault_start_step)
            if fault_start_step is not None
            else 10**12
        )

        if not 0.0 < self.restriction_factor <= 1.0:
            raise ValueError("restriction_factor must be in (0, 1].")

        self.input = {"requestedWaterFlowRate": tps.Scalar()}
        self.output = {"waterFlowRate": tps.Scalar(0)}
        self._config = {"parameters": []}

    @property
    def config(self):
        return self._config

    def initialize(self, start_time, end_time, step_size, simulator):
        self.input["requestedWaterFlowRate"].initialize(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            simulator=simulator,
        )
        self.output["waterFlowRate"].initialize(
            start_time=start_time,
            end_time=end_time,
            step_size=step_size,
            simulator=simulator,
        )

    def do_step(self, secondTime, dateTime, step_size, stepIndex):
        requested = torch.clamp(
            self.input["requestedWaterFlowRate"].get(),
            min=0.0,
        )
        factor = (
            self.restriction_factor
            if stepIndex >= self.fault_start_step
            else 1.0
        )
        self.output["waterFlowRate"].set(requested * factor, stepIndex)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one reproducible randomized Twin4Build building."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--case-id",
        default=None,
        help=(
            "Output case id. Default: random_building_seed_<seed>. "
            "The case is stored under data/raw/twin4build/<case-id>."
        ),
    )
    parser.add_argument("--fault-fraction", type=float, default=0.50)
    parser.add_argument("--n-levels", type=int, default=None)
    parser.add_argument("--rooms-per-level", type=int, default=None)
    parser.add_argument(
        "--ventilation-mode",
        choices=["occupied", "none"],
        default="occupied",
        help=(
            "Controlled ventilation ablation: 'occupied' retains the original "
            "occupancy-dependent balanced supply/exhaust airflow; 'none' sets "
            "both airflow rates to zero without changing occupancy or solar gains."
        ),
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-04-01")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing generated case.",
    )
    return parser.parse_args()


def constant_schedule(value: float) -> dict:
    return {
        "ruleset_default_value": float(value),
        "ruleset_start_hour": [],
        "ruleset_end_hour": [],
        "ruleset_start_minute": [],
        "ruleset_end_minute": [],
        "ruleset_value": [],
    }


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def scalar_history(system, port: str) -> np.ndarray:
    return tensor_to_numpy(system.output[port].history).reshape(-1)


def make_schedule(model, name: str, value: float):
    schedule = tb.ScheduleSystem(
        weekDayRulesetDict=constant_schedule(value),
        id=name,
    )
    model.add_component(schedule)
    return schedule


def make_timeseries_schedule(
    model,
    name: str,
    filename: Path,
    valuecolumn: int,
):
    schedule = tb.ScheduleSystem(
        useSpreadsheet=True,
        filename=str(filename),
        datecolumn=0,
        valuecolumn=valuecolumn,
        id=name,
    )
    model.add_component(schedule)
    return schedule


def to_utc_datetime(value: str) -> dt.datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def write_simulation_inputs(
    output_path: Path,
    start_time: dt.datetime,
    end_time: dt.datetime,
    step_size: int,
) -> pd.DataFrame:
    """Create case-local outdoor-temperature and disturbance schedules."""

    source = pd.read_csv(
        OPENSTUDIO_TOUT_PATH,
        usecols=["timestamp", "T_out"],
    )
    source["timestamp"] = pd.to_datetime(source["timestamp"]).dt.tz_localize(None)
    source = (
        source.groupby("timestamp", as_index=False)["T_out"]
        .first()
        .sort_values("timestamp")
    )

    start_naive = start_time.replace(tzinfo=None)
    end_naive = end_time.replace(tzinfo=None)

    target_index = pd.date_range(
        start=start_naive,
        end=end_naive,
        freq=f"{step_size}s",
        inclusive="left",
    )

    weather = (
        source.set_index("timestamp")
        .reindex(target_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    # Reproducible weekday occupancy profile. Room-specific people counts and
    # ventilation rates are obtained later by scaling this 0/1 base schedule.
    weekday = target_index.dayofweek < 5
    hour = target_index.hour + target_index.minute / 60.0
    occupied = weekday & (hour >= 8.0) & (hour < 17.0)
    occupancy_base = occupied.astype(float)

    # Simple deterministic winter solar disturbance. This is deliberately not
    # used by the diagnostic rules; it only prevents the thermal model from
    # being driven by outdoor temperature alone.
    solar_shape = np.zeros(len(target_index), dtype=float)
    daylight = (hour >= 8.0) & (hour <= 16.0)
    solar_shape[daylight] = np.sin(
        math.pi * (hour[daylight] - 8.0) / 8.0
    )
    day_of_year = target_index.dayofyear.to_numpy()
    slow_cloud_modulation = 0.65 + 0.20 * np.sin(
        2.0 * math.pi * day_of_year / 9.0
    )
    solar_wm2 = 180.0 * solar_shape * slow_cloud_modulation
    solar_wm2 = np.clip(solar_wm2, 0.0, None)

    inputs = pd.DataFrame(
        {
            "timestamp": target_index,
            "T_out": weather["T_out"].to_numpy(),
            "occupancy_base": occupancy_base,
            "solar_Wm2": solar_wm2,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs.to_csv(output_path, index=False)
    return inputs


def heat_loss_from_geometry(
    zone_area_m2: float,
    n_exterior_walls: int,
    window_area_m2: float,
    has_roof: bool,
    has_floor: bool,
) -> dict[str, float]:
    """Calculate aggregate room heat-loss coefficient H_total [W/K]."""

    area_scale = (zone_area_m2 / REFERENCE_ROOM_AREA_M2) ** 0.5
    wall_length = WALL_LENGTH_PER_25M2_ROOM_M * area_scale

    gross_external_wall_area = (
        n_exterior_walls * wall_length * ROOM_HEIGHT_M
    )
    net_external_wall_area = max(
        gross_external_wall_area - window_area_m2,
        0.0,
    )

    h_wall = U_VALUES_W_M2K["exterior_wall"] * net_external_wall_area
    h_window = U_VALUES_W_M2K["window"] * window_area_m2
    h_roof = U_VALUES_W_M2K["roof"] * zone_area_m2 if has_roof else 0.0
    h_floor = U_VALUES_W_M2K["floor"] * zone_area_m2 if has_floor else 0.0
    h_total = h_wall + h_window + h_roof + h_floor

    if h_total <= 0.0:
        raise ValueError("H_total must be positive.")

    return {
        "H_total_W_per_K": h_total,
        "H_wall_W_per_K": h_wall,
        "H_window_W_per_K": h_window,
        "H_roof_W_per_K": h_roof,
        "H_floor_W_per_K": h_floor,
        "gross_external_wall_area_m2": gross_external_wall_area,
        "net_external_wall_area_m2": net_external_wall_area,
        "window_area_m2": window_area_m2,
    }


def resistance_from_heat_loss(
    heat_loss_meta: dict[str, float],
) -> tuple[float, float]:
    """Split total room resistance equally between R_in and R_out."""

    r_total = 1.0 / heat_loss_meta["H_total_W_per_K"]
    r_half = r_total / 2.0
    return r_half, r_half


def count_exterior_sides(
    x: float,
    y: float,
    w: float,
    h: float,
    total_w: float,
    total_h: float,
    eps: float = 1e-6,
) -> int:
    sides = 0
    if abs(x) < eps:
        sides += 1
    if abs(y) < eps:
        sides += 1
    if abs((x + w) - total_w) < eps:
        sides += 1
    if abs((y + h) - total_h) < eps:
        sides += 1
    return max(1, sides)


def make_random_floor_plan(
    rng: random.Random,
    rooms_per_level: int,
    building_w: float,
    building_h: float,
) -> list[dict]:
    """Create one schematic plan that will be reused on every floor."""

    columns: list[str] = []
    remaining = rooms_per_level

    while remaining > 0:
        if remaining == 1:
            columns.append("full")
            remaining -= 1
        elif remaining == 2:
            columns.append("split")
            remaining -= 2
        elif rng.random() < 0.30:
            columns.append("full")
            remaining -= 1
        else:
            columns.append("split")
            remaining -= 2

    rng.shuffle(columns)

    raw_widths = [rng.uniform(0.7, 1.4) for _ in columns]
    width_sum = sum(raw_widths)
    widths = [building_w * value / width_sum for value in raw_widths]

    floor_rooms: list[dict] = []
    x = 0.0
    room_idx = 1

    for column_type, col_w in zip(columns, widths):
        if column_type == "full":
            floor_rooms.append(
                {
                    "room_idx": room_idx,
                    "plan_x": x,
                    "plan_y": 0.0,
                    "plan_w": col_w,
                    "plan_h": building_h,
                }
            )
            room_idx += 1
        else:
            split_ratio = rng.uniform(0.35, 0.65)
            top_h = building_h * split_ratio
            bottom_h = building_h - top_h

            floor_rooms.append(
                {
                    "room_idx": room_idx,
                    "plan_x": x,
                    "plan_y": 0.0,
                    "plan_w": col_w,
                    "plan_h": top_h,
                }
            )
            room_idx += 1

            floor_rooms.append(
                {
                    "room_idx": room_idx,
                    "plan_x": x,
                    "plan_y": top_h,
                    "plan_w": col_w,
                    "plan_h": bottom_h,
                }
            )
            room_idx += 1

        x += col_w

    for room in floor_rooms:
        room["zone_area_m2"] = float(rng.choice(ROOM_AREA_OPTIONS_M2))
        room["n_exterior_walls"] = count_exterior_sides(
            room["plan_x"],
            room["plan_y"],
            room["plan_w"],
            room["plan_h"],
            building_w,
            building_h,
        )

    return floor_rooms


def choose_fault_assignment(
    rng: random.Random,
    mechanism: str,
    n_steps: int,
) -> dict:
    """Return one randomized persistent fault specification."""

    severity = rng.choice(FAULT_SEVERITIES)
    onset_lo = max(1, int(FAULT_ONSET_FRACTION_RANGE[0] * n_steps))
    onset_hi = max(onset_lo, int(FAULT_ONSET_FRACTION_RANGE[1] * n_steps))
    fault_start_step = rng.randint(onset_lo, onset_hi)

    spec = {
        "fault_type": mechanism,
        "fault_severity": severity,
        "fault_start_step": fault_start_step,
        "restriction_factor": 1.0,
        "fixed_valve_position": None,
        "fixed_valve_quantile": None,
        "valve_upper_limit": None,
        "valve_lower_limit": None,
        "valve_limit_quantile": None,
        "fixed_valve_regime": None,
    }

    if mechanism == "restricted_flow":
        lo, hi = RESTRICTION_FACTOR_RANGES[severity]
        spec["restriction_factor"] = rng.uniform(lo, hi)

    elif mechanism == "fixed_valve":
        regime = rng.choice(["low", "high"])
        lo, hi = FIXED_VALVE_QUANTILE_RANGES[(regime, severity)]
        spec["fixed_valve_quantile"] = rng.uniform(lo, hi)
        spec["fixed_valve_regime"] = regime

    elif mechanism == "limited_opening":
        lo, hi = OPENING_LIMIT_QUANTILE_RANGES[severity]
        spec["valve_limit_quantile"] = rng.uniform(lo, hi)

    elif mechanism == "limited_closing":
        lo, hi = CLOSING_LIMIT_QUANTILE_RANGES[severity]
        spec["valve_limit_quantile"] = rng.uniform(lo, hi)

    else:
        raise ValueError(f"Unsupported fault mechanism: {mechanism}")

    return spec


def balanced_fault_mechanisms(
    rng: random.Random,
    n_faulty: int,
) -> list[str]:
    """Provide approximately balanced mechanism counts within each building."""

    mechanisms = [
        FAULT_MECHANISMS[i % len(FAULT_MECHANISMS)]
        for i in range(n_faulty)
    ]
    rng.shuffle(mechanisms)
    return mechanisms


# -----------------------------------------------------------------------------
# Room model construction
# -----------------------------------------------------------------------------

def add_room_branch(
    model,
    case: dict,
    t_out_schedule,
    t_set_schedule,
    t_supply_water_schedule,
    supply_air_temp_schedule,
    base_occupancy_schedule,
    solar_schedule,
    outdoor_co2_schedule,
    ventilation_mode: str = "occupied",
):
    suffix = case["suffix"]
    zone_area_m2 = case["zone_area_m2"]
    area_scale = zone_area_m2 / REFERENCE_ROOM_AREA_M2

    room = tb.BuildingSpaceTorchSystem(
        thermal_kwargs={
            "C_air": 1.0e6 * area_scale,
            "C_wall": 2.0e6 * area_scale,
            "C_int": 5.0e5 * area_scale,
            "R_out": case["R_out_K_per_W"],
            "R_in": case["R_in_K_per_W"],
            "R_int": 0.02,
            "f_wall": 0.2,
            "f_air": 0.05,
            "Q_occ_gain": 100.0,
        },
        mass_kwargs={},
        id=f"room_{suffix}",
    )

    heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=case["emitter_nominal_power_W"],
        T_a_nominal_sh=NOMINAL_SUPPLY_TEMP_C,
        T_b_nominal_sh=NOMINAL_RETURN_TEMP_C,
        TAir_nominal_sh=ROOM_SETPOINT_C,
        thermalMassHeatCapacity=5.0e5,
        nelements=10,
        id=f"baseboard_{suffix}",
    )

    controller = tb.PIDControllerSystem(
        kp=0.08,
        Ti=1800.0,
        Td=0.0,
        isReverse=True,
        id=f"pid_{suffix}",
    )

    valve_fault = ValveFaultInjectorSystem(
        mode=(
            case["fault_type"]
            if case["fault_type"] in {
                "fixed_valve",
                "limited_opening",
                "limited_closing",
            }
            else "healthy"
        ),
        fault_start_step=case["fault_start_step"],
        fixed_position=case["fixed_valve_position"],
        fixed_position_quantile=case["fixed_valve_quantile"],
        upper_limit=case["valve_upper_limit"],
        lower_limit=case["valve_lower_limit"],
        limit_quantile=case["valve_limit_quantile"],
        id=f"valve_fault_{suffix}",
    )

    valve = tb.ValveTorchSystem(
        waterFlowRateMax=case["healthy_valve_max_flow_kg_s"],
        valveAuthority=case["valve_authority"],
        id=f"valve_{suffix}",
    )

    hydraulic_restriction = HydraulicRestrictionSystem(
        restriction_factor=case["restriction_factor"],
        fault_start_step=case["fault_start_step"],
        id=f"restriction_{suffix}",
    )

    # Room-specific occupancy and ventilation scale with floor area while all
    # rooms share the same occupied/unoccupied timing.
    nominal_people = max(zone_area_m2 / 20.0, 0.25)
    # Keep occupancy and solar gains identical in both arms.  This switch
    # isolates ONLY mechanical supply/exhaust airflow from the room RC model.
    occupied_air_flow_kg_s = (
        0.010 * nominal_people if ventilation_mode == "occupied" else 0.0
    )

    occupancy_scaler = ScalarMultiplierSystem(
        multiplier=nominal_people,
        id=f"occupancy_scale_{suffix}",
    )
    airflow_scaler = ScalarMultiplierSystem(
        multiplier=occupied_air_flow_kg_s,
        id=f"airflow_scale_{suffix}",
    )

    components = [
        room,
        heater,
        controller,
        valve_fault,
        valve,
        hydraulic_restriction,
        occupancy_scaler,
        airflow_scaler,
    ]
    for component in components:
        model.add_component(component)

    # Controller -> fault injector -> physical valve.
    model.add_connection(t_set_schedule, controller, "scheduleValue", "setpointValue")
    model.add_connection(room, controller, "indoorTemperature", "actualValue")
    model.add_connection(
        controller,
        valve_fault,
        "inputSignal",
        "requestedValvePosition",
    )
    model.add_connection(
        valve_fault,
        valve,
        "actualValvePosition",
        "valvePosition",
    )

    # Healthy valve flow -> possible downstream hydraulic restriction -> heater.
    model.add_connection(
        valve,
        hydraulic_restriction,
        "waterFlowRate",
        "requestedWaterFlowRate",
    )
    model.add_connection(
        hydraulic_restriction,
        heater,
        "waterFlowRate",
        "waterFlowRate",
    )
    model.add_connection(
        t_supply_water_schedule,
        heater,
        "scheduleValue",
        "supplyWaterTemperature",
    )
    model.add_connection(room, heater, "indoorTemperature", "indoorTemperature")
    model.add_connection(heater, room, "Power", "heatGain")

    # Environmental inputs.
    model.add_connection(t_out_schedule, room, "scheduleValue", "outdoorTemperature")
    model.add_connection(
        supply_air_temp_schedule,
        room,
        "scheduleValue",
        "supplyAirTemperature",
    )
    model.add_connection(
        base_occupancy_schedule,
        occupancy_scaler,
        "scheduleValue",
        "inputSignal",
    )
    model.add_connection(
        occupancy_scaler,
        room,
        "outputSignal",
        "numberOfPeople",
    )
    model.add_connection(
        base_occupancy_schedule,
        airflow_scaler,
        "scheduleValue",
        "inputSignal",
    )
    model.add_connection(
        airflow_scaler,
        room,
        "outputSignal",
        "supplyAirFlowRate",
    )
    model.add_connection(
        airflow_scaler,
        room,
        "outputSignal",
        "exhaustAirFlowRate",
    )
    model.add_connection(solar_schedule, room, "scheduleValue", "globalIrradiation")
    model.add_connection(outdoor_co2_schedule, room, "scheduleValue", "outdoorCO2")

    return {
        **case,
        "room": room,
        "heater": heater,
        "controller": controller,
        "valve_fault": valve_fault,
        "valve": valve,
        "hydraulic_restriction": hydraulic_restriction,
        "nominal_people": nominal_people,
        "occupied_air_flow_kg_s": occupied_air_flow_kg_s,
    }


# -----------------------------------------------------------------------------
# Main generation routine
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not 0.0 <= args.fault_fraction <= 1.0:
        raise ValueError("--fault-fraction must be between 0 and 1.")

    seed = int(args.seed)
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    case_id = args.case_id or f"random_building_seed_{seed:04d}"
    case_dir = Path(f"data/raw/twin4build/{case_id}")
    operational_path = case_dir / "t4b_raw_timeseries.csv"
    truth_path = case_dir / "ground_truth.csv"
    building_manifest_path = case_dir / "building_manifest.csv"
    simulation_manifest_path = case_dir / "simulation_manifest.json"
    input_schedule_path = case_dir / "simulation_inputs.csv"

    existing_outputs = [
        p for p in (operational_path, truth_path, building_manifest_path,
                  simulation_manifest_path, input_schedule_path) if p.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            f"Existing case outputs: {existing_outputs}. "
            "Choose a NEW --case-id for each generator revision; "
            "use --overwrite only if intentional."
        )

    case_dir.mkdir(parents=True, exist_ok=True)

    start_time = to_utc_datetime(args.start_date)
    end_time = to_utc_datetime(args.end_date)
    if end_time <= start_time:
        raise ValueError("--end-date must be after --start-date.")

    inputs = write_simulation_inputs(
        output_path=input_schedule_path,
        start_time=start_time,
        end_time=end_time,
        step_size=TIMESTEP_SECONDS,
    )
    n_steps = len(inputs)

    design_outdoor_temp_c = float(
        inputs["T_out"].quantile(DESIGN_OUTDOOR_QUANTILE)
    )

    n_levels = args.n_levels if args.n_levels is not None else rng.randint(1, 3)
    rooms_per_level = (
        args.rooms_per_level
        if args.rooms_per_level is not None
        else rng.randint(4, 10)
    )
    if n_levels < 1:
        raise ValueError("--n-levels must be >= 1.")
    if rooms_per_level < 2:
        raise ValueError("--rooms-per-level must be >= 2.")

    building_w = rng.uniform(6.0, 20.0)
    building_h = rng.uniform(6.0, 15.0)

    # One floor plan is created and then deliberately reused on every level.
    base_floor_plan = make_random_floor_plan(
        rng=rng,
        rooms_per_level=rooms_per_level,
        building_w=building_w,
        building_h=building_h,
    )

    room_cases: list[dict] = []

    for level_index in range(n_levels):
        for room in base_floor_plan:
            suffix = f"l{level_index + 1}_room_{room['room_idx']}"
            has_floor = level_index == 0
            has_roof = level_index == n_levels - 1

            window_area_m2 = WINDOW_TO_FLOOR_AREA_RATIO * room["zone_area_m2"]
            heat_loss_meta = heat_loss_from_geometry(
                zone_area_m2=room["zone_area_m2"],
                n_exterior_walls=room["n_exterior_walls"],
                window_area_m2=window_area_m2,
                has_roof=has_roof,
                has_floor=has_floor,
            )
            r_out, r_in = resistance_from_heat_loss(heat_loss_meta)

            design_delta_t = max(
                ROOM_SETPOINT_C - design_outdoor_temp_c,
                1.0,
            )
            design_heat_load_w = (
                heat_loss_meta["H_total_W_per_K"] * design_delta_t
            )
            emitter_sizing_margin = rng.uniform(*EMITTER_SIZING_MARGIN_RANGE)
            emitter_nominal_power_w = design_heat_load_w * emitter_sizing_margin
            healthy_valve_max_flow = emitter_nominal_power_w / (
                NOMINAL_WATER_DELTA_T_K * WATER_CP_J_PER_KG_K
            )
            valve_authority = rng.uniform(*VALVE_AUTHORITY_RANGE)

            net_wall_ratio = (
                heat_loss_meta["net_external_wall_area_m2"]
                / room["zone_area_m2"]
            )
            wall_h_ratio = (
                heat_loss_meta["H_wall_W_per_K"]
                / room["zone_area_m2"]
            )

            room_cases.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "suffix": suffix,
                    "zone": f"ROOM_{suffix.upper()}",
                    "baseboard": f"BASEBOARD_{suffix.upper()}",
                    "zone_area_m2": room["zone_area_m2"],
                    "n_exterior_walls": room["n_exterior_walls"],
                    "exposure_group": f"{room['n_exterior_walls']}_external_wall",
                    "external_wall_net_area_per_floor_area": net_wall_ratio,
                    "external_wall_H_per_floor_area": wall_h_ratio,
                    "has_floor": has_floor,
                    "has_roof": has_roof,
                    "R_out_K_per_W": r_out,
                    "R_in_K_per_W": r_in,
                    **heat_loss_meta,
                    "design_outdoor_temp_C": design_outdoor_temp_c,
                    "design_heat_load_W": design_heat_load_w,
                    "emitter_sizing_margin": emitter_sizing_margin,
                    "emitter_nominal_power_W": emitter_nominal_power_w,
                    "healthy_valve_max_flow_kg_s": healthy_valve_max_flow,
                    "valve_authority": valve_authority,
                    "fault_type": "healthy",
                    "fault_severity": "none",
                    "fault_start_step": 10**12,
                    "fault_start": None,
                    "restriction_factor": 1.0,
                    "fixed_valve_position": None,
                    "fixed_valve_quantile": None,
                    "fixed_valve_regime": None,
                    "valve_upper_limit": None,
                    "valve_lower_limit": None,
                    "valve_limit_quantile": None,
                    "layout_position": "random_plan",
                    "plan_x": room["plan_x"],
                    "plan_y": room["plan_y"],
                    "plan_w": room["plan_w"],
                    "plan_h": room["plan_h"],
                    "level_name": f"Level {level_index + 1}",
                    "level_index": level_index,
                    "building_width_m": building_w,
                    "building_depth_m": building_h,
                    "layout_source": "shared_random_synthetic_floor_plan",
                }
            )

    # Approximately balanced fault mechanisms, with random room assignment.
    n_faulty = int(round(args.fault_fraction * len(room_cases)))
    if args.fault_fraction > 0 and n_faulty == 0:
        n_faulty = 1
    faulty_indices = rng.sample(range(len(room_cases)), n_faulty)
    mechanisms = balanced_fault_mechanisms(rng, n_faulty)

    for idx, mechanism in zip(faulty_indices, mechanisms):
        spec = choose_fault_assignment(
            rng=rng,
            mechanism=mechanism,
            n_steps=n_steps,
        )
        room_cases[idx].update(spec)
        onset_naive = pd.Timestamp(inputs.loc[spec["fault_start_step"], "timestamp"])
        room_cases[idx]["fault_start"] = onset_naive.tz_localize("UTC").isoformat()

    # Build the simulation only after the room design and injected ground truth
    # have been completely defined.
    model = tb.Model(id=case_id)
    t_out = make_timeseries_schedule(
        model,
        "schedule_outdoor_temperature",
        input_schedule_path,
        valuecolumn=1,
    )
    base_occupancy = make_timeseries_schedule(
        model,
        "schedule_occupancy_base",
        input_schedule_path,
        valuecolumn=2,
    )
    solar = make_timeseries_schedule(
        model,
        "schedule_solar",
        input_schedule_path,
        valuecolumn=3,
    )
    t_set = make_schedule(model, "schedule_room_setpoint", ROOM_SETPOINT_C)
    t_supply_water = make_schedule(
        model,
        "schedule_supply_water_temperature",
        NOMINAL_SUPPLY_TEMP_C,
    )
    supply_air_temp = make_schedule(
        model,
        "schedule_supply_air_temperature",
        SUPPLY_AIR_TEMP_C,
    )
    outdoor_co2 = make_schedule(model, "schedule_outdoor_co2", OUTDOOR_CO2_PPM)

    branches = [
        add_room_branch(
            model=model,
            case=case,
            t_out_schedule=t_out,
            t_set_schedule=t_set,
            t_supply_water_schedule=t_supply_water,
            supply_air_temp_schedule=supply_air_temp,
            base_occupancy_schedule=base_occupancy,
            solar_schedule=solar,
            outdoor_co2_schedule=outdoor_co2,
            ventilation_mode=args.ventilation_mode,
        )
        for case in room_cases
    ]

    print("Random building generated")
    print(f"  case id: {case_id}")
    print(f"  seed: {seed}")
    print(f"  levels: {n_levels}")
    print(f"  rooms per level: {rooms_per_level}")
    print(f"  total rooms: {len(room_cases)}")
    print(f"  faulty rooms: {n_faulty}")
    print(f"  ventilation mode: {args.ventilation_mode}")
    print(f"  building rectangle: {building_w:.1f} m x {building_h:.1f} m")
    print(f"  design outdoor temperature: {design_outdoor_temp_c:.2f} C")
    print(
        "  physical room areas on every level:",
        [room["zone_area_m2"] for room in base_floor_plan],
    )
    print("  fault mechanisms:")
    print(pd.Series([c["fault_type"] for c in room_cases]).value_counts().to_string())

    model.load(draw_simulation_model=False)
    simulator = tb.Simulator(model)
    simulator.simulate(
        step_size=TIMESTEP_SECONDS,
        start_time=start_time,
        end_time=end_time,
    )

    timestamps = pd.to_datetime(simulator.dateTimeSteps)
    t_out_values = inputs["T_out"].to_numpy()
    occupancy_base_values = inputs["occupancy_base"].to_numpy()

    operational_rows: list[pd.DataFrame] = []

    for branch in branches:
        room = branch["room"]
        heater = branch["heater"]
        valve = branch["valve"]
        restriction = branch["hydraulic_restriction"]
        valve_fault = branch["valve_fault"]

        # Evaluation-only realization metadata. These fields are intentionally
        # never added to the blind operational dataset.
        if branch["fault_type"] in {
            "fixed_valve",
            "limited_opening",
            "limited_closing",
        }:
            branch["valve_binding_fraction"] = float(valve_fault.binding_fraction)
            branch["valve_binding_count"] = int(valve_fault.binding_count)
            branch["valve_mean_absolute_position_change"] = float(
                valve_fault.mean_absolute_position_change
            )
            branch["valve_post_fault_step_count"] = int(
                valve_fault.post_fault_step_count
            )
            branch["valve_fault_activated"] = bool(valve_fault.binding_count > 0)
        else:
            branch["valve_binding_fraction"] = np.nan
            branch["valve_binding_count"] = np.nan
            branch["valve_mean_absolute_position_change"] = np.nan
            branch["valve_post_fault_step_count"] = np.nan
            branch["valve_fault_activated"] = np.nan

        if branch["fault_type"] == "fixed_valve":
            branch["fixed_valve_position"] = valve_fault.fixed_position
        elif branch["fault_type"] == "limited_opening":
            branch["valve_upper_limit"] = valve_fault.upper_limit
        elif branch["fault_type"] == "limited_closing":
            branch["valve_lower_limit"] = valve_fault.lower_limit

        t_zone = scalar_history(room, "indoorTemperature")
        q_bb = scalar_history(heater, "Power")
        # IMPORTANT: the observed flow is downstream of the injected hydraulic
        # restriction, not the healthy valve's unconstrained internal output.
        m_dot = scalar_history(restriction, "waterFlowRate")
        valve_position = scalar_history(valve, "valvePosition")
        t_return = scalar_history(heater, "outletWaterTemperature")

        n = min(
            len(timestamps),
            len(t_zone),
            len(q_bb),
            len(m_dot),
            len(valve_position),
            len(t_return),
            len(t_out_values),
            len(occupancy_base_values),
        )

        occupancy_values = (
            occupancy_base_values[:n] * branch["nominal_people"]
        )

        # Operational-only blind FDD input. Do NOT add fault_type,
        # restriction_factor, fixed/limited valve parameters, healthy maximum
        # flow, or the internal controller command to this file.
        operational_rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps[:n],
                    "case_id": case_id,
                    "archetype": "twin4build_randomized_synthetic_building",
                    "construction_year": CONSTRUCTION_YEAR,
                    "zone": branch["zone"],
                    "baseboard": branch["baseboard"],
                    "zone_area_m2": branch["zone_area_m2"],
                    "n_exterior_walls": branch["n_exterior_walls"],
                    "external_wall_net_area_per_floor_area": branch[
                        "external_wall_net_area_per_floor_area"
                    ],
                    "external_wall_H_per_floor_area": branch[
                        "external_wall_H_per_floor_area"
                    ],
                    "exposure_group": branch["exposure_group"],
                    "layout_position": branch["layout_position"],
                    "plan_x": branch["plan_x"],
                    "plan_y": branch["plan_y"],
                    "plan_w": branch["plan_w"],
                    "plan_h": branch["plan_h"],
                    "level_name": branch["level_name"],
                    "level_index": branch["level_index"],
                    "building_width_m": branch["building_width_m"],
                    "building_depth_m": branch["building_depth_m"],
                    "layout_source": branch["layout_source"],
                    "T_out": t_out_values[:n],
                    "occupancy": occupancy_values,
                    "T_zone": t_zone[:n],
                    "T_set": ROOM_SETPOINT_C,
                    "Q_bb": q_bb[:n],
                    "E_bb": q_bb[:n] * TIMESTEP_SECONDS,
                    "m_dot": m_dot[:n],
                    "T_supply": NOMINAL_SUPPLY_TEMP_C,
                    "T_return": t_return[:n],
                    "valve_position": valve_position[:n],
                }
            )
        )

    operational = pd.concat(operational_rows, ignore_index=True)
    operational.to_csv(operational_path, index=False)

    # Ground truth: one row per room, never consumed by the FDD classifier.
    truth_cols = [
        "case_id",
        "seed",
        "zone",
        "baseboard",
        "fault_type",
        "fault_severity",
        "fault_start",
        "fault_start_step",
        "restriction_factor",
        "fixed_valve_position",
        "fixed_valve_quantile",
        "fixed_valve_regime",
        "valve_upper_limit",
        "valve_lower_limit",
        "valve_limit_quantile",
        "valve_binding_fraction",
        "valve_mean_absolute_position_change",
        "valve_binding_count",
        "valve_post_fault_step_count",
        "valve_fault_activated",
    ]
    ground_truth = pd.DataFrame(branches)[truth_cols]
    ground_truth.to_csv(truth_path, index=False)

    # Design/building metadata are kept separate from the operational file so
    # the blind diagnostic input stays compact and explicit.
    building_cols = [
        "case_id",
        "seed",
        "zone",
        "baseboard",
        "zone_area_m2",
        "n_exterior_walls",
        "exposure_group",
        "has_floor",
        "has_roof",
        "external_wall_net_area_per_floor_area",
        "external_wall_H_per_floor_area",
        "R_out_K_per_W",
        "R_in_K_per_W",
        "H_total_W_per_K",
        "H_wall_W_per_K",
        "H_window_W_per_K",
        "H_roof_W_per_K",
        "H_floor_W_per_K",
        "gross_external_wall_area_m2",
        "net_external_wall_area_m2",
        "window_area_m2",
        "design_outdoor_temp_C",
        "design_heat_load_W",
        "emitter_sizing_margin",
        "emitter_nominal_power_W",
        "healthy_valve_max_flow_kg_s",
        "valve_authority",
        "plan_x",
        "plan_y",
        "plan_w",
        "plan_h",
        "level_name",
        "level_index",
        "building_width_m",
        "building_depth_m",
        "layout_source",
    ]
    building_manifest = pd.DataFrame(room_cases)[building_cols]
    building_manifest.to_csv(building_manifest_path, index=False)

    simulation_manifest = {
        "case_id": case_id,
        "seed": seed,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "timestep_seconds": TIMESTEP_SECONDS,
        "n_levels": n_levels,
        "rooms_per_level": rooms_per_level,
        "total_rooms": len(room_cases),
        "fault_fraction_requested": args.fault_fraction,
        "faulty_rooms": n_faulty,
        "ventilation_mode": args.ventilation_mode,
        "supply_air_temperature_C": SUPPLY_AIR_TEMP_C,
        "occupied_supply_exhaust_kg_s_per_person": (0.010 if args.ventilation_mode == "occupied" else 0.0),
        "fault_mechanisms": FAULT_MECHANISMS,
        "fault_onset_fraction_range": list(FAULT_ONSET_FRACTION_RANGE),
        "restriction_factor_ranges": RESTRICTION_FACTOR_RANGES,
        "opening_limit_quantile_ranges": OPENING_LIMIT_QUANTILE_RANGES,
        "fixed_valve_quantile_ranges": {
            f"{regime}_{severity}": list(bounds)
            for (regime, severity), bounds in FIXED_VALVE_QUANTILE_RANGES.items()
        },
        "closing_limit_quantile_ranges": CLOSING_LIMIT_QUANTILE_RANGES,
        "limited_travel_reference": "room-specific pre-fault requested-valve quantiles",
        "fixed_valve_reference": "room-specific pre-fault requested-valve quantiles",
        "valve_authority_range": list(VALVE_AUTHORITY_RANGE),
        "emitter_sizing_margin_range": list(EMITTER_SIZING_MARGIN_RANGE),
        "design_outdoor_quantile": DESIGN_OUTDOOR_QUANTILE,
        "design_outdoor_temp_C": design_outdoor_temp_c,
        "same_floor_plan_on_all_levels": True,
        "operational_file_contains_ground_truth": False,
        "controller_command_exported": False,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation_schema": "FDD_validation_candidate_v3",
    }
    with simulation_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(simulation_manifest, f, indent=2)

    print()
    print("Saved blind operational data:")
    print(f"  {operational_path}")
    print("Saved evaluation-only ground truth:")
    print(f"  {truth_path}")
    print("Saved building metadata:")
    print(f"  {building_manifest_path}")
    print("Saved simulation manifest:")
    print(f"  {simulation_manifest_path}")
    print()

    print("Operational means by room:")
    print(
        operational.groupby("zone")[
            ["T_zone", "Q_bb", "m_dot", "valve_position"]
        ].mean().round(5).to_string()
    )
    print()
    print("Injected ground truth:")
    print(
        ground_truth[
            [
                "zone",
                "fault_type",
                "fault_severity",
                "fault_start",
                "restriction_factor",
                "fixed_valve_position",
                "valve_limit_quantile",
                "valve_upper_limit",
                "valve_lower_limit",
                "valve_binding_fraction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
