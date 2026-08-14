from pathlib import Path
import datetime as dt
import random

import numpy as np
import pandas as pd
import torch
import twin4build as tb


CASE_ID = "t4b_random_fault_six_room"
OUT_PATH = Path("data/raw/twin4build/random_fault_six_room/t4b_raw_timeseries.csv")

OPENSTUDIO_TOUT_PATH = Path(
    "data/processed/openstudio/apartment_1970/"
    "apartment_baseboard_zone_timeseries_clean_area_new.csv"
)

T4B_TOUT_SCHEDULE_PATH = Path(
    "data/raw/twin4build/random_fault_six_room/"
    "openstudio_winter_tout_schedule.csv"
)

BASEBOARD_NOMINAL_POWER_W = 1800.0
NOMINAL_SUPPLY_TEMP_C = 60.0
NOMINAL_RETURN_TEMP_C = 45.0
WATER_CP_J_PER_KG_K = 4180.0

NORMAL_VALVE_MAX_FLOW = BASEBOARD_NOMINAL_POWER_W / (
    (NOMINAL_SUPPLY_TEMP_C - NOMINAL_RETURN_TEMP_C) * WATER_CP_J_PER_KG_K
)

# Random fault scenario settings.
# Change RANDOM_SEED to generate another repeatable random case.
RANDOM_SEED = 42

FAULT_OPTIONS = [
    "healthy",
    "restricted_flow",
    "stuck_open_valve",
]



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



def make_timeseries_schedule(model, name: str, filename: Path):
    schedule = tb.ScheduleSystem(
        useSpreadsheet=True,
        filename=str(filename),
        datecolumn=0,
        valuecolumn=1,
        id=name,
    )
    model.add_component(schedule)
    return schedule


def write_openstudio_winter_tout_schedule(
    start_time: dt.datetime,
    end_time: dt.datetime,
    step_size: int,
) -> pd.DataFrame:
    source = pd.read_csv(OPENSTUDIO_TOUT_PATH, usecols=["timestamp", "T_out"])
    source["timestamp"] = pd.to_datetime(source["timestamp"]).dt.tz_localize(None)

    source = (
        source
        .groupby("timestamp", as_index=False)["T_out"]
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

    winter = (
        source
        .set_index("timestamp")
        .reindex(target_index)
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )

    T4B_TOUT_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    winter.to_csv(T4B_TOUT_SCHEDULE_PATH, index=False)

    return winter

def add_room_branch(
    model,
    suffix: str,
    valve_max_flow: float,
    zone_area_m2: float,
    t_out_schedule,
    t_set_schedule,
    t_supply_water_schedule,
    supply_air_temp_schedule,
    air_flow_schedule,
    occupancy_schedule,
    solar_schedule,
    outdoor_co2_schedule,
    r_out: float = 0.012,
    r_in: float = 0.012,
    n_exterior_walls: int | None = None,
    external_wall_net_area_per_floor_area: float | None = None,
    external_wall_H_per_floor_area: float | None = None,
    fault_type: str = "healthy",
    layout_position: str | None = None,
    forced_valve_position_schedule=None,
):
    area_scale = zone_area_m2 / 25.0

    room = tb.BuildingSpaceTorchSystem(
        thermal_kwargs={
            "C_air": 1.0e6 * area_scale,
            "C_wall": 2.0e6 * area_scale,
            "C_int": 5.0e5 * area_scale,
            "R_out": r_out,
            "R_in": r_in,
            "R_int": 0.02,
            "f_wall": 0.2,
            "f_air": 0.05,
            "Q_occ_gain": 100.0 * area_scale,
        },
        mass_kwargs={},
        id=f"room_{suffix}",
    )

    heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=BASEBOARD_NOMINAL_POWER_W,
        T_a_nominal_sh=60.0,
        T_b_nominal_sh=45.0,
        TAir_nominal_sh=21.0,
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

    valve = tb.ValveTorchSystem(
        waterFlowRateMax=valve_max_flow,
        valveAuthority=1.0,
        id=f"valve_{suffix}",
    )

    for component in [room, heater, controller, valve]:
        model.add_component(component)

    # Control loop
    model.add_connection(t_set_schedule, controller, "scheduleValue", "setpointValue")
    model.add_connection(room, controller, "indoorTemperature", "actualValue")

    # Normal case: PID controls the valve.
    # Fault case: valve position is forced, while PID signal is still logged.
    if forced_valve_position_schedule is None:
        model.add_connection(controller, valve, "inputSignal", "valvePosition")
    else:
        model.add_connection(
            forced_valve_position_schedule,
            valve,
            "scheduleValue",
            "valvePosition",
        )

    # Hydronic loop
    model.add_connection(valve, heater, "waterFlowRate", "waterFlowRate")
    model.add_connection(t_supply_water_schedule, heater, "scheduleValue", "supplyWaterTemperature")
    model.add_connection(room, heater, "indoorTemperature", "indoorTemperature")

    # Heat into room
    model.add_connection(heater, room, "Power", "heatGain")

    # Room boundary and simple environmental inputs
    model.add_connection(t_out_schedule, room, "scheduleValue", "outdoorTemperature")
    model.add_connection(supply_air_temp_schedule, room, "scheduleValue", "supplyAirTemperature")
    model.add_connection(air_flow_schedule, room, "scheduleValue", "supplyAirFlowRate")
    model.add_connection(air_flow_schedule, room, "scheduleValue", "exhaustAirFlowRate")
    model.add_connection(occupancy_schedule, room, "scheduleValue", "numberOfPeople")
    model.add_connection(solar_schedule, room, "scheduleValue", "globalIrradiation")
    model.add_connection(outdoor_co2_schedule, room, "scheduleValue", "outdoorCO2")

    return {
        "suffix": suffix,
        "room": room,
        "heater": heater,
        "controller": controller,
        "valve": valve,
        "zone_area_m2": zone_area_m2,
        "valve_max_flow": valve_max_flow,
        "n_exterior_walls": n_exterior_walls,
        "external_wall_net_area_per_floor_area": external_wall_net_area_per_floor_area,
        "external_wall_H_per_floor_area": external_wall_H_per_floor_area,
        "exposure_group": f"{n_exterior_walls}_external_wall",
        "fault_type": fault_type,
        "layout_position": layout_position,
    }


def main():
    tz = dt.timezone.utc
    start_time = dt.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz)
    end_time = dt.datetime(2023, 4, 1, 0, 0, 0, tzinfo=tz)
    step_size = 900  # 15 minutes

    winter_tout = write_openstudio_winter_tout_schedule(
        start_time=start_time,
        end_time=end_time,
        step_size=step_size,
    )

    model = tb.Model(id=CASE_ID)

    # Shared schedules
    t_out = make_timeseries_schedule(model, "schedule_outdoor_temperature", T4B_TOUT_SCHEDULE_PATH)
    t_set = make_schedule(model, "schedule_room_setpoint", 21.0)
    t_supply_water = make_schedule(model, "schedule_supply_water_temperature", 60.0)
    supply_air_temp = make_schedule(model, "schedule_supply_air_temperature", 18.0)
    air_flow = make_schedule(model, "schedule_air_flow", 0.0)
    occupancy = make_schedule(model, "schedule_occupancy", 0.0)
    solar = make_schedule(model, "schedule_solar", 0.0)
    outdoor_co2 = make_schedule(model, "schedule_outdoor_co2", 420.0)

    # Valve leakage / stuck-open fault proxy:
    # the actuator remains partly open even if the controller would reduce demand.
    leakage_valve_position = make_schedule(model, "schedule_leakage_valve_position", 0.15)

    # Stuck-open valve fault proxy:
    # actuator remains partly open even if the controller would reduce demand.
    stuck_open_valve_position = make_schedule(model, "schedule_stuck_open_valve_position", 0.045)

    def resistance_from_exposure(zone_area_m2: float, n_exterior_walls: int) -> tuple[float, float]:
        """
        Approximate resistance scaling:
        2 external walls at 25 m²: R_out = R_in = 0.018
        1 external wall at 25 m²: R_out = R_in = 0.036

        Area scaling keeps heat-loss intensity comparable when room area changes.
        """
        base_r = 0.018 if n_exterior_walls == 2 else 0.036
        area_scaled_r = base_r * (25.0 / zone_area_m2)
        return area_scaled_r, area_scaled_r

    def exposure_values(n_exterior_walls: int) -> tuple[float, float]:
        """
        Approximate OpenStudio exposure values:
        2 external walls: H/A ≈ 0.45 W/K/m², net wall area/floor area ≈ 0.76
        1 external wall: H/A ≈ 0.22 W/K/m², net wall area/floor area ≈ 0.38
        """
        if n_exterior_walls == 2:
            return 0.76, 0.45
        return 0.38, 0.22

    rng = random.Random(RANDOM_SEED)

    base_room_cases = [
        {
            "suffix": "room_1",
            "zone_area_m2": 20.0,
            "n_exterior_walls": 2,
            "layout_position": "top_left_corner",
        },
        {
            "suffix": "room_2",
            "zone_area_m2": 20.0,
            "n_exterior_walls": 1,
            "layout_position": "top_middle_edge",
        },
        {
            "suffix": "room_3",
            "zone_area_m2": 30.0,
            "n_exterior_walls": 2,
            "layout_position": "top_right_corner",
        },
        {
            "suffix": "room_4",
            "zone_area_m2": 30.0,
            "n_exterior_walls": 2,
            "layout_position": "bottom_left_corner",
        },
        {
            "suffix": "room_5",
            "zone_area_m2": 15.0,
            "n_exterior_walls": 1,
            "layout_position": "bottom_middle_edge",
        },
        {
            "suffix": "room_6",
            "zone_area_m2": 15.0,
            "n_exterior_walls": 2,
            "layout_position": "bottom_right_corner",
        },
    ]

    room_cases = []

    # Ensure at least one room of each class appears in the test case.
    forced_faults = [
        "healthy",
        "restricted_flow",
        "stuck_open_valve",
    ]

    remaining_faults = [
        rng.choice(FAULT_OPTIONS)
        for _ in range(len(base_room_cases) - len(forced_faults))
    ]

    assigned_faults = forced_faults + remaining_faults
    rng.shuffle(assigned_faults)

    for base_case, fault_type in zip(base_room_cases, assigned_faults):
        case = dict(base_case)
        case["fault_type"] = fault_type

        if fault_type == "healthy":
            case["valve_max_flow_factor"] = 1.0
            case["forced_valve_position_schedule"] = None

        elif fault_type == "restricted_flow":
            case["valve_max_flow_factor"] = 0.35
            case["forced_valve_position_schedule"] = None

        elif fault_type == "stuck_open_valve":
            case["valve_max_flow_factor"] = 1.0
            case["forced_valve_position_schedule"] = stuck_open_valve_position

        else:
            raise ValueError(f"Unknown fault_type: {fault_type}")

        room_cases.append(case)

    print("Random fault assignment:")
    for case in room_cases:
        print(
            f"  {case['suffix']}: {case['fault_type']} "
            f"area={case['zone_area_m2']} m² "
            f"exposure={case['n_exterior_walls']}_external_wall"
        )

    branches = []

    for case in room_cases:
        r_out, r_in = resistance_from_exposure(
            case["zone_area_m2"],
            case["n_exterior_walls"],
        )
        ext_area_per_area, h_per_area = exposure_values(case["n_exterior_walls"])

        branch = add_room_branch(
            model=model,
            suffix=case["suffix"],
            valve_max_flow=case["valve_max_flow_factor"] * NORMAL_VALVE_MAX_FLOW,
            zone_area_m2=case["zone_area_m2"],
            t_out_schedule=t_out,
            t_set_schedule=t_set,
            t_supply_water_schedule=t_supply_water,
            supply_air_temp_schedule=supply_air_temp,
            air_flow_schedule=air_flow,
            occupancy_schedule=occupancy,
            solar_schedule=solar,
            outdoor_co2_schedule=outdoor_co2,
            r_out=r_out,
            r_in=r_in,
            n_exterior_walls=case["n_exterior_walls"],
            external_wall_net_area_per_floor_area=ext_area_per_area,
            external_wall_H_per_floor_area=h_per_area,
            fault_type=case["fault_type"],
            layout_position=case["layout_position"],
            forced_valve_position_schedule=case["forced_valve_position_schedule"],
        )
        branches.append(branch)

    model.load(draw_simulation_model=False)

    simulator = tb.Simulator(model)
    simulator.simulate(
        step_size=step_size,
        start_time=start_time,
        end_time=end_time,
    )

    timestamps = pd.to_datetime(simulator.dateTimeSteps)

    rows = []
    for branch in branches:
        room = branch["room"]
        heater = branch["heater"]
        controller = branch["controller"]
        valve = branch["valve"]

        T_zone = scalar_history(room, "indoorTemperature")
        Q_bb = scalar_history(heater, "Power")
        m_dot = scalar_history(valve, "waterFlowRate")
        valve_position = scalar_history(valve, "valvePosition")
        controller_signal = scalar_history(controller, "inputSignal")
        T_return = scalar_history(heater, "outletWaterTemperature")

        n = len(timestamps)
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "case_id": CASE_ID,
                "archetype": "twin4build_3x2_six_room",
                "construction_year": 1970,
                "zone": f"ROOM_{branch['suffix'].upper()}",
                "baseboard": f"BASEBOARD_{branch['suffix'].upper()}",
                "zone_area_m2": branch["zone_area_m2"],
                "n_exterior_walls": branch["n_exterior_walls"],
                "external_wall_net_area_per_floor_area": branch["external_wall_net_area_per_floor_area"],
                "external_wall_H_per_floor_area": branch["external_wall_H_per_floor_area"],
                "exposure_group": branch["exposure_group"],
                "fault_type": branch["fault_type"],
                "layout_position": branch["layout_position"],
                "T_out": winter_tout["T_out"].to_numpy()[:n],
                "occupancy": 0.0,
                "T_zone": T_zone[:n],
                "T_set": 21.0,
                "Q_bb": Q_bb[:n],
                "E_bb": Q_bb[:n] * step_size,
                "m_dot": m_dot[:n],
                "T_supply": 60.0,
                "T_return": T_return[:n],
                "valve_position": valve_position[:n],
                "controller_signal": controller_signal[:n],
                "valve_max_flow": branch["valve_max_flow"],
            }
        )
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print(out.groupby("zone")[["T_zone", "Q_bb", "m_dot", "valve_position"]].mean())
    print(out.groupby("zone")[["T_zone", "Q_bb", "m_dot", "valve_position"]].min())
    print(out.groupby("zone")[["T_zone", "Q_bb", "m_dot", "valve_position"]].max())


if __name__ == "__main__":
    main()
