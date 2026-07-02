from pathlib import Path
import datetime as dt

import numpy as np
import pandas as pd
import torch
import twin4build as tb


CASE_ID = "t4b_two_room_restricted_flow"
OUT_PATH = Path("data/raw/twin4build/two_room_restricted_flow/t4b_raw_timeseries.csv")


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
):
    room = tb.BuildingSpaceTorchSystem(
        thermal_kwargs={
            "C_air": 1.0e6,
            "C_wall": 2.0e6,
            "C_int": 5.0e5,
            "R_out": 0.08,
            "R_in": 0.04,
            "R_int": 0.02,
            "f_wall": 0.2,
            "f_air": 0.05,
            "Q_occ_gain": 100.0,
        },
        mass_kwargs={},
        id=f"room_{suffix}",
    )

    heater = tb.SpaceHeaterTorchSystem(
        Q_flow_nominal_sh=1000.0,
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
    model.add_connection(controller, valve, "inputSignal", "valvePosition")

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
    }


def main():
    tz = dt.timezone.utc
    start_time = dt.datetime(2023, 1, 1, 0, 0, 0, tzinfo=tz)
    end_time = dt.datetime(2023, 1, 8, 0, 0, 0, tzinfo=tz)
    step_size = 900  # 15 minutes

    model = tb.Model(id=CASE_ID)

    # Shared schedules
    t_out = make_schedule(model, "schedule_outdoor_temperature", 0.0)
    t_set = make_schedule(model, "schedule_room_setpoint", 21.0)
    t_supply_water = make_schedule(model, "schedule_supply_water_temperature", 60.0)
    supply_air_temp = make_schedule(model, "schedule_supply_air_temperature", 18.0)
    air_flow = make_schedule(model, "schedule_air_flow", 0.0)
    occupancy = make_schedule(model, "schedule_occupancy", 0.0)
    solar = make_schedule(model, "schedule_solar", 0.0)
    outdoor_co2 = make_schedule(model, "schedule_outdoor_co2", 420.0)

    normal = add_room_branch(
        model=model,
        suffix="normal",
        valve_max_flow=1000.0 / ((60.0 - 45.0) * 4180.0),
        zone_area_m2=25.0,
        t_out_schedule=t_out,
        t_set_schedule=t_set,
        t_supply_water_schedule=t_supply_water,
        supply_air_temp_schedule=supply_air_temp,
        air_flow_schedule=air_flow,
        occupancy_schedule=occupancy,
        solar_schedule=solar,
        outdoor_co2_schedule=outdoor_co2,
    )

    restricted = add_room_branch(
        model=model,
        suffix="restricted",
        valve_max_flow=0.35 * 1000.0 / ((60.0 - 45.0) * 4180.0),
        zone_area_m2=25.0,
        t_out_schedule=t_out,
        t_set_schedule=t_set,
        t_supply_water_schedule=t_supply_water,
        supply_air_temp_schedule=supply_air_temp,
        air_flow_schedule=air_flow,
        occupancy_schedule=occupancy,
        solar_schedule=solar,
        outdoor_co2_schedule=outdoor_co2,
    )

    model.load(draw_simulation_model=False)

    simulator = tb.Simulator(model)
    simulator.simulate(
        step_size=step_size,
        start_time=start_time,
        end_time=end_time,
    )

    timestamps = pd.to_datetime(simulator.dateTimeSteps)

    rows = []
    for branch in [normal, restricted]:
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
                "archetype": "twin4build_two_room",
                "construction_year": 1970,
                "zone": f"ROOM_{branch['suffix'].upper()}",
                "baseboard": f"BASEBOARD_{branch['suffix'].upper()}",
                "zone_area_m2": branch["zone_area_m2"],
                "T_out": 0.0,
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
