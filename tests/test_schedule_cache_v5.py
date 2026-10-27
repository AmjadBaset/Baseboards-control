"""Dependency-light tests for schedule-cache correction (no Twin4Build required)."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch.nn as nn
import pytest

SRC = Path(__file__).resolve().parents[1] / 'scripts' / '21_run_t4b_random_building_case_no_vent_physics_trace_v5.py'


class FakeSystem:
    def __init__(self, **kwargs):
        self.id = kwargs['id']


class FakeScalar:
    def __init__(self, is_leaf=False):
        self._history = None
        self.current = None

    def initialize(self, values=None, **kwargs):
        self._history = np.asarray(values, dtype=float).copy()

    def set(self, stepIndex):
        self.current = self._history[stepIndex]


class FakeModel:
    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)


def load_definitions():
    tree = ast.parse(SRC.read_text(encoding='utf-8'))
    names = {
        'InMemoryTimeSeriesScheduleSystem',
        'make_timeseries_schedule',
        'verify_actual_room_disturbance_inputs',
    }
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    ns = {
        'core': SimpleNamespace(System=FakeSystem),
        'nn': nn,
        'np': np,
        'pd': pd,
        'Path': Path,
        'tps': SimpleNamespace(Scalar=FakeScalar),
        'tensor_to_numpy': np.asarray,
        'scalar_history': lambda system, name: system.output[name]._history,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC), 'exec'), ns)
    return ns


def test_csv_columns_are_not_aliased(tmp_path):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=4, freq='15min').astype(str),
        'T_out': [-9., -2., 1., 3.],
        'occupancy_base': [0., 1., 0., 1.],
        'solar_Wm2': [0., 0., 10., 4.],
    })
    p = tmp_path / 'simulation_inputs.csv'
    df.to_csv(p, index=False)
    ns = load_definitions()
    model = FakeModel()
    schedules = {col: ns['make_timeseries_schedule'](model, col, p, idx)
                 for idx, col in [(1, 'T_out'), (2, 'occupancy_base'), (3, 'solar_Wm2')]}
    assert len(model.components) == 3
    sim = SimpleNamespace(dateTimeSteps=list(range(4)))
    for sched in schedules.values():
        sched.initialize(None, None, 900, sim)
        for i in range(4):
            sched.do_step(None, None, 900, i)
    for col, sched in schedules.items():
        assert np.allclose(sched.output['scheduleValue']._history, df[col])
    assert schedules['occupancy_base'].output['scheduleValue'].current == 1.

    ports = {
        'outdoorTemperature': FakeScalar(),
        'numberOfPeople': FakeScalar(),
        'globalIrradiation': FakeScalar(),
    }
    for port, arr in [
        ('outdoorTemperature', df['T_out']),
        ('numberOfPeople', df['occupancy_base'] * 1.5),
        ('globalIrradiation', df['solar_Wm2']),
    ]:
        ports[port]._history = np.asarray(arr, dtype=float)
    branches = [{'room': SimpleNamespace(input=ports), 'zone': 'ROOM1', 'nominal_people': 1.5}]
    ns['verify_actual_room_disturbance_inputs'](branches, df, {
        'T_out': schedules['T_out'],
        'occupancy_base': schedules['occupancy_base'],
        'solar_Wm2': schedules['solar_Wm2'],
    })

    ports['numberOfPeople']._history = df['T_out'].to_numpy(dtype=float)
    with pytest.raises(RuntimeError, match='negative numberOfPeople'):
        ns['verify_actual_room_disturbance_inputs'](branches, df, {
            'T_out': schedules['T_out'],
            'occupancy_base': schedules['occupancy_base'],
            'solar_Wm2': schedules['solar_Wm2'],
        })


def test_schedule_length_and_nonfinite_guards():
    ns = load_definitions()
    cls = ns['InMemoryTimeSeriesScheduleSystem']
    with pytest.raises(ValueError, match='finite'):
        cls(values=[1., float('nan')], id='bad')
    s = cls(values=[0,1,2], id='length')
    with pytest.raises(ValueError, match='samples'):
        s.initialize(None,None,900,SimpleNamespace(dateTimeSteps=[1,2]))
