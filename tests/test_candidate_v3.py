"""Smoke/unit tests for supplied validation candidate, independent of Twin4Build runtime."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / 'scripts'


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

SUM = load_module('12_summarize_t4b_room_diagnosis')
EVAL = load_module('24_evaluate_t4b_random_building_batch')


def test_oversupply_family_mappings():
    diag = 'possible_oversupply_or_overheating'
    assert SUM.family_from_diagnosis(diag) == 'oversupply_or_overheating'
    assert EVAL.DIAGNOSIS_TO_EVAL_FAMILY[diag] == 'oversupply_or_overheating'


def test_persistence_prevents_healthy_warning():
    assert SUM.choose_main_diagnosis({'possible_flow_or_control_instability': 33/720},
              {'possible_flow_or_control_instability': .10185}, .02)[0] == SUM.NORMAL_DIAGNOSIS
    assert SUM.choose_main_diagnosis({'possible_compensated_hydraulic_restriction': .4},
              {}, .02)[0] == 'possible_compensated_hydraulic_restriction'
    assert SUM.final_room_severity(.15, SUM.NORMAL_DIAGNOSIS) == 'normal'


def test_invalid_calibration_is_rejected(tmp_path):
    p = tmp_path / 'p.csv'
    p.write_text('diagnosis,persistence_threshold\nproblem,1.5\n')
    with pytest.raises(ValueError):
        SUM.load_persistence_thresholds(p)
    p.write_text('diagnosis,persistence_threshold\nproblem,.2\nproblem,.3\n')
    with pytest.raises(ValueError):
        SUM.load_persistence_thresholds(p)


def test_temporal_delay_uses_window_end_and_truth_remains_separate(tmp_path):
    base = tmp_path
    op=base/'raw.csv'
    pd.DataFrame([{'zone':'Z_fault','timestamp':'2023-01-01T00:00:00Z'},
                  {'zone':'Z_healthy','timestamp':'2023-01-01T00:00:00Z'}]).to_csv(op,index=False)
    gt=base/'truth.csv'
    pd.DataFrame([
     {'zone':'Z_fault','fault_type':'limited_opening','fault_severity':'severe',
      'fault_start':'2023-01-01T03:00:00Z','valve_binding_fraction':.12},
     {'zone':'Z_healthy','fault_type':'healthy','fault_severity':'none','fault_start':None}
    ]).to_csv(gt,index=False)
    summary=base/'summary.csv'
    pd.DataFrame([
      {'zone':'Z_fault','dominant_fault_family':'stuck_low_valve','main_diagnosis':'possible_partially_closed_or_stuck_low_valve'},
      {'zone':'Z_healthy','dominant_fault_family':'normal','main_diagnosis':'normal_or_no_rule_triggered'}
    ]).to_csv(summary,index=False)
    building=base/'design.csv';pd.DataFrame([{'zone':'Z_fault'},{'zone':'Z_healthy'}]).to_csv(building,index=False)
    diagnosed=base/'diag.csv'
    pd.DataFrame([
      {'zone':'Z_fault','window_start':'2023-01-01T00:00:00Z','diagnosis_text':'normal_or_no_rule_triggered'},
      {'zone':'Z_fault','window_start':'2023-01-01T03:00:00Z','diagnosis_text':'possible_partially_closed_or_stuck_low_valve'},
      {'zone':'Z_healthy','window_start':'2023-01-01T00:00:00Z','diagnosis_text':'normal_or_no_rule_triggered'},
      {'zone':'Z_healthy','window_start':'2023-01-01T03:00:00Z','diagnosis_text':'normal_or_no_rule_triggered'},
    ]).to_csv(diagnosed,index=False)
    case=pd.Series({'case_id':'synthetic_seed_1','operational_path':str(op),'ground_truth_path':str(gt),
      'room_summary_path':str(summary),'diagnosed_path':str(diagnosed),'building_manifest_path':str(building)})
    room,temporal=EVAL.read_case(case)
    row=temporal.loc[temporal['zone']=='Z_fault'].iloc[0]
    assert row['first_detection_delay_h'] == 3.0
    assert row['first_correct_family_delay_h'] == 3.0
    assert int(room['predicted_fault'].sum()) == 1


def test_blind_input_rejects_gt(tmp_path):
    p=tmp_path/'raw.csv'
    pd.DataFrame({'timestamp':['2023-01-01'], 'controller_signal':[0.5]}).to_csv(p,index=False)
    with pytest.raises(ValueError):EVAL.verify_blind_operational_file(p)


def test_room_relative_fixed_valve_and_opening_limits():
    tree=ast.parse((ROOT/'21_run_t4b_random_building_case.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ValveFaultInjectorSystem')
    # Evaluate only the fault-injector class, not the Twin4Build simulation imports.
    class DummySystem: pass
    class Scalar:
        def __init__(self, x=None):self.value=x
        def get(self):return self.value
        def set(self,v,step):self.value=v
    env={'core':SimpleNamespace(System=DummySystem), 'nn':torch.nn, 'torch':torch,
         'np':np,'tps':SimpleNamespace(Scalar=Scalar),'__name__':'validator'}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<valve_injector>', 'exec'),env)
    valve=env['ValveFaultInjectorSystem']
    v=valve(mode='fixed_valve', fault_start_step=0, fixed_position_quantile=.90)
    v._pre_fault_requested=[.10,.15,.20,.25,.35]
    v.input['requestedValvePosition'].value=torch.tensor(.70)
    v.do_step(0,None,900,0)
    expected=float(np.quantile([.10,.15,.20,.25,.35],.90))
    assert abs(v.fixed_position-expected)<1e-6
    assert abs(v.output['actualValvePosition'].value.item()-expected)<1e-6
    assert v.binding_fraction == 1.0
    assert v.mean_absolute_position_change > .30
    v2=valve(mode='limited_opening',fault_start_step=0,limit_quantile=.40)
    v2._pre_fault_requested=[.10,.15,.20,.25,.35]
    v2.input['requestedValvePosition'].value=torch.tensor(.75)
    v2.do_step(0,None,900,0)
    assert v2.upper_limit < .75
    assert abs(v2.output['actualValvePosition'].value.item()-v2.upper_limit)<1e-6
