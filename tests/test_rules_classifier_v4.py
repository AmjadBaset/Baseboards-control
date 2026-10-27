"""Causal FDD and blind-input checks; independent of Twin4Build installation."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / 'src/baseboard_diagnostics/diagnosis/rules.py'
CLASSIFIER = ROOT / 'scripts/11_classify_t4b_against_openstudio_bands.py'


def load_functions(path, names, namespace=None):
    source = ast.parse(path.read_text())
    body = [node for node in source.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
    ns={'pd':pd,'np':np}
    if namespace: ns.update(namespace)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
    return ns


def get_rules():
    import importlib.util
    spec=importlib.util.spec_from_file_location('fdd_rules_candidate_v4',RULES)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def windows(zone='R',kind='under',n_pre=4,n_post=4):
    records=[]
    for i in range(n_pre+n_post):
        post = i >= n_pre
        peak = .32 if post else [.51,.62,.49,.57][i%4]
        bottom = .10 if post else .03
        row={
            'case_id':'C','zone':zone,'baseboard':'B',
            'window_start':pd.Timestamp('2023-01-01T00:00:00Z')+pd.Timedelta(hours=3*i),
            'comfort_violation_fraction': 0.95 if post else 0.0,
            'overheating_fraction': 0.95 if post and kind=='over' else 0.0,
            'underheating_mean':1.5 if post and kind=='under' else 0.0,
            'overheating_mean':1.5 if post and kind=='over' else 0.0,
            'Q_density_mean_label':'normal',
            'm_dot_density_mean_label':'normal',
            'deltaT_water_mean_label':'normal',
            'valve_position_max': peak if kind=='under' else .62,
            'valve_position_min': .14 if kind=='over' and post else bottom,
            'valve_position_range': (.62-.14 if kind=='over' and post else peak-bottom),
            'valve_position_std': .15 if post else .20,
        }
        records.append(row)
    return pd.DataFrame(records)


def test_causal_upper_plateau_supports_limited_opening():
    r=get_rules()
    df=windows()
    diagnosed=r.apply_rules_to_labeled_windows(df)
    key='possible_partially_closed_or_stuck_low_valve'
    assert not diagnosed.iloc[:6]['diagnoses'].apply(lambda d:key in d).any()
    assert diagnosed.iloc[6]['diagnoses'].count(key)==1
    assert key in diagnosed.iloc[7]['diagnoses']
    assert 'possible_emitter_undersizing_or_high_load' not in diagnosed.iloc[6]['diagnoses']


def test_no_future_backpropagation():
    r=get_rules()
    df=windows()
    earlier=r.apply_rules_to_labeled_windows(df.iloc[:6].copy())
    full=r.apply_rules_to_labeled_windows(df)
    assert earlier['diagnosis_text'].tolist()==full.iloc[:6]['diagnosis_text'].tolist()
    # Reversing input row order must still give the same event time.
    shuffled=r.apply_rules_to_labeled_windows(df.sample(frac=1,random_state=8))
    assert shuffled.set_index('window_start')['diagnosis_text'].sort_index().tolist()==full.set_index('window_start')['diagnosis_text'].sort_index().tolist()


def test_group_isolation():
    r=get_rules()
    target=windows(zone='fault')
    healthy=windows(zone='healthy',n_post=0)
    combined=pd.concat([target,healthy],ignore_index=True)
    diagnosed=r.apply_rules_to_labeled_windows(combined)
    assert not diagnosed.loc[diagnosed['zone'].eq('healthy'),'has_diagnosis'].any()


def test_intermittent_underheating_does_not_make_travel_fault():
    r=get_rules()
    df=windows(n_pre=4,n_post=2)
    diagnosed=r.apply_rules_to_labeled_windows(df)
    assert not diagnosed['diagnosis_text'].str.contains('possible_partially_closed_or_stuck_low_valve').any()


def test_temporal_lower_plateau_detects_not_before_support():
    r=get_rules()
    df=windows(kind='over')
    diagnosed=r.apply_rules_to_labeled_windows(df)
    key='possible_valve_leakage_or_stuck_open'
    assert not diagnosed.iloc[:6]['diagnoses'].apply(lambda d:key in d).any()
    assert key in diagnosed.iloc[6]['diagnoses']


def test_blind_and_cadence_rejection():
    ns=load_functions(CLASSIFIER,['validate_blind_raw'],{'TIMESTEP_MINUTES':15,'FORBIDDEN_OPERATIONAL_COLUMNS':{'controller_signal','fault_type','valve_upper_limit'}})
    validate=ns['validate_blind_raw']
    raw=pd.DataFrame({'case_id':['C']*4,'zone':['Z']*4,'baseboard':['B']*4,
      'timestamp':pd.date_range('2023-01-01',periods=4,freq='15min',tz='UTC')})
    validate(raw)
    with pytest.raises(ValueError,match='controller'):
        validate(raw.assign(controller_signal=0.2))
    gap=raw.copy();gap.loc[2,'timestamp']=pd.Timestamp('2023-01-01T01:00:00Z')
    with pytest.raises(ValueError,match='missing/irregular'):
        validate(gap)
    duplicate=raw.copy();duplicate.loc[1,'timestamp']=duplicate.loc[0,'timestamp']
    with pytest.raises(ValueError,match='Duplicate'):
        validate(duplicate)
