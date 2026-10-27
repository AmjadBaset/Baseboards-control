"""Small regression test for the room-aggregation-only hydraulic labels."""
from pathlib import Path
import importlib.util
import pandas as pd

p = Path(__file__).resolve().parents[1] / 'scripts' / '12_summarize_t4b_room_diagnosis.py'
spec = importlib.util.spec_from_file_location('room_agg', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fake_room(diagnosis, normal=2, n_diag=4):
    return pd.DataFrame({
        'diagnosis_text': [diagnosis] * n_diag + [m.NORMAL_DIAGNOSIS] * normal,
        'comfort_violation_fraction': [0.0] * (normal + n_diag),
    })


cases = {
    'possible_compensated_hydraulic_restriction': 'compensation_suspected_comfort_maintained',
    'hydraulic_restriction_degraded': 'delivery_degraded_comfort_maintained',
    'hydraulic_restriction_with_comfort_impact': 'comfort_impacted',
    'possible_flow_or_control_instability': 'flow_or_control_instability',
}
for diagnosis, expected in cases.items():
    result = m.summarize_room('A', fake_room(diagnosis), None, {}, 0.02)
    assert result['hydraulic_operational_status'] == expected
    assert result['dominant_fault_family'] == 'restricted_or_hydraulic'
    assert result[m.HYDRAULIC_EVIDENCE_COLUMNS[diagnosis]] == 4 / 6

result = m.summarize_room('B', fake_room(m.NORMAL_DIAGNOSIS, 0, 6), None, {}, 0.02)
assert result['hydraulic_operational_status'] == 'not_hydraulic_main_diagnosis'
assert all(result[col] == 0 for col in m.HYDRAULIC_EVIDENCE_COLUMNS.values())

# Neither the previous severity nor detailed diagnosis should change.
result = m.summarize_room('C', fake_room('possible_compensated_hydraulic_restriction'), None, {}, 0.02)
assert result['main_diagnosis'] == 'possible_compensated_hydraulic_restriction'
assert result['severity'] == 'warning'
print('PASS: hydraulic room aggregation: status, fractions, existing outputs')
