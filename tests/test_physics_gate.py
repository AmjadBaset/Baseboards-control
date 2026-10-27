from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from importlib.machinery import SourceFileLoader
import pandas as pd

screen = SourceFileLoader('phys_screen', str(Path(__file__).resolve().parents[1] / 'scripts' / '28_screen_t4b_physics_v5.py')).load_module().screen


class PhysicsGateTests(unittest.TestCase):
    def test_clean_then_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'simulation_manifest.json').write_text(json.dumps({
                'start_time': '2023-01-01T00:00:00+00:00',
                'end_time': '2023-01-01T01:00:00+00:00',
                'timestep_seconds': 900, 'total_rooms': 2, 'ventilation_mode': 'none'
            }))
            dates = pd.date_range('2023-01-01', periods=4, freq='15min', tz='UTC')
            rows = [dict(timestamp=t.isoformat(), zone=z, T_zone=21.0, T_out=-4.0, T_set=21.0,
                         Q_bb=200.0, m_dot=.003, T_supply=60.0, T_return=44.0, valve_position=.2)
                    for z in ('R1','R2') for t in dates]
            data = pd.DataFrame(rows)
            data.to_csv(p/'t4b_raw_timeseries.csv', index=False)
            self.assertEqual(screen(p)['screen_status'], 'PASS')
            data.loc[5, 'T_zone'] = -500.0
            data.to_csv(p/'t4b_raw_timeseries.csv', index=False)
            self.assertEqual(screen(p)['screen_status'], 'FAIL')

if __name__ == '__main__':
    unittest.main()
