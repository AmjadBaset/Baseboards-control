#!/usr/bin/env python3
"""Read-only audit of Twin4Build room and radiator signals around impossible physics.

Usage:
    python scripts/26_audit_t4b_physics.py --case-id fdd_v4_trial_seed_2001

Does not modify simulation outputs and does not use ground truth for FDD.
Ground truth is used only to annotate whether an event occurred before/after onset.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CP_WATER_J_KG_K = 4180.0

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--root', default='data/raw/twin4build')
    parser.add_argument('--n-context', type=int, default=4)
    args = parser.parse_args()

    directory = Path(args.root) / args.case_id
    operational = directory / 't4b_raw_timeseries.csv'
    truth_file = directory / 'ground_truth.csv'
    if not operational.exists():
        parser.error(f'Missing operational data: {operational}')
    df = pd.read_csv(operational)
    needed = ['timestamp', 'zone', 'T_zone', 'T_out', 'T_set', 'Q_bb', 'T_supply', 'T_return', 'm_dot', 'valve_position']
    missing = [col for col in needed if col not in df.columns]
    if missing:
        parser.error(f'Missing columns: {missing}')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    truth = {}
    if truth_file.exists():
        gt = pd.read_csv(truth_file)
        if 'zone' in gt.columns:
            truth = {r.zone: r for r in gt.itertuples(index=False)}

    fmt_cols = ['timestamp', 'T_out', 'T_zone', 'T_set', 'dT_15min', 'Q_bb',
                'm_dot', 'valve_position', 'T_supply', 'T_return', 'Q_water_est_W']
    n_bad_rooms = 0
    summaries = []
    for zone, chunk in df.groupby('zone', sort=True):
        z = chunk.sort_values('timestamp').copy().reset_index(drop=True)
        z['dT_15min'] = z['T_zone'].diff()
        z['Q_water_est_W'] = CP_WATER_J_KG_K * z['m_dot'] * (z['T_supply'] - z['T_return'])
        cold_ambient = z['T_zone'] < z['T_out'] - 2.0
        extreme = (z['T_zone'] < -5.0) | (z['T_zone'] > 40.0)
        fast_change = z['dT_15min'].abs() > 5.0
        nonfinite = ~np.isfinite(z[['T_zone', 'Q_bb', 'm_dot', 'T_return']].to_numpy()).all(axis=1)
        fail = cold_ambient | extreme | fast_change | nonfinite
        truth_row = truth.get(zone)
        onset = None
        fault = '?'
        if truth_row is not None:
            fault = getattr(truth_row, 'fault_type', '?')
            onset_raw = getattr(truth_row, 'fault_start', None)
            if pd.notna(onset_raw):
                onset = pd.to_datetime(onset_raw, utc=True)
        summaries.append({
            'zone': zone, 'fault': fault, 'n_rows': len(z),
            'T_min_C': round(float(z['T_zone'].min()), 2),
            'below_Tout_minus2': int(cold_ambient.sum()),
            'extreme_T': int(extreme.sum()),
            'jump_over_5C': int(fast_change.sum()),
            'negative_heater_Q': int((z['Q_bb'] < -1.0).sum()),
            'nonfinite_rows': int(nonfinite.sum()),
        })
        if not fail.any():
            continue
        n_bad_rooms += 1
        first_index = int(np.flatnonzero(fail.to_numpy())[0])
        first = z.loc[first_index, 'timestamp']
        label = 'unknown' if onset is None else ('post-fault' if first >= onset else 'pre-fault')
        print(f'\n=== FIRST PHYSICS ANOMALY: {zone} ({fault}; {label}) ===')
        print(f'Fault onset: {onset}; first anomaly: {first}')
        lo = max(first_index - args.n_context, 0)
        hi = min(first_index + args.n_context + 1, len(z))
        print(z.loc[lo:hi-1, fmt_cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        if (z['T_zone'] < -5.0).any():
            worst_idx = int(z['T_zone'].idxmin())
            print('\nAt minimum T_zone:')
            print(z.loc[[worst_idx], fmt_cols].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\n=== CASE PHYSICS AUDIT SUMMARY ===')
    print(pd.DataFrame(summaries).to_string(index=False))
    if n_bad_rooms:
        print(f'\nINVALID SIMULATION: {n_bad_rooms} rooms violate basic thermal plausibility. Do not use for FDD performance metrics.')
    else:
        print('\nNo gross thermal-plausibility violations under these thresholds. This is a screening check, not a full physics validation.')

if __name__ == '__main__':
    main()
