# Twin4Build physics-trace check for FDD seed 2001

These are **diagnostic-only** scripts. Do not overwrite the production
`scripts/21_run_t4b_random_building_case.py` or use the trace CSV as FDD input.

1. Upload both Python files to `~/projects/Baseboards-control/scripts/`.
2. Activate project environment and set PYTHONPATH.
3. Run the physics-trace generator with seed 2001, a NEW case ID, and the
   same start/end dates as the failing trial.
4. Analyze its trace.

Commands:

```bash
cd ~/projects/Baseboards-control
source .venv/bin/activate
export PYTHONPATH=src:/home/amja/projects/Twin4Build

python scripts/21_run_t4b_random_building_case_physics_trace.py \
  --seed 2001 --case-id fdd_physics_trace_2001 \
  --start-date 2023-01-01 --end-date 2023-04-01

python scripts/27_analyze_t4b_physics_trace.py \
  --case-id fdd_physics_trace_2001
```

The generator copies v4's model and writes an **evaluation-only**
`physics_trace_evaluation_only.csv`, with the heat the radiator reports,
heat received at the room input, outdoor/air/solar/occupancy, and wall
state. It does not modify the thermal equations or blind operational CSV.

This exercise tests (i) radiator-to-room signal equality and (ii) whether
an approximate positive room-air energy balance contradicts falling measured
T_zone. The balance is only a directional check, not an exact one-step solver.

Do not use simulation output for FDD performance if physical checks fail.
