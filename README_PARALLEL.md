# Parallel Scatter Simulation (Hole Detection)

This document explains how to run the parallelized scatter simulation, compare it with the serial version, and save outputs.

## Contents
- Requirements
- Project layout (relevant files)
- Quick start
- Serial vs parallel usage
- Configuration
- Outputs
- Performance tips
- Limitations

## Requirements
- Python 3.8+
- Open3D
- NumPy, tqdm

Install (example):
```bash
pip install open3d numpy tqdm
```

## Relevant files
- `PLYManager.py` — point cloud loading and utilities
- `SimulationEngine.py` — serial simulator (original)
- `ParallelScatterSimulator.py` — parallel simulator
- `example_parallel_usage.py` — simple parallel usage (no saving)
- `example_parallel_usage_with_output.py` — parallel usage with saved outputs
- `run_parallel_simulation.py` — comparison harness (parallel vs serial)

## Quick start
1) Place your input under `ply_files/test ply inputs/` as either:
   - paired files: `<name>_in.ply` and `<name>_out.ply`, or
   - a single `<name>.ply`.
2) Edit `example_parallel_usage_with_output.py`:
   ```python
   config = {
       'ply_file': './ply_files/test ply inputs/<name>',
       'num_processes': 8,
       'num_balls': 50000,
       'render': False,  # required for parallel
   }
   ```
3) Run:
```bash
python example_parallel_usage_with_output.py
```
4) Outputs will be written to `ply_files/output/`.

## Serial vs parallel
- Serial (original): `main.py` / `SimulationEngine.py`
- Parallel: `ParallelScatterSimulator.py`

Minimal switch to parallel:
```python
from PLYManager import PLYManager
from ParallelScatterSimulator import ParallelScatterSimulator

pc = PLYManager('./ply_files/test ply inputs/<name>')
ball_radius = pc.get_average_separation(0.5) * 0.5
sim = ParallelScatterSimulator(pc, ball_radius, num_balls=50000, num_processes=8, render=False)
sim.simulate_balls_parallel(max_steps=50, max_collisions=5, diffusion=True)
```

Compare performance:
```bash
python run_parallel_simulation.py
```

## Configuration
Common keys:
- `ply_file`: base path to your model (no extension). `_in.ply` and `_out.ply` are used if present.
- `origin_point`: spawn origin, e.g. `[0, 0, 0]`.
- `ball_radius_factor`: scales the ball radius by the average point separation.
- `num_balls`: total balls to simulate.
- `max_steps`, `max_collisions`: per-ball limits.
- `diffusion`: whether to diffuse new spawn points.
- `render`: must be `False` for parallel.
- `num_processes`: CPU cores to use (e.g., 8). `None` auto-detects.
- `batch_size`: optional; controls per-iteration workload.

## Outputs
From `example_parallel_usage_with_output.py`:
- `ply_files/output/all_collisions_<simulation_name>.ply`: all collided points
- `ply_files/output/output_<simulation_name>.ply`: inner points (if `_in/_out` available)
- `ply_files/output/spawn_points_<simulation_name>.ply`: spawn points
- `ply_files/sim_data/sim_data_<simulation_name>.pkl`: collected metrics and arrays

Note: with `render=False`, collisions are computed from `point_counts`, not `data['points']`.

## Performance tips
- Use a single persistent pool (already implemented).
- Build KDTree once per worker via `initializer` (already implemented).
- Set `num_processes` to available cores.
- Tune `chunksize` (in code) and optionally set `batch_size=config['num_balls']`.
- Disable rendering: `render=False`.
- Large models: watch memory — each process holds a KDTree.

Memory estimate per process:
- points: `N * 3 * 8` bytes
- KDTree: ~2–3× points array
- total per process: ~3–4× points array + small overhead

## Limitations
- Open3D visualizer is not multiprocessing-safe across many workers. Keep `render=False` in parallel.
- Workers hold their own KDTree (memory scales with processes).

## License
Add your preferred license section here.
