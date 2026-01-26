


![Diffusion-based interlayer point cloud reconstruction method.](images/pipeline5.png)
# Diffusion-Driven Inter–Outer Surface Separation for Non-Watertight Point Clouds

This repository implements a **diffusion-driven** (physics-inspired) algorithm to separate **inter surface points** and **outer surface points** from **double-layered point clouds**, including **non-watertight** (open) geometries. The core idea is to simulate **particle diffusion via random walks** inside a hollow point-cloud shell using a moving **simulation ball**, and to log **collided cloud points** to recover the **true inter layer**.

Repository: https://github.com/lambo131/diffusion-scatter-algorithm

---

## Terminology (consistent with the paper)

- **Double-layered point cloud**: a point cloud containing an **inter layer** and an **outer layer** (a “double surface”).
- **Inter surface points / outer surface points**: the two point sets forming the inner and outer shells.
- **Simulation ball**: the moving particle used to probe the geometry.
- **Spawn point**: the starting location of a simulation ball.
  - **Initial spawn point**: the user-chosen spawn point inside the point cloud.
  - **New spawn points**: dynamically generated during the simulation (**dynamic spawn point generation**).
- **Collision / collided cloud point**: when the simulation ball hits the point cloud; the hit point is logged.
- **Scattering / reflected direction**: after collision, the motion direction is reflected (based on local geometry) with a small random perturbation.
- **Escape boundary sphere**: a bounding sphere used to terminate balls that escape the model (important for non-watertight cases).
- **Duplication rate** (*R_dup*, as in the paper): measures repeated collisions on already-hit points and can be used as a convergence/termination signal.

---

## Repository Structure

- `Serial/` — single-process implementation (recommended starting point)
- `Parallel/` — parallel acceleration version
- `ply_files/` — example `.ply` inputs
- `images/` — figures for documentation

---

## Serial Run

### 1) Edit the configuration in `Serial/main.py`

To run the serial version, edit the `config` dictionary in `Serial/main.py`. Example:

```python
config_1 = {
    'simulation_name': 'hourglass_closed',
    'ply_file': './ply_files/test ply inputs/hourglass_closed',
    'origin_point': [0, 0, 20],
    'render': False,
    'num_balls': 50000,
    'max_steps': 50,
    'max_collisions': 5,
    'ball_radius_factor': 2,
    'p': 0.98,
    'diffusion': True
}
```
### Configuration Notes

In this configuration:

- **`ply_file`**: Path to the input `.ply` point cloud file.
- **`origin_point`**: The **initial spawn point** (user-defined).
- **`render`**:
  - `False` disables the rendering process
  - `True` enables visualization
- **`num_balls`**: Maximum number of simulation balls (i.e., maximum number of simulation iterations).
- **`max_steps`** (default: `50`): Per-ball step limit.
- **`max_collisions`** (default: `5`): Per-ball collision limit.
- **`ball_radius_factor`**: Sets the simulation ball radius as a multiple of the point-cloud base length scale (e.g., the average nearest-neighbor distance, consistent with the paper’s unit-length definition).
- **`p`**: Controls spawn-point selection:
  - with probability `p`, sample from the dynamically generated spawn-point pool
  - with probability `1-p`, restart from `origin_point`
- **`diffusion`**: `True` enables diffusion-driven mode (with dynamic spawn-point generation).

---

### Step 2 — Run From the repository root:

```bash
python Serial/main.py
```
---

## Parallel Run

### 1) Edit the configuration in `example_parallel_usage_with_output.py`

To run the parallel version, edit the `config` dictionary in `example_parallel_usage_with_output.py`. Example:

```python
config = {
    'ply_file': './ply_files/test ply inputs/ball',
    'origin_point': [0, 0, 0],
    'ball_radius_factor': 4,
    'num_balls': 50000,
    'max_steps': 50,
    'max_collisions': 5,
    'diffusion': True,
    'render': False,  # Must be False for parallel mode
    'num_processes': 8,  # Number of CPU cores (None for auto-detect)
}
```

### Configuration Notes

In this configuration:

- **`ply_file`**: Path to the input `.ply` point cloud file (without extension).
- **`origin_point`**: The **initial spawn point** (user-defined).
- **`ball_radius_factor`**: Sets the simulation ball radius as a multiple of the point-cloud base length scale.
- **`num_balls`**: Maximum number of simulation balls.
- **`max_steps`** (default: `50`): Per-ball step limit.
- **`max_collisions`** (default: `5`): Per-ball collision limit.
- **`diffusion`**: `True` enables diffusion-driven mode (with dynamic spawn-point generation).
- **`render`**: **Must be `False`** for parallel mode (Open3D visualization is not multiprocessing-safe).
- **`num_processes`**: Number of parallel processes to use. Set to `None` for automatic detection (recommended), or specify a number (e.g., `8` for 8 CPU cores).

### Additional Parallel-Specific Notes

- **Memory usage**: Each process loads its own copy of the point cloud. For large point clouds, consider reducing `num_processes` if memory is limited.
- **Performance**: Parallel execution typically provides 2-8x speedup depending on CPU cores and workload.
- **Output files**: Results are saved to `ply_files/output/` directory:
  - `output_<simulation_name>.ply` — detected inter surface points
  - `all_collisions_<simulation_name>.ply` — all collision points
  - `spawn_points_<simulation_name>.ply` — spawn point locations
  - `ply_files/sim_data/sim_data_<simulation_name>.pkl` — complete simulation data

---

### Step 2 — Run From the repository root:

```bash
python example_parallel_usage_with_output.py
```

