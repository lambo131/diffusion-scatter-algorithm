


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

## Serial Run (recommended)

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



