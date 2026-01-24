
We propose a diffusion-based algorithm for separating the inter and outer layer surfaces from double-layered point clouds, particularly those exhibiting the "double surface artifact" caused by truncation in Truncated Signed Distance Function (TSDF) fusion during indoor or medical 3D reconstruction. This artifact arises from asymmetric truncation thresholds, leading to erroneous inter and outer shells in the fused volume, which our method addresses by extracting the true inter layer to mitigate challenges like overlapping surfaces and disordered normals. Our approach enables robust processing of both watertight and non-watertight 3D models, achieving extraction of the inter layer from 20,000 inter and 20,000 outer points in approximately 10 seconds. This solution is particularly effective for applications requiring accurate surface representations, such as indoor scene modeling and medical imaging, where double-layered point clouds are prevalent, and it accommodates both closed (watertight) and open (non-watertight) surface geometries. Moreover, this method is highly generalizable and does not require modifications to existing reconstruction algorithms.

![Diffusion-based interlayer point cloud reconstruction method.](images/pipeline5.png)
# Implementation

## Overview

The diffusion algorithm simulates the movement of a ball within a hollow
3D point cloud to identify inter points of the object's geometry. The
point cloud consists of two sets of points: inter surface points (red
points ) and outer surface points (black
points). The algorithm initializes a
simulation ball at a spawn point(the purple ball marked with 0), tracks its collisions with the
point cloud, and generates new spawn points to explore the geometry. The
simulation terminates when the simulation ball collision number reaches
a predefined limit or exits the escape boundary sphere which encloses
the entire point cloud as shown by the gray dashed sphere ).
![Diffusion algorithm visualization. The point cloud represents a hollow object with inter surface points (red) and outer surface points (black). Purple balls marked 0, 1, 2, and 3 represent the initial spawn point, reflected point, free-moving point, and escape point, respectively. The gray dashed sphere is the escape boundary sphere centered at the initial spawn point (0).](images/DiffusionAlgorithm.png)
### Initialization

The simulation begins by setting up the environment and parameters as
follows:

1.  **Point Cloud Loading**: Load the raw point cloud data of a hollow
    object as a NumPy array, representing 3D points that define the
    object's geometry. The point cloud consists of interior surface
    points (total number $N_{\text{inter}}$), representing the interior
    surface, and outer surface points (total number $N_{\text{outer}}$),
    representing the exterior surface. The total point count is
    $N_t = N_{\text{inter}} + N_{\text{outer}}$. The unit length of
    point cloud $R_0$ as the average length of the nearest point cloud
    distance.

2.  **Spawn Point Initialization**: Designate an initial spawn point
    manually (purple sphere marked \"0\" in
    Figure [\[fig:DiffusionAlgorithm\]](#fig:DiffusionAlgorithm){reference-type="ref"
    reference="fig:DiffusionAlgorithm"}) inside the point cloud, serving
    as the original spawn point of the simulation(usually the geometry
    center of the point cloud).

3.  **Escape Boundary Setup**: Define an escape boundary sphere centered
    at the bounding box center with a radius large enough to enclose the
    entire point cloud (gray dashed sphere in
    Figure [\[fig:DiffusionAlgorithm\]](#fig:DiffusionAlgorithm){reference-type="ref"
    reference="fig:DiffusionAlgorithm"}).

4.  **Simulation Parameters Setup**:

    -   $R_{\text{ball}}$: Radius of the simulation ball.

    -   $L_{\text{max}}$: Maximum distance the simulation ball moves in
        a single step if no collision occurs.

    -   $R_{eff}$: Effective collision radius, defined as
        $R_{eff}  = R_{\text{ball}} + \text{collision margin}$.
### Simulation Process

The simulation iteratively traces the trajectory of a simulation ball as
it moves, collides within the point cloud , or escapes from the point
cloud. The process is as follows:

1.  **Simulation Ball Initialization**:

    -   Spawn a simulation ball with radius $R_{\text{ball}}$ at a spawn
        point, chosen with probability $p_0$ for a random spawn point
        and $1-p_0$ for the initial spawn point. If no generated spawn
        points exist, default to the initial spawn point.

    -   Assign a random initial direction vector to the simulation ball.

2.  **Simulation Ball Trajectory**:

    -   The ball moves in a straight line for a distance up to
        $L_{\text{max}}$ unless a collision occurs.

    -   Upon collision with a point in the point cloud, compute a
        reflected direction based on the local surface geometry, adding
        a small random perturbation to simulate realistic scattering
        (e.g., transition from point 0 to 1 in
        Figure [\[fig:DiffusionAlgorithm\]](#fig:DiffusionAlgorithm){reference-type="ref"
        reference="fig:DiffusionAlgorithm"}).

    -   Track the number of steps (discrete movements) and collisions.
        Terminate the ball if it reaches predefined limits for steps or
        collisions.

    -   If the ball exits the point cloud (e.g., through a hole), it may
        collide with outer surface points(which can cause the false
        point cloud collision) or reach the escape boundary sphere
        (e.g., transition from point 2 to 3 in
        Figure [\[fig:DiffusionAlgorithm\]](#fig:DiffusionAlgorithm){reference-type="ref"
        reference="fig:DiffusionAlgorithm"}). Terminate the ball
        movement upon reaching the escape boundary sphere.

    More details about the ball movement and collision is written in
    Section [0.1](#sec:collision_detection){reference-type="ref"
    reference="sec:collision_detection"}.

3.  **Dynamic Spawn Point Generation**: Generate new spawn points during
    the simulation based on collision process (see
    Section [\[sec:spawn_point_generation\]](#sec:spawn_point_generation){reference-type="ref"
    reference="sec:spawn_point_generation"} for details).

4.  **Iterative Simulation**: Upon termination of a simulation ball (due
    to step/collision limits or escape), initialize a new simulation
    ball and repeat the process from **Simulation Ball Initialization**,
    incorporating generated spawn points and logged data.

### Simulation Termination

The simulation terminates when either of two conditions is met: the
total number of simulation balls reaches a predefined maximum, or the
duplication rate $R_{dup}$, as defined in
Section [\[sec:result\]](#sec:result){reference-type="ref"
reference="sec:result"}, reaches 0.99 for 10 consecutive iterations.
These conditions ensure the algorithm stops when sufficient exploration
is achieved or when further iterations yield minimal new information.

## Collision Detection 

During each simulation step, the simulation ball can either move freely,
collide with a point, or escape the boundary sphere:

-   **Free Movement**: Use Open3D's
    `kdtree.search_radius_vector_3d (current_ball_position, L_max +`$R_{eff}$` )`
    to identify cloud points within the ball's potential movement volume
    ($L_{\text{max}} + R_{eff}$). If no points are found, move the ball
    by $L_{\text{max}}$ in its current direction.

-   **Collision**: If points are found within the movement range,
    identify real collision points by checking for overlap with the
    ball's path (using effective collision radius $R_{eff}$). Select the
    closest point as the collision point. If no real collision occurs
    (e.g., no positive roots in the collision equation), move the ball
    by $L_{\text{max}}$.

-   **Escape**: After movement, check if the simulation ball's distance
    from the center of the escape boundary sphere exceeds the boundary's
    radius. If so, mark the simulation ball as escaped and terminate
    this simulation iterative.
## Spawn Point Generation 

To improve exploration of the 3D point cloud's internal geometry, new
spawn points are generated during the simulation. As the simulation ball
collides with the point cloud at multiple points, the two most recent
collision points are identified, and their midpoint is designated as a
new spawn point. This method leverages the likelihood that the midpoint
of two consecutive collisions lies within the internal volume of the 3D
model, facilitating effective exploration of its geometry. To maintain
diversity and avoid redundancy, a maximum of 200 spawn points is
enforced. Additionally, each new spawn point is checked for proximity to
existing spawn points; if it is too close (based on a predefined
distance threshold), it is discarded, and the algorithm waits for the
next collision to generate a new spawn point.

## Output Information

The algorithm outputs a Python dictionary containing the following
parameters: {Collide($i$)}, the set of point cloud points collided with
by the simulation ball at the $i$-th simulation step; {New($i$)}, the
set of unique (non-repeating) collided points at the $i$-th step;
{Dup($i$)}, the set of duplicate collided points (points hit multiple
times) at the $i$-th step; $C_{outer}$, the number of outer surface
points collided with by the simulation ball; $C_{inter}$, the number of
interior surface points collided with; and $N_{nescap}$, the number of
times the simulation ball escapes the designated escape boundary.


