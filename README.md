
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
