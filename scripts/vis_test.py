import numpy as np
import open3d as o3d

# Static test (no multiprocessing)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))  # Random points
pcd.paint_uniform_color([1, 0, 0])  # Red

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.reset_view_point(True)  # Focus camera
vis.get_render_option().point_size = 5.0  # Larger points

vis.run()  # Window should show red points
vis.destroy_window()