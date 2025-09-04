import numpy as np
import open3d as o3d
import math
import random
import time
from tqdm import tqdm
import os
import multiprocessing as mp
import queue
import copy
from pynput import keyboard
import json
import pickle as pkl
from SimulationEngine import *
import warnings
warnings.filterwarnings("ignore", module="open3d")  # Ignores all Open3D warnings

class PLYManager:
    '''
    Description: Manages inner and out layer ply files and provide quantification metrics
    '''

    def __init__(self, object_file_name, origin_point=[0, 0, 0], print_info=True):
        self.origin_point = origin_point # first ball spawn origin
        self.object_file_name = object_file_name
    
        self.inner_pcd = None
        self.outer_pcd = None
        self.combined = None
        self.points = None

        self.load_ply(print_info)
        self.points = np.asarray(self.combined.points)
        if self.inner_pcd is not None and self.outer_pcd is not None:
            inpoints = np.asarray(self.inner_pcd.points)  # shape (M, 3)
            self.inpoints_set = {tuple(point) for point in inpoints}
            outpoints = np.asarray(self.outer_pcd.points)  # shape (M, 3)
            self.outpoints_set = {tuple(point) for point in outpoints}

        self.kdtree = o3d.geometry.KDTreeFlann(self.combined)

        self.centroid = np.mean(self.points, axis=0)
        min_bound = np.min(self.points, axis=0)
        max_bound = np.max(self.points, axis=0)
        self.bbox = self.combined.get_axis_aligned_bounding_box()
        self.bbox_center = (min_bound + max_bound) / 2
        self.bbox_diagonal = np.linalg.norm(np.asarray(self.bbox.get_max_bound()) - np.asarray(self.bbox.get_min_bound()))
        self.outer_radius = self.bbox_diagonal*1.0

    def load_ply(self, print_info):
        """
        Description: Load the inner and outer ply files associated with the given object file name.
        
        Returns: A numpy array of all points in the inner and outer ply files.
        """
        if print_info: print(f">>> loading {self.object_file_name + '_in.ply'}...")
        self.inner_pcd = o3d.io.read_point_cloud(self.object_file_name + "_in.ply")
        if print_info: print(f">>> loading {self.object_file_name + '_out.ply'}...")
        self.outer_pcd = o3d.io.read_point_cloud(self.object_file_name + "_out.ply")
        self.combined = self.get_ply_combined()
        if self.combined.has_points():
            if print_info: print(f"    - loaded {len(self.combined.points)} points")
        else:
            print(f"!!! No _in & _out files, loading: {self.object_file_name}.ply")
            self.combined = o3d.io.read_point_cloud(self.object_file_name + ".ply")
            self.inner_pcd = None
            self.outer_pcd = None
        if not self.combined.has_points():
            raise ValueError("Point cloud is empty")

        return self.combined
    
    def get_ply_combined(self):
        """
        Description: Combine inner and outer ply files into a single point cloud.

        Returns: A numpy array of all points in the inner and outer ply files.
        """
        if self.inner_pcd is None or self.outer_pcd is None:
            return self.combined
        else:
            return self.inner_pcd + self.outer_pcd
    
    def get_evaluation_metrics(self, inner_points_detected: np.ndarray):
        """
        Description: Evaluate the quality of the inner point cloud segmentation.
        
        Returns: a dictionary of:
            {accuracy:, precision:, recall:, f1_score:, inner_points_percentage:, 
             outer_points_percentage:, total_inner_points, total_outer_points}
        """
        # check if _in and _out ply files were loaded
        if self.inner_pcd is None or self.outer_pcd is None:
            return {"msg": "Metrics not available because _in and _out ply files were NOT loaded..."}
        inner_points = np.asarray(self.inner_pcd.points)
        outer_points = np.asarray(self.outer_pcd.points)
        P = len(inner_points) # P = TP + FN
        N = len(outer_points) # N = TN + FP
        TP = self.count_identical_points_kdtree(inner_points, inner_points_detected)
        FP = self.count_identical_points_kdtree(outer_points, inner_points_detected)
        TN = N - FP
        FN = P - TP
        accuracy = TP / (TP + FP + FN + 1e-5)
        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-5)
        inner_points_percentage = TP / (P + 1e-5)
        outer_points_percentage = FP / (N + 1e-5)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "inner_points_percentage": inner_points_percentage,
            "outer_points_percentage": outer_points_percentage,
            "total_inner_points": P,
            "total_outer_points": N
        }
    
    def count_identical_points_kdtree(self, points1: np.ndarray, points2: np.ndarray, distance_threshold=1e-5):
        """
        Counts the number of identical points between two point clouds using a KDTree.

        Args:
            pcd1 (o3d.geometry.PointCloud): The first point cloud.
            pcd2 (o3d.geometry.PointCloud): The second point cloud.
            distance_threshold (float, optional): The distance threshold for considering two points as identical. Default is 1e-5.

        Returns:
            int: The count of identical points found in pcd1 with respect to pcd2 within the specified distance threshold.
        """

        # points1 = np.asarray(pcd1.points)
        # points2 = np.asarray(pcd2.points)
        if len(points1) == 0 or len(points2) == 0:
            return 0

        # Build KDTree for pcd2
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)  # pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
        count = 0
        for i, p1 in enumerate(points1):
            # Find nearest neighbor in pcd2
            #  _, idxs, dists = pcd2_tree.search_knn_vector_3d(p1, k=len(points2))  # Or use radius search
            # OR better: use radius search
            query_point = p1.reshape(3, 1)
            [k, idx, _] = pcd2_tree.search_radius_vector_3d(query_point, distance_threshold)
            if k > 0:
                count += 1

        return count
    
    def find_point_index(self, point):
        """Find the index of a point in the point cloud using KDTree"""
        _, idx, _ = self.kdtree.search_knn_vector_3d(point, 1)
        return idx[0]
    
    def get_collided_points(self, point_collision_counts):
        """
        Returns a tuple of:
        1. Array of points that have been collided with (count > 0)
        """
        mask = point_collision_counts > 0
        return self.points[mask]
    
    def get_collided_inpoints(self, collision_points: np.ndarray):
        if self.inner_pcd is None:
            return np.array([])
        # Convert collision_points (list) to a NumPy array
        collision_points_np = np.asarray(collision_points)
        # Create mask using the converted NumPy array
        mask = np.array([tuple(point) in self.inpoints_set for point in collision_points_np], dtype=bool)
        # Index the converted array with the mask
        return collision_points_np[mask]
    
    def get_collided_outpoints(self, collision_points: np.ndarray):
        if self.outer_pcd is None:
            return np.array([])
        # Convert collision_points (list) to a NumPy array
        collision_points_np = np.asarray(collision_points)
        # Create mask using the converted NumPy array
        mask = np.array([tuple(point) in self.outpoints_set for point in collision_points_np], dtype=bool)
        # Index the converted array with the mask
        return collision_points_np[mask]
    
    def get_average_separation(self, sample_size):
        """
        Compute the average distance from each point in the cloud to its closest other point.
        
        Args:
            sample_size (float): ratio (from 0 to 1.0) of points to sample.
        
        Returns:
            float: the average separation distance.
        """
        
        n = len(self.combined.points)
        if n < 2:
            return 0.0
            
        n_sample = int(n * sample_size)
        if n_sample == 0:
            return 0.0
            
        indices = random.sample(range(n), n_sample)
        
        total_distance = 0.0
        for i in indices:
            point = self.combined.points[i]
            [k, idx, dist_squared] = self.kdtree.search_knn_vector_3d(point, 2)
            if k >= 2:
                if idx[0] == i:
                    d = dist_squared[1]
                else:
                    d = dist_squared[0]
                total_distance += math.sqrt(d)
        return total_distance / n_sample
    
    
def on_press(key, process):
    try:
        if key == keyboard.Key.esc:
            print("\nESC pressed - terminating child process")
            process.terminate()
            process.join()
            return False  # Stop listener
    except Exception as e:
        print(f"Error: {e}")

def visualizer_process(collision_queue, ply_obj_name, origin):
    # Load point cloud in child process
    point_cloud = PLYManager(ply_obj_name, print_info=False)
    point_cloud_np = np.asarray(point_cloud.points)
    bbox_size = np.max(point_cloud_np, axis=0) - np.min(point_cloud_np, axis=0)
    auto_size = np.max(bbox_size) * 0.1
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=auto_size, origin=[0, 0, 0])
    
    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Configure original point cloud
    original_pcd = point_cloud.combined
    original_pcd.paint_uniform_color([0.2, 0.2, 0.2])
    vis.add_geometry(original_pcd)
    
    # Create origin indicator
    bbox_diag = point_cloud.bbox_diagonal
    sphere_radius = bbox_diag * 0.015
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    origin_sphere.paint_uniform_color([0, 1, 0])
    origin_sphere.translate(origin)
    origin_sphere.compute_vertex_normals()
    vis.add_geometry(origin_sphere)
    
    # Create containers for real time visualization elements
    inner_spheres = o3d.geometry.TriangleMesh()
    bounce_dir = o3d.geometry.TriangleMesh()
    spawn_points = o3d.geometry.TriangleMesh()
    vis.add_geometry(inner_spheres)
    vis.add_geometry(bounce_dir)
    vis.add_geometry(spawn_points)
    
    # Set rendering options
    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.light_on = True
    render_opt.background_color = [1, 1, 1]
    vis.add_geometry(coordinate_frame)
    ctrl = vis.get_view_control()
    ctrl.set_front([-1, -1, -1])
    ctrl.set_up([0, 0, 1])
    
    # Initialize with empty collision points
    data = {'points': [], 'ball_scatter_dir': []}
    
    # Main visualization loop
    while True:
        data_received = False
        try:
            # Get latest data from queue (non-blocking)
            new_data = collision_queue.get_nowait()
            if new_data is None:  # Termination signal (now unused but kept)
                break
            data = new_data
            data_received = True
        except queue.Empty:
            pass
        
        # Update visualization if new data received
        if data_received:
            inner_spheres, bounce_dir, spawn_points = update_visualizer_child(vis, inner_spheres, bounce_dir, spawn_points, data, bbox_diag)
        
        # Handle window events/closure
        try:
            vis.poll_events()
            vis.update_renderer()
        except RuntimeError as e:
            print("error:", e)
            if 'window' in str(e).lower():
                break  # Break loop when window closed
            else:
                raise
        
        # Sleep to control frame rate
        time.sleep(0.02)

    # Cleanup (safe to call even if window closed)
    vis.destroy_window()

def update_visualizer_child(vis, inner_spheres, bounce_dir_geometry, spawn_points, data, bbox_diag):
    # Create new sphere geometry
    new_spheres = o3d.geometry.TriangleMesh()
    sphere_radius = bbox_diag * 0.003
    # Create new arrow geometry
    new_arrows = o3d.geometry.TriangleMesh()
    arrow_length = bbox_diag * 0.03
    # create new spawn point geometry
    new_spawn_points = o3d.geometry.TriangleMesh()
    spawn_radius = bbox_diag * 0.009
    
    for ind, (point, direction) in enumerate(zip(data['points'], data['ball_scatter_dir'])):        
        # Add sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(point)
        color_factor = 1 - (ind + 1) / len(data['points']) * 0.9
        sphere.paint_uniform_color([color_factor, color_factor, 0])
        if ind == len(data['points']) - 1:
            sphere.paint_uniform_color([1, 0, 0])
        if ind <= 1:
            sphere.paint_uniform_color([0, 1, 0])
        new_spheres += sphere
        
        # Add arrow if direction is valid
        if np.linalg.norm(direction) > 0.1:
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=arrow_length*0.04,
                cone_radius=arrow_length*0.1,
                cylinder_height=arrow_length*0.7,
                cone_height=arrow_length*0.2
            )
            z_axis = np.array([0, 0, 1])
            rot_axis = np.cross(z_axis, direction)
            rot_angle = np.arccos(np.dot(z_axis, direction))
            if np.linalg.norm(rot_axis) > 1e-6:
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
                arrow.rotate(R, center=[0, 0, 0])
            arrow.translate(point - direction * arrow_length * 0.1)
            arrow.paint_uniform_color([0.2, 0.2, 1])
            new_arrows += arrow

    for ind, spawn_point in enumerate(data['spawn_points']):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=spawn_radius)
        sphere.translate(spawn_point)
        color_factor = 0.9 - (ind + 1) / len(data['spawn_points']) * 0.9
        sphere.paint_uniform_color([(1-color_factor), color_factor, 0])
        if ind == len(data['points']) - 1:
            sphere.paint_uniform_color([1, 0, 0])
        if ind <= 1:
            sphere.paint_uniform_color([0, 1, 0])
        new_spawn_points += sphere

    # Update geometries
    vis.remove_geometry(inner_spheres, reset_bounding_box=False)
    inner_spheres = new_spheres
    vis.add_geometry(inner_spheres, reset_bounding_box=False)
    
    vis.remove_geometry(bounce_dir_geometry, reset_bounding_box=False)
    bounce_dir_geometry = new_arrows
    vis.add_geometry(bounce_dir_geometry, reset_bounding_box=False)

    vis.remove_geometry(spawn_points, reset_bounding_box=False)
    spawn_points = new_spawn_points
    vis.add_geometry(spawn_points, reset_bounding_box=False)
    
    vis.poll_events()
    vis.update_renderer()
    
    return inner_spheres, bounce_dir_geometry, spawn_points

def run_simulation(ply_obj_name, config):
    render = config['render']
    num_balls = config['num_balls']
    max_steps = config['max_steps']
    max_collisions = config['max_collisions']
    ball_radius_factor = config['ball_radius_factor']
    p  = config['p']
    diffusion = config['diffusion']

    # Load point cloud
    point_cloud = PLYManager(ply_obj_name, origin_point=config['origin_point'])
    avg_separation = point_cloud.get_average_separation(0.5)
    ball_radius = ball_radius_factor * avg_separation

    # Initialize simulator
    simulator = ScatterSimulator(point_cloud, ball_radius, p=p, num_balls=num_balls, render=render)
    # add first spawn point
    simulator.data['points'].append(simulator.origin)
    simulator.data['ball_scatter_dir'].append(np.array([0, 0, 0]))
    print(f"----> avg seperation: {avg_separation}")
    print(f"----> ball radius: {simulator.ball_radius}, collision margin: {simulator.collision_margin}, outer radius: {simulator.outer_radius}")
    print(f"----> origin: {simulator.origin}, t_max: {simulator.t_max}, bbox_diagonal: {simulator.bbox_diagonal}")

    # Create communication queue
    if render:
        collision_queue = mp.Queue(maxsize=2)
        # Start visualization process
        vis_proc = mp.Process(target=visualizer_process, 
                            args=(collision_queue, ply_obj_name, simulator.origin))
        vis_proc.daemon = True
        vis_proc.start()
    
    # Main simulation loop----------------------------------------------------------------------------
    print(">>> starting simulation...")
    start_time = time.time()
    last_update_time = time.time()
    simulation_data = {'inner_points': [], 'dup_count_history': None, 'escaped_count': 0, 'time_elapsed': -1, 'config': config}
    
    pbar = tqdm(range(num_balls), desc=f"Ball: {0}, inner points: {len(simulator.data['points'])}")
    for i in pbar:
        if i % (num_balls / 1000) == 0:
            pbar.set_description(f"Collected points: {len(simulator.data['points'])}, dup rate: {simulator.data['dup_rate'][-1]:.2f}, avg_moves: {simulator.avg_move_count:.2f}")
        simulator.simulate_ball(max_steps=max_steps, max_collisions=max_collisions, diffusion=diffusion)
        # Update visualization every 5 seconds
        current_time = time.time()
        if render and (current_time - last_update_time >= 1 or i == 0):
            try:
                # Send a deep copy of sampled collision points
                keys_to_keep = ['points', 'ball_scatter_dir']
                collision_data = {k: simulator.data[k] for k in keys_to_keep if k in simulator.data}
                keys_to_keep = ['spawn_points']
                vis_proc_data = {k: simulator.data[k] for k in keys_to_keep if k in simulator.data}
                merge_dict(vis_proc_data, sample_dict(collision_data, sample_size=1000, ordered=True))
                collision_queue.put_nowait(copy.deepcopy(vis_proc_data))
                last_update_time = current_time
            except queue.Full:
                pass
    end_time = time.time()

    # Final update with all collision points
    if render:
        try:
            keys_to_keep = ['points', 'ball_scatter_dir']
            collision_data = {k: simulator.data[k] for k in keys_to_keep if k in simulator.data}
            keys_to_keep = ['spawn_points']
            vis_proc_data = {k: simulator.data[k] for k in keys_to_keep if k in simulator.data}
            merge_dict(vis_proc_data, sample_dict(collision_data, sample_size=2000, ordered=True))
            collision_queue.put_nowait(copy.deepcopy(vis_proc_data))
        except queue.Full:
            pass

        # wait for user to press ESC to terminate vis window 
        with keyboard.Listener(
            on_press=lambda key: on_press(key, vis_proc)
        ) as listener:
            try:
                while vis_proc.is_alive():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error: {e}")
            finally:
                if vis_proc.is_alive():
                    vis_proc.terminate()
                    vis_proc.join()
                listener.stop()

    # get performance metrics ---------------------------------------------------------------------
    simulation_data['inner_points'] = point_cloud.get_collided_points(simulator.data['point_counts']) # get inner points from point counts, not points
    metrics = point_cloud.get_evaluation_metrics(simulation_data['inner_points'])
    simulation_data['point_counts'] = simulator.data['point_counts']
    simulation_data['dup_count_history'] = simulator.data['duplicate_counts']
    simulation_data['add_count_history'] = np.array(simulator.data['collision_count']) - np.array(simulator.data['duplicate_counts'])
    simulation_data['dup_rate_history'] = simulator.data['dup_rate']
    simulation_data['escape_history'] = simulator.data['escape_history']
    simulation_data['inner_added'] = simulator.data['inner_added']
    simulation_data['outer_added'] = simulator.data['outer_added']
    simulation_data['time_elapsed'] = end_time - start_time
    simulation_data['metrics'] = metrics

    # Print summary ----------------------------------------------------------------------------
    print(f"{'-'*80}\n>>> scatter process finished (total time: {end_time - start_time :.2f})")
    print(f"---> file total points: {len(np.asarray(simulator.point_cloud.get_ply_combined().points))}")
    print(f"---> inner points: {len(simulation_data['inner_points'])}")
    print(f"---> collisions ({sum((simulator.data['collision_count']))}), avg collisions per ball: {np.mean(simulator.data['collision_count'])}")
    print(f"---> removed points ({sum((simulator.data['duplicate_counts']))}), avg dup count per ball: {np.mean(simulator.data['duplicate_counts'])}")
    avg_dup_rate_total = np.array(simulator.data['duplicate_counts']) / (np.array(simulator.data['collision_count'])+0.01)
    print(f"---> avg dup rate: {np.mean(avg_dup_rate_total)}, abs dup rate: {sum((simulator.data['duplicate_counts']))/(sum((simulator.data['collision_count']))+0.01)}")
    print(f"point counts range: min: {simulator.data['point_counts'].min()}, max: {simulator.data['point_counts'].max()}, mean: {np.mean(simulator.data['point_counts'])}")
    
    return simulation_data


config_1 = {
    'simulation_name': 'hourglass_closed',
    'ply_file': './ply_files/test ply inputs/hourglass_closed',
    'origin_point': [0, 0, 20],
    'render': False,
    'num_balls' : 50000,
    'max_steps' : 50,
    'max_collisions': 5,
    'ball_radius_factor' : 2,
    'p' : 0.98,
    'diffusion' : True
}
config_2 = {
    'simulation_name': 'hourglass_opened',
    'ply_file': './ply_files/test ply inputs/hourglass_opened',
    'origin_point': [0, 0, 20],
    'render': True,
    'num_balls' : 50000,
    'max_steps' : 50,
    'max_collisions': 5,
    'ball_radius_factor' : 2,
    'p' : 0.98,
    'diffusion' : True
}
config_3 = {
    'simulation_name': 'snail',
    'ply_file': './ply_files/test ply inputs/snail',
    'origin_point': [0, 0, 0],
    'render': True,
    'num_balls' : 50000,
    'max_steps' : 50,
    'max_collisions': 50,
    'ball_radius_factor' : 2,
    'p' : 0.98,
    'diffusion' : True
}
config_4 = {
    'simulation_name': 'sphere',
    'ply_file': './ply_files/test ply inputs/sphere',
    'origin_point': [0, 0, 0],
    'render': True,
    'num_balls' : 50000,
    'max_steps' : 50,
    'max_collisions': 5,
    'ball_radius_factor' : 2,
    'p' : 0.98,
    'diffusion' : True
}
config_5 = {
    'simulation_name': 'stomach_1',
    'ply_file': './ply_files/reconstructed/stomach_1',
    'origin_point': [0, 0, 0],
    'render': True,
    'num_balls' : 50000,
    'max_steps' : 50,
    'max_collisions': 5,
    'ball_radius_factor' : 2,
    'p' : 0.98,
    'diffusion' : True
}
config_6 = {
    'simulation_name': 'stomach_2_unique_score_off',
    'ply_file': './ply_files/reconstructed/pc_layer0_2025-07-07-10-07',
    'origin_point': [0, 0, 0],
    'render': True,
    'num_balls' : 500000,
    'max_steps' : 50,
    'max_collisions': 5,
    'ball_radius_factor' : 2,
    'p' : 0.98,
    'diffusion' : True
}
config_7 = {
    'simulation_name': 'stomach_3',
    'ply_file': './ply_files/reconstructed/pc_layer0_2025-07-21-01-59',
    'origin_point': [0, 0, 0],
    'render': True,
    'num_balls' : 100000,
    'max_steps' : 50,
    'max_collisions': 5,
    'ball_radius_factor' : 2,
    'p' : 0.98,
    'diffusion' : True
}
config_8 = {
    'simulation_name': 'compartment',
    'ply_file': './ply_files/test ply inputs/compartment_v2',
    'origin_point': [0, 0, 0],
    'render': True,
    'num_balls' : 500000,
    'max_steps' : 50,
    'max_collisions': 5,
    'ball_radius_factor' : 3,
    'p' : 0.999,
    'diffusion' : True
}
config_9 = {
    'simulation_name': 'ball_unique_score_on',
    'ply_file': './ply_files/test ply inputs/ball',
    'origin_point': [0, 0, 0],
    'render': True,
    'num_balls' : 50000,
    'max_steps' : 50,
    'max_collisions': 5,
    'ball_radius_factor' : 2,
    'p' : 0.999,
    'diffusion' : True
}

def main():
    SIMULATION_NAME = config['simulation_name']
    INPUT_PLY_FILE = config['ply_file']
    OUTPUT_DIR = "./ply_files/output/"
    OUTPUT_PLY_FILE = f"./ply_files/output/output_{SIMULATION_NAME}.ply"
    OUTPUT_SIM_DATA = f"./ply_files/sim_data/sim_data_{SIMULATION_NAME}.ply"
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    print("")

    # Run main simulation
    print(f"{'-'*80}\nrunning main simulation...\n{'-'*80}")
    simulation_data = run_simulation(INPUT_PLY_FILE, config)
    print('\n')
    # Performance metrics
    print(f"{'-'*80}\nshowing performance metrics...\n{'-'*80}")
    print_dict(simulation_data['metrics'])
    print('\n')
    # Save results
    print(f"{'-'*80}\nSaving results...\n{'-'*80}")
    print(f">>> Outputting inner point cloud to {OUTPUT_PLY_FILE}")
    output_pcd = o3d.geometry.PointCloud()
    output_pcd.points = o3d.utility.Vector3dVector(simulation_data['inner_points'])
    o3d.io.write_point_cloud(OUTPUT_PLY_FILE, output_pcd)
    print(f">>> saving simulation data to {OUTPUT_SIM_DATA}")
    with open(OUTPUT_SIM_DATA, 'wb') as f:
        pkl.dump(simulation_data, f)
    print('\n')


config = config_6
if __name__ == "__main__":
    main()

'''name_loop = ["compartment_ballSize_0.1", "compartment_ballSize_0.2","compartment_ballSize_0.4",
             "compartment_ballSize_0.7", "compartment_ballSize_1", "compartment_ballSize_2",
             "compartment_ballSize_4", "compartment_ballSize_7", "compartment_ballSize_10",]
arg_a_loop = [0.1, 0.2, 0.4, 0.7, 1, 2, 4, 7, 10]


if __name__ == "__main__":
    for (name, arg_a) in zip(name_loop, arg_a_loop): 
        config['simulation_name'] = name
        config['ball_radius_factor'] = arg_a
        main()'''