import numpy as np
import open3d as o3d
import math
import random
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
            points1 (np.ndarray): The first point cloud as numpy array.
            points2 (np.ndarray): The second point cloud as numpy array.
            distance_threshold (float, optional): The distance threshold for considering two points as identical. Default is 1e-5.

        Returns:
            int: The count of identical points found in points1 with respect to points2 within the specified distance threshold.
        """
        if len(points1) == 0 or len(points2) == 0:
            return 0

        # Build KDTree for points2
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
        count = 0
        for i, p1 in enumerate(points1):
            # Find nearest neighbor in pcd2
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
