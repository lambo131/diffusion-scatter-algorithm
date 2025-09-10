import numpy as np
import open3d as o3d
import math
import random
import time
from tqdm import tqdm
import multiprocessing as mp
# from sklearn.neighbors import KDTree
from utils import *
from PLYManager import PLYManager

class PointCloud:
    def __init__(self, file_path):
        self.pcd = o3d.io.read_point_cloud(file_path)
        if not self.pcd.has_points():
            raise ValueError("Point cloud is empty")
        self.points = np.asarray(self.pcd.points)
        self.kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        # Initialize scikit-learn KDTree
        # self.kdtree = KDTree(self.points, metric='euclidean')  # Use scikit-learn KDTree
        
        self.centroid = np.mean(self.points, axis=0)
        self.bbox = self.pcd.get_axis_aligned_bounding_box()
        self.bbox_diag = np.linalg.norm(np.asarray(self.bbox.get_max_bound()) - np.asarray(self.bbox.get_min_bound()))

    def get_points(self):
        return self.points

    def get_centroid(self):
        return self.centroid

    def get_bbox_diagonal(self):
        return self.bbox_diag

class Ball:
    def __init__(self, position, direction, radius, collision_margin):
        self.position = np.array(position, dtype=np.float64)
        assert self.position.shape == (3,), f"position must be a 3D vector [x,y,z], got shape {position.shape}"
        self.direction = np.array(direction, dtype=np.float64)
        self.direction /= np.linalg.norm(self.direction)
        assert self.direction.shape == (3,), f"direction must be a 3D vector [x,y,z], got shape {position.shape}"
        self.radius = radius
        self.collision_margin = collision_margin

    def set_position(self, position):
        self.position = np.array(position, dtype=np.float64)

    def set_direction(self, direction):
        self.direction = np.array(direction, dtype=np.float64)
        self.direction /= np.linalg.norm(self.direction)

class ScatterSimulator:
    def __init__(self, point_cloud: PLYManager, 
                 ball_radius, 
                 collision_margin_ratio=0.5, 
                 p=1, num_balls=1000, 
                 max_spawn_points=200, 
                 render=False):
        
        self.point_cloud = point_cloud
        self.bbox_center = point_cloud.bbox_center
        self.bbox_diagonal = point_cloud.bbox_diagonal
        self.outer_radius = point_cloud.outer_radius
        self.origin = point_cloud.origin_point
    
        self.ball_radius = ball_radius
        self.t_max = min(3 * ball_radius, self.bbox_diagonal*0.05)
        self.collision_margin = min(ball_radius * collision_margin_ratio, self.bbox_diagonal*0.02)
        self.p = p  # Likelyhood of spawning at new location
        self.total_balls = num_balls
        self.max_spawn_points = max_spawn_points
        self.render = render

        # data structures
        self.ball_count = 0
        self.escape_count = 0
        self.avg_move_count = 0
        self.spawn_point_collision_count = 0
        self.last_dup_rate = 0

        self.point_counts = np.zeros(len(self.point_cloud.points), dtype=np.int32)
        self.data = {'points': [], 'point_counts': self.point_counts, 'ball_scatter_dir': [], 
                    'duplicate_counts': [], 'collision_count': [], 'dup_rate': [0], 'escape_history': [0],
                    'inner_added': [0], 'outer_added': [0],
                    'spawn_points': [], 'spawn_point_unique_score': []
                    }
        self.existing_points_set = set()  # New: Track points for O(1) dup point lookups
        
        # Stop flag for early termination
        self.stop_flag = False
        self.recent_dup_rates = []  # Track last 10 duplicate rates

        # self.min_path_length = float('inf')
        # self.max_path_length = 0

    # def compute_normal(self, point, radius):
    #     [k, idx, _] = self.point_cloud.kdtree.search_radius_vector_3d(point, radius)
    #     if k < 3:
    #         return None
    #     points_norm = self.point_cloud.points[idx]
    #     cov = np.cov(points_norm.T)
    #     eigenvalues, eigenvectors = np.linalg.eigh(cov)
    #     return eigenvectors[:, 0]
    def compute_normal(self, point, radius):

        [k, idx, _] = self.point_cloud.kdtree.search_radius_vector_3d(point, radius)        
        k = len(idx)
        if k < 3:
            return None
        points_norm = self.point_cloud.points[idx]
        cov = np.cov(points_norm.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return eigenvectors[:, 0]

    def move_until_collision_or_escape(self, ball):
        r = ball.radius + ball.collision_margin
        current_pos, direction = ball.position, ball.direction

        query_point = current_pos.reshape(3, 1)  # Ensure shape is (3, 1)
        radius = float(r + self.t_max)  # Ensure radius is a float
        k, idx, _ = self.point_cloud.kdtree.search_radius_vector_3d(query_point, radius)
        # k, idx, _ = self.point_cloud.kdtree.search_hybrid_vector_3d(query_point, radius, 100)
        if k == 0:
            new_position = current_pos + self.t_max * direction
            if np.linalg.norm(new_position - self.bbox_center) > self.outer_radius:
                return 'escape', None, None
            else:
                ball.set_position(new_position)
                return 'moved', None, None

        candidate_points = self.point_cloud.points[idx]
        collision_point, t_min = find_collision(current_pos, direction, candidate_points, r)

        if t_min < self.t_max:
            ball.set_position(current_pos)
            return 'collision', collision_point, current_pos
        else:
            new_position = current_pos + self.t_max * direction
            ball.set_position(new_position)
            return 'moved', None, None

    def simulate_ball(self, max_steps=100, max_collisions=10, diffusion=False):
        last_collision_point = None  # Track last collision for midpoint calculation
        new_spawn_point = None
        set_spawn = False # flag after setting spawn
        generate_spawn = diffusion # whether to generate new spawn point
        escaped = False
        duplicated_count = 0
        collision_count = 0
        move_count = 0

        # spawn ball
        spawn_point = self.origin
        if np.random.rand() < self.p and len(self.data['spawn_points']) > 0 and diffusion:
            # spawn_point = self.data['spawn_points'][-1]
            spawn_point = random.choice(self.data['spawn_points'])
        initial_direction = np.random.randn(3)
        initial_direction /= np.linalg.norm(initial_direction)
        ball = Ball(spawn_point, initial_direction, self.ball_radius, self.collision_margin)

        for steps in range(max_steps):
            if collision_count >= max_collisions:
                break
            result, collision_point, new_position = self.move_until_collision_or_escape(ball)

            if result == 'collision':
                collision_count+=1
                # ------------compute new direction
                normal = self.compute_normal(collision_point, 3 * ball.radius)
                if normal is None:
                    normal = np.random.randn(3)
                    normal /= np.linalg.norm(normal)
                to_ball = ball.position - collision_point
                if np.dot(normal, to_ball) < 0:
                    normal = -normal
                v_old = ball.direction
                v_new = v_old - 2 * np.dot(v_old, normal) * normal + 0.05 * np.random.randn(3)
                v_new /= np.linalg.norm(v_new)
                # ------------track collisions
                collision_point_ind = self.point_cloud.find_point_index(collision_point)
                self.data['point_counts'][collision_point_ind] += 1
                if self.render:
                    if not contains_vector(collision_point, self.existing_points_set):  # Use the prebuilt set
                        self.data['points'].append(collision_point)
                        self.existing_points_set.add(tuple(collision_point.tolist()))  # Add to both list (for history) and set (for fast lookups)
                        self.data['ball_scatter_dir'].append(v_new)
                    else:
                        duplicated_count+=1
                else:
                    if self.data['point_counts'][collision_point_ind] > 1:
                        duplicated_count+=1
                    self.data['ball_scatter_dir'].append(v_new)
                # ------------update ball state
                ball.set_direction(v_new)
                ball.set_position(new_position)

                # Update spawn location when specific collision condition occurs
                new_spawn_point = None
                if len(self.data['collision_count'][-3:]) < 3:
                    valid_2 = False
                else:
                    valid_2 = self.spawn_point_collision_count > 0.3*np.mean(self.data['collision_count'][-3:])
                # check spawn point count
                spawn_point_count = len(self.data['spawn_points'])
                # if self.ball_count % 100 == 0 and spawn_point_count > 250:
                   #  self.data['spawn_points'] = random.sample(self.data['spawn_points'], math.floor(spawn_point_count * 0.85))

                if (collision_count > 2) and valid_2 and diffusion == True and generate_spawn == True:
                    # print((collision_count > max_steps/2), contains_vector(last_collision_point, self.data['points'][:-2]), contains_vector(collision_point, self.data['points'][:-1]))
                    # if (collision_count > 2) and not contains_vector(last_collision_point, self.data['points'][:-2]):
                    if last_collision_point is not None and self.data['dup_rate'][-1] > 0.5:
                        spawn_point_candidate = ((last_collision_point + collision_point) / 2.0)
                        k, idx, _ = self.point_cloud.kdtree.search_radius_vector_3d(spawn_point_candidate, self.ball_radius * 1.01) # avoid intersecting with existing points
                        if spawn_point_count > 0:
                            spawn_pcd = o3d.geometry.PointCloud()
                            spawn_pcd.points = o3d.utility.Vector3dVector(np.vstack(self.data['spawn_points']))
                            spawn_point_kdtree = o3d.geometry.KDTreeFlann(spawn_pcd)
                            k2, idx2, _ = spawn_point_kdtree.search_radius_vector_3d(spawn_point_candidate, self.ball_radius * 10) # avoid too close to existing spawn points
                            k+=k2
                        if k==0:
                            spawn_point_candidate = ((last_collision_point + collision_point) * 0.5) # Set new spawn as midpoint between last two collisions       
                            # path_distance = np.linalg.norm(last_collision_point - collision_point) 
                            # if path_distance > self.max_path_length:
                            #     self.max_path_length = path_distance
                            # if path_distance < self.min_path_length:
                            #     self.min_path_length = path_distance     

                            new_spawn_point = spawn_point_candidate
                            set_spawn = True
                        else:
                            pass
                            # print("\t>>> [filtered], bad spawn point")
                            
                if collision_count != max_steps: 
                    last_collision_point = collision_point  # Update for next collision

            elif result == 'moved':
                move_count += 1
                continue        
            elif result == 'escape':
                self.escape_count += 1
                escaped = True
                break

        if new_spawn_point is not None:
            # percentage to append spawn point
            # diff_prob = get_diffusion_probability(self.min_path_length, self.max_path_length, path_distance)
            # condition to add spawn point
            # if random.random() < diff_prob: 
            spawn_point_count = len(self.data['spawn_points'])
            if spawn_point_count >= self.max_spawn_points:
                #generate_spawn = False 
                self.data['spawn_points'] = self.data['spawn_points'][1:]
            else:
                generate_spawn = True
            self.data['spawn_points'].append(new_spawn_point)
                # print(f"---> last_collision_point: {last_collision_point}, collision_point: {collision_point}\n---> new spawn point: {new_spawn_point}")
                # print(f"---> path distance (this/min/max): {path_distance}/{self.min_path_length}/{self.max_path_length}, diff_prob: {diff_prob}")
        
        self.ball_count += 1
        self.avg_move_count = 0.9*self.avg_move_count + 0.1*move_count
        self.data['duplicate_counts'].append(duplicated_count)
        self.data['collision_count'].append(collision_count)
        this_dup_rate = float(duplicated_count / (collision_count+0.01))
        self.last_dup_rate = this_dup_rate
        self.data['dup_rate'].append(0.9*self.data['dup_rate'][-1] + 0.1 * this_dup_rate) # moving average filter on dup_rate
        
        # Check stop condition based on recent duplicate rates
        self.recent_dup_rates.append(this_dup_rate)
        if len(self.recent_dup_rates) > 10:
            self.recent_dup_rates.pop(0)  # Keep only last 10
        
        # Check if last 10 balls have high duplication rate
        if len(self.recent_dup_rates) == 10:
            avg_recent_dup_rate = sum(self.recent_dup_rates) / 10
            threshold = 1.0 - (1.0 / max_steps)  # e.g., 0.98 for max_steps=50
            if avg_recent_dup_rate > threshold:
                self.stop_flag = True
                print(f"\n*** STOP FLAG TRIGGERED ***")
                print(f"Recent 10 balls avg dup rate: {avg_recent_dup_rate:.3f} > threshold: {threshold:.3f}")
                print(f"Simulation stopped at ball {self.ball_count} out of {self.total_balls}")
        self.data['escape_history'].append(self.escape_count)
        if self.ball_count % (self.total_balls/100) == 0:
            in_points = self.point_cloud.get_collided_points(self.data['point_counts'])
            out_points = self.point_cloud.get_collided_points(self.data['point_counts'])
            self.data['inner_added'].append(len(self.point_cloud.get_collided_inpoints(in_points)))
            self.data['outer_added'].append(len(self.point_cloud.get_collided_outpoints(out_points)))

        if set_spawn:
            self.spawn_point_collision_count = 0
        else:
            self.spawn_point_collision_count+=collision_count
        
        # ___________print results_____________
        if self.ball_count % (self.total_balls/10) == 0:
            print(f"#############")
            print(f"--->Spawnpoint: {spawn_point}, spawn point len: {len(self.data['spawn_points'])}")
            print(f"---> steps: {steps}, collisions: {collision_count}, added new points: {collision_count-duplicated_count}, duplicated points: {duplicated_count}")
            print(f"---> this dup_rate: {this_dup_rate:.3f}, dup_rate: {self.data['dup_rate'][-1]:.3f}")
            print(f"---> inner added: {self.data['inner_added'][-1]}, outer added: {self.data['outer_added'][-1]}")
            print(f"#############\n")
