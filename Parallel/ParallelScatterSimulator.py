import numpy as np
import open3d as o3d
import math
import random
import time
from tqdm import tqdm
import multiprocessing as mp
from utils import *
from PLYManager import PLYManager

# Module-level globals initialized once per worker process
_kdtree = None
_points = None
_bbox_center = None
_bbox_diagonal = None
_outer_radius = None


def _init_worker(points_array, bbox_center, bbox_diagonal, outer_radius):
    global _kdtree, _points, _bbox_center, _bbox_diagonal, _outer_radius
    _points = points_array
    _bbox_center = bbox_center
    _bbox_diagonal = bbox_diagonal
    _outer_radius = outer_radius
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_points)
    _kdtree = o3d.geometry.KDTreeFlann(pcd)


def simulate_ball_worker(args):
    """
    Worker function for parallel ball simulation.
    This function runs in a separate process and simulates a single ball.
    
    Args:
        args: tuple containing simulation parameters
    
    Returns:
        dict: Results from the ball simulation
    """
    (ball_radius, collision_margin, t_max, p, origin, 
     spawn_points, max_steps, max_collisions, diffusion, ball_id, random_seed) = args
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Use prebuilt KDTree and shared arrays
    kdtree = _kdtree
    points = _points
    
    # Initialize result tracking
    last_collision_point = None
    new_spawn_point = None
    set_spawn = False
    generate_spawn = diffusion
    escaped = False
    duplicated_count = 0
    collision_count = 0
    move_count = 0
    
    # Spawn ball
    spawn_point = origin
    if np.random.rand() < p and len(spawn_points) > 0 and diffusion:
        spawn_point = random.choice(spawn_points)
    
    initial_direction = np.random.randn(3)
    initial_direction /= np.linalg.norm(initial_direction)
    ball = Ball(spawn_point, initial_direction, ball_radius, collision_margin)
    
    # Track collisions for this ball
    collision_points = []
    ball_scatter_dirs = []
    point_counts = np.zeros(len(points), dtype=np.int32)
    
    # Simulate ball movement
    for steps in range(max_steps):
        if collision_count >= max_collisions:
            break
            
        # Move until collision or escape
        r = ball.radius + ball.collision_margin
        current_pos, direction = ball.position, ball.direction
        
        query_point = current_pos.reshape(3, 1)
        radius = float(r + t_max)
        k, idx, _ = kdtree.search_radius_vector_3d(query_point, radius)
        
        if k == 0:
            new_position = current_pos + t_max * direction
            # Check if escaped (simplified check)
            if np.linalg.norm(new_position - origin) > 1000:  # Large escape radius
                escaped = True
                break
            else:
                ball.set_position(new_position)
                move_count += 1
                continue
        
        candidate_points = points[idx]
        collision_point, t_min = find_collision(current_pos, direction, candidate_points, r)
        
        if t_min < t_max:
            ball.set_position(current_pos)
            collision_count += 1
            
            # Compute normal (simplified)
            if k >= 3:
                points_norm = candidate_points
                cov = np.cov(points_norm.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]
            else:
                normal = np.random.randn(3)
                normal /= np.linalg.norm(normal)
            
            to_ball = ball.position - collision_point
            if np.dot(normal, to_ball) < 0:
                normal = -normal
            
            v_old = ball.direction
            v_new = v_old - 2 * np.dot(v_old, normal) * normal + 0.05 * np.random.randn(3)
            v_new /= np.linalg.norm(v_new)
            
            # Track collision
            collision_points.append(collision_point.copy())
            ball_scatter_dirs.append(v_new.copy())
            
            # Update ball state
            ball.set_direction(v_new)
            ball.set_position(current_pos)
            
            # Check for spawn point generation
            if (collision_count > 2 and diffusion and generate_spawn and 
                last_collision_point is not None):
                spawn_point_candidate = ((last_collision_point + collision_point) / 2.0)
                # Simple check for valid spawn point
                k_spawn, _, _ = kdtree.search_radius_vector_3d(spawn_point_candidate, ball_radius * 1.01)
                if k_spawn == 0:
                    new_spawn_point = spawn_point_candidate
                    set_spawn = True
            
            if collision_count != max_steps:
                last_collision_point = collision_point.copy()
        else:
            new_position = current_pos + t_max * direction
            ball.set_position(new_position)
            move_count += 1
    
    # Return results
    return {
        'ball_id': ball_id,
        'collision_points': collision_points,
        'ball_scatter_dirs': ball_scatter_dirs,
        'point_counts': point_counts,
        'collision_count': collision_count,
        'duplicated_count': duplicated_count,
        'move_count': move_count,
        'escaped': escaped,
        'new_spawn_point': new_spawn_point,
        'set_spawn': set_spawn
    }

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

class ParallelScatterSimulator:
    def __init__(self, point_cloud: PLYManager, 
                 ball_radius, 
                 collision_margin_ratio=0.5, 
                 p=1, num_balls=1000, 
                 max_spawn_points=200, 
                 render=False,
                 num_processes=None):
        
        self.point_cloud = point_cloud
        self.bbox_center = point_cloud.bbox_center
        self.bbox_diagonal = point_cloud.bbox_diagonal
        self.outer_radius = point_cloud.outer_radius
        self.origin = point_cloud.origin_point
    
        self.ball_radius = ball_radius
        self.t_max = min(3 * ball_radius, self.bbox_diagonal*0.05)
        self.collision_margin = min(ball_radius * collision_margin_ratio, self.bbox_diagonal*0.02)
        self.p = p  # Likelihood of spawning at new location
        self.total_balls = num_balls
        self.max_spawn_points = max_spawn_points
        self.render = render
        self.num_processes = num_processes or mp.cpu_count()

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
        self.existing_points_set = set()

    def simulate_balls_parallel(self, max_steps=100, max_collisions=10, diffusion=False, batch_size=None):
        """
        Simulate balls in parallel using multiprocessing.
        
        Args:
            max_steps: Maximum steps per ball
            max_collisions: Maximum collisions per ball
            diffusion: Whether to use diffusion mode
            batch_size: Number of balls to process in each batch (None for auto)
        """
        if batch_size is None:
            batch_size = max(1, self.total_balls // (self.num_processes * 4))
        
        print(f"Starting parallel simulation with {self.num_processes} processes, batch size: {batch_size}")
        
            # Prepare point data for worker initializer
        points_array = self.point_cloud.points
        bbox_center = self.bbox_center
        bbox_diagonal = self.bbox_diagonal
        outer_radius = self.outer_radius
        
        # Process balls in batches
        all_results = []
        processed_balls = 0
        
        with mp.Pool(processes=self.num_processes, initializer=_init_worker, initargs=(points_array, bbox_center, bbox_diagonal, outer_radius)) as pool:
            with tqdm(total=self.total_balls, desc="Simulating balls") as pbar:
                while processed_balls < self.total_balls:
                    # Determine batch size for this iteration
                    current_batch_size = min(batch_size, self.total_balls - processed_balls)
                    
                    # Prepare arguments for parallel processing
                    args_list = []
                    for i in range(current_batch_size):
                        ball_id = processed_balls + i
                        random_seed = int(time.time() * 1000000) % 2**32 + ball_id
                        
                        args = (self.ball_radius, self.collision_margin, 
                               self.t_max, self.p, self.origin, self.data['spawn_points'],
                               max_steps, max_collisions, diffusion, ball_id, random_seed)
                        args_list.append(args)
                    
                    # Choose a chunksize to reduce scheduling overhead
                    chunksize = max(1, current_batch_size // (self.num_processes * 4))
                    
                    # Process batch in parallel using persistent pool
                    batch_results = pool.map(simulate_ball_worker, args_list, chunksize)
                    
                    # Aggregate results
                    all_results.extend(batch_results)
                    self._aggregate_batch_results(batch_results)
                    
                    processed_balls += current_batch_size
                    pbar.update(current_batch_size)
                    
                    # Update progress periodically
                    if self.total_balls >= 10 and processed_balls % max(1, self.total_balls // 10) == 0:
                        self._print_progress()
        
        return all_results

    def _aggregate_batch_results(self, batch_results):
        """Aggregate results from a batch of parallel simulations."""
        for result in batch_results:
            self.ball_count += 1
            
            # Update collision data
            if result['collision_points']:
                for i, collision_point in enumerate(result['collision_points']):
                    collision_point_ind = self.point_cloud.find_point_index(collision_point)
                    self.data['point_counts'][collision_point_ind] += 1
                    
                    if self.render:
                        if not contains_vector(collision_point, self.existing_points_set):
                            self.data['points'].append(collision_point)
                            self.existing_points_set.add(tuple(collision_point.tolist()))
                            self.data['ball_scatter_dir'].append(result['ball_scatter_dirs'][i])
                        else:
                            result['duplicated_count'] += 1
                    else:
                        if self.data['point_counts'][collision_point_ind] > 1:
                            result['duplicated_count'] += 1
                        self.data['ball_scatter_dir'].append(result['ball_scatter_dirs'][i])
            
            # Update spawn points
            if result['new_spawn_point'] is not None:
                spawn_point_count = len(self.data['spawn_points'])
                if spawn_point_count >= self.max_spawn_points:
                    self.data['spawn_points'] = self.data['spawn_points'][1:]
                self.data['spawn_points'].append(result['new_spawn_point'])
            
            # Update statistics
            self.escape_count += result['escaped']
            self.avg_move_count = 0.9 * self.avg_move_count + 0.1 * result['move_count']
            self.data['duplicate_counts'].append(result['duplicated_count'])
            self.data['collision_count'].append(result['collision_count'])
            
            this_dup_rate = float(result['duplicated_count'] / (result['collision_count'] + 0.01))
            self.last_dup_rate = this_dup_rate
            self.data['dup_rate'].append(0.9 * self.data['dup_rate'][-1] + 0.1 * this_dup_rate)
            self.data['escape_history'].append(self.escape_count)

    def _print_progress(self):
        """Print simulation progress."""
        print(f"#############")
        print(f"---> Spawn points: {len(self.data['spawn_points'])}")
        print(f"---> Balls processed: {self.ball_count}, escape count: {self.escape_count}")
        print(f"---> Avg moves: {self.avg_move_count:.2f}, dup rate: {self.data['dup_rate'][-1]:.3f}")
        print(f"#############\n")

    def get_results(self):
        """Get simulation results."""
        return {
            'data': self.data,
            'ball_count': self.ball_count,
            'escape_count': self.escape_count,
            'avg_move_count': self.avg_move_count
        }
