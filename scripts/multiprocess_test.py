import multiprocessing as mp
import numpy as np
import open3d as o3d
import time

class SharedPointCloud:
    def __init__(self, num_points):
        self.shared_array = mp.Array('d', num_points * 3)
        self.lock = mp.Lock()
        self.exit_event = mp.Event()
        self.ready_event = mp.Event()
        
    def update_points(self, new_points):
        """Thread-safe point cloud update"""
        with self.lock:
            # Create fresh view of shared memory
            dest = np.frombuffer(self.shared_array.get_obj(), 
                               dtype=np.float64).reshape(-1, 3)
            np.copyto(dest, new_points)
            self.ready_event.set()

def visualizer_process(shared_data):
    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Initial dummy points
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    vis.add_geometry(pcd)
    
    reset_view = True
    
    while not shared_data.exit_event.is_set():
        if shared_data.ready_event.wait(timeout=0.1):
            with shared_data.lock:
                # Get fresh view of shared memory
                points = np.frombuffer(shared_data.shared_array.get_obj(),
                                     dtype=np.float64).reshape(-1, 3)
                pcd.points = o3d.utility.Vector3dVector(points)
                
                if reset_view:
                    vis.reset_view_point(True)
                    reset_view = False
                    
                vis.update_geometry(pcd)
                print(f"child: {np.sum(points)}")
                shared_data.ready_event.clear()
        
        vis.poll_events()
        vis.update_renderer()
    
    vis.destroy_window()

def main():
    num_points = 1000
    shared_data = SharedPointCloud(num_points+8)
    
    proc = mp.Process(target=visualizer_process, args=(shared_data,))
    proc.start()
    
    try:
        for i in range(1000):
            # Generate new point cloud data (e.g., rotating cube)
            theta = i * 0.1
            points = np.random.rand(num_points, 3) * 1 + np.array([
                np.cos(theta), np.sin(theta), np.sin(theta)
            ])
            # add 8 anchor points to points
            points = np.append(points, np.array([
                [-2, -2, -2],
                [-2, 2, -2],
                [2, -2, -2],
                [2,  2, -2],
                [-2, -2, 2],
                [-2, 2, 2],
                [2, -2, 2],
                [2,  2, 2],
            ]), axis=0)

            shared_data.update_points(points)
            print(f"parent: {np.sum(points)}")
            time.sleep(0.05)
    finally:
        shared_data.exit_event.set()
        proc.join(timeout=1.0)

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Critical for Windows
    main()