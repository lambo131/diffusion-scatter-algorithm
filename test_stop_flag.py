#!/usr/bin/env python3
"""
Test script for the stop flag functionality in SimulationEngine.py
"""

import numpy as np
import time
from PLYManager import PLYManager
from SimulationEngine import ScatterSimulator

def test_stop_flag():
    """Test the stop flag with different configurations"""
    
    # Test configuration
    config = {
        'ply_file': './ply_files/test ply inputs/ball',
        'origin_point': [0, 0, 0],
        'ball_radius_factor': 4,
        'num_balls': 100000,  # Large number to test early termination
        'max_steps': 100,    # Threshold will be 1 - 1/50 = 0.98
        'max_collisions': 5,
        'diffusion': True,
        'render': False,
    }
    
    print("="*80)
    print("TESTING STOP FLAG FUNCTIONALITY")
    print("="*80)
    
    try:
        # Load point cloud
        print("Loading point cloud...")
        point_cloud = PLYManager(config['ply_file'], origin_point=config['origin_point'])
        avg_separation = point_cloud.get_average_separation(0.5)
        ball_radius = config['ball_radius_factor'] * avg_separation
        
        print(f"Point cloud: {len(point_cloud.points)} points")
        print(f"Ball radius: {ball_radius:.6f}")
        print(f"Max steps: {config['max_steps']}")
        print(f"Stop threshold: {1.0 - (1.0/config['max_steps']):.3f}")
        
        # Create simulator
        simulator = ScatterSimulator(
            point_cloud=point_cloud,
            ball_radius=ball_radius,
            p=0.999,
            num_balls=config['num_balls'],
            render=config['render']
        )
        
        # Add initial spawn point
        simulator.data['points'].append(simulator.origin)
        simulator.data['ball_scatter_dir'].append(np.array([0, 0, 0]))
        
        # Run simulation with stop flag monitoring
        print(f"\nRunning simulation with stop flag monitoring...")
        print(f"Target: {config['num_balls']} balls")
        start_time = time.time()
        
        for i in range(config['num_balls']):
            simulator.simulate_ball(
                max_steps=config['max_steps'],
                max_collisions=config['max_collisions'],
                diffusion=config['diffusion']
            )
            
            # Check stop flag after each ball
            if simulator.stop_flag:
                print(f"\n*** STOP FLAG TRIGGERED ***")
                print(f"Simulation stopped at ball {simulator.ball_count}")
                break
            
            # Print progress every 100 balls
            if (i + 1) % 100 == 0:
                recent_avg = sum(simulator.recent_dup_rates) / len(simulator.recent_dup_rates) if simulator.recent_dup_rates else 0
                print(f"Ball {i+1}: Recent avg dup rate: {recent_avg:.3f}, Current dup rate: {simulator.data['dup_rate'][-1]:.3f}")
        
        end_time = time.time()
        simulation_time = end_time - start_time
        
        # Print results

        # Save outputs similar to main.py
        import open3d as o3d, os, pickle as pkl
        os.makedirs('./ply_files/output/', exist_ok=True)
        sim_name = 'stopflag_test'

        # Collisions via point_counts
        point_counts = simulator.data['point_counts']
        collision_points = point_cloud.get_collided_points(point_counts)

        # Inner/outer separation (if available)
        inner_points = point_cloud.get_collided_inpoints(collision_points)
        outer_points = point_cloud.get_collided_outpoints(collision_points)

        # Save inner points
        if len(inner_points) > 0:
            inner_pcd = o3d.geometry.PointCloud()
            inner_pcd.points = o3d.utility.Vector3dVector(inner_points)
            o3d.io.write_point_cloud(f'./ply_files/output/output_{sim_name}.ply', inner_pcd)
        
        # Save all collisions
        all_collision_pcd = o3d.geometry.PointCloud()
        all_collision_pcd.points = o3d.utility.Vector3dVector(collision_points)
        o3d.io.write_point_cloud(f'./ply_files/output/all_collisions_{sim_name}.ply', all_collision_pcd)

        # Save spawn points
        if len(simulator.data['spawn_points']) > 0:
            spawn_points = np.array(simulator.data['spawn_points'])
            spawn_pcd = o3d.geometry.PointCloud()
            spawn_pcd.points = o3d.utility.Vector3dVector(spawn_points)
            o3d.io.write_point_cloud(f'./ply_files/output/spawn_points_{sim_name}.ply', spawn_pcd)

        # Save simulation data
        os.makedirs('./ply_files/sim_data/', exist_ok=True)
        simulation_data = {
            'config': config,
            'final_data': {
                'data': simulator.data,
                'ball_count': simulator.ball_count,
                'escape_count': simulator.escape_count,
                'avg_move_count': simulator.avg_move_count
            },
            'inner_points': inner_points,
            'outer_points': outer_points,
            'all_collision_points': collision_points,
            'spawn_points': simulator.data['spawn_points'],
            'simulation_time': simulation_time,
        }
        with open(f'./ply_files/sim_data/sim_data_{sim_name}.pkl', 'wb') as f:
            pkl.dump(simulation_data, f)
        print(f"\n" + "="*80)
        print("SIMULATION RESULTS")
        print("="*80)
        print(f"Simulation time: {simulation_time:.2f} seconds")
        print(f"Balls processed: {simulator.ball_count}")
        print(f"Stop flag triggered: {simulator.stop_flag}")
        print(f"Final duplicate rate: {simulator.data['dup_rate'][-1]:.3f}")
        print(f"Total collision points: {len(simulator.data['points'])}")
        print(f"Escape count: {simulator.escape_count}")
        
        if simulator.recent_dup_rates:
            print(f"Last 10 duplicate rates: {[f'{r:.3f}' for r in simulator.recent_dup_rates]}")
            print(f"Average of last 10: {sum(simulator.recent_dup_rates)/len(simulator.recent_dup_rates):.3f}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("   Please update the config['ply_file'] path to point to a valid PLY file.")
        return False
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        return False

if __name__ == "__main__":
    success = test_stop_flag()
    if success:
        print("\n✅ Stop flag test completed!")
    else:
        print("\n❌ Stop flag test failed!")
