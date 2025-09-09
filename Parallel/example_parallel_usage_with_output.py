#!/usr/bin/env python3
"""
Example: How to use parallel scatter simulation with output saving

This example shows how to replace the serial simulation with parallel processing
and save the results to PLY files.
"""

import numpy as np
import open3d as o3d
import time
import os
import pickle as pkl
from PLYManager import PLYManager
from ParallelScatterSimulator import ParallelScatterSimulator

def main():
    # Your existing configuration
    config = {
        'ply_file': './ply_files/test ply inputs/ball',  # Update this path
        'origin_point': [0, 0, 0],
        'ball_radius_factor': 4,
        'num_balls': 50000,  # Number of balls to simulate
        'max_steps': 50,
        'max_collisions': 5,
        'diffusion': True,
        'render': False,
        'num_processes': 8,  # Number of CPU cores to use
    }
    
    # Output configuration
    output_config = {
        'output_dir': './ply_files/output/',
        'simulation_name': 'parallel_test_ball',
        'save_simulation_data': True
    }
    
    print("="*80)
    print("PARALLEL SCATTER SIMULATION WITH OUTPUT")
    print("="*80)
    
    print("Loading point cloud...")
    point_cloud = PLYManager(config['ply_file'], origin_point=config['origin_point'])
    avg_separation = point_cloud.get_average_separation(0.5)
    ball_radius = config['ball_radius_factor'] * avg_separation
    
    print(f"Point cloud: {len(point_cloud.points)} points")
    print(f"Ball radius: {ball_radius:.6f}")
    
    # Create parallel simulator
    simulator = ParallelScatterSimulator(
        point_cloud=point_cloud,
        ball_radius=ball_radius,
        p=0.999,  # Probability of spawning at new location
        num_balls=config['num_balls'],
        render=config['render'],
        num_processes=config['num_processes']
    )
    
    # Add initial spawn point
    simulator.data['points'].append(simulator.origin)
    simulator.data['ball_scatter_dir'].append(np.array([0, 0, 0]))
    
    # Run parallel simulation
    print(f"Running parallel simulation with {config['num_processes']} processes...")
    start_time = time.time()
    
    results = simulator.simulate_balls_parallel(
        max_steps=config['max_steps'],
        max_collisions=config['max_collisions'],
        diffusion=config['diffusion']
    )
    
    end_time = time.time()
    simulation_time = end_time - start_time
    
    # Get results
    final_data = simulator.get_results()
    
    print(f"\nSimulation completed in {simulation_time:.2f} seconds")
    print(f"Total collision points: {len(final_data['data']['points'])}")
    print(f"Escape count: {final_data['escape_count']}")
    print(f"Average moves per ball: {final_data['avg_move_count']:.2f}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_config['output_dir'], exist_ok=True)
    
    # Get collision points from point_counts (works even when render=False)
    point_counts = simulator.data['point_counts']
    collision_points = point_cloud.get_collided_points(point_counts)
    print(f"Collision points shape: {collision_points.shape}")
    
    if len(collision_points) > 0:
        # Separate inner and outer points
        inner_points = point_cloud.get_collided_inpoints(collision_points)
        outer_points = point_cloud.get_collided_outpoints(collision_points)
        
        print(f"Inner points detected: {len(inner_points)}")
        print(f"Outer points detected: {len(outer_points)}")
        
        # Save inner points (hole detection results)
        if len(inner_points) > 0:
            inner_pcd = o3d.geometry.PointCloud()
            inner_pcd.points = o3d.utility.Vector3dVector(inner_points)
            inner_output_file = f"{output_config['output_dir']}output_{output_config['simulation_name']}.ply"
            o3d.io.write_point_cloud(inner_output_file, inner_pcd)
            print(f"✅ Inner points saved to: {inner_output_file}")
        
        # Save all collision points
        all_collision_pcd = o3d.geometry.PointCloud()
        all_collision_pcd.points = o3d.utility.Vector3dVector(collision_points)
        all_output_file = f"{output_config['output_dir']}all_collisions_{output_config['simulation_name']}.ply"
        o3d.io.write_point_cloud(all_output_file, all_collision_pcd)
        print(f"✅ All collision points saved to: {all_output_file}")
        
        # Save spawn points
        if len(simulator.data['spawn_points']) > 0:
            spawn_points = np.array(simulator.data['spawn_points'])
            spawn_pcd = o3d.geometry.PointCloud()
            spawn_pcd.points = o3d.utility.Vector3dVector(spawn_points)
            spawn_output_file = f"{output_config['output_dir']}spawn_points_{output_config['simulation_name']}.ply"
            o3d.io.write_point_cloud(spawn_output_file, spawn_pcd)
            print(f"✅ Spawn points saved to: {spawn_output_file}")
        
        # Save simulation data
        if output_config['save_simulation_data']:
            sim_data_file = f"./ply_files/sim_data/sim_data_{output_config['simulation_name']}.pkl"
            os.makedirs(os.path.dirname(sim_data_file), exist_ok=True)
            
            simulation_data = {
                'config': config,
                'results': results,
                'final_data': final_data,
                'inner_points': inner_points,
                'outer_points': outer_points,
                'all_collision_points': collision_points,
                'spawn_points': simulator.data['spawn_points'],
                'simulation_time': simulation_time,
                'balls_per_second': config['num_balls'] / simulation_time
            }
            
            with open(sim_data_file, 'wb') as f:
                pkl.dump(simulation_data, f)
            print(f"✅ Simulation data saved to: {sim_data_file}")
        
        # Print performance metrics
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        print(f"Simulation time: {simulation_time:.2f} seconds")
        print(f"Balls per second: {config['num_balls'] / simulation_time:.2f}")
        print(f"Total balls: {config['num_balls']}")
        print(f"Collision points: {len(collision_points)}")
        print(f"Inner points: {len(inner_points)}")
        print(f"Outer points: {len(outer_points)}")
        print(f"Escape count: {final_data['escape_count']}")
        print(f"Average moves per ball: {final_data['avg_move_count']:.2f}")
        print(f"Final duplicate rate: {final_data['data']['dup_rate'][-1]:.3f}")
        
    else:
        print("❌ No collision points generated!")
    
    print("\n" + "="*80)
    print("✅ SIMULATION COMPLETED SUCCESSFULLY!")
    print("✅ Check the output files in the ply_files/output/ directory")
    print("="*80)
    
    return final_data

if __name__ == "__main__":
    result = main()
