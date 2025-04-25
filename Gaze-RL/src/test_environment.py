import os
import sys
import time
import random
import argparse

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from environments.ai2thor_gymnasium_env import AI2ThorEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Test AI2-THOR environment")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--target", type=str, default="Microwave",
                        help="Target object to search")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of steps to take")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between steps (seconds)")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Update config with target object
    env_config = config["environment"]
    env_config["target_object"] = args.target
    
    # Create environment with human rendering
    env = AI2ThorEnv(env_config, render_mode="human")
    
    # Reset environment
    obs, info = env.reset()
    print("Environment reset complete.")
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run random actions
    total_reward = 0
    success = False
    
    for i in range(args.steps):
        # Take random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Update total reward
        total_reward += reward
        
        # Print step information
        print(f"Step {i+1}/{args.steps}")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Total reward: {total_reward}")
        print(f"  Success: {info['success']}")
        print(f"  Object visible: {info['object_visible']}")
        
        if info['object_visible']:
            print(f"  Object distance: {info['object_distance']:.2f}")
            print(f"  Object visibility: {info['object_visibility']:.2f}")
        
        # Check for termination
        if terminated:
            success = info['success']
            print(f"\nEpisode terminated successfully: {success}")
            break
            
        if truncated:
            print("\nEpisode truncated (max steps reached)")
            break
        
        # Add delay
        time.sleep(args.delay)
    
    # Final results
    print("\n=== Final Results ===")
    print(f"Total steps: {i+1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Success: {success}")
    print(f"Explored {info['visited_positions']} positions")
    
    # Close environment
    env.close()
    print("\nEnvironment closed.")

if __name__ == "__main__":
    main()