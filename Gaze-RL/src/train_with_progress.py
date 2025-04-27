import os
import sys
import time
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment
from environments.ai2thor_gymnasium_env import AI2ThorEnv

# Custom callback to print training progress
class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_steps = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        
    def _on_step(self):
        # Get rewards from the most recent step
        for info in self.locals["infos"]:
            # Track episode reward
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                
                # Calculate elapsed time
                elapsed_time = time.time() - self.start_time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                
                # Get current timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Print detailed progress
                print(f"[{timestamp}] Episode {len(self.episode_rewards) + 1} | "
                      f"Steps: {self.num_timesteps}/{self.locals['total_timesteps']} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Length: {episode_length} steps | "
                      f"Elapsed: {elapsed_str}")
                
                # Store stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Print additional stats every 5 episodes
                if len(self.episode_rewards) % 5 == 0:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    print(f"[{timestamp}] Last 5 episodes: "
                          f"Mean reward = {mean_reward:.2f} | "
                          f"Mean length = {mean_length:.1f} steps")
                
                # Print overall stats every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards)
                    mean_length = np.mean(self.episode_lengths)
                    print(f"\n=== OVERALL STATISTICS (AFTER {len(self.episode_rewards)} EPISODES) ===")
                    print(f"Mean reward: {mean_reward:.2f}")
                    print(f"Mean episode length: {mean_length:.1f} steps")
                    print(f"Progress: {self.num_timesteps}/{self.locals['total_timesteps']} steps "
                          f"({100 * self.num_timesteps / self.locals['total_timesteps']:.1f}%)")
                    print(f"Elapsed time: {elapsed_str}")
                    print("=" * 60 + "\n")
        
        # Also print progress based on timesteps
        if self.num_timesteps % 1000 == 0:
            current_time = time.time()
            time_diff = current_time - self.last_print_time
            if time_diff > 5:  # Only print if more than 5 seconds passed
                self.last_print_time = current_time
                elapsed_time = current_time - self.start_time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                steps_per_second = 1000 / time_diff if time_diff > 0 else 0
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Progress: {self.num_timesteps}/{self.locals['total_timesteps']} steps "
                      f"({100 * self.num_timesteps / self.locals['total_timesteps']:.1f}%) | "
                      f"Speed: {steps_per_second:.1f} steps/s | "
                      f"Elapsed: {elapsed_str}")
        
        return True

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent with progress tracking")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--target", type=str, default="Microwave",
                        help="Target object to search")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total timesteps for training")
    return parser.parse_args()

def create_env(config, target_object):
    """Create environment with proper settings for Mac"""
    
    # Update config for Mac optimization
    env_config = config["environment"].copy()
    env_config["target_object"] = target_object
    env_config["width"] = 224
    env_config["height"] = 224
    env_config["grid_size"] = 0.25
    # Add Mac-friendly settings
    env_config["quality"] = "Medium"
    env_config["shadows"] = False
    
    def _init():
        # Create environment
        env = AI2ThorEnv(
            env_config,
            render_mode=None  # No rendering during training for performance
        )
        
        # Wrap with Monitor to track episode rewards
        monitor_env = Monitor(env)
        
        # Print when a new environment is created
        print(f"Created environment for target object: {target_object}")
        
        return monitor_env
    
    return _init

def main():
    """Training script with comprehensive progress tracking"""
    # Parse arguments
    args = parse_args()
    
    # Print system info
    if torch.cuda.is_available():
        device = "CUDA GPU"
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "MPS (Apple GPU)"
        device_name = "Apple Silicon"
    else:
        device = "CPU"
        device_name = "CPU"
        
    print("\n" + "="*60)
    print(f"Starting training run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device} ({device_name})")
    print(f"Target object: {args.target}")
    print(f"Total timesteps: {args.timesteps}")
    print("="*60 + "\n")
    
    try:
        # Load config
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Create log directory
        os.makedirs(args.log_dir, exist_ok=True)
        
        print("Creating environment...")
        # Create environment function
        env_fn = create_env(config, args.target)
        
        # Create vectorized environment
        print("Initializing vectorized environment...")
        env = DummyVecEnv([env_fn])
        
        # Set device for PyTorch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        # Create progress callback
        progress_callback = ProgressCallback()
        
        # Create a PPO model
        print("Creating PPO model...")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=0,  # Set to 0 to use our custom progress reporting
            device=device,
            tensorboard_log=os.path.join(args.log_dir, "tensorboard"),
            learning_rate=1e-4,
            n_steps=128,
            batch_size=32,  # Reduced for Mac GPU
            n_epochs=4
        )
        
        # Print when training starts
        print("\nStarting training...")
        print(f"Target: {args.target}")
        print(f"Timesteps: {args.timesteps}")
        print(f"Using device: {device}\n")
        
        # Train model with progress callback
        train_start_time = time.time()
        model.learn(
            total_timesteps=args.timesteps,
            callback=progress_callback
        )
        train_end_time = time.time()
        
        # Print training summary
        train_duration = train_end_time - train_start_time
        train_hours = int(train_duration // 3600)
        train_minutes = int((train_duration % 3600) // 60)
        train_seconds = int(train_duration % 60)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED!")
        print(f"Total training time: {train_hours:02d}:{train_minutes:02d}:{train_seconds:02d}")
        print(f"Episodes completed: {len(progress_callback.episode_rewards)}")
        print(f"Final mean reward: {np.mean(progress_callback.episode_rewards[-10:]):.2f}")
        print("="*60 + "\n")
        
        # Save the model
        model_path = os.path.join(args.log_dir, "ppo_ai2thor")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Success stories
        if progress_callback.episode_rewards:
            best_reward = max(progress_callback.episode_rewards)
            best_idx = progress_callback.episode_rewards.index(best_reward)
            best_length = progress_callback.episode_lengths[best_idx]
            
            print(f"Best episode: #{best_idx+1}")
            print(f"  Reward: {best_reward:.2f}")
            print(f"  Length: {best_length} steps")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            if 'env' in locals():
                env.close()
                print("Environment closed")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

if __name__ == "__main__":
    main()

# python src/train_with_progress.py --target Microwave --timesteps 50000  