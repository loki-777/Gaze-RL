#!/usr/bin/env python
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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pytorch_lightning as pl

# Import your environment and wrappers
from environments.ai2thor_gymnasium_env import AI2ThorEnv
from env_wrappers import FlattenObservationWrapper, GazeEnvWrapper, GazePreprocessEnvWrapper
from src.models.agents import GazePPO
from src.models.lightning_module import GazeLightningModule

# Custom callback to print training progress and track metrics
class GazeProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
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
                success = 1 if info.get("success", False) else 0
                
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
                      f"Success: {'✓' if success else '✗'} | "
                      f"Elapsed: {elapsed_str}")
                
                # Store stats
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_rates.append(success)
                
                # Print additional stats every 5 episodes
                if len(self.episode_rewards) % 5 == 0:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    success_rate = np.mean(self.success_rates[-5:]) * 100
                    print(f"[{timestamp}] Last 5 episodes: "
                          f"Mean reward = {mean_reward:.2f} | "
                          f"Mean length = {mean_length:.1f} steps | "
                          f"Success rate = {success_rate:.1f}%")
                
                # Print overall stats every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    mean_reward = np.mean(self.episode_rewards)
                    mean_length = np.mean(self.episode_lengths)
                    overall_success_rate = np.mean(self.success_rates) * 100
                    print(f"\n=== OVERALL STATISTICS (AFTER {len(self.episode_rewards)} EPISODES) ===")
                    print(f"Mean reward: {mean_reward:.2f}")
                    print(f"Mean episode length: {mean_length:.1f} steps")
                    print(f"Overall success rate: {overall_success_rate:.1f}%")
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
    parser = argparse.ArgumentParser(description="Train Gaze-Guided RL agent for object search")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--gaze_config", type=str, default="configs/gaze_config.yaml",
                        help="Path to gaze model config file")
    parser.add_argument("--gaze_checkpoint", type=str, required=True,
                        help="Path to pretrained gaze model checkpoint (.ckpt file)")
    parser.add_argument("--target", type=str, default="Microwave",
                        help="Target object to search")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total timesteps for training")
    # Add experiment name parameter
    parser.add_argument("--exp_name", type=str, default="gaze_guided",
                        help="Experiment name for logging")
    # Add comparison mode
    parser.add_argument("--comparison", action="store_true",
                        help="Run comparison experiment with and without gaze guidance")
    return parser.parse_args()

def load_gaze_model(checkpoint_path):
    """Load pretrained gaze prediction model from checkpoint"""
    print(f"Loading gaze model from {checkpoint_path}")
    try:
        # Load the checkpoint using PyTorch Lightning
        gaze_model = GazeLightningModule.load_from_checkpoint(checkpoint_path)
        gaze_model.eval()  # Set to evaluation mode
        
        # Move to appropriate device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        gaze_model = gaze_model.to(device)
        print(f"Gaze model loaded successfully and moved to {device}")
        return gaze_model
    except Exception as e:
        print(f"Error loading gaze model: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def create_env(config, target_object, gaze_model=None, use_gaze=False):
    """Create environment with proper settings and gaze integration"""
    
    # Update config for optimization
    env_config = config["environment"].copy()
    env_config["target_object"] = target_object
    env_config["width"] = 224
    env_config["height"] = 224
    env_config["grid_size"] = 0.25
    # Add Mac-friendly settings
    env_config["quality"] = "Medium"
    env_config["shadows"] = False
    
    def _init():
        # Create base environment
        env = AI2ThorEnv(
            env_config,
            render_mode=None  # No rendering during training for performance
        )
        
        # Add gaze wrapper if using gaze guidance
        if use_gaze and gaze_model is not None:
            env = GazeEnvWrapper(env, gaze_predictor=gaze_model)
            env = GazePreprocessEnvWrapper(env)
        
        # Make observations compatible with SB3
        env = FlattenObservationWrapper(env)
        
        # Wrap with Monitor to track episode rewards
        monitor_env = Monitor(env)
        
        # Print environment configuration
        print(f"Created environment for target object: {target_object}")
        print(f"Using gaze guidance: {use_gaze}")
        
        return monitor_env
    
    return _init

def train_agent(config, env, gaze_model=None, use_gaze=False, total_timesteps=50000, log_dir="logs", exp_name="gaze_guided"):
    """Train an agent with or without gaze guidance"""
    
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create unique experiment folder name with timestamp
    suffix = "with_gaze" if use_gaze else "no_gaze"
    experiment_id = f"{exp_name}_{suffix}_{timestamp}"
    
    # Create experiment-specific directories
    experiment_dir = os.path.join(log_dir, experiment_id)
    model_dir = os.path.join(experiment_dir, "models")
    tensorboard_dir = os.path.join(experiment_dir, "tensorboard")
    results_dir = os.path.join(experiment_dir, "results")
    
    # Create all directories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(experiment_dir, "training.log")
    
    # Create a log file handler
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
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
        
    logger.info("\n" + "="*60)
    logger.info(f"Experiment: {exp_name} ({'with' if use_gaze else 'without'} gaze)")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Starting training run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {device} ({device_name})")
    logger.info(f"Target object: {config['environment']['target_object']}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info("="*60 + "\n")
    
    try:
        # Save the configuration for reference
        config_path = os.path.join(experiment_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Set device for PyTorch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        # Create progress callback
        progress_callback = GazeProgressCallback()
        
        # Create a GazePPO model with appropriate configuration
        logger.info("Creating PPO model...")
        
        # Update model config with gaze settings
        model_config = config["model"].copy()
        model_config["use_gaze"] = use_gaze
        
        model = GazePPO(
            env=env,
            config=model_config,
            policy="CnnPolicy",
            device=device,
            tensorboard_log=tensorboard_dir,
        )
        
        # Print when training starts
        logger.info("\nStarting training...")
        logger.info(f"Experiment: {exp_name} ({'with' if use_gaze else 'without'} gaze)")
        logger.info(f"Target: {config['environment']['target_object']}")
        logger.info(f"Timesteps: {total_timesteps}")
        logger.info(f"Using device: {device}\n")
        
        # Train model with progress callback
        train_start_time = time.time()
        model.learn(
            total_timesteps=total_timesteps,
            callback=progress_callback
        )
        train_end_time = time.time()
        
        # Print training summary
        train_duration = train_end_time - train_start_time
        train_hours = int(train_duration // 3600)
        train_minutes = int((train_duration % 3600) // 60)
        train_seconds = int(train_duration % 60)
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED!")
        logger.info(f"Experiment: {exp_name} ({'with' if use_gaze else 'without'} gaze)")
        logger.info(f"Total training time: {train_hours:02d}:{train_minutes:02d}:{train_seconds:02d}")
        logger.info(f"Episodes completed: {len(progress_callback.episode_rewards)}")
        logger.info(f"Final mean reward: {np.mean(progress_callback.episode_rewards[-10:]):.2f}")
        logger.info(f"Overall success rate: {np.mean(progress_callback.success_rates) * 100:.1f}%")
        logger.info("="*60 + "\n")
        
        # Save the model
        model_path = os.path.join(model_dir, "ppo_ai2thor_final")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save performance metrics
        metrics = {
            "experiment": exp_name,
            "target_object": config['environment']['target_object'],
            "use_gaze": use_gaze,
            "timestamp": timestamp,
            "total_timesteps": total_timesteps,
            "episodes_completed": len(progress_callback.episode_rewards),
            "episode_rewards": progress_callback.episode_rewards,
            "episode_lengths": progress_callback.episode_lengths,
            "success_rates": progress_callback.success_rates,
            "mean_reward": float(np.mean(progress_callback.episode_rewards)),
            "mean_last_10_reward": float(np.mean(progress_callback.episode_rewards[-10:])),
            "mean_episode_length": float(np.mean(progress_callback.episode_lengths)),
            "overall_success_rate": float(np.mean(progress_callback.success_rates) * 100),
            "training_duration_seconds": train_duration,
        }
        
        # Save metrics to file
        metrics_path = os.path.join(results_dir, "metrics.yaml")
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f)
        
        # Success stories
        if progress_callback.episode_rewards:
            best_reward = max(progress_callback.episode_rewards)
            best_idx = progress_callback.episode_rewards.index(best_reward)
            best_length = progress_callback.episode_lengths[best_idx]
            
            logger.info(f"Best episode: #{best_idx+1}")
            logger.info(f"  Reward: {best_reward:.2f}")
            logger.info(f"  Length: {best_length} steps")
            logger.info(f"  Success: {'Yes' if progress_callback.success_rates[best_idx] else 'No'}")
        
        # Return metrics for comparison
        return metrics
    
    except Exception as e:
        logger.error(f"\nERROR during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Clean up
        try:
            if 'env' in locals():
                env.close()
                logger.info("Environment closed")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

def main():
    """Main function for training gaze-guided RL agents"""
    # Parse arguments
    args = parse_args()
    
    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    with open(args.gaze_config, "r") as f:
        gaze_config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config["environment"]["target_object"] = args.target
    
    # Load pretrained gaze model
    gaze_model = load_gaze_model(args.gaze_checkpoint)
    
    if args.comparison:
        # Run both with and without gaze guidance for comparison
        print("\n" + "="*60)
        print("RUNNING COMPARISON EXPERIMENT")
        print("Training without gaze guidance first...")
        print("="*60 + "\n")
        
        # Create environment without gaze
        env_fn_no_gaze = create_env(config, args.target, gaze_model=None, use_gaze=False)
        env_no_gaze = DummyVecEnv([env_fn_no_gaze])
        
        # Train without gaze
        metrics_no_gaze = train_agent(
            config, 
            env_no_gaze, 
            gaze_model=None, 
            use_gaze=False,
            total_timesteps=args.timesteps,
            log_dir=args.log_dir,
            exp_name=args.exp_name
        )
        
        # Close environment
        env_no_gaze.close()
        
        print("\n" + "="*60)
        print("Training with gaze guidance next...")
        print("="*60 + "\n")
        
        # Create environment with gaze
        env_fn_with_gaze = create_env(config, args.target, gaze_model=gaze_model, use_gaze=True)
        env_with_gaze = DummyVecEnv([env_fn_with_gaze])
        
        # Train with gaze
        metrics_with_gaze = train_agent(
            config, 
            env_with_gaze, 
            gaze_model=gaze_model, 
            use_gaze=True,
            total_timesteps=args.timesteps,
            log_dir=args.log_dir,
            exp_name=args.exp_name
        )
        
        # Close environment
        env_with_gaze.close()
        
        # Print comparison results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        if metrics_no_gaze and metrics_with_gaze:
            print(f"Mean reward without gaze: {metrics_no_gaze['mean_reward']:.2f}")
            print(f"Mean reward with gaze: {metrics_with_gaze['mean_reward']:.2f}")
            print(f"Success rate without gaze: {metrics_no_gaze['overall_success_rate']:.1f}%")
            print(f"Success rate with gaze: {metrics_with_gaze['overall_success_rate']:.1f}%")
            print(f"Mean episode length without gaze: {metrics_no_gaze['mean_episode_length']:.1f} steps")
            print(f"Mean episode length with gaze: {metrics_with_gaze['mean_episode_length']:.1f} steps")
            
            # Calculate improvements
            reward_improvement = ((metrics_with_gaze['mean_reward'] - metrics_no_gaze['mean_reward']) / 
                                 abs(metrics_no_gaze['mean_reward']) * 100)
            success_improvement = (metrics_with_gaze['overall_success_rate'] - 
                                  metrics_no_gaze['overall_success_rate'])
            length_improvement = ((metrics_no_gaze['mean_episode_length'] - metrics_with_gaze['mean_episode_length']) /
                                 metrics_no_gaze['mean_episode_length'] * 100)
            
            print("\nIMPROVEMENTS WITH GAZE:")
            print(f"Reward: {reward_improvement:.1f}% improvement")
            print(f"Success rate: {success_improvement:.1f} percentage points improvement")
            print(f"Episode length: {length_improvement:.1f}% reduction (shorter is better)")
        
    else:
        # Just run with gaze guidance
        print("\n" + "="*60)
        print("TRAINING WITH GAZE GUIDANCE")
        print("="*60 + "\n")
        
        # Create environment with gaze
        env_fn = create_env(config, args.target, gaze_model=gaze_model, use_gaze=True)
        env = DummyVecEnv([env_fn])
        
        # Train with gaze
        train_agent(
            config, 
            env, 
            gaze_model=gaze_model, 
            use_gaze=True,
            total_timesteps=args.timesteps,
            log_dir=args.log_dir,
            exp_name=args.exp_name
        )
        
        # Close environment
        env.close()

if __name__ == "__main__":
    main()

# Example usage:
# python train_gaze_guided_rl.py --exp_name gaze_guided_search --target Microwave --timesteps 50000 --gaze_checkpoint logs/epoch=19-step=6260.ckpt
# For comparison experiment:
# python train_gaze_guided_rl.py --exp_name comparison_experiment --target Microwave --timesteps 50000 --gaze_checkpoint checkpoints/gaze_model/best_model.ckpt --comparison