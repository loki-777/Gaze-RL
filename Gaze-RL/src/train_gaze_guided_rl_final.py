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
from env_wrappers import FlattenObservationWrapper, GazeEnvWrapper, GazePreprocessEnvWrapper, VideoRecorderWrapper
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
    # Gaze Integration Variations
    parser.add_argument("--gaze_integration", type=str, default="channel",
                        choices=["channel", "bottleneck", "weighted"],
                        help="Method to integrate gaze information")
    parser.add_argument("--record_freq", type=int, default=20,
                    help="Record every N-th episode (default: 1000)")
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

def create_env(config, target_object, gaze_model=None, video_dir=None, record_freq=20):
    """Create environment with gaze integration"""
    
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
        
        # Print initial observation space shape
        print(f"Base environment observation space: {env.observation_space.shape}")
        
        # Add gaze wrapper
        env = GazeEnvWrapper(env, gaze_predictor=gaze_model)
        
        # Add gaze to observation space
        env = GazePreprocessEnvWrapper(env)
        
        # Verify the observation shape
        print(f"After GazePreprocessEnvWrapper: {env.observation_space.shape}")
        
        # Make observations compatible with SB3
        env = FlattenObservationWrapper(env)
        
        # Print final observation shape
        print(f"After FlattenObservationWrapper: {env.observation_space.shape}")
        
        # Wrap with Monitor to track episode rewards
        monitor_env = Monitor(env)

        # Add video recording as the final wrapper if video_dir is provided
        if video_dir is not None:
            monitor_env = VideoRecorderWrapper(monitor_env, video_dir=video_dir, record_freq=record_freq)
            print(f"Video recording enabled: recording every {record_freq} episodes to {video_dir}")
        
        # Print environment configuration
        print(f"Created environment for target object: {target_object}")
        
        return monitor_env
    
    return _init

def train_agent(config, env, gaze_model, total_timesteps=50000, log_dir="logs", exp_name="gaze_guided", gaze_integration="ChannelCNN"):
    """Train an agent with gaze guidance"""
    
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create unique experiment folder name with timestamp
    experiment_id = f"{exp_name}_{gaze_integration}_{timestamp}"
    
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
    logger.info(f"Experiment: {exp_name} (with gaze, {gaze_integration} integration)")
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
        
        # Create a GazePPO model
        logger.info("Creating PPO model...")
        
        # Update model config
        model_config = config["model"].copy()
        model_config["use_gaze"] = True
        model_config["integration_method"] = gaze_integration
        
        model = GazePPO(
            env=env,
            config=model_config,
            policy="CnnPolicy",
            device=device,
            tensorboard_log=tensorboard_dir,
        )
        
        # Print when training starts
        logger.info("\nStarting training...")
        logger.info(f"Experiment: {exp_name} (with gaze, {gaze_integration} integration)")
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
        logger.info(f"Experiment: {exp_name} (with gaze, {gaze_integration} integration)")
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
            "use_gaze": True,
            "gaze_integration": gaze_integration,
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
    
    # Map integration method to feature extractor class name
    integration_to_extractor = {
        "channel": "ChannelCNN",
        "bottleneck": "GazeAttnCNN",
        "weighted": "WeightedCNN"
    }
    
    # Get integration method from command line arg
    gaze_integration = args.gaze_integration
    
    # Update config with the correct feature extractor
    config["model"]["features_extractor"] = integration_to_extractor[gaze_integration]
    config["model"]["use_gaze"] = True
    
    # Get gaze checkpoint path - command line arg overrides config
    gaze_checkpoint_path = args.gaze_checkpoint if args.gaze_checkpoint else config["gaze"]["model_path"]
    
    # Load pretrained gaze model
    gaze_model = load_gaze_model(gaze_checkpoint_path)
    
    print("\n" + "="*60)
    print(f"TRAINING WITH GAZE GUIDANCE ({gaze_integration} integration)")
    print(f"Using feature extractor: {config['model']['features_extractor']}")
    print("="*60 + "\n")
    
    # Create video directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = os.path.join("videos", f"{args.exp_name}_{gaze_integration}_{timestamp}")
    os.makedirs(video_dir, exist_ok=True)
    
    # Create environment with gaze
    env_fn = create_env(config, args.target, gaze_model=gaze_model, video_dir=video_dir, record_freq=args.record_freq)
    env = DummyVecEnv([env_fn])
    
    # Train with gaze
    train_agent(
        config, 
        env, 
        gaze_model=gaze_model,
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        exp_name=args.exp_name,
        gaze_integration=gaze_integration
    )
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()

# Example usage:
# python src/train_gaze_guided_rl_final.py --exp_name gaze_exp --target Microwave --timesteps 100000 --gaze_checkpoint logs/RESNET.ckpt

# python src/train_gaze_guided_rl_final.py --exp_name gaze_expt --target Microwave --timesteps 100000 --gaze_checkpoint logs/RESNET.ckpt --gaze_integration channel

# --gaze_integration channel will use ChannelCNN
# --gaze_integration bottleneck will use GazeAttnCNN
# --gaze_integration weighted will use WeightedCNN