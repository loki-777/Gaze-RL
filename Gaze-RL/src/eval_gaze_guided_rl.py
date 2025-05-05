#!/usr/bin/env python
import os
import sys
import time
import argparse
import numpy as np
import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your environment and wrappers
from environments.ai2thor_gymnasium_env import AI2ThorEnv
from env_wrappers import FlattenObservationWrapper, GazeEnvWrapper, GazePreprocessEnvWrapper, VideoRecorderWrapper, RetryTimeoutWrapper
from src.models.agents import GazePPO, CommonFeatureExtractor
from src.models.lightning_module import GazeLightningModule

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gaze-Guided RL agent for object search")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--gaze_checkpoint", type=str, required=True,
                        help="Path to pretrained gaze model checkpoint (.ckpt file)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing extracted model files")
    parser.add_argument("--integration", type=str, required=True, choices=["channel", "bottleneck", "weighted"],
                        help="Gaze integration method used for the model")
    parser.add_argument("--target", type=str, default="Microwave",
                        help="Target object to search for")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to evaluate")
    parser.add_argument("--record_video", action="store_true",
                        help="Record videos of evaluation episodes")
    parser.add_argument("--record_freq", type=int, default=25,
                        help="Record every N-th episode if videos are enabled")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions for evaluation")
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

def create_env(config, target_object, gaze_model=None, video_dir=None, record_freq=5, seed=None):
    """Create environment with gaze integration"""
    
    # Update config for optimization
    env_config = config["environment"].copy()
    env_config["target_object"] = target_object
    env_config["width"] = 224
    env_config["height"] = 224
    env_config["grid_size"] = 0.25
    env_config["quality"] = "Medium"
    env_config["shadows"] = False

    def _init():
        # Create base environment
        env = AI2ThorEnv(
            env_config,
            render_mode=None  # No rendering during evaluation for performance
        )
        
        # Add retry wrapper to handle occasional timeouts
        env = RetryTimeoutWrapper(env, max_retries=3, retry_delay=1.0)
        
        # Add gaze wrapper
        env = GazeEnvWrapper(env, gaze_predictor=gaze_model)
        
        # Add gaze to observation space
        env = GazePreprocessEnvWrapper(env)
        
        # Make observations compatible with SB3
        env = FlattenObservationWrapper(env)
        
        # Add video recording as the final wrapper if video_dir is provided
        if video_dir is not None:
            env = VideoRecorderWrapper(env, video_dir=video_dir, record_freq=record_freq)
            print(f"Video recording enabled: recording every {record_freq} episodes to {video_dir}")
        
        return env
    
    return _init

def create_model(model_dir, env, integration_method):
    """Create a model from extracted model files"""
    # Map integration method to feature extractor class name
    integration_to_extractor = {
        "channel": "ChannelCNN",
        "bottleneck": "GazeAttnCNN",
        "weighted": "WeightedCNN"
    }
    
    # Configure the feature extractor class
    CommonFeatureExtractor.configure(
        network_type=integration_to_extractor[integration_method],
        use_gaze=True
    )
    
    # Create model with manually loaded weights
    try:
        # Create a default model config
        model_config = {
            "features_extractor": integration_to_extractor[integration_method],
            "use_gaze": True,
            "lr": 3.0e-4,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
        
        # Create a new model instance
        model = GazePPO(
            env=env,
            config=model_config,
            policy="CnnPolicy",
            device="auto"
        )
        
        # Load policy weights from extracted files
        policy_path = os.path.join(model_dir, "policy.pth")
        if os.path.exists(policy_path):
            print(f"Loading policy weights from {policy_path}")
            model.policy.load_state_dict(torch.load(policy_path))
        else:
            print(f"Warning: Policy file not found at {policy_path}")
        
        # Load optimizer state if available
        optimizer_path = os.path.join(model_dir, "policy.optimizer.pth")
        if os.path.exists(optimizer_path):
            print(f"Loading optimizer state from {optimizer_path}")
            optimizer_state_dict = torch.load(optimizer_path)
            model.policy.optimizer.load_state_dict(optimizer_state_dict)
        
        print(f"Successfully created model from extracted files in {model_dir}")
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def evaluate_model(model, env, num_episodes=100, deterministic=True):
    """Evaluate model performance over multiple episodes"""
    
    # Create lists to store metrics
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    time_to_success = []
    exploration_coverage = []
    path_efficiency = []
    
    # Initialize progress bar
    pbar = tqdm(total=num_episodes, desc="Evaluating")
    
    for episode in range(num_episodes):
        # Reset environment - handle both return formats (obs, info) or just obs
        # Reset environment - handle both return formats (obs, info) or just obs
        reset_result = env.reset()

        # For VecEnv, the result is a list or tuple with just the observations
        obs = reset_result
        info = {}  # Start with empty info dict

        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        # Storage for episode metrics
        visited_positions = set()

        # Store initial time
        start_time = time.time()
        success = False

        try:
            # Run episode
            while not (done or truncated):
                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)
                
                try:
                    # Step environment - with VecEnv, returns (obs, rewards, dones, infos)
                    obs, reward, done, infos = env.step(action)
                    
                    # In VecEnv, these are all lists - get first environment's info
                    reward = reward[0]
                    done = done[0]
                    truncated = False  # Assume no truncation unless specified
                    info = infos[0] if isinstance(infos, list) and len(infos) > 0 else {}
                    
                    # Update metrics
                    episode_reward += reward
                    steps += 1
                    
                    # Track agent's path
                    if isinstance(info, dict) and "agent_position" in info:
                        visited_positions.add(info["agent_position"])
                    
                    # Check for success (make sure info is a dict first)
                    if isinstance(info, dict) and info.get("success", False):
                        success = True
                        time_to_success.append(time.time() - start_time)
                        break
                
                except (ValueError, TimeoutError) as e:
                    print(f"Error during episode step: {e}")
                    print("Terminating episode early")
                    break
        except Exception as e:
            print(f"Error during episode: {e}")
            # If we got any episodes completed, continue with those
            if episode > 0:
                print(f"Continuing with {episode} completed episodes")
                break
            else:
                raise  # Re-raise if no episodes completed
                
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        success_rates.append(1 if success else 0)
        
        # Calculate exploration metrics
        exploration_coverage.append(len(visited_positions))
        
        # Calculate path efficiency (only for successful episodes)
        if success:
            # Path efficiency = optimal path length / actual path length (lower is better)
            # Since we don't know optimal path, use a proxy: 1 / steps
            path_efficiency.append(1.0 / steps)
        else:
            path_efficiency.append(0.0)
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            "success_rate": f"{np.mean(success_rates) * 100:.1f}%",
            "avg_reward": f"{np.mean(episode_rewards):.2f}"
        })
    
    pbar.close()
    
    # Calculate final metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "success_rate": np.mean(success_rates) * 100,
        "mean_time_to_success": np.mean(time_to_success) if time_to_success else 0,
        "mean_exploration_coverage": np.mean(exploration_coverage),
        "mean_path_efficiency": np.mean(path_efficiency) * 100, # Convert to percentage
    }
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Success Rate: {metrics['success_rate']:.1f}%")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Episode Length: {metrics['mean_episode_length']:.1f} ± {metrics['std_episode_length']:.1f} steps")
    print(f"Mean Time to Success: {metrics['mean_time_to_success']:.2f} seconds")
    print(f"Mean Exploration Coverage: {metrics['mean_exploration_coverage']:.1f} positions")
    print(f"Mean Path Efficiency: {metrics['mean_path_efficiency']:.1f}%")
    
    return metrics, episode_rewards, episode_lengths, success_rates

def generate_report(episode_rewards, episode_lengths, success_rates, output_dir, target_object, integration_method):
    """Generate simplified evaluation report with just the data"""
    
    # Convert numpy arrays to regular Python lists to avoid serialization issues
    episode_rewards_list = [float(r) for r in episode_rewards]
    episode_lengths_list = [int(l) for l in episode_lengths]
    success_rates_list = [int(s) for s in success_rates]
    
    # Calculate summary metrics
    mean_reward = float(np.mean(episode_rewards))
    final_10_mean_reward = float(np.mean(episode_rewards[-10:])) if len(episode_rewards) >= 10 else mean_reward
    overall_success_rate = float(np.mean(success_rates) * 100) if len(success_rates) > 0 else 0.0
    
    # Create data dict in required format
    results = {
        "episode_rewards": episode_rewards_list,
        "episode_lengths": episode_lengths_list,
        "success_rates": success_rates_list,
        "mean_reward": mean_reward,
        "final_10_mean_reward": final_10_mean_reward,
        "overall_success_rate": overall_success_rate,
        "episodes_completed": len(episode_rewards),
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "target_object": target_object,
        "experiment": f"gaze_guided_{integration_method}",
        "status": "completed"
    }
    
    # Save results to YAML
    results_path = os.path.join(output_dir, f"{integration_method}_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)  # Disable flow style for better readability
    
    # Create just the rewards plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards_list, marker='o', markersize=3, linestyle='-', alpha=0.7)
    plt.axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    plt.title(f"Episode Rewards - {integration_method.capitalize()} Integration")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{integration_method}_rewards.png"), dpi=300)
    plt.close()
    
    print(f"Results saved to {results_path}")
    return results

def main():
    """Main function for evaluating a gaze-guided RL agent"""
    # Parse arguments
    args = parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{args.integration}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create video directory if recording
    video_dir = None
    if args.record_video:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    
    # Load configs
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        # Create a default config
        config = {
            "environment": {
                "target_object": args.target,
                "width": 224,
                "height": 224,
                "grid_size": 0.25,
                "quality": "Medium",
                "shadows": False,
                "max_steps": 500
            },
            "model": {
                "lr": 3.0e-4,
                "n_steps": 128,
                "batch_size": 64,
                "n_epochs": 4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "features_extractor": args.integration,
                "use_gaze": True
            }
        }
    
    # Update config with command line arguments
    config["environment"]["target_object"] = args.target
    
    # Log configuration to file
    config_path = os.path.join(output_dir, "eval_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            "evaluation_config": {
                "model_dir": args.model_dir,
                "integration_method": args.integration,
                "target_object": args.target,
                "num_episodes": args.num_episodes,
                "deterministic": args.deterministic,
                "record_video": args.record_video,
                "record_freq": args.record_freq
            },
            "environment_config": config["environment"],
            "model_config": config["model"]
        }, f)
    
    # Load gaze model
    gaze_model = load_gaze_model(args.gaze_checkpoint)
    if gaze_model is None:
        print("Failed to load gaze model. Exiting...")
        return
    
    # Print evaluation setup
    print("\n" + "="*60)
    print(f"EVALUATING GAZE-GUIDED RL POLICY")
    print(f"Integration Method: {args.integration}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Target Object: {args.target}")
    print(f"Number of Episodes: {args.num_episodes}")
    print("="*60 + "\n")
    
    # Create environment
    env_fn = create_env(
        config, 
        args.target, 
        gaze_model=gaze_model, 
        video_dir=video_dir, 
        record_freq=args.record_freq
    )
    env = DummyVecEnv([env_fn])
    
    try:
        # Create model from extracted files
        model = create_model(args.model_dir, env, args.integration)
        if model is None:
            print("Failed to create model. Exiting...")
            return
        
        # Evaluate model
        metrics, episode_rewards, episode_lengths, success_rates = evaluate_model(
            model, 
            env, 
            num_episodes=args.num_episodes, 
            deterministic=args.deterministic
        )
        
        # Check if we got any results
        if len(episode_rewards) > 0:
            # Generate report
            results = generate_report(
                episode_rewards, 
                episode_lengths, 
                success_rates, 
                output_dir, 
                args.target, 
                args.integration
            )
            print(f"Evaluation completed with {len(episode_rewards)} episodes")
            print(f"Mean reward: {results['mean_reward']:.2f}")
            print(f"Success rate: {results['overall_success_rate']:.1f}%")
        else:
            print("No evaluation episodes completed successfully.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
            print("Environment closed")

if __name__ == "__main__":
    main()

# Example usage:
# python src/eval_gaze_guided_rl.py --gaze_checkpoint logs/RESNET.ckpt --model_dir logs/gaze_exp_floorplan30_channel_20250501_163301/models/ppo_ai2thor_final --integration channel --target Microwave --num_episodes 50 

# python src/eval_gaze_guided_rl.py --gaze_checkpoint logs/RESNET.ckpt --model_dir logs/gaze_exp_floorplan30_bottleneck_20250501_163353/models/ppo_ai2thor_final --integration bottleneck --target Microwave --num_episodes 50 

# python src/eval_gaze_guided_rl.py --gaze_checkpoint logs/RESNET.ckpt --model_dir logs/gaze_exp_floorplan30_weighted_20250501_183725/models/ppo_ai2thor_final --integration weighted --target Microwave --num_episodes 50
