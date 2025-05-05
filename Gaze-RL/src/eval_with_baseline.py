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
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your environment
from environments.ai2thor_gymnasium_env import AI2ThorEnv
from env_wrappers import VideoRecorderWrapper, RetryTimeoutWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Baseline RL agent for object search")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (either .zip file or directory)")
    parser.add_argument("--target", type=str, default="Microwave",
                        help="Target object to search for")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to evaluate")
    parser.add_argument("--record_video", action="store_true",
                        help="Record videos of evaluation episodes")
    parser.add_argument("--record_freq", type=int, default=5,
                        help="Record every N-th episode if videos are enabled")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions for evaluation")
    parser.add_argument("--exp_name", type=str, default="baseline_ppo",
                        help="Experiment name for results")
    return parser.parse_args()

def create_env(config, target_object, video_dir=None, record_freq=5, seed=None):
    """Create environment with proper settings"""
    
    # Update config for optimization
    env_config = config["environment"].copy()
    env_config["target_object"] = target_object
    env_config["width"] = 128
    env_config["height"] = 128
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
        
        # Wrap with Monitor to track episode rewards
        env = Monitor(env)
        
        # Add video recording as the final wrapper if video_dir is provided
        if video_dir is not None:
            env = VideoRecorderWrapper(env, video_dir=video_dir, record_freq=record_freq)
            print(f"Video recording enabled: recording every {record_freq} episodes to {video_dir}")
        
        return env
    
    return _init

def load_model(model_path, env):
    """Load the trained model"""
    try:
        # Check if model_path is a zip file or a directory
        if os.path.isfile(model_path) and model_path.endswith('.zip'):
            # Load from zip file
            print(f"Loading model from zip file: {model_path}")
            model = PPO.load(model_path, env=env)
        elif os.path.isdir(model_path):
            # Load from directory with extracted model files
            print(f"Loading model from directory: {model_path}")
            policy_path = os.path.join(model_path, "policy.pth")
            
            # Create a new PPO model
            model = PPO(
                "CnnPolicy",
                env,
                verbose=0,
                device="auto"
            )
            
            # Load policy weights
            if os.path.exists(policy_path):
                print(f"Loading policy weights from {policy_path}")
                model.policy.load_state_dict(torch.load(policy_path))
            else:
                print(f"Warning: Policy file not found at {policy_path}")
        else:
            raise ValueError(f"Invalid model path: {model_path}. Must be a .zip file or directory.")
        
        print(f"Successfully loaded model")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
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
        try:
            # Reset environment - handle both return formats (obs, info) or just obs
            reset_result = env.reset()
            
            # Check if result is a tuple (compatible with gymnasium) or single value (older gym)
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                # If just the observation is returned
                obs = reset_result
                info = {}
            
            done = False
            truncated = False
            episode_reward = 0
            steps = 0
            
            # Storage for episode metrics
            visited_positions = set()
            
            # Store initial time
            start_time = time.time()
            success = False
            
            # Run episode
            while not (done or truncated):
                try:
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=deterministic)
                    
                    # Step environment
                    step_result = env.step(action)
                    
                    # Check the format of step_result
                    if len(step_result) == 5:  # New gymnasium format
                        obs, reward, done, truncated, info = step_result
                    else:  # Older gym format
                        obs, reward, done, info = step_result
                        truncated = False
                    
                    # Update metrics
                    episode_reward += reward
                    steps += 1
                    
                    # Track agent's path
                    if isinstance(info, dict) and "agent_position" in info:
                        visited_positions.add(info["agent_position"])
                    
                    # Check for success
                    if isinstance(info, dict) and info.get("success", False):
                        success = True
                        time_to_success.append(time.time() - start_time)
                        break
                
                except (ValueError, TimeoutError) as e:
                    print(f"Error during episode step: {e}")
                    print("Terminating episode early")
                    done = True
                    break
                    
            # Store episode metrics
            episode_rewards.append(float(episode_reward))
            episode_lengths.append(int(steps))
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
            
        except Exception as e:
            print(f"Error during episode {episode}: {e}")
            if episode > 0:
                print(f"Continuing with {episode} completed episodes")
                break
            else:
                raise  # Re-raise if no episodes completed
    
    pbar.close()
    
    # Calculate final metrics
    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(success_rates) * 100),
        "mean_time_to_success": float(np.mean(time_to_success)) if time_to_success else 0,
        "mean_exploration_coverage": float(np.mean(exploration_coverage)),
        "mean_path_efficiency": float(np.mean(path_efficiency) * 100), # Convert to percentage
    }
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Success Rate: {metrics['success_rate']:.1f}%")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Episode Length: {metrics['mean_episode_length']:.1f} ± {metrics['std_episode_length']:.1f} steps")
    print(f"Mean Time to Success: {metrics['mean_time_to_success']:.2f} seconds")
    print(f"Mean Exploration Coverage: {metrics['mean_exploration_coverage']:.1f} positions")
    print(f"Mean Path Efficiency: {metrics['mean_path_efficiency']:.1f}%")
    
    return episode_rewards, episode_lengths, success_rates, metrics

def generate_report(episode_rewards, episode_lengths, success_rates, metrics, output_dir, target_object, exp_name):
    """Generate evaluation report and save results in a standard format"""
    
    # Calculate final 10 episodes mean reward
    final_10_mean_reward = float(np.mean(episode_rewards[-10:])) if len(episode_rewards) >= 10 else float(np.mean(episode_rewards))
    
    # Create data dict in the uniform format (matching your examples)
    results = {
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths],
        "success_rates": [int(s) for s in success_rates],
        "mean_reward": float(metrics["mean_reward"]),
        "final_10_mean_reward": final_10_mean_reward,
        "overall_success_rate": float(metrics["success_rate"]),
        "episodes_completed": len(episode_rewards),
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "target_object": target_object,
        "experiment": exp_name,
        "status": "completed"
    }
    
    # Save results to YAML
    results_path = os.path.join(output_dir, f"{exp_name}_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    
    # Create rewards plot
    plt.figure(figsize=(10, 6))
    plt.plot(results["episode_rewards"], marker='o', markersize=3, linestyle='-', alpha=0.7)
    plt.axhline(y=results["mean_reward"], color='r', linestyle='--', label=f'Mean: {results["mean_reward"]:.2f}')
    plt.title(f"Episode Rewards - {exp_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f"{exp_name}_rewards.png"), dpi=300)
    plt.close()
    
    print(f"Results saved to {results_path}")
    return results

def main():
    """Main function for evaluating a baseline RL agent"""
    # Parse arguments
    args = parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{args.exp_name}_{timestamp}")
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
                "max_grad_norm": 0.5
            }
        }
    
    # Update config with command line arguments
    config["environment"]["target_object"] = args.target
    
    # Log configuration to file
    config_path = os.path.join(output_dir, "eval_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            "evaluation_config": {
                "model_path": args.model_path,
                "target_object": args.target,
                "num_episodes": args.num_episodes,
                "deterministic": args.deterministic,
                "record_video": args.record_video,
                "record_freq": args.record_freq,
                "experiment_name": args.exp_name
            },
            "environment_config": config["environment"],
            "model_config": config["model"]
        }, f)
    
    # Print evaluation setup
    print("\n" + "="*60)
    print(f"EVALUATING BASELINE RL POLICY")
    print(f"Experiment Name: {args.exp_name}")
    print(f"Model Path: {args.model_path}")
    print(f"Target Object: {args.target}")
    print(f"Number of Episodes: {args.num_episodes}")
    print("="*60 + "\n")
    
    # Create environment
    env_fn = create_env(
        config, 
        args.target, 
        video_dir=video_dir, 
        record_freq=args.record_freq
    )
    env = DummyVecEnv([env_fn])
    
    try:
        # Load model
        model = load_model(args.model_path, env)
        if model is None:
            print("Failed to load model. Exiting...")
            return
        
        # Evaluate model
        episode_rewards, episode_lengths, success_rates, metrics = evaluate_model(
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
                metrics,
                output_dir, 
                args.target, 
                args.exp_name
            )
            
            print(f"Evaluation completed with {len(episode_rewards)} episodes")
            print(f"Mean reward: {results['mean_reward']:.2f}")
            print(f"Success rate: {results['overall_success_rate']:.1f}%")
            print(f"All results saved to {output_dir}")
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
# python src/eval_with_baseline.py --model_path logs/baseline_ppo_new_20250502_075403/models/ppo_ai2thor_final --target Microwave --num_episodes 50 --exp_name baseline_ppo_new 