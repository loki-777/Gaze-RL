import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your environment
from environments.ai2thor_gymnasium_env import AI2ThorEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained AI2-THOR agent")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--target", type=str, default="Microwave",
                        help="Target object to search")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering during evaluation")
    parser.add_argument("--record", action="store_true",
                        help="Record videos of episodes")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--exp_name", type=str, default="evaluation",
                        help="Experiment name for logging")
    return parser.parse_args()

def create_env(config, target_object, render=False):
    """Create environment with proper settings"""
    
    # Update config
    env_config = config["environment"].copy()
    env_config["target_object"] = target_object
    env_config["width"] = 224
    env_config["height"] = 224
    env_config["grid_size"] = 0.25
    
    # Create environment
    env = AI2ThorEnv(
        env_config,
        render_mode="human" if render else None
    )
    
    return env

def setup_recording(output_dir, width=224, height=224, fps=10):
    """Set up video recording"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(output_dir, f"episode_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    return video_writer

def evaluate_model(model, env, num_episodes=20, render=False, record=False, output_dir=None):
    """Evaluate model performance over multiple episodes"""
    
    # Set up metrics collection
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    # Set up visualization directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        trajectory_dir = os.path.join(output_dir, "trajectories")
        os.makedirs(trajectory_dir, exist_ok=True)
    
    # Run evaluation episodes
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        # Reset environment
        obs, info = env.reset()
        
        # FIX: Ensure observation is C-contiguous and writable
        if isinstance(obs, np.ndarray):
            # Create a contiguous, writable copy of the array
            obs = np.array(obs, copy=True, dtype=np.float32)
        
        # Set up video recording if needed
        if record and output_dir:
            video_writer = setup_recording(os.path.join(output_dir, "videos"))
            # Record initial frame
            if isinstance(obs, np.ndarray):
                video_writer.write(obs.astype(np.uint8))
        
        # Initialize episode variables
        done = False
        truncated = False
        episode_reward = 0
        episode_steps = 0
        
        # Track agent trajectory for visualization
        trajectory = []
        if info and "agent_position" in info:
            trajectory.append(info["agent_position"])
        
        # Store initial observation
        if output_dir:
            first_frame = obs.copy() if isinstance(obs, np.ndarray) else None
        
        # Run episode
        while not (done or truncated):
            # FIX: Ensure observation is C-contiguous before predict
            if isinstance(obs, np.ndarray):
                obs = np.ascontiguousarray(obs)
            
            # Select action
            action, _ = model.predict(obs, deterministic=True)
            
            # FIX: Convert numpy array action to integer
            if isinstance(action, np.ndarray):
                action_int = action.item()
            else:
                action_int = action
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action_int)
            
            # FIX: Ensure observation is C-contiguous and writable
            if isinstance(obs, np.ndarray):
                obs = np.array(obs, copy=True, dtype=np.float32)
            
            # Record frame if needed
            if record and output_dir and isinstance(obs, np.ndarray):
                # Add visualization elements to frame
                frame = obs.copy()
                # Add action and reward text
                cv2.putText(frame, f"Action: {action_int}", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Reward: {reward:.2f}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # Write frame to video
                video_writer.write(frame.astype(np.uint8))
            
            # Track metrics
            episode_reward += reward
            episode_steps += 1
            
            # Track agent trajectory
            if info and "agent_position" in info:
                trajectory.append(info["agent_position"])
            
            # Safety check for episode length
            if episode_steps >= 500:  # Hard limit to prevent infinite loops
                print(f"Episode reached hard limit of 500 steps, terminating...")
                break
        
        # Track success
        if info and info.get("success", False):
            success_count += 1
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        # Close video writer if needed
        if record and output_dir:
            video_writer.release()
        
        # Save trajectory visualization
        if output_dir and trajectory and len(trajectory) > 1:
            try:
                # Plot trajectory
                plt.figure(figsize=(8, 8))
                
                # Extract x, z coordinates
                xs = [pos[0] for pos in trajectory]
                zs = [pos[1] for pos in trajectory]
                
                # Plot trajectory line
                plt.plot(xs, zs, 'b-', alpha=0.7)
                
                # Mark start and end points
                plt.scatter(xs[0], zs[0], color='green', s=100, marker='o', label='Start')
                plt.scatter(xs[-1], zs[-1], color='red', s=100, marker='x', label='End')
                
                # Add direction arrows every few steps
                arrow_steps = max(1, len(trajectory) // 10)
                for i in range(0, len(trajectory) - 1, arrow_steps):
                    dx = xs[i+1] - xs[i]
                    dz = zs[i+1] - zs[i]
                    # Normalize arrow length
                    length = np.sqrt(dx**2 + dz**2)
                    if length > 0:
                        dx, dz = dx / length * 0.2, dz / length * 0.2
                        plt.arrow(xs[i], zs[i], dx, dz, head_width=0.1, 
                                 head_length=0.1, fc='blue', ec='blue', alpha=0.7)
                
                # Plot formatting
                plt.title(f"Episode {episode+1}: {'Success' if info.get('success', False) else 'Failure'} "
                          f"(Reward: {episode_reward:.2f}, Steps: {episode_steps})")
                plt.xlabel("X coordinate")
                plt.ylabel("Z coordinate")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Save figure
                plt.savefig(os.path.join(trajectory_dir, f"trajectory_ep{episode+1}.png"))
                plt.close()
            except Exception as e:
                print(f"Error plotting trajectory: {e}")
        
        # Print episode summary
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward = {episode_reward:.2f}, "
              f"Steps = {episode_steps}, "
              f"Success = {info.get('success', False)}")
    
    # Calculate overall metrics
    success_rate = success_count / num_episodes
    mean_reward = np.mean(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # Compile metrics
    metrics = {
        "success_rate": float(success_rate),
        "mean_reward": float(mean_reward),
        "mean_episode_length": float(mean_length),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "num_episodes": num_episodes,
    }
    
    return metrics

def plot_evaluation_results(metrics, output_dir):
    """Create visualizations of evaluation results"""
    
    # Create figures directory
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics["episode_rewards"], bins=10, alpha=0.7, color='blue')
    plt.axvline(metrics["mean_reward"], color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {metrics["mean_reward"]:.2f}')
    plt.title("Episode Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "reward_distribution.png"))
    plt.close()
    
    # Plot episode length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics["episode_lengths"], bins=10, alpha=0.7, color='green')
    plt.axvline(metrics["mean_episode_length"], color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {metrics["mean_episode_length"]:.2f}')
    plt.title("Episode Length Distribution")
    plt.xlabel("Steps")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "length_distribution.png"))
    plt.close()
    
    # Create a summary visualization
    plt.figure(figsize=(12, 8))
    
    # Plot all episode rewards
    episodes = range(1, len(metrics["episode_rewards"]) + 1)
    plt.subplot(2, 1, 1)
    plt.bar(episodes, metrics["episode_rewards"], color='blue', alpha=0.7)
    plt.axhline(metrics["mean_reward"], color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {metrics["mean_reward"]:.2f}')
    plt.title("Evaluation Results")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot all episode lengths
    plt.subplot(2, 1, 2)
    plt.bar(episodes, metrics["episode_lengths"], color='green', alpha=0.7)
    plt.axhline(metrics["mean_episode_length"], color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {metrics["mean_episode_length"]:.2f}')
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "evaluation_summary.png"))
    plt.close()

def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create unique experiment folder name with timestamp
    experiment_id = f"{args.exp_name}_{timestamp}"
    
    # Create evaluation directory
    eval_dir = os.path.join(args.output_dir, experiment_id)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(eval_dir, "evaluation.log")
    
    # Create video directory if recording
    if args.record:
        video_dir = os.path.join(eval_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    
    # Display evaluation settings
    print("\n" + "="*60)
    print(f"EVALUATION SETTINGS")
    print(f"Model: {args.model_path}")
    print(f"Target object: {args.target}")
    print(f"Episodes: {args.episodes}")
    print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
    print(f"Recording: {'Enabled' if args.record else 'Disabled'}")
    print(f"Output directory: {eval_dir}")
    print("="*60 + "\n")
    
    try:
        # Load config
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Save the configuration for reference
        config_path = os.path.join(eval_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Create environment
        env = create_env(config, args.target, render=args.render)
        
        # Load model
        print(f"Loading model from {args.model_path}...")
        model = PPO.load(args.model_path)
        
        # Run evaluation
        print(f"Starting evaluation over {args.episodes} episodes...")
        eval_start_time = time.time()
        
        metrics = evaluate_model(
            model=model,
            env=env,
            num_episodes=args.episodes,
            render=args.render,
            record=args.record,
            output_dir=eval_dir
        )
        
        eval_end_time = time.time()
        eval_duration = eval_end_time - eval_start_time
        
        # Save metrics
        metrics_path = os.path.join(eval_dir, "metrics.yaml")
        with open(metrics_path, "w") as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        # Plot results
        plot_evaluation_results(metrics, eval_dir)
        
        # Print evaluation summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print(f"Success rate: {metrics['success_rate']*100:.2f}%")
        print(f"Mean reward: {metrics['mean_reward']:.2f}")
        print(f"Mean episode length: {metrics['mean_episode_length']:.2f}")
        print(f"Evaluation time: {time.strftime('%H:%M:%S', time.gmtime(eval_duration))}")
        print("="*60)
        print(f"Detailed results saved to: {eval_dir}")
        
    except Exception as e:
        print(f"\nERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
            print("Environment closed")

if __name__ == "__main__":
    main()


# python src/eval_with_progress.py --model_path Gaze-RL/logs/baseline_ppo_new_20250502_075403/models/ppo_ai2thor_final.zip --target Microwave --episodes 50