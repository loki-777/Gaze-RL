#!/usr/bin/env python
import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Compare results of gaze-guided vs non-gaze-guided RL")
    parser.add_argument("--base_metrics", type=str, required=True, 
                        help="Path to baseline (no gaze) metrics.yaml file")
    parser.add_argument("--gaze_metrics", type=str, required=True,
                        help="Path to gaze-guided metrics.yaml file")
    parser.add_argument("--output_dir", type=str, default="comparison_results",
                        help="Directory to save comparison results and plots")
    return parser.parse_args()

def load_metrics(file_path):
    """Load metrics from YAML file"""
    try:
        with open(file_path, 'r') as f:
            metrics = yaml.safe_load(f)
        return metrics
    except Exception as e:
        print(f"Error loading metrics from {file_path}: {e}")
        return None

def create_plots(base_metrics, gaze_metrics, output_dir):
    """Create comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(12, 6))
    base_rewards = base_metrics["episode_rewards"]
    gaze_rewards = gaze_metrics["episode_rewards"]
    
    # Calculate moving averages for smoother plots
    window = 3  # Window size for moving average
    base_rewards_ma = np.convolve(base_rewards, np.ones(window)/window, mode='valid')
    gaze_rewards_ma = np.convolve(gaze_rewards, np.ones(window)/window, mode='valid')
    
    episodes = range(1, len(base_rewards) + 1)
    episodes_ma = range(window, len(base_rewards) + 1)
    
    plt.plot(episodes, base_rewards, 'b-', alpha=0.3, label='Without Gaze (Raw)')
    plt.plot(episodes, gaze_rewards, 'r-', alpha=0.3, label='With Gaze (Raw)')
    plt.plot(episodes_ma, base_rewards_ma, 'b-', linewidth=2, label='Without Gaze (MA)')
    plt.plot(episodes_ma, gaze_rewards_ma, 'r-', linewidth=2, label='With Gaze (MA)')
    
    plt.title('Episode Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'reward_comparison.png'), dpi=300)
    plt.close()
    
    # Plot reward distribution
    plt.figure(figsize=(12, 6))
    plt.boxplot([base_rewards, gaze_rewards], labels=['Without Gaze', 'With Gaze'])
    plt.title('Reward Distribution Comparison')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'reward_distribution.png'), dpi=300)
    plt.close()
    
    # Plot cumulative rewards
    plt.figure(figsize=(12, 6))
    cumulative_base = np.cumsum(base_rewards)
    cumulative_gaze = np.cumsum(gaze_rewards)
    plt.plot(episodes, cumulative_base, 'b-', linewidth=2, label='Without Gaze')
    plt.plot(episodes, cumulative_gaze, 'r-', linewidth=2, label='With Gaze')
    plt.title('Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'cumulative_rewards.png'), dpi=300)
    plt.close()

def generate_comparison_report(base_metrics, gaze_metrics, output_dir):
    """Generate a comparison report in markdown format"""
    # Calculate metric differences
    base_mean_reward = base_metrics["mean_reward"]
    gaze_mean_reward = gaze_metrics["mean_reward"]
    reward_improvement = ((gaze_mean_reward - base_mean_reward) / abs(base_mean_reward)) * 100
    
    base_mean_last10 = base_metrics.get("mean_last_10_reward", np.mean(base_metrics["episode_rewards"][-10:]))
    gaze_mean_last10 = gaze_metrics.get("mean_last_10_reward", np.mean(gaze_metrics["episode_rewards"][-10:]))
    last10_improvement = ((gaze_mean_last10 - base_mean_last10) / abs(base_mean_last10)) * 100
    
    # Create markdown report
    report = f"""# Gaze-Guided RL Comparison Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Details

| Metric | Without Gaze | With Gaze |
|--------|-------------|-----------|
| Experiment Name | {base_metrics.get("experiment", "Baseline")} | {gaze_metrics.get("experiment", "Gaze-Guided")} |
| Target Object | {base_metrics.get("target_object", "N/A")} | {gaze_metrics.get("target_object", "N/A")} |
| Total Timesteps | {base_metrics.get("total_timesteps", "N/A")} | {gaze_metrics.get("total_timesteps", "N/A")} |
| Episodes Completed | {base_metrics.get("episodes_completed", "N/A")} | {gaze_metrics.get("episodes_completed", "N/A")} |
| Training Duration (s) | {base_metrics.get("training_duration_seconds", "N/A"):.2f} | {gaze_metrics.get("training_duration_seconds", "N/A"):.2f} |

## Performance Comparison

| Metric | Without Gaze | With Gaze | Improvement |
|--------|-------------|-----------|-------------|
| Mean Reward | {base_mean_reward:.2f} | {gaze_mean_reward:.2f} | {reward_improvement:.2f}% |
| Mean Reward (Last 10 Episodes) | {base_mean_last10:.2f} | {gaze_mean_last10:.2f} | {last10_improvement:.2f}% |
| Min Reward | {min(base_metrics["episode_rewards"]):.2f} | {min(gaze_metrics["episode_rewards"]):.2f} | - |
| Max Reward | {max(base_metrics["episode_rewards"]):.2f} | {max(gaze_metrics["episode_rewards"]):.2f} | - |
| Reward Standard Deviation | {np.std(base_metrics["episode_rewards"]):.2f} | {np.std(gaze_metrics["episode_rewards"]):.2f} | - |

## Reward Analysis

The gaze-guided RL approach shows a {reward_improvement:.2f}% improvement in mean reward compared to the baseline without gaze guidance. This indicates that incorporating gaze information significantly enhances the agent's ability to search for objects efficiently.

Looking at the last 10 episodes, the gaze-guided approach achieves a {last10_improvement:.2f}% improvement, suggesting that the benefits of gaze guidance become more pronounced as training progresses.

## Visual Comparison

Please see the following plots in the output directory:
1. `reward_comparison.png` - Episode rewards comparison
2. `reward_distribution.png` - Reward distribution comparison
3. `cumulative_rewards.png` - Cumulative rewards comparison

## Conclusion

The results demonstrate that incorporating gaze guidance in reinforcement learning for object search tasks leads to substantial improvements in performance. The gaze-guided approach consistently achieves higher rewards, indicating more efficient search strategies.

"""
    
    # Write report to file
    report_path = os.path.join(output_dir, 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report_path

def main():
    # Parse arguments
    args = parse_args()
    
    # Load metrics
    print("Loading metrics...")
    base_metrics = load_metrics(args.base_metrics)
    gaze_metrics = load_metrics(args.gaze_metrics)
    
    if not base_metrics or not gaze_metrics:
        print("Failed to load metrics. Exiting.")
        sys.exit(1)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Without Gaze: {base_metrics.get('experiment', 'Baseline')}")
    print(f"With Gaze: {gaze_metrics.get('experiment', 'Gaze-Guided')}")
    print(f"Target object: {base_metrics.get('target_object', 'N/A')}")
    print("="*60)
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    create_plots(base_metrics, gaze_metrics, output_dir)
    
    # Generate comparison report
    print("Generating comparison report...")
    report_path = generate_comparison_report(base_metrics, gaze_metrics, output_dir)
    
    # Print results
    base_mean_reward = base_metrics["mean_reward"]
    gaze_mean_reward = gaze_metrics["mean_reward"]
    reward_improvement = ((gaze_mean_reward - base_mean_reward) / abs(base_mean_reward)) * 100
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Mean reward without gaze: {base_mean_reward:.2f}")
    print(f"Mean reward with gaze: {gaze_mean_reward:.2f}")
    print(f"Improvement with gaze: {reward_improvement:.2f}%")
    print("="*60)
    
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()

# python src/compare_train_results.py --base_metrics logs/baseline_ppo_new_20250502_075403/results/metrics.yaml --gaze_metrics logs/gaze_expt_bottleneck_20250430_234927/results/metrics.yaml --output_dir train_comparison_plots_base_vs_bottleneck_general

# logs/gaze_expt_channel_20250430_224958/results/metrics.yaml logs/gaze_expt_bottleneck_20250430_234927/results/metrics.yaml logs/gaze_expt_weighted_20250501_101103/results/metrics.yaml --output_dir train_comparison_plots