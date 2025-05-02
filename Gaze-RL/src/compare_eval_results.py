#!/usr/bin/env python
import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple RL evaluation results")
    parser.add_argument("--results", type=str, nargs='+', required=True,
                        help="Paths to multiple result YAML files")
    parser.add_argument("--output", type=str, default="comparison_results",
                        help="Directory to save comparison plots")
    parser.add_argument("--window", type=int, default=5,
                        help="Smoothing window size (must be odd)")
    parser.add_argument("--title", type=str, default="Reward Comparison",
                        help="Plot title")
    parser.add_argument("--labels", type=str, nargs='+',
                        help="Custom labels for each result file (in same order as results)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 8],
                        help="Figure size (width, height) in inches")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for saved figures")
    parser.add_argument("--style", type=str, default="seaborn-v0_8-darkgrid",
                        help="Matplotlib style to use")
    return parser.parse_args()

def load_results(filepath):
    """Load results from a YAML file"""
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['episode_rewards', 'experiment', 'target_object']
        for field in required_fields:
            if field not in data:
                print(f"Warning: Missing required field '{field}' in {filepath}")
        
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def smooth_data(data, window_size=9):
    """Apply smoothing to the data using Savitzky-Golay filter"""
    if len(data) < window_size:
        # If data is shorter than window, use a smaller window
        window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
        # If window size is too small, return original data
        if window_size < 3:
            return data
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size -= 1
    
    # Apply Savitzky-Golay filter
    try:
        polynomial_order = min(3, window_size - 1)  # Order must be less than window_size
        smoothed = savgol_filter(data, window_size, polynomial_order)
        return smoothed
    except Exception as e:
        print(f"Error during smoothing: {e}. Using original data.")
        return data

def get_experiment_name(data, custom_label=None):
    """Extract a nice display name from the experiment data"""
    if custom_label:
        return custom_label
    
    # Try to generate a readable name from the experiment field
    experiment = data.get('experiment', 'Unknown')
    
    # Remove timestamp patterns if present
    import re
    experiment = re.sub(r'_\d{8}_\d{6}', '', experiment)
    
    # Replace underscores with spaces and capitalize words
    name = ' '.join(word.capitalize() for word in experiment.split('_'))
    
    # Add target object if available
    target = data.get('target_object', None)
    if target:
        name += f" ({target})"
    
    return name

def plot_comparison(results_data, output_dir, window_size=9, title="Reward Comparison", 
                   custom_labels=None, figsize=(12, 8), dpi=300):
    """Create comparison plots for multiple result files"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up colors - use tab10 colormap for distinct colors
    colors = plt.cm.tab10.colors
    
    # Create the main rewards over episodes plot
    plt.figure(figsize=figsize)
    
    # Track max episodes for x-axis
    max_episodes = 0
    
    # For legend entries
    legend_elements = []
    
    # Summary metrics for a table
    summary_data = []
    
    # Plot each result
    for i, (filepath, data) in enumerate(results_data.items()):
        if not data:
            continue
        
        # Get reward data and experiment name
        rewards = data.get('episode_rewards', [])
        custom_label = None if custom_labels is None else custom_labels[i] if i < len(custom_labels) else None
        name = get_experiment_name(data, custom_label)
        
        # Update max episodes
        max_episodes = max(max_episodes, len(rewards))
        
        # Create episode indices
        episodes = np.arange(1, len(rewards) + 1)
        
        # Get color for this experiment
        color = colors[i % len(colors)]
        
        # Smooth data and plot (no background raw data)
        if len(rewards) >= 3:  # Only smooth if we have enough data points
            smoothed_rewards = smooth_data(rewards, window_size)
            plt.plot(episodes, smoothed_rewards, color=color, linewidth=3, label=name)
            
            # Add to legend
            legend_elements.append(Line2D([0], [0], color=color, lw=3, label=name))
        else:
            # Not enough data for smoothing, just use the original
            plt.plot(episodes, rewards, color=color, linewidth=3, label=name)
            
            # Add to legend
            legend_elements.append(Line2D([0], [0], color=color, lw=3, label=name))
        
        # Collect summary metrics
        mean_reward = np.mean(rewards)
        final_10_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        success_rate = data.get('overall_success_rate', 0)
        
        summary_data.append({
            'Name': name,
            'Mean Reward': f"{mean_reward:.2f}",
            'Final 10 Reward': f"{final_10_reward:.2f}",
            'Success Rate': f"{success_rate:.1f}%",
            'Episodes': len(rewards)
        })
    
    # Set plot labels and title
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Adjust x-axis limits
    plt.xlim(1, max_episodes)
    
    # Add explanatory text about smoothing
    plt.figtext(0.02, 0.02, f"Note: Lines show smoothed rewards (window size={window_size})", 
                fontsize=8, ha='left')
    
    # Tight layout and save figure
    plt.tight_layout()
    reward_plot_path = os.path.join(output_dir, 'reward_comparison.png')
    plt.savefig(reward_plot_path, dpi=dpi)
    print(f"Saved reward comparison plot to {reward_plot_path}")
    
    # Create a summary plot with bar charts
    create_summary_chart(summary_data, output_dir, figsize, dpi)
    
    # Return the paths to the created plots
    return reward_plot_path

def create_summary_chart(summary_data, output_dir, figsize, dpi):
    """Create a summary bar chart with key metrics"""
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(summary_data)
    
    # Convert string values to numeric for plotting
    df['Mean Reward'] = df['Mean Reward'].astype(float)
    df['Final 10 Reward'] = df['Final 10 Reward'].astype(float)
    df['Success Rate'] = df['Success Rate'].str.rstrip('%').astype(float)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot Mean Reward
    df.plot(kind='bar', x='Name', y='Mean Reward', ax=axes[0], color='skyblue')
    axes[0].set_title('Mean Reward')
    axes[0].set_ylabel('Reward')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot Final 10 Reward
    df.plot(kind='bar', x='Name', y='Final 10 Reward', ax=axes[1], color='orange')
    axes[1].set_title('Final 10 Episodes Reward')
    axes[1].set_ylabel('Reward')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot Success Rate
    df.plot(kind='bar', x='Name', y='Success Rate', ax=axes[2], color='green')
    axes[2].set_title('Success Rate (%)')
    axes[2].set_ylabel('Percentage')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'metrics_summary.png')
    plt.savefig(summary_path, dpi=dpi)
    print(f"Saved metrics summary plot to {summary_path}")
    
    # Also create a metrics table as CSV
    csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics summary to {csv_path}")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set matplotlib style
    try:
        plt.style.use(args.style)
    except Exception as e:
        print(f"Warning: Could not set style '{args.style}': {e}")
        print("Using default style instead.")
    
    # Load results data
    results_data = {}
    for filepath in args.results:
        data = load_results(filepath)
        if data:
            results_data[filepath] = data
    
    if not results_data:
        print("Error: No valid results files found.")
        return
    
    print(f"Loaded {len(results_data)} result files successfully.")
    
    # Create comparison plot
    plot_comparison(
        results_data,
        args.output,
        window_size=args.window,
        title=args.title,
        custom_labels=args.labels,
        figsize=tuple(args.figsize),
        dpi=args.dpi
    )
    
    print(f"Comparison completed and saved to {args.output}")

if __name__ == "__main__":
    main()

# Example usage:
# python src/compare_eval_results.py --results evaluation_results/eval_baseline_ppo_20250502_015604/baseline_ppo_results.yaml evaluation_results/eval_gaze_expt_channel_20250430_224958/channel_results.yaml evaluation_results/eval_gaze_expt_bottleneck_20250430_234927/bottleneck_results.yaml evaluation_results/eval_gaze_expt_weighted_20250501_101103/weighted_results.yaml --labels "Baseline PPO" "Gaze Channel" "Gaze Bottleneck" "Gaze Weighted" --title "AI2THOR Object Search Performance, Target: Microwave" --output comparison_plots