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
import seaborn as sns

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
    """Create visually appealing comparison plots using Seaborn and advanced styling"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Import seaborn for enhanced styling
    import seaborn as sns
    
    # Set up better seaborn styling
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # Better color palette from seaborn
    palette = sns.color_palette("bright", n_colors=10)
    
    # Create the main rewards over episodes plot with seaborn styling
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add a subtle background color gradient for better aesthetics
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('subtle_gradient', ['#f8f9fa', '#ffffff'], N=100)
    ax.imshow(np.zeros((2, 2)), extent=[-1000, 1000, -1000, 1000], cmap=cmap, 
              aspect='auto', zorder=-1, alpha=0.8)
    
    # Track max episodes and rewards for axis limits
    max_episodes = 0
    min_reward = float('inf')
    max_reward = float('-inf')
    
    # For legend entries and summary data
    legend_elements = []
    summary_data = []
    
    # Plot each result with enhanced seaborn styling
    for i, (filepath, data) in enumerate(results_data.items()):
        if not data:
            continue
        
        # Get reward data and experiment name
        rewards = data.get('episode_rewards', [])
        custom_label = None if custom_labels is None else custom_labels[i] if i < len(custom_labels) else None
        name = get_experiment_name(data, custom_label)
        
        # Update max episodes and reward ranges
        max_episodes = max(max_episodes, len(rewards))
        min_reward = min(min_reward, min(rewards))
        max_reward = max(max_reward, max(rewards))
        
        # Create episode indices
        episodes = np.arange(1, len(rewards) + 1)
        
        # Get color for this experiment
        color = palette[i % len(palette)]
        
        # Prepare for both raw and smoothed data
        if len(rewards) >= 3:  # Only smooth if we have enough data points
            smoothed_rewards = smooth_data(rewards, window_size)
            
            # Plot the main line with seaborn
            sns.lineplot(
                x=episodes, 
                y=smoothed_rewards, 
                color=color, 
                linewidth=3, 
                label=name,
                zorder=3,
                alpha=0.9
            )
            
            # Add to legend
            legend_elements.append(Line2D([0], [0], color=color, lw=3, label=name))
        else:
            # Not enough data for smoothing, just use the original
            sns.lineplot(
                x=episodes, 
                y=rewards, 
                color=color, 
                linewidth=3, 
                label=name,
                zorder=3,
                alpha=0.9
            )
            
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
    
    # Enhance axes with seaborn styling
    sns.despine(left=False, bottom=False, right=False, top=False)
    
    # Set plot labels and title with seaborn styling
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Reward', fontsize=14, fontweight='bold', labelpad=10)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Create a custom legend with seaborn styling
    legend = ax.legend(
        handles=legend_elements,
        loc='best',
        fontsize=12,
        frameon=True,
        facecolor='white',
        framealpha=0.9,
        edgecolor='lightgrey',
        borderpad=1,
        title="Methods",
        title_fontsize=13
    )
    legend.get_frame().set_linewidth(0.5)
    
    # Add a subtle watermark annotation for smoothing info
    ax.annotate(
        f"Smoothed (window size={window_size})",
        xy=(0.98, 0.02),
        xycoords='axes fraction',
        fontsize=10,
        ha='right',
        va='bottom',
        color='#999999',
        style='italic',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey", alpha=0.8)
    )
    
    # Adjust spacing and limits
    plt.tight_layout()
    
    # Adjust x-axis limits with padding
    ax.set_xlim(1, max_episodes + int(max_episodes * 0.05))
    
    # Adjust y-axis limits with padding for better visualization
    y_range = max_reward - min_reward
    ax.set_ylim(min_reward - y_range * 0.05, max_reward + y_range * 0.1)
    
    # Save the enhanced figure
    reward_plot_path = os.path.join(output_dir, 'reward_comparison.png')
    plt.savefig(reward_plot_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved enhanced reward comparison plot to {reward_plot_path}")
    
    # Create a summary chart
    create_summary_chart(summary_data, output_dir, figsize, dpi)
    
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
# python src/compare_eval_results.py --results 
# Gaze-RL/evaluation_results/eval_baseline_ppo_floorplan30_20250502_033454/baseline_ppo_floorplan30_results.yaml
# Gaze-RL/evaluation_results/eval_channel_floorplan30_20250502_034142/channel_results.yaml 
# Gaze-RL/evaluation_results/eval_bottleneck_floorplan30_20250502_034316/bottleneck_results.yaml
# Gaze-RL/evaluation_results/eval_weighted_floorplan30_20250502_042117/weighted_results.yaml
# --labels "Baseline PPO" "Gaze Channel" "Gaze Bottleneck" "Gaze Weighted" 
# --title "AI2THOR Object Search Performance, Target: Microwave & Floorplan 30 - Evaluation" 
# --output comparison_plots_eval_floorplan30_all


# python src/compare_eval_results.py --results evaluation_results/eval_baseline_ppo_floorplan30_20250502_033454/baseline_ppo_floorplan30_results.yaml evaluation_results/eval_channel_floorplan30_20250502_034142/channel_results.yaml evaluation_results/eval_bottleneck_floorplan30_20250502_034316/bottleneck_results.yaml evaluation_results/eval_weighted_floorplan30_20250502_042117/weighted_results.yaml --labels "Baseline PPO" "Gaze Channel" "Gaze Bottleneck" "Gaze Weighted" --title "AI2THOR Object Search Performance, Target: Microwave & Floorplan 30 - Evaluation" --output comparison_plots_eval_floorplan30_all


# python src/compare_eval_results.py --results evaluation_results/eval_baseline_ppo_floorplan1_20250502_024948/baseline_ppo_floorplan1_results.yaml evaluation_results/eval_gaze_channel_floorplan1_20250502_023229/channel_results.yaml evaluation_results/eval_gaze_bottleneck_floorplan1_20250502_025804/bottleneck_results.yaml evaluation_results/eval_gaze_weighted_floorplan1_20250502_025939/weighted_results.yaml --labels "Baseline PPO" "Gaze Channel" "Gaze Bottleneck" "Gaze Weighted" --title "AI2THOR Object Search Performance, Target: Microwave & Floorplan 1 - Evaluation" --output comparison_plots_eval_floorplan1_all

# python src/compare_eval_results.py --results logs/baseline_ppo_100k_floorplan30_20250501_185056/results/metrics.yaml logs/gaze_exp_floorplan30_channel_20250501_163301/results/metrics.yaml logs/gaze_exp_floorplan30_bottleneck_20250501_163353/results/metrics.yaml logs/gaze_exp_floorplan30_weighted_20250501_183725/results/metrics.yaml --labels "Baseline PPO" "Gaze Channel" "Gaze Bottleneck" "Gaze Weighted" --title "AI2THOR Object Search Performance, Target: Microwave & Floorplan 30" --output comparison_plots_train_floorplan30_all

# python src/compare_eval_results.py --results logs/baseline_ppo_100k_floorplan1_20250501_143639/results/metrics.yaml logs/gaze_exp_floorplan1_channel_20250501_174434/results/metrics.yaml logs/gaze_exp_floorplan1_bottleneck_20250501_145550/results/metrics.yaml logs/gaze_exp_floorplan1_weighted_20250501_145407/results/metrics.yaml --labels "Baseline PPO" "Gaze Channel" "Gaze Bottleneck" "Gaze Weighted" --title "AI2THOR Object Search Performance, Target: Microwave & Floorplan 1" --output comparison_plots_train_floorplan1_all

# python src/compare_eval_results.py --results logs/baseline_ppo_100k_floorplan30_20250501_185056/results/metrics.yaml logs/gaze_pretrain_model_without_gaze_reward_channel_floorplan30_second_case/results/metrics.yaml logs/baseline_with_gaze_reward_only_floorplan30_third_case/results/metrics.yaml logs/gaze_exp_floorplan30_channel_20250501_163301/results/metrics.yaml --labels "PPO (No gaze)" "PPO (with gaze pretraining)" "PPO (with gaze incentives)" "PPO (with gaze pretraining & gaze rewards)" --title "AI2THOR Object Search Performance, Target: Microwave & Floorplan 30, Gaze Integration: Channel" --output comparison_plots_train_floorplan30_all_table_case