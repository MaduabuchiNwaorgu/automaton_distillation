"""
Publication-quality plotting utilities for multi-trial RL experiments.
Supports multiple seeds, parallel environments, and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import pandas as pd
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ExperimentPlotter:
    """
    A class for creating publication-quality plots from multi-seed RL experiments.
    """
    
    def __init__(self, figsize=(12, 8), dpi=300):
        """
        Initialize the plotter with publication settings.
        
        Args:
            figsize: Figure size in inches (width, height)
            dpi: Resolution for saving plots
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        
        # Publication settings
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2.0,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix'
        })
    
    def aggregate_trial_data(self, trial_data: Dict[str, List[np.ndarray]], 
                           window_size: int = 100) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Aggregate reward data across multiple trials for each algorithm.
        
        Args:
            trial_data: Dict mapping algorithm names to lists of reward arrays (one per seed)
            window_size: Window size for moving average smoothing
            
        Returns:
            Dict with aggregated statistics for each algorithm
        """
        aggregated = {}
        
        for algo_name, trials in trial_data.items():
            if not trials:
                continue
                
            # Find the minimum length across all trials
            min_length = min(len(trial) for trial in trials)
            
            # Truncate all trials to the same length
            aligned_trials = np.array([trial[:min_length] for trial in trials])
            
            # Calculate statistics
            mean_rewards = np.mean(aligned_trials, axis=0)
            std_rewards = np.std(aligned_trials, axis=0)
            sem_rewards = stats.sem(aligned_trials, axis=0)  # Standard error of mean
            
            # Apply smoothing
            if window_size > 1:
                mean_rewards = self._moving_average(mean_rewards, window_size)
                std_rewards = self._moving_average(std_rewards, window_size)
                sem_rewards = self._moving_average(sem_rewards, window_size)
            
            # Calculate confidence intervals (95%)
            confidence_level = 0.95
            alpha = 1 - confidence_level
            t_val = stats.t.ppf(1 - alpha/2, len(trials) - 1)
            ci_lower = mean_rewards - t_val * sem_rewards
            ci_upper = mean_rewards + t_val * sem_rewards
            
            aggregated[algo_name] = {
                'mean': mean_rewards,
                'std': std_rewards,
                'sem': sem_rewards,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'episodes': np.arange(len(mean_rewards)),
                'num_trials': len(trials)
            }
            
        return aggregated
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average smoothing to data."""
        if window_size <= 1:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def plot_learning_curves(self, aggregated_data: Dict[str, Dict[str, np.ndarray]], 
                           title: str = "Learning Curves",
                           xlabel: str = "Episodes",
                           ylabel: str = "Average Reward",
                           save_path: Optional[str] = None,
                           show_ci: bool = True,
                           show_std: bool = False,
                           log_scale: bool = False) -> plt.Figure:
        """
        Plot learning curves with error bands for multiple algorithms.
        
        Args:
            aggregated_data: Output from aggregate_trial_data()
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the figure (optional)
            show_ci: Whether to show confidence intervals
            show_std: Whether to show standard deviation bands
            log_scale: Whether to use log scale for y-axis
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for i, (algo_name, data) in enumerate(aggregated_data.items()):
            color = self.colors[i % len(self.colors)]
            episodes = data['episodes']
            mean_rewards = data['mean']
            
            # Plot main line
            ax.plot(episodes, mean_rewards, 
                   label=f"{algo_name} (n={data['num_trials']})",
                   color=color, linewidth=2.5, alpha=0.9)
            
            # Add error bands
            if show_ci:
                ax.fill_between(episodes, data['ci_lower'], data['ci_upper'],
                              alpha=0.2, color=color, label=f"{algo_name} 95% CI")
            
            if show_std:
                std_lower = mean_rewards - data['std']
                std_upper = mean_rewards + data['std']
                ax.fill_between(episodes, std_lower, std_upper,
                              alpha=0.15, color=color, linestyle='--')
        
        # Formatting
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_final_performance_comparison(self, aggregated_data: Dict[str, Dict[str, np.ndarray]],
                                        title: str = "Final Performance Comparison",
                                        save_path: Optional[str] = None,
                                        last_n_episodes: int = 100) -> plt.Figure:
        """
        Create a bar plot comparing final performance across algorithms.
        
        Args:
            aggregated_data: Output from aggregate_trial_data()
            title: Plot title
            save_path: Path to save the figure (optional)
            last_n_episodes: Number of final episodes to average over
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        algo_names = []
        final_means = []
        final_stds = []
        
        for algo_name, data in aggregated_data.items():
            # Calculate final performance (average of last N episodes)
            final_rewards = data['mean'][-last_n_episodes:]
            final_mean = np.mean(final_rewards)
            final_std = np.mean(data['std'][-last_n_episodes:])
            
            algo_names.append(f"{algo_name}\n(n={data['num_trials']})")
            final_means.append(final_mean)
            final_stds.append(final_std)
        
        # Create bar plot
        x_pos = np.arange(len(algo_names))
        bars = ax.bar(x_pos, final_means, yerr=final_stds, capsize=5,
                     color=self.colors[:len(algo_names)], alpha=0.8,
                     edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, final_means, final_stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.01,
                   f'{mean_val:.1f}±{std_val:.1f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Algorithm', fontweight='bold')
        ax.set_ylabel(f'Average Reward (Last {last_n_episodes} Episodes)', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algo_names)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Figure saved to: {save_path}")
        
        return fig
    
    def plot_statistical_summary(self, aggregated_data: Dict[str, Dict[str, np.ndarray]],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive statistical summary plot.
        
        Args:
            aggregated_data: Output from aggregate_trial_data()
            save_path: Path to save the figure (optional)
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        for i, (algo_name, data) in enumerate(aggregated_data.items()):
            color = self.colors[i % len(self.colors)]
            episodes = data['episodes']
            
            # 1. Learning curves with CI
            ax1.plot(episodes, data['mean'], label=algo_name, color=color, linewidth=2)
            ax1.fill_between(episodes, data['ci_lower'], data['ci_upper'],
                           alpha=0.3, color=color)
        
        ax1.set_title('Learning Curves with 95% Confidence Intervals', fontweight='bold')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Average Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Standard deviation over time
        for i, (algo_name, data) in enumerate(aggregated_data.items()):
            color = self.colors[i % len(self.colors)]
            ax2.plot(data['episodes'], data['std'], label=algo_name, color=color, linewidth=2)
        
        ax2.set_title('Standard Deviation Over Training', fontweight='bold')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Standard Deviation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot of final performance
        final_data = []
        labels = []
        for algo_name, data in aggregated_data.items():
            final_100 = data['mean'][-100:]  # Last 100 episodes
            final_data.append(final_100)
            labels.append(algo_name)
        
        ax3.boxplot(final_data, labels=labels)
        ax3.set_title('Final Performance Distribution\n(Last 100 Episodes)', fontweight='bold')
        ax3.set_ylabel('Average Reward')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Sample efficiency plot (episodes to reach threshold)
        thresholds = np.linspace(
            min(data['mean'].min() for data in aggregated_data.values()),
            max(data['mean'].max() for data in aggregated_data.values()) * 0.9,
            20
        )
        
        for i, (algo_name, data) in enumerate(aggregated_data.items()):
            color = self.colors[i % len(self.colors)]
            episodes_to_threshold = []
            
            for threshold in thresholds:
                # Find first episode where moving average exceeds threshold
                smoothed = self._moving_average(data['mean'], 50)
                indices = np.where(smoothed >= threshold)[0]
                if len(indices) > 0:
                    episodes_to_threshold.append(indices[0])
                else:
                    episodes_to_threshold.append(len(data['mean']))
            
            ax4.plot(thresholds, episodes_to_threshold, label=algo_name, 
                    color=color, linewidth=2, marker='o', markersize=4)
        
        ax4.set_title('Sample Efficiency', fontweight='bold')
        ax4.set_xlabel('Reward Threshold')
        ax4.set_ylabel('Episodes to Reach Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Statistical summary saved to: {save_path}")
        
        return fig


def load_experiment_data(data_dir: str, pattern: str = "*_rewards.npy") -> Dict[str, List[np.ndarray]]:
    """
    Load reward data from multiple experiment files.
    
    Args:
        data_dir: Directory containing experiment data
        pattern: File pattern to match (e.g., "*_rewards.npy")
        
    Returns:
        Dict mapping algorithm names to lists of reward arrays
    """
    data_dir = Path(data_dir)
    experiment_data = {}
    
    for file_path in data_dir.glob(pattern):
        # Extract algorithm name and seed from filename
        # Expected format: algo_name_seed_rewards.npy
        parts = file_path.stem.split('_')
        if len(parts) >= 3 and parts[-1] == 'rewards':
            algo_name = '_'.join(parts[:-2])  # Everything except seed and 'rewards'
            seed = parts[-2]
            
            if algo_name not in experiment_data:
                experiment_data[algo_name] = []
            
            rewards = np.load(file_path)
            experiment_data[algo_name].append(rewards)
    
    return experiment_data


# Example usage and utility functions
def create_sample_data(num_algorithms: int = 3, num_seeds: int = 5, 
                      num_episodes: int = 1000) -> Dict[str, List[np.ndarray]]:
    """
    Create sample data for testing plotting functions.
    """
    algorithms = [f"Algorithm_{i+1}" for i in range(num_algorithms)]
    sample_data = {}
    
    np.random.seed(42)  # For reproducible example
    
    for i, algo in enumerate(algorithms):
        sample_data[algo] = []
        
        for seed in range(num_seeds):
            # Create synthetic learning curves with different characteristics
            episodes = np.arange(num_episodes)
            
            # Base performance with algorithm-specific characteristics
            base_performance = -100 + i * 50  # Different starting points
            learning_rate = 0.01 + i * 0.005  # Different learning rates
            final_performance = base_performance + 150 + i * 20
            
            # Generate learning curve
            progress = 1 - np.exp(-learning_rate * episodes)
            rewards = base_performance + (final_performance - base_performance) * progress
            
            # Add noise
            noise = np.random.normal(0, 10 + i * 5, size=num_episodes)
            rewards += noise
            
            # Add some random fluctuations
            fluctuations = 20 * np.sin(episodes * 0.02) * np.exp(-episodes * 0.001)
            rewards += fluctuations
            
            sample_data[algo].append(rewards)
    
    return sample_data


if __name__ == "__main__":
    # Example usage
    print("Creating publication-quality plots...")
    
    # Create sample data
    sample_data = create_sample_data(num_algorithms=3, num_seeds=5, num_episodes=1000)
    
    # Initialize plotter
    plotter = ExperimentPlotter(figsize=(12, 8))
    
    # Aggregate data
    aggregated = plotter.aggregate_trial_data(sample_data, window_size=50)
    
    # Create plots
    fig1 = plotter.plot_learning_curves(
        aggregated, 
        title="HalfCheetah Patrol Task: Learning Curves",
        xlabel="Training Episodes",
        ylabel="Average Episode Reward",
        save_path="plots/learning_curves.pdf",
        show_ci=True
    )
    
    fig2 = plotter.plot_final_performance_comparison(
        aggregated,
        title="Final Performance Comparison",
        save_path="plots/final_performance.pdf"
    )
    
    fig3 = plotter.plot_statistical_summary(
        aggregated,
        save_path="plots/statistical_summary.pdf"
    )
    
    plt.show()
