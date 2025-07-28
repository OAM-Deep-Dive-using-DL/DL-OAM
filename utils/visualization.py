import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    """Class for logging training and evaluation metrics."""
    
    def __init__(self, log_dir: str):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save logs
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.metrics = {}
        
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag: Name of the metric
            value: Value of the metric
            step: Training step or episode number
        """
        self.writer.add_scalar(tag, value, step)
        
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append((step, value))
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """
        Log a histogram to TensorBoard.
        
        Args:
            tag: Name of the metric
            values: Array of values
            step: Training step or episode number
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_network_weights(self, model: torch.nn.Module, step: int) -> None:
        """
        Log neural network weights as histograms.
        
        Args:
            model: PyTorch model
            step: Training step or episode number
        """
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"weights/{name}", param.data, step)
    
    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()


def plot_training_curves(
    metrics_logger: MetricsLogger,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training curves from logged metrics.
    
    Args:
        metrics_logger: MetricsLogger instance with recorded metrics
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    metrics = metrics_logger.metrics
    
    if not metrics:
        raise ValueError("No metrics have been logged yet")
    
    # Create subplots based on available metrics
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
    
    # Handle case with only one subplot
    if n_metrics == 1:
        axes = [axes]
    
    for i, (tag, values) in enumerate(metrics.items()):
        steps, vals = zip(*values)
        axes[i].plot(steps, vals)
        axes[i].set_xlabel('Episode')
        axes[i].set_ylabel(tag)
        axes[i].set_title(f'{tag} vs. Episode')
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig


def create_interactive_dashboard(
    episode_rewards: List[float],
    episode_throughputs: List[float],
    episode_handovers: List[float],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create an interactive Plotly dashboard for visualizing RL agent performance.
    
    Args:
        episode_rewards: List of episode rewards
        episode_throughputs: List of episode throughputs
        episode_handovers: List of episode handover counts
        save_path: Path to save the HTML dashboard (if None, dashboard is not saved)
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Episode Rewards', 'Average Throughput', 'Handover Rate'),
        vertical_spacing=0.1
    )
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(
            x=list(range(len(episode_rewards))),
            y=episode_rewards,
            mode='lines',
            name='Reward'
        ),
        row=1, col=1
    )
    
    # Episode throughputs
    fig.add_trace(
        go.Scatter(
            x=list(range(len(episode_throughputs))),
            y=episode_throughputs,
            mode='lines',
            name='Throughput'
        ),
        row=2, col=1
    )
    
    # Episode handovers
    fig.add_trace(
        go.Scatter(
            x=list(range(len(episode_handovers))),
            y=episode_handovers,
            mode='lines',
            name='Handovers'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=900,
        width=1000,
        title_text="RL Agent Performance Dashboard",
        showlegend=False
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
    
    return fig


def visualize_q_values(
    q_values: np.ndarray,
    action_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize Q-values for different actions.
    
    Args:
        q_values: Array of Q-values with shape (n_samples, n_actions)
        action_names: List of action names
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate mean and std of Q-values for each action
    mean_q = np.mean(q_values, axis=0)
    std_q = np.std(q_values, axis=0)
    
    # Create bar plot with error bars
    x = np.arange(len(action_names))
    ax.bar(x, mean_q, yerr=std_q, alpha=0.7)
    
    ax.set_xlabel('Actions')
    ax.set_ylabel('Average Q-Value')
    ax.set_title('Average Q-Values by Action')
    ax.set_xticks(x)
    ax.set_xticklabels(action_names)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    
    # Save the plot  
    if save_path:
        os.makedirs('plots', exist_ok=True)
        if not save_path.startswith('plots/'):
            save_path = os.path.join('plots', os.path.basename(save_path))
        plt.savefig(save_path)
    
    return fig 