#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def find_latest_training_dirs():
    """Find the latest training directories for original and stable training."""
    # Find all training directories
    original_dirs = sorted(glob("results/train_2025*"))
    stable_dirs = sorted(glob("results/train_stable_2025*"))
    
    # Return the latest directories
    return original_dirs[-1] if original_dirs else None, stable_dirs[-1] if stable_dirs else None

def load_metrics(directory):
    """Load metrics from a training directory."""
    metrics_file = os.path.join(directory, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return None

def moving_average(data, window=10):
    """Calculate moving average of data."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def compare_rewards(original_dir, stable_dir, output_dir="plots", window=10):
    """Compare rewards from original and stable training."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics
    original_metrics = load_metrics(original_dir)
    stable_metrics = load_metrics(stable_dir)
    
    if not original_metrics or not stable_metrics:
        print("Could not load metrics from one or both directories.")
        return
    
    # Extract rewards
    original_rewards = original_metrics.get("rewards", [])
    stable_rewards = stable_metrics.get("rewards", [])
    
    # Calculate moving averages
    original_ma = moving_average(original_rewards, window)
    
    # Get stable moving average (may already be in metrics)
    if "avg_rewards" in stable_metrics:
        stable_ma = stable_metrics["avg_rewards"]
    else:
        stable_ma = moving_average(stable_rewards, window)
    
    # Ensure same length for comparison
    min_length = min(len(original_ma), len(stable_ma))
    original_ma = original_ma[:min_length]
    stable_ma = stable_ma[:min_length]
    
    # Calculate statistics
    original_std = np.std(original_rewards)
    stable_std = np.std(stable_rewards)
    variance_reduction = (original_std - stable_std) / original_std * 100 if original_std > 0 else 0
    
    print(f"Original reward standard deviation: {original_std:.2f}")
    print(f"Stable reward standard deviation: {stable_std:.2f}")
    print(f"Variance reduction: {variance_reduction:.2f}%")
    
    # Create comparison plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Raw rewards
    axs[0].plot(original_rewards, 'b-', alpha=0.5, label='Original')
    axs[0].plot(stable_rewards, 'r-', alpha=0.5, label='Stable')
    axs[0].set_title('Raw Reward Comparison')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Moving average rewards
    axs[1].plot(original_ma, 'b-', linewidth=2, label=f'Original (MA-{window})')
    axs[1].plot(stable_ma, 'r-', linewidth=2, label=f'Stable (MA-{window})')
    axs[1].set_title('Moving Average Reward Comparison')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel(f'Reward (MA-{window})')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Add variance reduction text
    plt.figtext(0.5, 0.01, 
                f"Variance Reduction: {variance_reduction:.2f}% | Original StdDev: {original_std:.2f} | Stable StdDev: {stable_std:.2f}",
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save plot
    output_file = os.path.join(output_dir, "reward_comparison.png")
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to {output_file}")
    
    # Create additional plots
    
    # Handovers comparison
    original_handovers = original_metrics.get("handovers", [])
    stable_handovers = stable_metrics.get("handovers", [])
    
    plt.figure(figsize=(10, 6))
    plt.plot(original_handovers, 'b-', alpha=0.7, label='Original')
    plt.plot(stable_handovers, 'r-', alpha=0.7, label='Stable')
    plt.title('Handovers Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Number of Handovers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "handovers_comparison.png")
    plt.savefig(output_file, dpi=300)
    print(f"Handovers comparison saved to {output_file}")
    
    # Throughput comparison
    original_throughput = original_metrics.get("throughputs", [])
    stable_throughput = stable_metrics.get("throughputs", [])
    
    plt.figure(figsize=(10, 6))
    plt.plot(original_throughput, 'b-', alpha=0.7, label='Original')
    plt.plot(stable_throughput, 'r-', alpha=0.7, label='Stable')
    plt.title('Throughput Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Throughput (bps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "throughput_comparison.png")
    plt.savefig(output_file, dpi=300)
    print(f"Throughput comparison saved to {output_file}")
    
    # Reward histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(original_rewards, bins=30, alpha=0.7, color='blue')
    plt.title('Original Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(stable_rewards, bins=30, alpha=0.7, color='red')
    plt.title('Stable Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "reward_distribution.png")
    plt.savefig(output_file, dpi=300)
    print(f"Reward distribution comparison saved to {output_file}")

if __name__ == "__main__":
    # Find latest training directories
    original_dir, stable_dir = find_latest_training_dirs()
    
    if original_dir and stable_dir:
        print(f"Comparing original training from {original_dir} with stable training from {stable_dir}")
        compare_rewards(original_dir, stable_dir)
    else:
        print("Could not find training directories for comparison.") 