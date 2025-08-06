#!/usr/bin/env python3
"""
Test script to compare the performance of the original and optimized channel simulators.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from simulator.channel_simulator import ChannelSimulator
from simulator.optimized_channel_simulator import OptimizedChannelSimulator

def test_simulator_performance(num_iterations=1000):
    """
    Test and compare the performance of the original and optimized simulators.
    
    Args:
        num_iterations: Number of iterations to run
        
    Returns:
        Dictionary with performance results
    """
    # Create configuration
    config = {
        'system': {
            'frequency': 28.0e9,
            'bandwidth': 100.0e6,
            'tx_power_dBm': 30.0,
            'noise_figure_dB': 5.0,
            'noise_temp': 290.0,
            'antenna_efficiency': 0.75,
            'implementation_loss_dB': 3.0,
        },
        'oam': {
            'min_mode': 1,
            'max_mode': 8,
            'beam_width': 0.03,
        },
        'environment': {
            'humidity': 50.0,
            'temperature': 20.0,
            'pressure': 101.325,  # 101.325 kPa (standard atmospheric pressure)
            'turbulence_strength': 1e-14,
            'rician_k_factor': 10.0,
            'pointing_error_sigma': 0.01,
        }
    }
    
    # Create simulators
    original_simulator = ChannelSimulator(config)
    optimized_simulator = OptimizedChannelSimulator(config)
    
    # Generate test data
    np.random.seed(42)  # For reproducibility
    user_positions = np.random.uniform(10, 500, (num_iterations, 3))
    oam_modes = np.random.randint(1, 9, num_iterations)
    
    # Warm-up
    print("Warming up simulators...")
    for _ in range(10):
        original_simulator.run_step(user_positions[0], oam_modes[0])
        optimized_simulator.run_step(user_positions[0], oam_modes[0])
    
    # Test original simulator
    print(f"Testing original simulator ({num_iterations} iterations)...")
    original_start_time = time.time()
    original_results = []
    for i in range(num_iterations):
        H, sinr = original_simulator.run_step(user_positions[i], oam_modes[i])
        original_results.append((H, sinr))
    original_end_time = time.time()
    original_total_time = original_end_time - original_start_time
    original_avg_time_ms = (original_total_time / num_iterations) * 1000
    
    # Test optimized simulator
    print(f"Testing optimized simulator ({num_iterations} iterations)...")
    optimized_start_time = time.time()
    optimized_results = []
    for i in range(num_iterations):
        H, sinr = optimized_simulator.run_step(user_positions[i], oam_modes[i])
        optimized_results.append((H, sinr))
    optimized_end_time = time.time()
    optimized_total_time = optimized_end_time - optimized_start_time
    optimized_avg_time_ms = (optimized_total_time / num_iterations) * 1000
    
    # Calculate speedup
    speedup = original_total_time / optimized_total_time
    
    # Check result consistency
    consistency_check = []
    for i in range(num_iterations):
        original_H, original_sinr = original_results[i]
        optimized_H, optimized_sinr = optimized_results[i]
        
        # Check SINR consistency (allow small numerical differences)
        sinr_diff = abs(original_sinr - optimized_sinr)
        consistency_check.append(sinr_diff < 1.0)  # Allow 1 dB difference
    
    consistency_percent = (sum(consistency_check) / len(consistency_check)) * 100
    
    # Get detailed performance metrics from optimized simulator
    perf_metrics = optimized_simulator.get_performance_metrics()
    
    # Print results
    print("\n===== PERFORMANCE COMPARISON =====")
    print(f"Original Simulator: {original_avg_time_ms:.3f} ms per iteration")
    print(f"Optimized Simulator: {optimized_avg_time_ms:.3f} ms per iteration")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Result Consistency: {consistency_percent:.1f}%")
    
    print("\n===== DETAILED METRICS =====")
    print(f"Cache Statistics:")
    print(f"  - Phase Structure Cache Size: {perf_metrics.get('phase_structure_cache_size', 0)}")
    print(f"  - Step Cache Size: {perf_metrics.get('step_cache_size', 0)}")
    print(f"  - Turbulence Cache Size: {perf_metrics.get('turbulence_cache_size', 0)}")
    print(f"  - Total Cache Size: {perf_metrics.get('total_cache_size', 0)}")
    
    print(f"Component Times:")
    print(f"  - Turbulence Time: {perf_metrics.get('avg_turbulence_time_ms', 0.0):.3f} ms ({perf_metrics.get('turbulence_percent', 0.0):.1f}%)")
    print(f"  - Crosstalk Time: {perf_metrics.get('avg_crosstalk_time_ms', 0.0):.3f} ms ({perf_metrics.get('crosstalk_percent', 0.0):.1f}%)")
    print(f"  - Path Loss Time: {perf_metrics.get('avg_path_loss_time_ms', 0.0):.3f} ms ({perf_metrics.get('path_loss_percent', 0.0):.1f}%)")
    print(f"  - Fading Time: {perf_metrics.get('avg_fading_time_ms', 0.0):.3f} ms ({perf_metrics.get('fading_percent', 0.0):.1f}%)")
    print(f"  - SINR Calculation Time: {perf_metrics.get('avg_sinr_time_ms', 0.0):.3f} ms ({perf_metrics.get('sinr_percent', 0.0):.1f}%)")
    
    # Create performance comparison plot
    create_performance_plot(original_avg_time_ms, optimized_avg_time_ms, perf_metrics)
    
    # Return results
    return {
        'original_avg_time_ms': original_avg_time_ms,
        'optimized_avg_time_ms': optimized_avg_time_ms,
        'speedup': speedup,
        'consistency_percent': consistency_percent,
        'perf_metrics': perf_metrics
    }

def create_performance_plot(original_avg_time_ms, optimized_avg_time_ms, perf_metrics):
    """
    Create performance comparison plots.
    
    Args:
        original_avg_time_ms: Average time for original simulator (ms)
        optimized_avg_time_ms: Average time for optimized simulator (ms)
        perf_metrics: Detailed performance metrics
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Overall comparison
    simulators = ['Original', 'Optimized']
    times = [original_avg_time_ms, optimized_avg_time_ms]
    
    bars = ax1.bar(simulators, times, color=['#3498db', '#2ecc71'])
    ax1.set_ylabel('Average Time per Step (ms)')
    ax1.set_title('Simulator Performance Comparison')
    
    # Add speedup text
    speedup = original_avg_time_ms / optimized_avg_time_ms
    ax1.text(0.5, 0.9, f'Speedup: {speedup:.2f}x', 
             horizontalalignment='center',
             transform=ax1.transAxes,
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add time values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{times[i]:.3f} ms',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Breakdown of optimized simulator time
    components = ['Turbulence', 'Crosstalk', 'Path Loss', 'Fading', 'SINR Calc']
    component_times = [
        perf_metrics.get('avg_turbulence_time_ms', 0.0),
        perf_metrics.get('avg_crosstalk_time_ms', 0.0),
        perf_metrics.get('avg_path_loss_time_ms', 0.0),
        perf_metrics.get('avg_fading_time_ms', 0.0),
        perf_metrics.get('avg_sinr_time_ms', 0.0)
    ]
    
    # Sort by time (descending)
    sorted_indices = np.argsort(component_times)[::-1]
    sorted_components = [components[i] for i in sorted_indices]
    sorted_times = [component_times[i] for i in sorted_indices]
    
    # Calculate percentages
    total_time = sum(component_times)
    percentages = [100 * t / total_time for t in sorted_times]
    
    # Create bars with custom colors
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    sorted_colors = [colors[i] for i in sorted_indices]
    
    bars = ax2.bar(sorted_components, sorted_times, color=sorted_colors)
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Optimized Simulator Component Breakdown')
    
    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{sorted_times[i]:.3f} ms ({percentages[i]:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('simulator_performance_comparison.png', dpi=300)
    print("Performance comparison plot saved as 'simulator_performance_comparison.png'")
    
    # Create additional plot for cache effectiveness
    create_cache_effectiveness_plot(perf_metrics)

def create_cache_effectiveness_plot(perf_metrics):
    """
    Create a plot showing cache effectiveness.
    
    Args:
        perf_metrics: Detailed performance metrics
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create pie chart of time distribution
    labels = ['Turbulence', 'Crosstalk', 'Path Loss', 'Fading', 'SINR Calc', 'Other']
    turbulence_pct = perf_metrics.get('turbulence_percent', 0.0)
    crosstalk_pct = perf_metrics.get('crosstalk_percent', 0.0)
    path_loss_pct = perf_metrics.get('path_loss_percent', 0.0)
    fading_pct = perf_metrics.get('fading_percent', 0.0)
    sinr_pct = perf_metrics.get('sinr_percent', 0.0)
    
    sizes = [
        turbulence_pct,
        crosstalk_pct,
        path_loss_pct,
        fading_pct,
        sinr_pct,
        max(0.0, 100.0 - (turbulence_pct + crosstalk_pct + path_loss_pct + fading_pct + sinr_pct))
    ]
    
    # Only include components with > 1% contribution
    filtered_labels = []
    filtered_sizes = []
    for label, size in zip(labels, sizes):
        if size > 1.0:
            filtered_labels.append(label)
            filtered_sizes.append(size)
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6']
    filtered_colors = colors[:len(filtered_labels)]
    
    plt.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
            autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Optimized Simulator Time Distribution')
    
    # Add cache size annotation
    cache_size = perf_metrics['phase_structure_cache_size']
    plt.annotate(f'Phase Structure Cache Size: {cache_size}', 
                 xy=(0.5, 0.02),
                 xycoords='figure fraction',
                 ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save figure
    plt.savefig('simulator_time_distribution.png', dpi=300)
    print("Time distribution plot saved as 'simulator_time_distribution.png'")

if __name__ == "__main__":
    # Run with different iteration counts to see scaling
    for iterations in [100, 1000]:
        print(f"\n\n===== TESTING WITH {iterations} ITERATIONS =====")
        test_simulator_performance(iterations)