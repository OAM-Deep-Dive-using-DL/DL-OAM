#!/usr/bin/env python3
"""
Test script to compare memory usage between original and enhanced figure generation.
"""

import os
import sys
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def run_script_and_measure_memory(script_path):
    """
    Run a Python script and measure its memory usage over time.
    
    Args:
        script_path: Path to the script to run
        
    Returns:
        Tuple of (peak_memory_mb, execution_time_s, memory_profile)
    """
    print(f"Running {script_path}...")
    
    # Start time
    start_time = time.time()
    
    # Run the script and capture output
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Memory profile data
    memory_profile = []
    peak_memory = 0
    
    # Monitor memory usage while script is running
    while process.poll() is None:
        # Get memory usage of the subprocess
        try:
            # Use ps command to get memory usage
            ps_process = subprocess.Popen(
                ['ps', '-o', 'rss=', str(process.pid)],
                stdout=subprocess.PIPE,
                universal_newlines=True
            )
            memory_str = ps_process.communicate()[0].strip()
            if memory_str:
                memory_kb = int(memory_str)
                memory_mb = memory_kb / 1024
                
                # Record timestamp and memory usage
                elapsed = time.time() - start_time
                memory_profile.append((elapsed, memory_mb))
                
                # Update peak memory
                peak_memory = max(peak_memory, memory_mb)
        except (ValueError, subprocess.SubprocessError):
            pass
        
        # Sleep briefly to avoid excessive CPU usage
        time.sleep(0.1)
    
    # Get execution time
    execution_time = time.time() - start_time
    
    # Get output
    stdout, stderr = process.communicate()
    
    # Print output
    print("Script output:")
    print(stdout)
    if stderr:
        print("Errors:")
        print(stderr)
    
    return peak_memory, execution_time, memory_profile

def plot_memory_comparison(original_profile, enhanced_profile, output_path):
    """
    Plot memory usage comparison between original and enhanced scripts.
    
    Args:
        original_profile: List of (time, memory) tuples for original script
        enhanced_profile: List of (time, memory) tuples for enhanced script
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Extract data
    original_times = [p[0] for p in original_profile]
    original_memory = [p[1] for p in original_profile]
    
    enhanced_times = [p[0] for p in enhanced_profile]
    enhanced_memory = [p[1] for p in enhanced_profile]
    
    # Plot data
    plt.plot(original_times, original_memory, 'r-', label='Original Script', linewidth=2)
    plt.plot(enhanced_times, enhanced_memory, 'g-', label='Enhanced Script', linewidth=2)
    
    # Add peak memory markers
    original_peak = max(original_memory)
    enhanced_peak = max(enhanced_memory)
    
    original_peak_time = original_times[original_memory.index(original_peak)]
    enhanced_peak_time = enhanced_times[enhanced_memory.index(enhanced_peak)]
    
    plt.plot(original_peak_time, original_peak, 'ro', markersize=8)
    plt.plot(enhanced_peak_time, enhanced_peak, 'go', markersize=8)
    
    plt.text(original_peak_time, original_peak + 5, f"{original_peak:.1f} MB", 
             color='red', fontweight='bold', ha='center')
    plt.text(enhanced_peak_time, enhanced_peak + 5, f"{enhanced_peak:.1f} MB", 
             color='green', fontweight='bold', ha='center')
    
    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison: Original vs. Enhanced')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add summary statistics
    memory_reduction = (original_peak - enhanced_peak) / original_peak * 100
    plt.figtext(0.5, 0.01, 
                f"Peak Memory Reduction: {memory_reduction:.1f}%", 
                ha='center', fontsize=12, fontweight='bold')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Memory comparison plot saved to {output_path}")

def main():
    """Main function to run the memory usage comparison."""
    # Paths to scripts
    original_script = "IEEE_images/generate_figure1_system_model.py"
    enhanced_script = "IEEE_images/generate_figure1_system_model_enhanced.py"
    
    # Check if scripts exist
    if not os.path.exists(original_script):
        print(f"Error: Original script not found at {original_script}")
        return
    
    if not os.path.exists(enhanced_script):
        print(f"Error: Enhanced script not found at {enhanced_script}")
        return
    
    # Run original script and measure memory
    original_peak, original_time, original_profile = run_script_and_measure_memory(original_script)
    print(f"Original script: Peak memory = {original_peak:.2f} MB, Time = {original_time:.2f} seconds")
    
    # Run enhanced script and measure memory
    enhanced_peak, enhanced_time, enhanced_profile = run_script_and_measure_memory(enhanced_script)
    print(f"Enhanced script: Peak memory = {enhanced_peak:.2f} MB, Time = {enhanced_time:.2f} seconds")
    
    # Calculate improvements
    memory_improvement = (original_peak - enhanced_peak) / original_peak * 100
    time_improvement = (original_time - enhanced_time) / original_time * 100
    
    print(f"\nMemory usage reduced by {memory_improvement:.1f}%")
    print(f"Execution time {time_improvement:.1f}% {'faster' if time_improvement > 0 else 'slower'}")
    
    # Plot memory comparison
    plot_memory_comparison(original_profile, enhanced_profile, "memory_usage_comparison.png")

if __name__ == "__main__":
    main()