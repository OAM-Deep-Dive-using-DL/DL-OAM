#!/usr/bin/env python3
"""
Performance test for the channel simulator.
"""

import time
import numpy as np
import cProfile
import pstats
from pstats import SortKey
from simulator.channel_simulator import ChannelSimulator

def test_simulator_performance():
    """Test the performance of the channel simulator."""
    # Create simulator
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
    simulator = ChannelSimulator(config)
    
    # Test parameters
    num_iterations = 1000
    user_positions = np.random.uniform(10, 500, (num_iterations, 3))
    oam_modes = np.random.randint(1, 9, num_iterations)
    
    # Warm-up
    for _ in range(10):
        simulator.run_step(user_positions[0], oam_modes[0])
    
    # Time the run_step method
    start_time = time.time()
    for i in range(num_iterations):
        simulator.run_step(user_positions[i], oam_modes[i])
    end_time = time.time()
    
    # Calculate statistics
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    
    print(f"Total time for {num_iterations} iterations: {total_time:.3f} seconds")
    print(f"Average time per iteration: {avg_time_ms:.3f} ms")
    
    return simulator, user_positions, oam_modes

def profile_simulator():
    """Profile the simulator to identify bottlenecks."""
    # Create simulator
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
    simulator = ChannelSimulator(config)
    
    # Test parameters
    num_iterations = 100
    user_positions = np.random.uniform(10, 500, (num_iterations, 3))
    oam_modes = np.random.randint(1, 9, num_iterations)
    
    # Profile the run_step method
    cProfile.runctx('for i in range(num_iterations): simulator.run_step(user_positions[i], oam_modes[i])',
                   globals(), locals(), 'simulator_stats')
    
    # Print results
    p = pstats.Stats('simulator_stats')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)
    
    return p

if __name__ == "__main__":
    print("Running performance test...")
    test_simulator_performance()
    
    print("\nRunning profiler...")
    profile_simulator()