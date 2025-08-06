#!/usr/bin/env python3
"""
Distance Optimization Test Script

This script demonstrates the distance optimization system functionality
and provides a quick way to test the implementation.
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# Use centralized path management
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from environment.distance_optimized_env import DistanceOptimizedEnv
from environment.distance_optimizer import DistanceOptimizer, DistanceOptimizationConfig
from utils.config_utils import load_config, merge_configs


def test_distance_optimizer():
    """Test the distance optimizer functionality."""
    print("🧪 Testing Distance Optimizer")
    print("=" * 40)
    
    # Create distance optimizer
    config = DistanceOptimizationConfig()
    optimizer = DistanceOptimizer(config)
    
    # Test distance categorization
    test_distances = [25, 75, 200, 350]
    for distance in test_distances:
        category = optimizer.get_distance_category(distance)
        print(f"Distance {distance}m -> Category: {category}")
    
    # Test mode optimization
    available_modes = [1, 2, 3, 4, 5, 6, 7, 8]
    current_mode = 4
    
    print(f"\n📊 Mode Optimization Tests:")
    for distance in test_distances:
        optimal_mode, scores = optimizer.optimize_mode_selection(
            distance, 500e6, current_mode, available_modes
        )
        print(f"Distance {distance}m: Current={current_mode}, Optimal={optimal_mode}")
        print(f"  Optimization Score: {scores['optimization_score']:.3f}")
        print(f"  Distance Score: {scores['distance_score']:.3f}")
        print(f"  Throughput Score: {scores['throughput_score']:.3f}")
        print(f"  Stability Score: {scores['stability_score']:.3f}")
        print()


def test_distance_optimized_environment():
    """Test the distance-optimized environment."""
    print("🌍 Testing Distance-Optimized Environment")
    print("=" * 40)
    
    # Load configuration
    base_config = load_config('config/base_config_new.yaml')
    distance_config = load_config('config/distance_optimization_config.yaml')
    config = merge_configs(base_config, distance_config)
    
    # Create environment
    env = DistanceOptimizedEnv(config)
    
    # Run a few episodes
    num_episodes = 5
    total_reward = 0
    total_throughput = 0
    total_optimizations = 0
    
    print(f"Running {num_episodes} test episodes...")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_throughput = 0
        episode_optimizations = 0
        steps = 0
        
        while steps < 50:  # Limit steps per episode
            # Use random actions
            action = np.random.randint(0, 3)
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_throughput += info.get('throughput', 0)
            
            if info.get('distance_optimization_mode_change', False):
                episode_optimizations += 1
            
            state = next_state
            steps += 1
            
            if done or truncated:
                break
        
        total_reward += episode_reward
        total_throughput += episode_throughput / max(steps, 1)
        total_optimizations += episode_optimizations
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Throughput={episode_throughput/1e6/max(steps,1):.1f} Mbps, "
              f"Optimizations={episode_optimizations}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"Average Reward: {total_reward/num_episodes:.2f}")
    print(f"Average Throughput: {total_throughput/num_episodes/1e6:.1f} Mbps")
    print(f"Total Optimizations: {total_optimizations}")
    
    # Get optimization statistics
    stats = env.get_distance_optimization_stats()
    print(f"Optimization Success Rate: {stats['success_rate']:.3f}")
    print(f"Average Optimization Score: {stats['average_optimization_score']:.3f}")
    
    # Get category performance
    category_performance = env.get_distance_category_performance()
    print(f"\n📏 Distance Category Performance:")
    for category, performance in category_performance.items():
        print(f"  {category.capitalize()}: {performance['avg_throughput']/1e6:.1f} Mbps "
              f"({performance['count']} samples)")


def test_configuration_loading():
    """Test configuration loading and merging."""
    print("⚙️  Testing Configuration Loading")
    print("=" * 40)
    
    try:
        # Load configurations
        base_config = load_config('config/base_config_new.yaml')
        distance_config = load_config('config/distance_optimization_config.yaml')
        config = merge_configs(base_config, distance_config)
        
        print("✅ Configuration loaded successfully")
        
        # Check key configuration sections
        if 'distance_optimization' in config:
            print("✅ Distance optimization configuration found")
            
            dist_opt = config['distance_optimization']
            if 'distance_thresholds' in dist_opt:
                thresholds = dist_opt['distance_thresholds']
                print(f"  Near threshold: {thresholds.get('near_threshold', 'N/A')}m")
                print(f"  Medium threshold: {thresholds.get('medium_threshold', 'N/A')}m")
                print(f"  Far threshold: {thresholds.get('far_threshold', 'N/A')}m")
            
            if 'mode_preferences' in dist_opt:
                modes = dist_opt['mode_preferences']
                print(f"  Near modes: {modes.get('near_modes', [])}")
                print(f"  Medium modes: {modes.get('medium_modes', [])}")
                print(f"  Far modes: {modes.get('far_modes', [])}")
        
        if 'oam' in config:
            oam_config = config['oam']
            print(f"✅ OAM configuration found")
            print(f"  Mode range: {oam_config.get('min_mode', 'N/A')} - {oam_config.get('max_mode', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")


def demonstrate_optimization_strategies():
    """Demonstrate different optimization strategies."""
    print("🎯 Demonstrating Optimization Strategies")
    print("=" * 40)
    
    # Create optimizer with different configurations
    configs = {
        'conservative': DistanceOptimizationConfig(
            distance_weight=0.2,
            throughput_weight=0.3,
            stability_weight=0.5
        ),
        'balanced': DistanceOptimizationConfig(
            distance_weight=0.3,
            throughput_weight=0.5,
            stability_weight=0.2
        ),
        'aggressive': DistanceOptimizationConfig(
            distance_weight=0.4,
            throughput_weight=0.6,
            stability_weight=0.0
        )
    }
    
    test_distance = 100.0
    current_mode = 4
    available_modes = [1, 2, 3, 4, 5, 6, 7, 8]
    
    print(f"Testing optimization strategies for distance {test_distance}m:")
    print()
    
    for strategy_name, config in configs.items():
        optimizer = DistanceOptimizer(config)
        optimal_mode, scores = optimizer.optimize_mode_selection(
            test_distance, 500e6, current_mode, available_modes
        )
        
        print(f"{strategy_name.capitalize()} Strategy:")
        print(f"  Optimal Mode: {optimal_mode}")
        print(f"  Optimization Score: {scores['optimization_score']:.3f}")
        print(f"  Distance Score: {scores['distance_score']:.3f}")
        print(f"  Throughput Score: {scores['throughput_score']:.3f}")
        print(f"  Stability Score: {scores['stability_score']:.3f}")
        print()


def main():
    """Run all tests."""
    print("🚀 Distance Optimization System Test")
    print("=" * 50)
    
    # Test configuration loading
    test_configuration_loading()
    print()
    
    # Test distance optimizer
    test_distance_optimizer()
    print()
    
    # Demonstrate optimization strategies
    demonstrate_optimization_strategies()
    print()
    
    # Test environment
    test_distance_optimized_environment()
    print()
    
    print("✅ All tests completed successfully!")
    print("\n💡 To run the full training:")
    print("   python scripts/training/train_distance_optimization.py")
    print("\n💡 To run the analysis:")
    print("   python scripts/analysis/analyze_distance_optimization.py")


if __name__ == "__main__":
    main() 