import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional
import math
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.oam_env import OAM_Env
from simulator.channel_simulator import ChannelSimulator


class StableOAM_Env(OAM_Env):
    """
    Gymnasium environment for OAM mode handover decisions with a more stable reward function.
    
    This environment extends the base OAM_Env with modifications to reduce reward variance:
    1. Reward normalization and scaling
    2. Moving average for throughput
    3. Exponential smoothing for rewards
    4. Relative improvements over baseline
    5. Bounded reward range
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the stable OAM environment.
        
        Args:
            config: Dictionary containing environment parameters
        """
        # Initialize additional parameters
        self.reward_smoothing_factor = 0.7  # Exponential smoothing factor (0-1)
        self.throughput_window_size = 10  # Size of moving average window
        self.throughput_history = []  # Store recent throughput values
        self.baseline_throughput = None  # Baseline throughput for relative improvement
        self.previous_reward = 0.0  # Previous reward for smoothing
        self.reward_scale = 1.0  # Scaling factor for reward
        self.reward_min = -10.0  # Minimum reward value
        self.reward_max = 10.0  # Maximum reward value
        self.sinr_scaling_factor = 0.1  # Scale SINR contribution to reward
        
        # Call parent constructor
        super().__init__(config)
        
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update environment parameters from configuration.
        
        Args:
            config: Dictionary containing environment parameters
        """
        # Call parent method first
        super()._update_config(config)
        
        # Update stable reward parameters
        if 'stable_reward' in config:
            stable_config = config['stable_reward']
            self.reward_smoothing_factor = stable_config.get('smoothing_factor', self.reward_smoothing_factor)
            self.throughput_window_size = stable_config.get('window_size', self.throughput_window_size)
            self.reward_scale = stable_config.get('reward_scale', self.reward_scale)
            self.reward_min = stable_config.get('reward_min', self.reward_min)
            self.reward_max = stable_config.get('reward_max', self.reward_max)
            self.sinr_scaling_factor = stable_config.get('sinr_scaling_factor', self.sinr_scaling_factor)
    
    def _get_moving_avg_throughput(self, current_throughput: float) -> float:
        """
        Calculate moving average of throughput.
        
        Args:
            current_throughput: Current throughput value
            
        Returns:
            Moving average throughput
        """
        # Add current throughput to history
        self.throughput_history.append(current_throughput)
        
        # Keep only the most recent values
        if len(self.throughput_history) > self.throughput_window_size:
            self.throughput_history.pop(0)
        
        # Calculate moving average
        return np.mean(self.throughput_history)
    
    def _normalize_throughput(self, throughput: float) -> float:
        """
        Normalize throughput to a reasonable range.
        
        Args:
            throughput: Raw throughput value
            
        Returns:
            Normalized throughput value
        """
        # Get theoretical maximum throughput (at 60 dB SINR)
        max_throughput = self.simulator.bandwidth * math.log2(1 + 10**(60/10))
        
        # Normalize to 0-1 range
        normalized = throughput / max_throughput
        
        return normalized
    
    def _calculate_stable_reward(self, throughput: float, sinr_dB: float, handover_occurred: bool) -> float:
        """
        Calculate a more stable reward.
        
        Args:
            throughput: Current throughput value
            sinr_dB: Current SINR in dB
            handover_occurred: Whether a handover occurred
            
        Returns:
            Stable reward value
        """
        # 1. Get moving average throughput
        avg_throughput = self._get_moving_avg_throughput(throughput)
        
        # 2. Initialize baseline throughput if not set
        if self.baseline_throughput is None:
            self.baseline_throughput = avg_throughput
        
        # 3. Calculate relative improvement over baseline
        if self.baseline_throughput > 0:
            relative_improvement = (avg_throughput - self.baseline_throughput) / self.baseline_throughput
        else:
            relative_improvement = 0
        
        # 4. Normalize throughput to 0-1 range
        normalized_throughput = self._normalize_throughput(avg_throughput)
        
        # 5. Calculate reward components
        throughput_reward = self.throughput_factor * normalized_throughput
        sinr_reward = self.sinr_scaling_factor * (sinr_dB / 60.0)  # Normalize SINR to -1 to 1 range
        handover_penalty = self.handover_penalty if handover_occurred else 0
        outage_penalty = self.outage_penalty if sinr_dB < self.sinr_threshold else 0
        
        # 6. Combine components
        raw_reward = throughput_reward + sinr_reward - handover_penalty - outage_penalty
        
        # 7. Apply exponential smoothing
        smoothed_reward = (self.reward_smoothing_factor * self.previous_reward + 
                          (1 - self.reward_smoothing_factor) * raw_reward)
        
        # 8. Scale reward
        scaled_reward = smoothed_reward * self.reward_scale
        
        # 9. Clip to bounds
        final_reward = np.clip(scaled_reward, self.reward_min, self.reward_max)
        
        # 10. Update previous reward and baseline
        self.previous_reward = smoothed_reward
        
        # Gradually update baseline (slow tracking)
        self.baseline_throughput = 0.99 * self.baseline_throughput + 0.01 * avg_throughput
        
        return final_reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (initial state, info dictionary)
        """
        # Reset stable reward variables
        self.throughput_history = []
        self.baseline_throughput = None
        self.previous_reward = 0.0
        
        # Call parent reset
        return super().reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment with a more stable reward function.
        
        Args:
            action: Action to take (0: STAY, 1: UP, 2: DOWN)
            
        Returns:
            Tuple of (next state, reward, done, truncated, info)
        """
        self.steps += 1
        
        # Track the previous mode for handover detection
        prev_mode = self.current_mode
        
        # Update OAM mode based on action
        if action == 0:  # STAY
            pass  # Keep the current mode
        elif action == 1:  # UP
            self.current_mode = min(self.current_mode + 1, self.max_mode)
        elif action == 2:  # DOWN
            self.current_mode = max(self.current_mode - 1, self.min_mode)
        
        # Detect if a handover occurred
        handover_occurred = (prev_mode != self.current_mode)
        if handover_occurred:
            self.episode_handovers += 1
        
        # Update user position using mobility model
        self._update_position()
        
        # Run simulator to get new channel state
        _, self.current_sinr = self.simulator.run_step(self.position, self.current_mode)
        
        # Calculate throughput
        throughput = self._calculate_throughput(self.current_sinr)
        self.episode_throughput += throughput
        
        # Calculate stable reward
        reward = self._calculate_stable_reward(throughput, self.current_sinr, handover_occurred)
            
        # Handle NaN or infinity in reward
        if np.isnan(reward) or np.isinf(reward):
            reward = -self.outage_penalty  # Default to penalty value
        
        # Construct the next state vector
        next_state = np.array([
            self.current_sinr,
            np.linalg.norm(self.position),  # distance
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            self.current_mode,
            self.min_mode,
            self.max_mode
        ], dtype=np.float32)
        
        # Check if episode is done
        done = False
        truncated = (self.steps >= self.max_steps)
        
        # Prepare info dictionary
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'throughput': throughput,
            'avg_throughput': self._get_moving_avg_throughput(throughput),
            'handovers': self.episode_handovers,
            'sinr': self.current_sinr,
            'mode': self.current_mode,
            'raw_reward': reward
        }
        
        return next_state, reward, done, truncated, info 