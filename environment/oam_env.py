import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional
import math
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.channel_simulator import ChannelSimulator


class OAM_Env(gym.Env):
    """
    Gymnasium environment for OAM mode handover decisions.
    
    This environment simulates a user moving in a wireless network with OAM-based
    transmission. The agent must decide when to switch OAM modes to maximize throughput
    while minimizing unnecessary handovers.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OAM environment.
        
        Args:
            config: Dictionary containing environment parameters
        """
        super(OAM_Env, self).__init__()
        
        # Default parameters
        self.distance_min = 50.0  # meters
        self.distance_max = 300.0  # meters
        self.velocity_min = 1.0  # m/s
        self.velocity_max = 5.0  # m/s
        self.area_size = np.array([500.0, 500.0])  # meters [x, y]
        self.pause_time_max = 5.0  # seconds
        self.min_mode = 1
        self.max_mode = 8
        
        # Reward function parameters
        self.throughput_factor = 1.0
        self.handover_penalty = 0.5
        self.outage_penalty = 10.0
        self.sinr_threshold = 0.0  # dB
        
        # Time step in seconds
        self.dt = 0.1
        
        # Update with provided config
        if config:
            self._update_config(config)
        
        # Validate parameters
        self._validate_environment_parameters()
        
        # Initialize the channel simulator
        self.simulator = ChannelSimulator(config)
        
        # Action space: 0 = STAY, 1 = UP, 2 = DOWN
        self.action_space = spaces.Discrete(3)
        
        # State space: [SINR, distance, velocity_x, velocity_y, velocity_z, current_mode, min_mode, max_mode]
        low = np.array(
            [-30.0,                 # Minimum SINR in dB
             self.distance_min,     # Minimum distance
             -self.velocity_max,    # Minimum velocity in x
             -self.velocity_max,    # Minimum velocity in y
             -self.velocity_max,    # Minimum velocity in z
             self.min_mode,         # Minimum OAM mode
             self.min_mode,         # Minimum possible mode
             self.min_mode],        # Minimum of max mode (doesn't change)
            dtype=np.float32
        )
        
        high = np.array(
            [50.0,                  # Maximum SINR in dB
             self.distance_max,     # Maximum distance
             self.velocity_max,     # Maximum velocity in x
             self.velocity_max,     # Maximum velocity in y
             self.velocity_max,     # Maximum velocity in z
             self.max_mode,         # Maximum OAM mode
             self.max_mode,         # Maximum of min mode (doesn't change)
             self.max_mode],        # Maximum possible mode
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Initialize state variables
        self.position = None
        self.velocity = None
        self.current_mode = None
        self.current_sinr = None
        self.target_position = None
        self.pause_time = 0
        
        # Episode tracking
        self.steps = 0
        self.max_steps = 1000
        self.episode_throughput = 0
        self.episode_handovers = 0
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update environment parameters from configuration.
        
        Args:
            config: Dictionary containing environment parameters
        """
        if 'environment' in config:
            env_config = config['environment']
            self.distance_min = env_config.get('distance_min', self.distance_min)
            self.distance_max = env_config.get('distance_max', self.distance_max)
        
        if 'mobility' in config:
            mob_config = config['mobility']
            self.velocity_min = mob_config.get('velocity_min', self.velocity_min)
            self.velocity_max = mob_config.get('velocity_max', self.velocity_max)
            self.area_size = np.array(mob_config.get('area_size', self.area_size))
            self.pause_time_max = mob_config.get('pause_time_max', self.pause_time_max)
        
        if 'oam' in config:
            oam_config = config['oam']
            self.min_mode = oam_config.get('min_mode', self.min_mode)
            self.max_mode = oam_config.get('max_mode', self.max_mode)
        
        if 'reward' in config:
            reward_config = config['reward']
            self.throughput_factor = reward_config.get('throughput_factor', self.throughput_factor)
            self.handover_penalty = reward_config.get('handover_penalty', self.handover_penalty)
            self.outage_penalty = reward_config.get('outage_penalty', self.outage_penalty)
            self.sinr_threshold = reward_config.get('sinr_threshold', self.sinr_threshold)
    
    def _validate_environment_parameters(self):
        """Validate environment parameters are within reasonable ranges."""
        
        # Distance validation
        if not (1.0 <= self.distance_min < self.distance_max <= 100000):
            raise ValueError(f"Distance range [{self.distance_min}, {self.distance_max}] is invalid")
        
        # Velocity validation
        if not (0.1 <= self.velocity_min < self.velocity_max <= 100):
            raise ValueError(f"Velocity range [{self.velocity_min}, {self.velocity_max}] m/s is invalid")
        
        # Area validation
        if not all(10.0 <= size <= 100000 for size in self.area_size):
            raise ValueError(f"Area size {self.area_size} is outside reasonable range")
        
        # Time step validation
        if not (0.01 <= self.dt <= 10.0):
            raise ValueError(f"Time step {self.dt} s is outside reasonable range (0.01-10.0 s)")
        
        # Reward parameters validation
        if not (0.1 <= self.throughput_factor <= 10.0):
            raise ValueError(f"Throughput factor {self.throughput_factor} is outside reasonable range")
        
        if not (0.0 <= self.handover_penalty <= 100.0):
            raise ValueError(f"Handover penalty {self.handover_penalty} is outside reasonable range")
        
        if not (0.0 <= self.outage_penalty <= 1000.0):
            raise ValueError(f"Outage penalty {self.outage_penalty} is outside reasonable range")
        
        print("✅ Environment parameters validated successfully")

    def _generate_random_position(self) -> np.ndarray:
        """
        Generate a random position within the area bounds.
        
        Returns:
            3D position array [x, y, z]
        """
        x = np.random.uniform(0, self.area_size[0])
        y = np.random.uniform(0, self.area_size[1])
        
        # Z coordinate is set to ensure the distance is within bounds
        # We'll set a random distance first
        distance = np.random.uniform(self.distance_min, self.distance_max)
        
        # Calculate z based on x, y, and desired distance
        # Using Pythagorean theorem: x^2 + y^2 + z^2 = distance^2
        xy_dist = np.sqrt(x**2 + y**2)
        if xy_dist > distance:
            # If x,y already exceeds the desired distance, set z=0 and normalize x,y
            scale = distance / xy_dist
            x *= scale
            y *= scale
            z = 0
        else:
            # Otherwise, calculate z to achieve the desired distance
            z = np.sqrt(distance**2 - xy_dist**2)
        
        return np.array([x, y, z])
    
    def _generate_random_velocity(self) -> np.ndarray:
        """
        Generate a random velocity vector.
        
        Returns:
            3D velocity vector [vx, vy, vz]
        """
        # Generate random direction (unit vector)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Generate random speed
        speed = np.random.uniform(self.velocity_min, self.velocity_max)
        
        # Return velocity vector
        return speed * direction
    
    def _update_position(self) -> None:
        """Update the user's position using the Random Waypoint mobility model."""
        if self.pause_time > 0:
            # User is paused at current position
            self.pause_time -= self.dt
        else:
            # Check if user has reached the target
            dist_to_target = np.linalg.norm(self.position - self.target_position)
            
            if dist_to_target < self.velocity_max * self.dt:
                # User has reached the target, generate a new one and maybe pause
                self.position = self.target_position
                self.target_position = self._generate_random_position()
                self.velocity = self._generate_random_velocity()
                
                # Randomly decide whether to pause
                if np.random.random() < 0.3:  # 30% chance to pause
                    self.pause_time = np.random.uniform(0, self.pause_time_max)
            else:
                # Continue moving towards target
                direction = (self.target_position - self.position)
                direction = direction / np.linalg.norm(direction)
                
                # Update velocity (direction * speed)
                speed = np.linalg.norm(self.velocity)
                self.velocity = direction * speed
                
                # Update position
                self.position += self.velocity * self.dt
    
    def _calculate_throughput(self, sinr_dB: float) -> float:
        """
        Calculate throughput using Shannon's formula with enhanced error handling.
        
        Args:
            sinr_dB: Signal-to-Interference-plus-Noise Ratio in dB
            
        Returns:
            Throughput in bits per second
        """
        # Input validation
        if not isinstance(sinr_dB, (int, float)):
            return 0.0
            
        # Handle NaN or infinity
        if np.isnan(sinr_dB) or np.isinf(sinr_dB):
            return 0.0
        
        # Clamp SINR to reasonable bounds
        sinr_dB = max(min(sinr_dB, 60.0), -40.0)
            
        # Convert SINR from dB to linear
        sinr_linear = 10 ** (sinr_dB / 10)
        
        # Shannon's formula: C = B * log2(1 + SINR)
        # Make sure bandwidth is a float
        try:
            bandwidth = float(self.simulator.bandwidth)
            
            # Add small epsilon to avoid log(1) = 0 issues
            sinr_for_log = max(sinr_linear, 1e-10)
            throughput = bandwidth * math.log2(1 + sinr_for_log)
            
            # Validate result
            if np.isnan(throughput) or np.isinf(throughput) or throughput < 0:
                return 0.0
            
            # Cap maximum throughput to theoretical maximum
            max_throughput = bandwidth * math.log2(1 + 10**(60/10))  # 60 dB SINR max
            throughput = min(throughput, max_throughput)
            
            return throughput
            
        except (ValueError, TypeError, OverflowError):
            return 0.0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (initial state, info dictionary)
        """
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.steps = 0
        self.episode_throughput = 0
        self.episode_handovers = 0
        
        # Initialize user position and velocity
        self.position = self._generate_random_position()
        self.target_position = self._generate_random_position()
        self.velocity = self._generate_random_velocity()
        self.pause_time = 0
        
        # Initialize OAM mode (start in the middle of the range)
        self.current_mode = (self.min_mode + self.max_mode) // 2
        
        # Run simulator to get initial channel state
        _, self.current_sinr = self.simulator.run_step(self.position, self.current_mode)
        
        # Construct the state vector
        state = np.array([
            self.current_sinr,
            np.linalg.norm(self.position),  # distance
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            self.current_mode,
            self.min_mode,
            self.max_mode
        ], dtype=np.float32)
        
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'throughput': self._calculate_throughput(self.current_sinr),
            'handovers': 0
        }
        
        return state, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
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
        
        # Calculate reward
        reward = self.throughput_factor * throughput / 1e9  # Scale by 10^9 for numerical stability
        
        # Apply handover penalty if mode was changed
        if handover_occurred:
            reward -= self.handover_penalty
        
        # Apply outage penalty if SINR is below threshold
        if self.current_sinr < self.sinr_threshold:
            reward -= self.outage_penalty
            
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
            'handovers': self.episode_handovers,
            'sinr': self.current_sinr,
            'mode': self.current_mode
        }
        
        return next_state, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass 