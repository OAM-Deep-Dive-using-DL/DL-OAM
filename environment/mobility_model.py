#!/usr/bin/env python3
"""
Mobility model for OAM environment.

This module handles user mobility and position updates, separating them
from the RL environment logic to follow the Single Responsibility Principle.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class MobilityModel:
    """
    Handles user mobility and position updates.
    
    This class is responsible for all mobility-related calculations,
    including position updates, velocity generation, and movement patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mobility model.
        
        Args:
            config: Configuration dictionary with mobility parameters
        """
        # Default parameters
        self.velocity_min = 1.0  # m/s
        self.velocity_max = 5.0  # m/s
        self.area_size = np.array([500.0, 500.0])  # meters [x, y]
        self.pause_time_max = 5.0  # seconds
        self.dt = 0.1  # Time step in seconds
        
        # Update with provided config
        if config and 'mobility' in config:
            mob_config = config['mobility']
            self.velocity_min = mob_config.get('velocity_min', self.velocity_min)
            self.velocity_max = mob_config.get('velocity_max', self.velocity_max)
            self.area_size = np.array(mob_config.get('area_size', self.area_size))
            self.pause_time_max = mob_config.get('pause_time_max', self.pause_time_max)
        
        # Initialize state
        self.position = None
        self.velocity = None
        self.target_position = None
        self.pause_time = 0.0
        
    def generate_random_position(self) -> np.ndarray:
        """
        Generate a random position within the area bounds.
        
        Returns:
            3D position array [x, y, z]
        """
        x = np.random.uniform(0, self.area_size[0])
        y = np.random.uniform(0, self.area_size[1])
        
        # Z coordinate is set to ensure the distance is within bounds
        # We'll set a random distance first
        distance = np.random.uniform(50.0, 300.0)  # Reasonable distance range
        
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
    
    def generate_random_velocity(self) -> np.ndarray:
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
        
        return direction * speed
    
    def update_position(self) -> None:
        """
        Update user position based on current velocity and mobility model.
        """
        if self.position is None or self.velocity is None:
            return
        
        # Check if we should pause (random pause model)
        if np.random.random() < 0.1:  # 10% chance of pause
            self.pause_time = np.random.uniform(0, self.pause_time_max)
        
        if self.pause_time > 0:
            # User is paused
            self.pause_time -= self.dt
            return
        
        # Update position based on velocity
        new_position = self.position + self.velocity * self.dt
        
        # Check boundary conditions
        if new_position[0] < 0 or new_position[0] > self.area_size[0]:
            # Bounce off x boundaries
            self.velocity[0] *= -1
            new_position[0] = np.clip(new_position[0], 0, self.area_size[0])
        
        if new_position[1] < 0 or new_position[1] > self.area_size[1]:
            # Bounce off y boundaries
            self.velocity[1] *= -1
            new_position[1] = np.clip(new_position[1], 0, self.area_size[1])
        
        # Update position
        self.position = new_position
        
        # Occasionally change direction (random walk model)
        if np.random.random() < 0.05:  # 5% chance of direction change
            self.velocity = self.generate_random_velocity()
    
    def reset(self) -> None:
        """
        Reset the mobility model to initial state.
        """
        self.position = self.generate_random_position()
        self.target_position = self.generate_random_position()
        self.velocity = self.generate_random_velocity()
        self.pause_time = 0.0
    
    def get_position(self) -> np.ndarray:
        """
        Get current position.
        
        Returns:
            Current 3D position
        """
        if self.position is None:
            self.reset()
        return self.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """
        Get current velocity.
        
        Returns:
            Current 3D velocity
        """
        if self.velocity is None:
            self.reset()
        return self.velocity.copy()
    
    def get_distance(self) -> float:
        """
        Calculate distance from origin.
        
        Returns:
            Distance in meters
        """
        if self.position is None:
            return 0.0
        return float(np.linalg.norm(self.position))
    
    def validate_mobility_parameters(self) -> Tuple[bool, str]:
        """
        Validate mobility model parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Validate velocity range
        if self.velocity_min >= self.velocity_max:
            errors.append(f"Velocity range invalid: min={self.velocity_min}, max={self.velocity_max}")
        
        if self.velocity_min < 0:
            errors.append(f"Minimum velocity cannot be negative: {self.velocity_min}")
        
        if self.velocity_max > 100:
            errors.append(f"Maximum velocity too high: {self.velocity_max}")
        
        # Validate area size
        if not all(size > 0 for size in self.area_size):
            errors.append(f"Area size must be positive: {self.area_size}")
        
        if not all(size <= 10000 for size in self.area_size):
            errors.append(f"Area size too large: {self.area_size}")
        
        # Validate time parameters
        if self.dt <= 0:
            errors.append(f"Time step must be positive: {self.dt}")
        
        if self.pause_time_max < 0:
            errors.append(f"Pause time cannot be negative: {self.pause_time_max}")
        
        if errors:
            return False, "; ".join(errors)
        else:
            return True, "Mobility parameters valid"
    
    def get_mobility_info(self) -> dict:
        """
        Get information about mobility parameters.
        
        Returns:
            Dictionary with mobility parameters
        """
        return {
            'velocity_min': self.velocity_min,
            'velocity_max': self.velocity_max,
            'area_size': self.area_size.tolist(),
            'pause_time_max': self.pause_time_max,
            'dt': self.dt,
            'current_position': self.position.tolist() if self.position is not None else None,
            'current_velocity': self.velocity.tolist() if self.velocity is not None else None
        } 