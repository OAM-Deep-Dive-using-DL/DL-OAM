#!/usr/bin/env python3
"""
State dimension validation utilities.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces


class StateDimensionValidator:
    """
    Validates state dimensions across configuration and environment.
    
    This class ensures that state dimensions are consistent between
    configuration files and environment implementations, preventing
    runtime errors due to dimension mismatches.
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.expected_state_components = [
            'SINR',           # Signal-to-Interference-plus-Noise Ratio
            'distance',        # Distance from base station
            'velocity_x',      # X-component of velocity
            'velocity_y',      # Y-component of velocity
            'velocity_z',      # Z-component of velocity
            'current_mode',    # Current OAM mode
            'min_mode',        # Minimum possible OAM mode
            'max_mode'         # Maximum possible OAM mode
        ]
        self.expected_state_dim = len(self.expected_state_components)
    
    def validate_environment_state_space(self, env) -> Tuple[bool, str]:
        """
        Validate that environment has correct state space.
        
        Args:
            env: Gymnasium environment
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check observation space type
            if not isinstance(env.observation_space, spaces.Box):
                return False, f"Environment observation space must be Box, got {type(env.observation_space)}"
            
            # Check state dimension
            state_dim = env.observation_space.shape[0]
            if state_dim != self.expected_state_dim:
                return False, f"Expected state dimension {self.expected_state_dim}, got {state_dim}"
            
            # Check that observation space has correct bounds
            low = env.observation_space.low
            high = env.observation_space.high
            
            if len(low) != self.expected_state_dim or len(high) != self.expected_state_dim:
                return False, f"State bounds have wrong dimension: low={len(low)}, high={len(high)}"
            
            # Validate specific bounds for each component
            validation_errors = []
            
            # SINR bounds (index 0)
            if not (-100.0 <= low[0] <= high[0] <= 100.0):
                validation_errors.append(f"SINR bounds invalid: [{low[0]}, {high[0]}]")
            
            # Distance bounds (index 1)
            if not (0.0 <= low[1] <= high[1]):
                validation_errors.append(f"Distance bounds invalid: [{low[1]}, {high[1]}]")
            
            # Velocity bounds (indices 2-4)
            for i in range(2, 5):
                if not (low[i] <= 0 <= high[i]):
                    validation_errors.append(f"Velocity bounds invalid for component {i}: [{low[i]}, {high[i]}]")
            
            # Mode bounds (indices 5-7)
            for i in range(5, 8):
                if not (1 <= low[i] <= high[i]):
                    validation_errors.append(f"Mode bounds invalid for component {i}: [{low[i]}, {high[i]}]")
            
            if validation_errors:
                return False, f"State bounds validation failed: {'; '.join(validation_errors)}"
            
            return True, "State space validation passed"
            
        except Exception as e:
            return False, f"State space validation error: {e}"
    
    def validate_config_state_dimension(self, config: Dict[str, Any], env_state_dim: int) -> Tuple[bool, str]:
        """
        Validate that config state dimension matches environment.
        
        Args:
            config: Configuration dictionary
            env_state_dim: State dimension from environment
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if config has explicit state_dim
            if 'network' in config and 'state_dim' in config['network']:
                config_state_dim = config['network']['state_dim']
                if config_state_dim != env_state_dim:
                    return False, f"Config state_dim ({config_state_dim}) != environment state_dim ({env_state_dim})"
            
            # Check if config has action_dim
            if 'network' in config and 'action_dim' in config['network']:
                config_action_dim = config['network']['action_dim']
                if config_action_dim != 3:  # Expected: STAY, UP, DOWN
                    return False, f"Config action_dim ({config_action_dim}) != expected (3)"
            
            return True, "Config state dimension validation passed"
            
        except Exception as e:
            return False, f"Config validation error: {e}"
    
    def get_state_component_description(self) -> str:
        """
        Get description of expected state components.
        
        Returns:
            Description of state components
        """
        description = "Expected state components:\n"
        for i, component in enumerate(self.expected_state_components):
            description += f"  {i}: {component}\n"
        return description
    
    def validate_agent_state_dimension(self, agent, env_state_dim: int) -> Tuple[bool, str]:
        """
        Validate that agent state dimension matches environment.
        
        Args:
            agent: RL agent
            env_state_dim: State dimension from environment
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check agent's state dimension
            if hasattr(agent, 'state_dim'):
                agent_state_dim = agent.state_dim
                if agent_state_dim != env_state_dim:
                    return False, f"Agent state_dim ({agent_state_dim}) != environment state_dim ({env_state_dim})"
            
            # Check policy network input dimension
            if hasattr(agent, 'policy_net'):
                policy_input_dim = agent.policy_net.model[0].in_features
                if policy_input_dim != env_state_dim:
                    return False, f"Policy network input dimension ({policy_input_dim}) != environment state_dim ({env_state_dim})"
            
            return True, "Agent state dimension validation passed"
            
        except Exception as e:
            return False, f"Agent validation error: {e}"
    
    def auto_detect_state_dimension(self, env) -> int:
        """
        Automatically detect state dimension from environment.
        
        Args:
            env: Gymnasium environment
            
        Returns:
            State dimension
        """
        try:
            return env.observation_space.shape[0]
        except Exception as e:
            raise ValueError(f"Cannot detect state dimension from environment: {e}")
    
    def create_state_dimension_report(self, env, config: Optional[Dict[str, Any]] = None, agent=None) -> str:
        """
        Create a comprehensive report of state dimensions.
        
        Args:
            env: Gymnasium environment
            config: Configuration dictionary (optional)
            agent: RL agent (optional)
            
        Returns:
            State dimension report
        """
        report = "📊 STATE DIMENSION VALIDATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Environment validation
        env_valid, env_error = self.validate_environment_state_space(env)
        env_state_dim = self.auto_detect_state_dimension(env)
        
        report += f"🌍 ENVIRONMENT:\n"
        report += f"   State dimension: {env_state_dim}\n"
        report += f"   Valid: {'✅' if env_valid else '❌'}\n"
        if not env_valid:
            report += f"   Error: {env_error}\n"
        report += "\n"
        
        # Config validation
        if config:
            config_valid, config_error = self.validate_config_state_dimension(config, env_state_dim)
            report += f"⚙️  CONFIGURATION:\n"
            if 'network' in config and 'state_dim' in config['network']:
                report += f"   Config state_dim: {config['network']['state_dim']}\n"
            else:
                report += f"   Config state_dim: (auto-detected)\n"
            report += f"   Valid: {'✅' if config_valid else '❌'}\n"
            if not config_valid:
                report += f"   Error: {config_error}\n"
            report += "\n"
        
        # Agent validation
        if agent:
            agent_valid, agent_error = self.validate_agent_state_dimension(agent, env_state_dim)
            report += f"🤖 AGENT:\n"
            if hasattr(agent, 'state_dim'):
                report += f"   Agent state_dim: {agent.state_dim}\n"
            if hasattr(agent, 'policy_net'):
                policy_input_dim = agent.policy_net.model[0].in_features
                report += f"   Policy input dim: {policy_input_dim}\n"
            report += f"   Valid: {'✅' if agent_valid else '❌'}\n"
            if not agent_valid:
                report += f"   Error: {agent_error}\n"
            report += "\n"
        
        # State component description
        report += f"📋 STATE COMPONENTS:\n"
        report += self.get_state_component_description()
        
        # Overall validation
        all_valid = env_valid
        if config:
            all_valid = all_valid and config_valid
        if agent:
            all_valid = all_valid and agent_valid
        
        report += f"\n🎯 OVERALL VALIDATION: {'✅ PASSED' if all_valid else '❌ FAILED'}\n"
        
        return report


def validate_state_dimensions(env, config: Optional[Dict[str, Any]] = None, agent=None) -> bool:
    """
    Convenience function to validate state dimensions.
    
    Args:
        env: Gymnasium environment
        config: Configuration dictionary (optional)
        agent: RL agent (optional)
        
    Returns:
        True if all validations pass, False otherwise
    """
    validator = StateDimensionValidator()
    
    # Environment validation
    env_valid, _ = validator.validate_environment_state_space(env)
    if not env_valid:
        return False
    
    # Config validation
    if config:
        config_valid, _ = validator.validate_config_state_dimension(config, validator.auto_detect_state_dimension(env))
        if not config_valid:
            return False
    
    # Agent validation
    if agent:
        agent_valid, _ = validator.validate_agent_state_dimension(agent, validator.auto_detect_state_dimension(env))
        if not agent_valid:
            return False
    
    return True


def get_state_dimension_report(env, config: Optional[Dict[str, Any]] = None, agent=None) -> str:
    """
    Convenience function to get state dimension report.
    
    Args:
        env: Gymnasium environment
        config: Configuration dictionary (optional)
        agent: RL agent (optional)
        
    Returns:
        State dimension report
    """
    validator = StateDimensionValidator()
    return validator.create_state_dimension_report(env, config, agent) 