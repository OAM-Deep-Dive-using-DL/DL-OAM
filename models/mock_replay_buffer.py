#!/usr/bin/env python3
"""
Mock replay buffer for testing.

This module provides mock implementations of replay buffers for testing
the agent with dependency injection.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Any
from models.replay_buffer_interface import ReplayBufferInterface


class MockReplayBuffer(ReplayBufferInterface):
    """
    Mock replay buffer for testing.
    
    This implementation provides predictable behavior for testing
    the agent's interaction with replay buffers.
    """
    
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        """
        Initialize the mock replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            device: PyTorch device to use for tensor operations
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        self.buffer = []
        self.push_count = 0
        self.sample_count = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        self.push_count += 1
        
        # Keep only the most recent transitions up to capacity
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        self.sample_count += 1
        
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        if batch_size == 0:
            # Return empty tensors if buffer is empty
            empty_states = torch.FloatTensor([]).to(self.device)
            empty_actions = torch.LongTensor([]).to(self.device)
            empty_rewards = torch.FloatTensor([]).to(self.device)
            empty_next_states = torch.FloatTensor([]).to(self.device)
            empty_dones = torch.FloatTensor([]).to(self.device)
            return empty_states, empty_actions, empty_rewards, empty_next_states, empty_dones
        
        # Sample random indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Get the sampled transitions
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self) >= batch_size
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer.clear()
    
    def get_info(self) -> dict:
        """
        Get information about the buffer.
        
        Returns:
            Dictionary with buffer information
        """
        return {
            'capacity': self.capacity,
            'current_size': len(self),
            'state_dim': self.state_dim,
            'device': str(self.device),
            'utilization': len(self) / self.capacity if self.capacity > 0 else 0.0,
            'push_count': self.push_count,
            'sample_count': self.sample_count,
            'type': 'MockReplayBuffer'
        }


class DeterministicMockReplayBuffer(ReplayBufferInterface):
    """
    Deterministic mock replay buffer for testing.
    
    This implementation provides deterministic behavior for testing,
    always returning the same samples in the same order.
    """
    
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        """
        Initialize the deterministic mock replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            device: PyTorch device to use for tensor operations
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        self.buffer = []
        self.sample_index = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        
        # Keep only the most recent transitions up to capacity
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer (deterministic).
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        if batch_size == 0:
            # Return empty tensors if buffer is empty
            empty_states = torch.FloatTensor([]).to(self.device)
            empty_actions = torch.LongTensor([]).to(self.device)
            empty_rewards = torch.FloatTensor([]).to(self.device)
            empty_next_states = torch.FloatTensor([]).to(self.device)
            empty_dones = torch.FloatTensor([]).to(self.device)
            return empty_states, empty_actions, empty_rewards, empty_next_states, empty_dones
        
        # Take sequential samples starting from sample_index
        start_idx = self.sample_index % len(self.buffer)
        indices = [(start_idx + i) % len(self.buffer) for i in range(batch_size)]
        self.sample_index += batch_size
        
        # Get the sampled transitions
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self) >= batch_size
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer.clear()
        self.sample_index = 0
    
    def get_info(self) -> dict:
        """
        Get information about the buffer.
        
        Returns:
            Dictionary with buffer information
        """
        return {
            'capacity': self.capacity,
            'current_size': len(self),
            'state_dim': self.state_dim,
            'device': str(self.device),
            'utilization': len(self) / self.capacity if self.capacity > 0 else 0.0,
            'sample_index': self.sample_index,
            'type': 'DeterministicMockReplayBuffer'
        }


class FailingMockReplayBuffer(ReplayBufferInterface):
    """
    Failing mock replay buffer for testing error handling.
    
    This implementation simulates failures to test how the agent
    handles replay buffer errors.
    """
    
    def __init__(self, capacity: int, state_dim: int, device: torch.device, fail_after: int = 5):
        """
        Initialize the failing mock replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            device: PyTorch device to use for tensor operations
            fail_after: Number of operations before failing
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        self.buffer = []
        self.operation_count = 0
        self.fail_after = fail_after
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the buffer (may fail).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.operation_count += 1
        if self.operation_count > self.fail_after:
            raise RuntimeError("Mock replay buffer push failed")
        
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        
        # Keep only the most recent transitions up to capacity
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer (may fail).
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors
        """
        self.operation_count += 1
        if self.operation_count > self.fail_after:
            raise RuntimeError("Mock replay buffer sample failed")
        
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        if batch_size == 0:
            # Return empty tensors if buffer is empty
            empty_states = torch.FloatTensor([]).to(self.device)
            empty_actions = torch.LongTensor([]).to(self.device)
            empty_rewards = torch.FloatTensor([]).to(self.device)
            empty_next_states = torch.FloatTensor([]).to(self.device)
            empty_dones = torch.FloatTensor([]).to(self.device)
            return empty_states, empty_actions, empty_rewards, empty_next_states, empty_dones
        
        # Sample random indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Get the sampled transitions
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self) >= batch_size
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer.clear()
        self.operation_count = 0
    
    def get_info(self) -> dict:
        """
        Get information about the buffer.
        
        Returns:
            Dictionary with buffer information
        """
        return {
            'capacity': self.capacity,
            'current_size': len(self),
            'state_dim': self.state_dim,
            'device': str(self.device),
            'utilization': len(self) / self.capacity if self.capacity > 0 else 0.0,
            'operation_count': self.operation_count,
            'fail_after': self.fail_after,
            'type': 'FailingMockReplayBuffer'
        } 