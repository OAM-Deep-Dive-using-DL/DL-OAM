import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple, Dict, Any, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn_model import DQN
from utils.replay_buffer import ReplayBuffer


class Agent:
    """
    RL agent that makes OAM handover decisions using a DQN.
    
    This agent manages the policy network, target network, and learning process.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: List[int] = [128, 128],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_layers: List of hidden layer sizes for the DQN
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            buffer_capacity: Maximum capacity of the replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency (in episodes) to update target network
            device: PyTorch device to use (defaults to CUDA if available)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_layers).to(self.device)
        
        # Copy policy network parameters to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, self.device)
        
        # Set hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Training tracking
        self.episode_count = 0
        self.loss_history = []
    
    def choose_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            epsilon: Probability of choosing a random action
            
        Returns:
            Selected action
        """
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Choose random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Choose greedy action
            with torch.no_grad():
                # Convert state to tensor and add batch dimension
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get Q-values from policy network
                q_values = self.policy_net(state_tensor)
                
                # Choose action with highest Q-value
                return torch.argmax(q_values).item()
    
    def learn(self) -> float:
        """
        Update the policy network using a batch from the replay buffer.
        
        Returns:
            Loss value from the update
        """
        # Check if buffer has enough samples
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Get Q-values for current states and actions
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            # Get maximum Q-value for next states from target network
            next_q_values = self.target_net(next_states).max(1)[0]
            
            # Compute target Q-values using Bellman equation
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
            # Reshape to match q_values
            target_q_values = target_q_values.unsqueeze(1)
        
        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # Store loss value
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def update_target_network(self) -> None:
        """Update the target network with the policy network's weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def end_episode(self) -> None:
        """
        Perform end-of-episode updates.
        
        Updates the target network if needed.
        """
        self.episode_count += 1
        
        # Update target network if it's time
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def save_models(self, save_dir: str) -> None:
        """
        Save both policy and target networks.
        
        Args:
            save_dir: Directory to save the models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        policy_path = os.path.join(save_dir, "policy_net.pth")
        target_path = os.path.join(save_dir, "target_net.pth")
        
        self.policy_net.save(policy_path)
        self.target_net.save(target_path)
    
    def load_models(self, save_dir: str) -> None:
        """
        Load both policy and target networks.
        
        Args:
            save_dir: Directory to load the models from
        """
        policy_path = os.path.join(save_dir, "policy_net.pth")
        target_path = os.path.join(save_dir, "target_net.pth")
        
        self.policy_net.load(policy_path, self.device)
        self.target_net.load(target_path, self.device) 