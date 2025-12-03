"""
Reinforcement Learning agents for hybrid CV-DV Hamiltonian compilation.

This module provides RL agents that can learn to compile quantum Hamiltonians.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import deque
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    
    # Network architecture
    hidden_sizes: List[int] = None  # Will default to [256, 256]
    
    # Training parameters
    learning_rate: float = 1e-3
    gamma: float = 0.99  # Discount factor
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Experience replay
    buffer_size: int = 100000
    batch_size: int = 64
    min_replay_size: int = 1000
    
    # Target network
    target_update_freq: int = 100
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


if TORCH_AVAILABLE:
    class QNetwork(nn.Module):
        """Q-Network for DQN agent."""
        
        def __init__(
            self,
            obs_dim: int,
            n_actions: int,
            hidden_sizes: List[int]
        ):
            """
            Initialize the Q-network.
            
            Args:
                obs_dim: Dimension of observation space
                n_actions: Number of discrete actions
                hidden_sizes: List of hidden layer sizes
            """
            super().__init__()
            
            layers = []
            prev_size = obs_dim
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, n_actions))
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            return self.network(x)


class DQNAgent:
    """
    Deep Q-Network agent for Hamiltonian compilation.
    
    Uses experience replay and target networks for stable training.
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: Optional[AgentConfig] = None,
        device: str = "cpu"
    ):
        """
        Initialize the DQN agent.
        
        Args:
            obs_dim: Dimension of observation space
            n_actions: Number of discrete actions
            config: Agent configuration
            device: Device for PyTorch ("cpu" or "cuda")
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQNAgent. Install with: pip install torch")
        
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config or AgentConfig()
        self.device = torch.device(device)
        
        # Networks
        self.q_network = QNetwork(
            obs_dim, n_actions, self.config.hidden_sizes
        ).to(self.device)
        
        self.target_network = QNetwork(
            obs_dim, n_actions, self.config.hidden_sizes
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        
        # Training stats
        self.train_steps = 0
        self.episode_rewards: List[float] = []
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current observation
            evaluate: If True, use greedy policy (no exploration)
        
        Returns:
            Selected action index
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform a single training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.config.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.train_steps += 1
        if self.train_steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
    
    def save(self, path: str):
        """
        Save agent to file.
        
        Note: Saves configuration alongside model weights. Only load checkpoints
        from trusted sources.
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
        }, path)
    
    def load(self, path: str):
        """
        Load agent from file.
        
        Warning: Only load checkpoints from trusted sources. This method uses
        weights_only=True for security but requires the checkpoint to contain
        only tensor data.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']


class PolicyGradientAgent:
    """
    Policy Gradient (REINFORCE) agent for Hamiltonian compilation.
    
    Directly optimizes the policy without Q-value estimation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: List[int] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu"
    ):
        """
        Initialize the Policy Gradient agent.
        
        Args:
            obs_dim: Dimension of observation space
            n_actions: Number of discrete actions
            hidden_sizes: List of hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            device: Device for PyTorch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PolicyGradientAgent")
        
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = torch.device(device)
        
        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        
        # Policy network
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, n_actions))
        
        self.policy = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Episode storage
        self.saved_log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action from the policy.
        
        Args:
            state: Current observation
            evaluate: If True, select most likely action (no sampling)
        
        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.policy(state_tensor)
        probs = F.softmax(logits, dim=1)
        
        if evaluate:
            return probs.argmax(dim=1).item()
        
        # Sample from distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward: float):
        """Store reward for the current step."""
        self.rewards.append(reward)
    
    def train_episode(self) -> float:
        """
        Perform policy gradient update at the end of an episode.
        
        Returns:
            Loss value
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode storage
        loss = policy_loss.item()
        self.saved_log_probs = []
        self.rewards = []
        
        return loss
    
    def save(self, path: str):
        """
        Save agent to file.
        
        Note: Only load checkpoints from trusted sources.
        """
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """
        Load agent from file.
        
        Warning: Only load checkpoints from trusted sources.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class RandomAgent:
    """Simple random agent for baseline comparisons."""
    
    def __init__(self, n_actions: int):
        """
        Initialize random agent.
        
        Args:
            n_actions: Number of discrete actions
        """
        self.n_actions = n_actions
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select a random action."""
        return random.randint(0, self.n_actions - 1)
    
    def store_transition(self, *args):
        """No-op for interface compatibility."""
        pass
    
    def train_step(self) -> None:
        """No-op for interface compatibility."""
        return None
