"""
Tests for agent module.
"""

import numpy as np
import pytest
from hybrid_compiler.agent import (
    ReplayBuffer, AgentConfig, RandomAgent, TORCH_AVAILABLE
)

# Only import torch-dependent classes if available
if TORCH_AVAILABLE:
    from hybrid_compiler.agent import DQNAgent, PolicyGradientAgent


class TestReplayBuffer:
    """Test suite for ReplayBuffer."""
    
    def test_buffer_creation(self):
        """Test buffer creation."""
        buffer = ReplayBuffer(capacity=100)
        
        assert len(buffer) == 0
    
    def test_push_single(self):
        """Test pushing single experience."""
        buffer = ReplayBuffer(capacity=100)
        
        state = np.array([1.0, 2.0, 3.0])
        action = 0
        reward = 1.0
        next_state = np.array([2.0, 3.0, 4.0])
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
    
    def test_push_multiple(self):
        """Test pushing multiple experiences."""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(10):
            buffer.push(
                np.array([float(i)]),
                i % 5,
                float(i),
                np.array([float(i + 1)]),
                i == 9
            )
        
        assert len(buffer) == 10
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=5)
        
        for i in range(10):
            buffer.push(
                np.array([float(i)]),
                0,
                0.0,
                np.array([0.0]),
                False
            )
        
        assert len(buffer) == 5
    
    def test_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(20):
            buffer.push(
                np.array([float(i), float(i + 1)]),
                i % 5,
                float(i),
                np.array([float(i + 1), float(i + 2)]),
                i == 19
            )
        
        states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)
        
        assert states.shape == (5, 2)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 2)
        assert dones.shape == (5,)


class TestAgentConfig:
    """Test suite for AgentConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig()
        
        assert config.learning_rate == 1e-3
        assert config.gamma == 0.99
        assert config.epsilon_start == 1.0
        assert config.hidden_sizes == [256, 256]
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            learning_rate=5e-4,
            gamma=0.95,
            hidden_sizes=[128, 128, 64]
        )
        
        assert config.learning_rate == 5e-4
        assert config.gamma == 0.95
        assert config.hidden_sizes == [128, 128, 64]


class TestRandomAgent:
    """Test suite for RandomAgent."""
    
    def test_creation(self):
        """Test random agent creation."""
        agent = RandomAgent(n_actions=10)
        
        assert agent.n_actions == 10
    
    def test_select_action(self):
        """Test action selection."""
        agent = RandomAgent(n_actions=10)
        
        for _ in range(100):
            action = agent.select_action(np.array([1.0, 2.0]))
            assert 0 <= action < 10
    
    def test_action_distribution(self):
        """Test that actions are approximately uniformly distributed."""
        agent = RandomAgent(n_actions=5)
        
        counts = [0] * 5
        n_samples = 1000
        
        for _ in range(n_samples):
            action = agent.select_action(np.array([0.0]))
            counts[action] += 1
        
        # Each action should be selected roughly n_samples/5 times
        expected = n_samples / 5
        for count in counts:
            assert abs(count - expected) < expected * 0.3  # Within 30% of expected


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDQNAgent:
    """Test suite for DQNAgent."""
    
    def test_creation(self):
        """Test DQN agent creation."""
        agent = DQNAgent(obs_dim=10, n_actions=5)
        
        assert agent.obs_dim == 10
        assert agent.n_actions == 5
    
    def test_select_action(self):
        """Test action selection."""
        agent = DQNAgent(obs_dim=10, n_actions=5)
        
        state = np.random.randn(10).astype(np.float32)
        action = agent.select_action(state)
        
        assert 0 <= action < 5
    
    def test_select_action_evaluate(self):
        """Test greedy action selection in evaluation mode."""
        agent = DQNAgent(obs_dim=10, n_actions=5)
        agent.epsilon = 0.0  # No exploration
        
        state = np.random.randn(10).astype(np.float32)
        
        # Same state should give same action in eval mode
        actions = [agent.select_action(state, evaluate=True) for _ in range(10)]
        assert len(set(actions)) == 1
    
    def test_store_transition(self):
        """Test storing transitions."""
        agent = DQNAgent(obs_dim=10, n_actions=5)
        
        state = np.random.randn(10).astype(np.float32)
        next_state = np.random.randn(10).astype(np.float32)
        
        agent.store_transition(state, 0, 1.0, next_state, False)
        
        assert len(agent.replay_buffer) == 1
    
    def test_train_step_insufficient_data(self):
        """Test that training doesn't happen with insufficient data."""
        config = AgentConfig(min_replay_size=100)
        agent = DQNAgent(obs_dim=10, n_actions=5, config=config)
        
        # Add only a few transitions
        for i in range(10):
            agent.store_transition(
                np.random.randn(10).astype(np.float32),
                0, 1.0,
                np.random.randn(10).astype(np.float32),
                False
            )
        
        loss = agent.train_step()
        assert loss is None
    
    def test_train_step(self):
        """Test training step with sufficient data."""
        config = AgentConfig(min_replay_size=10, batch_size=5)
        agent = DQNAgent(obs_dim=10, n_actions=5, config=config)
        
        # Add enough transitions
        for i in range(20):
            agent.store_transition(
                np.random.randn(10).astype(np.float32),
                i % 5, 1.0,
                np.random.randn(10).astype(np.float32),
                i == 19
            )
        
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_decay_epsilon(self):
        """Test epsilon decay."""
        agent = DQNAgent(obs_dim=10, n_actions=5)
        
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
    
    def test_save_load(self, tmp_path):
        """Test saving and loading agent."""
        agent = DQNAgent(obs_dim=10, n_actions=5)
        
        # Do some training
        for i in range(100):
            agent.store_transition(
                np.random.randn(10).astype(np.float32),
                i % 5, 1.0,
                np.random.randn(10).astype(np.float32),
                i == 99
            )
        
        agent.train_step()
        agent.decay_epsilon()
        
        save_path = tmp_path / "agent.pt"
        agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = DQNAgent(obs_dim=10, n_actions=5)
        new_agent.load(str(save_path))
        
        assert new_agent.epsilon == agent.epsilon
        assert new_agent.train_steps == agent.train_steps


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPolicyGradientAgent:
    """Test suite for PolicyGradientAgent."""
    
    def test_creation(self):
        """Test Policy Gradient agent creation."""
        agent = PolicyGradientAgent(obs_dim=10, n_actions=5)
        
        assert agent.obs_dim == 10
        assert agent.n_actions == 5
    
    def test_select_action(self):
        """Test action selection."""
        agent = PolicyGradientAgent(obs_dim=10, n_actions=5)
        
        state = np.random.randn(10).astype(np.float32)
        action = agent.select_action(state)
        
        assert 0 <= action < 5
    
    def test_store_reward(self):
        """Test storing rewards."""
        agent = PolicyGradientAgent(obs_dim=10, n_actions=5)
        
        agent.store_reward(1.0)
        agent.store_reward(2.0)
        
        assert len(agent.rewards) == 2
    
    def test_train_episode(self):
        """Test training at end of episode."""
        agent = PolicyGradientAgent(obs_dim=10, n_actions=5)
        
        # Simulate episode
        for _ in range(10):
            state = np.random.randn(10).astype(np.float32)
            agent.select_action(state)  # This stores log_prob
            agent.store_reward(np.random.randn())
        
        loss = agent.train_episode()
        
        assert isinstance(loss, float)
        # After training, storage should be cleared
        assert len(agent.rewards) == 0
        assert len(agent.saved_log_probs) == 0
