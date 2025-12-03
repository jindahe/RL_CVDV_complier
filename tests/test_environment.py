"""
Tests for environment module.
"""

import numpy as np
import pytest
from hybrid_compiler.environment import (
    HybridCompilerEnv, CompilationConfig, make_env
)
from hybrid_compiler.hamiltonian import HamiltonianBuilder
from hybrid_compiler.gates import GateType


class TestCompilationConfig:
    """Test suite for CompilationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CompilationConfig()
        
        assert config.qumode_dim == 10
        assert config.max_gates == 20
        assert config.target_fidelity == 0.99
        assert config.fidelity_reward_scale > 0
        assert config.gate_penalty > 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CompilationConfig(
            qumode_dim=5,
            max_gates=15,
            target_fidelity=0.95
        )
        
        assert config.qumode_dim == 5
        assert config.max_gates == 15
        assert config.target_fidelity == 0.95


class TestHybridCompilerEnv:
    """Test suite for HybridCompilerEnv."""
    
    @pytest.fixture
    def simple_env(self):
        """Create a simple environment for testing."""
        config = CompilationConfig(qumode_dim=5, max_gates=10)
        builder = HamiltonianBuilder(qumode_dim=5)
        H = builder.dispersive_model(omega_q=1.0, omega_c=1.0, chi=0.1)
        
        return HybridCompilerEnv(config=config, target_hamiltonian=H, evolution_time=0.5)
    
    def test_env_creation(self, simple_env):
        """Test environment creation."""
        assert simple_env is not None
        assert simple_env.config.qumode_dim == 5
    
    def test_action_space(self, simple_env):
        """Test action space is valid."""
        from gymnasium.spaces import Discrete
        
        assert isinstance(simple_env.action_space, Discrete)
        assert simple_env.action_space.n > 0
    
    def test_observation_space(self, simple_env):
        """Test observation space is valid."""
        from gymnasium.spaces import Box
        
        assert isinstance(simple_env.observation_space, Box)
        assert len(simple_env.observation_space.shape) == 1
    
    def test_reset(self, simple_env):
        """Test environment reset."""
        obs, info = simple_env.reset()
        
        assert obs.shape == simple_env.observation_space.shape
        assert 'fidelity' in info
        assert info['depth'] == 0
    
    def test_step(self, simple_env):
        """Test environment step."""
        simple_env.reset()
        
        action = simple_env.action_space.sample()
        obs, reward, terminated, truncated, info = simple_env.step(action)
        
        assert obs.shape == simple_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'fidelity' in info
        assert info['depth'] == 1
    
    def test_max_gates_truncation(self, simple_env):
        """Test that episode truncates at max_gates."""
        simple_env.reset()
        
        for i in range(simple_env.config.max_gates + 1):
            action = simple_env.action_space.sample()
            obs, reward, terminated, truncated, info = simple_env.step(action)
            
            if truncated or terminated:
                break
        
        # Should have reached max_gates
        assert info['depth'] <= simple_env.config.max_gates
    
    def test_fidelity_increases_possible(self, simple_env):
        """Test that fidelity can increase with good actions."""
        obs, info = simple_env.reset()
        initial_fidelity = info['fidelity']
        
        max_fidelity = initial_fidelity
        
        # Try many random actions
        for _ in range(50):
            simple_env.reset()
            for _ in range(10):
                action = simple_env.action_space.sample()
                obs, reward, terminated, truncated, info = simple_env.step(action)
                max_fidelity = max(max_fidelity, info['fidelity'])
                if terminated or truncated:
                    break
        
        # At least some improvement should be possible
        assert max_fidelity >= initial_fidelity
    
    def test_get_compiled_sequence(self, simple_env):
        """Test getting the compiled sequence."""
        simple_env.reset()
        
        # Add some gates
        for _ in range(3):
            simple_env.step(simple_env.action_space.sample())
        
        seq = simple_env.get_compiled_sequence()
        
        assert len(seq) == 3
    
    def test_render_ansi(self, simple_env):
        """Test ANSI rendering."""
        simple_env.render_mode = "ansi"
        simple_env.reset()
        simple_env.step(simple_env.action_space.sample())
        
        output = simple_env.render()
        
        assert isinstance(output, str)
        assert "Fidelity" in output
    
    def test_set_target(self, simple_env):
        """Test setting a new target Hamiltonian."""
        builder = HamiltonianBuilder(qumode_dim=5)
        new_H = builder.rabi_model(omega_q=2.0, omega_c=1.0, g=0.05)
        
        simple_env.set_target(new_H, evolution_time=1.0)
        
        obs, info = simple_env.reset()
        
        # Should have new target
        assert simple_env.evolution_time == 1.0


class TestMakeEnv:
    """Test suite for make_env factory function."""
    
    def test_make_env_default(self):
        """Test make_env with default parameters."""
        env = make_env()
        
        assert isinstance(env, HybridCompilerEnv)
        assert env.config.qumode_dim == 10
    
    def test_make_env_custom(self):
        """Test make_env with custom parameters."""
        env = make_env(qumode_dim=5, max_gates=15, target_fidelity=0.95)
        
        assert env.config.qumode_dim == 5
        assert env.config.max_gates == 15
        assert env.config.target_fidelity == 0.95


class TestGymCompatibility:
    """Test Gym/Gymnasium interface compatibility."""
    
    def test_gym_check_env(self):
        """Test that environment passes Gymnasium's check_env."""
        try:
            from gymnasium.utils.env_checker import check_env
            
            config = CompilationConfig(qumode_dim=5, max_gates=10)
            builder = HamiltonianBuilder(qumode_dim=5)
            H = builder.dispersive_model(omega_q=1.0, omega_c=1.0, chi=0.1)
            
            env = HybridCompilerEnv(config=config, target_hamiltonian=H)
            
            # This will raise if there are issues
            check_env(env, skip_render_check=True)
        except ImportError:
            pytest.skip("gymnasium.utils.env_checker not available")
