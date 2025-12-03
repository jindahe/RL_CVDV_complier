"""
Reinforcement Learning environment for hybrid CV-DV Hamiltonian compilation.

This module provides a Gym-compatible environment for training an RL agent
to compile target Hamiltonians into gate sequences.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

from .operators import Operator, tensor, qubit_identity, qumode_identity
from .hamiltonian import Hamiltonian, fidelity, process_fidelity
from .gates import Gate, GateLibrary, GateSequence, GateType


@dataclass
class CompilationConfig:
    """Configuration for the compilation environment."""
    
    # System parameters
    qumode_dim: int = 10  # Fock space truncation
    
    # Gate library parameters
    n_discrete_params: int = 5  # Number of discrete values per parameter
    gate_types: Optional[List[GateType]] = None  # None = all gates
    
    # Episode parameters
    max_gates: int = 20  # Maximum gates in a sequence
    target_fidelity: float = 0.99  # Fidelity threshold for success
    
    # Reward parameters
    fidelity_reward_scale: float = 10.0  # Scale for fidelity improvement reward
    gate_penalty: float = 0.01  # Penalty per gate (encourages shorter circuits)
    success_bonus: float = 100.0  # Bonus for reaching target fidelity
    
    # State representation
    include_hamiltonian_in_state: bool = True
    state_history_length: int = 5  # Number of recent gates to include in state


class HybridCompilerEnv(gym.Env):
    """
    Gym environment for hybrid qubit-qumode Hamiltonian compilation.
    
    The agent's goal is to find a sequence of quantum gates that implements
    the time evolution of a target Hamiltonian with high fidelity.
    
    State: 
        - Current fidelity
        - Steps taken
        - Recent action history
        - Target Hamiltonian features (optional)
    
    Action:
        - Discrete action selecting a gate from the library
    
    Reward:
        - Fidelity improvement
        - Gate count penalty
        - Success bonus
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        config: Optional[CompilationConfig] = None,
        target_hamiltonian: Optional[Hamiltonian] = None,
        evolution_time: float = 1.0,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the environment.
        
        Args:
            config: Configuration object
            target_hamiltonian: Target Hamiltonian to compile
            evolution_time: Time for target evolution U = exp(-i*H*t)
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.config = config or CompilationConfig()
        self.evolution_time = evolution_time
        self.render_mode = render_mode
        
        # Initialize gate library
        self.gate_library = GateLibrary(
            qumode_dim=self.config.qumode_dim,
            gate_types=self.config.gate_types,
            n_discrete_params=self.config.n_discrete_params
        )
        
        # Action space: discrete selection from gate library
        self.action_space = spaces.Discrete(self.gate_library.n_actions)
        
        # Calculate observation space size
        obs_size = self._calculate_obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Set target Hamiltonian
        self._target_hamiltonian = target_hamiltonian
        self._target_unitary: Optional[Operator] = None
        
        # Episode state
        self._gate_sequence: GateSequence = GateSequence()
        self._current_fidelity: float = 0.0
        self._step_count: int = 0
        self._action_history: List[int] = []
    
    def _calculate_obs_size(self) -> int:
        """Calculate the observation space dimension."""
        size = 2  # Current fidelity + normalized step count
        size += self.config.state_history_length  # Recent actions (normalized)
        
        if self.config.include_hamiltonian_in_state:
            # Flatten Hamiltonian matrix (real and imaginary parts)
            dim = 2 * self.config.qumode_dim
            size += 2 * (dim ** 2)  # Real and imaginary parts
        
        return size
    
    def set_target(self, hamiltonian: Hamiltonian, evolution_time: float = 1.0):
        """
        Set a new target Hamiltonian.
        
        Args:
            hamiltonian: Target Hamiltonian
            evolution_time: Evolution time for U = exp(-i*H*t)
        """
        self._target_hamiltonian = hamiltonian
        self.evolution_time = evolution_time
        self._target_unitary = hamiltonian.time_evolution(evolution_time)
    
    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector."""
        obs = []
        
        # Current fidelity
        obs.append(self._current_fidelity)
        
        # Normalized step count
        obs.append(self._step_count / self.config.max_gates)
        
        # Recent action history (normalized by action space size)
        # Use action_space.n as sentinel for empty slots (out of valid range)
        history = self._action_history[-self.config.state_history_length:]
        padding_value = self.action_space.n  # Out-of-range sentinel
        history = history + [padding_value] * (self.config.state_history_length - len(history))
        # Normalize to [0, 1] range (sentinel becomes 1.0)
        obs.extend([a / self.action_space.n for a in history])
        
        # Hamiltonian features
        if self.config.include_hamiltonian_in_state and self._target_hamiltonian is not None:
            H_matrix = self._target_hamiltonian.matrix
            obs.extend(H_matrix.real.flatten())
            obs.extend(H_matrix.imag.flatten())
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_fidelity(self) -> float:
        """Calculate fidelity between current sequence and target."""
        if self._target_unitary is None:
            return 0.0
        
        if len(self._gate_sequence) == 0:
            # Empty sequence = identity
            dim = self._target_unitary.dim
            identity = Operator(np.eye(dim), "I")
            return process_fidelity(self._target_unitary, identity)
        
        compiled_unitary = self._gate_sequence.to_operator()
        return process_fidelity(self._target_unitary, compiled_unitary)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Additional options (can include 'hamiltonian' and 'evolution_time')
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Check for new target in options
        if options is not None:
            if 'hamiltonian' in options:
                self._target_hamiltonian = options['hamiltonian']
            if 'evolution_time' in options:
                self.evolution_time = options['evolution_time']
        
        # Compute target unitary
        if self._target_hamiltonian is not None:
            self._target_unitary = self._target_hamiltonian.time_evolution(self.evolution_time)
        
        # Reset episode state
        self._gate_sequence = GateSequence()
        self._step_count = 0
        self._action_history = []
        self._current_fidelity = self._calculate_fidelity()
        
        obs = self._get_obs()
        info = {
            "fidelity": self._current_fidelity,
            "depth": 0,
            "target_dim": self._target_unitary.dim if self._target_unitary else 0
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment by adding a gate.
        
        Args:
            action: Index of the gate to add
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether the episode ended (success or failure)
            truncated: Whether the episode was truncated (max steps)
            info: Additional information
        """
        # Get the gate for this action
        gate = self.gate_library.get_gate(action)
        
        # Store previous fidelity
        prev_fidelity = self._current_fidelity
        
        # Add gate to sequence
        self._gate_sequence.append(gate)
        self._action_history.append(action)
        self._step_count += 1
        
        # Calculate new fidelity
        self._current_fidelity = self._calculate_fidelity()
        
        # Calculate reward
        reward = self._calculate_reward(prev_fidelity, self._current_fidelity)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if self._current_fidelity >= self.config.target_fidelity:
            # Success!
            terminated = True
            reward += self.config.success_bonus
        
        if self._step_count >= self.config.max_gates:
            truncated = True
        
        obs = self._get_obs()
        info = {
            "fidelity": self._current_fidelity,
            "depth": len(self._gate_sequence),
            "fidelity_improvement": self._current_fidelity - prev_fidelity,
            "gate": str(gate),
            "success": self._current_fidelity >= self.config.target_fidelity
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, prev_fidelity: float, new_fidelity: float) -> float:
        """
        Calculate the reward for a step.
        
        Args:
            prev_fidelity: Fidelity before the step
            new_fidelity: Fidelity after the step
        
        Returns:
            Reward value
        """
        # Fidelity improvement reward
        fidelity_improvement = new_fidelity - prev_fidelity
        reward = self.config.fidelity_reward_scale * fidelity_improvement
        
        # Gate penalty (encourages shorter circuits)
        reward -= self.config.gate_penalty
        
        return reward
    
    def render(self) -> Optional[str]:
        """Render the current state."""
        if self.render_mode == "human":
            print(self._render_string())
        elif self.render_mode == "ansi":
            return self._render_string()
        return None
    
    def _render_string(self) -> str:
        """Generate a string representation of the current state."""
        lines = [
            "=" * 50,
            "Hybrid CV-DV Compiler Environment",
            "=" * 50,
            f"Step: {self._step_count}/{self.config.max_gates}",
            f"Current Fidelity: {self._current_fidelity:.6f}",
            f"Target Fidelity: {self.config.target_fidelity}",
            f"Circuit Depth: {len(self._gate_sequence)}",
            "",
            "Current Gate Sequence:",
        ]
        
        if len(self._gate_sequence) == 0:
            lines.append("  (empty)")
        else:
            for i, gate in enumerate(self._gate_sequence.gates):
                lines.append(f"  {i+1}. {gate}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def get_compiled_sequence(self) -> GateSequence:
        """Get the current compiled gate sequence."""
        return self._gate_sequence.copy()
    
    def get_compiled_unitary(self) -> Optional[Operator]:
        """Get the unitary operator for the current sequence."""
        if len(self._gate_sequence) == 0:
            return None
        return self._gate_sequence.to_operator()


class RandomTargetWrapper(gym.Wrapper):
    """
    Wrapper that randomizes the target Hamiltonian each episode.
    
    This helps the agent learn a general compilation strategy.
    """
    
    def __init__(
        self,
        env: HybridCompilerEnv,
        hamiltonian_generator: callable,
        evolution_time_range: Tuple[float, float] = (0.1, 2.0)
    ):
        """
        Initialize the wrapper.
        
        Args:
            env: The base environment
            hamiltonian_generator: Function that returns a random Hamiltonian
            evolution_time_range: Range for random evolution times
        """
        super().__init__(env)
        self.hamiltonian_generator = hamiltonian_generator
        self.evolution_time_range = evolution_time_range
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with a new random target."""
        # Generate random Hamiltonian
        hamiltonian = self.hamiltonian_generator()
        
        # Random evolution time
        t = np.random.uniform(*self.evolution_time_range)
        
        # Update options
        options = options or {}
        options['hamiltonian'] = hamiltonian
        options['evolution_time'] = t
        
        return self.env.reset(seed=seed, options=options)


def make_env(
    qumode_dim: int = 10,
    max_gates: int = 20,
    target_fidelity: float = 0.99,
    **kwargs
) -> HybridCompilerEnv:
    """
    Factory function to create a compilation environment.
    
    Args:
        qumode_dim: Fock space truncation dimension
        max_gates: Maximum number of gates per episode
        target_fidelity: Fidelity threshold for success
        **kwargs: Additional config parameters
    
    Returns:
        Configured HybridCompilerEnv instance
    """
    config = CompilationConfig(
        qumode_dim=qumode_dim,
        max_gates=max_gates,
        target_fidelity=target_fidelity,
        **kwargs
    )
    return HybridCompilerEnv(config=config)
