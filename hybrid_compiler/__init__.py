"""
Hybrid CV-DV Quantum Hamiltonian Compiler

A reinforcement learning based compiler for hybrid qubit-qumode quantum systems.
This package provides tools for compiling target Hamiltonians into gate sequences
using RL agents.

Main components:
- operators: Quantum operators for CV-DV systems
- hamiltonian: Hamiltonian representation and construction
- gates: Gate library for compilation
- environment: RL environment for training
- agent: RL agents (DQN, Policy Gradient)
- training: Training utilities

Example usage:
    from hybrid_compiler import (
        HamiltonianBuilder, HybridCompilerEnv, DQNAgent,
        train_dqn, TrainingConfig, CompilationConfig
    )
    
    # Build a target Hamiltonian
    builder = HamiltonianBuilder(qumode_dim=5)
    hamiltonian = builder.jaynes_cummings_model(
        omega_q=1.0, omega_c=1.0, g=0.1
    )
    
    # Create environment
    config = CompilationConfig(qumode_dim=5, max_gates=15)
    env = HybridCompilerEnv(config=config, target_hamiltonian=hamiltonian)
    
    # Create and train agent
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n
    )
    
    metrics = train_dqn(env, agent, TrainingConfig(n_episodes=100))
"""

from .operators import (
    Operator,
    pauli_x, pauli_y, pauli_z, qubit_identity,
    sigma_plus, sigma_minus, hadamard,
    rotation_x, rotation_y, rotation_z,
    creation, annihilation, number_op, position, momentum, qumode_identity,
    displacement, squeeze, rotation_mode,
    tensor, embed_qubit, embed_qumode,
    jaynes_cummings, anti_jaynes_cummings, dispersive_coupling,
    conditional_displacement
)

from .hamiltonian import (
    Hamiltonian,
    HamiltonianBuilder,
    fidelity,
    process_fidelity,
    operator_distance
)

from .gates import (
    Gate,
    GateType,
    GateDefinition,
    GateLibrary,
    GateSequence,
    GATE_DEFINITIONS
)

from .environment import (
    HybridCompilerEnv,
    CompilationConfig,
    RandomTargetWrapper,
    make_env
)

from .agent import (
    DQNAgent,
    PolicyGradientAgent,
    RandomAgent,
    AgentConfig,
    ReplayBuffer,
    TORCH_AVAILABLE
)

from .training import (
    train_dqn,
    train_policy_gradient,
    evaluate_agent,
    compile_hamiltonian,
    TrainingConfig,
    TrainingMetrics
)

__version__ = "0.1.0"
__all__ = [
    # Operators
    "Operator",
    "pauli_x", "pauli_y", "pauli_z", "qubit_identity",
    "sigma_plus", "sigma_minus", "hadamard",
    "rotation_x", "rotation_y", "rotation_z",
    "creation", "annihilation", "number_op", "position", "momentum", "qumode_identity",
    "displacement", "squeeze", "rotation_mode",
    "tensor", "embed_qubit", "embed_qumode",
    "jaynes_cummings", "anti_jaynes_cummings", "dispersive_coupling",
    "conditional_displacement",
    
    # Hamiltonian
    "Hamiltonian", "HamiltonianBuilder",
    "fidelity", "process_fidelity", "operator_distance",
    
    # Gates
    "Gate", "GateType", "GateDefinition", "GateLibrary", "GateSequence",
    "GATE_DEFINITIONS",
    
    # Environment
    "HybridCompilerEnv", "CompilationConfig", "RandomTargetWrapper", "make_env",
    
    # Agents
    "DQNAgent", "PolicyGradientAgent", "RandomAgent", "AgentConfig", "ReplayBuffer",
    "TORCH_AVAILABLE",
    
    # Training
    "train_dqn", "train_policy_gradient", "evaluate_agent", "compile_hamiltonian",
    "TrainingConfig", "TrainingMetrics",
]
