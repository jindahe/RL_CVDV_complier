# RL-based Hybrid CV-DV Quantum Hamiltonian Compiler

A reinforcement learning framework for compiling target Hamiltonians into executable gate sequences for hybrid continuous-variable (CV) / discrete-variable (DV) quantum systems.

## Overview

This project provides tools for:

1. **Representing hybrid qubit-qumode quantum systems** - Operators and Hamiltonians for systems combining qubits (DV) with quantum harmonic oscillators/qumodes (CV)

2. **Defining a gate library** - Parameterized quantum gates including:
   - Qubit gates (Rx, Ry, Rz, Hadamard)
   - Qumode gates (displacement, squeezing, phase rotation)
   - Hybrid gates (Jaynes-Cummings, conditional displacement, dispersive coupling)

3. **RL environment** - A Gymnasium-compatible environment where:
   - State: Current fidelity, step count, action history, target Hamiltonian features
   - Action: Discrete selection from the gate library
   - Reward: Fidelity improvement - gate penalty + success bonus

4. **RL agents** - DQN and Policy Gradient agents for learning compilation strategies

## Installation

### Basic Installation

```bash
pip install -e .
```

### With PyTorch (for deep RL agents)

```bash
pip install -e ".[torch]"
```

### Development Installation

```bash
pip install -e ".[all]"
```

## Quick Start

```python
from hybrid_compiler import (
    HamiltonianBuilder,
    HybridCompilerEnv,
    CompilationConfig,
    DQNAgent,
    train_dqn,
    TrainingConfig,
)

# 1. Create a target Hamiltonian
builder = HamiltonianBuilder(qumode_dim=5)
target_H = builder.jaynes_cummings_model(
    omega_q=1.0,  # Qubit frequency
    omega_c=1.0,  # Cavity frequency
    g=0.1         # Coupling strength
)

# 2. Set up the environment
config = CompilationConfig(
    qumode_dim=5,
    max_gates=15,
    target_fidelity=0.95
)
env = HybridCompilerEnv(config=config, target_hamiltonian=target_H)

# 3. Create and train an agent
agent = DQNAgent(
    obs_dim=env.observation_space.shape[0],
    n_actions=env.action_space.n
)

metrics = train_dqn(env, agent, TrainingConfig(n_episodes=500))

# 4. Compile a Hamiltonian
from hybrid_compiler import compile_hamiltonian

result = compile_hamiltonian(target_H, agent, env, evolution_time=1.0)
print(f"Fidelity: {result['fidelity']:.4f}")
print(f"Gate sequence: {result['gate_sequence']}")
```

## Project Structure

```
hybrid_compiler/
├── __init__.py          # Package exports
├── operators.py         # Quantum operators (Pauli, CV operators, etc.)
├── hamiltonian.py       # Hamiltonian class and builders
├── gates.py             # Gate definitions and sequences
├── environment.py       # RL environment
├── agent.py             # RL agents (DQN, Policy Gradient)
└── training.py          # Training utilities

examples/
├── basic_usage.py                   # Basic example
└── train_multiple_hamiltonians.py   # Training on multiple targets

tests/
├── test_operators.py
├── test_hamiltonian.py
├── test_gates.py
├── test_environment.py
└── test_agent.py
```

## Key Components

### Operators

The `operators` module provides quantum operators for both DV (qubits) and CV (qumodes):

```python
from hybrid_compiler import (
    pauli_x, pauli_y, pauli_z,      # Qubit operators
    creation, annihilation,          # Qumode ladder operators
    number_op, position, momentum,   # Qumode observables
    displacement, squeeze,           # Qumode gates
    jaynes_cummings,                 # Hybrid interaction
    tensor,                          # Tensor product
)

# Create hybrid operators
dim = 10  # Fock space truncation
H_interaction = jaynes_cummings(dim, g=0.1)
```

### Hamiltonians

Build standard quantum optics Hamiltonians:

```python
from hybrid_compiler import HamiltonianBuilder

builder = HamiltonianBuilder(qumode_dim=10)

# Standard models
H_JC = builder.jaynes_cummings_model(omega_q=1.0, omega_c=1.0, g=0.1)
H_Rabi = builder.rabi_model(omega_q=1.0, omega_c=1.0, g=0.1)
H_disp = builder.dispersive_model(omega_q=1.0, omega_c=1.0, chi=0.05)

# Custom Hamiltonian
H_custom = builder.custom([
    (1.0, "Iz*n"),     # σz ⊗ n
    (0.5, "Ix*q"),     # σx ⊗ q
])
```

### Gate Library

The gate library defines the action space:

```python
from hybrid_compiler import GateLibrary, GateType

# Full library (all gate types)
library = GateLibrary(qumode_dim=10, n_discrete_params=5)

# Restricted library
library = GateLibrary(
    qumode_dim=10,
    gate_types=[GateType.RX, GateType.RY, GateType.JAYNES_CUMMINGS],
    n_discrete_params=5
)

# Get a gate
gate = library.get_gate(action_idx=0)
unitary = gate.to_operator()
```

### RL Environment

The environment follows the Gymnasium interface:

```python
from hybrid_compiler import HybridCompilerEnv, CompilationConfig

config = CompilationConfig(
    qumode_dim=10,
    max_gates=20,
    target_fidelity=0.99,
    fidelity_reward_scale=10.0,
    gate_penalty=0.01,
    success_bonus=100.0
)

env = HybridCompilerEnv(config=config, target_hamiltonian=H)

# Standard Gymnasium interface
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

### Agents

DQN and Policy Gradient agents are available:

```python
from hybrid_compiler import DQNAgent, PolicyGradientAgent, AgentConfig

# DQN Agent
config = AgentConfig(
    hidden_sizes=[256, 256],
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_decay=0.995
)
agent = DQNAgent(obs_dim=50, n_actions=100, config=config)

# Policy Gradient Agent
pg_agent = PolicyGradientAgent(obs_dim=50, n_actions=100)
```

## Running Tests

```bash
pytest tests/ -v
```

## Examples

Run the examples:

```bash
cd examples
python basic_usage.py
python train_multiple_hamiltonians.py
```

## Theory Background

### Hybrid CV-DV Systems

Hybrid quantum systems combine:
- **Discrete-variable (DV) qubits**: Two-level systems with basis |0⟩, |1⟩
- **Continuous-variable (CV) qumodes**: Quantum harmonic oscillators with infinite-dimensional Fock space |0⟩, |1⟩, |2⟩, ...

The Hilbert space is H = H_qubit ⊗ H_qumode.

### Hamiltonian Compilation

Given a target Hamiltonian H, the goal is to find a gate sequence U₁, U₂, ..., Uₙ such that:

```
U_compiled = Uₙ @ ... @ U₂ @ U₁ ≈ exp(-i·H·t)
```

The fidelity measure is:
```
F = |Tr(U_target† @ U_compiled)|² / d²
```

### Reinforcement Learning Formulation

- **State**: Encodes current fidelity, step count, action history, and target Hamiltonian
- **Action**: Select a gate from the discrete gate library
- **Reward**: r = α·ΔF - β + γ·success
  - ΔF: Change in fidelity
  - β: Gate penalty (encourages shorter circuits)
  - γ: Bonus for reaching target fidelity

## License

MIT License
