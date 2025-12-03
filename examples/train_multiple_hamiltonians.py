#!/usr/bin/env python3
"""
Example: Training with different Hamiltonians.

This example shows how to train an agent that can generalize
across different target Hamiltonians.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from hybrid_compiler import (
    HamiltonianBuilder,
    HybridCompilerEnv,
    CompilationConfig,
    DQNAgent,
    AgentConfig,
    train_dqn,
    TrainingConfig,
    evaluate_agent,
    TORCH_AVAILABLE
)


def random_hamiltonian_generator(qumode_dim: int = 5):
    """
    Generator function that creates random Hamiltonians.
    
    This helps the agent learn a general compilation strategy
    instead of overfitting to a single target.
    """
    builder = HamiltonianBuilder(qumode_dim=qumode_dim)
    
    # Randomly select a model type
    model_type = np.random.choice(['jc', 'rabi', 'dispersive'])
    
    # Random parameters
    omega_q = np.random.uniform(0.5, 2.0)
    omega_c = np.random.uniform(0.5, 2.0)
    
    if model_type == 'jc':
        g = np.random.uniform(0.05, 0.3)
        return builder.jaynes_cummings_model(omega_q, omega_c, g)
    elif model_type == 'rabi':
        g = np.random.uniform(0.05, 0.3)
        return builder.rabi_model(omega_q, omega_c, g)
    else:
        chi = np.random.uniform(0.01, 0.1)
        return builder.dispersive_model(omega_q, omega_c, chi)


def main():
    if not TORCH_AVAILABLE:
        print("This example requires PyTorch. Install with: pip install torch")
        return
    
    print("=" * 60)
    print("Training with Multiple Hamiltonians")
    print("=" * 60)
    
    qumode_dim = 5
    
    # Environment setup
    config = CompilationConfig(
        qumode_dim=qumode_dim,
        max_gates=15,
        target_fidelity=0.9,
        n_discrete_params=3
    )
    
    # Create initial environment with a default Hamiltonian
    builder = HamiltonianBuilder(qumode_dim=qumode_dim)
    initial_H = builder.jaynes_cummings_model(1.0, 1.0, 0.1)
    
    env = HybridCompilerEnv(config=config, target_hamiltonian=initial_H)
    
    # Create agent
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        config=AgentConfig(
            hidden_sizes=[256, 256],
            epsilon_decay=0.995
        )
    )
    
    # Create Hamiltonian generator
    def generator():
        return random_hamiltonian_generator(qumode_dim)
    
    # Training with random Hamiltonians
    print("\n[1] Training with random Hamiltonians...")
    
    training_config = TrainingConfig(
        n_episodes=200,
        max_steps_per_episode=15,
        log_interval=50,
        eval_interval=100,
        early_stopping=False,
        save_interval=1000,  # Don't save during this demo
        save_path="/tmp/compiler_agent"  # Save to temp directory
    )
    
    metrics = train_dqn(
        env, agent, training_config,
        hamiltonian_generator=generator
    )
    
    # Evaluation on different Hamiltonians
    print("\n[2] Evaluating on different Hamiltonian types...")
    
    test_cases = [
        ("Jaynes-Cummings", builder.jaynes_cummings_model(1.0, 1.0, 0.1)),
        ("Rabi Model", builder.rabi_model(1.0, 1.0, 0.2)),
        ("Dispersive", builder.dispersive_model(1.0, 1.0, 0.05)),
    ]
    
    for name, H in test_cases:
        env.set_target(H, evolution_time=0.5)
        
        # Evaluate
        eval_results = evaluate_agent(env, agent, n_episodes=5)
        
        print(f"\n  {name}:")
        print(f"    Average fidelity: {eval_results['avg_fidelity']:.4f}")
        print(f"    Success rate: {eval_results['success_rate']:.2%}")
        print(f"    Average depth: {eval_results['avg_length']:.1f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
