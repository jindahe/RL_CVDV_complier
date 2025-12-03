#!/usr/bin/env python3
"""
Example: Basic usage of the hybrid CV-DV quantum compiler.

This example demonstrates:
1. Creating a target Hamiltonian
2. Setting up the RL environment
3. Training an agent to compile the Hamiltonian
4. Evaluating the compiled result
"""

import numpy as np
import sys

# Add parent directory to path for running from examples folder
sys.path.insert(0, '..')

from hybrid_compiler import (
    HamiltonianBuilder,
    HybridCompilerEnv,
    CompilationConfig,
    DQNAgent,
    RandomAgent,
    AgentConfig,
    train_dqn,
    TrainingConfig,
    compile_hamiltonian,
    TORCH_AVAILABLE
)


def main():
    print("=" * 60)
    print("Hybrid CV-DV Quantum Hamiltonian Compiler - Basic Example")
    print("=" * 60)
    
    # 1. Create a target Hamiltonian
    print("\n[1] Creating target Hamiltonian...")
    
    qumode_dim = 5  # Small dimension for fast computation
    builder = HamiltonianBuilder(qumode_dim=qumode_dim)
    
    # Create a Jaynes-Cummings model Hamiltonian
    # H = (ω_q/2)σz + ω_c*a†a + g(σ+a + σ-a†)
    target_H = builder.jaynes_cummings_model(
        omega_q=1.0,  # Qubit frequency
        omega_c=1.0,  # Cavity frequency
        g=0.1         # Coupling strength
    )
    
    print(f"  Target Hamiltonian: {target_H}")
    print(f"  System dimension: {target_H.dim}")
    
    # Compute the ground state
    energy, ground_state = target_H.ground_state()
    print(f"  Ground state energy: {energy:.4f}")
    
    # 2. Set up the compilation environment
    print("\n[2] Setting up compilation environment...")
    
    config = CompilationConfig(
        qumode_dim=qumode_dim,
        max_gates=15,
        target_fidelity=0.95,
        n_discrete_params=3,  # Fewer params for faster training
        fidelity_reward_scale=10.0,
        gate_penalty=0.01
    )
    
    env = HybridCompilerEnv(
        config=config,
        target_hamiltonian=target_H,
        evolution_time=0.5  # Compile U = exp(-i*H*t) for t=0.5
    )
    
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n} discrete actions")
    print(f"  Gate library: {env.gate_library}")
    
    # 3. Test with a random agent (baseline)
    print("\n[3] Testing with random agent (baseline)...")
    
    random_agent = RandomAgent(n_actions=env.action_space.n)
    
    random_fidelities = []
    for _ in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            action = random_agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        random_fidelities.append(info['fidelity'])
    
    print(f"  Average fidelity (random): {np.mean(random_fidelities):.4f}")
    print(f"  Max fidelity (random): {np.max(random_fidelities):.4f}")
    
    # 4. Train a DQN agent (if PyTorch available)
    if TORCH_AVAILABLE:
        print("\n[4] Training DQN agent...")
        
        agent_config = AgentConfig(
            hidden_sizes=[128, 128],
            learning_rate=1e-3,
            epsilon_decay=0.99,
            batch_size=32,
            min_replay_size=100,
        )
        
        agent = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            n_actions=env.action_space.n,
            config=agent_config
        )
        
        training_config = TrainingConfig(
            n_episodes=100,  # Short training for demo
            max_steps_per_episode=15,
            log_interval=20,
            eval_interval=50,
            early_stopping=False,
            save_interval=1000,  # Don't save during this short demo
            save_path="/tmp/compiler_agent"  # Save to temp directory
        )
        
        metrics = train_dqn(env, agent, training_config)
        
        print(f"\n  Training completed!")
        print(f"  Final average fidelity: {np.mean(metrics.episode_fidelities[-20:]):.4f}")
        print(f"  Final success rate: {np.mean(metrics.episode_successes[-20:]):.2%}")
        
        # 5. Compile the Hamiltonian with trained agent
        print("\n[5] Compiling Hamiltonian with trained agent...")
        
        result = compile_hamiltonian(
            hamiltonian=target_H,
            agent=agent,
            env=env,
            evolution_time=0.5,
            max_attempts=3
        )
        
        print(f"  Compilation {'succeeded' if result['success'] else 'failed'}")
        print(f"  Final fidelity: {result['fidelity']:.4f}")
        print(f"  Circuit depth: {result['depth']} gates")
        
        if result['gate_sequence']:
            print(f"\n  Compiled gate sequence:")
            print(result['gate_sequence'].to_string())
    else:
        print("\n[4] Skipping DQN training (PyTorch not available)")
        print("    Install PyTorch to enable deep RL agents: pip install torch")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
