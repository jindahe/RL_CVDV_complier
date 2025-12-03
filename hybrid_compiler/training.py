"""
Training utilities for hybrid CV-DV Hamiltonian compiler.

This module provides training loops and utilities for training RL agents
to compile quantum Hamiltonians.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
import time
import logging

from .environment import HybridCompilerEnv, CompilationConfig, RandomTargetWrapper
from .agent import DQNAgent, PolicyGradientAgent, RandomAgent, AgentConfig
from .hamiltonian import Hamiltonian, HamiltonianBuilder
from .gates import GateSequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training duration
    n_episodes: int = 1000
    max_steps_per_episode: int = 50
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    n_eval_episodes: int = 5
    
    # Saving
    save_interval: int = 100
    save_path: str = "models/compiler_agent"
    
    # Early stopping
    early_stopping: bool = True
    target_success_rate: float = 0.9
    patience: int = 100
    
    # Evolution time range for random Hamiltonians
    evolution_time_min: float = 0.1
    evolution_time_max: float = 2.0


class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_fidelities: List[float] = []
        self.episode_successes: List[bool] = []
        self.losses: List[float] = []
        self.eval_rewards: List[float] = []
        self.eval_fidelities: List[float] = []
        self.eval_success_rates: List[float] = []
    
    def log_episode(
        self,
        reward: float,
        length: int,
        fidelity: float,
        success: bool
    ):
        """Log metrics for a single episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_fidelities.append(fidelity)
        self.episode_successes.append(success)
    
    def log_loss(self, loss: float):
        """Log training loss."""
        self.losses.append(loss)
    
    def log_eval(
        self,
        avg_reward: float,
        avg_fidelity: float,
        success_rate: float
    ):
        """Log evaluation metrics."""
        self.eval_rewards.append(avg_reward)
        self.eval_fidelities.append(avg_fidelity)
        self.eval_success_rates.append(success_rate)
    
    def get_recent_stats(self, n: int = 100) -> Dict[str, float]:
        """Get statistics over the last n episodes."""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-n:]
        recent_lengths = self.episode_lengths[-n:]
        recent_fidelities = self.episode_fidelities[-n:]
        recent_successes = self.episode_successes[-n:]
        
        return {
            "avg_reward": np.mean(recent_rewards),
            "avg_length": np.mean(recent_lengths),
            "avg_fidelity": np.mean(recent_fidelities),
            "success_rate": np.mean(recent_successes),
            "std_reward": np.std(recent_rewards),
        }


def train_dqn(
    env: HybridCompilerEnv,
    agent: DQNAgent,
    config: TrainingConfig,
    hamiltonian_generator: Optional[Callable[[], Hamiltonian]] = None
) -> TrainingMetrics:
    """
    Train a DQN agent on the Hamiltonian compilation task.
    
    Args:
        env: The compilation environment
        agent: DQN agent to train
        config: Training configuration
        hamiltonian_generator: Optional function to generate random Hamiltonians
    
    Returns:
        Training metrics
    """
    metrics = TrainingMetrics()
    best_success_rate = 0.0
    no_improvement_count = 0
    
    logger.info("Starting DQN training...")
    start_time = time.time()
    
    for episode in range(config.n_episodes):
        # Generate new target if generator provided
        options = {}
        if hamiltonian_generator is not None:
            options['hamiltonian'] = hamiltonian_generator()
            options['evolution_time'] = np.random.uniform(
                config.evolution_time_min, config.evolution_time_max
            )
        
        state, info = env.reset(options=options)
        
        episode_reward = 0.0
        episode_length = 0
        final_fidelity = info.get('fidelity', 0.0)
        success = False
        
        for step in range(config.max_steps_per_episode):
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                metrics.log_loss(loss)
            
            episode_reward += reward
            episode_length += 1
            final_fidelity = info.get('fidelity', 0.0)
            success = info.get('success', False)
            
            state = next_state
            
            if done:
                break
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Log episode
        metrics.log_episode(episode_reward, episode_length, final_fidelity, success)
        
        # Periodic logging
        if (episode + 1) % config.log_interval == 0:
            stats = metrics.get_recent_stats(config.log_interval)
            logger.info(
                f"Episode {episode + 1}/{config.n_episodes} | "
                f"Reward: {stats['avg_reward']:.2f} | "
                f"Fidelity: {stats['avg_fidelity']:.4f} | "
                f"Success: {stats['success_rate']:.2%} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
        
        # Periodic evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_metrics = evaluate_agent(
                env, agent, config.n_eval_episodes, hamiltonian_generator
            )
            metrics.log_eval(
                eval_metrics['avg_reward'],
                eval_metrics['avg_fidelity'],
                eval_metrics['success_rate']
            )
            
            logger.info(
                f"Evaluation | "
                f"Reward: {eval_metrics['avg_reward']:.2f} | "
                f"Fidelity: {eval_metrics['avg_fidelity']:.4f} | "
                f"Success: {eval_metrics['success_rate']:.2%}"
            )
            
            # Check for improvement
            if eval_metrics['success_rate'] > best_success_rate:
                best_success_rate = eval_metrics['success_rate']
                no_improvement_count = 0
                # Save best model
                agent.save(config.save_path + "_best.pt")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if config.early_stopping:
                if eval_metrics['success_rate'] >= config.target_success_rate:
                    logger.info(f"Reached target success rate! Stopping early.")
                    break
                if no_improvement_count >= config.patience:
                    logger.info(f"No improvement for {config.patience} evaluations. Stopping.")
                    break
        
        # Periodic saving
        if (episode + 1) % config.save_interval == 0:
            agent.save(config.save_path + f"_ep{episode + 1}.pt")
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s")
    logger.info(f"Best success rate: {best_success_rate:.2%}")
    
    return metrics


def train_policy_gradient(
    env: HybridCompilerEnv,
    agent: PolicyGradientAgent,
    config: TrainingConfig,
    hamiltonian_generator: Optional[Callable[[], Hamiltonian]] = None
) -> TrainingMetrics:
    """
    Train a Policy Gradient agent on the Hamiltonian compilation task.
    
    Args:
        env: The compilation environment
        agent: Policy Gradient agent to train
        config: Training configuration
        hamiltonian_generator: Optional function to generate random Hamiltonians
    
    Returns:
        Training metrics
    """
    metrics = TrainingMetrics()
    best_success_rate = 0.0
    no_improvement_count = 0
    
    logger.info("Starting Policy Gradient training...")
    start_time = time.time()
    
    for episode in range(config.n_episodes):
        # Generate new target if generator provided
        options = {}
        if hamiltonian_generator is not None:
            options['hamiltonian'] = hamiltonian_generator()
            options['evolution_time'] = np.random.uniform(
                config.evolution_time_min, config.evolution_time_max
            )
        
        state, info = env.reset(options=options)
        
        episode_reward = 0.0
        episode_length = 0
        final_fidelity = info.get('fidelity', 0.0)
        success = False
        
        for step in range(config.max_steps_per_episode):
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store reward
            agent.store_reward(reward)
            
            episode_reward += reward
            episode_length += 1
            final_fidelity = info.get('fidelity', 0.0)
            success = info.get('success', False)
            
            state = next_state
            
            if done:
                break
        
        # Train at end of episode
        loss = agent.train_episode()
        if loss is not None:
            metrics.log_loss(loss)
        
        # Log episode
        metrics.log_episode(episode_reward, episode_length, final_fidelity, success)
        
        # Periodic logging
        if (episode + 1) % config.log_interval == 0:
            stats = metrics.get_recent_stats(config.log_interval)
            logger.info(
                f"Episode {episode + 1}/{config.n_episodes} | "
                f"Reward: {stats['avg_reward']:.2f} | "
                f"Fidelity: {stats['avg_fidelity']:.4f} | "
                f"Success: {stats['success_rate']:.2%}"
            )
        
        # Periodic evaluation
        if (episode + 1) % config.eval_interval == 0:
            eval_metrics = evaluate_agent(
                env, agent, config.n_eval_episodes, hamiltonian_generator
            )
            metrics.log_eval(
                eval_metrics['avg_reward'],
                eval_metrics['avg_fidelity'],
                eval_metrics['success_rate']
            )
            
            # Check for improvement and early stopping (same as DQN)
            if eval_metrics['success_rate'] > best_success_rate:
                best_success_rate = eval_metrics['success_rate']
                no_improvement_count = 0
                agent.save(config.save_path + "_best.pt")
            else:
                no_improvement_count += 1
            
            if config.early_stopping and no_improvement_count >= config.patience:
                logger.info(f"No improvement for {config.patience} evaluations. Stopping.")
                break
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s")
    
    return metrics


def evaluate_agent(
    env: HybridCompilerEnv,
    agent,
    n_episodes: int = 10,
    hamiltonian_generator: Optional[Callable[[], Hamiltonian]] = None
) -> Dict[str, float]:
    """
    Evaluate an agent without training.
    
    Args:
        env: The compilation environment
        agent: Agent to evaluate (DQN, PolicyGradient, or Random)
        n_episodes: Number of evaluation episodes
        hamiltonian_generator: Optional function to generate random Hamiltonians
    
    Returns:
        Dictionary of evaluation metrics
    """
    rewards = []
    fidelities = []
    successes = []
    lengths = []
    
    for _ in range(n_episodes):
        options = {}
        if hamiltonian_generator is not None:
            options['hamiltonian'] = hamiltonian_generator()
            options['evolution_time'] = np.random.uniform(0.1, 2.0)
        
        state, info = env.reset(options=options)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if episode_length >= env.config.max_gates:
                break
        
        rewards.append(episode_reward)
        fidelities.append(info.get('fidelity', 0.0))
        successes.append(info.get('success', False))
        lengths.append(episode_length)
    
    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_fidelity': np.mean(fidelities),
        'std_fidelity': np.std(fidelities),
        'success_rate': np.mean(successes),
        'avg_length': np.mean(lengths)
    }


def compile_hamiltonian(
    hamiltonian: Hamiltonian,
    agent,
    env: HybridCompilerEnv,
    evolution_time: float = 1.0,
    max_attempts: int = 5
) -> Dict[str, Any]:
    """
    Compile a Hamiltonian using a trained agent.
    
    Args:
        hamiltonian: Target Hamiltonian
        agent: Trained agent
        env: Compilation environment
        evolution_time: Evolution time for target unitary
        max_attempts: Maximum compilation attempts
    
    Returns:
        Dictionary with compilation results
    """
    best_fidelity = 0.0
    best_sequence = None
    
    for attempt in range(max_attempts):
        state, info = env.reset(options={
            'hamiltonian': hamiltonian,
            'evolution_time': evolution_time
        })
        
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        fidelity = info.get('fidelity', 0.0)
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_sequence = env.get_compiled_sequence()
        
        if info.get('success', False):
            break
    
    return {
        'success': best_fidelity >= env.config.target_fidelity,
        'fidelity': best_fidelity,
        'gate_sequence': best_sequence,
        'depth': len(best_sequence) if best_sequence else 0,
        'target_hamiltonian': hamiltonian,
        'evolution_time': evolution_time
    }
