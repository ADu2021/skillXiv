---
name: complementary-rl
title: "Complementary Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.17621"
keywords: [Reinforcement Learning, Experience Management, Co-Evolution, Multi-Task Learning]
description: "Improve RL sample efficiency through co-evolution of policy and experience extractor, enabling dynamic experience replay that adapts to the agent's skill level."
---

# Complementary Reinforcement Learning

Traditional reinforcement learning maintains a static experience replay buffer that decouples from the agent's growing capabilities. Early-episode experiences that were useful for a novice agent may become misleading for an expert, yet they remain in the buffer unchanged. Complementary RL solves this fundamental asymmetry by making experience management itself an evolving component.

The core innovation draws inspiration from neuroscience: the brain maintains complementary learning systems where fast (episodic) and slow (semantic) learning co-evolve. Applying this principle, Complementary RL optimizes both the policy and the experience extractor simultaneously, ensuring that stored experiences remain aligned with the agent's current capability level throughout training.

## Core Concept

Complementary RL implements two coupled optimization objectives within a single training loop:

**Policy Actor:** Improved using sparse outcome-based rewards. Traditional RL maximizes task success.

**Experience Extractor:** Trained to identify which experiences demonstrably improve actor performance. Rather than storing all experiences uniformly, the extractor learns *which* experiences are most valuable for the current stage of learning.

The key insight: as the actor improves, the experience extractor adapts to select increasingly sophisticated training data. Early-stage learning focuses on diverse experiences; later-stage learning focuses on edge cases and refinement.

## Architecture Overview

- **Policy Actor**: Standard RL policy parameterized by neural network, optimized on task rewards
- **Experience Extractor**: Auxiliary network that assigns importance weights to trajectories
- **Shared Replay Buffer**: Stores trajectories; extractor determines which to sample
- **Co-Evolution Objective**: Combined loss for policy and extractor prevents gradient conflicts
- **Multi-Task Evaluation**: Track improvements in both single-task and multi-task settings

## Implementation Steps

### Step 1: Define the Complementary Loss

Design a loss that couples policy and experience extractor training.

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class ComplementaryRLTrainer:
    """
    Trainer implementing policy-experience co-evolution.
    Simultaneously optimizes actor and extractor networks.
    """

    def __init__(self, policy_net, extractor_net, learning_rate=1e-4):
        self.policy_net = policy_net
        self.extractor_net = extractor_net
        self.policy_optimizer = Adam(policy_net.parameters(), lr=learning_rate)
        self.extractor_optimizer = Adam(extractor_net.parameters(), lr=learning_rate)

    def compute_actor_loss(self, states, actions, rewards, next_states, dones):
        """
        Standard RL loss (e.g., PPO, A3C).
        For simplicity, using policy gradient with importance weighting.
        """
        log_probs = self.policy_net.compute_log_prob(states, actions)
        value_targets = self._compute_returns(rewards, dones)

        # Policy gradient with importance weighting from extractor
        importance_weights = self.extractor_net(states).detach()  # Detach to prevent gradient collision

        actor_loss = -(log_probs * (value_targets - baseline) * importance_weights).mean()
        return actor_loss

    def compute_extractor_loss(self, states, actions, rewards, improved_actor_metrics):
        """
        Extractor learns to assign high weight to experiences that improved the actor.
        improved_actor_metrics: list of bool indicating which trajectories led to actor improvement
        """
        extractor_weights = self.extractor_net(states)

        # Binary classification: did this experience improve the actor?
        improvement_labels = torch.tensor(improved_actor_metrics, dtype=torch.float32)

        # Extractor loss: high weight -> high improvement probability
        extractor_loss = nn.BCELoss()(extractor_weights.squeeze(), improvement_labels)

        return extractor_loss

    def compute_complementary_loss(self, batch):
        """
        Combined loss that drives co-evolution.
        Balances actor and extractor improvements.
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # Check which trajectories actually improved the actor's performance
        baseline_actor = self.policy_net.clone().eval()
        improved_labels = self._measure_improvement(
            baseline_actor, self.policy_net, states, rewards
        )

        # Co-evolved losses
        actor_loss = self.compute_actor_loss(states, actions, rewards, next_states, dones)
        extractor_loss = self.compute_extractor_loss(states, actions, rewards, improved_labels)

        # Coupling term: penalize if extractor ignores actually-improving experiences
        coupling_penalty = self._coupling_penalty(improved_labels, states)

        total_loss = actor_loss + 0.5 * extractor_loss + 0.1 * coupling_penalty
        return total_loss, {
            'actor_loss': actor_loss.item(),
            'extractor_loss': extractor_loss.item(),
            'coupling_penalty': coupling_penalty.item()
        }

    def _measure_improvement(self, old_actor, new_actor, states, rewards):
        """
        Measure which trajectories improved actor performance.
        Returns: binary labels for each trajectory.
        """
        with torch.no_grad():
            old_values = old_actor(states)
            new_values = new_actor(states)

        # Improvement: new value estimate higher than old
        improvement = new_values > old_values
        return improvement.cpu().numpy()

    def _coupling_penalty(self, improved_labels, states):
        """
        Penalize if extractor assigns low weight to improving experiences.
        Encourages alignment between extraction and actual improvement.
        """
        extractor_weights = self.extractor_net(states)
        improved_tensor = torch.tensor(improved_labels, dtype=torch.float32)

        # For experiences marked as improving, weight should be high
        penalty = torch.abs(extractor_weights - improved_tensor).mean()
        return penalty

    def update(self, batch):
        """Single training step."""
        total_loss, losses = self.compute_complementary_loss(batch)

        self.policy_optimizer.zero_grad()
        self.extractor_optimizer.zero_grad()

        total_loss.backward()

        self.policy_optimizer.step()
        self.extractor_optimizer.step()

        return losses
```

### Step 2: Dynamic Experience Sampling

Use the extractor to adaptively sample from replay buffer.

```python
class DynamicReplayBuffer:
    """
    Experience replay buffer with adaptive sampling.
    Extractor network determines which experiences to prioritize.
    """

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.importance_scores = []

    def add_experience(self, state, action, reward, next_state, done):
        """Store experience with initial importance score."""
        experience = (state, action, reward, next_state, done)
        initial_score = 0.5  # Start neutral

        if len(self.buffer) >= self.capacity:
            # Remove oldest experience
            self.buffer.pop(0)
            self.importance_scores.pop(0)

        self.buffer.append(experience)
        self.importance_scores.append(initial_score)

    def update_importance_scores(self, extractor_net, num_samples=1000):
        """
        Update importance scores using extractor network.
        Experiences predicted to improve actor get higher priority.
        """
        if len(self.buffer) == 0:
            return

        # Sample batch to compute importance
        indices = torch.randint(0, len(self.buffer), (num_samples,))
        states = torch.stack([torch.tensor(self.buffer[i][0]) for i in indices])

        with torch.no_grad():
            new_scores = extractor_net(states).squeeze().cpu().numpy()

        # Update scores for sampled experiences
        for idx, score in zip(indices, new_scores):
            self.importance_scores[idx] = float(score)

    def sample_batch(self, batch_size=32):
        """
        Sample batch prioritized by importance scores.
        Higher-scored experiences more likely to be selected.
        """
        # Normalize scores to probabilities
        scores = np.array(self.importance_scores)
        probs = scores / (scores.sum() + 1e-8)

        # Weighted random sampling
        indices = np.random.choice(
            len(self.buffer), size=batch_size, p=probs, replace=False
        )

        batch = {
            'states': torch.stack([torch.tensor(self.buffer[i][0]) for i in indices]),
            'actions': torch.stack([torch.tensor(self.buffer[i][1]) for i in indices]),
            'rewards': torch.stack([torch.tensor(self.buffer[i][2]) for i in indices]),
            'next_states': torch.stack([torch.tensor(self.buffer[i][3]) for i in indices]),
            'dones': torch.stack([torch.tensor(self.buffer[i][4]) for i in indices])
        }

        return batch, indices
```

### Step 3: Training Loop with Co-Evolution

Integrate complementary optimization into the main training loop.

```python
def train_complementary_rl(env, policy_net, extractor_net, num_episodes=1000):
    """
    Main training loop with complementary RL.
    Alternates between experience collection and co-evolved optimization.
    """
    trainer = ComplementaryRLTrainer(policy_net, extractor_net)
    replay_buffer = DynamicReplayBuffer(capacity=100000)

    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        # Collect experience
        for step in range(1000):  # Max steps per episode
            # Policy action selection
            action = policy_net.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store in replay buffer
            replay_buffer.add_experience(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        # Update importance scores periodically (every 10 episodes)
        if episode % 10 == 0:
            replay_buffer.update_importance_scores(trainer.extractor_net)

        # Training updates
        for update_step in range(50):
            batch, indices = replay_buffer.sample_batch(batch_size=32)
            losses = trainer.update(batch)

            if update_step % 10 == 0:
                print(f"Episode {episode}, Update {update_step}: Actor Loss = {losses['actor_loss']:.4f}, "
                      f"Extractor Loss = {losses['extractor_loss']:.4f}")

        # Log progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_importance = np.mean(replay_buffer.importance_scores)
            print(f"\nEpisode {episode}: Avg Reward = {avg_reward:.2f}, Avg Importance = {avg_importance:.3f}")

    return policy_net, episode_rewards
```

### Step 4: Multi-Task Evaluation

Validate that complementary RL generalizes across task variations.

```python
def evaluate_complementary_rl(policy_net, extractor_net, task_variants=5):
    """
    Evaluate on multiple task variants to verify generalization.
    Complements single-task gains with multi-task robustness.
    """
    results = {}

    for task_id in range(task_variants):
        env = create_task_variant(task_id)
        cumulative_reward = 0

        state = env.reset()
        done = False

        while not done:
            action = policy_net.select_action(state)
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward

        results[f'task_{task_id}'] = cumulative_reward

    avg_performance = np.mean(list(results.values()))
    print(f"Multi-task Performance: {results}")
    print(f"Average: {avg_performance:.2f}")

    return results
```

## Practical Guidance

**Hyperparameters:**
- Extractor network capacity: 1-2 layers, 64-256 hidden units (keep small relative to policy)
- Importance score update frequency: every 5-20 episodes (balance freshness vs. stability)
- Coupling penalty weight: 0.05-0.2 (controls co-evolution strength)
- Experience extractor learning rate: 2-10x lower than policy (slower adaptation)

**When to Use:**
- Multi-task or lifelong learning scenarios where task distribution changes
- Environments where early and late-stage learning require different data
- When you need sample efficiency across diverse task difficulties
- Robotics and control: agent needs both diverse exploration and refined exploitation

**When NOT to Use:**
- Single well-defined task with fixed training distribution
- Environments requiring strict replay buffer size (dynamic priorities add overhead)
- Real-time systems where extraction network inference adds latency
- Highly stochastic environments where causality between experience and improvement is unclear

**Pitfalls:**
- Feedback loops: if extractor is poorly initialized, it may eliminate important early-stage experiences
- Coupling instability: aggressive coupling penalties can cause policy-extractor conflict; tune conservatively
- Non-stationary labels: "improvement" changes as the policy evolves; update labels frequently
- Multi-task drift: ensure evaluation set remains fixed to measure true improvement

## Reference

Paper: [arxiv.org/abs/2603.17621](https://arxiv.org/abs/2603.17621)
