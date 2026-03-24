---
name: dreamgym-experience-synthesis-rl
title: "Scaling Agent Learning via Experience Synthesis"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.03773"
keywords: [Experience Synthesis, Reinforcement Learning, World Models, Agent Training, Simulation]
description: "Scale agent learning by synthesizing diverse experiences using reasoning-based models instead of costly real-world rollouts, maintaining replay buffers with both real and synthetic interactions while using adaptive curriculum to focus on challenging tasks."
---

# Title: Train Agents Efficiently With Synthesized Experience From World Models

Reinforcement learning typically requires millions of real-world interactions, which is expensive and slow. DreamGym solves this by training a reasoning-based experience model that generates synthetic rollouts: step-by-step predictions of state transitions and rewards. These synthetic experiences augment offline real-world data, enabling RL to converge faster with fewer real interactions.

The approach combines three components: experience model (synthesizes dynamics), replay buffer management (balances real and synthetic), and curriculum learning (focuses on useful tasks).

## Core Concept

**Reasoning-Based Experience Generation for RL**:
- **Experience Model**: Uses step-by-step reasoning to predict state transitions
- **Synthetic Rollout Generation**: Creates full trajectory sequences without environment interaction
- **Hybrid Replay Buffer**: Initialized with real data, continuously enriched with synthetic experiences
- **Adaptive Curriculum**: Actively generates challenging tasks matched to current agent capability
- **Efficient RL**: Train on diverse synthetic experiences, occasionally validate on real environment

## Architecture Overview

- **Experience Model**: Reasoning-based predictor of (state, action) → (next_state, reward)
- **Replay Buffer Manager**: Maintains real and synthetic experience ratio
- **Curriculum Controller**: Determines which synthetic tasks to generate
- **Policy Network**: Standard RL agent trained on combined data
- **Validation Loop**: Periodic real-world evaluation to ensure transfer

## Implementation Steps

**1. Build Reasoning-Based Experience Model**

Create a model that generates realistic synthetic experiences using step-by-step reasoning.

```python
class ReasoningBasedExperienceModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

        # Transformer for reasoning
        self.reasoning_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=6
        )

        # Output heads
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def predict_step(self, state, action):
        """Predict (next_state, reward, done) from (state, action)"""
        # Encode inputs
        state_emb = self.state_encoder(state)
        action_emb = self.action_encoder(action)

        # Combine with reasoning
        combined = state_emb + action_emb
        reasoning = self.reasoning_transformer(combined.unsqueeze(1)).squeeze(1)

        # Predict outputs
        next_state = self.next_state_head(reasoning)
        reward = self.reward_head(reasoning)
        done = torch.sigmoid(self.done_head(reasoning))

        return next_state, reward, done

    def generate_trajectory(self, initial_state, policy, max_steps=100):
        """Generate full trajectory using reasoning"""
        states = [initial_state]
        actions = []
        rewards = []
        dones = []

        state = initial_state
        for step in range(max_steps):
            # Policy selects action
            action = policy.select_action(state)
            actions.append(action)

            # Experience model predicts outcome
            next_state, reward, done = self.predict_step(state, action)

            states.append(next_state.detach())
            rewards.append(reward.item())
            dones.append(done.item() > 0.5)

            if done.item() > 0.5:
                break

            state = next_state

        return {
            'states': torch.stack(states),
            'actions': torch.stack(actions),
            'rewards': torch.tensor(rewards),
            'dones': torch.tensor(dones)
        }

    def train(self, real_transitions, learning_rate=1e-4):
        """Train experience model on real transitions"""
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for state, action, next_state, reward, done in real_transitions:
            pred_next, pred_reward, pred_done = self.predict_step(state, action)

            # Supervised loss
            loss_state = F.mse_loss(pred_next, next_state)
            loss_reward = F.mse_loss(pred_reward, reward.unsqueeze(-1))
            loss_done = F.binary_cross_entropy(pred_done, done.unsqueeze(-1))

            total_loss = loss_state + loss_reward + loss_done

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

**2. Implement Hybrid Replay Buffer**

Manage mix of real and synthetic experiences.

```python
class HybridReplayBuffer:
    def __init__(self, capacity=100000, real_ratio=0.5):
        self.real_buffer = []
        self.synthetic_buffer = []
        self.capacity = capacity
        self.real_ratio = real_ratio

    def add_real_experience(self, transition):
        """Add actual environment interaction"""
        self.real_buffer.append(transition)
        if len(self.real_buffer) > self.capacity * self.real_ratio:
            self.real_buffer.pop(0)

    def add_synthetic_experience(self, trajectory):
        """Add generated experience from model"""
        # Convert trajectory to transitions
        for i in range(len(trajectory['states']) - 1):
            transition = {
                'state': trajectory['states'][i],
                'action': trajectory['actions'][i],
                'reward': trajectory['rewards'][i],
                'next_state': trajectory['states'][i + 1],
                'done': trajectory['dones'][i]
            }
            self.synthetic_buffer.append(transition)

        # Maintain capacity
        max_synthetic = self.capacity * (1 - self.real_ratio)
        while len(self.synthetic_buffer) > max_synthetic:
            self.synthetic_buffer.pop(0)

    def sample_batch(self, batch_size=32):
        """Sample balanced batch from both buffers"""
        # Proportion real and synthetic
        num_real = int(batch_size * self.real_ratio)
        num_synthetic = batch_size - num_real

        real_batch = random.sample(self.real_buffer, min(num_real, len(self.real_buffer)))
        synthetic_batch = random.sample(self.synthetic_buffer, min(num_synthetic, len(self.synthetic_buffer)))

        batch = real_batch + synthetic_batch
        random.shuffle(batch)

        return {
            'states': torch.stack([t['state'] for t in batch]),
            'actions': torch.stack([t['action'] for t in batch]),
            'rewards': torch.stack([t['reward'] for t in batch]),
            'next_states': torch.stack([t['next_state'] for t in batch]),
            'dones': torch.stack([t['done'] for t in batch])
        }
```

**3. Implement Adaptive Curriculum**

Generate synthetic tasks tailored to agent capability.

```python
class AdaptiveCurriculum:
    def __init__(self, experience_model, policy, initial_tasks):
        self.experience_model = experience_model
        self.policy = policy
        self.completed_tasks = set()
        self.task_difficulty = defaultdict(float)

    def estimate_agent_capability(self):
        """Assess current policy strength"""
        # Test on standard benchmarks
        success_rates = {}
        for task in self.completed_tasks:
            trajectory = self.experience_model.generate_trajectory(
                task['initial_state'], self.policy
            )
            success = self.evaluate_trajectory(trajectory, task)
            success_rates[task] = success

        avg_success = np.mean(list(success_rates.values())) if success_rates else 0.5
        return avg_success

    def generate_curriculum_tasks(self, num_tasks=10):
        """Generate tasks at appropriate difficulty"""
        agent_capability = self.estimate_agent_capability()

        # Target 70% success rate
        target_difficulty = agent_capability + 0.2

        new_tasks = []
        for _ in range(num_tasks):
            # Sample difficulty
            difficulty = target_difficulty + np.random.normal(0, 0.1)
            difficulty = np.clip(difficulty, 0.0, 1.0)

            # Generate task at this difficulty
            task = self.generate_task_at_difficulty(difficulty)
            new_tasks.append(task)

        return new_tasks

    def generate_task_at_difficulty(self, difficulty):
        """Procedurally generate task with specified difficulty"""
        # Difficulty controls: obstacle density, goal distance, etc.
        task = {
            'initial_state': self.sample_initial_state(difficulty),
            'difficulty': difficulty,
            'goal': self.sample_goal(difficulty)
        }
        return task
```

**4. Integrate Experience Synthesis into RL Training Loop**

Combine model learning, experience generation, and policy optimization.

```python
def train_with_experience_synthesis(
    experience_model, policy, optimizer,
    real_env, initial_buffer,
    num_iterations=1000
):
    replay_buffer = HybridReplayBuffer()

    # Populate with initial real data
    for transition in initial_buffer:
        replay_buffer.add_real_experience(transition)

    curriculum = AdaptiveCurriculum(experience_model, policy, [])

    for iteration in range(num_iterations):
        # Phase 1: Collect real data (occasionally)
        if iteration % 10 == 0:
            real_trajectory = collect_real_trajectory(real_env, policy)
            for transition in real_trajectory:
                replay_buffer.add_real_experience(transition)

        # Phase 2: Train experience model on recent real data
        real_batch = [t for t in replay_buffer.real_buffer[-100:]]
        experience_model.train(real_batch)

        # Phase 3: Generate synthetic experiences
        curriculum_tasks = curriculum.generate_curriculum_tasks(num_tasks=5)
        for task in curriculum_tasks:
            synthetic_trajectory = experience_model.generate_trajectory(
                task['initial_state'], policy
            )
            replay_buffer.add_synthetic_experience(synthetic_trajectory)

        # Phase 4: Train policy on mixed batch
        for _ in range(10):
            batch = replay_buffer.sample_batch(batch_size=32)
            policy_loss = compute_rl_loss(policy, batch)

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        # Phase 5: Validation on real environment
        if iteration % 100 == 0:
            real_eval = evaluate_on_real_env(policy, real_env)
            print(f"Iteration {iteration}: Real performance {real_eval:.2%}")
```

## Practical Guidance

**When to Use**:
- Limited real environment interaction budget
- Expensive real-world rollouts (robotics, simulation)
- Tasks where experience model can be accurate

**Hyperparameters**:
- real_ratio: 0.5 (balance real and synthetic)
- curriculum_difficulty_step: 0.2 per update
- experience_model_train_freq: Every 10 policy updates

**When NOT to Use**:
- Environments where model predictions are unreliable
- High-dimensional state spaces (vision) where models struggle
- Tasks requiring perfect memory of environment dynamics

**Pitfalls**:
- **Model drift**: Experience model diverges from real environment; validate predictions regularly
- **Distribution mismatch**: Synthetic data drawn from different distribution than real; use importance weighting
- **Compounding errors**: Multi-step predictions accumulate errors; keep max trajectory length reasonable

**Key Insight**: This only works if the experience model is reasonably accurate. Validate model performance on held-out real transitions frequently.

## Reference

arXiv: https://arxiv.org/abs/2511.03773
