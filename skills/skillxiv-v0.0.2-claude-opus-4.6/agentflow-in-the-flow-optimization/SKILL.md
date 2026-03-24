---
name: agentflow-in-the-flow-optimization
title: "In-the-Flow Agentic System Optimization for Effective Planning and Tool Use"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.05592"
keywords: [agent planning, multi-turn RL, credit assignment, tool use, modular agents]
description: "Decompose agent work across four specialized modules (planner, executor, verifier, generator) coordinated via evolving memory. Use Flow-GRPO to convert multi-turn sparse-reward optimization into sequential single-turn updates with outcome broadcasting, achieving 4-15% accuracy gains on benchmarks while scaling better than monolithic agent policies."
---

# In-the-Flow Agentic System Optimization (AgentFlow)

## Core Concept

Monolithic agent policies struggle with long-horizon tool-use tasks and generalization to new scenarios. AgentFlow decomposes agent responsibilities across four specialized modules—planner, executor, verifier, generator—coordinated through an evolving memory. Flow-GRPO handles sparse multi-turn rewards by broadcasting outcomes to align local decisions with global success, enabling efficient credit assignment without reward decomposition.

## Architecture Overview

- **Modular Components**: Planner (high-level strategy), Executor (action selection), Verifier (quality checks), Generator (text output)
- **Evolving Memory**: Persistent state tracks task context, tool outputs, and decision rationale across turns
- **Flow-GRPO Algorithm**: Converts multi-turn sparse-reward RL into sequential single-turn training by broadcasting episode outcomes
- **Group-Normalized Advantages**: Stabilizes training across heterogeneous tasks and reward signals
- **Scalability**: Demonstrates improved gains with model size (7B baseline, compared against GPT-4o)

## Implementation Steps

### 1. Modular Agent Architecture

Design four specialized modules with clear responsibilities.

```python
class ModularAgent:
    def __init__(self, model_backbone, memory_size=2048):
        # Shared backbone (7B model)
        self.backbone = model_backbone

        # Specialized modules
        self.planner = PlannerModule(self.backbone)        # Strategy generation
        self.executor = ExecutorModule(self.backbone)      # Tool selection
        self.verifier = VerifierModule(self.backbone)      # Verification
        self.generator = GeneratorModule(self.backbone)    # Text generation

        # Evolving memory: persistent state across turns
        self.memory = Memory(max_size=memory_size)

    def step(self, task_instruction, current_observation, memory_context=None):
        """
        Execute one agent step using modular pipeline.
        """

        # Step 1: Planner - high-level strategy
        plan = self.planner.generate_plan(
            task_instruction,
            current_observation,
            memory=self.memory
        )

        # Step 2: Executor - select tool/action
        action = self.executor.select_action(
            plan,
            available_tools,
            memory=self.memory
        )

        # Step 3: Executor runs tool → observe result
        observation = execute_action(action)

        # Step 4: Verifier - quality check
        is_correct = self.verifier.verify(
            action,
            observation,
            plan,
            memory=self.memory
        )

        # Step 5: Generator - text response (if needed)
        response = self.generator.generate_text(
            plan, action, observation, is_correct,
            memory=self.memory
        )

        # Update memory with decision rationale
        self.memory.append({
            'plan': plan,
            'action': action,
            'observation': observation,
            'verification': is_correct,
            'response': response
        })

        return response, observation

class Memory:
    def __init__(self, max_size=2048):
        self.buffer = []
        self.max_size = max_size

    def append(self, decision_record):
        """Add decision to evolving memory."""
        self.buffer.append(decision_record)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def summarize(self):
        """Compact memory for context efficiency."""
        if len(self.buffer) > 100:
            # Summarize old entries
            old_records = self.buffer[:-100]
            summary = self.compact_summary(old_records)
            self.buffer = [summary] + self.buffer[-100:]
        return self.buffer
```

### 2. Flow-GRPO: Credit Assignment for Multi-Turn Tasks

Convert sparse episode-end rewards into sequential single-turn updates via outcome broadcasting.

```python
class FlowGRPO:
    def __init__(self, policy_model, learning_rate=1e-6, group_size=8):
        self.policy = policy_model
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate)
        self.group_size = group_size

    def compute_group_normalized_advantage(self, rewards, eps=1e-8):
        """
        Group-normalize advantages for stability across heterogeneous tasks.
        """
        group_mean = rewards.mean()
        group_std = rewards.std() + eps

        # Normalize to mean 0, std 1 within group
        advantages = (rewards - group_mean) / group_std

        return advantages

    def flow_grpo_step(self, trajectory, episode_reward):
        """
        Multi-turn trajectory with sparse episode-end reward.
        Broadcast outcome backward to align all decisions.

        Args:
            trajectory: List of (observation, action, log_prob) tuples
            episode_reward: Final episode reward (sparse)
        """

        # Broadcast outcome signal to all steps
        # Each step's advantage: episode_reward weighted by step distance
        step_advantages = []
        for step_idx in range(len(trajectory)):
            # Distance decay: closer steps get stronger signal
            distance_weight = 1.0  # Can apply exponential decay if desired
            advantage = episode_reward * distance_weight

            step_advantages.append(advantage)

        # Compute group-normalized advantages
        step_advantages = torch.tensor(step_advantages)
        normalized_advantages = self.compute_group_normalized_advantage(step_advantages)

        # Policy gradient: maximize reward-weighted log probability
        loss = 0
        for step_idx, (obs, action_logit, log_prob) in enumerate(trajectory):
            # Clipped objective: standard PPO-style clipping
            ratio = torch.exp(log_prob - log_prob.detach())
            clipped_ratio = torch.clamp(ratio, 0.9, 1.1)

            advantage = normalized_advantages[step_idx]
            loss += -torch.min(ratio, clipped_ratio) * advantage

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_on_episode_batch(self, episodes, batch_size=32):
        """
        Train on batch of completed episodes (multi-turn trajectories).
        """
        for _ in range(len(episodes) // batch_size):
            batch = episodes[:batch_size]

            for trajectory, episode_reward in batch:
                self.flow_grpo_step(trajectory, episode_reward)
```

### 3. Integration: Multi-Turn Agent Training Loop

Combine modular architecture with Flow-GRPO training.

```python
def train_agentflow(agent, task_environments, num_episodes=1000, num_steps_per_episode=20):
    """
    End-to-end training loop for modular agent with Flow-GRPO.
    """

    flow_grpo = FlowGRPO(agent.backbone)

    for episode in range(num_episodes):
        # Select task randomly
        env = random.choice(task_environments)
        task_instruction = env.get_task()
        observation = env.reset()

        trajectory = []
        total_reward = 0

        # Execute episode
        for step in range(num_steps_per_episode):
            # Agent step
            response, next_observation = agent.step(task_instruction, observation)

            # Record: (observation, action_logit, log_prob)
            trajectory.append((observation, response))

            observation = next_observation

            # Check intermediate rewards (task-specific)
            step_reward = env.get_reward(observation)

            # Final step: episode completion signal
            if step == num_steps_per_episode - 1:
                episode_complete = env.is_episode_complete()
                episode_reward = float(episode_complete)
                total_reward = episode_reward

        # Training: Flow-GRPO update
        flow_grpo.flow_grpo_step(trajectory, total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Task success={total_reward:.1%}")
```

## Performance Results

Evaluation on 10 benchmarks demonstrates consistent improvements:

```python
results = {
    'workbench_tasks': {
        'agentflow': '4.1-14.9% accuracy gains vs baseline',
        'gpt4o_comparison': 'Outperforms on several categories',
        'scaling': 'Gains increase with model scale (7B → 70B)'
    },
    'tool_use_benchmarks': {
        'avg_improvement': '8.5%',
        'generalization': 'Better zero-shot to unseen tool combinations'
    }
}
```

## Practical Guidance

**Module Specialization**: Separate planner (strategic), executor (tactical), and verifier (checking) roles. Specialization improves interpretability and allows targeted improvement.

**Memory Design**: Evolving memory should summarize old decisions to avoid unbounded growth. Compact summaries preserve decision rationale without token explosion.

**Flow-GRPO Configuration**: Group size 8-16 balances variance reduction with representation diversity. Clipping range [0.9, 1.1] controls optimization drift.

**Scaling**: Modular design shows consistent improvements across 7B, 13B, and larger backbones. Specialized modules fine-tune faster than monolithic policies.

## When to Use / When NOT to Use

**Use When**:
- Training agents on extended multi-turn tasks with tool use
- Generalization to unseen task combinations is important
- You need interpretable intermediate decisions (planning, verification)
- Model scaling is planned (modular design improves with size)

**NOT For**:
- Single-turn language tasks (RLHF sufficient)
- Scenarios requiring true end-to-end fine-tuning
- Extremely latency-sensitive deployments (multiple module calls)

## Reference

This skill encodes techniques from "In-the-Flow Agentic System Optimization" (arXiv:2510.05592). Flow-GRPO and modular decomposition enable efficient credit assignment in sparse-reward agent training.
