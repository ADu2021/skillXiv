---
name: era-embodied-agents
title: "ERA: Transforming VLMs into Embodied Agents via Embodied Prior Learning and Online RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.12693"
keywords: [embodied-agents, vla, trajectory-augmentation, online-rl, self-summarization]
description: "Transform vision-language models into embodied agents through two stages: learning embodied priors from trajectory-augmented data with LLM reasoning, then online RL with self-summarization and dense rewards for long-horizon tasks."
---

# ERA: Two-Stage Framework for VLM-to-Agent Transformation

Vision-language models excel at understanding but lack embodied action grounding. ERA transforms VLMs into effective embodied agents through a two-stage approach: first distilling embodied knowledge, then refining with online RL.

Core insight: VLMs can learn embodied reasoning by combining trajectory demonstrations with LLM-generated reasoning, then adapting through online RL with proper credit assignment and context management. Small models (3B) can match or exceed larger models (GPT-4o) on embodied tasks.

## Core Concept

**Embodied Prior Learning**: Augment trajectory data with structured reasoning from stronger models, grounding language understanding in action and environment constraints.

**Online RL Refinement**: Adapt learned priors through RL with three mechanisms: self-summarization for context, dense reward shaping, and turn-level policy optimization.

## Architecture Overview

- **Trajectory Augmentation Engine**: Enriches raw trajectory data with reasoning steps
- **Prior Learning Module**: Fine-tunes VLM on augmented trajectories
- **Online RL Agent**: Explores environment with shaped rewards
- **Context Manager**: Self-summarization for long-horizon state tracking

## Implementation Steps

**Stage 1: Trajectory Augmentation with LLM Reasoning**

Enrich trajectories with structured reasoning from stronger models:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class TrajectoryAugmenter:
    def __init__(self, reasoning_model='claude-opus'):
        self.reasoning_model = reasoning_model
        self.tokenizer = AutoTokenizer.from_pretrained('llama-7b')

    def augment_trajectory(self, trajectory):
        """
        Add reasoning steps to trajectory.
        trajectory: list of (observation, action) pairs
        """

        augmented = []

        for step_idx, (obs, action) in enumerate(trajectory):
            # Get reasoning from strong model
            reasoning = self._generate_reasoning(
                trajectory[:step_idx],
                obs,
                action
            )

            augmented_step = {
                'observation': obs,
                'action': action,
                'reasoning': reasoning,
                'step_idx': step_idx,
                'trajectory_context': trajectory[:step_idx]
            }

            augmented.append(augmented_step)

        return augmented

    def _generate_reasoning(self, history, obs, action):
        """
        Generate reasoning for why this action is taken.
        """

        prompt = f"""
        Environment state: {obs}
        Previous steps: {self._format_history(history)}
        Action taken: {action}

        Why is this action appropriate? Consider:
        - Current goal
        - Environment constraints
        - Long-term implications

        Reasoning:
        """

        reasoning = self.reasoning_model.generate(
            prompt,
            max_tokens=256,
            temperature=0.3
        )

        return reasoning

class EmbodiedPriorLearner(nn.Module):
    def __init__(self, vla_model_name):
        super().__init__()

        self.vla = AutoModelForCausalLM.from_pretrained(vla_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(vla_model_name)

    def forward(self, obs, reasoning, action):
        """
        Learn to predict action given observation and reasoning.
        """

        # Tokenize input
        obs_tokens = self.tokenizer.encode(f"Observation: {obs}")
        reasoning_tokens = self.tokenizer.encode(
            f"Reasoning: {reasoning}"
        )
        action_tokens = self.tokenizer.encode(f"Action: {action}")

        # Concatenate tokens
        input_ids = torch.tensor(
            obs_tokens + reasoning_tokens + action_tokens[:-1]
        )

        # Teacher forcing: predict action tokens
        logits = self.vla(input_ids).logits

        # Compute loss on action tokens
        target = torch.tensor(action_tokens[1:])
        loss = torch.nn.functional.cross_entropy(
            logits[-len(target):],
            target
        )

        return loss

def train_embodied_priors(
    vla,
    augmented_trajectories,
    num_epochs=3,
    lr=5e-5
):
    """
    Train VLA on augmented trajectory data.
    """

    optimizer = torch.optim.AdamW(vla.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for trajectory in augmented_trajectories:
            for step in trajectory:
                loss = vla(
                    step['observation'],
                    step['reasoning'],
                    step['action']
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss / len(augmented_trajectories)}")

    return vla
```

**Stage 2: Online RL with Self-Summarization**

Refine learned priors through environment interaction:

```python
class SelfSummarizer(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()

        self.summarizer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, current_summary, new_observation):
        """
        Update summary with new observation.
        """

        combined = torch.cat([current_summary, new_observation], dim=-1)
        updated_summary = self.summarizer(combined)

        return updated_summary

def run_online_rl_episode(
    vla,
    env,
    goal,
    max_steps=20,
    dense_reward_scale=0.1
):
    """
    Run single episode with online RL training.
    """

    obs = env.reset()
    summary = torch.zeros(768)  # Initialize summary
    episode_trajectories = []

    for step in range(max_steps):
        # Generate action from VLA + summary
        action = vla.generate_action(obs, summary)

        # Execute in environment
        next_obs, task_reward = env.step(action)

        # Compute dense reward
        progress_reward = compute_progress_reward(
            obs,
            next_obs,
            goal
        )

        dense_reward = (
            task_reward +
            dense_reward_scale * progress_reward
        )

        # Self-summarization: update summary with new obs
        summarizer = SelfSummarizer()
        new_summary = summarizer(summary, next_obs)

        # Store for RL update
        trajectory_step = {
            'obs': obs,
            'action': action,
            'reward': dense_reward,
            'next_obs': next_obs,
            'summary': summary,
            'next_summary': new_summary
        }

        episode_trajectories.append(trajectory_step)
        summary = new_summary
        obs = next_obs

        if task_reward > 0.9:  # Task complete
            break

    return episode_trajectories, sum(r['reward'] for r in episode_trajectories)

def rl_training_loop(
    vla,
    env,
    goal,
    num_episodes=50,
    learning_rate=1e-5
):
    """
    Main RL training loop with policy optimization.
    """

    optimizer = torch.optim.AdamW(vla.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        # Run episode
        trajectories, total_reward = run_online_rl_episode(
            vla,
            env,
            goal
        )

        # Compute returns
        returns = []
        R = 0.0
        for trajectory in reversed(trajectories):
            R = trajectory['reward'] + 0.99 * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # Policy gradient update
        log_probs = []
        for trajectory, ret in zip(trajectories, returns):
            action_log_prob = vla.compute_log_prob(
                trajectory['obs'],
                trajectory['summary'],
                trajectory['action']
            )

            log_probs.append(action_log_prob)

        policy_loss = -(torch.stack(log_probs) * returns).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}")
```

## Practical Guidance

**When to Use ERA:**
- Adapting pretrained VLMs to embodied domains
- Long-horizon tasks with sparse rewards (self-summarization helps)
- Resource-constrained scenarios (smaller models competitive)

**When NOT to Use:**
- Short-horizon tasks without memory needs
- Domains without good trajectory data for augmentation
- When full-scale RL from scratch is more efficient

**Augmentation Strategy:**

| Source | Benefit | Data Required |
|--------|---------|---------------|
| Strong LLM reasoning | High-quality reasoning supervision | Good prompts |
| Environment simulator | Diverse trajectories | Simulator access |
| Human demonstrations | Natural action distribution | Human effort |

**RL Hyperparameters:**

| Parameter | Typical Value | Guidance |
|-----------|---------------|----------|
| Dense Reward Scale | 0.05-0.2 | Higher = more emphasis on progress |
| Summary Dimension | 768-1024 | Match VLA hidden dimension |
| Learning Rate | 1e-5 to 5e-5 | Lower = more stable, slower |
| Discount Factor | 0.99 | Standard value for long-horizon |

**Common Pitfalls:**
- Augmentation reasoning too verbose (dilutes signal)
- Dense rewards overshadowing task reward (imbalance)
- Summary forgetting important details (dimension too small)
- Not validating that augmented trajectories match environment dynamics

## Reference

Based on the research at: https://arxiv.org/abs/2510.12693
