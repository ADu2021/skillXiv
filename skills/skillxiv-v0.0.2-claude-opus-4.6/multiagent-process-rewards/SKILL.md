---
name: multiagent-process-rewards
title: "Scaling Multiagent Systems with Process Rewards"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.23228"
keywords: [Multiagent Systems, Process Rewards, Credit Assignment, REINFORCE, Pipeline Agents]
description: "Train specialized agents in pipelines using dense per-action process rewards from AI coaching. Solves credit assignment in sequential workflows, enabling better generalization and faster convergence than outcome-only training."
---

# MAPPA: Process Rewards for Multiagent Pipelines

## Problem
Multiagent pipelines suffer from credit assignment problems. When downstream agents encounter errors from upstream work, outcome-based rewards penalize the reporting agent rather than the source of error. This creates incorrect learning signals.

GRPO assumes consistent rollouts from fixed initial states, but multiagent systems produce different intermediate states from the same initial prompt due to upstream stochasticity.

## Core Concept
MAPPA (MultiAgent Per-Action Process rewards) trains agents using dense 0-10 scale quality ratings from a coach LLM that evaluates each intermediate action. The coach assesses actions holistically considering role, input, action, and execution results.

This provides correct attribution—upstream errors receive low scores directly, and downstream agents can report issues without penalty.

## Architecture Overview

- **Coach LLM**: Evaluates each agent action independently on 0-10 quality scale
- **Per-Action Supervision**: Every intermediate step gets explicit quality feedback
- **Role-Aware Evaluation**: Coach considers agent's role and input context
- **Tool Execution Verification**: Coach validates whether actions achieved intended effects
- **REINFORCE++ Training**: Global batch normalization instead of GRPO (accounts for stochastic inputs)

## Implementation

### Step 1: Define Coach Evaluation Prompt
Structure coach LLM to provide consistent 0-10 quality ratings.

```python
def create_coach_evaluation_prompt(agent_role, input_context, action, execution_result):
    """Create prompt for coach to evaluate action quality."""
    prompt = f"""Evaluate this action on a scale of 0-10.

Agent Role: {agent_role}
Input Context: {input_context}
Action: {action}
Execution Result: {execution_result}

Rate the quality of this action considering:
1. Correctness: Did it achieve intended effect?
2. Efficiency: Did it take optimal approach?
3. Robustness: Did it handle edge cases?

Score (0-10):"""

    return prompt
```

### Step 2: Collect Per-Action Rewards
Call coach for each agent action to obtain dense supervision.

```python
def collect_process_rewards(agent_trajectory, coach_model):
    """Collect per-action rewards from coach for entire trajectory."""
    rewards = []

    for step in agent_trajectory:
        # Get coach evaluation
        eval_prompt = create_coach_evaluation_prompt(
            step['agent_role'],
            step['input'],
            step['action'],
            step['result']
        )

        evaluation = coach_model.evaluate(eval_prompt)
        score = extract_score(evaluation)  # Extract 0-10 from response
        rewards.append(score)

    return rewards
```

### Step 3: Account for Stochastic Inputs
Use REINFORCE++ instead of GRPO due to different intermediate states.

```python
def reinforce_plus_plus_training(agent, trajectory_batch, process_rewards_batch, learning_rate=1e-4):
    """Train with REINFORCE++ using global batch normalization."""
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    # Compute global batch mean/std for normalization
    all_rewards = torch.cat(process_rewards_batch)
    reward_mean = all_rewards.mean()
    reward_std = all_rewards.std() + 1e-6

    # Normalize rewards globally
    normalized_rewards = [(r - reward_mean) / reward_std for r in process_rewards_batch]

    for trajectory, norm_rewards in zip(trajectory_batch, normalized_rewards):
        # Compute policy gradient
        for action, reward in zip(trajectory, norm_rewards):
            log_prob = agent.compute_log_prob(action)
            loss = -reward * log_prob
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
```

### Step 4: Multi-Agent Training Loop
Coordinate training across specialized agents.

```python
def train_multiagent_system(agents, data_stream, num_iterations=1000):
    """Train multiple specialized agents with process rewards."""
    for iteration in range(num_iterations):
        # Process one batch from data stream
        batch = data_stream.get_batch(batch_size=16)

        trajectories = []
        all_process_rewards = []

        # Execute pipeline and collect trajectories
        current_input = batch
        for agent_idx, agent in enumerate(agents):
            outputs, trajectory = agent.execute(current_input)
            trajectories.append(trajectory)

            # Get process rewards for this agent
            process_rewards = collect_process_rewards(trajectory, coach_model)
            all_process_rewards.append(process_rewards)

            current_input = outputs

        # Train each agent with its process rewards
        for agent, trajectory, rewards in zip(agents, trajectories, all_process_rewards):
            reinforce_plus_plus_training(agent, [trajectory], [rewards])
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Coach rating scale | 0-10 | Granular feedback signal |
| Reward normalization | Global batch | Accounts for stochastic inputs |
| REINFORCE learning rate | 1e-4 to 5e-4 | Stable policy optimization |
| Batch size | 16-32 | Process reward collection overhead |
| Coach model | GPT-4 or equivalent | Quality and consistency critical |

### When to Use

- Multiagent pipelines (math solving, data analysis, research)
- Tasks with clear intermediate checkpoints
- When upstream errors create downstream issues
- Systems with specialized agent roles
- Scaling beyond single large model capabilities

### When Not to Use

- Single-agent tasks (standard RL sufficient)
- Environments without clear intermediate feedback points
- When human expert evaluation is feasible (use direct human ratings)
- Tightly coupled agents where actions are interdependent

### Common Pitfalls

1. **Inconsistent coach evaluation**: Coach must apply consistent criteria. Validate on ground truth.
2. **Inappropriate reward scale**: 0-10 scale must match domain difficulty. Pilot test on known examples.
3. **Ignoring input stochasticity**: Using GRPO despite different intermediate states biases learning.
4. **Sparse intermediate steps**: Agents need sufficient actions to develop distinct policies per role.

## Reference
Scaling Multiagent Systems with Process Rewards
https://arxiv.org/abs/2601.23228
