---
name: info-driven-policy-optimization-agents
title: "InfoPO: Information-Driven Policy Optimization for User-Centric Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.00656"
keywords: [Multi-Turn Agents, Information Gain, Credit Assignment, Reinforcement Learning, User Interaction]
description: "Optimize multi-turn agent policies by measuring turn-level information gain via counterfactual reasoning. Provide dense reward signals identifying which clarifying questions and observations improve the agent's decision distribution, then adaptively blend information rewards with outcome rewards."
---

# InfoPO: Information-Driven Policy Optimization for Multi-Turn Agents

Sparse outcome rewards make early-stage multi-turn agent training inefficient. Standard GRPO-based methods struggle to credit valuable clarification steps that improve downstream decisions but don't directly affect final task success. InfoPO solves this through **turn-level counterfactual information-gain rewards** that identify which observations genuinely shift the agent's action distribution.

The core insight is to measure how much each observation changes what the agent would do next, then reward observations that drive meaningful behavioral shifts. This approach provides dense supervision during exploration and automatically down-weights information once discriminative outcome signals emerge.

## Core Concept

InfoPO treats each interaction turn as a decision point with two scenarios:

1. **Factual**: The agent receives real feedback; its action distribution reflects both prior knowledge and the new observation
2. **Counterfactual**: The feedback is masked; the agent relies only on prior context

The information gain is the KL divergence between these two distributions—a principled measure of how much the observation matters to the agent's thinking.

The method then uses adaptive variance gating to balance two reward sources:
- Early training: Information gain drives learning when outcome signals are sparse
- Late training: Outcome rewards dominate as discriminative feedback emerges

## Architecture Overview

- **Input**: Multi-turn trajectory with observations at each step {o₁, o₂, ..., oₜ}
- **Counterfactual Masking**: For each step, create masked version with observation set to null/padding
- **Policy Evaluation**: Run forward passes to compute action distributions with real and masked observations
- **Information Gain Computation**: Compute KL divergence between the two distributions
- **Variance Gating**: Blend information and outcome rewards based on outcome signal strength
- **Output**: Dense per-turn rewards for policy optimization

## Implementation Steps

**Step 1: Prepare factual and counterfactual trajectories**

For each interaction turn, create two versions of the context: one with the real observation and one with it masked.

```python
def prepare_counterfactual_pair(trajectory, turn_idx):
    """Create factual and counterfactual context for turn t."""
    # Factual: full context including observation at turn t
    factual_context = trajectory[:turn_idx + 1]

    # Counterfactual: mask the observation at turn t
    counterfactual_context = trajectory[:turn_idx]
    # Append masked observation (e.g., <MASKED> token or zero embedding)
    counterfactual_context.append(mask_observation(trajectory[turn_idx]))

    return factual_context, counterfactual_context

# For trajectory with 5 turns, generate 5 factual-counterfactual pairs
trajectory = [context, obs1, obs2, obs3, obs4, obs5]
pairs = [
    prepare_counterfactual_pair(trajectory, t)
    for t in range(1, len(trajectory))
]
```

**Step 2: Forward pass with both versions to get action distributions**

Run the agent policy on both contexts to extract action distributions before the next decision.

```python
def get_action_distribution(context, agent_model):
    """Get softmax distribution over next-step actions given context."""
    logits = agent_model.forward(context)  # Raw logits for action space
    return softmax(logits, axis=-1)

# Evaluate agent on both factual and counterfactual contexts
factual_dist = []
counterfactual_dist = []

for factual_ctx, counterfactual_ctx in pairs:
    # Forward passes (can batch together)
    factual_logits = agent_model.forward(factual_ctx)
    counterfactual_logits = agent_model.forward(counterfactual_ctx)

    # Softmax to get probability distributions
    p_factual = softmax(factual_logits)
    p_counterfactual = softmax(counterfactual_logits)

    factual_dist.append(p_factual)
    counterfactual_dist.append(p_counterfactual)
```

**Step 3: Compute turn-level information gain via KL divergence**

Measure how much the observation changes the agent's next action distribution using reverse KL divergence (from counterfactual to factual).

```python
def kl_divergence(p, q, epsilon=1e-8):
    """KL(p || q) = sum p * (log p - log q)"""
    return np.sum(p * (np.log(p + epsilon) - np.log(q + epsilon)))

# Compute information gain for each turn
information_gains = []

for p_fact, p_cfact in zip(factual_dist, counterfactual_dist):
    # Information gain = KL(factual || counterfactual)
    # High KL = observation strongly influences agent behavior
    ig = kl_divergence(p_fact, p_cfact)
    information_gains.append(ig)

# Normalize information gains to [0, 1] for stable reward scaling
ig_array = np.array(information_gains)
ig_normalized = (ig_array - ig_array.min()) / (ig_array.max() - ig_array.min() + 1e-8)
```

**Step 4: Compute outcome rewards and extract variance**

Evaluate downstream task success and quantify how much outcome variance is explained by different factors.

```python
def compute_outcome_reward(trajectory, task_success, reference_reward=None):
    """Binary outcome reward for task completion."""
    if reference_reward is None:
        reference_reward = 0.0  # Baseline for "no clarification" case

    return float(task_success) - reference_reward

# Collect outcome rewards across all trajectories in batch
trajectory_batch = sample_trajectories(agent, env, batch_size=32)
outcome_rewards = [
    compute_outcome_reward(traj, traj.success)
    for traj in trajectory_batch
]

# Compute variance as proxy for outcome discriminativeness
outcome_var = np.var(outcome_rewards)
outcome_variance_normalized = min(outcome_var / 0.25, 1.0)  # Normalize to ~[0, 1]
```

**Step 5: Adaptive variance-gated reward blending**

Combine information gain and outcome rewards using a gate that tracks outcome signal strength. Early training emphasizes information; late training emphasizes outcomes.

```python
def compute_adaptive_reward(ig_normalized, outcome_reward, outcome_variance,
                           alpha_base=0.5):
    """
    Blend information gain and outcome rewards adaptively.
    alpha = gate strength based on outcome variance.
    """
    # Outcome variance gate: when variance is low (unclear outcomes), trust info gain more
    # When variance is high (clear outcomes), trust outcome more
    alpha = 1.0 - outcome_variance_normalized  # High variance → α close to 0 (outcome focused)

    # Blended reward
    reward = alpha * ig_normalized + (1 - alpha) * outcome_reward

    return reward

# Apply to each turn in trajectory
adaptive_rewards = []
for turn_idx, (ig, outcome) in enumerate(zip(ig_normalized, outcome_rewards)):
    adaptive_r = compute_adaptive_reward(ig, outcome, outcome_variance_normalized)
    adaptive_rewards.append(adaptive_r)

# Use adaptive_rewards as targets for policy gradient (GRPO, REINFORCE, etc.)
```

**Step 6: Execute policy gradient update**

Use the adaptive rewards to train the agent policy via standard RL algorithms.

```python
def policy_gradient_step(agent, trajectory, adaptive_rewards, learning_rate=0.001):
    """Perform policy gradient update using computed rewards."""
    loss = 0.0

    for t, (context, action, reward) in enumerate(zip(trajectory, actions, adaptive_rewards)):
        # Forward: get action logits
        logits = agent.forward(context)
        action_prob = softmax(logits)[action]

        # Policy gradient: maximize log-prob of high-reward actions
        pg_loss = -np.log(action_prob + 1e-8) * reward

        loss += pg_loss

    # Gradient step
    agent.backward(loss)
    agent.optimizer.step(learning_rate)

    return loss.item()
```

## Practical Guidance

**Hyperparameter Selection:**
- **Masking strategy**: Use <MASKED> token or zero embedding; ensure consistency with pretraining
- **Variance threshold for gating**: Calibrate based on task; typically 0.1-0.3 for normalized variance
- **Initial alpha (information weight)**: Start at 0.7-0.8 to emphasize exploration; decreases over training
- **KL epsilon for stability**: 1e-8 standard; increase to 1e-6 for numerical instability

**When to Use:**
- Multi-turn interactive agents (conversational bots, Q&A systems, exploratory search)
- Scenarios where information-seeking steps don't directly improve immediate rewards
- Task settings where early clarification enables late-stage performance (hierarchical reasoning)

**When NOT to Use:**
- Single-turn decision tasks (no intermediate observations to credit)
- Environments with dense immediate rewards (outcome signal already informative)
- Scenarios where masking observations invalidates counterfactuals (state-dependent observations)

**Common Pitfalls:**
- **Masking invalidates policy**: If agent's history is deterministic and observation masked, counterfactual may be unrealistic. Use soft masking (small noise) instead.
- **KL divergence inflation**: Small models may show artificially high KL when outcome rewards are sparse. Regularize with KL upper bound.
- **Variance gate malfunction**: If outcome variance stays near zero (deterministic tasks), gate never activates. For deterministic tasks, use curriculum (introduce noise → reduce noise).
- **Credit misattribution**: Information gain rewards high-entropy observations. Filter out noisy observations explicitly.

## Reference

arXiv: https://arxiv.org/abs/2603.00656
