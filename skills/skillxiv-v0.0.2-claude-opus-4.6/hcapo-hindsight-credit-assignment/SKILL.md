---
name: hcapo-hindsight-credit-assignment
title: "Hindsight Credit Assignment for Long-Horizon LLM Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.08754"
keywords: [Credit Assignment, RL, Long-Horizon, LLM, Agents, GRPO]
description: "Compute step-level credit assignments via hindsight generative verification: condition the LLM on successful outcomes to compute importance ratios that filter credit by causal relevance. Synergizes macro stability with micro precision."
---

# Technique: Hindsight Importance Ratios for Causal Step-Level Credit

Long-horizon tasks suffer from **credit assignment myopia**: trajectory-level rewards don't reveal which intermediate steps were responsible for success. HCAPO addresses this by using the LLM itself as a post-hoc critic: it conditions on successful outcomes to estimate **hindsight importance ratios** that amplify credit for causally relevant steps while suppressing less instrumental ones.

Rather than training separate value models, this method leverages self-normalized importance sampling with the base LLM, making it training-free and readily integrated into existing RLVR frameworks like GRPO.

## Core Concept

The key innovation is computing hindsight Q-values by:

1. **Self-Conditioning on Success**: Inject the successful outcome into the LLM's prompt context
2. **Importance Ratio Estimation**: Measure how step probability changes when conditioned on success
3. **Causal Filtering**: Steps whose probability increases given success are causally responsible
4. **Multi-Scale Integration**: Combine macro-level trajectory rewards with micro-level Q-values

This produces step-wise credit signals grounded in the model's own conditional distributions, avoiding external value function errors.

## Architecture Overview

- **Base LLM policy**: Unchanged from baseline RLVR method
- **Hindsight conditional**: Re-prompt the model with outcome information
- **Importance ratio computation**: Token-level log-probability deltas
- **Self-normalized importance sampling**: Aggregate step-level credits
- **Multi-scale reward aggregation**: Blend trajectory and step signals

## Implementation Steps

### Step 1: Compute Self-Normalized Importance Ratios

For each step, estimate how probability changes when conditioned on successful outcome.

```python
import torch
import torch.nn.functional as F

def compute_hindsight_importance_ratios(
    model,
    trajectory,
    input_ids,
    successful_outcome
):
    """
    Compute importance ratios ρ for each step in trajectory.

    trajectory: list of (token_id, log_prob) tuples
    input_ids: original input token sequence
    successful_outcome: target outcome to condition on
    """
    trajectory_length = len(trajectory)
    importance_ratios = []

    # Original forward pass (already computed, use cache)
    original_log_probs = torch.tensor(
        [log_prob for _, log_prob in trajectory],
        dtype=torch.float32
    )

    # Hindsight forward pass: condition on successful outcome
    # Prepend outcome to input as conditioning
    hindsight_input = torch.cat([
        input_ids,
        torch.tensor([[model.tokenizer.encode(successful_outcome)]])
    ], dim=1)

    with torch.no_grad():
        hindsight_outputs = model(hindsight_input)
        hindsight_logits = hindsight_outputs.logits

    # Extract log probabilities for trajectory tokens
    hindsight_log_probs = []
    for t, (token_id, _) in enumerate(trajectory):
        logits = hindsight_logits[0, input_ids.shape[1] + t]
        log_prob = F.log_softmax(logits, dim=-1)[token_id]
        hindsight_log_probs.append(log_prob.item())

    hindsight_log_probs = torch.tensor(hindsight_log_probs, dtype=torch.float32)

    # Importance ratio: exp(log_prob_hindsight - log_prob_original)
    # This measures probability boost when conditioned on success
    log_ratios = hindsight_log_probs - original_log_probs
    importance_ratios = torch.exp(torch.clamp(log_ratios, min=-5, max=5))

    return importance_ratios
```

### Step 2: Formulate Hindsight Q-Values

Combine importance ratios with trajectory returns to create step-level credits.

```python
def compute_hindsight_q_values(
    trajectory,
    returns,
    importance_ratios,
    gamma=0.99
):
    """
    Compute Q^H = ρ × G where G is discounted return from each step.

    trajectory: list of steps
    returns: trajectory-level return G
    importance_ratios: tensor of importance ratios per step
    gamma: discount factor
    """
    trajectory_length = len(trajectory)

    # Discounted cumulative return from each step
    q_values = []
    cumulative_return = returns

    for t in range(trajectory_length - 1, -1, -1):
        # Q-value: hindsight importance weighted return
        q_hindsight = importance_ratios[t] * cumulative_return

        q_values.insert(0, q_hindsight)

        # Update cumulative return for previous step
        cumulative_return = cumulative_return / gamma  # Undo discount

    q_values = torch.stack(q_values)

    return q_values
```

### Step 3: Multi-Scale Reward Integration

Blend macro-level trajectory signals (GRPO baseline) with micro-level Q-values.

```python
def compute_multi_scale_returns(
    trajectory_return,
    hindsight_q_values,
    macro_weight=0.6,
    micro_weight=0.4
):
    """
    Combine trajectory-level and step-level signals.

    trajectory_return: scalar return for full trajectory
    hindsight_q_values: tensor of step-level Q-values
    macro_weight, micro_weight: balance between scales
    """
    # Macro signal: repeat trajectory return
    macro_signal = torch.full_like(
        hindsight_q_values,
        trajectory_return
    ) * macro_weight

    # Micro signal: hindsight Q-values normalized
    micro_signal = (
        hindsight_q_values / (hindsight_q_values.std() + 1e-8)
    ) * micro_weight

    # Combined: weighted sum
    multi_scale_returns = macro_signal + micro_signal

    return multi_scale_returns
```

### Step 4: Policy Gradient with Multi-Scale Advantages

Update policy using blended advantage estimates (placeholder for GRPO integration).

```python
def grpo_step_with_hindsight(
    model,
    input_ids,
    trajectory,
    trajectory_return,
    model_optimizer,
    num_reference_samples=4
):
    """
    GRPO training step augmented with hindsight credit assignment.
    """
    # Compute importance ratios from hindsight conditioning
    successful_outcome = "Correct"  # Placeholder
    importance_ratios = compute_hindsight_importance_ratios(
        model,
        trajectory,
        input_ids,
        successful_outcome
    )

    # Compute hindsight Q-values
    hindsight_q_values = compute_hindsight_q_values(
        trajectory,
        trajectory_return,
        importance_ratios
    )

    # Multi-scale integration
    multi_scale_returns = compute_multi_scale_returns(
        trajectory_return,
        hindsight_q_values,
        macro_weight=0.6,
        micro_weight=0.4
    )

    # Forward pass for policy gradient
    outputs = model(input_ids)
    log_probs = F.log_softmax(outputs.logits, dim=-1)

    # Extract log probs for trajectory tokens
    trajectory_log_probs = []
    for t, (token_id, _) in enumerate(trajectory):
        logit = outputs.logits[0, input_ids.shape[1] + t]
        log_prob = F.log_softmax(logit, dim=-1)[token_id]
        trajectory_log_probs.append(log_prob)

    trajectory_log_probs = torch.stack(trajectory_log_probs)

    # Policy gradient: advantage-weighted log probs
    baseline = multi_scale_returns.mean()
    advantages = multi_scale_returns - baseline

    policy_loss = -(trajectory_log_probs * advantages).mean()

    model_optimizer.zero_grad()
    policy_loss.backward()
    model_optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'importance_ratio_mean': importance_ratios.mean().item(),
        'advantage_mean': advantages.mean().item()
    }
```

## Practical Guidance

**When to Use:**
- Long-horizon reasoning tasks (10+ steps)
- Scenarios where trajectory rewards are sparse or weak
- GRPO-based training where step-level credit is bottleneck
- Multi-step problem solving with clear intermediate milestones

**When NOT to Use:**
- Short horizon tasks (<3 steps) where trajectory signals suffice
- Extreme latency constraints (hindsight pass adds 2-3x inference)
- Tasks without clear success conditions to condition on
- Very large models where re-inference is prohibitively expensive

**Hyperparameters:**
- **macro_weight**: 0.5-0.8; increase if trajectory signal is reliable
- **micro_weight**: 0.2-0.5; complement with macro signal
- **importance ratio clipping**: [-5, 5] prevents numerical instability
- **number of hindsight samples**: 1 usually sufficient (could ensemble)

**Common Pitfalls:**
- Over-relying on micro signal when trajectory supervision is weak
- Importance ratios exploding due to log-probability differences
- Hindsight conditioning conflicting with original input (prompt carefully)
- Forgetting to normalize Q-values before aggregation

## Reference

[Hindsight Credit Assignment paper on arXiv](https://arxiv.org/abs/2603.08754)
