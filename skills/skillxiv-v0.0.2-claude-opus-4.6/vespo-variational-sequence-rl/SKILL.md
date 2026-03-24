---
name: vespo-variational-sequence-rl
title: "VESPO: Variational Sequence-Level Soft Policy Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.10693"
keywords: [reinforcement learning, off-policy learning, LLM training, importance weighting, policy stability]
description: "Stabilize off-policy RL training for LLMs by deriving principled importance weight reshaping from variational optimization. Instead of heuristic clamping, VESPO uses closed-form exponential weighting W^c1 * exp(c2*(1-W)) to suppress overweighted samples while maintaining smooth gradients. Enables stable training at 64× policy staleness and under fully asynchronous execution with sequence-level operations that avoid length-dependent biases."
---

# VESPO: Principled Off-Policy Weighting for Stable LLM RL

Large language model training with reinforcement learning often requires collecting interaction data from older policy versions due to computational constraints. This off-policy distribution shift—where training samples come from stale policies—destabilizes learning, causing gradient explosions or policy collapse. Standard approaches use heuristic importance weight functions (clipping, softmax) to suppress overweighted samples, but these lack principled justification and often require extensive hyperparameter tuning.

The challenge is discovering weight transformation functions that maintain training stability while preserving signal from informative samples. Existing methods operate at token level and apply length normalization, introducing biases where longer sequences receive different effective learning rates than short ones.

## Core Concept

VESPO reframes importance weight transformation through measure-change perspective. Rather than manually designing weight functions, the method formulates an optimization problem seeking a proposal distribution that:

1. Remains efficient (close to the behavior distribution for importance-sampled learning)
2. Incorporates the target policy (reduces bias)
3. Constrains variance (bounds importance weight magnitudes)

The closed-form solution is an exponential reshaping kernel: W^c₁ × exp(c₂(1-W)), where W represents sequence-level importance weights. This provides smooth, principled suppression that scales correctly with off-policy staleness.

## Architecture Overview

- **Sequence-Level Importance**: Compute importance weight per full sequence (not per-token), avoiding length-dependent biases
- **Variational Optimization**: Formulate weight transformation as constrained optimization problem over proposal distribution
- **Exponential Reshaping**: Apply closed-form kernel W^c₁ × exp(c₂(1-W)) to transform raw importance weights
- **Smooth Gradient Flow**: Exponential decay prevents hard clipping artifacts and enables stable backpropagation
- **Asynchronous Compatible**: Works with fully asynchronous data collection (no synchronization requirements)

## Implementation

Compute sequence-level importance weights from policy log probabilities:

```python
def compute_sequence_importance_weights(
    sequences, target_logprobs, behavior_logprobs, eps=1e-6
):
    """
    Compute importance weights at sequence level (not token level).
    sequences: (B, T) token ids
    target_logprobs: (B, T) log probs under current policy
    behavior_logprobs: (B, T) log probs under data collection policy
    Returns: (B,) importance weights
    """
    # Sum log probabilities across sequence
    target_log_prob = target_logprobs.sum(dim=1)  # (B,)
    behavior_log_prob = behavior_logprobs.sum(dim=1)  # (B,)

    # Importance weight: exp(log p_target - log p_behavior)
    log_importance = target_log_prob - behavior_log_prob
    importance_weights = torch.exp(log_importance.clamp(-10, 10))

    return importance_weights
```

Apply variational exponential reshaping to importance weights:

```python
def reshape_importance_weights(importance_weights, c1=0.5, c2=1.0):
    """
    Apply learned importance weight transformation.
    W^c1 * exp(c2 * (1 - W))
    This provides smooth suppression without hard clipping.
    """
    W = importance_weights.clamp(1e-6, 1e2)  # Avoid numerical issues

    # Exponential reshaping
    reshaped = (W ** c1) * torch.exp(c2 * (1.0 - W))

    return reshaped
```

Integrate into policy gradient loss with sequence-level weighting:

```python
def vespo_policy_loss(
    logits, targets, advantages, behavior_logprobs,
    c1=0.5, c2=1.0
):
    """
    Compute policy gradient loss with VESPO importance reweighting.
    logits: (B, T, V) model predictions
    targets: (B, T) ground truth tokens
    advantages: (B, T) or (B,) advantage estimates
    behavior_logprobs: (B, T) log probs from behavior policy
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Compute importance weights
    importance_weights = compute_sequence_importance_weights(
        targets, selected_logprobs, behavior_logprobs
    )

    # Reshape weights using exponential kernel
    reshaped_weights = reshape_importance_weights(importance_weights, c1, c2)

    # Standard policy gradient, scaled by reshaped importance weight
    if advantages.dim() == 1:
        # Sequence-level advantage
        pg_loss = -selected_logprobs * advantages.unsqueeze(1)
        pg_loss = (pg_loss * reshaped_weights.unsqueeze(1)).sum()
    else:
        # Token-level advantage
        pg_loss = -selected_logprobs * advantages
        pg_loss = (pg_loss * reshaped_weights.unsqueeze(1)).sum()

    # Normalize by batch size, not token count (avoids length bias)
    return pg_loss / (targets.shape[0] + 1e-8)
```

Implement hyperparameter learning or tuning:

```python
def learn_reshaping_coefficients(
    validation_data, base_model, num_trials=10
):
    """
    Grid search to find optimal c1, c2 values.
    Optimize for validation loss stability.
    """
    best_c1, best_c2 = 0.5, 1.0
    best_loss_variance = float('inf')

    for c1 in [0.3, 0.5, 0.7, 1.0]:
        for c2 in [0.5, 1.0, 2.0, 3.0]:
            losses = []

            for batch in validation_data:
                loss = vespo_policy_loss(
                    batch['logits'], batch['targets'],
                    batch['advantages'], batch['behavior_logprobs'],
                    c1=c1, c2=c2
                )
                losses.append(loss.item())

            loss_variance = np.var(losses)

            if loss_variance < best_loss_variance:
                best_loss_variance = loss_variance
                best_c1, best_c2 = c1, c2

    return best_c1, best_c2
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| c1 (polynomial weight) | 0.5 | Range 0.3–1.0; lower → more aggressive suppression |
| c2 (exponential strength) | 1.0 | Range 0.5–3.0; higher → stronger suppression of overweighted samples |
| Max importance ratio | 100 | Clamp weights to avoid extreme values; usually redundant with reshaping |
| Policy staleness | 64 updates | Tested stable at 64× staleness with MoE models |

**When to use**: For asynchronous or offline RL training of LLMs where data collection lags policy updates.

**When not to use**: For on-policy algorithms where all data is current; overhead of importance weighting is unnecessary.

**Common pitfalls**:
- Using token-level weighting instead of sequence-level; sequence-level avoids spurious length normalization
- Forgetting to account for policy staleness in advantage estimates; use GAE with behavior policy baseline
- Setting c1, c2 without validation tuning; grid search on small validation set to find stable coefficients

## Reference

VESPO enables stable off-policy RL training for LLMs even at 64× policy staleness and under fully asynchronous execution. The exponential kernel provides smooth gradient flow compared to hard-clipped importance weights, preventing optimization instabilities common in standard importance sampling approaches.
