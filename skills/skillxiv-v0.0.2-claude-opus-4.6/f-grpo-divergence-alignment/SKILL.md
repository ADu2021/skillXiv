---
name: f-grpo-divergence-alignment
title: "f-GRPO and Beyond: Divergence-Based RL for General LLM Alignment"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.05946"
keywords: [f-Divergence, RL Alignment, GRPO Generalization, Policy Improvement, Verifiable Rewards]
description: "Unify LLM alignment methods through f-divergence theory. f-GRPO extends GRPO to handle any divergence measure (KL, Jensen-Shannon, Hellinger), enabling tailored alignment objectives. f-HAL combines on-policy and off-policy preference learning to prevent reward hacking while maintaining safety alignment."
---

# f-GRPO: Divergence-Based RL for LLM Alignment

Popular LLM alignment methods optimize divergence between aligned and unaligned distributions, yet lack a unified framework. f-GRPO grounds GRPO in f-divergence theory, enabling selection of divergence measures matching your alignment objectives. For verifiable rewards (correct/incorrect), f-GRPO concentrates probability on high-reward responses. For preference-based alignment, f-HAL balances on-policy exploration with off-policy preference learning to prevent reward hacking.

## Core Concept

f-GRPO generalizes GRPO by parameterizing the divergence: min_π f(π || π_ref) where f is any f-divergence (KL, Jensen-Shannon, Hellinger, etc.). Different f-divergences yield different concentration behaviors:

- **KL divergence**: Punishes low-probability modes in π_ref; stays close to reference
- **Jensen-Shannon**: Symmetric; balances exploration and exploitation
- **Hellinger**: More aggressive concentration on high-reward modes
- **Reverse KL**: Avoids modes with low rewards in π_ref

For verifiable rewards: f-GRPO estimates divergence between above-average and below-average reward distributions. For preference-based: f-HAL uses on-policy RL (exploration) combined with off-policy preference signals (exploitation).

## Architecture Overview

- **f-Divergence Family**: Choose divergence matching alignment goals (concentration vs. exploration)
- **Reward-Based f-GRPO**: For verifiable rewards (math, code); converges to max-reward policies
- **f-HAL**: Hybrid on-policy + off-policy for preference alignment; prevents reward hacking
- **Monotonic Improvement**: All variants guarantee reward improvement until convergence
- **Theoretical Guarantees**: Alignment consistency proofs for each divergence choice

## Implementation

Implement f-GRPO for verifiable rewards:

```python
import torch
import torch.nn.functional as F

def compute_f_divergence(p_logits, q_logits, divergence_type='kl'):
    """
    Compute f-divergence between two distributions.
    Args:
        p_logits: Logits from policy distribution [batch, vocab_size]
        q_logits: Logits from reference distribution [batch, vocab_size]
        divergence_type: 'kl', 'js', 'hellinger', etc.
    Returns:
        divergence: Scalar divergence value
    """
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)

    if divergence_type == 'kl':
        # KL(p || q) = sum p * log(p/q)
        return (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()

    elif divergence_type == 'js':
        # Jensen-Shannon: symmetric divergence
        m = 0.5 * (p + q)
        return 0.5 * (p * (torch.log(p) - torch.log(m))).sum(dim=-1).mean() + \
               0.5 * (q * (torch.log(q) - torch.log(m))).sum(dim=-1).mean()

    elif divergence_type == 'hellinger':
        # Hellinger distance (squared)
        return (torch.sqrt(p * q + 1e-8)).sum(dim=-1).mean()

    else:
        raise ValueError(f"Unknown divergence: {divergence_type}")

def f_grpo_advantage(rewards, divergence_strength=1.0):
    """Compute advantages for f-GRPO with reward-based concentration."""
    # Separate above-average and below-average rewards
    mean_reward = rewards.mean()
    above_avg = (rewards > mean_reward).float()

    # Advantage: how much better than average
    advantage = rewards - mean_reward

    return advantage * divergence_strength

def f_grpo_loss(policy_logits, reference_logits, rewards, divergence_type='kl',
                 divergence_strength=1.0, entropy_coef=0.01):
    """Compute f-GRPO loss combining divergence and reward optimization."""
    # Compute divergence penalty
    divergence = compute_f_divergence(policy_logits, reference_logits, divergence_type)

    # Compute advantages for reward optimization
    advantages = f_grpo_advantage(rewards, divergence_strength)

    # Policy gradient
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(log_probs.detach() * advantages.unsqueeze(-1)).mean()

    # Entropy regularization (prevent mode collapse)
    entropy = -(log_probs * F.softmax(policy_logits, dim=-1)).sum(dim=-1).mean()

    # Combined loss
    loss = policy_loss + divergence + entropy_coef * entropy

    return loss

def f_grpo_training_step(policy, reference_model, batch, optimizer, divergence_type='js'):
    """Single f-GRPO training step."""
    inputs, rewards = batch

    # Get logits
    policy_logits = policy(inputs).logits
    with torch.no_grad():
        ref_logits = reference_model(inputs).logits

    # Compute loss
    loss = f_grpo_loss(policy_logits, ref_logits, rewards, divergence_type=divergence_type)

    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

Implement f-HAL for preference-based alignment:

```python
def f_hal_loss(policy_logits, reference_logits, preference_pairs, on_policy_rewards,
               divergence_type='js', on_policy_weight=0.5, off_policy_weight=0.5):
    """
    f-HAL: Hybrid on-policy + off-policy preference alignment.
    Args:
        policy_logits: [batch, vocab_size]
        reference_logits: [batch, vocab_size]
        preference_pairs: [(preferred_seq, dispreferred_seq), ...] from offline data
        on_policy_rewards: Rewards from on-policy rollouts
        on_policy_weight: Balance between on-policy and off-policy
        off_policy_weight: Weight for preference learning
    """
    # On-policy component: standard f-GRPO with on-policy rewards
    on_policy_loss = f_grpo_loss(policy_logits, reference_logits, on_policy_rewards,
                                  divergence_type=divergence_type)

    # Off-policy component: preference learning from offline pairs
    off_policy_loss = 0.0
    for preferred, dispreferred in preference_pairs:
        # Bradley-Terry preference model: log(π(preferred)) - log(π(dispreferred))
        pref_logprob = F.log_softmax(policy_logits, dim=-1)[preferred].sum()
        dispref_logprob = F.log_softmax(policy_logits, dim=-1)[dispreferred].sum()

        # Loss: maximize log odds of preferred
        off_policy_loss += -(pref_logprob - dispref_logprob)

    off_policy_loss = off_policy_loss / len(preference_pairs)

    # Combine with weight balance
    total_loss = on_policy_weight * on_policy_loss + off_policy_weight * off_policy_loss

    return total_loss
```

## Practical Guidance

| Parameter | Recommendation | Notes |
|-----------|-----------------|-------|
| Divergence type | Start with 'js' | Jensen-Shannon balances exploration; tune if needed. |
| Divergence strength | 0.5-2.0 | Higher values strengthen alignment pressure. |
| On-policy weight | 0.5-0.7 | Balance exploration (on-policy) vs. preference (off-policy). |
| Off-policy weight | 0.3-0.5 | Too high causes reward hacking; too low loses preference signal. |
| Entropy coefficient | 0.01-0.05 | Prevents mode collapse; adjust if needed. |

**When to Use**
- Alignment with verifiable rewards (math, code) → use f-GRPO
- Preference-based alignment with safety concerns → use f-HAL
- Need flexible divergence choices for domain-specific goals
- Want theoretical guarantees on reward improvement

**When NOT to Use**
- Simple reward signals (standard GRPO sufficient)
- Extremely limited offline data (off-policy weight too high)

**Common Pitfalls**
- Divergence weight too high → policy stops learning meaningful patterns
- Mixing on-policy and off-policy improperly → reward hacking reappears
- Not monitoring entropy; mode collapse if too low
- Choosing wrong divergence for task (test JS first, then adjust)

## Reference

See https://arxiv.org/abs/2602.05946 for theoretical analysis of f-divergence, convergence proofs, and empirical validation on math reasoning and safety alignment benchmarks.
