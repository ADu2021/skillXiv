---
name: llds-grpo-collapse
title: "On GRPO Collapse in Search-R1: Lazy Likelihood Displacement Suppression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.04220
keywords: [reinforcement-learning, grpo-training, rl-stability, tool-integration, token-level-optimization]
description: "Identifies Lazy Likelihood Displacement (LLD) as GRPO failure mechanism in tool-integrated LLMs and proposes lightweight LLDS regularization that penalizes only tokens responsible for likelihood decrease, achieving up to 45.2% performance gains."
---

## Summary

On GRPO Collapse identifies Lazy Likelihood Displacement (LLD) as the fundamental failure mechanism when training tool-integrated LLMs with GRPO, and proposes LLDS (LLD Suppression), a lightweight regularization that prevents unintended confidence reduction. The method activates only when a response's likelihood decreases and penalizes only responsible tokens.

## Core Technique

**Lazy Likelihood Displacement (LLD):** During GRPO training, the policy can unintentionally reduce confidence in correct tool calls while improving on other aspects. This "lazy" displacement occurs because the RL objective optimizes for reward, not confidence preservation.

**Fine-Grained Token Penalties:** Rather than penalizing entire sequences, identify which specific tokens caused likelihood decrease and penalize only those:
```
likelihood_per_token_t = log p(token_t | context)
if likelihood_t < baseline_likelihood_t:
    penalty_t = -λ * (baseline_likelihood_t - likelihood_t)
```

**Selective Activation:** Only apply penalty when overall sequence likelihood decreases:
```
if likelihood_total < baseline_likelihood_total:
    apply_llds_penalty()
```

## Implementation

**Likelihood tracking:** Compute baseline likelihood from reference model:
```python
def compute_baseline_likelihood(sequence, reference_model):
    with torch.no_grad():
        logits = reference_model(sequence)
        likelihood = F.log_softmax(logits, dim=-1)
    return likelihood
```

**Token-wise penalty computation:**
```python
def compute_llds_penalty(current_logits, baseline_likelihood, sequence):
    current_likelihood = F.log_softmax(current_logits, dim=-1)

    # Per-token likelihood change
    likelihood_change = current_likelihood - baseline_likelihood

    # Penalize tokens that decreased likelihood
    penalties = torch.where(
        likelihood_change < 0,
        -likelihood_change,  # Penalty proportional to decrease
        torch.zeros_like(likelihood_change)
    )

    return penalties.sum() / sequence.numel()
```

**LLDS in GRPO loss:**
```python
def grpo_loss_with_llds(policy_logits, reference_logits, reward, baseline_likelihood, lambda_llds=0.1):
    # Standard GRPO component
    policy_log_prob = F.log_softmax(policy_logits, dim=-1)
    reference_log_prob = F.log_softmax(reference_logits, dim=-1).detach()

    # KL divergence
    kl_div = F.kl_div(policy_log_prob, reference_log_prob, reduction='batchmean')

    # Advantage and policy gradient
    policy_loss = -policy_log_prob * (reward - baseline)

    # LLDS penalty (only if likelihood decreased overall)
    llds_penalty = compute_llds_penalty(policy_logits, baseline_likelihood, input_ids)

    # Total loss
    total_loss = policy_loss + beta * kl_div + lambda_llds * llds_penalty
    return total_loss
```

## When to Use

- GRPO training of tool-integrated language models
- Scenarios where tool calls require high confidence
- Applications experiencing GRPO collapse or performance degradation
- Tasks where preserving baseline capabilities is important

## When NOT to Use

- Non-GRPO training methods (PPO, DPO, etc.)
- Scenarios without baseline likelihood reference
- Real-time training where penalty overhead matters
- Applications where likelihood decrease is acceptable

## Key References

- Group Relative Policy Optimization (GRPO)
- Reinforcement learning for language models
- Tool-use and function calling in LLMs
- Training stability and collapse prevention
