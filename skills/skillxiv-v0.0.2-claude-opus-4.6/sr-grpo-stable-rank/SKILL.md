---
name: sr-grpo-stable-rank
title: "SR-GRPO: Stable Rank as Intrinsic Reward for LLM Alignment"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.02807
keywords: [llm-alignment, reinforcement-learning, intrinsic-rewards, representation-geometry, grpo]
description: "Uses stable rank (variance distribution across semantic dimensions) as annotation-free reward signal in GRPO to incentivize high-dimensional representation structures, eliminating dependency on human preference annotations or learned reward models."
---

## Summary

SR-GRPO introduces stable rank as an intrinsic, annotation-free reward signal for LLM alignment. Stable rank measures how information distributes across semantic dimensions in hidden representations. The approach leverages this geometric property in reinforcement learning to guide policy optimization, rewarding responses that maintain higher-dimensional representational structure rather than collapsing into narrow activation patterns.

## Core Technique

**Stable Rank Definition:** Measures the ratio of total variance to dominant-direction variance:
```
SR(X) = (sum of all eigenvalues)² / (sum of squared eigenvalues)
```
Higher stable rank indicates information spread across dimensions; lower indicates collapse.

**Intrinsic Reward Signal:** Compute stable rank of hidden states at each generation step:
```
reward_t = stable_rank(hidden_states_t) - stable_rank(hidden_states_ref)
```
Encourage diversity of representations without external labels.

**SR-GRPO Integration:** Add stable rank to group relative policy optimization:
```
total_reward = task_reward + λ_sr * sr_reward
```

## Implementation

**Stable rank computation:**
```python
def compute_stable_rank(X):
    # X: [batch, seq_len, hidden_dim]
    # Flatten to [N, hidden_dim]
    X_flat = X.reshape(-1, X.shape[-1])

    # SVD or eigendecomposition
    _, singular_values, _ = torch.svd(X_flat)
    sv_normalized = singular_values / singular_values.sum()

    # Stable rank formula
    sr = (sv_normalized.sum() ** 2) / (sv_normalized ** 2).sum()
    return sr
```

**Reference model setup:** Use frozen reference model for baseline:
```python
reference_model = load_pretrained_model()
reference_model.eval()
with torch.no_grad():
    ref_hidden = reference_model(input_ids)
    ref_sr = compute_stable_rank(ref_hidden)
```

**SR-GRPO reward:**
```python
def compute_sr_reward(generated_sequence, reference_sr, lambda_sr=0.1):
    hidden_states = model.get_hidden_states(generated_sequence)
    sr_current = compute_stable_rank(hidden_states)
    sr_reward = sr_current - reference_sr
    return lambda_sr * sr_reward
```

**Policy gradient with SR:**
```python
def grpo_step(prompt, reference_sr):
    # Generate trajectory
    response = model.generate(prompt)

    # Compute task and SR rewards
    task_reward = evaluate_task(response)
    sr_reward = compute_sr_reward(response, reference_sr)

    # Total reward
    total_reward = task_reward + sr_reward

    # Standard GRPO update
    loss = -log_prob(response) * (total_reward - baseline)
    loss.backward()
```

## When to Use

- LLM alignment without human preference data
- Scenarios where you want to encourage diverse representations
- Applications requiring intrinsic reward signals
- Tasks where representation geometry correlates with quality

## When NOT to Use

- Scenarios with abundant human preference annotations
- Tasks where task-specific rewards are more important than representation quality
- Models where stable rank doesn't correlate with performance
- Real-time inference where computing stable rank is prohibitive

## Key References

- Representation learning and spectral analysis
- Geometric properties of neural networks
- Group relative policy optimization (GRPO)
- Intrinsic motivation in reinforcement learning
