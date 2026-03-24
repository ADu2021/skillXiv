---
name: f-grpo-focal-policy-optimization
title: "F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.06717"
keywords: [Policy Optimization, Diversity, Rare Solution Mining, RL Verifiable Rewards, Focal Weighting]
description: "Prevent policy collapse onto common solutions during reinforcement learning by applying difficulty-aware focal weighting to gradient contributions, maintaining diversity across solution modes while preserving performance."
---

# F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare

## Problem Context

Reinforcement learning with verifiable rewards (RLVR) for language models exhibits distribution sharpening at intermediate group sizes. Policies improve pass@1 (best single solution) but degrade pass@256 (solution diversity), indicating concentration onto a narrow set of solutions. This erases rare but valuable solutions, reducing the value of the learned model for exploring multiple reasoning paths.

## Core Concept

F-GRPO applies [focal weighting, difficulty-aware scaling, group-relative advantage] to address this phenomenon. The insight is that distribution collapse peaks at intermediate group sizes due to how probability mass redistributes. A single scalar multiplier per prompt down-weights high-success cases where concentration pressure peaks, preserving rare solutions without additional compute.

## Architecture Overview

- **Theoretical foundation**: Closed-form tail-miss probability showing non-monotonic dependence on group size
- **Focal weight function**: g(x) = (1 − μ̂_pos(x))^γ scaling advantage contributions
- **Group-relative optimization**: Asymmetric pressure preserving diversity at specific group sizes
- **Hyperparameter**: Single γ (focal strength, typically 0.5-1.0) with intuitive effects
- **Integration**: Drop-in modification to GRPO, DAPO, CISPO without new networks

## Implementation

### Step 1: Estimate success probability per prompt

Compute empirical pass rate (probability of getting any correct solution) for each prompt during training.

```python
# Estimate success probability
def estimate_success_probability(
    prompt_ids, group_results, smoothing_alpha=0.5
):
    """
    Estimate μ_pos(x): probability of success for each prompt.
    Smooth with alpha for stability on small groups.
    """
    success_probs = {}

    for prompt_id in set(prompt_ids):
        # Get all results for this prompt
        prompt_results = [
            r for r, p in zip(group_results, prompt_ids)
            if p == prompt_id
        ]

        # Count correct solutions (non-zero reward)
        num_correct = sum(1 for r in prompt_results if r > 0)
        total = len(prompt_results)

        # Estimate with smoothing
        raw_prob = num_correct / max(total, 1)
        smoothed_prob = (num_correct + smoothing_alpha) / (total + 2 * smoothing_alpha)

        success_probs[prompt_id] = smoothed_prob

    return success_probs
```

### Step 2: Compute focal weights for each prompt

Calculate difficulty-aware weights based on estimated success probability. Down-weight high-success prompts where concentration risk is highest.

```python
# Focal weighting
def compute_focal_weights(success_probs, gamma=1.0):
    """
    Compute focal weights: g(x) = (1 - μ_pos(x))^gamma
    Higher gamma → stronger down-weighting of easy prompts
    """
    focal_weights = {}

    for prompt_id, prob in success_probs.items():
        # Focal weight: prioritizes hard prompts
        weight = (1.0 - prob) ** gamma
        focal_weights[prompt_id] = weight

    return focal_weights
```

### Step 3: Apply focal weighting to advantage calculation

Modify the advantage computation in GRPO to include focal weights. This directly affects gradient magnitudes.

```python
# Modified advantage with focal weighting
def compute_focal_advantages(
    log_probs, rewards, prompt_ids, focal_weights,
    group_size=8
):
    """
    Compute GRPO advantages with focal weighting.
    F-GRPO advantage: Â^F-GRPO = g(x) * Â^GRPO
    """
    # First compute standard GRPO advantages
    batch_size = len(log_probs)
    num_groups = batch_size // group_size

    standard_advantages = []

    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = (group_idx + 1) * group_size

        group_rewards = rewards[group_start:group_end]
        group_probs = log_probs[group_start:group_end]

        # Compute per-token advantage within group
        mean_reward = group_rewards.mean()

        for i in range(group_size):
            advantage = group_rewards[i] - mean_reward
            standard_advantages.append(advantage)

    # Apply focal weights
    focal_advantages = []
    for adv, prompt_id in zip(standard_advantages, prompt_ids):
        weight = focal_weights.get(prompt_id, 1.0)
        focal_adv = weight * adv
        focal_advantages.append(focal_adv)

    return torch.tensor(focal_advantages)
```

### Step 4: Integrate into GRPO training step

Replace standard advantage computation with focal advantages in the main training loop.

```python
# GRPO training with focal weighting
def grpo_step_with_focal(
    model, batch_prompts, group_size=8, gamma=1.0,
    optimizer=None, clip_ratio=0.2
):
    """
    Single GRPO training step with focal weighting.
    """
    # Generate responses
    responses = []
    log_probs_list = []

    for prompt in batch_prompts:
        response, log_prob = model.generate_with_logprobs(
            prompt, max_tokens=200
        )
        responses.append(response)
        log_probs_list.append(log_prob)

    # Compute rewards (assuming verifier available)
    rewards = []
    for response in responses:
        reward = verifier(response)
        rewards.append(reward)

    rewards = torch.tensor(rewards)

    # Estimate success probability per prompt
    prompt_ids = list(range(len(batch_prompts)))  # Simplification
    success_probs = estimate_success_probability(
        prompt_ids, rewards.tolist()
    )

    # Compute focal weights
    focal_weights = compute_focal_weights(success_probs, gamma=gamma)

    # Compute advantages with focal weighting
    log_probs = torch.stack(log_probs_list)
    focal_advantages = compute_focal_advantages(
        log_probs, rewards, prompt_ids, focal_weights,
        group_size=group_size
    )

    # Standard GRPO clipping objective
    log_prob_ratio = log_probs - log_probs.detach()
    clipped_ratio = torch.clamp(
        torch.exp(log_prob_ratio), 1 - clip_ratio, 1 + clip_ratio
    )

    loss = -torch.min(
        log_prob_ratio * focal_advantages,
        clipped_ratio * focal_advantages
    ).mean()

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
```

### Step 5: Monitor diversity metrics during training

Track pass@k metrics to verify that focal weighting preserves solution diversity.

```python
# Diversity monitoring
def compute_pass_at_k_metrics(
    model, test_prompts, verifier, k_values=[1, 8, 256],
    num_samples_per_prompt=256
):
    """
    Compute pass@k: probability of having at least one correct solution
    in k samples per prompt.
    """
    results = {f'pass@{k}': [] for k in k_values}

    for prompt in test_prompts:
        samples = []

        for _ in range(num_samples_per_prompt):
            response = model.generate(prompt, max_tokens=200)
            is_correct = verifier(response)
            samples.append(is_correct)

        for k in k_values:
            # Pass@k: has at least one correct in first k
            has_correct = any(samples[:k])
            results[f'pass@{k}'].append(float(has_correct))

    # Average over prompts
    final_results = {
        k: sum(v) / len(v) for k, v in results.items()
    }

    return final_results
```

## Practical Guidance

**When to use**: Reasoning tasks (math, code, logic) where solution diversity matters for downstream use (ensemble, reranking, exploration). Less critical for single-solution tasks (generation, summarization).

**Hyperparameters**:
- **γ (focal strength)**: 0.5 (mild) to 2.0 (strong)
  - γ=0.5: gentle diversity preservation
  - γ=1.0: balanced (recommended baseline)
  - γ>1.5: aggressive diversity, may hurt pass@1
- **Group size**: Typically 4-8; focal weighting effects stronger at 8-16
- **Smoothing alpha**: 0.5-1.0 for probability estimation stability

**Tuning guidance**:
1. Measure baseline pass@1 and pass@256 without focal weighting
2. Apply γ=1.0 and evaluate; if pass@1 drops >1%, reduce γ
3. If pass@256 still dropping, increase γ slightly
4. Use pass@256 / pass@1 ratio to track diversity relative to quality

**Common pitfalls**:
- Focal weights computed with insufficient samples; use averaging over recent batches
- Over-aggressive γ (>1.5) hurts quality; start conservative
- Forgetting to smooth success probabilities; sparse data leads to unstable weights
- Not evaluating on test set; training pass@256 may not match test

**Scaling**: Negligible computational overhead; single scalar multiplication per prompt. Effective across model sizes (1B-70B parameters) and group sizes (4-256).

## Reference

Paper: https://arxiv.org/abs/2602.06717
Code: Available at author's repository
Related work: GRPO, policy divergence, solution diversity in RL
Metrics: pass@k evaluation, diversity-quality tradeoffs
