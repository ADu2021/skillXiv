---
name: lspo-length-aware-dynamic-sampling
title: "LSPO: Length-aware Dynamic Sampling for Policy Optimization in LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.01459"
keywords: [Reinforcement Learning, LLM Training, Policy Optimization, Response Length Filtering, Reasoning Models]
description: "Filter training samples by response length to identify high-confidence correct solutions and complex problems, improving sample efficiency in LLM reasoning RL without additional metrics."
---

# Technique: Length-Aware Dynamic Sampling for Efficient RL Training

Large language models trained on reasoning tasks generate responses of varying lengths. Existing approaches treat all responses equally, but response length contains useful training signal: short responses indicate high-confidence solutions, while long responses reveal problems that required substantial effort. LSPO exploits this signal to filter training data more efficiently.

The core insight is that intermediate-length responses lack the signal value of either extreme. Short responses demonstrate correct reasoning with minimal deliberation (the ideal case). Long responses show where models struggle and need reinforcement. Responses of moderate length neither demonstrate confidence nor provide clear learning opportunities.

## Core Concept

LSPO implements a filtering strategy based on response length percentiles during each RL training iteration. The algorithm retains the shortest 30% of responses (high-confidence cases) and the longest 65-95% of responses (difficult cases requiring effort), filtering out the middle 35-65% as uninformative.

The percentile thresholds are computed dynamically from the current batch rather than fixed globally. This ensures the filtering adapts as the model's response distribution evolves during training—early in training when responses cluster in a narrow range, the percentiles shift automatically to maintain the retention ratio.

## Architecture Overview

- **Input**: A batch of (prompt, response) pairs from an LLM rollout
- **Filtering Stage**: Compute response length for each example, determine 30th and 65th-95th percentile boundaries
- **Selection**: Retain responses shorter than 30th percentile and longer than dynamic threshold
- **Output**: Filtered samples fed into standard RL algorithms (GRPO, DAPO, PPO)
- **Integration**: Works as a preprocessing layer before policy gradient computation

## Implementation Steps

Initialize the filtering parameters and generate a batch of responses. The filtering operates on response lengths computed as token count (typically using the model's tokenizer).

```python
def lspo_filter(responses, prompts, retention_short=0.30, retention_long=0.30):
    """
    Filter responses by length using LSPO strategy.

    Args:
        responses: List of generated response strings
        prompts: List of corresponding prompts
        retention_short: Fraction of shortest responses to keep (default 0.30)
        retention_long: Fraction of longest responses to keep (default 0.30)

    Returns:
        filtered_indices: Boolean array of samples to retain
    """
    lengths = np.array([len(r.split()) for r in responses])

    # Compute percentile thresholds
    short_threshold = np.percentile(lengths, retention_short * 100)
    long_threshold = np.percentile(lengths, (1 - retention_long) * 100)

    # Retain short (confident) and long (difficult) responses
    keep_mask = (lengths <= short_threshold) | (lengths >= long_threshold)

    return keep_mask
```

Create reward signals for the retained samples. These rewards should align with task correctness (exact match, tool verification, etc.).

```python
def compute_advantages(responses, rewards, baseline_value=0.0):
    """
    Compute advantage values for policy gradient updates.

    Args:
        responses: Filtered response list
        rewards: Correctness rewards for each response
        baseline_value: Baseline for advantage estimation (default 0.0)

    Returns:
        advantages: Advantage values for policy optimization
    """
    advantages = rewards - baseline_value
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return normalized_advantages
```

Integrate the filtered samples into your standard RL algorithm (GRPO shown here as example).

```python
def train_step_with_lspo(model, prompts, responses, rewards, optimizer, lspo_retention):
    """
    Single training step combining LSPO filtering with GRPO.

    Args:
        model: Language model policy
        prompts: Input prompts
        responses: Generated responses
        rewards: Task correctness signals
        optimizer: PyTorch optimizer
        lspo_retention: Dict with retention fractions

    Returns:
        loss: Training loss for monitoring
    """
    # Apply LSPO filtering
    keep_mask = lspo_filter(
        responses,
        prompts,
        retention_short=lspo_retention['short'],
        retention_long=lspo_retention['long']
    )

    # Filter to retained samples
    filtered_prompts = [p for p, k in zip(prompts, keep_mask) if k]
    filtered_responses = [r for r, k in zip(responses, keep_mask) if k]
    filtered_rewards = rewards[keep_mask]

    # Compute policy log probs and advantages
    log_probs = model.get_log_probs(filtered_prompts, filtered_responses)
    advantages = compute_advantages(filtered_responses, filtered_rewards)

    # Standard GRPO-style loss with advantage weighting
    loss = -(log_probs * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Short retention % | 25-35% | Captures high-confidence solutions; adjust based on task success rate |
| Long retention % | 65-95% | Captures complex cases; higher threshold keeps more difficult examples |
| Batch size | 32-256 | Percentiles are computed per-batch; larger batches provide more stable thresholds |
| When to use | Any reasoning task with response length variation | Most effective when task difficulty causes length variation |
| When NOT to use | Fixed-length outputs or constrained generation | LSPO assumes length correlates with confidence/difficulty |
| Common pitfall | Using fixed percentiles across training | Dynamic recomputation per-batch is essential as response distribution evolves |

### When to Use LSPO

- Training on math, coding, or complex reasoning tasks where response length naturally varies
- When you have reliable reward signals (exact match, verifiable outputs) but noisy intermediate signals
- In RL pipelines where computational cost is a concern—filtering reduces required forward passes
- Scenarios with high response length variation (some solutions 10 tokens, others 500+)

### When NOT to Use LSPO

- Tasks with fixed or tightly constrained response lengths
- Scenarios where length doesn't correlate with confidence (e.g., stylistic writing)
- When all responses are already similarly successful (no difficulty gradient)
- Instruction-following tasks without objective correctness metrics

### Common Pitfalls

- **Computing percentiles incorrectly**: Must recompute from current batch distribution, not globally
- **Filtering too aggressively**: Setting retention thresholds too low loses valuable training signal
- **Ignoring task domain differences**: A 50-token response means different things in code generation vs. essay writing
- **Mixing with curriculum learning**: Be cautious combining LSPO with explicit difficulty ordering
- **Not normalizing advantages**: Skipping normalization causes training instability

## Reference

Paper: https://arxiv.org/abs/2510.01459
