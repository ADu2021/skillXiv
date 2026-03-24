---
name: repro-process-reward
title: "Rectifying LLM Thought from Lens of Optimization: Process-Level Rewards"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.01925
keywords: [reinforcement-learning, reasoning-optimization, process-rewards, llm-alignment, chain-of-thought]
description: "Dual-scoring mechanism (Magnitude and Stability scores) enabling process-level rewards that penalize inefficient reasoning patterns like overthinking and backtracking without trained reward models. Improve reasoning efficiency in RL without additional supervision."
---

## Summary

RePro (Rectifying Process-level Reward) frames chain-of-thought reasoning as an optimization process and develops a dual-scoring mechanism to improve LLM reasoning during RL training. The method computes Magnitude Score (optimization intensity) and Stability Score (smoothness), combined into process-level rewards that penalize inefficient patterns while incentivizing efficient reasoning steps.

## Core Technique

**Surrogate Objective Function:** Rather than evaluating reasoning directly, compute a surrogate measuring the model's confidence in ground truth via perplexity:
```
confidence_t = -log p(ground_truth_token | context_t)
```
This provides a continuous signal throughout reasoning about whether the model is on the right track.

**Magnitude Score:** Quantifies optimization intensity:
```
magnitude = |confidence_0 - confidence_T| / T
```
High magnitude indicates strong improvement or degradation; moderate magnitude indicates steady progress.

**Stability Score:** Assesses smoothness of reasoning trajectory:
```
stability = 1 - (variance of confidence across steps) / mean(confidence)
```
High stability indicates consistent, directed reasoning; low stability indicates zigzagging.

**Process-Level Reward:** Combine both scores:
```
reward_t = magnitude * stability - λ_overthink * overthinking_penalty
```

## Implementation

**Perplexity tracking:** At each reasoning step, compute cross-entropy loss with ground truth:
```python
step_loss = cross_entropy(logits_t, target)
confidence_t = -log(softmax(logits_t)[target])
```

**Score computation:**
```python
magnitudes = [abs(confidence[i+1] - confidence[i]) for i in range(len(confidence)-1)]
magnitude_score = mean(magnitudes) / max(magnitudes)

variance = var(confidence)
stability_score = 1 - (variance / (mean(confidence) + ε))
```

**Reward integration:** In your RL pipeline:
```python
process_reward = magnitude_score * stability_score
# Add to existing RL loss without extra models
total_reward = sequence_reward + 0.1 * process_reward
```

**Penalty for overthinking:** Detect and penalize reasoning loops:
```python
overthink_penalty = num_repeated_tokens / total_tokens
```

## When to Use

- RL training on reasoning tasks where efficiency matters
- Scenarios where you want to penalize inefficient patterns without labeled data
- Applications where reasoning transparency (via perplexity) aids training
- Tasks combining multiple reasoning attempts where early stopping helps

## When NOT to Use

- Supervised fine-tuning (RL not used)
- Scenarios where reasoning length is uniformly helpful
- Tasks where perplexity doesn't correlate with correctness
- Applications requiring external reward models for alignment

## Key References

- Process rewards and intermediate feedback in RL
- Optimization landscape and convergence analysis
- Perplexity-based confidence estimation
- Chain-of-thought reasoning and efficiency
