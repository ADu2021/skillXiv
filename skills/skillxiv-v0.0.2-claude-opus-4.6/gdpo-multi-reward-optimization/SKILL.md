---
name: gdpo-multi-reward-optimization
title: "GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05242"
keywords: [Reinforcement Learning, Multi-objective Optimization, LLM Alignment, Policy Optimization]
description: "Optimize language models against multiple reward signals simultaneously by decoupling reward normalization. GDPO prevents reward combination collapse that undermines training signal quality when aligning models to multiple human preferences like accuracy, safety, efficiency, and format compliance."
---

## When to Use This Skill
- Training LLMs to satisfy multiple objectives (tool calling accuracy + format compliance)
- Multi-reward RL settings where reward signals have different scales or difficulties
- Aligning models where early rewards (e.g., correctness) must be satisfied before optimizing secondary objectives (e.g., length constraints)
- Extending GRPO-based training to handle >2 reward objectives

## When NOT to Use This Skill
- Single-objective reward optimization (use standard GRPO)
- When rewards are naturally on identical scales without collapse risk
- Scenarios with fully independent reward signals requiring no prioritization

## Problem Summary
Applying Group Relative Policy Optimization (GRPO) directly to multiple rewards causes advantage value collapse: distinct reward combinations map to identical normalized advantages. For example, with two binary rewards, reward combinations (0,1), (0,2), and (1,2) produce identical advantages despite representing fundamentally different satisfaction levels. This collapses six distinct signal types into two advantage groups, degrading training efficiency and model performance.

## Solution: GDPO Three-Step Algorithm

Decouple normalization at the reward level before aggregation:

```python
# Step 1: Normalize each reward independently using group statistics
normalized_rewards = []
for reward_signal in reward_signals:
    mean = reward_signal.mean()
    std = reward_signal.std()
    norm_reward = (reward_signal - mean) / (std + eps)
    normalized_rewards.append(norm_reward)

# Step 2: Aggregate normalized rewards
aggregated_advantage = sum(normalized_rewards)

# Step 3: Stabilize via batch-wise normalization
final_advantage = (aggregated_advantage - aggregated_advantage.mean()) / (aggregated_advantage.std() + eps)
```

Instead of: `A = norm(r₁ + r₂ + ... + rₙ)`
Use: `A = norm(norm(r₁) + norm(r₂) + ... + norm(rₙ))`

## Key Implementation Details

**Handling Reward Difficulty Gaps:**
When objectives differ substantially in achievability (e.g., correctness is harder than format compliance), condition lower-priority rewards on higher-priority success:

```python
# Force format compliance to activate only when correctness is met
length_reward = 1 if (length <= L and correctness == 1) else 0
```

**Preventing Magnitude Explosion:**
Without batch-wise normalization, convergence failures occur as reward scales increase. Apply normalization as final step to stabilize advantage magnitudes across training updates.

**Training Stability Configuration:**
- Apply small weight decay to reward signals (λ ≈ 1e-4)
- Use batch-wise normalization to normalize advantages after aggregation
- Validate advantage distribution prevents >10x magnitude fluctuations between batches

## Empirical Results

**Tool Calling (Qwen2.5-1.5B):**
- +2.7% overall accuracy vs. GRPO
- +4.3% format correctness improvement

**Math Reasoning (DeepSeek-R1-7B on AIME):**
- +3% accuracy while reducing length violations to 0.2% (vs. 2.1% with GRPO)
- Eliminates training collapse GRPO exhibits after ~400 steps

**Coding (3-reward simultaneous optimization):**
- Maintains comparable pass rates while reducing bug ratio 26% more than GRPO

## Framework Compatibility
Implementations available in HF-TRL, verl, and Nemo-RL frameworks. Requires minimal code changes to existing GRPO implementations—typically adding separate normalization layers per reward signal.

## References
Refer to original paper for detailed ablations on rollout batch sizes, reward weighting strategies, and curriculum learning approaches for sequential reward mastery.
