---
name: fastcurl-curriculum-rl-efficient-reasoning
title: "FastCuRL: Curriculum Reinforcement Learning with Stage-wise Context Scaling for Efficient Training R1-like Reasoning Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.17287"
keywords: [Curriculum Learning, Reinforcement Learning, Context Scaling, Reasoning Models, GRPO, CoT Optimization]
description: "Train efficient reasoning models using stage-wise context scaling and complexity-aware data selection. Achieves 49.6% accuracy on AIME 2024 while reducing training steps by 50% through alternating compress-extend cycles that progressively refine reasoning quality."
---

## Core Concept

FastCuRL addresses the training inefficiency of large reasoning models by jointly optimizing context length and training data complexity through a curriculum learning framework. The key insight is that controlling context length and selecting data based on problem complexity can significantly improve RL training efficiency while generating more concise Chain-of-Thought (CoT) outputs. The approach uses a cyclical compress-extend strategy that iteratively refines reasoning outputs.

## Architecture Overview

FastCuRL integrates three main components:

- **Group Relative Policy Optimization (GRPO)**: A resource-efficient RL algorithm that eliminates the need for a critic model by computing advantages from group-level scores rather than individual baseline scores
- **Complexity-Aware Data Selection**: Divides training data into three categories (L1, L2, L3) based on input prompt length correlation with output complexity
- **Stage-wise Context Scaling**: Alternates between compress phases (reducing context length) and extend phases (increasing context length) across multiple training stages

## Implementation Steps

### 1. Group Relative Policy Optimization (GRPO) Algorithm

The GRPO objective maximizes policy improvements with KL divergence regularization and optional entropy bonus:

```python
# Simplified GRPO loss computation
import torch
import torch.nn.functional as F

def compute_grpo_loss(policy_logits, old_policy_logits, rewards,
                      kl_coefficient=0.02, entropy_coeff=0.01, epsilon=0.2):
    """
    Compute Group Relative Policy Optimization loss.

    Args:
        policy_logits: logits from current policy
        old_policy_logits: logits from reference policy
        rewards: shape [batch, group_size]
        kl_coefficient: KL penalty weight
        entropy_coeff: entropy bonus weight
        epsilon: clipping threshold
    """
    # Normalize advantages across group
    mean_reward = rewards.mean(dim=-1, keepdim=True)
    std_reward = rewards.std(dim=-1, keepdim=True) + 1e-8
    advantages = (rewards - mean_reward) / std_reward

    # Compute log probability ratio
    log_ratio = policy_logits - old_policy_logits
    ratio = torch.exp(log_ratio)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # KL divergence penalty
    kl_loss = kl_coefficient * ((ratio - 1) - log_ratio).mean()

    # Optional entropy bonus for exploration
    entropy = -torch.sum(torch.softmax(policy_logits, -1) *
                         torch.log_softmax(policy_logits, -1), dim=-1)
    entropy_loss = -entropy_coeff * entropy.mean()

    return policy_loss + kl_loss + entropy_loss
```

### 2. Complexity-Aware Data Selection

Segment the training dataset by input prompt length to match problem complexity with context requirements:

```python
def partition_dataset_by_complexity(dataset,
                                   complexity_thresholds=None):
    """
    Partition dataset into complexity levels based on prompt length.

    Args:
        dataset: list of (prompt, response) tuples
        complexity_thresholds: percentile cutoffs [25, 75] for L1/L2/L3

    Returns:
        dict with keys 'L1' (short), 'L2' (medium), 'L3' (long)
    """
    prompt_lengths = [len(item['prompt'].split()) for item in dataset]

    if complexity_thresholds is None:
        # Use percentile-based splitting
        p25 = np.percentile(prompt_lengths, 25)
        p75 = np.percentile(prompt_lengths, 75)
        complexity_thresholds = [p25, p75]

    partitions = {'L1': [], 'L2': [], 'L3': []}

    for item, length in zip(dataset, prompt_lengths):
        if length <= complexity_thresholds[0]:
            partitions['L1'].append(item)
        elif length <= complexity_thresholds[1]:
            partitions['L2'].append(item)
        else:
            partitions['L3'].append(item)

    return partitions
```

### 3. Stage-wise Context Scaling Training Loop

Implement the multi-stage training with alternating context lengths:

```python
def train_with_stage_scaling(model, train_partitions, num_stages=5,
                            context_lengths=None, epochs_per_stage=1):
    """
    Train model using stage-wise context scaling curriculum.

    Args:
        model: the reasoning model to train
        train_partitions: dict with 'L1', 'L2', 'L3' datasets
        num_stages: number of training stages
        context_lengths: list of context window sizes per stage
        epochs_per_stage: epochs per stage (typically 1 for efficiency)
    """
    if context_lengths is None:
        # Example: compress-extend cycle
        context_lengths = [8192, 8192, 16384, 16384, 24576]

    stage_configs = [
        # Stage 1: L1 (short) at 8K context
        {'dataset': train_partitions['L1'], 'context': context_lengths[0]},
        # Stage 2: L2 (medium) at 8K context
        {'dataset': train_partitions['L2'], 'context': context_lengths[1]},
        # Stage 3: L2 (medium) at 16K context (extend phase)
        {'dataset': train_partitions['L2'], 'context': context_lengths[2]},
        # Stage 4: L3 (long) at 16K context
        {'dataset': train_partitions['L3'], 'context': context_lengths[3]},
        # Stage 5: L3 (long) at 24K context
        {'dataset': train_partitions['L3'], 'context': context_lengths[4]},
    ]

    for stage_idx, config in enumerate(stage_configs):
        print(f"Training Stage {stage_idx + 1}: "
              f"dataset={list(train_partitions.keys())[stage_idx % 3]}, "
              f"context={config['context']}")

        # Set model context window
        model.config.max_position_embeddings = config['context']

        # Train on this stage's dataset
        for epoch in range(epochs_per_stage):
            for batch in DataLoader(config['dataset'], batch_size=8):
                # Forward pass with GRPO
                loss = compute_grpo_loss(batch, model)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```

## Practical Guidance

### When to Use FastCuRL

- Training reasoning models on mathematical or complex problem-solving tasks
- Resource-constrained environments where reducing training steps is critical
- When you need both quality reasoning and concise CoT outputs
- Fine-tuning R1-distilled models (e.g., DeepSeek-R1-Distill-Qwen)

### When NOT to Use

- Tasks requiring long, detailed explanations as a feature
- Limited access to compute for initial dataset partitioning and analysis
- When training data doesn't show prompt-length to output-length correlation
- Few-shot scenarios with very limited training data

### Hyperparameters & Configuration

- **KL coefficient (β)**: Typically 0.02; controls policy divergence from reference
- **Entropy coefficient (α)**: Around 0.01 when needed to prevent entropy collapse
- **Epsilon (clip threshold)**: Standard PPO value of 0.2
- **Context lengths**: Start with 8K→8K→16K→16K→24K sequence for compress-extend
- **Epochs per stage**: Keep at 1 for efficiency in low-resource settings
- **Group size (G)**: Number of rollout responses per prompt; typically 4-8

### Common Pitfalls

- **Not analyzing data distribution**: Failing to verify input-output length correlation before applying complexity-based partitioning
- **Entropy collapse**: Occurs when context window is too large for current dataset; add entropy bonus term to mitigate
- **Stage ordering**: Placing complex data (L3) too early can cause training instability; start with simpler data
- **Context thrashing**: Rapidly changing context lengths without stable intermediate stages can prevent convergence
- **Ignoring clipping rates**: High percentage of clipped outputs indicates context is too small; monitor clipping ratio per stage

## Reference

- DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability.
- Shao et al. 2024. Group Relative Policy Optimization (GRPO).
- Luo et al. 2025. DeepScaleR: Iterative Context Scaling for Efficient LLM Training.
- Official FastCuRL repository: https://github.com/nick7nlp/FastCuRL
