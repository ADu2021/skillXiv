---
name: clipo-contrastive-policy-optimization
title: "CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.10101"
keywords: [RLVR, Contrastive Learning, Policy Optimization, Reasoning, Dense Rewards]
description: "Augment verifiable reward RL (RLVR) with contrastive learning to generate dense auxiliary rewards. Enforce proximity among correct reasoning trajectories in embedding space while suppressing errors, amplifying invariant reasoning patterns."
---

# Technique: Dense Contrastive Rewards for Reasoning Trajectory Clustering

Sparse verifiable rewards (binary success/failure) provide limited training signal for complex reasoning tasks. CLIPO adds **contrastive learning** as an auxiliary objective: it embeds reasoning trajectories in latent space and applies InfoNCE loss to cluster correct responses together while repelling errors.

The insight is that successful reasoning paths share consistent underlying logic structures. By enforcing this structure in embedding space, contrastive learning acts as a denoising mechanism, amplifying invariant reasoning patterns while suppressing spurious shortcuts and hallucinations.

## Core Concept

CLIPO extends RLVR policy optimization algorithms (GRPO, GSPO, DAPO, GMPO) by introducing a lightweight contrastive head and auxiliary reward:

1. **Contrastive Head**: Projects reasoning trajectories to embedding space
2. **InfoNCE Loss**: Treats correct responses as positives, incorrect as negatives
3. **Dense Auxiliary Reward**: Converted contrastive loss to reward signal
4. **Combined Loss**: Final reward = verifiable reward + contrastive reward

This dual signal prevents optimization collapse on narrow heuristics while maintaining grounding in task-specific verifiable rewards.

## Architecture Overview

- **Trajectory encoder**: Linear or small MLP embedding trajectories
- **Contrastive head**: Projects to (typically) 256-512 dimensional embedding space
- **InfoNCE comparator**: Computes similarity matrices and contrastive loss
- **Reward converter**: Translates contrastive loss to auxiliary signal
- **Main policy**: Unchanged from baseline RLVR method

## Implementation Steps

### Step 1: Build Trajectory Encoder and Contrastive Head

Create embeddings for reasoning trajectories by processing token hidden states.

```python
import torch
import torch.nn as nn

class TrajectoryContrastiveHead(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=256, projection_dim=128):
        super().__init__()
        # Encode trajectory via mean pooling + projection
        self.projection = nn.Linear(hidden_dim, projection_dim)
        self.contrastive_head = nn.Sequential(
            nn.Linear(projection_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )

    def forward(self, hidden_states):
        """
        hidden_states: (batch_size, seq_len, hidden_dim)
        returns: (batch_size, projection_dim) embeddings
        """
        # Mean pooling over sequence dimension
        trajectory_repr = hidden_states.mean(dim=1)  # (batch, hidden_dim)

        # Project through network
        projected = self.projection(trajectory_repr)
        embedding = self.contrastive_head(projected)

        return embedding
```

### Step 2: Compute InfoNCE Contrastive Loss

Within each batch, group by correctness and compute contrastive pairs.

```python
def infonce_loss(embeddings, labels, temperature=0.1):
    """
    InfoNCE loss for trajectory embeddings.

    embeddings: (batch_size, embedding_dim)
    labels: (batch_size,) binary correctness labels
    temperature: temperature parameter for softmax
    """
    batch_size = embeddings.shape[0]

    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.mm(embeddings, embeddings.t()) / temperature

    # Create positive and negative masks
    labels_expanded = labels.unsqueeze(1)
    positive_mask = (labels_expanded == labels_expanded.t()).float()

    # Set diagonal (self-similarity) to 0 for positive mask
    positive_mask.fill_diagonal_(0)

    # Negative mask is complement
    negative_mask = 1.0 - positive_mask
    negative_mask.fill_diagonal_(0)

    # InfoNCE: log(exp(sim_pos) / sum(exp(sim_neg)))
    exp_sim = torch.exp(similarity_matrix)

    # Sum of positive similarities per row
    pos_sum = (exp_sim * positive_mask).sum(dim=1, keepdim=True)

    # Sum of negative similarities per row
    neg_sum = (exp_sim * negative_mask).sum(dim=1, keepdim=True)

    # InfoNCE loss (avoiding division by zero)
    infonce = -torch.log(
        (pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8)
    )

    return infonce.mean()
```

### Step 3: Convert Contrastive Loss to Auxiliary Reward

Transform contrastive objective into a dense reward signal that complements verifiable reward.

```python
def contrastive_reward(
    embeddings,
    labels,
    verifiable_rewards,
    contrastive_weight=0.5,
    temperature=0.1
):
    """
    Compute combined reward: verifiable + contrastive auxiliary.

    Returns dense reward per trajectory.
    """
    batch_size = embeddings.shape[0]

    # Normalize embeddings
    embeddings_norm = torch.nn.functional.normalize(embeddings, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / temperature

    # Create positive mask (other correct trajectories)
    labels_expanded = labels.unsqueeze(1)
    positive_mask = (labels_expanded == labels_expanded.t()).float()
    positive_mask.fill_diagonal_(0)

    # Average positive similarity per trajectory
    pos_similarities = (similarity_matrix * positive_mask).sum(dim=1) / (
        positive_mask.sum(dim=1) + 1e-8
    )

    # Average negative similarity per trajectory
    negative_mask = 1.0 - positive_mask
    negative_mask.fill_diagonal_(0)
    neg_similarities = (similarity_matrix * negative_mask).sum(dim=1) / (
        negative_mask.sum(dim=1) + 1e-8
    )

    # Contrastive reward: pull positives, push negatives
    contrastive_reward_signal = pos_similarities - neg_similarities

    # Combine with verifiable reward
    total_reward = (
        verifiable_rewards +
        contrastive_weight * contrastive_reward_signal
    )

    return total_reward, contrastive_reward_signal
```

### Step 4: Integrate into RLVR Training Loop

Extend baseline policy optimization with contrastive auxiliary objective.

```python
def train_step_with_clipo(
    model,
    input_ids,
    verifiable_rewards,
    contrastive_head,
    policy_optimizer,
    contrastive_weight=0.5,
    temperature=0.1
):
    """
    Single training step combining RLVR with contrastive learning.
    """
    # Forward pass through model
    outputs = model(
        input_ids,
        output_hidden_states=True
    )

    # Extract trajectory embeddings
    hidden_states = outputs.hidden_states[-1]
    trajectory_embeddings = contrastive_head(hidden_states)

    # Compute correctness labels (binarize verifiable rewards)
    correctness_labels = (verifiable_rewards > 0).long()

    # Compute combined rewards
    total_rewards, contrastive_signal = contrastive_reward(
        trajectory_embeddings,
        correctness_labels,
        verifiable_rewards,
        contrastive_weight=contrastive_weight,
        temperature=temperature
    )

    # Policy gradient: GRPO style (placeholder)
    log_probs = outputs.logits.log_softmax(dim=-1)
    baseline = total_rewards.mean()
    advantages = total_rewards - baseline

    policy_loss = -(log_probs * advantages.unsqueeze(-1)).mean()

    # Backward pass
    policy_loss.backward()
    policy_optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'contrastive_signal_mean': contrastive_signal.mean().item(),
        'total_reward_mean': total_rewards.mean().item()
    }
```

## Practical Guidance

**When to Use:**
- Complex reasoning tasks (MATH, GSM8K, competition-level problems)
- Distributions with spurious correlations that mislead optimization
- Scenarios where verifiable rewards alone lead to hallucinations
- Tasks requiring robust generalization to perturbed inputs

**When NOT to Use:**
- Simple classification with clean, unambiguous labels
- When correct solutions form disconnected regions (contrastive smoothness harmful)
- Extreme computational budget constraints (adds embedding computation)

**Hyperparameter Tuning:**
- **contrastive_weight**: 0.3-1.0; balance with task difficulty
- **temperature**: 0.05-0.2; lower sharpens contrastive distinctions
- **embedding_dim**: 128-512 depending on trajectory complexity
- **batch size**: Larger batches provide more negative samples

**Common Pitfalls:**
- Weight too high, overshadowing verifiable signals (verify with ablations)
- Temperature too low causing training instability (increase if loss oscillates)
- Insufficient positive pairs early in training (consider warm-up scheduling)
- Embedding space not well-aligned with task semantics

## Reference

[CLIPO paper on arXiv](https://arxiv.org/abs/2603.10101)
