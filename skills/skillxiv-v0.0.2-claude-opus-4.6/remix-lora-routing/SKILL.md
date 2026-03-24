---
name: remix-lora-routing
title: "ReMix: Reinforcement routing for mixtures of LoRAs in LLM finetuning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.10160"
keywords: [LoRA, Routing, Mixture-of-Experts, RLVR, Reinforcement Learning]
description: "Learn to route requests across multiple LoRA adapters using RL-based router training with constant routing weights. Prevents weight collapse and ensures balanced contribution from all selected adapters during inference."
---

# Technique: Reinforcement-Learned LoRA Routing with Balanced Activation

Standard mixture-of-LoRA routers suffer from **weight collapse**: learned routing weights concentrate on single adapters, wasting computational resources. ReMix addresses this by treating routing as a reinforcement learning problem with constant (non-learnable) weights that ensure balanced contributions from all activated adapters.

The key insight is that router training can be reformulated as learning which LoRAs to *select* (via discrete sampling) rather than how much to weight their continuous contributions. This ensures the effective support size equals exactly k activated modules, not a smaller collapsed subset.

## Core Concept

ReMix separates training and inference strategies:

**Training Phase**: Use RL to optimize a categorical routing distribution that decides which LoRAs activate for each input.

**Inference Phase**: Select top-k LoRAs deterministically based on trained probabilities, apply constant routing weight ω to all k selected adapters.

This two-phase approach combines the expressiveness of learned routing during training with the efficiency and predictability of constant weights during inference.

## Architecture Overview

- **Router network**: Learns categorical distribution q(i|x) over LoRA selections
- **RL training**: Uses RLOO estimator to optimize which LoRAs activate
- **Constant weights**: All selected LoRAs contribute equally with weight ω > 0
- **Inference mechanism**: Top-k deterministic selection from learned probabilities
- **Task-specific adaptation**: Each LoRA specializes on different input patterns

## Implementation Steps

### Step 1: Define Router Network

The router produces probabilities for each LoRA to be selected.

```python
import torch
import torch.nn as nn

class LoRARouter(nn.Module):
    def __init__(self, hidden_dim, num_loras, k_selected):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_loras = num_loras
        self.k_selected = k_selected

        self.router_head = nn.Linear(hidden_dim, num_loras)

    def forward(self, hidden_state):
        # Shape: (batch_size, hidden_dim)
        logits = self.router_head(hidden_state)
        probs = torch.softmax(logits, dim=-1)
        return probs
```

### Step 2: RL Training with RLOO Estimator

Train the router using RLOO (Rank-Leave-One-Out) to optimize sampling-based routing selections.

```python
def train_router_rl(router, model_with_loras, input_ids, rewards, num_samples=8):
    batch_size = input_ids.shape[0]

    # Get router probabilities
    router_probs = router(hidden_states)  # (batch, num_loras)

    # Sample LoRA selections from categorical distribution
    sampled_selections = torch.multinomial(
        router_probs,
        num_samples,
        replacement=False
    )  # (batch, k_selected, num_samples)

    # Evaluate each sampled selection with the model
    sample_rewards = []
    for sample_idx in range(num_samples):
        lora_indices = sampled_selections[:, :, sample_idx]
        # Apply selected LoRAs with constant weight and evaluate
        output = model_with_loras(input_ids, lora_indices, weight=1.0/k_selected)
        sample_reward = compute_reward(output)
        sample_rewards.append(sample_reward)

    # RLOO baseline: leave-one-out reward
    sample_rewards = torch.stack(sample_rewards, dim=-1)
    baseline = sample_rewards.mean(dim=-1, keepdim=True)
    advantages = sample_rewards - baseline

    # Policy gradient loss
    log_probs = torch.log(router_probs + 1e-8)
    loss = -(log_probs * advantages).sum()

    return loss
```

### Step 3: Inference with Top-k Selection

During inference, use deterministic selection of top-k LoRAs and apply constant weights.

```python
def inference_with_remix(router, model_with_loras, input_ids, k_selected):
    batch_size = input_ids.shape[0]

    # Get router probabilities
    router_probs = router(hidden_states)  # (batch, num_loras)

    # Top-k deterministic selection
    top_k_probs, top_k_indices = torch.topk(
        router_probs,
        k=k_selected,
        dim=-1
    )

    # Constant routing weight for all selected LoRAs
    constant_weight = 1.0 / k_selected

    # Apply selected LoRAs with equal weight
    output = model_with_loras(
        input_ids,
        lora_indices=top_k_indices,
        routing_weights=torch.full_like(top_k_probs, constant_weight)
    )

    return output
```

## Practical Guidance

**When to Use:**
- Multi-task LoRA adaptation where tasks partition naturally into subgroups
- Inference environments where computational budget is constrained
- Scenarios requiring stable, predictable routing patterns
- Mixture-of-experts where adapter specialization matters

**When NOT to Use:**
- Single-task fine-tuning (use single LoRA directly)
- Scenarios requiring soft gating or continuous weight modulation
- Applications where one adapter dominates all inputs (use coarse-grained routing instead)

**Hyperparameter Tuning:**
- **k_selected**: Balance between expressiveness and inference speed (typically 2-4)
- **Constant weight ω**: Always use 1/k; provides principled normalization
- **Number of LoRA samples during training**: 4-8 samples balance variance and computation
- **RLOO baseline**: Leave-one-out strategy reduces gradient variance

**Common Pitfalls:**
- Learning continuous weights instead of discrete selection (reverts to collapse problem)
- Using unequal weights for selected LoRAs (breaks the routing principle)
- Insufficient RL training steps (router distributions remain poorly optimized)
- k_selected too large relative to number of input tokens

## Reference

[ReMix paper on arXiv](https://arxiv.org/abs/2603.10160)
