---
name: rlkv-cache-compression
title: "Which Heads Matter for Reasoning? RL-Guided KV Cache Compression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.08525"
keywords: [kv-cache, compression, attention-heads, reinforcement-learning, reasoning-efficiency]
description: "Use reinforcement learning to identify reasoning-critical attention heads and apply selective KV cache compression, reducing cache by 20-50% while preserving reasoning quality. Maintains speedups up to 1.21x with minimal performance loss."
---

# RL-Guided KV Cache Compression: Selective Head Importance

Transformer inference scales poorly with sequence length due to KV cache growth. Naive compression removes important information; reasoning-critical attention heads degrade when compressed uniformly. This technique uses reinforcement learning to discover which heads matter for reasoning and compresses selectively.

The key insight: attention heads have different roles. Some heads are critical for coherent reasoning chains, while others handle retrieval or aggregation and tolerate aggressive compression. By identifying these patterns through RL, you preserve reasoning quality while gaining 20-50% cache reduction.

## Core Concept

**Head-Importance Discovery**: Use RL to directly optimize cache allocation against actual reasoning outcomes. Rather than heuristic importance measures, train an agent that observes head activations and decides cache allocation while generation quality is the reward signal.

**Asymmetric Allocation**: Allocate full KV cache to reasoning-critical heads while aggressively compressing others. This prevents the quality degradation that comes from uniform compression.

## Architecture Overview

- **Head Analyzer**: Computes importance scores for each attention head using RL-derived signals
- **Selective Compression Policy**: Maintains full cache for critical heads, applies compression to others
- **Reward Mechanism**: Direct optimization against generation quality, not proxy metrics
- **Generation Monitor**: Tracks completion quality to guide policy updates

## Implementation Steps

**Stage 1: Establish Head Importance Baseline**

Profile which heads contribute to reasoning by analyzing attention patterns during generation:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def analyze_head_activations(model, batch):
    """
    Profile attention head activation patterns during generation.
    Identifies which heads show sustained activity during reasoning.
    """
    head_importance = {}

    # Hook into attention layers
    activations = []
    def capture_attention(module, input, output):
        activations.append(output[0])  # attention weights

    # Register hooks
    for layer_idx, layer in enumerate(model.transformer.h):
        layer.self_attention.register_forward_hook(capture_attention)

    # Generate with reasoning
    with torch.no_grad():
        outputs = model.generate(
            batch,
            max_new_tokens=512,
            output_attentions=True
        )

    return activations

# Profile baseline
activations = analyze_head_activations(model, reasoning_prompts)
```

**Stage 2: RL Training for Head Importance**

Train a policy that learns which heads to preserve:

```python
def rl_head_selection_training(model, train_data, num_episodes=100):
    """
    Train RL agent to select which heads to keep full cache.
    Reward = generation quality (match human verification).
    """

    cache_allocations = []
    rewards = []

    for episode in range(num_episodes):
        # Sample reasoning problems
        problems = sample_batch(train_data, batch_size=8)

        # Current policy: which heads get full KV cache
        head_mask = compute_allocation_policy(episode)  # 0=compress, 1=full

        # Generate with selective caching
        outputs = generate_with_selective_cache(
            model,
            problems,
            head_mask=head_mask
        )

        # Evaluate quality
        quality_scores = verify_reasoning_outputs(outputs)
        reward = torch.tensor(quality_scores).mean()

        # Policy gradient update
        log_prob = compute_policy_logprob(head_mask)
        loss = -(log_prob * reward).mean()

        update_policy(loss)

        rewards.append(reward.item())
        cache_allocations.append(head_mask)

    return cache_allocations[-1]  # Best policy
```

**Stage 3: Inference with Selective Compression**

Apply learned allocation policy during generation:

```python
def generate_with_rl_cache(model, prompt, cache_allocation):
    """
    Generate while applying learned head-based cache compression.
    cache_allocation: binary mask where 1=full cache, 0=compressed.
    """

    input_ids = tokenize(prompt)
    kv_cache = None

    for step in range(max_steps):
        with torch.no_grad():
            outputs = model(
                input_ids[:, -1:],
                past_key_values=kv_cache,
                return_dict=True,
                output_attentions=True
            )

        # Apply selective caching
        new_cache = []
        for layer_idx, (key, value) in enumerate(outputs.past_key_values):
            for head_idx in range(key.shape[1]):
                if cache_allocation[layer_idx, head_idx] == 0:
                    # Compress this head: keep only recent tokens
                    key = key[:, head_idx, -256:, :]
                    value = value[:, head_idx, -256:, :]
            new_cache.append((key, value))

        kv_cache = new_cache
        next_token = outputs.logits[0, -1].argmax()
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)])

    return input_ids
```

## Practical Guidance

**When to Use RL-Guided Cache Compression:**
- Reasoning tasks where head roles vary significantly by task type
- Long-sequence generation where cache overhead is prohibitive (>10K tokens)
- Models where you can afford RL training upfront for inference speedup

**When NOT to Use:**
- Short sequences where cache isn't a bottleneck
- Tasks requiring all heads equally (retrieval, copying)
- When you need inference speedup immediately without training

**Typical Head Patterns:**

| Head Type | Reasoning Role | Compression Tolerance |
|-----------|----------------|----------------------|
| Reasoning-Path Heads | Track solution chains | 0-10% compression |
| Retrieval Heads | Attend to factual content | 50-80% compression |
| Aggregation Heads | Combine sub-components | 40-70% compression |
| Noise Heads | Low meaningful activity | 90%+ compression |

**Common Pitfalls:**
- Using single importance metric instead of task-aware importance
- Compressing too aggressively in early layers (contains necessary structure)
- Ignoring layer depth: deep layers tolerate more compression than shallow
- Not validating that compressed generations pass verification

## Reference

Based on the research at: https://arxiv.org/abs/2510.08525
