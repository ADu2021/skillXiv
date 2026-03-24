---
name: psa-pyramid-sparse-attention
title: "PSA: Pyramid Sparse Attention for Video Understanding and Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.04025
keywords: [sparse-attention, video-understanding, video-generation, hierarchical-pooling, efficient-transformers]
description: "Replaces binary keep/drop masks with multi-level pooled key-value representations, allowing queries to access larger receptive fields under same compute budget through hierarchical aggregation without discarding information."
---

## Summary

Pyramid Sparse Attention (PSA) improves block sparse attention by replacing binary keep/drop decisions with multi-level pooled KV representations. Instead of discarding unimportant blocks entirely, PSA assigns them to progressively coarser pooling levels, maintaining information fidelity while enabling larger receptive fields under computational constraints.

## Core Technique

**Hierarchical Pooling:** Instead of binary keep/drop:
```
keep_blocks: full attention  [level 0]
skip_blocks: 2× pooled       [level 1]
skip_blocks: 4× pooled       [level 2]
skip_blocks: 8× pooled       [level 3]
```

**Progressive Coarsening:** Create multiple downsampled versions of each KV cache:
- Level 0: Original KV (keep blocks)
- Level 1: 2× pooled KV (medium importance)
- Level 2: 4× pooled KV (low importance)
- Level 3: 8× pooled KV (minimal importance)

**Query-Aware Routing:** Route queries to appropriate pooling levels based on importance scores.

## Implementation

**Pooled KV cache creation:**
```python
def create_pyramid_cache(kv_cache, num_levels=4):
    pyramid = {}
    pyramid[0] = kv_cache  # Full resolution

    for level in range(1, num_levels):
        pool_factor = 2 ** level
        # Average pooling over tokens
        pooled_k = average_pool(kv_cache[0], pool_factor)
        pooled_v = average_pool(kv_cache[1], pool_factor)
        pyramid[level] = (pooled_k, pooled_v)

    return pyramid
```

**Block importance scoring:**
```python
def score_blocks(query, kv_cache, block_size=64):
    num_blocks = len(kv_cache) // block_size
    scores = []

    for block_idx in range(num_blocks):
        block_kv = kv_cache[block_idx*block_size:(block_idx+1)*block_size]
        # Compute attention score
        score = query @ block_kv.mean().T
        scores.append(score)

    return scores
```

**Level assignment:** Assign blocks to pooling levels based on scores:
```python
def assign_to_levels(scores, keep_ratio=0.1):
    num_blocks = len(scores)
    num_keep = int(num_blocks * keep_ratio)

    assignment = {}
    keep_indices = argsort(scores)[-num_keep:]
    assignment[0] = keep_indices  # Full resolution

    # Remaining blocks distributed across pooling levels
    remaining = [i for i in range(num_blocks) if i not in keep_indices]
    assignment[1] = remaining[:len(remaining)//3]
    assignment[2] = remaining[len(remaining)//3:2*len(remaining)//3]
    assignment[3] = remaining[2*len(remaining)//3:]

    return assignment
```

**Attention with pyramid:**
```python
def pyramid_attention(query, pyramid_cache, assignment):
    output = zeros_like(query)

    for level, block_indices in assignment.items():
        kv = pyramid_cache[level]
        for idx in block_indices:
            block_output = attention(query, kv[idx])
            output += block_output

    return output
```

## When to Use

- Video understanding and generation with limited compute
- Long sequences (1000+ tokens) requiring sparse attention
- Scenarios where maintaining information from all tokens matters
- Tasks balancing latency and receptive field size

## When NOT to Use

- Short sequences where full attention is feasible
- Applications requiring exact attention over all tokens
- Scenarios where pooling loses critical information
- Real-time inference where hierarchy overhead matters

## Key References

- Sparse attention mechanisms and efficiency
- Multi-scale representation learning
- Hierarchical attention and pyramid structures
- Video understanding and temporal modeling
