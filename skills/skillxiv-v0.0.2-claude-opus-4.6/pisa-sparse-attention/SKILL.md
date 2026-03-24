---
name: pisa-sparse-attention
title: "PISA: Piecewise Sparse Attention"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.01077"
keywords: [Sparse Attention, Diffusion Transformers, Inference Efficiency, Attention Mechanisms]
description: "Accelerate diffusion transformers through training-free sparse attention combining exact computation for critical blocks with efficient approximation for non-critical ones. Achieves 2-3× speedup without retraining."
---

# PISA: Piecewise Sparse Attention for Efficient Diffusion Transformers

Attention mechanisms scale quadratically with sequence length O(n²), creating bottlenecks for high-resolution image and video generation in diffusion models. Existing sparse attention methods discard non-critical blocks entirely, but this loses information and degrades quality. PISA introduces a hybrid approach: compute attention exactly for critical blocks identified through covariance analysis, and approximate remaining blocks using block-wise Taylor expansion. This "exact-or-approximate" paradigm maintains quality while accelerating inference.

The key insight is that non-critical attention blocks have smooth, predictable distributions suitable for efficient approximation without discarding.

## Core Concept

PISA implements a unified "exact-or-approximate" strategy:

1. **Exact Computation**: Critical blocks identified through query-key relevance and block covariance norms receive full attention
2. **Efficient Approximation**: Non-critical blocks use block-wise zeroth-order Taylor expansion with global first-order correction
3. **Covariance-Aware Selection**: Block selection scoring incorporates norm of block covariance matrices, ensuring heterogeneous blocks receive exact computation

The hybrid-order approximation combines efficiency (single shared global correction) with accuracy (block-wise base expansion).

## Architecture Overview

- **Block Partitioning**: Divide attention into regular blocks
- **Relevance Scoring**: Compute query-key affinity per block
- **Covariance Analysis**: Evaluate block-wise covariance norms
- **Block Classification**: Select critical blocks for exact computation
- **Exact Attention**: Full softmax for critical blocks
- **Approximate Attention**: Taylor expansion for non-critical blocks
- **Global Correction**: First-order term applied uniformly across approximations
- **Fused Kernel**: Custom GPU implementation combining exact and approximate paths

## Implementation

The implementation involves block selection, exact computation, and approximation with global correction.

Score blocks for criticality using relevance and covariance:

```python
import torch
import torch.nn.functional as F

def score_attention_blocks(query, key, block_size=64):
    """Score blocks for criticality based on relevance and covariance."""
    batch_size, num_heads, seq_len, d_k = query.shape
    num_blocks = (seq_len + block_size - 1) // block_size

    block_scores = []

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)

        # Extract block
        query_block = query[:, :, start:end, :]
        key_block = key[:, :, start:end, :]

        # Compute relevance: average attention affinity
        relevance = torch.matmul(query_block, key_block.transpose(-2, -1)).mean()

        # Compute covariance norm (heterogeneity measure)
        key_centered = key_block - key_block.mean(dim=2, keepdim=True)
        cov = torch.matmul(key_centered.transpose(-2, -1), key_centered)
        cov_norm = torch.norm(cov, p='fro')

        # Combined score: relevance + covariance heterogeneity
        score = relevance + 0.1 * cov_norm
        block_scores.append(score)

    return torch.stack(block_scores)

block_scores = score_attention_blocks(query, key)
```

Select critical blocks and compute exact attention:

```python
def select_critical_blocks(block_scores, sparsity_ratio=0.5):
    """Select top blocks for exact computation."""
    num_critical = max(1, int(len(block_scores) * sparsity_ratio))
    critical_indices = torch.topk(block_scores, num_critical)[1]
    return sorted(critical_indices.tolist())

def compute_exact_attention(query, key, value, start_idx, end_idx):
    """Compute full softmax attention for critical block."""
    # Standard scaled dot-product attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / (key.shape[-1] ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights

critical_indices = select_critical_blocks(block_scores)
```

Approximate non-critical blocks with Taylor expansion and global correction:

```python
def compute_approximate_attention(query_block, key_block, value_block):
    """Approximate attention using block-wise Taylor expansion."""
    batch_size, num_heads, block_len, d_k = query_block.shape

    # Zeroth-order approximation: mean attention
    # Treat non-critical blocks as having uniform attention
    mean_value = value_block.mean(dim=2, keepdim=True)  # (B, H, 1, D)

    # Replicate across sequence dimension
    approx_output = mean_value.expand(-1, -1, block_len, -1)

    return approx_output

def apply_global_correction(exact_outputs, approx_outputs, critical_mask, alpha=0.1):
    """Apply first-order global correction to approximations."""
    batch_size, num_heads, seq_len, d_v = approx_outputs.shape

    # Compute average correction from exact blocks
    exact_correction = exact_outputs[critical_mask].mean(dim=0, keepdim=True)

    # Add correction scaled by alpha
    corrected_output = approx_outputs + alpha * exact_correction

    return corrected_output
```

Assemble exact and approximate outputs:

```python
def hybrid_sparse_attention(query, key, value, block_size=64, sparsity_ratio=0.5):
    """Hybrid exact-approximate attention mechanism."""
    batch_size, num_heads, seq_len, d_k = query.shape
    num_blocks = (seq_len + block_size - 1) // block_size

    # Score and select critical blocks
    block_scores = score_attention_blocks(query, key, block_size)
    critical_indices = select_critical_blocks(block_scores, sparsity_ratio)

    output = torch.zeros_like(value)
    attn_weights = torch.zeros((batch_size, num_heads, seq_len, seq_len), device=value.device)

    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)

        query_block = query[:, :, start:end, :]
        key_block = key[:, :, start:end, :]
        value_block = value[:, :, start:end, :]

        if i in critical_indices:
            # Exact computation
            block_output, block_attn = compute_exact_attention(
                query_block, key_block, value_block, start, end
            )
        else:
            # Approximate computation
            block_output = compute_approximate_attention(
                query_block, key_block, value_block
            )
            block_attn = None

        output[:, :, start:end, :] = block_output
        if block_attn is not None:
            attn_weights[:, :, start:end, start:end] = block_attn

    # Apply global correction to approximated regions
    approx_mask = torch.ones(num_blocks, dtype=bool)
    approx_mask[critical_indices] = False

    if approx_mask.any():
        output = apply_global_correction(output, output, approx_mask)

    return output

attention_output = hybrid_sparse_attention(query, key, value)
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|-----------------|-------|
| Sparsity Ratio | 30-50% critical blocks | Higher ratio reduces approximation error |
| Block Size | 64-128 tokens | Balanced compute and memory |
| Global Correction Weight (α) | 0.05-0.15 | Tune based on FID metrics |
| Hardware Target | GPU with tensor cores | Custom fused kernel critical |
| Sequence Length | 512-2048 (video/image) | Benefits increase with length |
| Validation Metric | FID/LPIPS, not speed alone | Quality preservation critical |

**When to use**: For diffusion models on high-resolution images/video. When you need drop-in replacement for pretrained attention without retraining.

**When NOT to use**: For small models where attention isn't bottleneck. For tasks requiring precise spatial relationships (impossible to approximate accurately).

**Common pitfalls**:
- Sparsity too aggressive creates visible artifacts—monitor FID scores closely
- Global correction weight requires tuning—start conservative (α=0.05)
- Block covariance computation can be expensive—compute offline if possible
- Hardware kernel critical for speedup—naive PyTorch implementation may be slower
- Critical block selection must be deterministic—avoid data-dependent selection variability

## Reference

PISA: Piecewise Sparse Attention
https://arxiv.org/abs/2602.01077
