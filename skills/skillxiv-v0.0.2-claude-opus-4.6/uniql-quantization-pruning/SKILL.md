---
name: uniql-quantization-pruning
title: "UniQL: Unified Quantization and Low-Rank Compression for Edge Deployment"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.03383
keywords: [model-compression, quantization, pruning, edge-deployment, efficient-inference]
description: "Single cloud-side weight-sorting and fine-tuning supporting multiple on-device pruning rates via efficient SVD and MLP decomposition, achieving 4-5.7× memory reduction and 2.7-3.4× throughput across Transformers, SSMs, and hybrid architectures."
---

## Summary

UniQL introduces a unified post-training framework enabling flexible on-device pruning after cloud-side quantization and fine-tuning. Key innovations include pseudo-inverse-free MLP decomposition, quantization-aware SVD for attention layers, and state-aware sorting for SSMs, enabling a single cloud training pass to support multiple pruning rates at deployment.

## Core Technique

**Single Cloud Pass:** Perform all heavy computation once in cloud:
1. Weight sorting and ranking
2. Joint fine-tuning with quantization
3. Store sorted weights for deployment

**Multiple Pruning Rates at Device:** On edge devices, select different pruning rates without retraining:
```
cloud: [w_1, w_2, ..., w_n]  # Sorted weights
device_1: keep top-30% weights
device_2: keep top-20% weights
device_3: keep top-50% weights
```

**Architecture-Specific Compression:**
- **Transformers:** Quantization-aware SVD for attention+FFN layers
- **SSMs:** State-aware sorting for selective sparsification
- **Hybrids:** Coordinated compression across mixed architectures

## Implementation

**Weight sorting:** Rank weights by importance (magnitude-based or gradient-based):
```python
def rank_weights(weights, method='magnitude'):
    if method == 'magnitude':
        importance = abs(weights)
    else:  # gradient-based
        importance = abs(weights * gradients)

    sorted_idx = argsort(importance)
    return sorted_idx
```

**MLP decomposition (pseudo-inverse-free):**
```python
def decompose_mlp(weight, target_rank):
    # Instead of SVD: use iterative power method
    # Avoid explicit pseudo-inverse computation
    U, s, Vt = power_iteration(weight, rank=target_rank)
    W_low_rank = U @ diag(s) @ Vt
    return W_low_rank
```

**Quantization-aware SVD:**
```python
def quantization_aware_svd(weight, bits=4):
    # Account for quantization error in SVD
    U, s, Vt = svd(weight)

    # Truncate based on both rank AND quantization noise
    quantization_noise = estimate_quant_noise(weight, bits)
    threshold = quantization_noise
    keep_idx = s > threshold

    U_trunc = U[:, keep_idx]
    s_trunc = s[keep_idx]
    Vt_trunc = Vt[keep_idx, :]

    return U_trunc @ diag(s_trunc) @ Vt_trunc
```

**On-device pruning:**
```python
def prune_at_device(cloud_weights, prune_rate):
    # cloud_weights already sorted
    num_keep = int(len(cloud_weights) * (1 - prune_rate))
    pruned = cloud_weights[:num_keep]
    # Pad or interpolate zeros
    result = pad(pruned, original_shape)
    return result
```

## When to Use

- Mobile and edge deployment with strict memory constraints
- Scenarios where cloud training is feasible but device retraining is not
- Applications supporting multiple device configurations from single model
- Tasks combining quantization and pruning for maximum compression

## When NOT to Use

- Cloud-only inference where efficiency is less critical
- Scenarios where on-device fine-tuning is available
- Applications requiring custom per-device pruning patterns
- Tasks where sorted importance ranking is not reliable

## Key References

- Model compression and quantization techniques
- Low-rank decomposition and SVD
- Pruning and sparsification methods
- On-device inference and edge deployment
