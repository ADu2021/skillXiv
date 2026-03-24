---
name: wush-adaptive-quantization-transforms
title: "WUSH: Near-Optimal Adaptive Transforms for LLM Quantization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.00956
keywords: [quantization, llm-compression, adaptive-transforms, linear-algebra, weight-activation-quantization]
description: "Derives closed-form optimal linear blockwise transforms combining fixed Hadamard matrices with data-dependent components from second-order statistics, providing provably near-optimal quantization for both weights and activations with GPU efficiency."
---

## Summary

WUSH derives closed-form optimal linear blockwise transforms for joint weight-activation quantization in LLMs. The key innovation combines a fixed Hadamard matrix with a data-dependent component derived from second-order statistics of weights and activations, yielding non-orthogonal adaptive transforms that are provably near-optimal for both floating-point and integer quantizers while remaining computationally efficient.

## Core Technique

**Joint Quantization Optimization:** Rather than quantizing weights and activations separately, optimize a transform that works well for both:
```
x_quantized = Q(T * x)  # T is learned transform, Q is quantizer
```

**Optimal Transform Derivation:** The transform minimizes quantization error under the constraint that it's a linear combination:
```
T = H @ D  # H is fixed Hadamard, D is data-dependent diagonal
```

The Hadamard component provides rotation invariance, while the diagonal component adapts to data statistics.

**Closed-Form Solution:** Compute optimal D from:
```
D_optimal = diag(sqrt(var(weights) @ var(activations)))
```
using second-order statistics—no iterative optimization needed.

## Implementation

**Hadamard matrix:** Use fixed Walsh-Hadamard matrices (highly structured, GPU-efficient):
```python
import hadamard
H = hadamard.walsh_hadamard_matrix(dim)
```

**Data-dependent diagonal:** Compute from training data:
```python
# Collect weight and activation statistics
weight_var = var(all_weights)  # Per-channel variance
act_var = var(all_activations)  # Per-channel variance

# Derive diagonal scaling
D = diag(sqrt(weight_var * act_var))
```

**Transform matrix:** Combine components:
```python
T = H @ D @ H.T  # Final transform
```

**Quantization application:**
```python
def quantize_with_transform(x, bits=4):
    x_transformed = T @ x
    x_quantized = round_to_bits(x_transformed, bits)
    return inv(T) @ x_quantized
```

**Blockwise application:** Apply transforms per block (e.g., 128-dim blocks):
```python
for block_idx in range(0, dim, block_size):
    block = x[block_idx:block_idx+block_size]
    T_block = compute_transform(block)
    x[block_idx:block_idx+block_size] = quantize_with_transform(block, T_block, bits)
```

## When to Use

- Quantizing large language models with memory constraints
- Scenarios requiring both weight and activation quantization
- Applications needing theoretical near-optimality guarantees
- Tasks where GPU-efficient transforms matter for deployment

## When NOT to Use

- Scenarios where simple uniform quantization suffices
- Applications where transform overhead is prohibitive
- Models where post-quantization fine-tuning is unavailable
- Real-time inference where quantization computation adds latency

## Key References

- Quantization-aware training and post-training quantization
- Linear algebra and matrix factorization
- Hadamard transforms and structured matrices
- Joint optimization for multiple objectives
