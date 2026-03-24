---
name: signround-v2-low-bit-quant
title: "SignRoundV2: Extremely Low-Bit Post-Training Quantization via DeltaLoss Sensitivity"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.04746
keywords: [quantization, model-compression, low-bit, post-training, layer-wise-bit-allocation]
description: "DeltaLoss sensitivity metric combining gradient and quantization-induced parameter deviation for adaptive bit-width allocation, with lightweight pre-tuning search for scale initialization, enabling competitive accuracy at 4-5 bits in 2.5-6 hours."
---

## Summary

SignRoundV2 presents two main technical innovations for extremely low-bit post-training quantization. The primary contribution is DeltaLoss, a sensitivity metric capturing both local parameter distortions and global impact on task loss for reliable layer-wise bit allocation. The secondary contribution is lightweight pre-tuning search for initialization, enabling stable and accurate quantization at extremely low bit-widths (2-5 bits) in minimal time.

## Core Technique

**DeltaLoss Sensitivity Metric:** Measures importance of each layer by combining:
1. Gradient information (how sensitive loss is to weight changes)
2. Quantization-induced deviation (how much quantization distorts parameters)

```
DeltaLoss_layer = ||∇_W L|| · ||W - Q(W)||
```

This captures both local and global impact—sensitive layers get more bits.

**Pre-Tuning Scale Search:** Before main quantization, perform lightweight search for optimal quantization scale (step size) per layer:
```
scale_opt = argmin_scale MSE(Q_scale(W), W)
```

**Adaptive Bit-Width Allocation:** Use DeltaLoss to allocate bits from a budget:
```
bits_layer = base_bits + extra_bits[DeltaLoss_rank(layer)]
```
Critical layers get extra bits; less important layers get fewer.

## Implementation

**DeltaLoss computation:**
```python
def compute_deltaloss_sensitivity(weight, gradient, quant_weight):
    # Gradient magnitude (local sensitivity)
    gradient_norm = torch.norm(gradient.flatten())

    # Quantization deviation (distortion magnitude)
    deviation = torch.norm(weight - quant_weight)

    # Combined sensitivity
    deltaloss = gradient_norm * deviation
    return deltaloss
```

**Pre-tuning scale search:**
```python
def search_optimal_scale(weight, bits=4):
    scales = torch.linspace(0.001, 1.0, 100)
    best_scale = None
    best_mse = float('inf')

    for scale in scales:
        # Quantize with this scale
        quant_weight = quantize_symmetric(weight / scale, bits) * scale
        mse = torch.mean((weight - quant_weight) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_scale = scale

    return best_scale
```

**Adaptive bit allocation:**
```python
def allocate_bits_adaptive(layers, total_bits_budget, deltaloss_scores):
    num_layers = len(layers)
    base_bits = total_bits_budget // num_layers

    # Normalize DeltaLoss scores
    dl_normalized = deltaloss_scores / deltaloss_scores.sum()

    bit_allocation = {}
    for layer_idx, layer in enumerate(layers):
        # Allocate bits proportional to importance
        extra_bits = int(total_bits_budget * dl_normalized[layer_idx])
        bit_allocation[layer] = base_bits + extra_bits

    return bit_allocation
```

**Quantization with allocated bits:**
```python
def quantize_with_allocation(model, bit_allocation):
    quantized_model = {}

    for layer, bits in bit_allocation.items():
        weight = model[layer]

        # Search optimal scale for this bit-width
        optimal_scale = search_optimal_scale(weight, bits)

        # Quantize
        normalized_weight = weight / optimal_scale
        quant_weight = quantize_symmetric(normalized_weight, bits) * optimal_scale

        quantized_model[layer] = quant_weight

    return quantized_model
```

## When to Use

- Extreme model compression for mobile/edge deployment (2-5 bits)
- Scenarios where quantization must complete in hours not days
- Applications requiring reliable bit-width allocation across layers
- Tasks where post-training quantization is preferred over QAT

## When NOT to Use

- Scenarios with abundant GPU for QAT training
- Applications requiring per-model custom tuning
- Tasks where 8-bit quantization is sufficient
- Real-time quantization where search overhead matters

## Key References

- Post-training quantization and sensitivity analysis
- Layer-wise bit allocation and adaptive compression
- Low-bit quantization and extreme compression
- Quantization-aware scale selection
