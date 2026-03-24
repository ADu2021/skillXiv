---
name: sq-format-sparse-quantized
title: "SQ-format: A Unified Sparse-Quantized Hardware-friendly Data Format for LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.05409
keywords: [quantization, sparsification, post-training quantization, hardware efficiency, LLM inference]
description: "Unify sparse and quantized representations in a single hardware-friendly format for efficient LLM inference. Exploit complementary acceleration properties—high precision for sparse operations, low precision for dense—when W4A8 bottlenecks GPU throughput."
---

## Overview

SQ-format combines quantization and sparsification in a unified framework compatible with both new hardware and existing GPUs. The innovation leverages complementary acceleration properties where sparse matrices accelerate effectively at high precision while low-precision multiplication also benefits from acceleration.

## When to Use

- Post-training quantization of large language models
- LLM inference with W4A8 or similar bottlenecking GPU performance
- Scenarios where activation patterns contain outlier inequality
- Requiring static compression with minimal accuracy loss
- Deploying LLMs on resource-constrained hardware
- Applications needing Pareto improvements in accuracy-efficiency tradeoff

## When NOT to Use

- Models where sparsity is minimal (dense computations dominate)
- Cases where activation distributions are uniform
- Scenarios already achieving good results with W8A8
- Applications requiring full-precision model inference
- Hardware without sparse operation support

## Core Technique

Unified sparse-quantized format exploiting dual acceleration pathways:

```python
# SQ-format implementation overview
class SparseQuantizedFormat:
    def __init__(self, quantization_bits=4, sparse_threshold=0.5):
        self.q_bits = quantization_bits
        self.sparse_threshold = sparse_threshold

    def compress_weights(self, weights):
        """
        Compress weights using combined sparsification and quantization.
        Weights are both sparsified (set small values to zero) and quantized.
        """
        # Identify sparse patterns (large magnitude threshold)
        mask = torch.abs(weights) > self.sparse_threshold
        sparse_weights = weights * mask

        # Quantize sparse weights at high precision (8-bit)
        # Dense regions at low precision (4-bit)
        quantized = torch.zeros_like(weights)

        # High precision path for sparse regions
        sparse_region = sparse_weights[mask]
        quantized[mask] = self.quantize_8bit(sparse_region)

        # Low precision path for dense regions
        dense_region = weights[~mask]
        quantized[~mask] = self.quantize_4bit(dense_region)

        return quantized, mask

    def compress_activations(self, activations):
        """
        Compress activations with focus on outlier patterns.
        Particularly effective when activations have high inequality.
        """
        # Detect outlier inequality in activation distribution
        percentile_90 = torch.quantile(activations, 0.9)
        percentile_10 = torch.quantile(activations, 0.1)
        inequality_ratio = percentile_90 / (percentile_10 + 1e-8)

        if inequality_ratio > 2.0:
            # High inequality: use sparse representation
            threshold = percentile_10 + 0.5 * (percentile_90 - percentile_10)
            mask = torch.abs(activations) > threshold
            compressed = activations * mask
            return self.quantize_8bit(compressed), mask
        else:
            # Low inequality: dense quantization sufficient
            return self.quantize_4bit(activations), None

    def quantize_4bit(self, values):
        """Low-precision quantization for dense regions."""
        # Affine quantization to 4-bit range
        scale = (values.max() - values.min()) / 15.0
        zero_point = values.min()
        quantized = torch.clamp(
            (values - zero_point) / scale,
            0, 15
        ).round()
        return quantized * scale + zero_point

    def quantize_8bit(self, values):
        """High-precision quantization for sparse regions."""
        # Affine quantization to 8-bit range
        scale = (values.max() - values.min()) / 255.0
        zero_point = values.min()
        quantized = torch.clamp(
            (values - zero_point) / scale,
            0, 255
        ).round()
        return quantized * scale + zero_point

    def forward(self, input_tensor, weight, bias):
        """
        Inference-time computation using mixed precision.
        Sparse path uses 8-bit precision, dense path uses 4-bit.
        """
        weight_q, weight_mask = self.compress_weights(weight)
        act_q, act_mask = self.compress_activations(input_tensor)

        # Execute on hardware supporting both sparse and dense ops
        output = torch.nn.functional.linear(act_q, weight_q, bias)
        return output
```

The format provides hardware guidance for next-generation AI accelerators to support dual-path computation.

## Key Results

- State-of-the-art post-training quantization performance
- Effective for activations with outlier inequality patterns
- Enables Pareto improvements in accuracy-efficiency
- Compatible with both new and existing GPU hardware

## Implementation Notes

- Sparsification and quantization are complementary
- Outlier detection guides compression strategy selection
- Hardware implementation can parallelize sparse and dense paths
- No retraining required for post-training quantization

## References

- Original paper: https://arxiv.org/abs/2512.05409
- Focus: Hardware-efficient LLM inference
- Domain: Quantization, model compression, inference optimization
