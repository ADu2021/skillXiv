---
name: jacobi-parallel-decoding
title: "Jacobi Forcing: Progressive Distillation for Fast and Accurate Causal Parallel Decoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.14681
keywords: [parallel-decoding, distillation, language-models, inference-speedup, causal-attention]
description: "Enable efficient parallel decoding while preserving causal inference properties through progressive distillation with cyclic noise scheduling. Reduce training complexity via block-wise sparse attention, implement rejection recycling and multi-block decoding. Achieve 3.8-4.0× wall-clock speedup on coding benchmarks."
---

## Skill Summary

Jacobi Forcing introduces progressive distillation for training autoregressive language models as efficient parallel decoders while maintaining their causal inference properties. The method uses cyclic noise scheduling with linearly increasing ratios to reduce longest spans of consecutive noisy inputs, employs sequence packing and block-wise sparse attention to reduce training complexity, and iteratively trains on progressively larger block sizes. Inference optimizations include rejection recycling and multi-block decoding, achieving 3.8-4.0× wall-clock speedup on coding benchmarks.

## When To Use

- Accelerating autoregressive model inference with parallel decoding
- Scenarios requiring preservation of causal attention properties
- Projects where token-level parallelism can improve latency
- Research on efficient training for parallel decoding

## When NOT To Use

- Applications already meeting latency goals with simpler methods
- Models with strict sequence length constraints preventing block parallelism
- Domains where preserving causal properties isn't critical
- Scenarios with limited training compute for progressive distillation

## Core Technique

Multiple innovations enable efficient parallel decoding:

**1. Progressive Noise Schedule**
Rather than randomly masking tokens during training, use "cyclic strategy with window size w, where the noise ratio linearly increases from 0 to 1 within each window." This reduces the longest span of consecutive noisy inputs, making learning more tractable and enabling progressive difficulty increase.

**2. Noise-Aware Causal Attention**
Employ sequence packing and block-wise sparse attention masks to reduce training complexity from O(N) to O(1) forward passes, enabling efficient gradient computation. Maintain causal properties within and across blocks.

**3. Progressive Distillation**
Iteratively train on trajectories with progressively larger block sizes, allowing the model to handle increasingly complex parallel decoding tasks. Start with small blocks where dependencies are simple, gradually increase complexity.

**4. Inference Optimization**
- Rejection recycling: reuse high-quality tokens from previous iterations
- Multi-block decoding: maintain multiple blocks simultaneously
These techniques boost practical speedup to nearly 4.0× wall-clock speedup.

## Implementation Notes

Implement cyclic progressive noise scheduling with window-based increasing ratios. Design block-wise sparse attention masks for causal parallel decoding. Start training with small block sizes, progressively increase block size during distillation. Implement rejection recycling selecting high-quality tokens. Add multi-block decoding for throughput optimization.

## References

- Original paper: Jacobi Forcing (Dec 2025)
- Jacobi iteration for parallel decoding
- Progressive distillation strategies
