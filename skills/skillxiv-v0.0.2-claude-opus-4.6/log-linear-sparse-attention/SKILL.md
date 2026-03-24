---
name: log-linear-sparse-attention
title: "Trainable Log-linear Sparse Attention for Efficient Diffusion Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.16615
keywords: [sparse-attention, diffusion-transformers, efficient-inference, training-efficiency, log-linear-complexity]
description: "Reduce self-attention complexity from O(N²) to O(N log N) through hierarchical token selection and enrichment. Perform hierarchical Top-K selection progressively adopting sparse Top-K at each level. Implement sparse index transpose algorithm avoiding dense mask construction. Achieve 28.27× faster inference and 6.09× faster training."
---

## Skill Summary

This work introduces Log-linear Sparse Attention (LLSA), a trainable mechanism that reduces self-attention complexity from quadratic O(N²) to log-linear O(N log N) for Diffusion Transformers. The approach performs hierarchical compression through progressive Top-K selection, adopts sparse indices at each hierarchy level, and incorporates coarse representations from multiple levels. An efficient GPU implementation develops a sparse index transpose algorithm operating directly on sparse indices for both forward and backward passes, eliminating expensive dense mask construction. Results show 28.27× faster inference and 6.09× faster training on 256×256 pixel sequences while maintaining generation quality.

## When To Use

- Accelerating diffusion transformer inference for image/video generation
- Scenarios where sequence length creates O(N²) attention bottleneck
- Projects with limited compute budget for generation
- Research on sparse attention mechanisms for transformers

## When NOT To Use

- Short sequences already efficient with full attention
- Applications where sparse approximation introduces artifacts
- Domains requiring full attention interactions between all tokens
- Scenarios where sparse overhead isn't justified by sequence length

## Core Technique

Two key components enable log-linear attention:

**1. Hierarchical Compression and Selection**
Rather than single-level token selection, perform "hierarchical Top-K selection, progressively adopting sparse Top-K selection with the indices found at the previous level." This multi-scale approach enables:
- Efficient long-range token selection
- Reduced total selected tokens while maintaining context
- Progressive refinement across hierarchy levels

**2. Hierarchical KV Enrichment**
"Preserve global context while using fewer tokens of different granularity during attention computation" by incorporating coarse representations from multiple hierarchy levels. This enables modeling tokens at different levels of abstraction simultaneously.

**3. Efficient GPU Implementation**
Develop sparse index transpose algorithm "operating directly on sparse indices" for both forward and backward passes. This eliminates expensive dense mask construction that would negate sparsity benefits.

## Practical Results

- 256×256 pixel sequences:
  - Inference speedup: 28.27×
  - Training speedup: 6.09×
- Smaller K values than competing methods
- Generation quality maintained comparable to full attention

## Implementation Notes

Implement hierarchical Top-K selection operating progressively across hierarchy levels. Add hierarchical KV enrichment incorporating multi-scale context. Implement efficient GPU kernel for sparse index operations avoiding dense masks. Validate inference and training speedups on your sequence lengths.

## References

- Original paper: Trainable Log-linear Sparse Attention (Dec 2025)
- Sparse attention mechanisms
- Diffusion transformer architectures
