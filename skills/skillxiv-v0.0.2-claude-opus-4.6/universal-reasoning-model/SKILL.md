---
name: universal-reasoning-model
title: "Universal Reasoning Model: Enhanced Universal Transformers with ConvSwiGLU and Truncated Backpropagation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.14693
keywords: [universal-transformers, reasoning, recurrent-networks, complex-reasoning, ARC-benchmark]
description: "Enhance Universal Transformers for complex reasoning through ConvSwiGLU modules integrating depthwise convolution into feed-forward blocks and truncated backpropagation through loops (TBPTL) restricting gradient computation to final iterations. Achieve state-of-the-art on ARC-AGI: 53.8% on ARC-1, 16.0% on ARC-2."
---

## Skill Summary

The Universal Reasoning Model (URM) enhances Universal Transformers through two key innovations: (1) ConvSwiGLU module integrating depthwise short convolution into standard SwiGLU feed-forward blocks to strengthen non-linearity through local contextual interactions, and (2) Truncated Backpropagation Through Loops (TBPTL) restricting gradient computation to only final recurrent loops, avoiding noise accumulation and instability. Results demonstrate performance gains stem from recurrent inductive bias and nonlinear components rather than elaborate designs, achieving state-of-the-art ARC-AGI performance.

## When To Use

- Building systems for complex reasoning tasks like ARC benchmarks
- Scenarios requiring recurrent computation without elaborate architectural complexity
- Projects exploring Universal Transformers as alternatives to standard transformers
- Research on efficient training of recurrent deep networks

## When NOT To Use

- Simple tasks where recurrent computation adds unnecessary overhead
- Scenarios where TBPTL training stability concerns are critical
- Applications with strict parameter budgets where Universal Transformers don't fit
- Domains not benefiting from recurrent processing patterns

## Core Technique

Two key innovations enhance reasoning capability:

**1. ConvSwiGLU Module**
Integrate depthwise short convolution into standard SwiGLU feed-forward block. As the module combines:
- SwiGLU gating mechanism for nonlinearity
- Depthwise convolution for local token interactions

This "strengthens the non-linearity of Universal Transformer" by injecting local contextual interactions between tokens, enabling richer feature combinations.

**2. Truncated Backpropagation Through Loops (TBPTL)**
Restrict gradient computation to only the final recurrent loops during training, avoiding "noise accumulation and instability" from propagating gradients through all iterations. This focuses learning on the most recent, most informative loop iterations.

**3. Key Insight**
Extensive analysis reveals performance gains in Universal Transformers stem primarily from recurrent inductive bias and nonlinear components rather than elaborate architectural designs. This justifies the relatively simple ConvSwiGLU + TBPTL approach.

## Results

- ARC-AGI 1: 53.8% accuracy
- ARC-AGI 2: 16.0% accuracy
- State-of-the-art performance on complex reasoning benchmarks

## Implementation Notes

Start with Universal Transformer base. Replace standard SwiGLU in feed-forward layers with ConvSwiGLU (adding depthwise convolution). Implement TBPTL during training: truncate gradient flow to final K recurrent loops. Validate on reasoning benchmarks. Monitor training stability improvements from TBPTL.

## References

- Original paper: Universal Reasoning Model (Dec 2025)
- Universal Transformers architecture
- Recurrent neural network training techniques
