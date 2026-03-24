---
name: repo-context-repositioning
title: "RePo: Dynamic Context Re-positioning for Language Models via Learnable Position Assignment"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.14391
keywords: [position-encoding, context-reorganization, language-models, learned-positioning, cognitive-load]
description: "Enable language models to dynamically assign continuous position values to tokens based on contextual relevance instead of fixed linear positions. Implement learnable SwiGLU module extracting position information, integrate with RoPE for end-to-end optimization. Improves performance on noisy contexts and long-sequence tasks."
---

## Skill Summary

RePo introduces context re-positioning, a lightweight neural module enabling models to dynamically assign continuous position values based on token relevance rather than fixed linear order. The module comprises two components: a SwiGLU sub-layer extracting position information from token hidden states and a linear transformation assigning real-valued positions. Experiments on OLMo-2 models demonstrate gains on noisy contexts, structured data, and long-context tasks while maintaining performance on general benchmarks.

## When To Use

- Working with noisy contexts where irrelevant tokens hurt performance
- Long-context tasks where linear position assumptions don't hold
- Structured data with meaningful hierarchies not captured by left-to-right order
- Research exploring dynamic position encoding alternatives

## When NOT To Use

- Tasks already performing well with fixed position encodings
- Real-time applications requiring fixed position computation cost
- Domains where left-to-right order is inherently meaningful
- Scenarios where additional learned parameters exceed budget constraints

## Core Technique

The RePo module enables dynamic position assignment:

**1. Position Representation**
A SwiGLU sub-layer extracts position information from token hidden states. This gated linear unit learns which tokens should be positioned early (high priority) vs. late (low priority).

**2. Position Assignment**
A linear transformation assigns real-valued positions in continuous space based on extracted position information. As authors state, RePo enables models to "assign token positions that capture contextual dependencies, rather than relying on pre-defined order."

**3. Integration with RoPE**
Assigned positions are integrated with differentiable position encodings like RoPE (Rotary Position Embedding), allowing end-to-end optimization. The model jointly learns position assignment and position encoding.

**4. Cognitive Load Theory Inspiration**
Drawn from Cognitive Load Theory, the approach argues that linear position assignments unnecessarily burden working memory. By allowing the model to reassign positions based on token relevance—grouping related information and deprioritizing noise—the model can allocate more capacity to deeper reasoning tasks.

## Implementation Notes

Start with your language model using RoPE or similar position encoding. Add learnable SwiGLU module that outputs position scores for each token. Compute real-valued continuous positions from scores. Integrate positions with your existing position encoding mechanism. Fine-tune end-to-end on your task.

## References

- Original paper: RePo (Dec 2025)
- RoPE position encoding
- Cognitive load theory in neural networks
