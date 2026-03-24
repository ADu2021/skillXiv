---
name: sliding-window-attention-adaptation
title: "Practical Sliding Window Attention Adaptation for Pre-trained Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.10411
keywords: [sliding-window-attention, long-context, adaptation, inference-speedup, language-models]
description: "Adapt full-attention language models to sliding window attention without expensive retraining. Combine five synergistic strategies (full decode, sink tokens, interleaved layers, chain-of-thought, fine-tuning) achieving 30-100% speedups while maintaining 90-100% accuracy."
---

## Skill Summary

Rather than retraining from scratch, this method systematically combines five complementary adaptation strategies to convert full-attention LLMs to sliding window attention (SWA). The key insight is that "no single ingredient suffices"—specific synergistic combinations of these methods effectively recover original performance. The approach identifies configurations achieving substantial speedups while preserving accuracy, depending on whether efficiency or quality takes priority.

## When To Use

- Converting existing full-attention models to efficient sliding window attention
- Scenarios requiring 30-100% inference speedups with minimal accuracy loss
- Long-context applications where full attention is computationally prohibitive
- Projects already deployed with full-attention models seeking efficiency gains

## When NOT To Use

- Models requiring full global attention for certain reasoning patterns
- Tasks where 10% accuracy loss is unacceptable (some configurations may not preserve perfect accuracy)
- Scenarios where the 5 adaptation strategies contradict your specific architectural constraints
- Models designed for short sequences where attention is already efficient

## Core Technique

The approach systematically combines five complementary strategies:

**1. Full Attention Decode**
Apply SWA only during prefilling (context encoding), switch to full attention during generation when sequence lengths are short.

**2. Keep First k Tokens**
Preserve attention to initial "sink" tokens while using SWA elsewhere, maintaining access to document-level context.

**3. Interleaving FA/SWA Layers**
Alternate between full-attention and SWA layers throughout the model, balancing efficiency and global context awareness.

**4. Chain-of-Thought**
Enable explicit reasoning during decoding to compensate for limited context windows, allowing the model to compensate for SWA context constraints.

**5. Fine-tuning**
Lightweight SWA-aware supervised fine-tuning on long-context data to adapt the model to the new attention pattern.

## Implementation Notes

Start with your full-attention model. Experiment with combinations of these five strategies:
- For maximum speed: use SWA prefill + SWA decode + interleaved layers
- For maximum quality: use full decode + keep-first-k + chain-of-thought + fine-tuning
- For balanced approach: use SWA prefill + full decode + interleaved layers + light fine-tuning

Recommended: Test recommended configurations on your specific task to find the right accuracy-efficiency trade-off.

## References

- Original paper: SWAA (Dec 2025)
- Sliding window attention mechanisms
- Long-context fine-tuning strategies
