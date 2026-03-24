---
name: efficient-dlm-ar-conversion
title: "Efficient-DLM: Converting Autoregressive Models to Diffusion Language Models with Superior Accuracy-Throughput Trade-offs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.14067
keywords: [diffusion-lm, autoregressive-conversion, parallel-generation, block-wise-attention, position-dependent-masking]
description: "Systematically convert pretrained autoregressive models into efficient diffusion language models via block-wise attention and position-dependent masking. Efficient-DLM family (1.5B/4B/8B) maintains comparable accuracy to standard AR models while delivering 4.5× higher throughput."
---

## Skill Summary

Efficient-DLM presents a systematic framework for converting pretrained autoregressive models into parallel-decoding diffusion language models. The method combines block-wise attention with clean context, position-dependent token masking aligned with inference behavior, and comprehensive design analysis. Results show Efficient-DLM 8B maintains accuracy comparable to Qwen3 8B while delivering 4.5× higher throughput versus Dream 7B.

## When To Use

- Converting existing AR checkpoints to parallel-decoding diffusion models
- Projects requiring dramatic throughput improvements without expensive retraining from scratch
- Scenarios balancing accuracy preservation with substantial speedup gains
- Research on efficient alternatives to autoregressive decoding

## When NOT To Use

- Applications requiring maximum per-token quality over throughput
- Models where AR properties are fundamentally necessary
- Scenarios where the AR-to-DLM conversion introduces unacceptable artifacts
- Domains where single-pass AR inference already meets latency requirements

## Core Technique

Three key technical components enable efficient AR-to-DLM conversion:

**1. Block-wise Attention Pattern**
Employ "block-wise attention with clean context" where each corrupted block conditions only on previously decoded clean context. This better preserves pretrained AR model weights while enabling KV caching, avoiding full bidirectional attention that would substantially alter learned representations.

**2. Position-Dependent Token Masking**
Identify training-test gap where uniform masking during training mismatches confidence-based sampling during inference (exhibiting left-to-right bias). Propose position-dependent masking assigning higher masking probabilities to later tokens:

> w_i(t) = exp[β(1-t)i]

This aligns training with test-time behavior, improving sample efficiency.

**3. Comprehensive Design Analysis**
Systematically study optimal block sizes, attention patterns, and training dynamics. Provide actionable guidelines for scalable AR-to-DLM conversion, enabling practitioners to apply the approach to different architectures.

## Implementation Notes

Start with pretrained AR checkpoint. Implement block-wise attention maintaining context from previous blocks. Apply position-dependent masking with learned parameters aligned to inference-time confidence. Train progressively scaling block sizes. Validate accuracy preservation and measure throughput improvements over AR baselines.

## References

- Original paper: Efficient-DLM (Dec 2025)
- Block diffusion language models
- Autoregressive model conversion techniques
