---
name: sparse-lavida-diffusion
title: "Sparse-LaViDa: Efficient Sparse Parameterization for Multimodal Discrete Diffusion Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.14008
keywords: [sparse-diffusion, multimodal-models, discrete-diffusion, inference-efficiency, attention-optimization]
description: "Accelerate masked discrete diffusion models by dynamically truncating unnecessary masked tokens while maintaining complete representation through positional information. Use register tokens as compressed representations, implement step-causal attention masks for bidirectional context. Achieve 1.95-2.83× speedups across generation tasks."
---

## Skill Summary

Sparse-LaViDa introduces efficient parameterization for Masked Discrete Diffusion Models (MDMs) that accelerates inference without sacrificing quality. The framework dynamically truncates unnecessary masked tokens while preserving representation through positional metadata, uses sixty-four learnable register tokens as compressed representations of truncated content, and implements step-causal attention enabling efficient KV-cache usage. Results show 1.95-2.83× speedups across text-to-image, image editing, and visual reasoning tasks.

## When To Use

- Accelerating discrete diffusion models for real-time applications
- Projects requiring efficient bidirectional attention (unlike block-causal alternatives)
- Scenarios where both generation and editing are required (supports both tasks)
- Research on sparse attention mechanisms for diffusion models

## When NOT To Use

- Applications already achieving target latency with full attention
- Domains where register tokens reduce model expressiveness unacceptably
- Scenarios where position-only representation of truncated tokens is insufficient
- Models where step-causal attention causes artifacts

## Core Technique

Three key innovations enable sparse diffusion:

**1. Sparse Parameterization**
Rather than processing all tokens at each diffusion step, "dynamically truncate unnecessary masked tokens" while maintaining complete representation through positional information and sequence length metadata. This reduces computation without losing semantic information.

**2. Register Tokens**
Sixty-four special learnable tokens serve as "compressed representations of truncated tokens" to preserve modeling capacity lost during truncation. These registers capture aggregate information about truncated content.

**3. Step-Causal Attention Mask**
Implement specialized attention pattern enabling efficient KV-cache usage during inference while preserving bidirectional context necessary for image generation and editing tasks. This avoids limitations of block-causal approaches used in prior semi-autoregressive work.

## Speedup Results

- Text-to-image generation: 1.95× speedup
- Image editing: 2.83× speedup
- Visual math reasoning: 2.80× speedup

All speedups maintain performance comparable to baseline LaViDa-O while supporting bidirectional tasks like image inpainting.

## Implementation Notes

Start with masked discrete diffusion model. Implement dynamic token truncation based on masking status. Add learnable register tokens capturing compressed truncated information. Implement step-causal attention mask: bidirectional within steps, causal between steps. Measure speedups and quality on your target tasks.

## References

- Original paper: Sparse-LaViDa (Dec 2025)
- Masked discrete diffusion models
- Sparse attention mechanisms
