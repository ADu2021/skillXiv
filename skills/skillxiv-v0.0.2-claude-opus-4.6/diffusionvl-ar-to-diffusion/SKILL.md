---
name: diffusionvl-ar-to-diffusion
title: "DiffusionVL: Converting Autoregressive Vision-Language Models into Efficient Diffusion VLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.15713
keywords: [vision-language, diffusion-models, paradigm-shift, parallel-generation, model-conversion]
description: "Convert pre-trained autoregressive vision-language models into diffusion VLMs without architectural modifications. Use block diffusion strategy enabling arbitrary-length generation and KV-cache reuse. Hybrid attention enforces bidirectional within blocks, causal between blocks. Requires less than 5% of data compared to prior diffusion VLM methods."
---

## Skill Summary

DiffusionVL demonstrates direct conversion of already vision-language aligned autoregressive models into diffusion VLMs through full-parameter diffusion finetuning. The approach converts the "next-token prediction paradigm into a diffusion paradigm." Block diffusion strategy enables arbitrary-length generation with intra-block parallel denoising and inter-block autoregressive decoding. Hybrid attention mechanism enforces bidirectional attention within blocks and causal attention between blocks. Remarkably, requires less than 5% of data compared to prior diffusion VLM methods, proving "the gap between dVLMs and AR-VLMs is minimal."

## When To Use

- Converting existing AR vision-language models to parallel-generation diffusion models
- Scenarios requiring efficient VLM inference with minimal retraining
- Projects where parallel generation benefits outweigh minor quality trade-offs
- Research on paradigm-agnostic model conversion techniques

## When NOT To Use

- Building VLMs from scratch (end-to-end diffusion training may be preferable)
- Applications where AR properties are critical (causal attention constraints)
- Scenarios with very limited training data (even 5% is substantial for some projects)
- Domains where hybrid attention mechanism causes artifacts

## Core Technique

Two main pathways enable vision-language paradigm conversion:

**1. AR-VLM to dVLM (Paradigm Shift)**
Direct conversion of already vision-language aligned autoregressive models through full-parameter diffusion finetuning. Convert the "next-token prediction paradigm into a diffusion paradigm" using minimal data.

**2. AR-LM to dVLM (Modality + Paradigm Shift)**
Two-stage approach: connector first aligns vision and text spaces using autoregressive training, then diffusion finetuning completes the conversion. Enables VLM creation from vision and language separately.

**3. Block Diffusion Strategy**
Enable arbitrary-length generation and KV-cache reuse through:
- Intra-block parallel denoising: generate block contents in parallel
- Inter-block autoregressive decoding: generate blocks sequentially

**4. Hybrid Attention Mechanism**
Enforce bidirectional attention within blocks (full context for parallel denoising) and causal attention between blocks (respecting generation order). This balances efficiency with coherence.

## Implementation Notes

Start with pre-trained AR-VLM. Implement block diffusion strategy with hybrid attention pattern. Fine-tune entire model with diffusion objective using your target data (only 5% of original AR training needed). Monitor accuracy preservation and measure parallel generation speedup. Validate on your target VLM tasks.

## References

- Original paper: DiffusionVL (Dec 2025)
- Block diffusion language models
- Vision-language model architectures
