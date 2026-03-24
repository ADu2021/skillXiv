---
name: llada2-diffusion-scaling
title: "LLaDA2.0: Scaling Diffusion Language Models to 100B Parameters via Warmup-Stable-Decay Training"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.15745
keywords: [diffusion-lm, large-scale-models, parallel-generation, 100B-parameters, training-efficiency]
description: "Convert pre-trained autoregressive models into large-scale diffusion language models via Warmup-Stable-Decay training strategy. Progressively increase block size during warmup, perform stable diffusion training, then decay to smaller blocks for inference. Achieve 535 tokens-per-second with document-level masking and confidence-aware training."
---

## Skill Summary

LLaDA2.0 scales diffusion language models to 100B parameters by systematically converting pre-trained autoregressive checkpoints through Warmup-Stable-Decay (WSD) training strategy. Rather than training from scratch, the approach progressively transforms AR checkpoints across three phases: Warmup increases block size gradually transforming AR model into full-sequence masked diffusion model, Stable performs full-sequence diffusion training on large-scale corpora, and Decay reverts to smaller blocks for practical inference. Additional innovations include document-level attention masking and confidence-aware parallel training, achieving 535 tokens-per-second inference speed.

## When To Use

- Scaling diffusion language models to 100B+ parameters
- Projects converting existing large AR checkpoints to parallel generation
- Scenarios where parallel decoding benefits outweigh autoregressive efficiency
- Research on efficient diffusion model scaling

## When NOT To Use

- Applications requiring pure autoregressive inference properties
- Scenarios with strict latency requirements (WSD training has overhead)
- Small models where diffusion complexity adds overhead
- Domains benefiting specifically from left-to-right generation order

## Core Technique

Three coordinated training phases enable efficient transformation:

**1. Warmup Phase**
"Progressively increase the block size in block diffusion language models (BDLM) to gradually transform the AR model into a full-sequence masked diffusion language model." This gradual transformation preserves learned representations while shifting from AR to diffusion paradigm.

**2. Stable Phase**
Perform full-sequence diffusion training on large-scale corpora with two innovations:
- Document-level attention masking: prevents spurious cross-document dependencies in packed training
- Complementary masking: during post-training ensures near-100% data utilization

**3. Decay Phase**
Revert to efficient smaller block sizes for practical inference, maintaining diffusion model benefits while achieving practical throughput (535 tokens-per-second).

**4. Confidence-Aware Parallel Training**
Sharpen predictions for faster decoding, allowing better utilization of parallel generation.

## Results

- 100B parameter model
- 535 tokens-per-second inference speed
- Competitive performance vs. comparable autoregressive baselines
- Diffusion models' parallel decoding advantages preserved

## Implementation Notes

Start with pre-trained 100B AR checkpoint. Implement progressive block size increase during warmup. Implement document-level attention masking for packed training. Apply complementary masking for 100% data utilization. Decay to smaller blocks for efficient inference. Measure throughput and accuracy.

## References

- Original paper: LLaDA2.0 (Dec 2025)
- Block diffusion language models
- Large-scale model training and conversion
