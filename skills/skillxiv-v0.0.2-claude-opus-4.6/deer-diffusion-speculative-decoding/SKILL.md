---
name: deer-diffusion-speculative-decoding
title: "DEER: Speculative Decoding via Diffusion Language Models with Parallel Draft Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.15176
keywords: [speculative-decoding, diffusion-lm, efficient-inference, parallel-generation, language-models]
description: "Enable efficient speculative decoding by training discrete diffusion language models for parallel draft generation. Use AR-style distillation and scribe refinement to train dLLMs. Eliminate left-to-right error accumulation through independent parallel proposals. Achieve 5.54× speedup on HumanEval vs. 2.41× for AR-based methods."
---

## Skill Summary

DEER (Draft with diffusion, vErify with autoRegressive) introduces speculative decoding using discrete diffusion language models for efficient parallel draft generation. The approach trains dLLMs through two-stage alignment: AR-style distillation enabling prefix-conditioned continuation and scribe refinement sharpening predictions near verification boundaries. Unlike AR drafters suffering left-to-right uncertainty accumulation, DEER's parallel generation makes "the proposal at position i independent of previously drafted tokens," enabling acceptance lengths up to 32 tokens vs. ~10 for competing methods. Results show 5.54× speedup on HumanEval.

## When To Use

- Building efficient LLM inference systems requiring high acceptance rates
- Scenarios where speculative decoding speedups are critical for latency
- Projects where parallel draft generation prevents error accumulation
- Research on efficient alternatives to autoregressive drafting

## When NOT To Use

- Applications already meeting latency goals with simpler methods
- Scenarios where training discrete diffusion models is computationally infeasible
- Domains where AR drafters' lower acceptance rates aren't performance bottlenecks
- Models with strict parameter budgets for additional draft model

## Core Technique

Two-stage alignment pipeline trains diffusion drafters:

**1. AR-Style Distillation (Stage I)**
Train discrete diffusion language model (dLLM) for prefix-conditioned continuation. Learn from truncated teacher answers marked with SEP token, enabling model to generate coherent suffixes given fixed prefix. This bridges AR teacher and diffusion student paradigms.

**2. Scribe Refinement (Stage II)**
Enhance accuracy near verification boundary through "weighted suffix masking with exponentially decaying loss," focusing training on tokens most critical for acceptance. Improve predictions most likely to affect speculative decoding outcome.

**3. Parallel Generation Advantage**
Unlike AR drafters with left-to-right uncertainty accumulation, dLLM generates entire token blocks in single denoising step. "The proposal at position i is independent of previously drafted tokens," preventing error propagation and enabling acceptance lengths up to 32 tokens versus ~10 for competing methods.

## Speedup Results

- 5.54× speedup on HumanEval vs. EAGLE-3's 2.41×
- Substantially better than AR-based speculative decoding approaches
- Maintains competitive verification overhead

## Implementation Notes

Start with pretrained teacher AR model. Train discrete diffusion LM for parallel generation conditioned on prefixes. Implement AR-style distillation with SEP token marking. Add scribe refinement focusing on boundary tokens. Integrate into verifier framework for speculative decoding. Measure acceptance rates and overall speedup.

## References

- Original paper: DEER (Dec 2025)
- Discrete diffusion language models
- Speculative decoding frameworks
