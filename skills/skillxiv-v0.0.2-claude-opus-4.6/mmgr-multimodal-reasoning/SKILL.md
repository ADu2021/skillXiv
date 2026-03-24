---
name: mmgr-multimodal-reasoning
title: "MMGR: Multi-Modal Generative Reasoning Framework for Evaluating Video and Image Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.14691
keywords: [multimodal, reasoning, evaluation, video-generation, benchmark, generative-models]
description: "Evaluate whether generative models possess foundational reasoning capabilities. Develop five-ability framework (physical, logical, 3D spatial, 2D spatial, temporal reasoning) across abstract reasoning, embodied navigation, and physical commonsense benchmarks. Use structured rubric requiring simultaneous satisfaction of all sub-metrics."
---

## Skill Summary

MMGR introduces a comprehensive evaluation framework assessing reasoning capabilities of generative models across five complementary abilities: physical, logical, 3D spatial, 2D spatial, and temporal reasoning. The benchmark operationalizes this framework across three domains—abstract reasoning, embodied navigation, and physical commonsense—using structured rubrics requiring simultaneous satisfaction of all sub-metrics. Human evaluation on 1,853 samples validates findings and reveals critical misalignments between automated VLM assessment and ground-truth performance.

## When To Use

- Evaluating generative models for reasoning capabilities beyond perceptual quality
- Developing benchmarks assessing structured reasoning in video/image generation
- Research exploring what generative models actually understand about the physical world
- Projects requiring comprehensive reasoning evaluation beyond standard metrics

## When NOT To Use

- Simple text-to-image quality benchmarking (too comprehensive for basic use cases)
- Real-time model evaluation where benchmark overhead is prohibitive
- Domains where simpler metrics already adequately capture model capabilities
- Scenarios focused solely on fidelity without reasoning assessment

## Core Technique

The framework consists of three integrated components:

**1. Five-Ability Reasoning Framework**
Evaluates generative models across:
- Physical Reasoning: intuitive physics and object dynamics
- Logical Reasoning: abstract rule-following and symbolic manipulation
- 3D Spatial Reasoning: volumetric environment understanding
- 2D Spatial Reasoning: planar layout and composition
- Temporal Reasoning: causality and event sequencing

**2. Benchmark Structure**
Operationalizes framework across three complementary domains:
- Abstract Reasoning: Maze, Sudoku, ARC-AGI, Math (testing logic and 2D spatial skills)
- Embodied Navigation: 3D navigation, egocentric views, top-down planning (assessing spatial and temporal coherence)
- Physical Commonsense: fundamental physics concepts and sports scenarios (evaluating intuitive physics understanding)

**3. Structured Evaluation Rubric**
Unlike partial success metrics, use "a structured rubric requiring simultaneous satisfaction of all sub-metrics" to properly assess reasoning correctness. Combine automated evaluation with human validation across 1,853 test samples.

## Implementation Notes

Design benchmark tasks requiring multi-ability reasoning. Implement structured rubrics checking all sub-metrics simultaneously. Conduct human evaluation to calibrate automated metrics. Use findings to identify which reasoning abilities generative models possess or lack.

## References

- Original paper: MMGR (Dec 2025)
- Generative model evaluation frameworks
- Reasoning benchmarks for vision models
