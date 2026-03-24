---
name: nemotron-math-long-context
title: "Nemotron-Math: Large-Scale Mathematical Reasoning Dataset with Multi-Mode Supervision and Sequential Bucketing"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.15489
keywords: [mathematical-reasoning, long-context, dataset, reasoning-modes, training-efficiency]
description: "Create 7.5M long-form mathematical solution traces with multi-mode supervision (high/medium/low reasoning depths, with/without Python). Integrate 85K competition problems (AoPS) and 262K community questions (StackExchange). Implement sequential bucketing training achieving 2-3× speedup while maintaining accuracy."
---

## Skill Summary

Nemotron-Math introduces a large-scale mathematical reasoning dataset combining diverse problem sources with multi-mode supervision. The dataset contains "7.5M long-form solution traces produced by gpt-oss-120b under three distinct reasoning modes (high, medium, and low), both with and without Python TIR." The approach integrates 85,000 curated competition problems (AoPS) with 262,000 community-driven questions (StackExchange-Math), balancing structured rigor with diverse real-world mathematical queries. A sequential bucketed training strategy groups samples by sequence length and trains progressively from 16K to 128K tokens, achieving "2–3× faster training while maintaining accuracy within 1–3% of full-length joint training."

## When To Use

- Training models on complex mathematical reasoning
- Projects requiring diverse problem sources (competition + community)
- Scenarios with long sequences benefiting from bucketed training
- Research on efficient long-context reasoning model training

## When NOT To Use

- Simple arithmetic or short-form math tasks
- Domains not involving long-form reasoning or multi-step derivation
- Scenarios where single-mode supervision suffices
- Models already trained on large math corpora

## Core Technique

Three key components enable effective mathematical reasoning training:

**1. Multi-Mode Data Generation**
Leverage controllable reasoning depths and tool-integrated reasoning (Python execution) to capture diverse solution styles:
- High reasoning mode: detailed step-by-step derivation
- Medium reasoning mode: balanced explanation and conciseness
- Low reasoning mode: streamlined solution
- With/without Python TIR: symbolic computation options

This flexibility captures the diverse ways mathematicians solve problems.

**2. Hybrid Problem Sourcing**
Integrate two complementary sources:
- 85K AoPS (Art of Problem Solving) problems: curated, competition-quality
- 262K StackExchange-Math questions: community-driven, diverse difficulty

Balancing structured rigor with real-world problem distribution.

**3. Sequential Bucketed Training Strategy**
Group samples by sequence length and train progressively from 16K → 128K tokens. This strategy achieves "2–3× faster training while maintaining accuracy within 1–3% of full-length joint training." Enables efficient scaling to ultra-long sequences without uniform padding overhead.

## Implementation Notes

Collect or access multi-mode supervision dataset with varying reasoning depths. Integrate problem sources spanning competition and community. Implement sequential bucketing: group training examples by sequence length, train stage-by-stage from shorter to longer. Monitor accuracy maintenance as sequence length increases. Use learned representations from shorter stages to initialize longer stages.

## References

- Original paper: Nemotron-Math (Dec 2025)
- Long-context language model training
- Mathematical reasoning datasets
