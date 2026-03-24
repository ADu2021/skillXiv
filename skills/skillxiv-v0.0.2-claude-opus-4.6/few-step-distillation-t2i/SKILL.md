---
name: few-step-distillation-t2i
title: "Few-Step Distillation for Text-to-Image Generation: Practical Guide and Unified Framework"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.13006
keywords: [distillation, text-to-image, consistency-models, fast-generation, few-step]
description: "Systematically adapt state-of-the-art distillation methods for T2I generation. Compare sCM (stabilized Consistency Models), MeanFlow, and IMM within unified framework. sCM excels at extreme few-step regimes (52.81% GenEval at 2 steps), MeanFlow achieves superior fidelity at 4 NFEs."
---

## Skill Summary

This work presents a systematic study adapting existing distillation techniques for text-to-image generation within a unified framework. It compares three primary methods—sCM (stabilized Consistency Models), MeanFlow, and IMM (Inductive Moment Matching)—applied to FLUX.1-lite. Theoretical insights establish relationships between methods, practical adaptations address T2I-specific challenges, and empirical results guide method selection based on step requirements.

## When To Use

- Building fast text-to-image systems requiring 2-4 generation steps
- Scenarios needing extreme speedup from full-step diffusion models
- Projects where generation latency is critical for user experience
- Research comparing different distillation approaches for image generation

## When NOT To Use

- Applications requiring maximum visual quality over inference speed
- Scenarios where the overhead of distillation training isn't justified by speedup needs
- Models already highly optimized for few-step inference
- Domains where more than 4 steps are necessary for acceptable quality

## Core Technique

Three primary distillation methods compared within unified framework:

**1. Stabilized Consistency Models (sCM)**
Excels at extreme few-step regimes, achieving 52.81% GenEval score at 2 steps. Practical T2I adaptations include timestep normalization for sCM to handle the different scale ranges between vision and language domains.

**2. MeanFlow**
Requires more steps but achieves superior fidelity at 4 NFEs (Network Function Evaluations). Employs dual-timestep mechanisms for T2I generation and maintains better quality with additional steps. Can be understood as a synchronization limit of sCM.

**3. Inductive Moment Matching (IMM)**
Provides a third approach with distinct theoretical properties. Theoretical framework shows Flow Matching is precisely recovered when the reference time equals the current time.

**4. Practical Adaptations**
- Timestep normalization for sCM across modality scales
- Dual-timestep mechanisms for MeanFlow
- Improved classifier-free guidance configurations for all methods

## Implementation Notes

Cast your distillation problem within the unified sCM/MeanFlow/IMM framework. Choose sCM for extreme 2-step regimes, MeanFlow for 4+ step quality, or IMM for balanced approaches. Implement T2I-specific adaptations: timestep normalization, dual-timestep mechanisms, and guidance tuning. Use modular code structure enabling easy comparison between methods.

## References

- Original paper: Few-Step Distillation for T2I (Dec 2025)
- Consistency Models for diffusion
- Flow Matching and MeanFlow frameworks
- FLUX.1 architecture details
