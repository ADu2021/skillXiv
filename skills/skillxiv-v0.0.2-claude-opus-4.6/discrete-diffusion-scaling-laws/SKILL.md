---
name: discrete-diffusion-scaling-laws
title: "Scaling Behavior of Discrete Diffusion Language Models and SNR-Based Reformulation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.10858
keywords: [discrete-diffusion, scaling-laws, language-models, SNR, masked-vs-uniform-diffusion]
description: "Reformulate discrete diffusion using signal-to-noise ratio for hybrid masked-uniform noise scheduling. Derive compute-optimal scaling laws through careful hyperparameter tuning, showing uniform diffusion scales more favorably in token-constrained settings than autoregressive models."
---

## Skill Summary

This research presents three interconnected contributions: (1) an SNR-based diffusion framework enabling smooth interpolation between masked and uniform noise, (2) systematic scaling law methodology tuning batch size and learning rate across model scales, and (3) empirical evidence that uniform diffusion models scale more favorably in token-constrained training, requiring more parameters and less data for compute-efficient training compared to masked diffusion alternatives.

## When To Use

- Training discrete diffusion language models and needing to understand scaling dynamics
- Projects comparing masked vs. uniform diffusion for optimal compute allocation
- Scenarios where you need to predict performance of diffusion models at different scales
- Research exploring alternatives to autoregressive pretraining that handle token budgets differently

## When NOT To Use

- Inference-optimized scenarios (diffusion requires multiple denoising steps vs. single AR pass)
- Projects already deeply committed to autoregressive models without flexibility to pivot
- Domains where the scaling laws don't apply (limited data regimes, highly specialized tasks)
- Real-time generation requirements where AR models' single-pass inference is critical

## Core Technique

Three key contributions establish the framework:

**1. SNR-Based Diffusion Framework**
Reformulate Generalized Interpolating Discrete Diffusion (GIDD) using signal-to-noise ratio instead of time steps. This enables "smoothly interpolating between masked and uniform diffusion" by defining a hybrid mixing distribution that transitions based on SNR values, aligning discrete diffusion with continuous-state theory.

**2. Systematic Scaling Law Methodology**
Rather than fixing hyperparameters, carefully tune batch size and learning rate across different model scales. Use CompleteP parameterization for stable learning rate transfer, deriving compute-optimal scaling laws without learning rate annealing. Key finding: "optimal batch size depends primarily on the total number of training tokens" rather than model size or target loss.

**3. Empirical Scaling Results**
Through experiments from 25M to 10B parameters: uniform diffusion models scale more favorably in token-constrained settings, requiring "more parameters and less data for compute-efficient training compared to masked diffusion." Discovered scaling laws extrapolate accurately across 50× larger compute budgets than the fitting range.

## Implementation Notes

Use SNR-based parameterization for hybrid noise scheduling. Systematically tune batch size and learning rate as model scales increase. Compare uniform vs. masked diffusion variants on your compute budget to identify which scales better for your token allocation. Apply the derived scaling laws to predict performance across different compute budgets.

## References

- Original paper: Scaling Behavior of Discrete Diffusion LMs (Dec 2025)
- CompleteP parameterization for stable scaling
- Signal-to-noise ratio diffusion frameworks
