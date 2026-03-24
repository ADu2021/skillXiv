---
name: adversarial-flow-models
title: "Adversarial Flow Models: Optimal Transport Regularization for One-Step Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2511.22475
keywords: [generative-models, flow-matching, adversarial-training, optimal-transport, one-step-generation]
description: "Combines adversarial training with optimal transport constraints enabling deterministic, transport-optimal one-step image generation with FID 2.38 on ImageNet-256. Apply when you need fast generative models without teacher-student distillation complexity."
---

## Summary

Adversarial Flow Models unifies adversarial training with flow-based generative models by adding an optimal transport regularization term that constrains generators to learn transport-optimal mappings. This enables native one-step generation without distillation, stabilizes adversarial training on transformers, and achieves FID 2.38 on ImageNet-256 with single-step generation.

## Core Technique

The key innovation adds an optimal transport regularization to the adversarial objective. While adversarial training learns distribution matching through min-max optimization, it doesn't guarantee learning a deterministic, Wasserstein-2 optimal transport plan like flow matching does.

**Optimal Transport Loss Term:** ℒ_ot^G = E_z[1/n ||G(z) - z||²_2] constrains generator outputs to match the transport plan induced by random Gaussian noise. This ensures G(z) satisfies optimal transport properties—straightness and determinism—even under adversarial training.

**Combined Objective:** Minimize: ℒ_adv^G + λ * ℒ_ot^G where ℒ_adv^G is the standard adversarial generator loss and λ balances both objectives.

## Implementation

**Regularization weight selection:** Start with λ = 1.0 and adjust based on FID convergence. Too high prioritizes transport optimality over distribution matching; too low loses benefits of the constraint.

**Noise handling:** Sample z ~ N(0, I) and pass through generator G(z). Compute both the adversarial loss (discriminator feedback) and the L2 distance ||G(z) - z||², averaging over the batch.

**One-step generation:** Train for native one-step inference without teacher-student distillation. Sample z once and evaluate G(z) directly—no iterative denoising.

**Transformer stability:** The optimal transport constraint stabilizes transformer-based generators compared to pure adversarial training, reducing training divergence and gradient scaling issues.

## When to Use

- Real-time image generation where latency is critical
- Applications needing diverse generation without distillation overhead
- Scenarios where straightforward adversarial training diverges on transformer architectures
- Tasks requiring deterministic, principled generation mappings

## When NOT to Use

- Scenarios where teacher-distilled multi-step models are acceptable
- Applications where discriminator training overhead is prohibitive
- Tasks needing iterative refinement or coarse-to-fine generation
- Models where likelihood-based training is preferred over adversarial objectives

## Key References

- Optimal transport theory and Wasserstein distances in generative modeling
- Flow matching and matching networks for generative tasks
- Adversarial training stabilization techniques for transformers
