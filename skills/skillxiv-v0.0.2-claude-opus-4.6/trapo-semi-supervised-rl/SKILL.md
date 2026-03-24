---
name: trapo-semi-supervised-rl
title: "TraPO: Trajectory-based Policy Optimization for Semi-supervised Reasoning with Limited Labels"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.13106
keywords: [reinforcement-learning, semi-supervised, reasoning, trajectory-alignment, label-efficient]
description: "Bridge labeled and unlabeled data through trajectory similarity in reinforcement learning. Select reliable unlabeled samples by comparing pass-rate evolution trajectories against labeled data. Achieve 42.6% accuracy with 1K labeled + 3K unlabeled samples, surpassing fully-supervised training on 45K labels with 10% annotation budget."
---

## Skill Summary

TraPO introduces Trajectory-based Policy Optimization, a semi-supervised RL framework addressing limitations in unsupervised approaches. Rather than concatenating supervised and unsupervised objectives, the method measures "pass rate trajectory similarity" between unlabeled and labeled samples across training epochs. This selects reliable unlabeled instances for policy optimization, achieving significant data efficiency gains—42.6% accuracy with 1K labeled + 3K unlabeled samples surpasses fully-supervised training on 45K labels.

## When To Use

- Training reasoning models with limited labeled data
- Scenarios where unlabeled data is abundant but labels are expensive
- Projects where unsupervised RL leads to mode collapse or divergence
- Research on efficient label use in reinforcement learning

## When NOT To Use

- Domains with abundant labeled data where simple supervised learning suffices
- Tasks where reliable trajectory patterns don't emerge during training
- Applications requiring immediate performance without trajectory evolution analysis
- Scenarios where the computational overhead of trajectory tracking is prohibitive

## Core Technique

The core breakthrough leverages trajectory-level insights rather than step-level labels:

**1. Pass Rate Trajectory Measurement**
Track the evolution of correctness rates for each question throughout training epochs. Represent this as a trajectory showing how model performance on that sample changes over time.

**2. Trajectory Similarity Analysis**
Compare an unlabeled sample's trajectory against the average trajectory from labeled data using cosine similarity. This identifies unlabeled samples following learning patterns similar to labeled examples.

**3. Trajectory-Aligned Selection**
Select only trajectory-aligned unlabeled instances for policy optimization. This moves "from what the model learns to how it learns," using the "pass rate change trajectory as a medium" connecting heterogeneous solution spaces.

**4. Empirical Results**
With 1K labeled + 3K unlabeled: 42.6% accuracy. With 4K labeled: surpasses fully-supervised training on 45K labels using only 10% of annotation budget. Demonstrates trajectory alignment captures crucial learning dynamics beyond individual correctness labels.

## Implementation Notes

Implement trajectory tracking for each sample: record pass/fail status at each training epoch. Compute average trajectory from labeled data. For each unlabeled sample, compute cosine similarity between its trajectory and labeled average. Include only trajectory-aligned samples in RL policy optimization. Monitor total accuracy and annotation efficiency.

## References

- Original paper: TraPO (Dec 2025)
- Semi-supervised learning for reasoning
- Policy optimization with limited supervision
