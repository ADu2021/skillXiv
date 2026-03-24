---
name: puzzle-curriculum-grpo
title: "Puzzle Curriculum: Vision Language Model Post-training via Self-Supervised Puzzle Tasks and Exploration-Aware Curriculum"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.14944
keywords: [vision-language, reinforcement-learning, curriculum-learning, self-supervised, puzzle-tasks]
description: "Post-train vision-language models using automatically-verifiable puzzle environments (Jigsaw, Rotation, PatchFit) with graded rewards. Implement exploration-aware curriculum combining difficulty weighting with solution-space diversity metrics. Track reasoning-answer consistency to prevent divergence during training."
---

## Skill Summary

PuzzleCraft presents three main technical contributions for vision-language model post-training. Puzzle-based RLVR uses three automatically-verifiable environments inspired by pretext tasks, with Jigsaw notably using graded partial-credit rewards. An exploration-aware curriculum combines difficulty weighting with entropy/permutation diversity metrics to prevent solution-space collapse. Reasoning-Answer Consistency (RAC) monitoring tracks whether reasoning traces support answers, addressing the critical observation that consistency can degrade during training.

## When To Use

- Post-training vision-language models for improved reasoning
- Scenarios with automatically-verifiable reward signals
- Projects requiring exploration-aware curriculum learning
- Research on self-supervised reasoning for vision-language models

## When NOT To Use

- Domains without naturally verifiable puzzle structures
- Applications where simple supervised finetuning already works well
- Scenarios with limited computational resources for extensive post-training
- Tasks where exploration-diversity metrics don't apply

## Core Technique

Three key innovations enable effective vision-language post-training:

**1. Puzzle-Based RLVR Framework**
Introduce three automatically-verifiable puzzle environments:
- Jigsaw: Tile-placement with graded rewards based on correct tile placement
- Rotation: Image rotation prediction and correction
- PatchFit: Patch fitting and alignment tasks

Jigsaw notably uses partial-credit rewards addressing sparsity problem of binary rewards, enabling learning from partial progress.

**2. Exploration-Aware Curriculum**
Combine difficulty weighting with exploration signal: "detection of when rollouts within a prompt group collapse to the same solution" prevents vanishing-advantage dynamics. Curriculum downweights groups showing solution-space collapse using:
- Entropy: for binary puzzles
- Permutation diversity: for Jigsaw puzzles

This balances difficulty with exploration to maintain learning signal.

**3. Reasoning-Answer Consistency (RAC) Monitoring**
Track whether model reasoning traces actually support final answers during training. Address observation that "reasoning–answer inconsistency" can emerge as post-training progresses despite improving puzzle rewards. Use RAC as diagnostic metric guiding training.

## Implementation Notes

Design or adopt puzzle environments with automatic verification. Implement graded reward signals for partial progress. Track solution entropy/diversity per prompt group. Downweight groups showing collapse. Monitor RAC throughout training. Adjust curriculum based on both task difficulty and solution diversity.

## References

- Original paper: Puzzle Curriculum GRPO (Dec 2025)
- Vision-language model post-training
- Curriculum learning and exploration strategies
