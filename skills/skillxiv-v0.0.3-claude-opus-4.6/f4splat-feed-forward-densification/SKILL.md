---
name: f4splat-feed-forward-densification
title: "F4Splat: Feed-Forward Predictive Densification for 3D Gaussian Splatting"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.21304
keywords: [3DGS, Gaussian Splatting, Densification, Feed-Forward, Adaptive Allocation]
description: "Enable spatially adaptive Gaussian allocation in 3D Gaussian Splatting without iterative optimization. Reduces primitive count by 70-90% while maintaining reconstruction quality through learned densification score prediction."
---

## Component ID
Predictive Densification Module for Feed-Forward 3DGS

## Motivation
Traditional feed-forward 3D Gaussian Splatting uniformly distributes Gaussians across pixels or voxels, creating redundancy in simple regions and insufficient representation in complex areas. Each scene requires many primitives to maintain quality because allocation is not adaptive.

## The Modification
F4Splat replaces fixed pixel-to-Gaussian allocation with a learned densification score predictor that enables spatially adaptive allocation at inference time:

1. **Densification Score Prediction Network** - During training, the model observes per-region homodirectional view-space positional gradients (which indicate where additional Gaussians would help during optimization). These supervise a learned densification score (computed as log-scaled L2 norm of gradients). At inference, the score is available without per-scene optimization.

2. **Multi-Scale Gaussian Maps** - Instead of single-resolution allocation, the architecture predicts multi-scale parameter maps across three pyramid levels, enabling flexible representation levels per spatial region.

3. **Adaptive Thresholding Mechanism** - A region-wise selector chooses representation levels for each spatial location based on predicted densification scores, ensuring non-overlapping allocations across pyramid levels.

## Ablation Results
The paper demonstrates:
- With 24-29% of baseline Gaussians, maintains competitive LPIPS/PSNR metrics on novel-view synthesis
- Achieves "on-par or superior" quality compared to standard approaches while using significantly fewer primitives
- 70-90% reduction in Gaussian count is achievable while preserving reconstruction quality
- Performance is consistent across multi-view and two-view settings on RE10K and ACID datasets

## Conditions
- Applicable to feed-forward 3D Gaussian Splatting pipelines where inference-time optimization is not performed
- Requires training data with sufficient geometric complexity to learn meaningful densification scores
- Post-hoc budget adjustment is possible—the learned scores can be thresholded at different levels without retraining

## Drop-In Checklist
- [ ] Integrate densification score prediction into the main encoding network
- [ ] Add multi-scale parameter map prediction heads (three pyramid levels)
- [ ] Implement adaptive region-wise thresholding selector
- [ ] During training: supervise scores with backpropagated rendering gradients
- [ ] At inference: apply learned scores for spatially adaptive allocation
- [ ] Validate Gaussian count reduction (expect 70-90% fewer primitives)
- [ ] Benchmark novel-view synthesis quality (LPIPS, PSNR) against baseline
