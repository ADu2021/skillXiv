---
name: rl-awb-nighttime-white-balance
title: "RL-AWB: Deep Reinforcement Learning for Auto White Balance Correction in Low-Light Night-time Scenes"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05249"
keywords: [Reinforcement Learning, Computational Photography, Low-Light Processing, White Balance]
description: "Correct color distortion in nighttime photos by combining statistical gray-pixel detection with reinforcement learning parameter optimization. Achieves superior cross-camera generalization without extensive labeled nighttime training data through a hybrid architecture that preserves interpretability while gaining adaptive tuning capability."
---

## When to Use This Skill
- Processing nighttime or low-light photography with unknown camera sensors
- Applications requiring cross-camera generalization without retraining
- Real-world white balance correction where sensor-specific optimization is impractical
- Scenarios where interpretable statistical methods provide baseline reliability

## When NOT to Use This Skill
- Well-lit image processing (standard AWB algorithms suffice)
- Applications with access to extensive labeled nighttime datasets and GPU capacity
- Scenarios requiring sub-millisecond inference (RL adds latency)

## Problem Summary
Traditional automatic white balance (AWB) assumes sufficient scene diversity and reliable gray pixel detection, which fail under extreme low-light conditions where sensor noise dominates signal. Deep learning approaches require extensive labeled nighttime data and suffer catastrophic generalization loss across different camera sensors. Existing methods either sacrifice interpretability or generalization capability.

## Solution: Hybrid SGP-LRD + RL Framework

Combine an interpretable statistical algorithm with learned parameter optimization via Soft Actor-Critic reinforcement learning.

```python
# SGP-LRD Algorithm: Statistical Gray Pixel with Local Reflectance
class SGPLRDWBCorrector:
    def __init__(self, N_percentile=35, p_norm=4):
        self.N = N_percentile  # Gray pixel percentage
        self.p = p_norm        # Minkowski norm parameter

    def detect_salient_gray_pixels(self, image):
        """Two-stage filtering: local variance + color deviation"""
        # Stage 1: Local contrast analysis
        local_variance = compute_local_variance(image)
        # Stage 2: Color deviation from gray
        color_deviation = measure_chroma_deviation(image)
        return apply_confidence_weighting(local_variance, color_deviation)

    def estimate_illuminant(self, gray_pixels):
        """Apply local reflectance normalization"""
        norm_pixels = gray_pixels ** (1/self.p)
        illuminant = compute_avg_illuminant(norm_pixels)
        return illuminant
```

**RL Refinement Layer (Soft Actor-Critic):**
- **State**: Log-chrominance histograms (60×60×3) + adjustment history (11 dims)
- **Actions**: Continuous relative adjustments to N and p parameters
- **Reward**: Relative angular error improvement with difficulty-aware scaling
- **Architecture**: Dual-branch MLPs processing histograms and history separately

## Key Implementation Details

**Training Configuration:**
- Soft Actor-Critic with twin Q-heads
- Batch size: 256
- Discount factor γ: 0.99
- Target update rate τ: 0.005
- Learning rate: 3×10⁻⁴
- 150,000 timesteps over 16 parallel environments
- Hardware: NVIDIA RTX 3080 GPU for image processing

**Curriculum Learning (Two Stages):**
1. **Single-image stabilization**: Establish baseline convergence behavior
2. **Cyclic multi-image tuning**: Optimal pool M=5 images with 5 consecutive episodes per image

**State Architecture Rationale:**
Use dual-branch processing instead of concatenation: one branch for high-dimensional histogram (10,800 dims), another for low-dimensional adjustment history (11 dims). Direct concatenation drowns adjustment signals in histogram noise.

## Empirical Performance

**In-Domain Accuracy (NCC Dataset):**
- RL-AWB: 1.98° median angular error vs. 2.12° for pure SGP-LRD
- +5.9% improvement through per-image parameter tuning

**Cross-Sensor Generalization (LEVI Dataset):**
- Maintains 3.01°-3.03° median error when training crosses sensors (NCC→LEVI)
- Baseline deep learning methods degrade dramatically (2.46°→9.40° error)
- Only method achieving "sensor-agnostic robustness"

**Cross-Domain Daytime Performance:**
- Achieves 2.24° error on daytime Gehler-Shi dataset vs. 2.38° for pure SGP-LRD
- Proves applicability beyond target nighttime domain

## Dataset Contribution
LEVI dataset: 700 RAW nighttime images across iPhone 16 Pro and Sony ILCE-6400 cameras at ISOs 500-16,000. Enables rigorous cross-sensor evaluation critical for real-world mobile and professional camera deployment.

## Advantages Over Pure Approaches
- **vs. Statistical Only**: Adaptive per-image parameter tuning outperforms fixed dataset optimization
- **vs. Deep Learning Only**: Hybrid architecture requires <1% training data while maintaining cross-sensor generalization
- **Interpretability**: Statistical foundation preserves explainability for quality assurance and debugging
