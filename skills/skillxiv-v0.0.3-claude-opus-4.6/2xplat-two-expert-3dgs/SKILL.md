---
name: 2xplat-two-expert-3dgs
title: "2Xplat: Two-Expert Framework for Pose-Free 3D Gaussian Splatting"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.21064"
keywords: [3D Gaussian Splatting, Pose Estimation, Modular Architecture, Geometry-Appearance Decomposition]
description: "Replace monolithic 3D Gaussian Splatting with two-expert architecture separating geometry (pose) estimation from appearance synthesis. Converges 30× faster (5K vs 150K iterations) while matching pose-dependent methods. Works best for multi-view reconstruction when geometry and appearance have conflicting optimization dynamics. Trigger: When doing feed-forward 3DGS and need faster convergence without sacrificing quality."
category: "Component Innovation"
---

## What This Skill Does

Swap a monolithic 3D Gaussian Splatting architecture with a modular two-expert framework that explicitly separates geometry estimation (camera pose prediction) from appearance synthesis (Gaussian generation). Achieves 30× faster convergence while maintaining quality comparable to state-of-the-art pose-dependent methods.

## Problem with Monolithic 3D Gaussian Splatting

Monolithic architectures jointly optimize all parameters (camera poses, Gaussian positions, colors, covariances) in a single network. This creates inherent conflicts:
- Strict geometric accuracy (tight pose constraints) can degrade visual appearance (fewer degrees of freedom for colors)
- Unified representations cannot leverage specialized architectures proven effective for each sub-task
- Large parameter space (poses + gaussians) requires extensive training (150K+ iterations on 16 GH200 GPUs)
- Dense architecture cannot reuse pretrained geometric or appearance priors

The paper's insight: Decompose into specialized experts. Geometry expert predicts precise camera poses; appearance expert (already trained on multi-view synthesis) generates Gaussians conditioned on known poses. This enables reuse of pretrained models and faster convergence.

## The Swap: Monolithic → Two-Expert Decomposition

Replace unified architecture with sequential expert pipeline:

```python
# Monolithic approach (baseline)
class MonolithicGaussianSplatter(nn.Module):
    """Single network jointly predicts poses + Gaussians"""
    def __init__(self):
        self.image_encoder = Encoder()
        self.pose_head = nn.Linear(feat_dim, 6)      # Predict 6-DoF poses
        self.gaussian_head = nn.Linear(feat_dim, 14*N_gaussians)  # Predict positions, colors, covariance

    def forward(self, images):
        features = self.image_encoder(images)
        poses = self.pose_head(features)       # Joint optimization
        gaussians = self.gaussian_head(features)
        return poses, gaussians

# Two-expert decomposition (proposed)
class TwoExpertGaussianSplatter(nn.Module):
    """Separate experts for geometry and appearance"""
    def __init__(self):
        # Geometry expert: specialized for pose estimation
        self.geometry_expert = DepthAnything3()  # Pretrained depth/pose model

        # Appearance expert: specialized for multi-view Gaussian synthesis
        self.appearance_expert = MultiViewPyramidTransformer()  # Pretrained multi-view synthesizer

    def forward(self, images):
        # Stage 1: Geometry expert predicts camera poses
        depth_maps = self.geometry_expert(images)
        poses = depth_to_poses(depth_maps)  # Extract poses from depth

        # Stage 2: Appearance expert generates Gaussians conditioned on estimated poses
        gaussians = self.appearance_expert(images, poses=poses)  # Pose-conditioned synthesis

        # Stage 3: End-to-end fine-tuning (optional, joint loss)
        return poses, gaussians

# Fine-tuning: joint loss allows appearance expert to adapt to noisy poses
def joint_loss(rendered, target, poses, poses_target):
    render_loss = l2_loss(rendered, target)
    pose_supervision = mse_loss(poses, poses_target)  # Regularize pose estimates
    return render_loss + lambda * pose_supervision
```

Key differences:
- Monolithic: One network, shared features, conflicting gradients
- Two-expert: Reuse pretrained Depth Anything 3 for geometry, pretrained multi-view model for appearance; fine-tune jointly

## Performance Impact

**Baseline (Monolithic, feed-forward):**
- Training iterations: 150,000 on 16 GH200 GPUs
- Convergence time: ~48 hours
- Quality: Baseline for comparison

**With Two-Expert Architecture:**
- Training iterations: < 5,000 on 8 H200 GPUs
- Speedup: 30× faster (150K → 5K iterations, 16×8 → 8×8 GPUs)
- Quality: Substantially outperforms prior pose-free methods; matches pose-dependent methods
- Cross-dataset generalization: Strong (DL3DV → ScanNet++) with minimal fine-tuning
- Robustness to view count: Scales from 6 to 128 input views

**Ablation (implicit):**
- Monolithic baseline: slow convergence due to conflicting gradients
- Geometry expert alone: accurate poses but no appearance
- Two-expert (frozen): decent performance, no adaptation
- Two-expert + fine-tuning: best results (poses + appearance jointly optimized)

## When to Use

- Feed-forward 3D Gaussian Splatting when pose is unknown (casual capture, internet images)
- Multi-view reconstruction on diverse datasets (10-100 views)
- When you have access to pretrained geometry (Depth Anything) and appearance experts (multi-view models)
- Large-scale 3D reconstruction where convergence speed matters (avoid training-from-scratch)
- Cross-dataset scenarios (pretrained experts generalize better than monolithic models)

## When NOT to Use

- Single-image 3D reconstruction (two-expert assumes multiple views)
- When you have ground truth poses (pose-dependent methods will be simpler)
- Highly constrained domains where monolithic end-to-end training is standard (e.g., synthetic data)
- If pretrained experts (Depth Anything, multi-view synthesizer) are unavailable or domain-specific
- Real-time applications needing absolute minimal latency (two stages add minor overhead)

## Implementation Checklist

To integrate two-expert architecture:

1. **Prepare geometry expert**:
   - Use Depth Anything 3 or comparable depth estimator
   - Input: multi-view images
   - Output: depth maps → camera poses (via SfM-like recovery or direct pose regression)

2. **Prepare appearance expert**:
   - Use Multi-View Pyramid Transformer or pretrained multi-view synthesis model
   - Input: images + camera intrinsics/extrinsics
   - Output: 3D Gaussian parameters (position, color, covariance)

3. **Implement sequential pipeline**:
   ```
   Multi-view images → Geometry expert → Pose estimates
                                            ↓
                                    Appearance expert + pose-conditioning
                                            ↓
                                      3D Gaussians
   ```

4. **Add joint fine-tuning** (optional but recommended):
   - Forward: render Gaussians, compare to input images
   - Loss: render loss + pose supervision (small weight ~0.01)
   - Fine-tune both experts jointly for 5K iterations

5. **Verify convergence and generalization**:
   - Baseline: train monolithic from scratch, measure time to convergence
   - New: train two-expert, target 30× faster
   - Cross-dataset: evaluate on different domain (e.g., test ScanNet++ after training on DL3DV)
   - Benchmark: PSNR/SSIM on standard multi-view datasets

6. **Optional tuning**:
   - Pose supervision weight: 0.001-0.1 (balance rendering vs pose accuracy)
   - Freezing strategies: which layers to fine-tune (try all, appearance-only, geometry-only)
   - View count: test on 6, 12, 24, 48, 128 views for scaling study

## Known Issues

- **Noisy pose estimates degrade appearance**: If geometry expert produces poor poses, fine-tuning may not fully recover. Monitor pose estimation accuracy early.
- **Incompatibility of pretrained models**: If geometry expert and appearance expert were trained differently (e.g., different camera models), fine-tuning reconciles but takes longer.
- **Mode collapse on few views**: With < 5 views, geometry expert may be ambiguous. Add pose uncertainty or regularization.
- **Generalization to narrow baselines**: If training data has wide baselines but test data is narrow (or vice versa), performance drops. Ensure training set matches test distribution.

## Related Work

Builds on 3D Gaussian Splatting (Kerbl et al.), Depth Anything (Yang et al.), and multi-view synthesis (Transformer-based approaches). Relates to prior modular 3D reconstruction methods (NeRF with pose networks) but applies to Gaussian representations. Distinct from monolithic end-to-end 3D learning by leveraging pretrained, specialized experts.
