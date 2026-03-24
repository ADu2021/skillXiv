---
name: seedvr2-video-restoration
title: "SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.05301"
keywords: [video-restoration, diffusion-models, adversarial-training, efficient-inference, high-resolution]
description: "Achieves single-step video restoration at 1080p resolution with 4x speedup over multi-step diffusion approaches via adversarial training, adaptive window attention, and feature matching loss."
---

# SeedVR2: One-Step Video Restoration

## Core Concept

SeedVR2 tackles computational inefficiency in diffusion-based video restoration by eliminating multi-step iterative refinement. Rather than requiring 64+ denoising steps, the model completes restoration in a single forward pass while maintaining or improving quality over iterative methods. This is achieved through adversarial training against real data, learnable attention mechanisms that adapt to input resolution, and efficient loss functions optimized for high-resolution processing.

## Architecture Overview

- **Diffusion Transformer Base**: Swin-MMDIT architecture with 16B total parameters (generator + discriminator)
- **Adaptive Window Attention**: Dynamic spatial window sizing that scales to arbitrary resolutions while maintaining computational efficiency
- **Causal Video VAE**: Temporal compression for efficient processing of multi-frame sequences
- **Feature Matching Loss**: Efficient alternative to LPIPS for adversarial training on high-resolution video without costly pixel-space decoding
- **Progressive Distillation**: Training from 64-step teacher model with gradual temporal length increase for stability
- **RpGAN Stabilization**: Approximate R2 regularization to maintain adversarial training stability across thousands of iterations

## Implementation

The following code illustrates the core architectural components:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AdaptiveWindowAttention(nn.Module):
    """
    Adaptive window sizing for efficient high-resolution video processing.
    """
    def __init__(self, dim: int, window_base_size: int = 7):
        super().__init__()
        self.dim = dim
        self.window_base_size = window_base_size

    def compute_adaptive_window(self, h: int, w: int) -> Tuple[int, int]:
        """
        Compute window size based on input resolution.
        Maintains constant computation while scaling to arbitrary resolutions.
        """
        # Base window for 512x512: 7x7
        scale_h = h / 512.0
        scale_w = w / 512.0

        adaptive_h = int(self.window_base_size * (scale_h ** 0.5))
        adaptive_w = int(self.window_base_size * (scale_w ** 0.5))

        # Ensure odd window size
        adaptive_h = adaptive_h if adaptive_h % 2 == 1 else adaptive_h + 1
        adaptive_w = adaptive_w if adaptive_w % 2 == 1 else adaptive_w + 1

        return adaptive_h, adaptive_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive window attention with computed window size.
        x: (B, T, C, H, W) video tensor
        """
        B, T, C, H, W = x.shape
        win_h, win_w = self.compute_adaptive_window(H, W)

        # Reshape for windowed attention
        x_flat = x.view(B * T, C, H, W)

        # Apply local window attention (simplified)
        x_out = F.avg_pool2d(x_flat, kernel_size=max(H // win_h, 1))
        x_out = F.interpolate(x_out, size=(H, W), mode='bilinear')

        return x_out.view(B, T, C, H, W)


class FeatureMatchingLoss(nn.Module):
    """
    Efficient alternative to LPIPS for adversarial training.
    Compares feature maps rather than pixel values.
    """
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, generated: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """
        Compute feature-level matching loss.
        generated, real: (B, C, H, W) video frames
        """
        gen_features = self.feature_extractor(generated)
        real_features = self.feature_extractor(real)

        # Compute L2 distance between feature maps
        loss = F.mse_loss(gen_features, real_features)

        return loss


class SeedVR2Trainer:
    def __init__(self, generator: nn.Module, discriminator: nn.Module,
                 learning_rate: float = 1e-4):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
        self.dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        self.feature_loss = FeatureMatchingLoss(self._build_feature_extractor())

    def _build_feature_extractor(self) -> nn.Module:
        """Build lightweight feature extractor for loss computation."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def training_step(self, degraded_video: torch.Tensor,
                     real_video: torch.Tensor) -> Tuple[float, float]:
        """
        Single training step with adversarial loss.
        """
        # Generator step
        restored = self.generator(degraded_video)
        dis_fake = self.discriminator(restored)

        gen_loss = -dis_fake.mean()  # Adversarial loss
        gen_loss += 10.0 * self.feature_loss(restored, real_video)  # Feature matching

        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        # Discriminator step with R2 regularization
        dis_real = self.discriminator(real_video)
        dis_fake = self.discriminator(restored.detach())

        dis_loss = F.softplus(dis_fake).mean() + F.softplus(-dis_real).mean()

        # Approximate R2 regularization
        r2_reg = 0.01 * (dis_real ** 2).mean()
        dis_loss = dis_loss + r2_reg

        self.dis_optimizer.zero_grad()
        dis_loss.backward()
        self.dis_optimizer.step()

        return float(gen_loss), float(dis_loss)
```

## Practical Guidance

**Progressive Distillation Schedule**: Begin training with a 64-step teacher model, then gradually compress to single-step inference. Use KL divergence to maintain distribution alignment during distillation.

**Temporal Length Curriculum**: Start training with 4-8 frame clips, then progressively increase to 16-32 frames. This prevents training instability in early stages.

**Resolution Proxy Technique**: Train at 720p but maintain consistency at arbitrary test resolutions by using a spatial proxy that preserves aspect ratios during inference.

**Adversarial Training Stability**: The RpGAN loss with R2 regularization is critical for preventing discriminator collapse. Monitor the discriminator loss; if it drops below 0.1 during training, increase regularization weight.

**Feature Loss Coefficient**: Weight the feature matching loss 5-15x higher than the adversarial loss. This balances perceptual quality with adversarial realism.

**Batch Size and Data**: Use approximately 15M image-video pairs for comprehensive training. Batch size of 16-32 on A100 GPUs provides good convergence.

## Reference

SeedVR2 achieves exceptional efficiency gains:
- **Speed**: 4× faster than existing diffusion-based video restoration
- **Quality**: Comparable or superior to multi-step methods in single forward pass
- **Resolution**: Handles 1080p restoration with faithful detail preservation

The method is particularly valuable for real-time video enhancement applications where computational budgets are constrained but quality standards are high.
