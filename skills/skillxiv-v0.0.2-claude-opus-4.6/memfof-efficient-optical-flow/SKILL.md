---
name: memfof-efficient-optical-flow
title: "MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.23151"
keywords: [OpticalFlow, ComputerVision, MemoryEfficiency, VideoAnalysis, HighResolution]
description: "Reduces optical flow GPU memory 3.9× while maintaining state-of-the-art accuracy through correlation volume downsampling and dimension compensation. Enables native FullHD training with 2.09GB inference memory. Use for motion estimation in memory-constrained environments or high-resolution video processing."
---

# MEMFOF: Memory-Efficient Multi-Frame Optical Flow at Full Resolution

Optical flow estimation—computing pixel motion between frames—is fundamental to video understanding, but dense correlation volume computation requires enormous GPU memory at high resolutions. Standard methods like RAFT fail at 1080p or require aggressive downsampling that loses detail. MEMFOF solves this through three complementary mechanisms: reducing correlation volume resolution from 1/8 to 1/16 (60× memory savings), compensating with increased feature channel dimensions, and training on high-resolution upsampled data to capture large motions. The result: native FullHD training and 2.09GB inference memory with top-tier accuracy.

The insight is that correlation volume resolution doesn't need to match input resolution; feature dimension density can compensate. Additionally, high-resolution training data (2× upsampled) prevents underfitting and properly captures motion distributions that low-resolution training misses.

## Core Concept

MEMFOF extends the RAFT family with three key memory optimizations:

1. **Correlation Volume Downsampling**: Reduces from 1/8 to 1/16 resolution, decreasing memory from 10.4GB to 0.65GB for multi-frame setup while maintaining motion tracking accuracy.

2. **Dimensional Compensation**: Increases feature dimensions (encoder output Df: 256→1024, correlation Dc: 128→512) to preserve representational capacity despite lower correlation volume resolution.

3. **High-Resolution Training Strategy**: Uses 2× upsampled training frames to align motion distributions with FullHD inference, preventing the distribution mismatch that causes standard models to underfit at high resolutions.

The framework extends SEA-RAFT to process three consecutive frames, predicting bidirectional flows simultaneously while reusing computations across overlapping frame pairs.

## Architecture Overview

- **Three-Frame Processing**: Predicts current-to-previous and current-to-next flows in single forward pass
- **Feature Extraction**: Enhanced encoder with increased dimensions (256→1024 base features)
- **Downsampled Correlation Volume**: 1/16 input resolution instead of 1/8, with increased channel dimensions (128→512)
- **GMA Attention Module**: Geometric-matching attention enhancing match quality at reduced resolution
- **Computation Reuse**: Shares feature maps across overlapping frame pairs, reducing redundant computation
- **FullHD Training Data**: 2× upsampled frames capturing large motion properly

## Implementation

Memory-efficient correlation volume computation with dimensional compensation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryEfficientCorrelationVolume(nn.Module):
    """
    Computes correlation volume at 1/16 resolution with increased channel
    dimensions to compensate for spatial downsampling.
    Memory usage: 0.65GB for 3 frames vs. 10.4GB standard.
    """
    def __init__(self, feature_dim=1024, correlation_dim=512, downsampling=16):
        super().__init__()
        self.feature_dim = feature_dim
        self.correlation_dim = correlation_dim
        self.downsampling = downsampling

        # Project features to correlation space (higher dimension = compensation)
        self.context_proj = nn.Conv2d(feature_dim, correlation_dim, 1)
        self.feature_proj = nn.Conv2d(feature_dim, correlation_dim, 1)

    def forward(self, context_features, feature_maps):
        """
        Compute dense correlation volume at low resolution.

        Args:
            context_features: (B, H/16, W/16, C) reference frame features
            feature_maps: (B, H/16, W/16, C) current frame features

        Returns:
            correlation_volume: (B, H/16*W/16, H/16, W/16) at 1/16 resolution
        """
        batch_size, height, width, _ = context_features.shape

        # Project to correlation dimension (increased from 128 → 512)
        context_proj = self.context_proj(context_features.permute(0, 3, 1, 2))
        feature_proj = self.feature_proj(feature_maps.permute(0, 3, 1, 2))

        # Reshape for correlation: (B, C, H*W) and (B, C, H, W)
        context_flat = context_proj.reshape(batch_size, self.correlation_dim, -1)
        features_spatial = feature_proj

        # Correlation: matrix multiply between all positions
        # (B, H*W, C) @ (B, C, H, W) → (B, H*W, H, W)
        correlation = torch.einsum('bch,bchw->bhw', context_flat, features_spatial)

        # Normalize correlation
        correlation = correlation / (self.correlation_dim ** 0.5)

        return correlation


class EnhancedFeatureExtractor(nn.Module):
    """
    Encoder with increased feature dimensions for compensation.
    Standard: 256 dim; MEMFOF: 1024 dim at matching memory cost through efficient design.
    """
    def __init__(self, in_channels=3, output_dim=1024, downsampling_factor=16):
        super().__init__()
        self.downsampling_factor = downsampling_factor

        # Efficient multi-scale feature extraction
        self.conv1 = self._make_layer(in_channels, 64, stride=1)
        self.conv2 = self._make_layer(64, 128, stride=2)  # 1/2
        self.conv3 = self._make_layer(128, 256, stride=2)  # 1/4
        self.conv4 = self._make_layer(256, 512, stride=2)  # 1/8
        self.conv5 = self._make_layer(512, output_dim, stride=2)  # 1/16

    def _make_layer(self, in_channels, out_channels, stride=1):
        """Residual layer with efficient computation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, image):
        """Extract features at 1/16 resolution."""
        feat1 = self.conv1(image)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)  # Output: 1/16 resolution, 1024 channels
        return feat5


class MultiFrameOpticalFlowNetwork(nn.Module):
    """
    Three-frame optical flow with correlation volume reuse for efficiency.
    Processes frame0, frame1, frame2 computing flows 0→1 and 1→2.
    """
    def __init__(self, feature_dim=1024, correlation_dim=512):
        super().__init__()
        self.feature_extractor = EnhancedFeatureExtractor(output_dim=feature_dim)
        self.correlation_volume = MemoryEfficientCorrelationVolume(
            feature_dim, correlation_dim
        )

        # GMA attention for enhanced matching at low resolution
        self.gma_attention = GeometricMatchingAttention(correlation_dim)

        # Flow refinement
        self.flow_refiner = FlowRefiner(correlation_dim + feature_dim)

    def forward(self, frame0, frame1, frame2):
        """
        Compute bidirectional flows: frame0→frame1 and frame1→frame2.
        Reuses frame1 features to reduce computation.

        Args:
            frame0, frame1, frame2: (B, 3, H, W) consecutive frames

        Returns:
            flow_01: (B, 2, H, W) optical flow from frame0 to frame1
            flow_12: (B, 2, H, W) optical flow from frame1 to frame2
        """
        # Extract features once per frame (1/16 resolution)
        feat0 = self.feature_extractor(frame0)  # (B, 1024, H/16, W/16)
        feat1 = self.feature_extractor(frame1)
        feat2 = self.feature_extractor(frame2)

        # Compute flow 0→1
        # Correlation: search frame0 features within frame1 window
        corr_01 = self.correlation_volume(feat0, feat1)  # (B, H/16*W/16, H/16, W/16)

        # GMA attention enhances matching at reduced resolution
        corr_01_refined = self.gma_attention(corr_01, feat1)

        # Flow refinement: iterative improvement
        flow_01 = self.flow_refiner(corr_01_refined, feat1)

        # Compute flow 1→2 (reuses feat1)
        corr_12 = self.correlation_volume(feat1, feat2)
        corr_12_refined = self.gma_attention(corr_12, feat2)
        flow_12 = self.flow_refiner(corr_12_refined, feat2)

        return flow_01, flow_12
```

Training strategy using high-resolution upsampled data:

```python
class HighResolutionTrainingStrategy:
    """
    Training on 2× upsampled frames prevents underfitting at high-resolution.
    Motion distributions: standard 256×256 data has smaller motion than
    FullHD 1080p data upsampled 2×.
    """
    def __init__(self, base_dataset, upsample_factor=2):
        self.base_dataset = base_dataset
        self.upsample_factor = upsample_factor

    def get_training_sample(self, idx):
        """Load sample and upsample to full resolution."""
        sample = self.base_dataset[idx]

        # Upsample frames to higher resolution
        frame0 = F.interpolate(
            sample['frame0'],
            scale_factor=self.upsample_factor,
            mode='bilinear',
            align_corners=False
        )
        frame1 = F.interpolate(
            sample['frame1'],
            scale_factor=self.upsample_factor,
            mode='bilinear',
            align_corners=False
        )

        # Scale optical flow accordingly (double resolution → double motion magnitude)
        flow = sample['flow'] * self.upsample_factor

        return {
            'frame0': frame0,
            'frame1': frame1,
            'flow': flow
        }


def train_memfof_high_resolution(model, train_loader, num_epochs=100):
    """
    Training loop on 2× upsampled high-resolution data.
    Ensures motion distribution matches FullHD inference.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            frame0 = batch['frame0'].cuda()
            frame1 = batch['frame1'].cuda()
            frame2 = batch['frame2'].cuda()
            flow_gt_01 = batch['flow_01'].cuda()
            flow_gt_12 = batch['flow_12'].cuda()

            with torch.cuda.amp.autocast():
                # Forward pass: predict bidirectional flows
                flow_01_pred, flow_12_pred = model(frame0, frame1, frame2)

                # Loss: photometric + smoothness
                loss_01 = photometric_loss(flow_01_pred, frame0, frame1) + \
                          smoothness_loss(flow_01_pred)
                loss_12 = photometric_loss(flow_12_pred, frame1, frame2) + \
                          smoothness_loss(flow_12_pred)

                total_loss = loss_01 + loss_12

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {total_loss:.4f}")


class GeometricMatchingAttention(nn.Module):
    """GMA attention: improves matching quality at low correlation resolution."""
    def __init__(self, dim=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)

    def forward(self, correlation, features):
        """Apply attention to correlation volume."""
        # Simplified: reshape and apply attention
        b, hw, h, w = correlation.shape
        corr_flat = correlation.reshape(b, hw, -1)
        attn_out, _ = self.attention(corr_flat, corr_flat, corr_flat)
        return attn_out.reshape(b, hw, h, w)


class FlowRefiner(nn.Module):
    """Iterative flow refinement module."""
    def __init__(self, input_dim=1536):
        super().__init__()
        self.refine_conv = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 3, padding=1)
        )

    def forward(self, correlation, features):
        """Refine flow estimate."""
        combined = torch.cat([correlation, features], dim=1)
        flow = self.refine_conv(combined)
        return flow
```

## Practical Guidance

| Aspect | Value | Notes |
|--------|-------|-------|
| Inference Memory (1080p) | 2.09 GB | Versus 8.2 GB standard RAFT |
| Training Memory (FullHD) | 28.5 GB | With 2× upsampling |
| Memory Reduction Factor | 3.9× | Compared to SEA-RAFT baseline |
| Correlation Volume Resolution | 1/16 input | Versus 1/8 standard |
| Feature Dimension Compensation | 256→1024 | Offset spatial reduction |
| Training Data Strategy | 2× upsampling | Matches FullHD motion distribution |
| Accuracy Maintained | Yes | State-of-the-art on benchmarks |

**When to use:**
- Processing 1080p or higher resolution video on memory-constrained GPUs
- Training on high-resolution datasets (FullHD video collections)
- Deploying optical flow in embedded or edge environments
- Video analysis requiring motion estimation at native resolution
- Applications with strict memory budgets (2-4GB GPU)

**When NOT to use:**
- If you have unlimited GPU memory (direct approach simpler)
- Very low-resolution video where standard methods already fit
- Real-time processing with strict latency requirements (multi-frame processing adds latency)
- Scenarios where 1/16 resolution correlation is insufficient (very fine motion)
- Applications where you cannot afford 2× upsampled training (data augmentation cost)

**Common pitfalls:**
- Inadequate feature dimension increase (must match spatial reduction)
- Forgetting to 2× upsample training data (causes distribution mismatch)
- Correlation resolution too low, losing fine motion details
- Not accounting for flow magnitude doubling with 2× upsampling
- GMA attention threshold tuned on low-resolution data then applied directly
- Skipping multi-frame reuse optimization (defeats memory savings)

## Reference

"MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation", 2025. [arxiv.org/abs/2506.23151](https://arxiv.org/abs/2506.23151)
