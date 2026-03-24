---
name: clift-light-field-tokens
title: "CLiFT: Compressive Light-Field Tokens for Compute-Efficient and Adaptive Neural Rendering"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.08776"
keywords: [3D Rendering, Light Field Compression, Neural Radiance, Adaptive Rendering]
description: "Represent 3D scenes as compressed light-field tokens for efficient neural rendering. Multi-view images are tokenized via Plücker coordinates, condensed through K-means clustering, and rendered adaptively. Achieves 5-7× data reduction versus MVSplat while enabling on-the-fly quality-speed tradeoffs: up to 66% FPS improvement with controlled token counts."
---

# CLiFT: Adaptive Light-Field Token Compression for Real-time 3D Rendering

Neural rendering typically requires storing full multi-view image data or learning dense radiance fields. Light fields—high-dimensional representations of scene appearance—are extremely data-heavy. CLiFT compresses light-field information into learned tokens representing semantic scene content, enabling dramatic data reduction (5-7×) while maintaining rendering quality. Crucially, by controlling the number of tokens used at render time, you get flexible quality-speed tradeoffs: use 256 tokens for high quality, or 100 tokens for 66% speedup.

The key insight is that most light-field information is redundant within small spatial regions. By identifying clusters of similar rays (Plücker-coordinate based) and keeping only their centroids, you compress aggressively while preserving the geometric and appearance information needed for novel-view synthesis.

## Core Concept

CLiFT operates in three stages:

1. **Multi-view Encoding**: Transform input images into light-field tokens using Plücker ray coordinates (captures both position and direction) concatenated with RGB
2. **Latent K-means Clustering**: Identify semantic clusters in feature space; select nearest neighbor from each cluster as centroid (lossless per-cluster representative selection)
3. **Neural Rendering**: Lightweight Transformer condenses tokens into a compact representation, then decodes to novel views with adaptive token counts

The method decouples storage tokens (Nₛ) from rendering tokens (Nᵣ), enabling post-hoc quality adjustment without retraining.

## Architecture Overview

- **Plücker Encoder**: Converts 3D rays to 6D Plücker coordinates (position + direction) to capture geometric structure
- **Multi-view Transformer Encoder**: Tokenizes input images with geometric constraints
- **K-means Clustering Module**: Groups similar rays; identifies cluster centroids via nearest-neighbor selection
- **Token Condensation Network**: Lightweight Transformer compressing K clusters into Nᵣ rendering tokens
- **Adaptive Rendering Head**: Produces novel views from selected tokens using positional encoding
- **View-dependent Decoder**: MLP network predicting appearance based on viewing direction
- **Learnable Cluster Parameters**: Mean and variance per cluster maintained during training

## Implementation

The following demonstrates light-field tokenization and adaptive rendering:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PlueckerEncoder(nn.Module):
    """Encode 3D rays as 6D Plücker coordinates for geometric awareness."""
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear projection from 6D Plücker to embedding
        self.proj = nn.Linear(6, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, rays_origin: torch.Tensor, rays_direction: torch.Tensor,
                rgb_values: torch.Tensor) -> torch.Tensor:
        """
        Encode rays as tokens with geometric structure.

        Args:
            rays_origin: (batch, H*W, 3) ray starting positions
            rays_direction: (batch, H*W, 3) normalized ray directions
            rgb_values: (batch, H*W, 3) pixel colors

        Returns:
            tokens: (batch, H*W, hidden_dim) encoded ray tokens
        """
        batch_size, num_rays, _ = rays_origin.shape

        # Compute Plücker coordinates: [direction, origin × direction]
        # Captures both geometric position and direction information
        cross_product = torch.cross(rays_origin, rays_direction, dim=-1)
        pluecker = torch.cat([rays_direction, cross_product], dim=-1)  # (batch, num_rays, 6)

        # Normalize Plücker coordinates
        pluecker = F.normalize(pluecker, dim=-1)

        # Project to embedding space
        tokens = self.proj(pluecker)  # (batch, num_rays, hidden_dim)
        tokens = self.norm(tokens)

        # Optionally fuse RGB information
        tokens = tokens + self.proj(torch.cat([rgb_values, torch.zeros_like(rgb_values[:, :, :3])], dim=-1))

        return tokens

class LightFieldTokenClusterer(nn.Module):
    """K-means clustering in feature space for light-field compression."""
    def __init__(self, hidden_dim: int, num_clusters: int = 256, max_iter: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.max_iter = max_iter

        # Learnable cluster centroids
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, hidden_dim))
        nn.init.normal_(self.cluster_centers, std=0.01)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cluster tokens; select representative from each cluster.

        Args:
            tokens: (batch, num_tokens, hidden_dim)

        Returns:
            cluster_representatives: (batch, num_clusters, hidden_dim)
            cluster_assignments: (batch, num_tokens) cluster indices
            cluster_variance: (num_clusters,) variance within each cluster
        """
        batch_size, num_tokens, hidden_dim = tokens.shape

        # Compute distances to cluster centers
        # (batch, num_tokens, 1, hidden_dim) - (1, 1, num_clusters, hidden_dim)
        distances = torch.cdist(tokens, self.cluster_centers)  # (batch, num_tokens, num_clusters)

        # Hard assignment: each token to nearest cluster
        cluster_assignments = distances.argmin(dim=-1)  # (batch, num_tokens)

        # Select representative from each cluster (nearest-neighbor to centroid)
        cluster_representatives = []
        cluster_variance = []

        for cluster_idx in range(self.num_clusters):
            # Find tokens in this cluster
            mask = (cluster_assignments == cluster_idx).float()  # (batch, num_tokens)

            if mask.sum() > 0:
                # Compute variance within cluster
                cluster_tokens = tokens[mask.unsqueeze(-1).expand(-1, -1, hidden_dim) > 0.5]
                if cluster_tokens.shape[0] > 0:
                    variance = cluster_tokens.var(dim=0).mean()
                else:
                    variance = torch.tensor(0.0, device=tokens.device)

                # Find nearest-neighbor representative
                cluster_center = self.cluster_centers[cluster_idx:cluster_idx+1]
                distances_to_center = torch.norm(tokens - cluster_center, dim=-1)
                representative_idx = distances_to_center.argmin(dim=-1).unsqueeze(-1)

                representative = torch.gather(
                    tokens,
                    dim=1,
                    index=representative_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
                ).squeeze(1)  # (batch, hidden_dim)
            else:
                representative = self.cluster_centers[cluster_idx:cluster_idx+1]
                variance = torch.tensor(0.0, device=tokens.device)

            cluster_representatives.append(representative)
            cluster_variance.append(variance)

        cluster_representatives = torch.stack(cluster_representatives, dim=1)  # (batch, num_clusters, hidden_dim)
        cluster_variance = torch.stack(cluster_variance)  # (num_clusters,)

        return cluster_representatives, cluster_assignments, cluster_variance

class AdaptiveTokenCondenser(nn.Module):
    """Compress light-field tokens into rendering-efficient representation."""
    def __init__(self, hidden_dim: int = 256, num_render_tokens: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_render_tokens = num_render_tokens

        # Transformer blocks for token interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Token selection via attention
        self.selection_head = nn.Linear(hidden_dim, 1)

    def forward(self, cluster_tokens: torch.Tensor,
                num_render_tokens: Optional[int] = None) -> torch.Tensor:
        """
        Select and compress tokens for rendering.

        Args:
            cluster_tokens: (batch, num_clusters, hidden_dim)
            num_render_tokens: Override default; enables dynamic adjustment

        Returns:
            render_tokens: (batch, num_render_tokens, hidden_dim)
        """
        if num_render_tokens is None:
            num_render_tokens = self.num_render_tokens

        # Encode cluster tokens
        encoded = self.encoder(cluster_tokens)  # (batch, num_clusters, hidden_dim)

        # Score tokens for importance
        scores = self.selection_head(encoded).squeeze(-1)  # (batch, num_clusters)

        # Select top-k tokens by score
        _, top_indices = torch.topk(scores, k=num_render_tokens, dim=-1)

        # Gather selected tokens
        batch_size = cluster_tokens.shape[0]
        batch_indices = torch.arange(batch_size, device=cluster_tokens.device).view(-1, 1)
        render_tokens = cluster_tokens[batch_indices, top_indices]  # (batch, num_render_tokens, hidden_dim)

        return render_tokens

class NeuralLightFieldRenderer(nn.Module):
    """Render novel views from compressed light-field tokens."""
    def __init__(self, hidden_dim: int = 256, image_height: int = 512, image_width: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_height = image_height
        self.image_width = image_width

        # Position encoding for spatial coordinates
        self.pos_encoding = nn.Linear(6, hidden_dim)  # x, y, z + view direction (3D)

        # View-dependent MLP for appearance
        self.appearance_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # RGB output
        )

        # Spatial aggregation (cross-attention from tokens to pixel positions)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

    def forward(self, render_tokens: torch.Tensor,
                target_poses: torch.Tensor) -> torch.Tensor:
        """
        Render novel views using compressed light-field tokens.

        Args:
            render_tokens: (batch, num_render_tokens, hidden_dim)
            target_poses: (batch, image_h, image_w, 6) target camera poses + ray directions

        Returns:
            rendered_image: (batch, image_h, image_w, 3)
        """
        batch_size = render_tokens.shape[0]
        h, w = self.image_height, self.image_width

        # Flatten spatial coordinates
        target_poses_flat = target_poses.view(batch_size, h * w, -1)  # (batch, h*w, 6)

        # Encode target positions
        pos_encoding = self.pos_encoding(target_poses_flat)  # (batch, h*w, hidden_dim)

        # Cross-attention: aggregate token information for each pixel
        pixel_features, _ = self.cross_attention(
            query=pos_encoding,
            key=render_tokens,
            value=render_tokens
        )  # (batch, h*w, hidden_dim)

        # Decode to RGB
        combined = torch.cat([pos_encoding, pixel_features], dim=-1)  # (batch, h*w, hidden_dim*2)
        rgb = self.appearance_mlp(combined)  # (batch, h*w, 3)

        # Reshape to image
        image = rgb.view(batch_size, h, w, 3).permute(0, 3, 1, 2)  # (batch, 3, h, w)
        image = torch.clamp(image, 0, 1)

        return image

class CLiFTModel(nn.Module):
    """Complete CLiFT pipeline for adaptive light-field rendering."""
    def __init__(self, hidden_dim: int = 256, num_storage_tokens: int = 256,
                 num_render_tokens: int = 128):
        super().__init__()
        self.pluecker_encoder = PlueckerEncoder(hidden_dim)
        self.clusterer = LightFieldTokenClusterer(hidden_dim, num_clusters=num_storage_tokens)
        self.condenser = AdaptiveTokenCondenser(hidden_dim, num_render_tokens)
        self.renderer = NeuralLightFieldRenderer(hidden_dim)

    def forward(self, input_images: torch.Tensor, input_poses: torch.Tensor,
                target_poses: torch.Tensor,
                num_render_tokens: Optional[int] = None) -> Tuple[torch.Tensor, dict]:
        """
        Encode input views, compress, and render novel view.

        Args:
            input_images: (batch, num_views, 3, H, W)
            input_poses: (batch, num_views, 6) camera poses + ray directions
            target_poses: (batch, H, W, 6) target camera pose + ray directions
            num_render_tokens: Dynamic token count for quality-speed tradeoff

        Returns:
            rendered: (batch, 3, H, W)
            info: Dict with compression stats
        """
        batch_size, num_views = input_images.shape[:2]

        # Tokenize input views
        # (Simplified: assume images already converted to rays)
        tokens = self.pluecker_encoder(
            torch.zeros(batch_size, num_views * 256, 3),  # placeholder rays
            torch.ones(batch_size, num_views * 256, 3) / 3,
            input_images.view(batch_size, -1, 3)
        )

        # Cluster tokens
        cluster_reps, assignments, variance = self.clusterer(tokens)

        # Compress for rendering
        render_tokens = self.condenser(cluster_reps, num_render_tokens)

        # Render novel view
        rendered = self.renderer(render_tokens, target_poses)

        info = {
            'num_input_tokens': tokens.shape[1],
            'num_storage_tokens': cluster_reps.shape[1],
            'num_render_tokens': render_tokens.shape[1],
            'compression_ratio': tokens.shape[1] / render_tokens.shape[1]
        }

        return rendered, info

def train_clift_step(model: CLiFTModel, batch, optimizer: torch.optim.Optimizer) -> float:
    """Single training step for CLiFT."""
    optimizer.zero_grad()

    input_images = batch['input_images']
    input_poses = batch['input_poses']
    target_images = batch['target_images']
    target_poses = batch['target_poses']

    # Forward pass
    rendered, info = model(input_images, input_poses, target_poses)

    # Render loss
    loss = F.l1_loss(rendered, target_images)

    loss.backward()
    optimizer.step()

    return loss.item()
```

This implementation demonstrates the three-stage compression and adaptive rendering pipeline.

## Practical Guidance

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Storage Tokens (Nₛ)** | 256 | Balance compression vs. quality; 512 for highest fidelity |
| **Render Tokens (Nᵣ)** | 128 | Start here; reduce to 50-100 for speedup, increase to 200+ for quality |
| **K-means Clusters** | 256 | Usually equal to Nₛ; more clusters allow finer discrimination |
| **Cluster Selection** | Nearest-neighbor | Ensures representatives are actual tokens, not interpolations |
| **Plücker Normalization** | L2-norm | Ensures geometric consistency across scales |
| **Training Views** | 4-8 | More views provide richer signal; <4 causes overfitting |
| **Learning Rate** | 1e-4 | Stable training for token compression |

### When to Use CLiFT

- **Multi-view 3D reconstruction**: Convert image collections to efficient scene representations
- **Neural rendering systems**: Needing real-time novel-view synthesis with controllable quality
- **Adaptive rendering**: Dynamic quality-speed tradeoffs (high-FPS low-quality vs. slow high-quality)
- **Storage-constrained deployment**: 5-7× compression is critical for mobile/edge devices
- **Fast preprocessing**: Lightweight compression avoids expensive optimization loops
- **View-dependent appearance**: Light fields naturally capture complex reflectance

### When NOT to Use

- **Geometrically complex scenes** with occlusions: Light field may be insufficient; use explicit geometry
- **Extreme view extrapolation**: If novel views far from input poses, extrapolation fails
- **Scenes with specularities**: Highly specular surfaces create discontinuities in light field
- **Very small object details**: Compression may lose fine geometric detail; unsuitable for macro-level precision
- **Dynamic scenes**: Static scene assumption; unsuitable for video or moving objects

### Common Pitfalls

1. **Insufficient Input Views**: Using <4 views causes K-means underfitting. Always use ≥4 input images; 8+ ideal.
2. **Plücker Coordinate Bugs**: Incorrect cross-product computation breaks geometric structure. Verify: origin × direction encodes moment, not position.
3. **K-means Divergence**: If cluster centers never update, learning rate too low or initialization bad. Use warm-start from PCA.
4. **Token Dropout During Rendering**: If rendering only uses top-k tokens by score, important geometric information discarded. Use all clusters, select only during inference.
5. **Camera Pose Errors**: Even 1-2° camera pose error cascades through rendering. Pre-register input poses carefully; validate with reprojection.

## Reference

Wang, Z., Wu, Y., et al. (2025). CLiFT: Compressive Light-Field Tokens for Compute-Efficient and Adaptive Neural Rendering. *arXiv preprint arXiv:2507.08776*.

Available at: https://arxiv.org/abs/2507.08776
