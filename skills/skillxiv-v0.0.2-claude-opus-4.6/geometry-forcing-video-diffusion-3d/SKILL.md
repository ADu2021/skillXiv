---
name: geometry-forcing-video-diffusion-3d
title: "Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07982"
keywords: [Video Diffusion, 3D Consistency, Spatial Coherence, Temporal Stability, Flow Matching]
description: "Improve video diffusion consistency by aligning intermediate diffusion features with 3D geometric representations from pretrained foundation models, enabling spatially coherent and temporally stable video generation through angular and scale alignment losses."
---

# Geometry Forcing: Anchoring Video Diffusion in 3D Structure

Standard video diffusion models generate pixel sequences without geometric constraints. They overlook a fundamental truth: videos capture 2D projections of dynamic 3D worlds. This causes artifacts—objects warp unrealistically, camera motion becomes incoherent, temporal consistency breaks down. Geometry Forcing addresses this by constraining video diffusion features to align with explicit 3D geometric representations.

The method extracts geometric features from a pretrained 3D foundation model (VGGT), then optimizes video diffusion to match these 3D constraints through dual alignment losses: angular alignment (direction preservation) and scale alignment (magnitude preservation). The result is video generation with improved spatial consistency and realistic temporal evolution.

## Core Concept

The key insight is that intermediate features of a video diffusion model should embed 3D geometric understanding. Rather than training from scratch, leverage pretrained 3D foundation models as geometric supervisors. By aligning diffusion features with 3D representations, the model learns to generate videos respecting 3D structure: cameras follow coherent paths, objects maintain consistent geometry, and scenes evolve realistically.

Two alignment mechanisms ensure this: (1) angular alignment preserves feature directions (relative relationships), and (2) scale alignment preserves magnitudes (absolute scales). Together, they ground video diffusion in 3D geometry while respecting the diffusion training dynamics.

## Architecture Overview

- **Video Diffusion Backbone**: Flow Matching with autoregressive transformer, generates frame sequences
- **3D Foundation Model**: Visual Geometry Grounded Transformer (VGGT), provides geometric supervision
- **Angular Alignment**: Cosine similarity loss between diffusion and geometric features
- **Scale Alignment**: MSE loss on normalized geometric feature prediction
- **Lightweight Projectors**: Feature transformation heads for alignment
- **Dual Loss Weighting**: λ_Angular and λ_Scale balance geometric constraints vs diffusion quality
- **Inference Reconstruction**: Generate 3D geometry during video generation for 4D understanding

## Implementation

### Step 1: Extract 3D Geometric Features from Foundation Model

Use pretrained 3D models (VGGT) to extract geometric supervision for video sequences:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class GeometricFeatureExtractor:
    """
    Extract 3D geometric representations from pretrained foundation models.
    Uses Visual Geometry Grounded Transformer (VGGT) for supervision.
    """
    def __init__(self, model_name: str = "vggt-base"):
        self.model = torch.hub.load('pretrained_models', model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_geometric_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract 3D geometric features from video frames.
        frames: [T, 3, H, W] video tensor
        Returns: [T, H, W, D] geometric feature maps
        """
        geometric_features = []

        with torch.no_grad():
            for frame in frames:
                # Extract geometric embeddings (depth, normals, etc.)
                frame_expanded = frame.unsqueeze(0)
                geom_feat = self.model.encode_geometry(frame_expanded)
                geometric_features.append(geom_feat.squeeze(0))

        return torch.stack(geometric_features, dim=0)

    def extract_camera_pose(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract estimated camera poses from frame sequence.
        Returns: [T, 4, 4] camera extrinsic matrices
        """
        poses = []

        with torch.no_grad():
            for frame in frames:
                frame_expanded = frame.unsqueeze(0)
                pose = self.model.estimate_camera_pose(frame_expanded)
                poses.append(pose.squeeze(0))

        return torch.stack(poses, dim=0)

# Example usage
extractor = GeometricFeatureExtractor()
video_frames = torch.randn(16, 3, 256, 256)  # 16-frame video
geometric_features = extractor.extract_geometric_features(video_frames)
camera_poses = extractor.extract_camera_pose(video_frames)
```

### Step 2: Implement Angular and Scale Alignment Losses

Design loss functions that align diffusion features with 3D geometric representations:

```python
def angular_alignment_loss(diffusion_features: torch.Tensor,
                          geometric_features: torch.Tensor) -> torch.Tensor:
    """
    Angular alignment: preserve direction of features (relative structure).
    Uses cosine similarity to measure alignment without scale dependence.

    diffusion_features: [B, T, H, W, D] from video diffusion model
    geometric_features: [B, T, H, W, D] from 3D foundation model
    """
    # Normalize features to unit vectors (remove scale)
    diff_norm = F.normalize(diffusion_features, p=2, dim=-1)
    geom_norm = F.normalize(geometric_features, p=2, dim=-1)

    # Cosine similarity (dot product of normalized vectors)
    cos_sim = (diff_norm * geom_norm).sum(dim=-1)  # [B, T, H, W]

    # Loss: maximize similarity (minimize negative similarity)
    loss = -cos_sim.mean()

    return loss

def scale_alignment_loss(diffusion_features: torch.Tensor,
                        geometric_features: torch.Tensor,
                        scale_predictor: nn.Module) -> torch.Tensor:
    """
    Scale alignment: predict geometric feature magnitudes from normalized diffusion features.
    Ensures absolute scales match between domains.

    scale_predictor: MLP that maps normalized diffusion features to geometric scales
    """
    # Normalize diffusion features
    diff_norm = F.normalize(diffusion_features, p=2, dim=-1)

    # Compute geometric feature scales (magnitudes)
    geom_scales = torch.norm(geometric_features, p=2, dim=-1, keepdim=True)

    # Predict scales from normalized diffusion features
    batch_size, t, h, w, d = diff_norm.shape
    diff_flat = diff_norm.reshape(-1, d)
    scale_pred = scale_predictor(diff_flat).reshape(batch_size, t, h, w, 1)

    # MSE loss on scale prediction
    loss = F.mse_loss(scale_pred, geom_scales)

    return loss

class ScalePredictor(nn.Module):
    """MLP for predicting geometric scales from diffusion features."""
    def __init__(self, feat_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
```

### Step 3: Integrate Alignment Losses into Video Diffusion Training

Modify the video diffusion training loop to include geometric alignment:

```python
from diffusers import DDIMScheduler, FlowMatchEulerScheduler

class GeometryAwareDiffusion(nn.Module):
    """Video diffusion model with geometry forcing."""

    def __init__(self, base_diffusion_model: nn.Module,
                 scale_predictor: nn.Module,
                 lambda_angular: float = 0.5,
                 lambda_scale: float = 0.05):
        super().__init__()
        self.diffusion_model = base_diffusion_model
        self.scale_predictor = scale_predictor
        self.lambda_angular = lambda_angular
        self.lambda_scale = lambda_scale
        self.geometric_extractor = GeometricFeatureExtractor()

    def forward(self, video_frames: torch.Tensor,
               timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with geometry forcing losses.
        Returns: denoised frames, combined loss
        """
        batch_size, t_frames, c, h, w = video_frames.shape

        # Standard diffusion: denoise frames
        noise = torch.randn_like(video_frames)
        noisy_frames = self.scheduler.add_noise(
            video_frames, noise, timesteps
        )

        # Diffusion model forward pass
        diffusion_output = self.diffusion_model(
            noisy_frames, timesteps
        )

        # Base diffusion loss (predict noise)
        diffusion_loss = F.mse_loss(diffusion_output, noise)

        # Extract intermediate features from diffusion model
        # (typically from hidden layers of transformer)
        diff_features = self.diffusion_model.get_intermediate_features(
            noisy_frames, timesteps
        )  # [B, T, H, W, D]

        # Extract geometric supervision
        geometric_features = self.geometric_extractor.extract_geometric_features(
            video_frames
        )  # [B, T, H, W, D]

        # Compute alignment losses
        angular_loss = angular_alignment_loss(diff_features, geometric_features)
        scale_loss = scale_alignment_loss(
            diff_features, geometric_features, self.scale_predictor
        )

        # Combined loss
        total_loss = (
            diffusion_loss +
            self.lambda_angular * angular_loss +
            self.lambda_scale * scale_loss
        )

        return diffusion_output, total_loss

def train_geometry_aware_diffusion(model: GeometryAwareDiffusion,
                                  train_dataset,
                                  num_epochs: int = 10,
                                  lr: float = 8e-6):
    """Train video diffusion with geometry forcing."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataset) * num_epochs
    )

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataset):
            video_frames = batch["video"]  # [B, T, 3, H, W]
            timesteps = torch.randint(0, 1000, (video_frames.shape[0],))

            # Forward pass
            _, loss = model(video_frames, timesteps)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {batch_idx}: loss = {loss.item():.4f}")

    return model
```

### Step 4: Inference with 3D Reconstruction

Generate videos while reconstructing 3D geometry, enabling 4D understanding:

```python
def generate_video_with_geometry(model: GeometryAwareDiffusion,
                                prompt: str,
                                num_frames: int = 16,
                                height: int = 256,
                                width: int = 224,
                                num_inference_steps: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate video and simultaneously reconstruct 3D geometry.
    Returns: (video_frames, geometric_features)
    """
    # Initialize random noise
    noise = torch.randn(1, num_frames, 3, height, width)

    # Encode prompt to conditioning
    prompt_embeddings = model.diffusion_model.encode_prompt(prompt)

    # Denoising loop (reverse diffusion process)
    scheduler = FlowMatchEulerScheduler()
    scheduler.set_timesteps(num_inference_steps)

    generated_video = noise
    all_geometric_features = []

    for t in scheduler.timesteps:
        # Predict noise
        with torch.no_grad():
            noise_pred = model.diffusion_model(
                generated_video,
                t,
                encoder_hidden_states=prompt_embeddings
            )

        # Denoising step
        generated_video = scheduler.step(
            noise_pred, t, generated_video
        ).prev_sample

        # Extract geometric features at this step
        with torch.no_grad():
            diff_features = model.diffusion_model.get_intermediate_features(
                generated_video, t
            )
            all_geometric_features.append(diff_features)

    # Reconstruct 3D geometry from final features
    final_features = all_geometric_features[-1]
    reconstructed_geometry = model.geometric_extractor.reconstruct_3d(
        final_features
    )

    return generated_video, reconstructed_geometry

def evaluate_temporal_consistency(video_frames: torch.Tensor,
                                 geometric_features: torch.Tensor) -> Dict:
    """Evaluate video quality via geometric consistency metrics."""
    metrics = {}

    # Reprojection Error: how well does 3D geometry reproject to 2D?
    reprojection_error = compute_reprojection_error(
        video_frames, geometric_features
    )
    metrics["reprojection_error"] = reprojection_error

    # Revisit Error: camera revisits same 3D point, should see similar projection
    revisit_error = compute_revisit_error(geometric_features)
    metrics["revisit_error"] = revisit_error

    # Optical Flow Consistency: flows should match 3D motion
    flow_consistency = compute_flow_consistency(video_frames, geometric_features)
    metrics["flow_consistency"] = flow_consistency

    return metrics

def compute_reprojection_error(video_frames, geometric_features) -> float:
    """Compute how well 3D geometry projects to 2D video."""
    # Use DROID-SLAM or similar to validate 3D -> 2D consistency
    pass

def compute_revisit_error(geometric_features) -> float:
    """Compute error when camera revisits 3D points."""
    pass

def compute_flow_consistency(video_frames, geometric_features) -> float:
    """Verify optical flow matches 3D motion."""
    pass
```

## Practical Guidance

| Component | Recommended Value | Notes |
|---|---|---|
| λ_Angular | 0.5 | Weight for angular alignment loss |
| λ_Scale | 0.05 | Weight for scale alignment loss (lower than angular) |
| Base Learning Rate | 8×10⁻⁶ | Conservative for stable training |
| Frame Resolution | 256×256 (RealEstate10K), 384×224 (Minecraft) | Dataset-specific |
| Batch Size RealEstate10K | 8 | Smaller due to high resolution |
| Batch Size Minecraft | 32 | Larger for synthetic data |
| Video Length | 16 frames (RealEstate10K), 32 frames (Minecraft) | Varies by dataset |
| GPU Setup | 8 NVIDIA A100 GPUs | For efficient training |
| Inference Steps | 50 | Balance quality vs speed |
| Scale Predictor Hidden | 256 | Small network for feature transformation |

**When to use Geometry Forcing:**
- Video generation requiring spatial coherence and realism
- Scenarios where camera motion consistency matters
- Object-centric generation (avoiding warp artifacts)
- Scene understanding tasks needing 3D-aware features
- Applications requiring 4D (video + 3D) understanding
- Synthetic video generation (Minecraft-style environments)

**When NOT to use Geometry Forcing:**
- Artistic/stylized video where geometric realism is secondary
- Real-time generation (adds geometric supervision overhead)
- Extremely fast inference (geometry extraction adds latency)
- Highly abstract content without clear 3D structure
- Computational budget extremely limited (requires 3D supervisor model)

**Common pitfalls:**
- λ_Angular too high (> 1.0), over-constraining diffusion dynamics
- λ_Scale too high (> 0.1), learning to fit scales over generating quality frames
- Not normalizing features before angular alignment, conflating direction and magnitude
- 3D foundation model misaligned with video domain (using ImageNet features)
- Forgetting to freeze geometric extractor, adding unnecessary parameters
- Not validating geometric reconstructions match video content
- Camera motion estimation unreliable for dynamic scenes (moving objects)

## Reference

Li, Z., Song, X., Chen, J., & Zhou, B. (2025). Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling. arXiv:2507.07982. https://arxiv.org/abs/2507.07982
