---
name: longvie-multimodal-video-generation
title: LongVie - Multimodal-Guided Controllable Ultra-Long Video Generation
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03694
keywords: [video-generation, multimodal-control, diffusion, long-form-video]
description: "Generate consistent ultra-long videos (up to one minute) using depth and trajectory controls with autoregressive segment generation and unified noise initialization."
---

## LongVie: Multimodal-Guided Ultra-Long Video Generation

LongVie enables controlled generation of ultra-long videos (up to 60 seconds) by combining autoregressive short-clip synthesis with sophisticated multi-modal control. The breakthrough is maintaining temporal coherence across many segments through unified noise initialization and global control normalization, while accepting spatial and motion guidance from both dense (depth) and sparse (trajectory) control signals.

### Core Concept

Traditional video diffusion models struggle with two constraints: (1) context windows limit sequence length, and (2) controlling both visual appearance and motion simultaneously requires careful signal balancing. LongVie solves these by:

1. **Autoregressive Generation**: Generate short video clips sequentially, using each clip's final frame to initialize the next, avoiding context explosion
2. **Dual-Modal Control**: Process depth maps (dense, per-pixel guidance) and point trajectories (sparse, semantic motion) through separate control branches, then fuse them
3. **Temporal Consistency Mechanisms**: Apply identical noise across segments and normalize control signals globally to prevent quality degradation and motion jitter

### Architecture Overview

- **Base Model**: CogVideoX diffusion backbone extended with ControlNet-style conditioning
- **Multi-Modal Control Branches**: Separate processing paths for depth maps and trajectories, with additive fusion into DiT blocks
- **Control Injection**: Frozen pre-trained blocks with trainable control sub-branches and zero-initialized layers, preserving base model knowledge
- **Unified Noise Initialization**: Apply same noise across all video segments for seamless temporal continuity
- **Global Normalization**: Compute depth/trajectory scale factors across entire video sequence, not per-segment, ensuring consistent control interpretation
- **Degradation-Aware Training**: Adaptive weighting between control modalities to prevent dense signals from overwhelming sparse ones

### Implementation Steps

**Step 1: Prepare Multi-Modal Control Signals**

Extract or synthesize depth maps and motion trajectories for your video:

```python
import numpy as np
import cv2

def extract_depth_control(video_frames, depth_model):
    """
    Extract dense depth maps from video frames for spatial grounding.
    Depth maps provide per-pixel guidance for scene structure.
    """
    depth_maps = []
    for frame in video_frames:
        # Use pre-trained depth estimator (e.g., MiDaS)
        depth = depth_model(frame)  # Shape: (H, W)
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        depth_maps.append(depth_normalized)

    return np.stack(depth_maps)  # Shape: (T, H, W)

def extract_trajectory_control(video_frames, trajectory_detector):
    """
    Extract sparse point trajectories for motion guidance.
    Tracks semantic keypoints (e.g., person joints, object corners) across frames.
    """
    trajectories = []

    for t, frame in enumerate(video_frames):
        keypoints = trajectory_detector.detect(frame)  # List of (x, y) positions
        trajectories.append(keypoints)

    # Convert to trajectory format: list of tracks, each with (frame_idx, x, y)
    all_trajectories = []
    for keypoint_id in range(len(trajectories[0])):
        track = [(t, trajectories[t][keypoint_id][0], trajectories[t][keypoint_id][1])
                 for t in range(len(trajectories))]
        all_trajectories.append(track)

    return all_trajectories

# Usage
depth_maps = extract_depth_control(frames, depth_estimator)
trajectories = extract_trajectory_control(frames, trajectory_detector)
```

**Step 2: Global Control Signal Normalization**

Compute normalization bounds across the entire video sequence to ensure consistent interpretation:

```python
def compute_global_control_bounds(depth_maps, trajectories, percentile=2):
    """
    Compute global normalization bounds using percentile-based thresholding.
    This prevents per-segment normalization from causing inconsistent depth/motion scales.
    """
    # Depth normalization: use global min/max across all segments
    all_depths = np.concatenate(depth_maps)
    depth_min = np.percentile(all_depths, percentile)
    depth_max = np.percentile(all_depths, 100 - percentile)
    depth_scale = depth_max - depth_min

    # Trajectory normalization: compute velocity bounds globally
    all_velocities = []
    for track in trajectories:
        for i in range(1, len(track)):
            prev_x, prev_y = track[i-1][1], track[i-1][2]
            curr_x, curr_y = track[i][1], track[i][2]
            velocity = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            all_velocities.append(velocity)

    velocity_max = np.percentile(all_velocities, 100 - percentile)

    return {
        'depth_min': depth_min,
        'depth_max': depth_max,
        'depth_scale': depth_scale,
        'velocity_max': velocity_max
    }

bounds = compute_global_control_bounds(depth_maps, trajectories)
```

**Step 3: Unified Noise Initialization**

Generate a single noise schedule and apply it identically across all video segments:

```python
def create_unified_noise_schedule(num_segments, segment_length, latent_shape, seed=42):
    """
    Create unified noise that will be applied to all segments.
    Same noise across segments ensures seamless temporal transitions.
    """
    np.random.seed(seed)

    # Noise for initial segment
    initial_noise = np.random.randn(*latent_shape)

    # Noise schedule (e.g., linear schedule for diffusion steps)
    noise_schedule = np.linspace(1.0, 0.0, num_diffusion_steps=50)

    # For subsequent segments, blend noise to maintain continuity at boundaries
    segment_noises = [initial_noise]
    for seg_idx in range(1, num_segments):
        # Create new noise but blend with previous segment's final frame
        new_noise = np.random.randn(*latent_shape)
        blend_ratio = 0.3  # 30% previous, 70% new
        blended = (1 - blend_ratio) * new_noise + blend_ratio * segment_noises[-1]
        segment_noises.append(blended)

    return segment_noises, noise_schedule

noises, schedule = create_unified_noise_schedule(
    num_segments=10,
    segment_length=48,  # frames
    latent_shape=(4, 8, 8)  # typical latent size
)
```

**Step 4: Implement Dual-Branch Control Fusion**

Create separate control pathways for depth and trajectory, then fuse additively:

```python
import torch
import torch.nn as nn

class MultiModalControlFusion(nn.Module):
    """
    Process depth (dense) and trajectory (sparse) controls separately,
    then fuse into diffusion model through ControlNet-style injection.
    """
    def __init__(self, hidden_dim=768):
        super().__init__()

        # Depth processing branch
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1)
        )

        # Trajectory processing branch (converts sparse points to dense heatmap)
        self.trajectory_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1)
        )

        # Fusion: additive combination with learnable scaling
        self.depth_scale = nn.Parameter(torch.tensor(1.0))
        self.trajectory_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, depth_map, trajectory_heatmap):
        """
        Args:
            depth_map: (B, 1, H, W) normalized depth map
            trajectory_heatmap: (B, 1, H, W) sparse point heatmap

        Returns:
            control_signal: (B, hidden_dim, H, W) fused control
        """
        # Encode both modalities
        depth_feat = self.depth_encoder(depth_map)
        traj_feat = self.trajectory_encoder(trajectory_heatmap)

        # Additive fusion with learned weighting
        control = self.depth_scale * depth_feat + self.trajectory_scale * traj_feat

        return control

def trajectory_to_heatmap(trajectories, frame_idx, h, w, sigma=5):
    """
    Convert sparse trajectory points to dense heatmap via Gaussian blurring.
    """
    heatmap = np.zeros((h, w))

    for track in trajectories:
        if frame_idx < len(track):
            _, x, y = track[frame_idx]
            # Add Gaussian blob at point location
            y_int, x_int = int(y), int(x)
            if 0 <= y_int < h and 0 <= x_int < w:
                heatmap[max(0, y_int-sigma):min(h, y_int+sigma),
                       max(0, x_int-sigma):min(w, x_int+sigma)] += 1.0

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap
```

**Step 5: Degradation-Aware Training Balance**

Prevent dense depth signals from overwhelming sparse trajectory signals:

```python
def compute_adaptive_control_weight(depth_map, trajectory_heatmap, epoch, total_epochs):
    """
    Adaptively balance control modalities to prevent one from dominating.
    Early epochs: emphasize depth (more stable). Later: increase trajectory weight.
    """
    # Variance-based balancing: high-variance modalities get lower weight
    depth_variance = np.var(depth_map)
    traj_variance = np.var(trajectory_heatmap)

    # Normalize by inverse variance
    depth_weight = 1.0 / (depth_variance + 1e-6)
    traj_weight = 1.0 / (traj_variance + 1e-6)

    # Normalize to sum to 1
    total = depth_weight + traj_weight
    depth_weight /= total
    traj_weight /= total

    # Curriculum: gradually increase trajectory influence
    progress = epoch / total_epochs
    depth_weight = depth_weight * (1 - 0.2 * progress)
    traj_weight = traj_weight * (1 + 0.2 * progress)

    return depth_weight, traj_weight

# Multi-scale fusion with feature-level scaling
def multi_scale_control_fusion(depth_features, trajectory_features, scales=[1.0, 0.5, 0.25]):
    """
    Fuse controls at multiple scales to handle both fine details and coarse structure.
    """
    fused = []
    for depth_feat, traj_feat, scale in zip(depth_features, trajectory_features, scales):
        # Resize to scale
        scaled_depth = torch.nn.functional.interpolate(depth_feat, scale_factor=scale)
        scaled_traj = torch.nn.functional.interpolate(traj_feat, scale_factor=scale)
        fused.append(scaled_depth + scaled_traj)

    return fused
```

**Step 6: Autoregressive Segment Generation**

Generate video segments sequentially, using final frame for initialization:

```python
def generate_video_segments(initial_frame, depth_maps, trajectories,
                           diffusion_model, num_segments, segment_length):
    """
    Generate ultra-long video through autoregressive segment composition.
    """
    all_frames = [initial_frame]
    current_frame = initial_frame

    for seg_idx in range(num_segments):
        # Prepare control signals for this segment
        segment_depth = depth_maps[seg_idx * segment_length:(seg_idx + 1) * segment_length]
        segment_trajectories = [trajectories[t] for t in range(
            seg_idx * segment_length, min((seg_idx + 1) * segment_length, len(trajectories))
        )]

        # Generate segment starting from current_frame
        segment = diffusion_model.generate(
            initial_frame=current_frame,
            depth_control=segment_depth,
            trajectory_control=segment_trajectories,
            num_frames=segment_length,
            noise=noises[seg_idx],
            control_bounds=bounds
        )

        # Append segment (skip first frame to avoid duplication)
        all_frames.extend(segment[1:])

        # Use last frame as initialization for next segment
        current_frame = segment[-1]

    return np.stack(all_frames)
```

### Practical Guidance

**When to Use:**
- Ultra-long video generation (30+ seconds) with dense control requirements
- Scenarios where both spatial structure (depth) and motion (trajectories) matter
- Applications requiring reproducible, controllable generation (e.g., animation, synthetic data)
- Cases where blending dense and sparse guidance improves output quality

**When NOT to Use:**
- Real-time video generation (autoregressive generation is inherently sequential)
- Tasks requiring only spatial or only temporal control (simpler methods suffice)
- Very short videos (<5 seconds) where full-sequence diffusion is more efficient
- Scenarios with limited computational resources (control fusion adds overhead)

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `segment_length` | 48 | Longer segments = fewer boundaries but harder to control; balance around 32-64 frames |
| `blend_ratio` | 0.3 | Controls temporal coherence; higher = smoother transitions, lower = more diverse frames |
| `depth_percentile` | 2 | Lower = more aggressive outlier removal; affects depth scale normalization |
| `velocity_scale` | 1.0 | Controls trajectory influence on motion; increase for stronger motion guidance |
| `curriculum_warmup` | 0.2 | Fraction of training to emphasize depth before trajectory; prevents early trajectory dominance |

**Common Issues:**
- **Flicker between segments**: Increase `blend_ratio` or use stronger temporal consistency loss
- **Depth overwhelming motion**: Lower `depth_scale` parameter or increase `trajectory_scale`
- **Inconsistent colors**: Ensure depth maps are normalized consistently; verify brightness levels across segments
- **Motion jitter**: Reduce `num_diffusion_steps` per segment for smoother, more constrained generation

### Reference

**Paper**: LongVie: Multimodal-Guided Controllable Ultra-Long Video Generation (2508.03694)
- Generates 60-second videos with synchronized depth and trajectory control
- Achieves temporal coherence through unified noise initialization and global normalization
- Introduces LongVGenBench: 100-video benchmark for evaluating long-form generation quality
