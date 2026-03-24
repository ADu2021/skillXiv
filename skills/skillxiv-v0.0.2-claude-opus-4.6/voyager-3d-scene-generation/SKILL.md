---
name: voyager-3d-scene-generation
title: "Voyager: World-Consistent Video Diffusion for 3D Scene Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.04225"
keywords: [3d-generation, video-diffusion, depth-estimation, world-caching]
description: "Generate spatially-coherent 3D point-cloud videos from single images using depth-fused diffusion with efficient world caching for infinite scene exploration."
---

# Voyager: Long-Range and World-Consistent Video Diffusion

## Core Concept

Voyager addresses a critical limitation in 3D scene generation: most methods cannot simultaneously generate RGB and depth sequences that remain spatially consistent across arbitrarily long videos. By jointly generating aligned RGB-depth pairs and implementing efficient world caching with point culling, Voyager enables direct 3D reconstruction without expensive post-processing pipelines.

## Architecture Overview

- **Geometry-Injected Conditioning**: Use partial RGB and depth maps rather than RGB-only conditioning to reduce hallucinations in complex occlusions
- **Depth-Fused Diffusion**: Concatenate RGB and depth along spatial dimensions for pixel-level interaction
- **World Caching System**: Accumulate 3D points from generated frames; point culling removes redundancy, reducing memory ~40% while maintaining essential geometry
- **Auto-Regressive Sampling**: Generate arbitrary-length videos while maintaining world consistency through incremental point cloud updates
- **Camera Path Control**: Accept user-specified camera trajectories for controlled scene exploration

## Implementation

### Step 1: Prepare Dual-Modal Training Dataset

```python
from typing import List, Dict, Tuple
import numpy as np

class DualModalDatasetBuilder:
    def __init__(self, num_training_pairs=100000):
        self.target_size = num_training_pairs
        self.auto_annotate_pipeline = self.AnnotationPipeline()

    class AnnotationPipeline:
        def estimate_metric_depth(self, single_image):
            """Use pretrained depth estimator (MiDaS, etc) to get metric depth"""
            depth_estimator = load_pretrained_depth_model('midas')
            relative_depth = depth_estimator(single_image)

            # Convert to metric scale using known scene structures
            metric_depth = self.scale_to_metric(relative_depth)
            return metric_depth

        def generate_camera_poses(self, image, num_frames=30):
            """Automatically generate camera trajectories"""
            poses = []

            # Infer scene bounds from depth and image content
            scene_bounds = self.estimate_scene_bounds(image)

            # Generate smooth camera path (circular, spiral, etc)
            for t in np.linspace(0, 1, num_frames):
                pose = self.generate_smooth_camera_motion(
                    scene_bounds, t
                )
                poses.append(pose)

            return poses

        def render_target_frames(self, single_image, camera_poses):
            """Render RGB-D video from single image using 3D warping"""
            # Use depth map to create 3D point cloud from image
            point_cloud = self.image_to_point_cloud(
                single_image,
                self.estimate_metric_depth(single_image)
            )

            rgb_frames = []
            depth_frames = []

            for pose in camera_poses:
                # Render point cloud from camera pose
                rgb_frame = self.render_rgb(point_cloud, pose)
                depth_frame = self.render_depth(point_cloud, pose)

                rgb_frames.append(rgb_frame)
                depth_frames.append(depth_frame)

            return rgb_frames, depth_frames

    def build_training_dataset(self, single_images: List):
        """Create training pairs: single image -> RGB-D video"""
        dataset = []

        for img in single_images:
            # Auto-annotate with depth
            depth_map = self.auto_annotate_pipeline.estimate_metric_depth(img)

            # Generate camera poses
            camera_poses = self.auto_annotate_pipeline.generate_camera_poses(
                img, num_frames=30
            )

            # Render target RGB-D sequence
            rgb_frames, depth_frames = (
                self.auto_annotate_pipeline.render_target_frames(
                    img, camera_poses
                )
            )

            # Create training example
            example = {
                'input_image': img,
                'input_depth': depth_map,
                'camera_poses': camera_poses,
                'target_rgb_frames': rgb_frames,
                'target_depth_frames': depth_frames,
                'scene_id': len(dataset),
            }

            dataset.append(example)

        print(f"Built dataset: {len(dataset)} single-image -> video pairs")
        print(f"Total frames: {len(dataset) * 30}")

        return dataset

# Build 100K+ training pairs
builder = DualModalDatasetBuilder(num_training_pairs=100000)
training_data = builder.build_training_dataset(images)
```

### Step 2: Implement Depth-Fused Diffusion Model

```python
import torch
import torch.nn as nn

class DepthFusedDiffusionModel(nn.Module):
    def __init__(self, num_channels=8, num_layers=12):
        super().__init__()
        # 8 channels: 3 RGB + 1 Depth + 4 extra features
        self.num_channels = num_channels

        # Transformer blocks for spatial interaction
        self.spatial_fusion = nn.ModuleList([
            self.SpatialFusionBlock(num_channels)
            for _ in range(num_layers)
        ])

        # Temporal consistency module
        self.temporal_consistency = self.TemporalModule()

    class SpatialFusionBlock(nn.Module):
        """Enable pixel-level interaction between RGB and depth"""

        def __init__(self, channels):
            super().__init__()
            # Cross-modal attention: RGB queries, depth keys/values
            self.rgb_to_depth_attn = nn.MultiheadAttention(
                embed_dim=channels, num_heads=8
            )
            self.depth_to_rgb_attn = nn.MultiheadAttention(
                embed_dim=channels, num_heads=8
            )

        def forward(self, rgb, depth):
            """
            Args:
                rgb: [B, 3, H, W] - RGB channels
                depth: [B, 1, H, W] - Depth channel
            Returns:
                fused: [B, 4, H, W] - Interacted RGB-D
            """
            B, _, H, W = rgb.shape

            # Flatten spatial dimensions for attention
            rgb_flat = rgb.view(B, 3, -1).transpose(1, 2)  # [B, HW, 3]
            depth_flat = depth.view(B, 1, -1).transpose(1, 2)  # [B, HW, 1]

            # Cross-attention: let depth inform RGB
            rgb_informed, _ = self.depth_to_rgb_attn(
                rgb_flat, depth_flat, depth_flat
            )

            # Cross-attention: let RGB inform depth
            depth_informed, _ = self.rgb_to_depth_attn(
                depth_flat, rgb_flat, rgb_flat
            )

            # Reshape back to spatial
            rgb_informed = rgb_informed.transpose(1, 2).view(B, 3, H, W)
            depth_informed = depth_informed.transpose(1, 2).view(B, 1, H, W)

            # Concatenate fused representations
            fused = torch.cat([rgb_informed, depth_informed], dim=1)

            return fused

    class TemporalModule(nn.Module):
        """Ensure consistency across video frames"""

        def __init__(self):
            super().__init__()
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=4, num_heads=4
            )

        def forward(self, frame_sequence):
            """
            Args:
                frame_sequence: [T, B, 4, H, W] - T frames of RGB-D
            Returns:
                consistent_frames: [T, B, 4, H, W]
            """
            T, B, C, H, W = frame_sequence.shape

            # Compute temporal attention to smooth transitions
            frames_flat = frame_sequence.view(T, B*H*W, C)

            attended, _ = self.temporal_attention(
                frames_flat, frames_flat, frames_flat
            )

            return attended.view(T, B, C, H, W)

    def forward(self, image, depth_init, camera_poses, num_frames=30):
        """
        Generate RGB-D video from single image and camera poses.

        Args:
            image: [B, 3, H, W] - Input RGB image
            depth_init: [B, 1, H, W] - Estimated depth of input image
            camera_poses: [B, num_frames, 4, 4] - Camera transformation matrices
            num_frames: int - Number of frames to generate
        """

        # Diffusion forward pass for first frame
        rgb_frames = []
        depth_frames = []

        # Use input as conditioning
        rgb_cond = image
        depth_cond = depth_init

        for frame_idx in range(num_frames):
            # Camera transformation: warp previous frame using depth
            if frame_idx > 0:
                warped_rgb = self.warp_frame(
                    rgb_frames[-1],
                    depth_frames[-1],
                    camera_poses[:, frame_idx]
                )
            else:
                warped_rgb = rgb_cond

            # Denoise RGB-D jointly
            noisy_frame = self.add_noise(
                torch.cat([warped_rgb, depth_cond], dim=1),
                noise_level=frame_idx / num_frames  # Less noise as we progress
            )

            # Apply spatial fusion blocks
            denoised = noisy_frame
            for fusion_block in self.spatial_fusion:
                rgb_part = denoised[:, :3, :, :]
                depth_part = denoised[:, 3:4, :, :]
                denoised = fusion_block(rgb_part, depth_part)

            # Split into RGB and depth
            rgb_frame = denoised[:, :3, :, :]
            depth_frame = denoised[:, 3:4, :, :]

            rgb_frames.append(rgb_frame)
            depth_frames.append(depth_frame)

        # Apply temporal consistency
        frame_sequence = torch.stack(rgb_frames + depth_frames, dim=0)
        consistent = self.temporal_consistency(frame_sequence)

        return rgb_frames, depth_frames

    def warp_frame(self, rgb, depth, camera_pose):
        """Warp frame using depth and camera transformation"""
        # Implement backward warping using depth and pose
        return rgb  # Simplified placeholder
```

### Step 3: Implement World Caching System

```python
import open3d as o3d
from collections import defaultdict

class WorldCachingSystem:
    def __init__(self, memory_reduction_target=0.6):
        self.point_cloud = o3d.geometry.PointCloud()
        self.frame_points = defaultdict(list)
        self.memory_reduction_target = memory_reduction_target

    def accumulate_points_from_frame(self, rgb_frame, depth_frame,
                                    camera_pose, intrinsics):
        """Add 3D points from new frame to world cache"""

        # Project depth + camera pose to get 3D points
        points_3d = self.depth_to_world_coordinates(
            depth_frame, camera_pose, intrinsics
        )

        # Get colors from RGB frame
        colors = self.sample_colors_from_rgb(rgb_frame, points_3d)

        # Add to accumulating point cloud
        new_points = o3d.geometry.PointCloud()
        new_points.points = o3d.utility.Vector3dVector(points_3d)
        new_points.colors = o3d.utility.Vector3dVector(colors)

        self.point_cloud += new_points

        return len(points_3d)

    def point_culling(self, culling_ratio=0.4):
        """Remove redundant points to reduce memory"""

        initial_size = len(self.point_cloud.points)

        # Downsample point cloud aggressively
        culled = self.point_cloud.voxel_down_sample(
            voxel_size=0.05  # Adjust based on scene scale
        )

        # Use statistical outlier removal
        culled, _ = culled.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        final_size = len(culled.points)
        actual_reduction = 1.0 - (final_size / initial_size)

        print(f"Point culling: {initial_size} -> {final_size} points")
        print(f"Memory reduction: {actual_reduction:.1%}")

        self.point_cloud = culled

        return actual_reduction

    def auto_regressive_generation(self, model, image, depth_init,
                                   camera_poses_all, batch_size=8):
        """
        Generate video auto-regressively, periodically calling point culling.
        This enables generating arbitrary-length videos without memory explosion.
        """

        rgb_frames = []
        depth_frames = []
        culling_interval = 30  # Cull points every N frames

        for frame_batch_idx in range(0, len(camera_poses_all), batch_size):
            batch_poses = camera_poses_all[
                frame_batch_idx : frame_batch_idx + batch_size
            ]

            # Generate batch of frames
            rgb_batch, depth_batch = model(
                image, depth_init, batch_poses
            )

            rgb_frames.extend(rgb_batch)
            depth_frames.extend(depth_batch)

            # Accumulate points
            for rgb_f, depth_f, pose in zip(rgb_batch, depth_batch, batch_poses):
                self.accumulate_points_from_frame(
                    rgb_f, depth_f, pose, intrinsics
                )

            # Periodically cull redundant points
            if len(rgb_frames) % culling_interval == 0:
                reduction = self.point_culling(culling_ratio=0.4)
                print(f"Frame {len(rgb_frames)}: Memory reduction {reduction:.1%}")

        # Final output
        return rgb_frames, depth_frames, self.point_cloud
```

## Practical Guidance

1. **Geometry-Aware Conditioning**: Always provide depth alongside RGB conditioning. Depth prevents the diffusion model from hallucinating surfaces in occluded regions, dramatically improving consistency.

2. **Joint RGB-Depth Generation**: Design diffusion models to generate both modalities simultaneously with cross-attention, not as separate pipelines. Pixel-level RGB-depth interaction is crucial.

3. **Camera Path Specification**: Support user-defined camera trajectories (circular, spiral, free-form). This gives users control and enables reproducible generation.

4. **World Caching for Infinite Video**: Don't generate monolithic long videos. Instead, accumulate points, periodically cull redundancy (removing ~40% without visible quality loss), and continue generating. This enables arbitrarily long videos.

5. **Point Culling Parameters**: Use voxel downsampling with small voxel sizes (0.05 unit scale) and statistical outlier removal (20 neighbors, 2.0 std ratio). Tune based on target scene density.

6. **Evaluation**: Assess both visual quality (RGB fidelity) and geometric consistency (3D reconstruction error). A good model excels at both.

## Reference

- Paper: Voyager (2506.04225)
- Architecture: Depth-fused diffusion transformer with world caching
- Dataset: 100,000+ automatic RGB-D video pairs via single-image rendering
- Key Innovation: Joint RGB-depth generation with efficient world state management
