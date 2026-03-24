---
name: video-world-models-spatial-memory
title: "Video World Models with Long-term Spatial Memory"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.05284"
keywords: [video-generation, world-models, spatial-memory, 3d-consistency, long-context]
description: "Enables long-term consistent video generation through three-tier memory architecture combining working memory, geometry-grounded point clouds, and episodic keyframes."
---

# Video World Models with Long-term Spatial Memory

## Core Concept

Video world models struggle to maintain scene consistency when revisiting previously generated locations due to limited temporal context windows. This work introduces a neuroscience-inspired memory architecture with three tiers—working memory (recent frames), spatial memory (static geometry), and episodic memory (historical keyframes)—enabling coherent generation across extended sequences. The key insight is grounding generation in persistent 3D point cloud representations that maintain physical consistency across time.

## Architecture Overview

- **Working Memory**: Recent context frames (4-8 frames) capturing dynamic elements and temporal dependencies
- **Spatial Memory**: TSDF-fused point clouds representing static scene geometry, filtering out transient objects
- **Episodic Memory**: Sparse historical keyframes at regular intervals providing long-range context
- **Static Point Cloud Rendering**: Conditioning input that guides generation while preserving spatial consistency
- **TSDF-Fusion Filtering**: Removes dynamic elements, retaining only persistent scene structure
- **Autoregressive Update**: Newly generated frames' static components update spatial memory for future predictions

## Implementation

The following code demonstrates the memory architecture and fusion process:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from collections import deque

class TrieredMemoryVideoModel(nn.Module):
    """
    Three-tier memory architecture for long-term consistent video generation.
    """
    def __init__(self, feature_dim: int = 768, max_keyframes: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_keyframes = max_keyframes

        # Working memory: deque of recent frames
        self.working_memory = deque(maxlen=8)

        # Episodic memory: sparse keyframes
        self.episodic_memory = deque(maxlen=max_keyframes)

        # Spatial memory: point cloud representation
        self.spatial_memory = None

    def update_working_memory(self, frame: torch.Tensor) -> None:
        """Add frame to working memory (most recent context)."""
        self.working_memory.append(frame)

    def tsdf_fusion(self, depth_map: np.ndarray,
                    camera_pose: np.ndarray,
                    voxel_size: float = 0.02) -> np.ndarray:
        """
        Truncated Signed Distance Function fusion to build spatial memory.
        Filters dynamic elements by preserving only consistent geometric structures.

        depth_map: (H, W) depth values from generated frame
        camera_pose: (4, 4) camera extrinsic matrix
        Returns: filtered point cloud array
        """
        # Convert depth to point cloud
        h, w = depth_map.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Intrinsic matrix (simplified)
        fx, fy = w / 2, h / 2
        cx, cy = w / 2, h / 2

        z = depth_map
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy

        # Stack into point cloud
        points = np.stack([X, Y, z], axis=-1).reshape(-1, 3)

        # Transform to world coordinates
        points_homog = np.hstack([points, np.ones((len(points), 1))])
        points_world = (camera_pose @ points_homog.T)[:3].T

        # TSDF filtering: keep points with consistent geometry
        # Simple heuristic: remove points with high variance in local neighborhood
        filtered_points = self._filter_dynamic_points(points_world)

        return filtered_points

    def _filter_dynamic_points(self, points: np.ndarray,
                              neighbor_radius: float = 0.1,
                              max_variance: float = 0.05) -> np.ndarray:
        """Remove dynamic points by variance in local neighborhoods."""
        from scipy.spatial import cKDTree

        if len(points) < 10:
            return points

        tree = cKDTree(points)
        variances = []

        for i, point in enumerate(points):
            neighbors = tree.query_ball_point(point, neighbor_radius)
            if len(neighbors) > 3:
                neighbor_variance = np.var(points[neighbors], axis=0).mean()
                variances.append(neighbor_variance)
            else:
                variances.append(float('inf'))

        # Keep points with low local variance (static)
        static_mask = np.array(variances) < max_variance
        return points[static_mask]

    def add_episodic_keyframe(self, frame: torch.Tensor,
                             point_cloud: np.ndarray) -> None:
        """Store frame and associated point cloud as episodic memory."""
        self.episodic_memory.append({
            'frame': frame.detach().cpu(),
            'point_cloud': point_cloud
        })

    def get_memory_condition(self) -> torch.Tensor:
        """
        Prepare conditioning input combining all three memory types.
        Returns: tensor suitable for concatenation with model input
        """
        condition_parts = []

        # Working memory encoding
        if len(self.working_memory) > 0:
            working = torch.stack(list(self.working_memory)).mean(dim=0)
            condition_parts.append(working)

        # Point cloud rendering (spatial memory)
        if self.spatial_memory is not None:
            point_cloud_rendered = self._render_point_cloud(self.spatial_memory)
            condition_parts.append(point_cloud_rendered)

        # Episodic memory: average features from keyframes
        if len(self.episodic_memory) > 0:
            episodic_frames = torch.stack([
                kf['frame'] for kf in list(self.episodic_memory)
            ])
            episodic_cond = episodic_frames.mean(dim=0)
            condition_parts.append(episodic_cond)

        return torch.cat(condition_parts, dim=0) if condition_parts else torch.zeros(1)

    def _render_point_cloud(self, point_cloud: np.ndarray,
                            resolution: Tuple[int, int] = (480, 720)) -> torch.Tensor:
        """Render point cloud to image plane for conditioning."""
        h, w = resolution
        rendering = np.zeros((h, w, 3), dtype=np.float32)

        # Simple orthographic projection
        if len(point_cloud) > 0:
            x_norm = ((point_cloud[:, 0] - point_cloud[:, 0].min()) /
                      (point_cloud[:, 0].max() - point_cloud[:, 0].min() + 1e-8))
            y_norm = ((point_cloud[:, 1] - point_cloud[:, 1].min()) /
                      (point_cloud[:, 1].max() - point_cloud[:, 1].min() + 1e-8))

            x_px = (x_norm * (w - 1)).astype(int)
            y_px = (y_norm * (h - 1)).astype(int)

            valid = (x_px >= 0) & (x_px < w) & (y_px >= 0) & (y_px < h)
            rendering[y_px[valid], x_px[valid]] = [1.0, 1.0, 1.0]

        return torch.from_numpy(rendering).permute(2, 0, 1).unsqueeze(0)

    def forward(self, current_frame: torch.Tensor,
                depth_map: np.ndarray,
                camera_pose: np.ndarray) -> torch.Tensor:
        """
        Generate next frame conditioned on memory architecture.
        """
        # Update working memory
        self.update_working_memory(current_frame)

        # Update spatial memory via TSDF fusion
        new_points = self.tsdf_fusion(depth_map, camera_pose)
        if self.spatial_memory is None:
            self.spatial_memory = new_points
        else:
            self.spatial_memory = np.vstack([self.spatial_memory, new_points])

        # Periodically add episodic keyframe
        if len(self.working_memory) % 4 == 0:
            self.add_episodic_keyframe(current_frame, self.spatial_memory)

        # Get conditioning from all memory types
        memory_condition = self.get_memory_condition()

        return memory_condition
```

## Practical Guidance

**Memory Update Cadence**: Update spatial memory every 4-8 generated frames to balance consistency with computational cost. More frequent updates improve geometric accuracy but increase overhead.

**Point Cloud Density**: Typical TSDF fusion produces 50K-500K points per view. Downsample to 100K points if memory becomes constrained; this minimally impacts quality.

**Camera Pose Tracking**: Ensure accurate camera pose estimation from the video generation model. Pose errors directly propagate to spatial memory misalignment.

**Episodic Keyframe Interval**: Store keyframes every 16-32 frames. This provides sufficient long-range context without excessive memory overhead.

**TSDF Truncation Distance**: Set truncation distance to 2-4 voxel sizes. This controls which depth variations are treated as dynamic vs. measurement noise.

**Training Resolution**: Train on 480×720 clips; spatial memory efficiently generalizes to 1080p at test time due to 3D geometry grounding.

## Reference

The memory-augmented approach achieves improved consistency metrics:
- **View Recall**: Higher pixel-level consistency when revisiting scenes
- **Camera Accuracy**: Fewer off-trajectory hallucinations
- **Temporal Coherence**: More stable object and structure persistence

This method is particularly valuable for long-form video generation (100+ frames) and applications requiring geometric accuracy such as virtual environment synthesis or video editing.
