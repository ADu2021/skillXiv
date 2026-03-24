---
name: wan-move-motion-control
title: "Wan-Move: Motion-controllable Video Generation via Latent Trajectory Guidance"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.08765
keywords: [video generation, motion control, trajectory guidance, latent space, image-to-video]
description: "Enable precise motion control in video generation using dense point trajectories as latent space features. Wan-Move integrates with existing models without architecture changes—ideal when you need fine-grained scene control without auxiliary motion encoders."
---

## Overview

Wan-Move introduces a framework for controlling motion in video generation models through latent trajectory guidance. Rather than coarse-grained motion guidance, the method enables precise control using dense point trajectories propagated through latent space without requiring architectural modifications to base models.

## When to Use

- Video generation with precise object motion control
- Integrating motion control into existing image-to-video models
- Scenarios requiring dense, fine-grained trajectory specification
- Applications avoiding additional motion encoder overhead
- Need for diverse, complex motion patterns beyond standard guidance

## When NOT to Use

- Simple global camera movement (simpler approaches sufficient)
- Models where architectural modification is acceptable
- Scenarios not requiring dense point-level control
- Real-time generation with strict latency requirements

## Core Technique

Latent space trajectory propagation for motion-aware guidance:

```python
# Motion-controllable video generation via latent trajectories
class WanMoveController:
    def __init__(self, base_video_model):
        """
        Wrap existing video generation model with motion control.
        No architectural modifications to base model.
        """
        self.video_model = base_video_model
        self.vae = base_video_model.vae  # Access VAE for latent space

    def parse_dense_point_trajectories(self, motion_specification):
        """
        Dense point trajectory representation for granular scene control.
        Objects represented as sets of point trajectories rather than
        coarse bounding boxes or simple vectors.
        """
        trajectories = []

        for obj in motion_specification['objects']:
            # Each object gets multiple tracking points
            # e.g., 16-64 points distributed across object
            point_trajectory = {
                'object_id': obj['id'],
                'start_positions': obj['start_points'],  # [num_points, 2]
                'end_positions': obj['end_points'],      # [num_points, 2]
                'num_frames': motion_specification['num_frames']
            }

            # Interpolate trajectory through time
            full_trajectory = self.interpolate_point_trajectory(
                point_trajectory
            )

            trajectories.append(full_trajectory)

        return trajectories

    def interpolate_point_trajectory(self, point_trajectory):
        """
        Temporal interpolation of point positions.
        """
        start = point_trajectory['start_positions']
        end = point_trajectory['end_positions']
        num_frames = point_trajectory['num_frames']

        # Linear interpolation through frames
        timesteps = torch.linspace(0, 1, num_frames)

        full_trajectory = []
        for t in timesteps:
            frame_positions = (1 - t) * start + t * end
            full_trajectory.append(frame_positions)

        return torch.stack(full_trajectory)  # [num_frames, num_points, 2]

    def project_trajectories_to_latent_space(self, trajectories, first_frame_image):
        """
        Project point trajectories into latent space for conditioning.
        Leverages VAE's learned latent representations.
        """
        # Encode first frame to get latent features
        first_latent = self.vae.encode(first_frame_image)

        # Map trajectory points to latent space coordinates
        latent_trajectories = []

        for trajectory in trajectories:
            # Trajectory shape: [num_frames, num_points, 2]
            latent_traj = []

            for frame_idx, frame_points in enumerate(trajectory):
                # Project 2D pixel coordinates to latent space
                latent_points = self.project_pixels_to_latent(
                    frame_points,
                    first_latent,
                    self.video_model.latent_scale
                )
                latent_traj.append(latent_points)

            latent_trajectories.append(torch.stack(latent_traj))

        return latent_trajectories

    def propagate_latent_features_along_trajectories(self, first_frame_latent, latent_trajectories):
        """
        Core insight: propagate first-frame latent features along trajectories
        to create spatiotemporal guidance maps.
        """
        batch_size, latent_h, latent_w, latent_c = first_frame_latent.shape
        num_frames = latent_trajectories[0].shape[0]

        # Initialize spatiotemporal feature map
        guidance_map = torch.zeros(
            batch_size,
            num_frames,
            latent_h,
            latent_w,
            latent_c
        )

        # For each point trajectory
        for traj_idx, latent_traj in enumerate(latent_trajectories):
            # Extract features from first frame at starting positions
            start_positions = latent_traj[0]  # [num_points, 2]

            for point_idx, pos in enumerate(start_positions):
                # Bilinear interpolate feature at position
                h, w = int(pos[0]), int(pos[1])
                h = torch.clamp(h, 0, latent_h - 1)
                w = torch.clamp(w, 0, latent_w - 1)

                start_feature = first_frame_latent[:, h, w, :]

                # Propagate this feature along trajectory
                for frame_idx, frame_pos in enumerate(latent_traj):
                    pos = frame_pos[point_idx]
                    h, w = int(pos[0]), int(pos[1])
                    h = torch.clamp(h, 0, latent_h - 1)
                    w = torch.clamp(w, 0, latent_w - 1)

                    # Add (blend) feature at trajectory position
                    guidance_map[:, frame_idx, h, w, :] += start_feature

        # Normalize to reasonable range
        guidance_map = guidance_map / (len(latent_trajectories) + 1e-8)

        return guidance_map

    def generate_with_trajectory_guidance(self, prompt, first_frame, trajectories):
        """
        Generate video conditioned on trajectory guidance.
        Seamless integration with existing models.
        """
        # Parse trajectories
        parsed_trajectories = self.parse_dense_point_trajectories(trajectories)

        # Project to latent space
        first_latent = self.vae.encode(first_frame)
        latent_trajectories = self.project_trajectories_to_latent_space(
            parsed_trajectories,
            first_frame
        )

        # Create guidance map
        guidance_map = self.propagate_latent_features_along_trajectories(
            first_latent,
            latent_trajectories
        )

        # Concatenate guidance with model input
        # No architectural changes needed: just additional conditioning
        conditioned_input = torch.cat([
            first_latent,
            guidance_map
        ], dim=-1)

        # Generate video using base model
        generated_latent = self.video_model.generate(
            prompt=prompt,
            initial_latent=conditioned_input,
            num_frames=guidance_map.shape[1]
        )

        # Decode to pixel space
        generated_video = self.vae.decode(generated_latent)

        return generated_video

    def create_movebench_evaluation(self, videos):
        """
        MoveBench: new evaluation dataset for motion control.
        Features diverse content, longer videos, high-quality annotations.
        """
        benchmark_data = {
            'videos': videos,
            'annotations': {
                'object_trajectories': [],
                'motion_complexity': [],
                'semantic_relevance': []
            }
        }

        return benchmark_data
```

The framework maintains scalability through standard training procedures, integrating trajectory guidance without auxiliary modules.

## Key Results

- Fine-grained motion control without architecture changes
- Integration with existing image-to-video models (Wan-I2V-14B)
- Dense point trajectory representation enables precise control
- Eliminates need for separate motion encoding modules
- MoveBench benchmark for comprehensive evaluation

## Implementation Notes

- Dense point trajectories capture complex motion patterns
- Latent space propagation avoids pixel-space operations
- Feature propagation creates spatiotemporal guidance
- Seamless integration as conditioning mechanism
- No model retraining required for existing architectures

## References

- Original paper: https://arxiv.org/abs/2512.08765
- Focus: Motion control in video generation
- Domain: Video synthesis, generative models
