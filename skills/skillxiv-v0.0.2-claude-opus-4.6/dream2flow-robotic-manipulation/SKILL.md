---
name: dream2flow-robotic-manipulation
title: "Dream2Flow: Bridging Video Generation and Open-World Manipulation with 3D Object Flow"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2512.24766"
keywords: [robotic manipulation, video generation, 3D object flow, embodiment gap, zero-shot learning, trajectory optimization]
description: "Convert video generation model outputs into executable robotic manipulation by extracting 3D object flow trajectories as an intermediate representation. Enables zero-shot manipulation of diverse object types (rigid, articulated, deformable, granular) without task-specific training. Use when pre-trained video models capture plausible manipulation patterns but need grounding in low-level robot control."
---

## When to Use This Skill

- Zero-shot robotic manipulation from video models without task-specific training
- Open-world scenarios with diverse object types and materials
- Problems where desired manipulation can be visualized as a video
- Systems with pre-trained video generation models available
- Tasks requiring adaptation to object variations without retraining

## When NOT to Use This Skill

- Real-time control requiring <100ms latency (video generation is slow)
- Manipulation requiring precise collision avoidance (flow-based approximation may miss fine contacts)
- Scenarios without visual groundtruth for flow validation
- Tasks with hidden or occluded state critical to manipulation
- Highly stochastic domains (uncertainty in flow prediction hurts execution)

## The Embodiment Gap Problem

Video generation models excel at predicting plausible human-like manipulation. But they generate sequences of images, not robot commands:

```
Video Model Output:
Frame 0: Hand above object
Frame 1: Hand grasps object
Frame 2: Hand moves object right
Frame 3: Object at goal location

Needed for Robot:
Joint angles α, β, γ → Joint velocity dα/dt, dβ/dt, dγ/dt

The gap: How do we get from visual frames to motor commands?
```

Dream2Flow bridges this gap using **3D object flow** as an intermediate representation.

## Core Approach: 3D Object Flow as Interface

Instead of trying to reverse-engineer gripper positions and object poses from video frames, Dream2Flow:

1. **Extracts 3D flow**: Track object motion trajectories across video frames in 3D space
2. **Represents motion**: Encode as dense optical flow (where each 3D point moves to)
3. **Plans trajectory**: Convert flow to continuous object trajectory in 3D
4. **Executes via control**: Either trajectory optimization or RL to achieve the target trajectory

This intermediate representation is:
- **Visual-to-motor bridge**: Connects visual understanding (video) to motor control (robot)
- **Object-centric**: Focuses on what matters (object motion) not robot kinematics
- **Generalizable**: Same flow extraction works for rigid, articulated, deformable, or granular objects

## Architecture Pattern

```python
# Dream2Flow pipeline for robotic manipulation
class Dream2FlowManipulator:
    def __init__(self, video_model, flow_extractor, controller):
        self.video_model = video_model  # Pre-trained video generation
        self.flow_extractor = flow_extractor  # Extract 3D flow from frames
        self.controller = controller  # Execute trajectories (RL or optimization)

    def plan_manipulation(self, current_image, goal_image, object_mask):
        """Plan manipulation from visual goal without task-specific training"""
        # Step 1: Generate video of manipulation from current to goal
        video_sequence = self.video_model.generate(
            start_frame=current_image,
            goal_frame=goal_image,
            num_frames=32  # Typical video length
        )

        # Step 2: Extract 3D object motion trajectories
        # Track each pixel in object mask across frames
        flow_trajectory = self.flow_extractor.extract_3d_flow(
            video_sequence,
            object_mask=object_mask,
            num_trajectories=200  # Dense flow: many points tracked
        )

        # Step 3: Aggregate into reference trajectory for object center/pose
        object_trajectory = self.flow_extractor.aggregate_trajectory(flow_trajectory)
        # object_trajectory shape: (T, 3) where T=32, 3=x,y,z position

        # Step 4: Execute trajectory on real robot
        # Option A: Trajectory optimization
        if self.controller.mode == 'optimization':
            actions = self.controller.optimize_trajectory(
                current_state=self.robot_state(),
                target_trajectory=object_trajectory,
                constraints=['collision_free', 'gripper_constraints']
            )

        # Option B: Reinforcement learning (pre-trained policy)
        elif self.controller.mode == 'rl':
            state = self.robot_state()
            for t in range(len(object_trajectory)):
                action = self.controller.get_action(
                    current_state=state,
                    target_position=object_trajectory[t]
                )
                state = self.execute_action(action)

        return actions or trajectory_log

    def extract_and_filter_flow(self, video_frames, object_mask):
        """Robust 3D flow extraction for various object types"""
        flow_3d = []
        for t in range(len(video_frames) - 1):
            # Compute optical flow between consecutive frames
            frame_t = video_frames[t]
            frame_t1 = video_frames[t + 1]
            optical_flow_2d = self.compute_optical_flow(frame_t, frame_t1)

            # Lift to 3D using depth (monocular or from reconstruction)
            depth = self.estimate_depth(frame_t)
            flow_3d_frame = self.lift_to_3d(
                optical_flow_2d,
                depth,
                camera_intrinsics=self.camera_K
            )
            flow_3d.append(flow_3d_frame)

        return flow_3d
```

## Object Type Generalization

A key advantage: same approach works across different object types without retraining:

| Object Type | How It Works |
|---|---|
| **Rigid** (mug, box) | Flow defines how center and orientation change |
| **Articulated** (scissors, drawer) | Flow captures relative motion of linked parts |
| **Deformable** (cloth, rope) | Dense flow tracks surface deformation |
| **Granular** (rice, sand) | Flow approximates bulk motion dynamics |

No task-specific training needed because the video model already understands these physics.

## Execution Strategies

**Strategy A: Trajectory Optimization**
- Optimize robot joint angles to track the target object trajectory
- Fast execution but requires good initial trajectory
- Works with any robot model
- Pseudo-code:
```python
def trajectory_optimize(start_state, target_trajectory):
    def loss(actions):
        state = start_state
        total_loss = 0
        for t, target_pos in enumerate(target_trajectory):
            state = simulate_forward(state, actions[t])
            actual_object_pos = get_object_position(state)
            total_loss += mse(actual_object_pos, target_pos)
        return total_loss + collision_penalty(state)

    return minimize(loss, init_actions=random())
```

**Strategy B: Reinforcement Learning**
- Pre-train policy to track target positions
- More robust to execution noise
- Learns to adapt online
- Slower per-step but more adaptive

## Training-Free vs. Fine-Tuning Trade-off

| Approach | Pros | Cons |
|---|---|---|
| **Pure zero-shot** (no training) | Fast to deploy, generalizes widely | Flow extraction errors compound |
| **Flow-level fine-tuning** | Adapt to specific robot | Still avoids task-specific training |
| **Controller fine-tuning** | Better execution quality | Requires target task data |

## When Flow Extraction Fails

Dream2Flow relies on accurate flow extraction. It struggles when:

- **Occlusions**: Object hidden by gripper or other objects
- **Motion blur**: Fast movements render flow ambiguous
- **Specular reflections**: Shiny surfaces confuse optical flow computation
- **Multiple objects**: Ambiguous which object's flow to track
- **Out-of-distribution physics**: Video model generates unrealistic motion

For these cases:
1. Use multiple video frames and temporal smoothing
2. Combine flow with semantic segmentation for robustness
3. Validate flow plausibility before execution
4. Fall back to trajectory optimization if learning-based control fails

## Implementation Checklist

- Optical flow computation library (OpenCV, PWCNet)
- 3D lifting from optical flow + depth (camera calibration needed)
- Trajectory smoothing (not all flow is equally valid)
- Collision checking (robot constraints)
- Controller (optimization or RL-based)
- Video model (pre-trained required)

## References

- Original paper: https://arxiv.org/abs/2512.24766
- Related: Diffusion models for video, visual imitation learning, inverse models
- Implementation: RAFT/PWCNet for optical flow, trajectory optimization frameworks
