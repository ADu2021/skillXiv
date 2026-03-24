---
name: world-compass-rl-world-models
title: "WorldCompass: RL for Long-Horizon World Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.09022"
keywords: [World Models, Reinforcement Learning, Video Generation, Action Following, Long-Horizon]
description: "Improve long-horizon world model fidelity using RL with clip-level rollouts and complementary reward functions for action accuracy and visual quality. Breaks computational constraints by evaluating candidate clips incrementally rather than full sequences, enabling efficient multi-objective optimization."
---

# WorldCompass: RL for Long-Horizon World Models

Training diffusion-based world models for long-horizon video generation requires balancing two conflicting objectives: following user actions precisely and maintaining visual quality. Standard supervised fine-tuning struggles because errors compound across timesteps, and determining correct action interpretation is difficult without reward signals.

WorldCompass applies RL with three key innovations: clip-level rollouts reduce computation from O(N·G) to O(N+G), complementary reward functions act as mutual constraints preventing single-objective collapse, and negative-aware fine-tuning makes multi-step optimization feasible for diffusion models.

## Core Concept

Standard approach: generate full video, evaluate once. Expensive and provides weak learning signals for early timesteps.

WorldCompass: generate clips at specific positions, score them independently with dual rewards (action accuracy + visual quality), use scores to update diffusion model. Clip-level granularity makes computation tractable and provides per-segment learning signals.

## Architecture Overview

- **Clip-Level Rollout**: Generate candidate video clips at timesteps k within sequence, not full sequences
- **Dual Rewards**: Action-following accuracy AND visual quality, balancing competing objectives
- **Efficient Sampling**: Use best-of-N selection and timestep subsampling to reduce training cost
- **Diffusion Model Update**: Apply RL gradients to diffusion model parameters via policy gradient
- **Error Mitigation**: Negative-aware fine-tuning prevents model collapse on hard negative examples

## Implementation

Implement clip-level rollout and dual-reward evaluation:

```python
import torch
import torch.nn as nn
from diffusion_model import DiffusionWorldModel

class ClipLevelRollout:
    """Generates and evaluates video clips at specific timesteps."""

    def __init__(self, world_model, action_detector, quality_scorer):
        """
        Args:
            world_model: Diffusion-based world model
            action_detector: Model to evaluate action-following accuracy
            quality_scorer: Model to evaluate visual quality
        """
        self.world_model = world_model
        self.action_detector = action_detector
        self.quality_scorer = quality_scorer

    def generate_clip_at_position(self, prefix_video, position, action_sequence, num_candidates=5):
        """
        Generate candidate video clips starting at position.
        Args:
            prefix_video: Video frames up to position [T_prefix, H, W, 3]
            position: Starting position for clip generation
            action_sequence: Actions to apply during clip [T_clip]
            num_candidates: Number of rollouts
        Returns:
            clips: Generated video clips [num_candidates, T_clip, H, W, 3]
        """
        clips = []
        for _ in range(num_candidates):
            # Diffusion model generates clip conditioned on prefix and actions
            clip = self.world_model.generate_clip(
                prefix=prefix_video,
                actions=action_sequence,
                steps=50  # Diffusion steps
            )
            clips.append(clip)

        return torch.stack(clips)

    def compute_action_accuracy_reward(self, generated_clip, action_sequence):
        """Reward for how well model followed actions."""
        # Detect actions in generated video
        detected_actions = self.action_detector(generated_clip)

        # Compare detected vs intended actions
        action_accuracy = (detected_actions == action_sequence).float().mean()

        return action_accuracy

    def compute_visual_quality_reward(self, generated_clip):
        """Reward for visual fidelity (no artifacts, smooth transitions)."""
        # Score based on:
        # 1. Perceptual quality (LPIPS or similar)
        # 2. Temporal consistency (optical flow variance)
        # 3. Semantic preservation (CLIP embeddings)

        quality_score = self.quality_scorer(generated_clip)
        return quality_score

    def evaluate_clips(self, clips, action_sequence, action_weight=0.6, quality_weight=0.4):
        """
        Compute dual rewards for each candidate clip.
        Args:
            clips: [num_candidates, T, H, W, 3]
            action_sequence: Ground-truth actions [T]
            action_weight, quality_weight: Reward function weights
        Returns:
            rewards: [num_candidates] weighted combined rewards
        """
        action_rewards = torch.stack([
            self.compute_action_accuracy_reward(clip, action_sequence)
            for clip in clips
        ])

        quality_rewards = torch.stack([
            self.compute_visual_quality_reward(clip)
            for clip in clips
        ])

        # Normalize and combine
        action_rewards = (action_rewards - action_rewards.mean()) / (action_rewards.std() + 1e-8)
        quality_rewards = (quality_rewards - quality_rewards.mean()) / (quality_rewards.std() + 1e-8)

        combined_rewards = action_weight * action_rewards + quality_weight * quality_rewards

        return combined_rewards, action_rewards, quality_rewards
```

Integrate into world model training loop:

```python
def world_compass_training_step(world_model, clip_evaluator, prefix_video, action_sequence, optimizer):
    """Single RL training step for world model."""
    # Generate candidate clips
    clips = clip_evaluator.generate_clip_at_position(
        prefix_video,
        position=len(prefix_video),
        action_sequence=action_sequence,
        num_candidates=8
    )

    # Evaluate candidates with dual rewards
    rewards, action_rewards, quality_rewards = clip_evaluator.evaluate_clips(
        clips, action_sequence,
        action_weight=0.6, quality_weight=0.4
    )

    # Select best-of-N
    best_idx = rewards.argmax()
    best_clip = clips[best_idx]
    best_reward = rewards[best_idx]

    # RL update: maximize reward
    # For diffusion, this requires computing log-likelihood gradient
    optimizer.zero_grad()

    # Diffusion loss: KL divergence between denoising and target
    # Weighted by reward signal
    loss = -best_reward * world_model.compute_diffusion_loss(
        best_clip, action_sequence
    )

    loss.backward()
    optimizer.step()

    return {
        'combined_reward': best_reward.item(),
        'action_reward': action_rewards[best_idx].item(),
        'quality_reward': quality_rewards[best_idx].item()
    }

def train_world_compass(world_model, evaluator, dataset, num_epochs=100):
    """Full WorldCompass training loop."""
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        total_reward = 0.0

        for batch_idx, (prefix_video, action_sequence, target_video) in enumerate(dataset):
            metrics = world_compass_training_step(
                world_model, evaluator, prefix_video, action_sequence, optimizer
            )
            total_reward += metrics['combined_reward']

            if (batch_idx + 1) % 50 == 0:
                avg_reward = total_reward / (batch_idx + 1)
                print(f"Epoch {epoch}, Batch {batch_idx + 1}: Avg Reward = {avg_reward:.4f}")

    return world_model
```

## Practical Guidance

| Parameter | Recommendation | Notes |
|-----------|-----------------|-------|
| Clip generation position | Middle of sequence | Avoids boundary artifacts; easier learning. |
| Num rollouts per clip | 4-8 | Balance sample efficiency vs computation. |
| Action weight | 0.5-0.7 | Higher emphasizes action-following. |
| Quality weight | 0.3-0.5 | Balance ensures visual fidelity. |
| Timestep subsampling | Every 3-5 steps | Reduce computation; still provides learning signal. |
| Diffusion steps | 25-50 | Fewer steps speed training; adjust quality tradeoff. |

**When to Use**
- Training long-horizon video world models
- Tasks requiring both action control and visual consistency
- When clip-level evaluation is tractable (e.g., action detection available)
- Interactive environments where action-following matters (robotics, games)

**When NOT to Use**
- Short-horizon or single-frame prediction tasks
- When only one objective matters (pure visual quality or pure action control)
- Environments without clear action semantics

**Common Pitfalls**
- Reward weights unbalanced; one objective collapses (test both extremes, then balance)
- Not computing rewards independently per clip; ensure objectives don't interfere
- Subsampling too aggressively; world model loses fine-grained action signals
- Forgetting error propagation; clips early in sequence get better learning signals

## Reference

See https://arxiv.org/abs/2602.09022 for full implementation, including action detection networks, quality metrics, and validation on complex interactive video generation tasks.
