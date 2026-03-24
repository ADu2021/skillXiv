---
name: exevrm-video-reward-modeling
title: "Video-Based Reward Modeling for Computer-Use Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.10178"
keywords: [Reward Modeling, Computer Use, Video Understanding, RL, Vision Language]
description: "Build robust reward models for computer-use agents by processing execution videos with spatiotemporal token pruning (STP+TTP) to remove redundancy while preserving task-relevant UI details. Achieve 84.7% accuracy with adversarial instruction translation."
---

# Technique: Spatiotemporal Token Pruning for Efficient Video Reward Modeling

Computer-use agents generate long, high-resolution execution videos that are expensive to process. Standard vision models struggle with this modality's inherent redundancy: static backgrounds, repeated layouts, and unchanged UI elements dominate frames. Execution Video Reward Models (ExeVRM) use **spatiotemporal token pruning** to eliminate this redundancy while preserving subtle visual cues determining task correctness.

The approach combines spatial pruning (removing homogeneous regions) with temporal pruning (suppressing unchanged tokens) and augments training with adversarial instruction translation to handle distribution shifts.

## Core Concept

ExeVRM operates through three mechanisms:

1. **Spatial Token Pruning (STP)**: Identifies and removes large homogeneous regions (backgrounds, toolbars) while preserving localized UI elements
2. **Temporal Token Pruning (TTP)**: Suppresses tokens unchanged across consecutive frames, focusing on state transitions
3. **Adversarial Instruction Translation**: Generates hard negatives by pairing successful trajectories with semantically mismatched instructions

This enables efficient processing of long, high-resolution videos while maintaining robustness to instruction variations.

## Architecture Overview

- **Video encoder**: ViT-based visual processor
- **Spatial pruner**: Identifies homogeneous regions to mask
- **Temporal pruner**: Compares frame deltas to suppress unchanged tokens
- **MLLM backbone**: Vision-language model for understanding instructions
- **Reward head**: Classifier outputting execution and consistency scores
- **Adversarial augmentation**: Generates hard negatives for training

## Implementation Steps

### Step 1: Spatial Token Pruning

Remove large homogeneous regions while preserving UI-relevant patches.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTokenPruner:
    def __init__(self, homogeneity_threshold=0.95, min_region_size=32):
        self.threshold = homogeneity_threshold
        self.min_region_size = min_region_size

    def compute_region_homogeneity(self, frame):
        """
        Measure color/texture homogeneity of spatial regions.

        frame: (height, width, 3) RGB values
        """
        # Compute local standard deviation using convolution
        kernel_size = 16
        stride = 8

        # Unfold frame into patches
        patches = F.unfold(
            frame.permute(2, 0, 1).unsqueeze(0),
            kernel_size=kernel_size,
            stride=stride
        )  # (1, 3*k*k, num_patches)

        # Compute variance per patch (homogeneity metric)
        patch_variance = patches.std(dim=1)  # (1, num_patches)

        return patch_variance

    def get_pruning_mask(self, frame):
        """
        Generate mask: 1 for regions to keep, 0 for pruning.
        """
        homogeneity = self.compute_region_homogeneity(frame)

        # Mask: keep tokens from low-homogeneity regions (higher variance)
        keep_mask = homogeneity < (self.threshold * homogeneity.max())

        return keep_mask

    def prune_tokens(self, visual_tokens, frame):
        """
        visual_tokens: (batch, num_patches, dim)
        frame: (height, width, 3)
        returns: pruned tokens with mask
        """
        keep_mask = self.get_pruning_mask(frame)

        # Extract indices to keep
        keep_indices = torch.nonzero(keep_mask).squeeze(-1)

        # Select tokens
        pruned_tokens = visual_tokens[:, keep_indices, :]

        return pruned_tokens, keep_mask
```

### Step 2: Temporal Token Pruning

Suppress tokens unchanged across frames, focus on state transitions.

```python
class TemporalTokenPruner:
    def __init__(self, change_threshold=0.1):
        self.threshold = change_threshold
        self.prev_frame_tokens = None

    def compute_token_delta(self, current_tokens, prev_tokens):
        """
        Measure L2 distance between consecutive frame representations.

        current_tokens: (batch, num_patches, dim)
        prev_tokens: (batch, num_patches, dim)
        """
        delta = torch.norm(current_tokens - prev_tokens, dim=-1)  # (batch, num_patches)

        return delta

    def get_change_mask(self, current_tokens, prev_tokens):
        """
        Mask: 1 for tokens with significant change, 0 for static.
        """
        delta = self.compute_token_delta(current_tokens, prev_tokens)

        # Adaptive threshold (relative to frame's max change)
        frame_max_delta = delta.max(dim=-1, keepdim=True)[0]
        threshold = self.threshold * frame_max_delta

        change_mask = delta > threshold

        return change_mask

    def prune_tokens(self, current_tokens, prev_tokens):
        """
        Filter tokens to keep only those with significant changes.
        """
        if prev_tokens is None:
            # First frame: keep all
            return current_tokens, torch.ones_like(current_tokens[:, :, 0])

        change_mask = self.get_change_mask(current_tokens, prev_tokens)

        # Select tokens with changes
        batch_size = current_tokens.shape[0]
        pruned_tokens_list = []

        for b in range(batch_size):
            selected = current_tokens[b, change_mask[b]]
            pruned_tokens_list.append(selected)

        # Pad to same length
        max_len = max(t.shape[0] for t in pruned_tokens_list)
        padded_tokens = torch.stack([
            F.pad(t, (0, 0, 0, max_len - t.shape[0]))
            for t in pruned_tokens_list
        ])

        self.prev_frame_tokens = current_tokens  # Cache for next frame

        return padded_tokens, change_mask
```

### Step 3: Combined STP+TTP Processing Pipeline

Integrate spatial and temporal pruning into coherent video processing.

```python
class VideoRewardModel(nn.Module):
    def __init__(self, backbone_model, hidden_dim=768):
        super().__init__()
        self.backbone = backbone_model
        self.spatial_pruner = SpatialTokenPruner()
        self.temporal_pruner = TemporalTokenPruner()

        # Reward heads
        self.execution_head = nn.Linear(hidden_dim, 1)
        self.consistency_head = nn.Linear(hidden_dim, 1)

    def forward(self, instruction, video_frames):
        """
        instruction: str description
        video_frames: (num_frames, height, width, 3)
        """
        pruned_frame_features = []

        prev_pruned_tokens = None

        for frame_idx, frame in enumerate(video_frames):
            # Extract visual features
            with torch.no_grad():
                visual_tokens = self.backbone.encode_image(frame)

            # Spatial pruning
            spatial_pruned, spatial_mask = self.spatial_pruner.prune_tokens(
                visual_tokens,
                frame
            )

            # Temporal pruning (accumulates over frames)
            if prev_pruned_tokens is not None:
                temporal_pruned, temporal_mask = self.temporal_pruner.prune_tokens(
                    spatial_pruned,
                    prev_pruned_tokens
                )
            else:
                temporal_pruned = spatial_pruned

            prev_pruned_tokens = spatial_pruned

            pruned_frame_features.append(temporal_pruned)

        # Aggregate across frames
        aggregated_features = torch.cat(pruned_frame_features, dim=1)  # (1, total_tokens, dim)

        # Process with language model
        instruction_tokens = self.backbone.encode_text(instruction)

        # Combined understanding
        combined = torch.cat([
            instruction_tokens.mean(dim=0, keepdim=True),
            aggregated_features.mean(dim=0, keepdim=True)
        ], dim=-1)

        # Predict rewards
        execution_score = self.execution_head(combined)
        consistency_score = self.consistency_head(combined)

        return {
            'execution': execution_score.sigmoid(),
            'consistency': consistency_score.sigmoid()
        }
```

### Step 4: Adversarial Instruction Translation

Synthesize hard negatives by pairing trajectories with semantically mismatched instructions.

```python
def adversarial_instruction_translation(
    successful_trajectory,
    task_instructions,
    llm_model,
    num_hard_negatives=3
):
    """
    Generate plausible but incorrect instruction-trajectory pairs.
    """
    hard_negatives = []

    # Extract trajectory semantics
    trajectory_description = llm_model.summarize_trajectory(successful_trajectory)

    # For each instruction, generate semantically different variants
    for instruction in task_instructions:
        # Generate modifications: addition, deletion, paraphrasing
        modifications = [
            f"{instruction} and additionally...",  # Addition variant
            instruction.split()[:-2],  # Deletion variant
            llm_model.paraphrase(instruction)  # Paraphrase variant
        ]

        for modified in modifications[:num_hard_negatives]:
            # Verify it's different from original
            similarity = cosine_similarity(
                llm_model.encode(instruction),
                llm_model.encode(str(modified))
            )

            if similarity < 0.8:  # Ensure sufficient divergence
                hard_negatives.append({
                    'trajectory': successful_trajectory,
                    'instruction': str(modified),
                    'label': 0  # Mismatch label
                })

    return hard_negatives

def train_step_with_adversarial_augmentation(
    model,
    instruction,
    video,
    label,
    adversarial_examples,
    optimizer
):
    """
    Training step with hard negative mining.
    """
    # Standard forward pass
    output = model(instruction, video)
    execution_loss = F.binary_cross_entropy(output['execution'], label)

    # Adversarial hard negatives
    adversarial_loss = 0
    for adv_ex in adversarial_examples:
        adv_output = model(adv_ex['instruction'], video)
        # Should output low score (mismatch)
        adv_loss = F.binary_cross_entropy(
            adv_output['execution'],
            torch.tensor(adv_ex['label'])
        )
        adversarial_loss += adv_loss

    total_loss = execution_loss + 0.5 * (adversarial_loss / len(adversarial_examples))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

## Practical Guidance

**When to Use:**
- Building reward models for computer-use agents
- Scenarios with long, high-resolution execution videos
- Tasks with significant spatial/temporal redundancy (UIs, multi-step processes)
- Distribution shift from train to deployment (adversarial augmentation helps)

**When NOT to Use:**
- Short, simple videos where pruning adds overhead
- Tasks requiring frame-by-frame granularity
- Extreme real-time constraints (pruning adds preprocessing latency)

**Hyperparameter Tuning:**
- **homogeneity_threshold**: 0.85-0.95; lower keeps more details
- **spatial_region_size**: 16-32 pixels; smaller = finer pruning
- **change_threshold**: 0.05-0.2; lower = more sensitive to changes
- **num_hard_negatives**: 2-5; more improves robustness

**Common Pitfalls:**
- Pruning too aggressive, losing task-relevant UI details
- Temporal pruning suppressing important state transitions
- Insufficient adversarial examples leading to instruction brittleness
- Cache not properly managed between video sequences

## Reference

[ExeVRM paper on arXiv](https://arxiv.org/abs/2603.10178)
