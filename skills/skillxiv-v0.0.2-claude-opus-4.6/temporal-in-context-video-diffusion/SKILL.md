---
name: temporal-in-context-video-diffusion
title: "Temporal In-Context Fine-Tuning for Versatile Control of Video Diffusion Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.00996"
keywords: [Video Generation, Diffusion Models, Conditional Generation, In-Context Learning]
description: "Adapt pretrained video diffusion models to conditional tasks using only 10-30 samples without architectural changes."
---

# Temporal In-Context: Control Video Diffusion With Minimal Data

Video diffusion models are powerful but require extensive fine-tuning to adapt to new conditional tasks (image-to-video, video-to-video, style transfer). Temporal in-context fine-tuning inverts this constraint: teach the model to condition on frames using only 10-30 training examples and no architectural changes. The trick is concatenating condition and target frames temporally with progressively noisier buffer frames between them, enabling smooth transitions and condition fidelity without retraining embeddings or attention layers.

This enables rapid adaptation of large video models to custom tasks and domains with minimal data, making video generation accessible for specialized applications.

## Core Concept

Instead of modifying model architecture or embeddings, inject condition information through the temporal dimension. Place condition frames at the start of the sequence, target frames at the end, and progressively noisier interpolation frames between them. The diffusion model learns to reconstruct clean target frames conditioned on the condition context. This leverages the model's existing temporal reasoning without any architectural surgery, works with any pretrained video diffusion model, and requires minimal data.

## Architecture Overview

- **Temporal Concatenation**: Stack condition frames + buffer frames (with escalating noise) + target frames along time axis
- **Noise Scheduling**: Buffer frames transition smoothly from condition (no noise) to full diffusion (maximum noise) to target frames
- **Diffusion Forward Pass**: Standard diffusion operates on concatenated temporal sequence; learns to reconstruct target given condition context
- **Pretrained Model Reuse**: No changes to model weights or architecture; works as an in-context learning mechanism
- **Multi-Task Adaptation**: Same approach handles image-to-video, video-to-video, and other conditional generation tasks

## Implementation

This implementation demonstrates temporal in-context fine-tuning for conditional video generation.

Build the temporal concatenation pipeline:

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ConditionalVideoFrames:
    condition_frames: torch.Tensor  # [C, H, W] or [T_cond, C, H, W]
    target_frames: torch.Tensor    # [T_target, C, H, W]
    buffer_frames: int = 4  # Number of interpolation frames

class TemporalContextBuilder:
    """Build temporal sequences for in-context video conditioning."""

    @staticmethod
    def create_noise_schedule(num_steps: int) -> torch.Tensor:
        """
        Create noise schedule for buffer frames.
        Start at 0 (clean condition) -> increase to ~0.7 (diffusion regime) -> 1.0 (target).
        """
        # Quadratic schedule: smoother than linear
        t = torch.linspace(0, 1, num_steps)
        noise_levels = t ** 2  # Quadratic
        return noise_levels

    @staticmethod
    def add_noise_to_frame(frame: torch.Tensor, noise_level: float,
                          num_timesteps: int = 1000) -> torch.Tensor:
        """
        Add Gaussian noise to a frame proportional to noise level.
        noise_level: 0 = clean, 1 = full noise
        """
        variance = noise_level
        noise = torch.randn_like(frame) * variance
        noisy_frame = frame + noise
        return torch.clamp(noisy_frame, -1, 1)

    def build_temporal_context(self, condition: ConditionalVideoFrames,
                               add_buffer: bool = True) -> torch.Tensor:
        """
        Build full temporal sequence: condition + buffer + target.
        """
        cond_frames = condition.condition_frames
        target_frames = condition.target_frames

        # Handle single condition frame
        if cond_frames.dim() == 3:
            cond_frames = cond_frames.unsqueeze(0)

        # Ensure consistent shape
        B, C, H, W = target_frames.shape if target_frames.dim() == 4 else \
                     (1, *target_frames.shape)

        if add_buffer:
            # Create buffer frames with escalating noise
            num_buffer = condition.buffer_frames
            noise_schedule = self.create_noise_schedule(num_buffer)

            buffer_frames = []
            for i, noise_level in enumerate(noise_schedule):
                # Interpolate between condition and target
                alpha = (i + 1) / num_buffer
                interpolated = (1 - alpha) * cond_frames[-1] + alpha * target_frames[0]

                # Add noise
                noisy = self.add_noise_to_frame(interpolated, noise_level)
                buffer_frames.append(noisy)

            # Stack: [cond, buffer, target]
            buffer_frames = torch.stack(buffer_frames, dim=0)
            full_sequence = torch.cat(
                [cond_frames, buffer_frames, target_frames],
                dim=0
            )
        else:
            full_sequence = torch.cat([cond_frames, target_frames], dim=0)

        return full_sequence

# Example usage
builder = TemporalContextBuilder()

condition = ConditionalVideoFrames(
    condition_frames=torch.randn(3, 256, 256),  # Single image
    target_frames=torch.randn(8, 3, 256, 256),  # 8 target frames
    buffer_frames=4
)

full_seq = builder.build_temporal_context(condition)
print(f"Full temporal sequence shape: {full_seq.shape}")
# Expected: [1 (cond) + 4 (buffer) + 8 (target), 3, 256, 256] = [13, 3, 256, 256]
```

Implement efficient fine-tuning on the concatenated sequences:

```python
from transformers import AutoModel
import torch.optim as optim

class TemporalInContextFinetuner:
    """Fine-tune video diffusion model on conditional tasks."""

    def __init__(self, pretrained_model_name: str = "cogvideo",
                 num_condition_tokens: int = 8,
                 learning_rate: float = 1e-4):
        # Load pretrained video diffusion model
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.num_condition_tokens = num_condition_tokens

        # Only optimize certain parameters for efficiency
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
        self.context_builder = TemporalContextBuilder()

    def forward_pass(self, temporal_sequence: torch.Tensor,
                    target_indices: Tuple[int, int]) -> torch.Tensor:
        """
        Forward pass through diffusion model.
        target_indices: (start_idx, end_idx) of target frames in sequence.
        """
        # Diffusion model processes full temporal sequence
        # We'll compute loss only on target frames
        output = self.model(temporal_sequence, output_hidden_states=True)

        return output

    def compute_conditional_loss(self, temporal_sequence: torch.Tensor,
                                target_start_idx: int) -> torch.Tensor:
        """
        Compute reconstruction loss on target frames conditioned on context.
        """
        # Forward pass
        outputs = self.forward_pass(temporal_sequence,
                                   (target_start_idx, len(temporal_sequence)))

        # Extract target frame predictions
        target_predictions = outputs.last_hidden_state[target_start_idx:]

        # Get ground truth target frames
        target_frames = temporal_sequence[target_start_idx:]

        # MSE loss between predictions and targets
        loss = F.mse_loss(target_predictions, target_frames)

        return loss

    def train_step(self, condition: ConditionalVideoFrames) -> dict:
        """Single training step on conditional video pair."""
        # Build temporal context
        full_seq = self.context_builder.build_temporal_context(condition)

        # Compute target frame start index
        # = num_condition + num_buffer frames
        target_start_idx = condition.condition_frames.shape[0] + condition.buffer_frames

        # Forward and backward
        loss = self.compute_conditional_loss(full_seq, target_start_idx)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss.item()}

# Training loop example
finetuner = TemporalInContextFinetuner(learning_rate=1e-4)

# Load training data: small set of condition-target pairs
# In practice: 10-30 image/video pairs for task
training_data = [
    ConditionalVideoFrames(
        condition_frames=torch.randn(3, 256, 256),
        target_frames=torch.randn(8, 3, 256, 256),
        buffer_frames=4
    )
    for _ in range(5)  # Small dataset
]

print("Starting fine-tuning on conditional generation task...")
for epoch in range(3):
    epoch_loss = 0
    for sample in training_data:
        stats = finetuner.train_step(sample)
        epoch_loss += stats["loss"]

    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(training_data):.4f}")
```

Implement inference for multiple conditional generation modes:

```python
class ConditionalVideoGenerator:
    """Generate videos conditioned on images/videos using fine-tuned model."""

    def __init__(self, finetuned_model):
        self.model = finetuned_model.model
        self.context_builder = TemporalContextBuilder()

    def generate_image_to_video(self, condition_image: torch.Tensor,
                               num_frames: int = 8,
                               steps: int = 50) -> torch.Tensor:
        """
        Generate video sequence starting from a single image.
        Uses temporal context: condition_image + buffer + target_frames.
        """
        # Build temporal context
        condition = ConditionalVideoFrames(
            condition_frames=condition_image,
            target_frames=torch.randn(num_frames, *condition_image.shape),
            buffer_frames=4
        )

        temporal_seq = self.context_builder.build_temporal_context(condition)

        # Denoise diffusion steps (iterative refinement)
        for step in range(steps):
            # Denoise one step
            with torch.no_grad():
                denoised = self.model(temporal_seq)

            # Update target frames (keep condition fixed)
            condition_len = 1 + 4  # condition + buffer
            temporal_seq[condition_len:] = denoised.last_hidden_state[condition_len:]

        # Extract final video (skip condition and buffer frames)
        video = temporal_seq[5:]  # Skip 1 condition + 4 buffer
        return video

    def generate_video_to_video(self, condition_video: torch.Tensor,
                               num_target_frames: int = 8,
                               steps: int = 50) -> torch.Tensor:
        """
        Adapt existing video: condition_video -> transformed_video.
        Useful for style transfer, aspect ratio change, etc.
        """
        # Build context: use last frame of condition video
        condition = ConditionalVideoFrames(
            condition_frames=condition_video[-1:],  # Last frame as condition
            target_frames=torch.randn(num_target_frames, *condition_video.shape[1:]),
            buffer_frames=4
        )

        temporal_seq = self.context_builder.build_temporal_context(condition)

        # Denoise steps
        for step in range(steps):
            with torch.no_grad():
                denoised = self.model(temporal_seq)
            temporal_seq[5:] = denoised.last_hidden_state[5:]

        return temporal_seq[5:]

# Example generation
generator = ConditionalVideoGenerator(finetuner)

# Image to video
test_image = torch.randn(3, 256, 256)
generated_video = generator.generate_image_to_video(test_image, num_frames=8)
print(f"Generated video shape: {generated_video.shape}")

# Video to video
test_video = torch.randn(4, 3, 256, 256)  # 4-frame condition video
adapted_video = generator.generate_video_to_video(test_video, num_target_frames=8)
print(f"Adapted video shape: {adapted_video.shape}")
```

## Practical Guidance

| Aspect | Details |
|--------|---------|
| **Training Samples** | 10-30 pairs sufficient; more helps but diminishing returns |
| **Buffer Frames** | 4-6 optimal; allows smooth transition between condition and generation |
| **Training Epochs** | 3-10 depending on dataset size; watch for overfitting |
| **Learning Rate** | Start 1e-4, reduce to 1e-5 if loss oscillates |
| **Target Frame Count** | 4-16 per sample; balance between diversity and memory |

**When to Use:**
- Need to adapt pretrained video diffusion to custom tasks quickly
- Limited training data (10-30 samples) for your specific application
- Want to preserve base model capabilities while specializing on new task
- Image-to-video, video-to-video, style transfer, aspect ratio conversion
- Don't have resources for full model retraining

**When NOT to Use:**
- Large training datasets available (full fine-tuning better)
- Need to modify model architecture or capabilities fundamentally
- Real-time generation required (iterative denoising adds latency)
- Task requires pixel-perfect control beyond what conditioning allows
- Model's base knowledge is insufficient for task domain

**Common Pitfalls:**
- Too few buffer frames: abrupt transitions between condition and target
- Too many target frames: model spreads attention thin, quality degrades
- Mismatch between condition and target: model learns artifacts; validate data
- Insufficient training iterations: underfitting causes blurry or incoherent frames
- Not preserving condition frames: model learns to ignore context if targets always different

## Reference

Temporal In-Context Fine-Tuning for Versatile Control of Video Diffusion Models
https://arxiv.org/abs/2506.00996
