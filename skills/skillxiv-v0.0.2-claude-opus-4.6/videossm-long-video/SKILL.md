---
name: videossm-long-video
title: "VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.04519
keywords: [video generation, state-space models, long-form synthesis, temporal consistency, autoregressive generation]
description: "Generate minute-scale coherent videos using state-space models as evolving memory for scene dynamics. VideoSSM achieves linear computational complexity while reducing motion drift—ideal when temporal consistency matters across long video sequences."
---

## Overview

VideoSSM unifies autoregressive diffusion with hybrid state-space memory to maintain temporal consistency in long-form video synthesis. The SSM serves as global scene dynamics memory while local context captures fine details, enabling streaming video synthesis without error accumulation.

## When to Use

- Long-form video generation (minute-scale)
- Temporal consistency requirements
- Motion-controlled video synthesis
- Streaming video capabilities needed
- Scenarios avoiding accumulated error drift

## When NOT to Use

- Short video clips
- Scenarios with strict latency requirements
- Applications without temporal structure importance

## Core Technique

Hybrid memory architecture for long-sequence video:

```python
# VideoSSM: Hybrid state-space memory for long videos
class VideoSSMGenerator:
    def __init__(self):
        self.ssm = StateSpaceModel(dim=512)  # Global memory
        self.local_context = ContextBuffer(size=16)  # Local window

    def generate_long_video(self, prompt, num_frames=300):
        """Generate minute-scale video with consistency."""
        # Initialize SSM state from prompt
        ssm_state = self.ssm.initialize(prompt)
        frames = []

        for frame_idx in range(num_frames):
            # Update SSM: evolving scene dynamics
            ssm_state = self.ssm.step(ssm_state, prompt)

            # Local context: recent frames for detail
            local_context = self.local_context.get_context()

            # Generate frame combining global + local
            frame = self.generate_frame(
                ssm_state,
                local_context,
                frame_idx / num_frames
            )

            frames.append(frame)
            self.local_context.add_frame(frame)

        return torch.stack(frames)

    def generate_frame(self, ssm_state, local_context, progress):
        """Generate single frame with hybrid memory."""
        # Diffusion denoising
        noise = torch.randn(1, 3, 512, 512)

        for denoising_step in range(50):
            # Condition on SSM state (scene dynamics)
            global_cond = ssm_state

            # Condition on local context (fine details)
            local_cond = self.embed_local_context(local_context)

            # Fused conditioning
            full_cond = torch.cat([global_cond, local_cond], dim=-1)

            # Denoise step
            noise = self.denoise(noise, full_cond, denoising_step)

        frame = self.decode_to_image(noise)
        return frame
```

## Key Results

- Linear computational complexity vs quadratic
- Minute-scale temporal consistency
- Interactive prompt control
- Content diversity maintained

## References

- Original paper: https://arxiv.org/abs/2512.04519
- Focus: Long-form video generation
- Domain: Video synthesis, temporal modeling
