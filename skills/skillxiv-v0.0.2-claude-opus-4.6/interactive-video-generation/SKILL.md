---
name: interactive-video-generation
title: "Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.09350"
keywords: [video generation, adversarial training, real-time inference, interactive]
description: "Convert pre-trained latent video diffusion into real-time autoregressive generators using adversarial post-training, achieving 24fps streaming on single H100."
---

# Real-Time Interactive Video Generation

## Core Concept

This work transforms pre-trained latent video diffusion models into real-time interactive generators via adversarial post-training. Rather than iterative refinement, the model generates complete frames autoregressively (1 NFE per frame), enabling genuine real-time streaming while accepting user controls. Adversarial training proves more efficient than diffusion for autoregressive generation.

## Architecture Overview

- **Latent autoregressive generation**: 1 neural function evaluation per frame
- **Adversarial training framework**: Discriminator ensures quality via adversarial loss
- **Student forcing**: Mitigates error accumulation in long sequences
- **Interactive control**: Accept user inputs during generation
- **Full KV caching**: Computational efficiency for long sequences
- **Real-time performance**: 24fps at 736x416, 1280x720 on 8xH100

## Implementation

### Step 1: Design Latent Frame Generator

Create autoregressive decoder in latent space:

```python
class LatentFrameGenerator(torch.nn.Module):
    def __init__(self, latent_dim: int = 8,
                 context_dim: int = 768,
                 num_layers: int = 12):
        super().__init__()
        self.latent_dim = latent_dim

        # Latent encoder/decoder (from pretrained VAE)
        self.vae = LatentVAE.from_pretrained()

        # Autoregressive transformer for latent generation
        self.transformer = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=context_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Generate next frame latents
        self.latent_head = torch.nn.Linear(
            context_dim,
            latent_dim
        )

        # Context encoder (e.g., text to image prompt)
        self.context_encoder = torch.nn.Sequential(
            torch.nn.Linear(768, context_dim),
            torch.nn.ReLU()
        )

    def encode_frames_to_latents(self, frames: torch.Tensor
                                ) -> torch.Tensor:
        """Convert RGB frames to latent vectors."""

        # Use VAE encoder
        with torch.no_grad():
            latents = self.vae.encode(frames)

        return latents  # (batch, num_frames, latent_dim)

    def generate_frame_latent(self,
                             prev_latents: torch.Tensor,
                             context: torch.Tensor,
                             kv_cache = None
                             ) -> tuple:
        """Generate next frame in latent space."""

        # Encode context
        context_features = self.context_encoder(context)

        # Autoregressive generation
        if kv_cache is not None:
            # Use KV cache for efficiency
            decoder_input = prev_latents[-1:].unsqueeze(0)
            hidden = self.transformer(
                decoder_input,
                memory=context_features,
                memory_key_padding_mask=None
            )
            # Update cache (simplified)
            kv_cache = hidden
        else:
            # Full autoregressive
            hidden = self.transformer(
                prev_latents,
                memory=context_features
            )

        # Predict next latent
        next_latent = self.latent_head(hidden[-1])

        return next_latent, kv_cache

    def decode_latent_to_frame(self, latent: torch.Tensor
                               ) -> torch.Tensor:
        """Convert latent vector back to RGB frame."""

        with torch.no_grad():
            frame = self.vae.decode(latent)

        return frame
```

### Step 2: Implement Adversarial Training

Train discriminator to ensure autoregressive quality:

```python
class VideoDiscriminator(torch.nn.Module):
    def __init__(self, num_frames: int = 8):
        super().__init__()
        self.num_frames = num_frames

        # 3D CNN for spatiotemporal discrimination
        self.spatial_temporal_blocks = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=(3, 4, 4),
                           stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(64, 128, kernel_size=(3, 4, 4),
                           stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv3d(128, 256, kernel_size=(3, 4, 4),
                           stride=(1, 2, 2), padding=(1, 1, 1)),
            torch.nn.LeakyReLU(0.2)
        )

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128, 1)
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs generated video."""

        # Expects (batch, num_frames, 3, height, width)
        video = video.permute(0, 2, 1, 3, 4)

        features = self.spatial_temporal_blocks(video)
        score = self.classifier(features)

        return score
```

### Step 3: Adversarial Post-Training Loop

Train generator and discriminator with student forcing:

```python
class AdversarialPostTrainer:
    def __init__(self, generator: LatentFrameGenerator,
                 discriminator: VideoDiscriminator):
        self.generator = generator
        self.discriminator = discriminator

        self.g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=1e-4,
            betas=(0.5, 0.999)
        )

    def student_forcing_rollout(self, initial_latent,
                               context, num_frames,
                               teacher_forcing_prob=0.5):
        """Rollout with scheduled teacher forcing."""

        generated_latents = [initial_latent]

        for frame_idx in range(1, num_frames):
            # With probability p, use teacher (real) latent
            if torch.rand(1).item() < teacher_forcing_prob:
                # During training, mix real and generated
                prev_latent = initial_latent  # Would use real
            else:
                prev_latent = generated_latents[-1]

            # Generate next frame
            next_latent, _ = self.generator.generate_frame_latent(
                torch.stack(generated_latents),
                context
            )

            generated_latents.append(next_latent)

        return torch.stack(generated_latents)

    def train_step(self, real_video: torch.Tensor,
                  context: torch.Tensor) -> dict:
        """Single adversarial training step."""

        batch_size = real_video.shape[0]
        num_frames = real_video.shape[1]

        # Encode real video to latents
        real_latents = self.generator.encode_frames_to_latents(
            real_video
        )

        # Generate fake video with student forcing
        fake_latents = self.student_forcing_rollout(
            real_latents[:, 0:1],
            context,
            num_frames,
            teacher_forcing_prob=0.3
        )

        # Decode fake latents to frames
        fake_frames = torch.stack([
            self.generator.decode_latent_to_frame(lat)
            for lat in fake_latents
        ], dim=1)

        # Discriminator loss
        real_scores = self.discriminator(real_video)
        fake_scores = self.discriminator(fake_frames.detach())

        d_loss_real = torch.nn.functional.softplus(-real_scores).mean()
        d_loss_fake = torch.nn.functional.softplus(fake_scores).mean()
        d_loss = d_loss_real + d_loss_fake

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Generator loss
        fake_scores = self.discriminator(fake_frames)
        g_loss_adv = torch.nn.functional.softplus(
            -fake_scores
        ).mean()

        # Reconstruction loss
        g_loss_recon = torch.nn.functional.mse_loss(
            fake_frames,
            real_video
        )

        g_loss = g_loss_adv + 10.0 * g_loss_recon

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "g_loss_adv": g_loss_adv.item(),
            "g_loss_recon": g_loss_recon.item()
        }
```

### Step 4: Real-Time Streaming Inference

Generate and stream video with user controls:

```python
class RealtimeVideoStreamer:
    def __init__(self, generator: LatentFrameGenerator,
                 target_fps: int = 24):
        self.generator = generator
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps

    def stream_video(self, prompt: str,
                    user_control_stream = None,
                    output_path: str = None,
                    duration_seconds: float = 10.0):
        """Stream video generation in real-time."""

        import time
        import cv2

        total_frames = int(self.target_fps * duration_seconds)

        # Initialize video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path, fourcc,
                self.target_fps,
                (1280, 720)
            )

        # Encode context once
        context = self.generator.context_encoder(
            encode_text(prompt)
        )

        # Generate initial frame
        initial_latent = torch.randn(1, self.generator.latent_dim)
        prev_frames = [initial_latent]

        start_time = time.time()
        kv_cache = None

        for frame_idx in range(1, total_frames):
            frame_start = time.time()

            # Check for user controls
            if user_control_stream is not None:
                control = user_control_stream.get_latest()
                if control is not None:
                    # Modify context based on user input
                    context = update_context(context, control)

            # Generate next frame
            with torch.no_grad():
                next_latent, kv_cache = (
                    self.generator.generate_frame_latent(
                        torch.stack(prev_frames),
                        context,
                        kv_cache
                    )
                )

                # Decode to RGB
                frame = self.generator.decode_latent_to_frame(
                    next_latent
                )

            prev_frames.append(next_latent)

            # Maintain real-time speed
            frame_elapsed = time.time() - frame_start
            if frame_elapsed < self.frame_time:
                time.sleep(self.frame_time - frame_elapsed)

            # Save frame
            if output_path:
                frame_np = frame[0].cpu().numpy()
                frame_np = (frame_np * 255).astype('uint8')
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                writer.write(frame_np)

            # Status
            if frame_idx % 12 == 0:
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed
                print(f"Frame {frame_idx}: {fps:.1f} FPS")

        if output_path:
            writer.release()
```

## Practical Guidance

**Student Forcing**: Critical for preventing error accumulation. Start with high teacher probability (0.5), decay during training.

**Adversarial vs Diffusion**: Adversarial training is more efficient than iterative diffusion for autoregressive generation. Single forward pass per frame enables real-time performance.

**KV Caching**: Essential for real-time inference. Cache attention states to avoid recomputation.

**Interactive Control**: Accept user inputs (text prompts, control signals) during streaming by updating context features without resetting generation.

**Performance Targets**: 24fps at 736x416 on single H100 is achievable; 1280x720 requires multi-GPU. Optimize for your hardware.

**When to Apply**: Use for interactive video generation systems, real-time content creation, or streaming applications where latency matters.

## Reference

Adversarial post-training enables efficient real-time video generation by converting diffusion models to autoregressive form. Key innovations: (1) single NFE per frame, (2) student forcing to prevent error accumulation, (3) full KV cache utilization. Together these achieve 24fps performance while maintaining quality.
