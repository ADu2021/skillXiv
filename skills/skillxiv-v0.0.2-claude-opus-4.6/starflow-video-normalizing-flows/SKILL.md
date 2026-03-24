---
name: starflow-video-normalizing-flows
title: "STARFlow-V: End-to-End Video Generative Modeling with Normalizing Flow"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.20462"
keywords: [Normalizing Flows, Video Generation, Exact Likelihood, Invertible Transformations, Flow Matching]
description: "Generate videos end-to-end using normalizing flows instead of diffusion: leverage invertible transformations for exact likelihood computation, eliminate train-test mismatch, and achieve non-iterative sampling with native support for multiple tasks (text-to-video, image-to-video) without fine-tuning."
---

# STARFlow-V: Flow-Based Video Generation

While diffusion models dominate video generation through iterative denoising, normalizing flows offer complementary advantages: exact likelihood evaluation, non-iterative sampling, and unified treatment of multiple generation tasks. This skill demonstrates how to implement STARFlow-V, a flow-based video generator that learns invertible transformations mapping video distributions to simple priors.

The core innovation is combining global-local latent structures (compact global temporal context + local spatial detail) with flow-score matching to maintain causality while preserving temporal consistency.

## Core Concept

STARFlow-V implements video generation via normalizing flows:

1. **Invertible Transformation**: Learn mapping V = g_θ(Z) where Z ~ N(0,I) and g is invertible
2. **Global-Local Structure**: Global latent carries temporal context; local blocks preserve spatial detail
3. **Flow-Score Matching**: Lightweight denoiser maintains causality without breaking invertibility
4. **End-to-End Training**: Direct maximum likelihood estimation via change-of-variables formula

## Architecture Overview

- **Encoder**: Maps videos to latent space via variational compression
- **Global Latent Sequence**: Compact representation of temporal dynamics
- **Local Latent Blocks**: Spatial detail information
- **Flow Model**: Invertible transformation mapping prior to latent distribution
- **Decoder**: Reconstructs video from latents
- **Flow-Score Matching Module**: Lightweight causality-preserving denoiser

## Implementation Steps

The flow-based generation system operates through encoding, flow transformation, and decoding.

**1. Implement Invertible Latent Encoder**

Create invertible compression mapping videos to latent space.

```python
class InvertibleVideoEncoder(torch.nn.Module):
    """
    Encodes videos to latent space with invertible transformation.
    Enables exact likelihood computation through change-of-variables formula.
    """
    def __init__(self, channels=3, hidden_dim=64):
        super().__init__()

        self.channels = channels
        self.hidden_dim = hidden_dim

        # Spatial encoder (per-frame)
        self.spatial_encoder = torch.nn.Sequential(
            InvertibleConv1x1(channels, hidden_dim),
            CouplingLayer(hidden_dim, hidden_dim * 2),
            InvertibleConv1x1(hidden_dim, hidden_dim),
            CouplingLayer(hidden_dim, hidden_dim * 2)
        )

        # Temporal encoder (across frames)
        self.temporal_encoder = torch.nn.Sequential(
            TemporalCouplingLayer(hidden_dim, hidden_dim),
            TemporalCouplingLayer(hidden_dim, hidden_dim)
        )

    def forward(self, video):
        """
        Invertible encoding of video to latent space.
        Args:
            video: (batch, time, channels, height, width)
        Returns:
            latents: (batch, time, hidden_dim, height, width)
            log_det_jacobian: Log determinant of transformation for likelihood
        """
        batch, time, c, h, w = video.shape

        # Encode spatially per frame
        spatial_latents = []
        log_det_sum = 0.0

        for t in range(time):
            frame = video[:, t]
            latent, log_det = self.spatial_encoder(frame)
            spatial_latents.append(latent)
            log_det_sum += log_det

        spatial_latents = torch.stack(spatial_latents, dim=1)

        # Encode temporally
        latents, log_det_temporal = self.temporal_encoder(spatial_latents)

        total_log_det = log_det_sum + log_det_temporal

        return latents, total_log_det

    def inverse(self, latents):
        """
        Decode from latent space back to video space (invertible).
        """
        # Inverse temporal transform
        spatial_latents = self.temporal_encoder.inverse(latents)

        # Inverse spatial transform per frame
        batch, time = spatial_latents.shape[:2]
        video = []

        for t in range(time):
            frame_latent = spatial_latents[:, t]
            frame = self.spatial_encoder.inverse(frame_latent)
            video.append(frame)

        video = torch.stack(video, dim=1)

        return video
```

**2. Implement Global-Local Latent Structure**

Design latent representation separating temporal dynamics from spatial detail.

```python
class GlobalLocalLatentStructure(torch.nn.Module):
    """
    Decomposes video latents into global (temporal) and local (spatial) components.
    Reduces error accumulation in autoregressive generation.
    """
    def __init__(self, latent_dim=256, num_frames=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frames = num_frames

        # Global stream: compact temporal sequence
        self.global_compressor = torch.nn.Sequential(
            torch.nn.Conv3d(latent_dim, latent_dim // 4, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(latent_dim // 4, latent_dim // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        )

        # Local stream: per-frame spatial detail
        self.local_extractor = torch.nn.Sequential(
            torch.nn.Conv3d(latent_dim, latent_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            torch.nn.ReLU()
        )

    def decompose(self, latents):
        """
        Decompose latents into global and local components.
        Args:
            latents: (batch, time, latent_dim, height, width)
        Returns:
            global_latent: (batch, global_dim, time) temporal dynamics
            local_latent: (batch, time, local_dim, height, width) spatial details
        """
        # Reshape for 3D convolution
        batch, time, dim, h, w = latents.shape
        latents_3d = latents.permute(0, 2, 1, 3, 4)  # (batch, dim, time, h, w)

        # Extract global: compress spatial dimensions
        global_latent = self.global_compressor(latents_3d)
        global_latent = global_latent.mean(dim=(3, 4))  # Average spatial → (batch, global_dim, time)

        # Extract local: preserve all details
        local_latent = self.local_extractor(latents_3d)
        local_latent = local_latent.permute(0, 2, 1, 3, 4)  # (batch, time, local_dim, h, w)

        return global_latent, local_latent

    def recombine(self, global_latent, local_latent):
        """
        Recombine global and local latents (inverse of decompose).
        """
        # Expand global to spatial dimensions
        batch, global_dim, time = global_latent.shape
        _, _, local_dim, h, w = local_latent.shape

        global_expanded = global_latent.unsqueeze(-1).unsqueeze(-1)
        global_expanded = global_expanded.expand(-1, -1, -1, h, w)
        global_expanded = global_expanded.permute(0, 2, 1, 3, 4)  # (batch, time, global_dim, h, w)

        # Combine: scaled addition
        combined = local_latent + 0.1 * global_expanded

        return combined
```

**3. Implement Normalizing Flow Model**

Build the invertible transformation learning flow.

```python
class NormalizingFlowForVideo(torch.nn.Module):
    """
    Normalizing flow mapping simple prior to video latent distribution.
    Enables exact likelihood computation via change-of-variables formula.
    """
    def __init__(self, latent_dim=256, num_flows=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_flows = num_flows

        # Stack of invertible transformations
        self.flows = torch.nn.ModuleList([
            CouplingFlowLayer(latent_dim, hidden_dim=512)
            for _ in range(num_flows)
        ])

    def forward(self, latents):
        """
        Forward pass through flow (encode to prior space).
        Args:
            latents: (batch, total_latent_dim) flattened latents
        Returns:
            z: (batch, total_latent_dim) samples from prior
            log_det_jacobian: Likelihood adjustment
        """
        z = latents
        log_det_sum = 0.0

        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det

        return z, log_det_sum

    def inverse(self, z):
        """
        Inverse pass through flow (decode from prior to latent space).
        For generation, sample z ~ N(0,I) and apply inverse.
        """
        latents = z

        # Reverse flow order for inverse
        for flow in reversed(self.flows):
            latents = flow.inverse(latents)

        return latents

    def sample(self, batch_size, device):
        """
        Generate samples from prior and decode through flow.
        Args:
            batch_size: Number of samples
            device: Torch device
        Returns:
            latents: Decoded samples in latent space
        """
        # Sample from standard normal prior
        z = torch.randn(batch_size, self.latent_dim, device=device)

        # Decode through inverse flow
        latents = self.inverse(z)

        return latents
```

**4. Implement Flow-Score Matching for Causality**

Add lightweight denoiser maintaining temporal causality without breaking invertibility.

```python
class FlowScoreMatching(torch.nn.Module):
    """
    Lightweight score-matching denoiser that maintains causality.
    Applied to latent space for refined generation without breaking invertibility.
    """
    def __init__(self, latent_dim=256, num_frames=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_frames = num_frames

        # Causal transformer: only attends to past frames
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=512,
            batch_first=True
        )

        self.causal_transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=4
        )

        # Score network: predicts ∇logp(x)
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, latent_dim)
        )

    def forward(self, latents, t=None):
        """
        Predict score (gradient of log probability) with causal attention.
        Args:
            latents: (batch, time, latent_dim)
            t: Optional diffusion timestep (not used for flows, but compatible)
        Returns:
            score: (batch, time, latent_dim) score predictions
        """
        # Create causal mask: each position only attends to itself and past
        seq_len = latents.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=latents.device) * float('-inf'),
            diagonal=1
        )

        # Apply causal transformer
        refined = self.causal_transformer(
            latents, src_mask=causal_mask
        )

        # Predict score
        score = self.score_head(refined)

        return score
```

**5. Training with Maximum Likelihood Estimation**

Train flow model using exact likelihood through change-of-variables.

```python
def train_normalizing_flow(
    encoder,
    flow_model,
    video_batch,
    optimizer,
    learning_rate=1e-4
):
    """
    Single training step for STARFlow-V.
    Optimizes exact likelihood using change-of-variables formula.
    Args:
        encoder: Invertible video encoder
        flow_model: Normalizing flow
        video_batch: (batch, time, channels, height, width) videos
        optimizer: PyTorch optimizer
        learning_rate: Gradient step size
    Returns:
        loss: Negative log-likelihood (for minimization)
    """
    # Encode video to latent space
    latents, log_det_encoder = encoder(video_batch)

    # Flatten for flow
    batch_size = latents.shape[0]
    latents_flat = latents.reshape(batch_size, -1)

    # Forward through flow
    z, log_det_flow = flow_model(latents_flat)

    # Likelihood computation (change-of-variables)
    # log p(x) = log p(z) + log |det(dz/dx)|
    log_p_z = -0.5 * (z ** 2).sum(dim=-1)  # Standard normal log-prob

    log_likelihood = log_p_z + log_det_flow + log_det_encoder

    # Negative log-likelihood (for minimization)
    nll_loss = -log_likelihood.mean()

    # Backward and update
    optimizer.zero_grad()
    nll_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(flow_model.parameters()),
        1.0
    )
    optimizer.step()

    return nll_loss.item()
```

**6. Generation with Multiple Conditions**

Use trained flow for flexible multi-task generation.

```python
def generate_with_flow(
    encoder,
    flow_model,
    condition_type='text',
    condition_input=None,
    num_frames=16,
    device='cuda'
):
    """
    Generate video using trained flow model.
    Supports text, image, or unconditioned generation.
    Args:
        encoder: Trained invertible encoder
        flow_model: Trained normalizing flow
        condition_type: 'text', 'image', or 'none'
        condition_input: Condition (text string or image tensor)
        num_frames: Number of frames to generate
        device: Torch device
    Returns:
        generated_video: (1, num_frames, 3, height, width) video
    """
    # Sample from prior
    latent_dim = flow_model.latent_dim
    z = torch.randn(1, latent_dim, device=device)

    # Decode through inverse flow
    latents = flow_model.inverse(z)

    # If conditioned, modulate latents
    if condition_type == 'text':
        # Encode text and modulate latents
        text_embedding = encode_text(condition_input)
        latents = latents + text_embedding

    elif condition_type == 'image':
        # Use image to guide latent generation
        image_latent = encoder(condition_input.unsqueeze(1))
        latents = 0.7 * latents + 0.3 * image_latent.mean(dim=1)

    # Reshape latents
    latents = latents.reshape(1, num_frames, -1, 8, 8)  # Example spatial size

    # Decode to video
    generated_video = encoder.inverse(latents)

    return generated_video
```

## Practical Guidance

**When to Use STARFlow-V:**
- Scenarios requiring exact likelihood computation (probability estimation, anomaly detection)
- Non-iterative sampling important (real-time applications)
- Need for unified multi-task generation (text-to-video, image-to-video)
- Applications where train-test mismatch is problematic

**When NOT to Use:**
- Very high-resolution generation (>1080p) where flow computational cost dominates
- Scenarios heavily optimized for diffusion models already
- Tasks where diffusion's iterative refinement provides quality advantages

**Key Hyperparameters:**
- `num_flows`: Number of invertible layers (8-32; more = better expressiveness)
- `coupling_layers_per_flow`: Depth of coupling transformations (2-4)
- `global_compression_ratio`: How much to compress temporal info (4-16×)
- `flow_score_mixing_weight`: How much denoiser influences generation (0.1-0.3)

**Performance Tips:**
- Pre-train encoder on large unlabeled video corpus before flow training
- Use batch normalization cautiously (breaks invertibility); prefer layer norm
- Cache encoder output to avoid redundant computation during flow training
- For long videos, process in overlapping chunks and stitch results

**Integration Pattern:**
STARFlow-V naturally integrates into video diffusion pipelines as alternative backbone. Use flow for initial coarse generation, optionally refine with diffusion in high-detail region if needed.

## Reference

Research paper: https://arxiv.org/abs/2511.20462
