---
name: tokenbridge-visual-generation
title: "Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.16430"
keywords: [Visual Generation, Discrete Tokens, Autoregressive, Quantization, VAE]
description: "Build autoregressive image generators using post-training quantization that bridges continuous VAE tokens with discrete vocabulary modeling. Achieves state-of-the-art visual quality via dimension-wise token prediction without training instability."
---

## Core Concept

Visual generation faces a tradeoff: discrete tokens enable simple modeling via cross-entropy loss but lose visual information; continuous tokens preserve quality but require complex distribution modeling. TokenBridge solves this by applying **post-training quantization** to pretrained VAE features, obtaining discrete tokens while maintaining visual fidelity. The method then uses **dimension-wise factorization** to predict high-dimensional token spaces efficiently in an autoregressive manner.

## Architecture Overview

- **Post-Training Quantization**: Non-uniform quantization of continuous VAE features based on standard normal distribution properties
- **Dimension-Wise Factorization**: Decomposes large token vocabulary (B^C combinations) into sequential per-dimension predictions
- **Spatial-Dimension Autoregression**: Combines spatial generation order with channel-wise token prediction
- **FFT-Guided Generation Order**: Prioritizes low-frequency (structural) information early in generation
- **Lightweight Autoregressive Head**: Single MLP conditioned on previously generated channels

## Implementation Steps

### Step 1: Post-Training Quantization of VAE Features

Apply dimension-wise non-uniform quantization to pretrained continuous VAE latents without retraining the VAE.

```python
import torch
import numpy as np
from scipy import stats

def post_training_quantization(
    continuous_features,
    num_levels=256,
    assume_gaussian=True
):
    """
    Quantize continuous VAE features to discrete tokens using
    non-uniform quantization. Assumes Gaussian distribution in latent space.
    continuous_features: shape (batch, channels, height, width)
    """
    batch, channels, h, w = continuous_features.shape
    quantized = torch.zeros_like(continuous_features, dtype=torch.long)

    for c in range(channels):
        channel_data = continuous_features[:, c, :, :].flatten()

        if assume_gaussian:
            # Compute quantile levels based on standard normal
            quantiles = np.linspace(0, 1, num_levels + 1)
            q_vals = stats.norm.ppf(quantiles[1:-1])  # Exclude 0 and 1

            # Normalize channel data to standard normal
            mean = channel_data.mean()
            std = channel_data.std() + 1e-8
            normalized = (channel_data - mean) / std

            # Assign tokens based on quantile membership
            tokens = torch.searchsorted(
                torch.tensor(q_vals, dtype=torch.float32),
                normalized
            )
        else:
            # Uniform quantization fallback
            min_val = channel_data.min()
            max_val = channel_data.max()
            normalized = (channel_data - min_val) / (max_val - min_val + 1e-8)
            tokens = (normalized * (num_levels - 1)).long()

        quantized[:, c, :, :] = tokens.reshape(batch, h, w)

    return quantized
```

### Step 2: Build Dimension-Wise Autoregressive Head

Create an efficient MLP-based head that predicts one channel at a time, conditioning on previously generated channels.

```python
import torch.nn as nn

class DimensionWiseAutoregressiveHead(nn.Module):
    """
    Predicts discrete tokens one channel (dimension) at a time.
    Conditions each prediction on all previously generated channels.
    """
    def __init__(
        self,
        hidden_dim=512,
        num_channels=16,
        num_token_levels=256,
        spatial_size=32
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.num_token_levels = num_token_levels
        self.spatial_size = spatial_size

        # Embedding for previously generated channel tokens
        self.token_embedding = nn.Embedding(num_token_levels, hidden_dim // 4)

        # Position embedding for spatial locations
        self.spatial_embedding = nn.Parameter(
            torch.randn(spatial_size * spatial_size, hidden_dim // 4)
        )

        # MLP layers for predicting next channel
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_token_levels)
        )

    def forward(self, previous_channels, spatial_context):
        """
        previous_channels: (batch, num_channels_generated, spatial_size, spatial_size)
        spatial_context: (batch, hidden_dim // 2, spatial_size, spatial_size)
        Returns logits: (batch, spatial_size, spatial_size, num_token_levels)
        """
        batch_size = previous_channels.shape[0]
        spatial_h, spatial_w = spatial_context.shape[-2:]
        total_spatial = spatial_h * spatial_w

        # Embed previous channel tokens
        prev_embeds = self.token_embedding(previous_channels)
        # Global average pool previous channels
        channel_context = prev_embeds.mean(dim=(2, 3))  # (batch, hidden_dim // 4)

        # Expand spatial context and combine
        spatial_flat = spatial_context.permute(0, 2, 3, 1).reshape(
            batch_size, total_spatial, -1
        )

        # Broadcast channel context and concatenate
        channel_context_expanded = channel_context.unsqueeze(1).expand(
            batch_size, total_spatial, -1
        )

        combined = torch.cat([spatial_flat, channel_context_expanded], dim=-1)

        # Predict tokens for this dimension
        logits = self.mlp(combined)  # (batch, spatial, num_levels)
        logits = logits.reshape(batch_size, spatial_h, spatial_w, -1)

        return logits
```

### Step 3: Determine Generation Order via FFT Analysis

Compute generation order by analyzing frequency content; low-frequency (structural) dimensions should be generated first.

```python
def compute_generation_order_fft(latent_samples, num_channels):
    """
    Analyze frequency distribution across channels using FFT.
    Return channel indices ordered from low-frequency to high-frequency.
    latent_samples: (num_samples, channels, height, width)
    """
    frequency_energy = []

    for c in range(num_channels):
        channel_data = latent_samples[:, c, :, :].cpu().numpy()

        # Compute 2D FFT per sample and average
        ffts = np.abs(np.fft.fft2(channel_data, axes=(1, 2)))

        # Shift zero-frequency to center
        ffts_shifted = np.fft.fftshift(ffts, axes=(1, 2))

        # Compute energy: sum of low-frequency components (center region)
        h, w = ffts_shifted.shape[1:]
        center_h, center_w = h // 3, w // 3
        low_freq_region = ffts_shifted[
            :,
            h // 2 - center_h:h // 2 + center_h,
            w // 2 - center_w:w // 2 + center_w
        ]
        low_freq_energy = low_freq_region.sum()

        frequency_energy.append((c, low_freq_energy))

    # Sort by frequency energy (descending) - generate low-freq first
    frequency_energy.sort(key=lambda x: x[1], reverse=True)
    generation_order = [idx for idx, _ in frequency_energy]

    return generation_order
```

### Step 4: Autoregressive Generation Loop

Generate discrete tokens spatially and dimension-wise, using the order determined by FFT analysis.

```python
def autoregressive_generate(
    model,
    batch_size,
    spatial_size=32,
    num_channels=16,
    num_levels=256,
    generation_order=None
):
    """
    Generate discrete tokens autoregressively: iterate through
    spatial locations and dimensions in specified order.
    Returns quantized token grid ready for VAE decoding.
    """
    if generation_order is None:
        generation_order = list(range(num_channels))

    # Initialize token grid
    generated_tokens = torch.zeros(
        batch_size, num_channels, spatial_size, spatial_size,
        dtype=torch.long
    )

    # Get spatial context (e.g., from image encoder or as learned embeddings)
    spatial_context = torch.randn(
        batch_size, 256, spatial_size, spatial_size
    )

    # Iterate through channels in generation order
    for channel_idx in generation_order:
        # Prepare previously generated channels
        prev_channels = generated_tokens[:, :channel_idx, :, :]

        # Get logits for this channel
        logits = model.ar_head(prev_channels, spatial_context)
        # logits shape: (batch, spatial_size, spatial_size, num_levels)

        # Sample tokens (or use argmax for deterministic generation)
        sampled = torch.argmax(logits, dim=-1)  # Greedy
        generated_tokens[:, channel_idx, :, :] = sampled

    return generated_tokens
```

### Step 5: Decode to Continuous Image via VAE

De-quantize discrete tokens back to continuous features and decode through VAE decoder.

```python
def dequantize_tokens(quantized_tokens, vae_decoder, num_levels=256):
    """
    Convert discrete tokens back to continuous values and decode via VAE.
    Uses inverse of the non-uniform quantization scheme.
    """
    batch, channels, h, w = quantized_tokens.shape
    continuous_features = torch.zeros_like(quantized_tokens, dtype=torch.float32)

    for c in range(channels):
        # Map tokens back to continuous range
        token_vals = quantized_tokens[:, c, :, :].float()

        # Inverse quantization: assume tokens map to quantile positions
        quantiles = np.linspace(0, 1, num_levels)
        q_vals = stats.norm.ppf(quantiles)  # Map to standard normal

        # Interpolate: token i corresponds to quantile at q_vals[i]
        continuous_vals = torch.zeros_like(token_vals)
        for i in range(batch):
            for j in range(h):
                for k in range(w):
                    t = token_vals[i, j, k].long()
                    t_clamped = torch.clamp(t, 0, len(q_vals) - 1)
                    continuous_vals[i, j, k] = q_vals[int(t_clamped)]

        continuous_features[:, c, :, :] = continuous_vals

    # Decode through VAE decoder
    images = vae_decoder(continuous_features)
    return images
```

## Practical Guidance

**When to Use:**
- Building autoregressive image generators that need both quality and training stability
- Projects requiring real-time or near-real-time generation with discrete token modeling
- Scenarios with limited computational resources where continuous token models are too slow
- Systems needing confidence-guided generation (sampling from discrete probabilities)

**When NOT to Use:**
- Conditional generation with extreme conditioning signals (discrete token vocab may be insufficient)
- Very high-resolution generation (>1024×1024) where spatial autoregression becomes prohibitively slow
- Scenarios requiring precise continuous latent manipulation

**Hyperparameter Tuning:**
- **num_levels (quantization levels)**: 256 standard; increase to 512 for higher fidelity, decrease to 128 for speed
- **Channel order**: FFT-guided order typically outperforms random or reverse-frequency ordering
- **Temperature**: For sampling during generation, start at 1.0; reduce to 0.7-0.8 for less stochastic output
- **Batch size during generation**: Larger batches improve efficiency; start at 32 if memory permits

**Common Pitfalls:**
- Insufficient quantization levels lead to visible banding artifacts; validate with FID scores
- FFT-based generation order assumes frequency-structured data; may not work for highly chaotic distributions
- Over-aggressive spatial autoregression on large images; use efficient spatial orderings (Z-order, Hilbert curve)
- Mismatch between VAE training data and test generation; ensure VAE is robust to quantization artifacts

## References

- arXiv:2503.16430 - TokenBridge paper
- https://arxiv.org/abs/2006.11239 - VAE-based image generation background
- https://arxiv.org/abs/2302.04761 - Autoregressive visual generation techniques
