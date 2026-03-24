---
name: unicom-compressed-multimodal-representations
title: "UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.10702"
keywords: [Multimodal, Compression, Semantic Representations, Generation, Understanding]
description: "Compress visual embeddings into compact latent space for unified image understanding and generation. Combines attention-based compression with diffusion decoding to bridge comprehension and generation through a shared semantic bottleneck."
---

# Technique: Channel-Wise Visual Compression for Unified Multimodal Modeling

Multimodal models struggle with the fundamental tension: dense visual features enable fine-grained understanding, but high dimensionality is wasteful for generation. UniCom inverts this by compressing along the *channel* axis rather than spatially, maintaining spatial structure while reducing feature richness. This creates a unified semantic space for both understanding and generation tasks.

The key insight is that channel reduction is more effective than spatial downsampling—it preserves the spatial layout needed for tasks like visual grounding while compressing redundancy in feature representation.

## Core Concept

UniCom operates through three stages:

1. **Semantic Compression**: Attention-based compressor reduces visual features from 1152-d to 64-d per spatial location
2. **Transfusion Prediction**: Single transformer processes interleaved text and compressed latents
3. **Diffusion Reconstruction**: Flow-matching decoder expands latents to pixels for generation

This architecture enables efficient bidirectional flow: text → latents (comprehension) and latents → pixels (generation), all within a single model.

## Architecture Overview

- **Visual encoder**: Standard ViT producing 1152-d features
- **Semantic compressor**: Attention-based channel reduction module
- **Transfusion backbone**: Unified transformer for text and latents
- **Latent predictor**: Maps text to compressed representations
- **Diffusion decoder**: Flow-matching model from latents to pixels
- **Joint optimizer**: Reconstruction + perceptual loss for both tasks

## Implementation Steps

### Step 1: Attention-Based Semantic Compressor

Compress visual features while preserving semantic content via learned attention weighting.

```python
import torch
import torch.nn as nn

class SemanticCompressor(nn.Module):
    def __init__(self, input_dim=1152, output_dim=64, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Attention-based compression
        self.query = nn.Linear(input_dim, num_heads * output_dim)
        self.key = nn.Linear(input_dim, num_heads * output_dim)
        self.value = nn.Linear(input_dim, num_heads * output_dim)

        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

    def forward(self, visual_features):
        """
        visual_features: (batch, height, width, input_dim)
        returns: (batch, height, width, output_dim)
        """
        batch_size, height, width, feat_dim = visual_features.shape

        # Reshape to (batch*h*w, feat_dim)
        features_flat = visual_features.reshape(-1, feat_dim)

        # Compute Q, K, V for channel dimension compression
        Q = self.query(features_flat)  # (batch*h*w, num_heads*output_dim)
        K = self.key(features_flat)
        V = self.value(features_flat)

        # Reshape for multi-head attention over feature dimensions
        Q = Q.reshape(-1, self.num_heads, self.head_dim)
        K = K.reshape(-1, self.num_heads, self.head_dim)
        V = V.reshape(-1, self.num_heads, self.head_dim)

        # Attention over feature dimension
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Weighted compression
        compressed = torch.bmm(attn, V)  # (batch*h*w, num_heads, head_dim)
        compressed = compressed.reshape(-1, self.output_dim)

        # Reshape back to spatial layout
        compressed = compressed.reshape(batch_size, height, width, self.output_dim)

        return compressed
```

### Step 2: Transfusion Processing Pipeline

Interleave text and visual tokens for efficient unified processing.

```python
class TransfusionBackbone(nn.Module):
    def __init__(self, vocab_size, latent_dim=64, hidden_dim=768, num_layers=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Token embeddings
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)

        # Position embeddings
        self.position_embedding = nn.Embedding(2048, hidden_dim)

        # Transformer layers
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=3072,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

    def forward(self, text_ids, latent_features):
        """
        text_ids: (batch, text_len)
        latent_features: (batch, num_latents, latent_dim) from compressor
        """
        batch_size = text_ids.shape[0]

        # Embed text tokens
        text_embed = self.text_embedding(text_ids)  # (batch, text_len, hidden_dim)

        # Project latent features
        latent_embed = self.latent_projection(latent_features)  # (batch, num_latents, hidden_dim)

        # Interleave: alternate text and latent tokens
        # Simple approach: concatenate; sophisticated: true interleaving
        mixed_sequence = torch.cat([text_embed, latent_embed], dim=1)  # (batch, text_len+num_latents, hidden_dim)

        # Add position embeddings
        positions = torch.arange(mixed_sequence.shape[1], device=mixed_sequence.device)
        position_embed = self.position_embedding(positions.unsqueeze(0))
        mixed_sequence = mixed_sequence + position_embed

        # Process through transformer layers
        for layer in self.transformer:
            mixed_sequence = layer(mixed_sequence)

        # Extract latent portion for downstream generation
        latent_portion = mixed_sequence[:, text_ids.shape[1]:]

        return latent_portion
```

### Step 3: Latent Predictor Head

Predict compressed representations from text for generation tasks.

```python
class LatentPredictorHead(nn.Module):
    def __init__(self, hidden_dim=768, latent_dim=64, num_latents=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        # MLP to predict latent features
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_latents * latent_dim)
        )

    def forward(self, text_representations):
        """
        text_representations: (batch, hidden_dim)
        returns: (batch, num_latents, latent_dim)
        """
        predictions = self.predictor(text_representations)
        predictions = predictions.reshape(-1, self.num_latents, self.latent_dim)
        return predictions
```

### Step 4: Diffusion Decoder from Latents to Pixels

Use flow matching to reconstruct images from compressed latents.

```python
class DiffusionDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_diffusion_steps=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_diffusion_steps

        # Predict pixel residuals at each diffusion step
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),  # +1 for time embedding
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 3)  # RGB output
        )

    def forward(self, compressed_latents, num_pixels):
        """
        Reconstruct pixels from compressed latents via iterative refinement.
        """
        batch_size = compressed_latents.shape[0]
        pixels = torch.randn(batch_size, num_pixels, 3)  # Start from noise

        # Iterative denoising
        for t in range(self.num_steps):
            time_embedding = torch.tensor(t / self.num_steps, dtype=torch.float32)

            # Expand latents to pixel resolution via upsampling
            expanded_latents = self.upsample(compressed_latents, num_pixels)

            # Predict residual
            residual = self.net(
                torch.cat([expanded_latents, time_embedding.unsqueeze(0).unsqueeze(0).expand_as(expanded_latents)], dim=-1)
            )

            # Update pixels
            pixels = pixels + residual

        return torch.clamp(pixels, -1, 1)

    def upsample(self, latents, target_size):
        """Simple upsampling from latents to pixel space."""
        return torch.nn.functional.interpolate(
            latents.permute(0, 2, 1).unsqueeze(-1),
            size=(target_size, 1),
            mode='bilinear',
            align_corners=False
        ).squeeze(-1).permute(0, 2, 1)
```

### Step 5: Joint Training with Reconstruction + Perceptual Loss

Optimize compressor and decoder jointly to preserve semantics across both directions.

```python
def train_step_unicom(
    compressor,
    transfusion,
    predictor,
    decoder,
    images,
    text_ids,
    lpips_model,
    optimizer
):
    """
    Unified training combining reconstruction and generation.
    """
    batch_size = images.shape[0]

    # Forward: image → compressed → text
    # Compress images
    visual_features = extract_visual_features(images)
    compressed = compressor(visual_features)

    # Backward: text → compressed → images
    # Predict latents from text
    text_embed = transfusion(text_ids, compressed)
    predicted_latents = predictor(text_embed.mean(dim=1))

    # Reconstruct pixels
    reconstructed = decoder(predicted_latents, images.shape[1] * images.shape[2])

    # Reconstruction loss: pixel MSE
    recon_loss = torch.nn.functional.mse_loss(reconstructed, images)

    # Perceptual loss: semantic similarity
    perceptual_loss = lpips_model(reconstructed, images).mean()

    # Combined loss
    total_loss = recon_loss + 0.1 * perceptual_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'perceptual_loss': perceptual_loss.item()
    }
```

## Practical Guidance

**When to Use:**
- Unified multimodal systems requiring both comprehension and generation
- Scenarios where visual grounding (spatial structure) matters
- Training efficiency is important (compressed representations reduce compute)
- Applications needing consistent semantic space across tasks

**When NOT to Use:**
- Ultra-high-resolution generation (compression may lose fine details)
- Tasks requiring extreme pixel accuracy
- Single-task systems (overhead of unified architecture not justified)

**Hyperparameter Tuning:**
- **output_dim (compression ratio)**: 64 good default; 32-128 depending on semantic richness needed
- **num_heads in compressor**: 8 standard; more heads for richer compression
- **diffusion_steps**: 20-50 balance quality and speed
- **latent_dim vs num_latents**: Trade spatial resolution vs feature dimensionality

**Common Pitfalls:**
- Compression too aggressive, losing important visual details
- Spatial downsampling instead of channel compression (harmful for grounding)
- Diffusion decoder undertrained relative to compressor
- Missing perceptual loss leading to semantically poor reconstructions

## Reference

[UniCom paper on arXiv](https://arxiv.org/abs/2603.10702)
