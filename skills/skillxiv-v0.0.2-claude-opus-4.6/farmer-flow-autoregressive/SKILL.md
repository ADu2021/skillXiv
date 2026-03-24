---
name: farmer-flow-autoregressive
title: "FARMER: Flow AutoRegressive Transformer over Pixels"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.23588"
keywords: [Generative Model, Autoregressive, Flow Matching, Pixels, Likelihoods]
description: "Generates high-quality images directly from pixels using flow-matching-based latent sequences. Transforms images via invertible flows into manageable latent sequences, applies autoregressive modeling, and uses classifier-free guidance. Provides exact likelihood estimates and one-step distillation capabilities."
---

# FARMER: Flow Autoregressive Model for Pixel Generation

Direct pixel-space autoregressive modeling fails due to sequence length and dimensionality. FARMER uses invertible flows to compress images into latent sequences, applying autoregressive modeling to the compressed space.

The flow-based approach provides exact likelihoods and enables efficient distillation to one-step generators.

## Core Concept

Key innovation: **decompose image generation into flow compression + autoregressive modeling**:
- Invertible flow transforms images → manageable latent sequences
- Autoregressive transformer models latent sequence distribution
- Self-supervised dimension reduction identifies informative channels
- One-step distillation accelerates inference

## Architecture Overview

- Normalizing flow encoder (invertible)
- Channel pruning for dimension reduction
- Autoregressive sequence modeling in latent space
- Classifier-free guidance for quality control

## Implementation Steps

Implement an invertible flow that transforms high-dimensional images into sequences. Use coupling layers or masked autoencoders:

```python
class InvertibleImageFlow(nn.Module):
    def __init__(self, num_channels=3, flow_depth=8):
        super().__init__()

        # Stack of coupling layers for invertible transformation
        self.flow_layers = nn.ModuleList([
            CouplingLayer(num_channels, hidden_dim=128)
            for _ in range(flow_depth)
        ])

        # Learnable scaling for stability
        self.log_scale = nn.Parameter(torch.zeros(num_channels))

    def forward(self, images):
        """Transform images to latent sequences."""
        # images shape: (batch, 3, H, W)
        batch_size = images.shape[0]

        # Apply coupling layers for invertible transformation
        z = images
        log_det_jacobian = 0

        for layer in self.flow_layers:
            z, ldj = layer(z)
            log_det_jacobian += ldj

        # Reshape to sequence format
        # From (batch, C, H, W) to (batch, H*W, C)
        z_seq = z.permute(0, 2, 3, 1).reshape(batch_size, -1, z.shape[1])

        return z_seq, log_det_jacobian

    def inverse(self, z_seq):
        """Transform latent sequences back to images."""
        batch_size = z_seq.shape[0]
        num_channels = z_seq.shape[-1]
        H = W = int(np.sqrt(z_seq.shape[1]))

        # Reshape from sequence to spatial
        z = z_seq.reshape(batch_size, H, W, num_channels).permute(0, 3, 1, 2)

        # Inverse flow
        for layer in reversed(self.flow_layers):
            z = layer.inverse(z)

        return z
```

Implement channel-wise dimension reduction to identify which latent channels carry information:

```python
class ChannelSelector(nn.Module):
    def __init__(self, num_channels, reduction_ratio=0.5):
        super().__init__()
        self.num_keep = max(1, int(num_channels * reduction_ratio))

        # Learnable importance scores per channel
        self.channel_importance = nn.Parameter(torch.randn(num_channels))

    def select_channels(self, z_seq):
        """Keep only informative channels."""
        # Compute channel importance
        importance = torch.abs(self.channel_importance)

        # Select top-k important channels
        _, top_indices = torch.topk(importance, self.num_keep)
        top_indices = torch.sort(top_indices)[0]

        # Select channels
        z_reduced = z_seq[:, :, top_indices]

        return z_reduced, top_indices

    def expand_channels(self, z_reduced, top_indices, num_channels):
        """Restore full channel dimension."""
        batch_size, seq_len = z_reduced.shape[0], z_reduced.shape[1]

        # Create zero tensor for full channels
        z_full = torch.zeros(
            batch_size, seq_len, num_channels,
            device=z_reduced.device, dtype=z_reduced.dtype
        )

        # Fill selected channels
        z_full[:, :, top_indices] = z_reduced

        return z_full
```

Apply autoregressive modeling to the latent sequence. Standard transformer with causal masking:

```python
class LatentSequenceAR(nn.Module):
    def __init__(self, latent_dim, num_layers=8, num_heads=12):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Logit prediction for next token
        self.to_logits = nn.Linear(latent_dim, 256)  # Quantize to 256 levels

    def forward(self, z_seq):
        """Model distribution of latent sequence."""
        # Causal mask: can only attend to past tokens
        seq_len = z_seq.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        ).bool().to(z_seq.device)

        # Forward through transformer
        z_encoded = self.transformer(z_seq, src_mask=causal_mask)

        # Predict next token logits
        logits = self.to_logits(z_encoded)

        return logits
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Flow depth | 6-8 coupling layers (balance expressiveness and training) |
| Channel reduction | 50-75% (compress sequence length) |
| Transformer depth | 8-12 layers |
| Quantization levels | 256 (standard byte representation) |

**When to use:**
- High-quality image generation with exact likelihoods
- Applications requiring one-step distillation
- Scenarios where pixel-space modeling is needed
- Research requiring interpretable generative models

**When NOT to use:**
- Real-time generation (slower than diffusion)
- Very high-resolution images (memory constraints)
- When latency is critical (autoregressive generation is inherently slow)

**Common pitfalls:**
- Insufficient flow invertibility (numerical instability)
- Channel reduction too aggressive (information loss)
- Not using classifier-free guidance (lower quality)
- Training without proper log-determinant regularization

Reference: [FARMER on arXiv](https://arxiv.org/abs/2510.23588)
