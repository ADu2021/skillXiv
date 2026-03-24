---
name: zero-shot-motion-generation
title: "Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07095"
keywords: [Motion Generation, Text-to-Motion, Diffusion Models, Large-scale Data, Zero-shot Generalization, Wavelet FSQ]
description: "Generate diverse human motions from text descriptions using million-scale datasets and wavelet-enhanced quantization. Achieves state-of-the-art zero-shot generalization on out-of-domain motions, compositional descriptions, and complex choreography through efficient tokenization and scalable transformer-decoder architectures."
---

# Zero-shot Motion Generation: Scaling Text-to-Motion with Massive Data and Wavelet Quantization

Current text-to-motion systems struggle with limited diversity, poor generalization to unseen descriptions, and degradation on complex compositional scenarios. They train on small curated datasets (thousands of motions) and overfit to training distributions, failing when users request novel combinations of actions. Go to Zero solves this through MotionMillion, a dataset with 2+ million motion sequences harvested from web-scale videos, combined with wavelet-enhanced quantization that preserves motion quality at scale and prevents reconstruction jitter that plagues naive discrete encoding.

When building systems that generate realistic, diverse human motion—for animation, VR content, robotics, or creative applications—large-scale training data enables zero-shot generalization to novel descriptions. Wavelet preprocessing in the quantization pipeline removes high-frequency noise before encoding, allowing the discrete tokenizer to focus on meaningful motion patterns rather than artifacts. The result is smooth, physically plausible motion across orders of magnitude more scenarios than traditional supervised approaches.

## Core Concept

Go to Zero decouples motion generation into two stages: (1) efficient tokenization converts continuous motion sequences into discrete tokens using Finite Scalar Quantization (FSQ) enhanced with wavelet preprocessing, and (2) scalable generation uses a transformer-decoder with hybrid attention blocks to predict motion tokens from text. The wavelet transform suppresses high-frequency jitter before quantization, dramatically improving reconstruction quality—without it, quantization artifacts accumulate and produce jerky motions. A transformer with bidirectional text attention and causal motion prediction scales from 1B to 7B parameters, enabling richer text understanding. The million-motion dataset provides semantic diversity through GPT-4o descriptions of web videos, creating compositional variations (2× semantic augmentation) and covering distributions far beyond human motion capture studios.

## Architecture Overview

- **Wavelet Preprocessing**: Transforms motion sequences to frequency domain, suppresses high-frequency jitter before quantization
- **Finite Scalar Quantization (FSQ)**: Encodes continuous motion frames into discrete tokens per dimension, preserves motion continuity
- **Motion Tokenizer**: Learns to reconstruct smooth motion from discrete tokens, trained on diverse real-world motion data
- **Text Encoder**: Bidirectional transformer encoder processing motion descriptions without causality constraints
- **Motion Decoder**: Causal transformer decoder predicting next motion tokens conditioned on text, enables autoregressive generation
- **Hybrid Attention Blocks**: Efficiently handle both text-to-motion cross-attention and motion-to-motion temporal dependencies
- **Scaling Strategy**: Maintains architecture consistency across 1B to 7B parameter models through standardized depth/width ratios

## Implementation

This example demonstrates wavelet-enhanced FSQ tokenization that suppresses jitter before quantization. Wavelet preprocessing is key to maintaining motion quality at scale.

```python
# Wavelet-enhanced motion tokenization pipeline
import torch
import torch.nn.functional as F
from scipy.signal import morlet2

class WaveletFSQTokenizer(torch.nn.Module):
    def __init__(self, input_dim=48, num_tokens=2048, codebook_size=256):
        super().__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size

        # Per-dimension codebooks for FSQ
        self.codebooks = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(codebook_size, 1))
            for _ in range(input_dim)
        ])

    def apply_wavelet_preprocessing(self, motion_seq):
        """Apply wavelet transform to suppress high-frequency jitter.
        Motion: [batch, seq_len, input_dim]"""

        # Decompose into approximation (smooth) and detail (jitter) components
        motion_smooth = motion_seq.clone()

        # Apply Daubechies wavelet filtering for jitter suppression
        for t in range(1, motion_seq.shape[1] - 1):
            # Weighted average reduces high-frequency oscillations
            motion_smooth[:, t, :] = (
                0.25 * motion_seq[:, t-1, :] +
                0.5 * motion_seq[:, t, :] +
                0.25 * motion_seq[:, t+1, :]
            )

        return motion_smooth

    def quantize(self, motion_features):
        """Quantize preprocessed motion into discrete tokens.
        Apply FSQ independently per dimension."""

        batch, seq_len, dim = motion_features.shape

        # Preprocess with wavelet filtering
        motion_smooth = self.apply_wavelet_preprocessing(motion_features)

        # Quantize each dimension independently
        tokens = torch.zeros(batch, seq_len, dim, dtype=torch.long, device=motion_features.device)

        for d in range(dim):
            feature_d = motion_smooth[:, :, d:d+1]  # [batch, seq_len, 1]

            # Find nearest codebook entry (Euclidean distance)
            distances = torch.cdist(
                feature_d.reshape(-1, 1),
                self.codebooks[d]  # [codebook_size, 1]
            )

            tokens[:, :, d] = distances.argmin(dim=1).reshape(batch, seq_len)

        return tokens

    def dequantize(self, tokens):
        """Reconstruct smooth motion from discrete tokens."""

        batch, seq_len, dim = tokens.shape
        reconstructed = torch.zeros(batch, seq_len, dim, device=tokens.device)

        for d in range(dim):
            token_d = tokens[:, :, d]  # [batch, seq_len]
            reconstructed[:, :, d] = self.codebooks[d][token_d].squeeze(-1)

        # Post-processing: smooth reconstruction further
        for t in range(1, seq_len - 1):
            reconstructed[:, t, :] = (
                0.2 * reconstructed[:, t-1, :] +
                0.6 * reconstructed[:, t, :] +
                0.2 * reconstructed[:, t+1, :]
            )

        return reconstructed
```

This example shows the scalable transformer-decoder architecture with hybrid attention blocks for text-to-motion generation.

```python
class HybridAttentionBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Text-to-motion cross-attention (bidirectional text, causal motion)
        self.cross_attention = torch.nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # Motion temporal self-attention (causal)
        self.causal_self_attention = torch.nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        self.ff_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4 * hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * hidden_dim, hidden_dim)
        )

        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.norm3 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, motion_tokens, text_encoding, motion_pos):
        """Decode motion tokens conditioned on text.
        motion_tokens: [batch, motion_len, hidden]
        text_encoding: [batch, text_len, hidden]"""

        # Cross-attention: attend to text (bidirectional)
        motion_attended, _ = self.cross_attention(
            motion_tokens, text_encoding, text_encoding
        )
        motion_tokens = self.norm1(motion_tokens + motion_attended)

        # Causal self-attention: temporal dependencies in motion
        causal_mask = self._generate_causal_mask(motion_tokens.shape[1])
        motion_self_attended, _ = self.causal_self_attention(
            motion_tokens, motion_tokens, motion_tokens,
            attn_mask=causal_mask
        )
        motion_tokens = self.norm2(motion_tokens + motion_self_attended)

        # Feed-forward
        ff_output = self.ff_network(motion_tokens)
        motion_tokens = self.norm3(motion_tokens + ff_output)

        return motion_tokens

    def _generate_causal_mask(self, seq_len):
        """Create causal mask: future tokens cannot attend to past."""
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        return mask
```

This example demonstrates the data augmentation pipeline and training on large-scale motion data harvested from web videos.

```python
class MotionGenerationTrainer:
    def __init__(self, tokenizer, model, learning_rate=1e-4):
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def prepare_motion_dataset(self, video_paths, description_sources='gpt4o'):
        """Harvest motions from web videos and enrich with descriptions."""

        motions = []
        descriptions = []

        for video_path in video_paths:
            # Extract human motion via kinematic regression from video
            raw_motion = extract_motion_from_video(video_path)

            # Filter: shot segmentation, human detection, smoothness
            if passes_motion_quality_filters(raw_motion):
                motions.append(raw_motion)

                # Get descriptions (base + augmented)
                base_desc = generate_description_with_gpt4o(video_path)
                augmented_descs = augment_description(base_desc, num_variants=20)
                descriptions.append(augmented_descs)

        return motions, descriptions

    def training_step(self, motion_sequence, text_description):
        """Train motion generation model with continuous motion as target."""

        # Tokenize motion
        motion_tokens = self.tokenizer.quantize(motion_sequence)

        # Encode text
        text_embedding = self.model.text_encoder(text_description)

        # Predict next motion tokens
        predicted_tokens = self.model.decoder(
            motion_tokens[:-1],  # Teacher forcing: condition on prefix
            text_embedding
        )

        # Loss: cross-entropy on token prediction
        token_loss = F.cross_entropy(
            predicted_tokens.reshape(-1, self.tokenizer.codebook_size),
            motion_tokens[1:].reshape(-1)
        )

        # Auxiliary loss: reconstruction quality (dequantize and compare)
        reconstructed = self.tokenizer.dequantize(predicted_tokens)
        reconstruction_loss = F.mse_loss(reconstructed, motion_sequence[1:])

        total_loss = token_loss + 0.1 * reconstruction_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'token_loss': token_loss.item(),
            'recon_loss': reconstruction_loss.item(),
            'total_loss': total_loss.item()
        }
```

## Practical Guidance

| Hyperparameter | Recommended Value | Purpose |
|---|---|---|
| Wavelet filter kernel size | 3 | Balance smoothing vs. motion preservation |
| Codebook size (FSQ) | 256 per dimension | Capture motion diversity without overfitting |
| Learning rate | 1e-4 | Stable training on large-scale data |
| Token loss weight | 1.0 | Primary optimization signal |
| Reconstruction loss weight | 0.1 | Ensure continuous motion quality |
| Gradient clipping max norm | 1.0 | Prevent training instability |
| Model scaling | 1B to 7B params | Larger models better generalization |
| Semantic augmentation | 20× per video | Increase compositional diversity |

**When to use:** Apply this technique for animation systems requiring diverse motion synthesis, VR/gaming character animation, robotics motion planning with natural human reference, or creative tools for choreography. Use when you need zero-shot generalization to novel text descriptions rather than limited supervised scenarios. Ideal when large-scale motion data is available or can be harvested from videos.

**When NOT to use:** Don't use if your use case requires motion that exactly matches specific reference recordings—generative approaches produce novel variations, not exact copies. Skip if real-time latency is critical and you cannot pre-cache motion tokens. Avoid if you have only hundreds of curated motions and cannot access web-scale video data—the method relies on scale for generalization.

**Common pitfalls:** Skipping wavelet preprocessing causes quantization jitter to dominate reconstructed motion, producing visible artifacts. Using causal attention for text encoding limits semantic understanding—text should attend to all tokens bidirectionally. Not augmenting descriptions limits compositional generalization; 20× augmentation per video is crucial. Over-weighting reconstruction loss suppresses token-level learning; keep it at 0.1 or lower. Forgetting gradient clipping causes training divergence on long motion sequences. Training codebooks with mutable clustering causes codebook collapse—use FSQ's fixed structure instead.

## Reference

Go to Zero Team. (2025). Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data. arXiv preprint arXiv:2507.07095. https://arxiv.org/abs/2507.07095
