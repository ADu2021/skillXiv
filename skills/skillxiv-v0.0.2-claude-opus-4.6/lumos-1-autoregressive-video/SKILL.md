---
name: lumos-1-autoregressive-video
title: "Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.08801"
keywords: [Video Generation, Autoregressive Models, Positional Embeddings, Diffusion Forcing]
description: "Generate videos autoregressively by extending LLM architectures to spatiotemporal data. MM-RoPE balances frequency spectra across temporal and spatial dimensions, while Autoregressive Diffusion Forcing enables efficient parallel decoding. Lumos-1 (0.5B-3B variants) matches or exceeds Show-o2 and COSMOS on text-to-video with training on 48 GPUs."
---

# Lumos-1: Unifying Video Generation Through Autoregressive LLMs

Video generation requires capturing temporal dynamics and spatial detail across multiple frames. Standard approaches treat frames independently (diffusion) or apply sequential RNNs (slow). Lumos-1 extends LLM architectures to video by addressing two fundamental challenges: (1) positional encodings designed for 1D text perform poorly on 3D video data, causing frequency imbalances, and (2) token-by-token generation is prohibitively slow. The solution is MM-RoPE (multimodal rotary position embeddings) for balanced spatiotemporal encoding, plus Autoregressive Diffusion Forcing for efficient parallel decoding. The result is a unified LLM-style model generating competitive video quality at tractable speed.

The key insight is that video is just another modality—positional encodings need careful frequency allocation across temporal and spatial axes, and generation can be parallelized through diffusion-style masking rather than sequential decoding.

## Core Concept

Lumos-1 combines two technical innovations:

1. **MM-RoPE**: Extends 1D rotary position embeddings to 3D (time, height, width) with distributed frequency allocation preventing temporal domination of the spectrum
2. **Autoregressive Diffusion Forcing (AR-DF)**: Parallel mask-based generation where multiple frames are generated simultaneously while maintaining temporal consistency through temporal tube masking

The model shares a single backbone (Llama-style) between text understanding and video generation, eliminating separate components.

## Architecture Overview

- **Llama Base Architecture**: Standard decoder-only transformer (0.5B to 3B variants)
- **MM-RoPE Positional Encoding**: 3D position embeddings balancing temporal, height, and width frequencies
- **Discrete Video Tokenizer**: Cosmos tokenizer compressing video into spatiotemporal patches (8×8×4 compression)
- **Unified Embedding**: Single token vocabulary for both text and video tokens
- **QK-Norm Stabilization**: Layer normalization on Q and K in attention (stabilizes large-scale training)
- **Parallel Diffusion Decoder**: Mask-based generation with temporal tube constraints (all frames in same spatial position share mask)
- **Multi-stage Pre-training**: Text→ image → image-to-video → text-to-video progression

## Implementation

The following demonstrates MM-RoPE and Autoregressive Diffusion Forcing:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MultimodalRotaryPositionEmbedding(nn.Module):
    """MM-RoPE: Balanced 3D positional encoding for video."""
    def __init__(self, hidden_dim: int = 4096, max_seq_len: int = 4096,
                 video_height: int = 512, video_width: int = 512, video_frames: int = 48):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.video_height = video_height
        self.video_width = video_width
        self.video_frames = video_frames

        # Compute inverse frequencies with balanced distribution
        # Standard 1D: inv_freq = 1 / (10000 ^ (i / d))
        # MM-RoPE: distribute frequencies across temporal (T), height (H), width (W)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Frequency scaling factors (compress ratio)
        # Text: standard 1D rope
        # Video: 3D rope with per-axis compression to balance spectrum
        self.time_compression = 0.5    # Reduce temporal frequencies
        self.space_compression = 1.0   # Keep spatial frequencies

    def forward(self, seq_positions: torch.Tensor,
                modality: str = "text") -> torch.Tensor:
        """
        Compute MM-RoPE embeddings.

        Args:
            seq_positions: (seq_len,) or (T, H, W) position indices
            modality: "text" (1D) or "video" (3D)

        Returns:
            freqs: (seq_len, hidden_dim) frequency embeddings
        """
        if modality == "text":
            # Standard 1D RoPE for text
            t = seq_positions.type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, hidden_dim//2)

        elif modality == "video":
            # 3D RoPE for video: combine temporal and spatial components
            if seq_positions.dim() == 1:
                # Flatten spatial indices to (T, H, W)
                seq_len = seq_positions.shape[0]
                total_spatial = self.video_height * self.video_width
                seq_positions = seq_positions.view(-1, total_spatial)

            T, spatial = seq_positions.shape
            H, W = self.video_height, self.video_width

            # Compute frequencies for each dimension separately
            t_indices = torch.arange(T, device=seq_positions.device).float()
            h_indices = torch.arange(H, device=seq_positions.device).float()
            w_indices = torch.arange(W, device=seq_positions.device).float()

            # Scale frequencies per dimension
            inv_freq_t = self.inv_freq * self.time_compression
            inv_freq_h = self.inv_freq * self.space_compression
            inv_freq_w = self.inv_freq * self.space_compression

            # Compute 3D frequency tensor
            # Expand inv_freq to match dimensionality
            freqs_t = torch.einsum("i,j->ij", t_indices, inv_freq_t[:self.hidden_dim//6])
            freqs_h = torch.einsum("i,j->ij", h_indices, inv_freq_h[:self.hidden_dim//6])
            freqs_w = torch.einsum("i,j->ij", w_indices, inv_freq_w[:self.hidden_dim//6])

            # Combine temporal and spatial components
            # Replicate across spatial grid
            freqs_t = freqs_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            freqs_h = freqs_h.unsqueeze(0).unsqueeze(-1).expand(T, -1, -1, W)
            freqs_w = freqs_w.unsqueeze(0).unsqueeze(0).expand(T, H, -1)

            freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=1)
            freqs = freqs.view(-1, self.hidden_dim // 2)

        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Compute cos and sin
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, hidden_dim)
        return emb.cos(), emb.sin()

class AutoregressiveDiffusionForcing(nn.Module):
    """Parallel generation with temporal tube masking."""
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 4096,
                 video_height: int = 512, video_width: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.video_height = video_height
        self.video_width = video_width

    def create_temporal_tube_mask(self, batch_size: int, seq_len: int,
                                  frames: int, mask_fraction: float = 0.5) -> torch.Tensor:
        """
        Create mask ensuring same spatial positions share mask pattern across time.

        Args:
            batch_size: Batch size
            seq_len: Total sequence length (frames * height * width)
            frames: Number of video frames
            mask_fraction: Fraction of tokens to mask

        Returns:
            mask: (batch, seq_len) binary mask (True = keep, False = mask)
        """
        spatial_dim = seq_len // frames
        h = w = int(spatial_dim ** 0.5)

        mask = torch.ones(batch_size, frames, h, w, dtype=torch.bool)

        # Temporal tube masking: mask same spatial position across all frames together
        num_spatial_positions = h * w
        num_to_mask = int(num_spatial_positions * mask_fraction)
        spatial_mask_indices = torch.randperm(num_spatial_positions)[:num_to_mask]

        for idx in spatial_mask_indices:
            i, j = idx // w, idx % w
            mask[:, :, i, j] = False

        return mask.view(batch_size, seq_len)

    def forward(self, embeddings: torch.Tensor, target_tokens: torch.Tensor,
                frames: int, mask_fraction: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive diffusion forcing: predict masked tokens in parallel.

        Args:
            embeddings: (batch, seq_len, hidden_dim) token embeddings
            target_tokens: (batch, seq_len) ground truth token ids
            frames: Number of video frames
            mask_fraction: Fraction to mask per generation step

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Mean squared error on masked positions
        """
        batch_size, seq_len, hidden_dim = embeddings.shape

        # Create mask (same spatial position masked across all frames)
        mask = self.create_temporal_tube_mask(batch_size, seq_len, frames, mask_fraction)

        # Predict all tokens in parallel (masked positions are targets)
        # In practice: pass through decoder
        logits = torch.randn(batch_size, seq_len, self.vocab_size)  # Placeholder

        # Compute loss only on masked positions (not observed)
        loss_mask = ~mask
        if loss_mask.sum() > 0:
            # Cross-entropy on masked positions
            loss = F.cross_entropy(
                logits[loss_mask],
                target_tokens[loss_mask],
                reduction='mean'
            )
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return logits, loss

class LumosVideoModel(nn.Module):
    """Unified LLM for text and video generation."""
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 4096,
                 num_layers: int = 32, num_heads: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Token embeddings (shared text+video)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)

        # MM-RoPE positional encoding
        self.rope = MultimodalRotaryPositionEmbedding(hidden_dim)

        # Transformer layers with QK-Norm stabilization
        self.layers = nn.ModuleList([
            LumosTransformerLayer(hidden_dim, num_heads, ff_dim=hidden_dim*4)
            for _ in range(num_layers)
        ])

        # Output projection
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Autoregressive diffusion forcing
        self.ar_df = AutoregressiveDiffusionForcing(vocab_size, hidden_dim)

    def forward(self, input_ids: torch.Tensor, modality: str = "text",
                frames: Optional[int] = None) -> torch.Tensor:
        """
        Generate text or video tokens.

        Args:
            input_ids: (batch, seq_len) token indices
            modality: "text" or "video"
            frames: Number of frames (only for video)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        embeddings = self.embed_tokens(input_ids)

        # Apply positional embeddings (MM-RoPE)
        cos, sin = self.rope(
            torch.arange(seq_len, device=input_ids.device),
            modality=modality
        )
        embeddings = apply_rotary_pos_emb(embeddings, cos, sin)

        # Forward through transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings)

        # Project to vocabulary
        logits = self.lm_head(embeddings)

        return logits

class LumosTransformerLayer(nn.Module):
    """Single transformer layer with QK-Norm."""
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # QK-Norm attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.qk_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with QK-Norm
        x_norm = self.norm1(x)
        q = self.qk_norm(x_norm)  # Normalize Q and K
        attn_out, _ = self.attention(q, x_norm, x_norm)
        x = x + attn_out

        # Feed-forward
        x = x + self.mlp(self.norm2(x))
        return x

def apply_rotary_pos_emb(embeddings: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to token embeddings."""
    # Simplified: rotate embedding dimensions
    dim = embeddings.shape[-1]
    x1 = embeddings[..., :dim//2]
    x2 = embeddings[..., dim//2:]

    rotated = torch.cat([
        x1 * cos[..., :dim//2] - x2 * sin[..., :dim//2],
        x1 * sin[..., dim//2:] + x2 * cos[..., dim//2:]
    ], dim=-1)

    return rotated

def train_lumos_step(model: LumosVideoModel, batch: dict,
                    optimizer: torch.optim.Optimizer,
                    modality: str = "video") -> float:
    """Single training step for Lumos-1."""
    optimizer.zero_grad()

    input_ids = batch['input_ids']
    target_ids = batch['target_ids']
    frames = batch.get('frames', 48)

    # Forward pass
    logits = model(input_ids, modality=modality, frames=frames)

    # Loss
    if modality == "video":
        # Autoregressive diffusion forcing
        _, loss = model.ar_df(
            model.embed_tokens(input_ids),
            target_ids,
            frames=frames
        )
    else:
        # Standard language modeling loss
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()
```

This implementation demonstrates MM-RoPE for balanced spatiotemporal encoding and parallel diffusion-forcing decoding.

## Practical Guidance

| Component | Value | Notes |
|-----------|-------|-------|
| **MM-RoPE Time Compression** | 0.3-0.5 | Reduce temporal frequencies; prevents domination |
| **MM-RoPE Space Compression** | 1.0 | Keep spatial frequencies; fine detail matters |
| **Temporal Tube Mask Ratio** | 0.3-0.5 | Mask 30-50% of spatial positions across all frames |
| **Video Frames** | 24-48 | Longer sequences = harder training; start at 24 |
| **Discrete Tokenizer** | Cosmos 8×8×4 | Compression ratio essential; 256 tokens = 16×16 spatial |
| **Batch Size** | 512 (on 48 GPUs) | Large batches improve training stability |
| **Learning Rate** | 1e-4 → 1e-5 | Decay over 500K steps |

### When to Use Lumos-1

- **Text-to-video generation**: Unified LLM approach leverages text understanding
- **Image-to-video prediction**: Temporal extension of image models without separate video modules
- **Efficient video synthesis**: Training on modest compute (48 GPUs, not 1K)
- **Multi-task video understanding**: Single model for generation and potentially understanding tasks
- **Research on autoregressive video**: Exploring LLM-style temporal modeling for video
- **Transfer from language models**: Reuse text-trained LLM weights; fine-tune sparingly

### When NOT to Use

- **Real-time video applications**: Generation speed still slower than diffusion baselines
- **Extreme spatial resolution** (>4K): Sequence length explodes; computational cost prohibitive
- **High-frequency motion**: Coarse temporal tokenization (8-frame intervals) loses fast motion
- **Models requiring architectural changes**: If you need adaptive computation or external memory, modify sparingly
- **Domain-specific quality**: Natural scene videos excellent; synthetic/art styles underexplored

### Common Pitfalls

1. **Imbalanced MM-RoPE**: If time_compression ≈ space_compression ≈ 1.0, temporal channels dominate spectrum. Always use time_compression = 0.3-0.5.
2. **Temporal Tube Mismatch**: If you mask different spatial positions at each timestep, temporal consistency breaks. Ensure same positions masked across all frames.
3. **Frame-wise Loss Dominance**: Later frames in sequence see lower loss due to information leakage from earlier frames. Counteract with frame-specific loss weighting: earlier frames weight 1.0, later frames weight 1.5.
4. **Tokenizer Bottleneck**: Cosmos tokenizer at 8×8×4 compression may be too aggressive. Validate reconstruction quality on sample videos before committing to training.
5. **Insufficient Pre-training**: Skipping text and image pre-training stages hurts video quality. Always follow staged pre-training: text → image → video.

## Reference

Xin, Z., Zhang, Y., et al. (2025). Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective. *arXiv preprint arXiv:2507.08801*.

Available at: https://arxiv.org/abs/2507.08801
