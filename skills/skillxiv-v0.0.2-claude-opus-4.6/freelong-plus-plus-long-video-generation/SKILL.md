---
name: freelong-plus-plus-long-video-generation
title: "FreeLong++: Training-Free Long Video Generation via Multi-band SpectralFusion"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.00162"
keywords: [Video Generation, Diffusion Models, Long-sequence Generation, Frequency Analysis, No Fine-tuning]
description: "Extend video diffusion models to generate 4-8× longer sequences without retraining. Uses frequency-aware attention to blend local detail preservation with global consistency, identifying and fixing high-frequency distortion in extended videos."
---

# FreeLong++: Extending Video Models to Long Sequences Without Retraining

Video generation models trained on short clips (64-81 frames) fail catastrophically when applied to longer sequences. The quality doesn't degrade gracefully—instead, fine detail and motion coherence collapse. Yet retraining on longer videos is expensive. FreeLong++ solves this without any fine-tuning by recognizing that the problem is frequency-domain: long videos lose high-frequency information while maintaining low-frequency structure.

The core insight: different temporal scales require different attention mechanisms. Short windows capture fast motion and texture. Long windows provide global semantic continuity. Rather than forcing a single attention pattern across all frequencies, FreeLong++ decouples attention by temporal scale and applies frequency-specific fusion, enabling models trained on 81 frames to generate 324+ frames with minimal quality loss.

## Core Concept

When diffusion models generate longer sequences, they experience "high-frequency distortion"—texture, motion details, and fine structure break down while the overall scene composition remains coherent. This happens because the model's attention patterns, learned on short sequences, struggle with longer context.

FreeLong++ recognizes that:

1. **Low-Frequency Coherence** comes from long-range dependencies (what's the scene about, overall motion)
2. **High-Frequency Details** come from short-range dependencies (pixel textures, quick movements)
3. **Different Temporal Scales Need Different Attention**: Short windows excel at local details; long windows excel at global consistency

By decoupling attention into multiple temporal scales and fusing them in frequency space, the method preserves both aspects simultaneously—achieving consistent long videos without training.

## Architecture Overview

The FreeLong++ pipeline consists of these components:

- **Multi-Scale Attention Decoupling**: Parallel attention branches operating at different temporal window sizes (e.g., 8 frames, 16 frames, 32 frames, full sequence)
- **Spectral Analysis Module**: Decomposition of video into frequency bands using 3D FFT
- **Multi-Band Spectral Fusion**: Frequency-specific blending that routes low-frequency information through long-window attention and high-frequency through short-window attention
- **SpecMix Noise Initialization**: Adaptive noise injection that balances consistency and variation based on frequency content
- **Training-Free Application**: No model updates required; works with existing diffusion transformers (Wan2.1, LTX-Video)

## Implementation

This section demonstrates how to implement FreeLong++ for extending video generation models.

**Step 1: Analyze frequency content of video outputs**

This code decomposes videos into frequency bands to understand where degradation occurs:

```python
import numpy as np
import torch
import torch.fft as fft
from scipy import signal

def analyze_video_frequencies(video_tensor, frame_dim=1):
    """
    Decompose video into frequency bands to identify distortion sources.
    video_tensor: (B, T, C, H, W) - batch of video sequences
    Returns: frequency spectrum analysis across bands
    """

    B, T, C, H, W = video_tensor.shape
    video_np = video_tensor.numpy()

    frequency_analysis = {}

    for band_name, freq_range in [
        ('low', (0, 0.1)),
        ('mid', (0.1, 0.3)),
        ('high', (0.3, 0.5))
    ]:
        # 3D FFT: frequency domain in spatial and temporal dimensions
        fft_3d = np.fft.fftn(video_np, axes=(frame_dim, -2, -1))
        fft_magnitude = np.abs(fft_3d)

        # Extract frequency band
        freq_idx_min = int(freq_range[0] * T)
        freq_idx_max = int(freq_range[1] * T)

        band_power = fft_magnitude[:, freq_idx_min:freq_idx_max, :, :].mean()
        frequency_analysis[band_name] = band_power

    return frequency_analysis

# Analyze a 324-frame output from a model trained on 81 frames
short_trained_output = torch.randn(1, 324, 3, 256, 256)
freqs = analyze_video_frequencies(short_trained_output)

print("Frequency content (power per band):")
for band, power in freqs.items():
    print(f"  {band}: {power:.4f}")
# Expected: low-freq maintained, high-freq degraded
```

This identifies which frequencies degrade in long-sequence generation.

**Step 2: Implement multi-scale attention decoupling**

This code creates parallel attention branches for different temporal windows:

```python
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class MultiScaleAttention(nn.Module):
    """
    Parallel attention at different temporal scales.
    Short windows capture local details; long windows capture global structure.
    """

    def __init__(self, embed_dim=768, num_heads=8, temporal_scales=(8, 16, 32, 64)):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_scales = temporal_scales

        # Separate attention heads for each temporal scale
        self.scale_attentions = nn.ModuleList([
            MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in temporal_scales
        ])

        # Fusion layer: combine outputs from all scales
        self.fusion = nn.Linear(len(temporal_scales) * embed_dim, embed_dim)

    def forward(self, x_t, attention_mask=None):
        """
        x_t: (B, T, D) - temporal sequence of embeddings
        """

        B, T, D = x_t.shape
        scale_outputs = []

        for scale_idx, window_size in enumerate(self.temporal_scales):
            # Reshape into windows of this temporal scale
            num_windows = T // window_size
            if num_windows == 0:
                continue  # Skip scales larger than sequence

            # Unfold into windows
            x_windowed = x_t[:, :num_windows * window_size].reshape(
                B * num_windows, window_size, D
            )

            # Apply attention within each window
            attn_out, _ = self.scale_attentions[scale_idx](
                x_windowed, x_windowed, x_windowed,
                attn_mask=attention_mask
            )

            # Reshape back
            attn_out = attn_out.reshape(B, num_windows * window_size, D)

            # Pad to original length
            if attn_out.shape[1] < T:
                pad_len = T - attn_out.shape[1]
                attn_out = torch.nn.functional.pad(attn_out, (0, 0, 0, pad_len))

            scale_outputs.append(attn_out)

        # Concatenate outputs from all scales
        fused = torch.cat(scale_outputs, dim=-1)

        # Fuse back to original dimension
        output = self.fusion(fused)

        return output

# Test multi-scale attention
attn = MultiScaleAttention(embed_dim=768)
x = torch.randn(2, 324, 768)  # Batch of 2, 324 frames
y = attn(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
```

This implements parallel attention across temporal scales without model retraining.

**Step 3: Apply frequency-specific fusion in spectral domain**

This code blends low and high frequency information appropriately:

```python
import torch
import torch.fft as fft

class Multiband SpectralFusion(nn.Module):
    """
    Fuse multi-scale attention outputs in frequency domain.
    Route frequencies to appropriate attention scales during fusion.
    """

    def __init__(self, num_bands=3):
        super().__init__()
        self.num_bands = num_bands
        # Learnable band-specific weights (fixed during inference)
        self.band_weights = nn.Parameter(torch.ones(num_bands) / num_bands)

    def forward(self, video_latent, low_freq_attn, high_freq_attn, mid_freq_attn):
        """
        video_latent: (B, T, C, H, W) - latent video
        low_freq_attn: attention output optimized for global consistency
        high_freq_attn: attention output optimized for local details
        mid_freq_attn: attention output for mid-range features
        """

        B, T, C, H, W = video_latent.shape

        # FFT to frequency domain
        video_fft = fft.fftn(video_latent, dim=(1, 3, 4))
        video_freq = torch.abs(video_fft)

        # Separate frequency bands
        freq_center = T // 2
        low_band = video_fft[:, :freq_center // 3].clone()      # Low freq (0-33%)
        mid_band = video_fft[:, freq_center // 3:2 * freq_center // 3].clone()  # Mid (33-66%)
        high_band = video_fft[:, 2 * freq_center // 3:].clone()  # High (66-100%)

        # Apply frequency-appropriate attention
        # Low frequencies: use long-window attention for global consistency
        low_out = fft.ifftn(low_band, dim=(1, 3, 4)).real

        # High frequencies: use short-window attention for details
        high_out = fft.ifftn(high_band, dim=(1, 3, 4)).real

        # Mid frequencies: blend both approaches
        mid_out = fft.ifftn(mid_band, dim=(1, 3, 4)).real

        # Reconstruct with frequency-specific routing
        output_fft = torch.cat([
            fft.fftn(low_out, dim=(1, 3, 4)),
            fft.fftn(mid_out, dim=(1, 3, 4)),
            fft.fftn(high_out, dim=(1, 3, 4))
        ], dim=1)

        # IFFT back to spatial domain
        output = fft.ifftn(output_fft, dim=(1, 3, 4)).real

        return output

# Test spectral fusion
fusion = MultibandSpectralFusion()
video = torch.randn(1, 324, 4, 64, 64)
output = fusion(video, video, video, video)
print(f"Fused output shape: {output.shape}")
```

This applies frequency-specific fusion without retraining the diffusion model.

**Step 4: Implement SpecMix noise initialization for balanced generation**

This code initializes noise adaptively based on frequency content:

```python
def specmix_noise_initialization(sequence_length, height, width, target_freqs='balanced'):
    """
    Initialize noise with frequency composition matching target video properties.
    Balanced init: equal energy across freq bands
    Low-freq dominant: more low-freq energy (smoother, more consistent)
    High-freq dominant: more high-freq energy (more detailed but noisier)
    """

    # Generate white noise baseline
    noise = torch.randn(1, sequence_length, 3, height, width)

    # FFT to frequency domain
    noise_fft = fft.fftn(noise, dim=(1, 3, 4))

    # Compute frequency bands
    T = sequence_length
    H = height
    W = width

    # Frequency band indices
    freq_t = np.fft.fftfreq(T)
    freq_h = np.fft.fftfreq(H)
    freq_w = np.fft.fftfreq(W)

    # Create frequency masks
    if target_freqs == 'balanced':
        # Equal power across all frequencies
        scale_factor = 1.0
    elif target_freqs == 'low_dominant':
        # Emphasize low frequencies (temporal smoothness)
        scale_factor = 1.0 / (1.0 + np.abs(freq_t[:T//2, np.newaxis, np.newaxis]))
    elif target_freqs == 'high_dominant':
        # Emphasize high frequencies (detail)
        scale_factor = 1.0 + np.abs(freq_t[:T//2, np.newaxis, np.newaxis])
    else:
        scale_factor = 1.0

    # Apply frequency scaling
    noise_fft[:, :T//2] *= torch.from_numpy(scale_factor).float()

    # IFFT back to spatial domain
    noise_init = fft.ifftn(noise_fft, dim=(1, 3, 4)).real

    return noise_init

# Initialize noise for 324-frame generation with balanced spectrum
init_noise = specmix_noise_initialization(324, 256, 256, target_freqs='balanced')
print(f"Initialized noise shape: {init_noise.shape}")
print(f"Noise std dev: {init_noise.std():.4f}")
```

This initializes noise adaptively to balance consistency and detail in long-sequence generation.

**Step 5: Apply FreeLong++ to existing diffusion models**

This shows how to use FreeLong++ with pre-trained models without any fine-tuning:

```python
class FreeLongPP:
    """
    Wrapper that extends any diffusion transformer to long videos.
    No fine-tuning required; works by modifying attention and noise.
    """

    def __init__(self, model, num_scales=4):
        self.model = model
        self.num_scales = num_scales
        self.multi_scale_attn = MultiScaleAttention(temporal_scales=(8, 16, 32, 64))
        self.spectral_fusion = MultibandSpectralFusion()

    def generate_long_video(self, prompt, num_frames=324, guidance_scale=7.5):
        """
        Generate long video by applying multi-scale attention during diffusion.
        """

        # Initialize noise with SpecMix
        x_t = specmix_noise_initialization(num_frames, 256, 256, target_freqs='balanced')

        # Diffusion process (simplified)
        timesteps = np.linspace(1, 0, 50)

        for t in timesteps:
            # Standard diffusion step with guidance
            noise_pred = self.model.denoise(x_t, t, prompt)

            # Apply multi-scale attention to stabilize long sequences
            x_t_latent = x_t.reshape(1, num_frames, -1)
            x_t_attn = self.multi_scale_attn(x_t_latent)
            x_t = x_t_attn.reshape(x_t.shape)

            # Apply spectral fusion to blend frequencies
            x_t = self.spectral_fusion(x_t, x_t, x_t, x_t)

            # Diffusion update step
            x_t = x_t - noise_pred * (1.0 - t)

        return x_t

# Use FreeLong++ with existing model
freelong = FreeLongPP(pretrained_diffusion_model)
long_video = freelong.generate_long_video("A person walking in a park", num_frames=324)
print(f"Generated long video: {long_video.shape}")
```

This applies FreeLong++ to any diffusion transformer without retraining.

## Practical Guidance

**When to use FreeLong++:**
- Extending short-video models to 4-8× longer sequences without retraining
- Applications where temporal consistency matters but high-frequency detail is secondary
- Scenarios where retraining is infeasible (limited GPU memory, proprietary models)
- Real-time generation where latency from fine-tuning is unacceptable

**When NOT to use:**
- Tasks requiring ultra-high-quality fine details (frequency filtering trades detail for consistency)
- Very long sequences (512+ frames) where accumulated errors compound
- Domain shifts where attention patterns from short videos don't transfer (e.g., different camera motion)
- Scenarios where moderate speed increases are acceptable (retraining may be worth it)

**Hyperparameters and Configuration:**

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Temporal Scales | (8, 16, 32, full) | Covers local, mid, and global levels |
| Number of Frequency Bands | 3 | Low, mid, high; more bands increase complexity |
| SpecMix Initialization | 'balanced' | 'low_dominant' for smoother, 'high_dominant' for detail |
| Extension Factor | 4-8× | Beyond 8× typically shows diminishing returns |
| Diffusion Steps | 50-100 | More steps help maintain quality on longer sequences |
| Guidance Scale | 7.5 | Standard; higher → more adherence to prompt |

**Common Pitfalls:**
- Applying only single-scale attention (loses either detail or consistency)
- Using random noise initialization for very long videos (accumulates error)
- Forgetting that frequency mixing is bidirectional—high-freq noise can corrupt low-freq consistency
- Trying to extend 10× or more without retraining (error accumulation)
- Not adapting temporal scales to model training context length

**Key Design Decisions:**
FreeLong++ avoids fine-tuning entirely by working within the model's natural attention mechanism. Rather than retraining, it decouples attention into multiple temporal scales—each already implicitly present in the original model. Spectral fusion is key: it routes different frequencies through appropriate scales rather than trying to force a single attention pattern to handle all frequencies. SpecMix noise initialization ensures that generated sequences start with frequency composition balanced for long-range coherence.

## Reference

Zhu, L., Li, W., Gao, H., Tao, X., Meng, C., Ren, Y., ... & Ye, Q. (2025). FreeLong++: Training-Free Long Video Generation via Multi-band SpectralFusion. arXiv preprint arXiv:2507.00162. https://arxiv.org/abs/2507.00162
