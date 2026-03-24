---
name: seacache-spectral-evolution-diffusion
title: "SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.18993"
keywords: [diffusion models, caching, acceleration, frequency domain, training-free]
description: "Accelerate diffusion models through spectral-evolution-aware caching. Exploit insight that early timesteps establish low-frequency structure while later timesteps refine high-frequency details. Apply FFT-based frequency filtering to feature cache decisions: preserve content-relevant frequency components while suppressing noise. Plug-and-play training-free enhancement achieving 1.5–2.5× speedup across FLUX, HunyuanVideo, Wan2.1 models."
---

# SeaCache: Frequency-Aware Feature Reuse in Diffusion Models

Diffusion models generate high-quality images through iterative refinement: starting from noise, they gradually add structure across timesteps. However, each timestep requires expensive forward passes through large models. Naively, every step must compute all features, but adjacent timesteps often share similar feature patterns—if properly measured, cache decisions can reuse features from nearby timesteps.

The challenge is determining when features are similar "enough" to cache. Pixel-level differences are noisy (stochastic variations don't indicate meaningful changes); frequency-domain similarity is more stable. Early timesteps focus on low-frequency structure (edges, composition); late timesteps add high-frequency detail (texture, fine structure).

## Core Concept

SeaCache leverages this frequency progression: early timesteps emphasize low-frequency components; late timesteps progressively incorporate higher frequencies. Rather than comparing raw features (noise-sensitive), the method:

1. Applies timestep-dependent frequency filters via FFT
2. Compares filtered features to identify meaningful similarity
3. Caches when features are similar in content-relevant frequency bands

The approach is training-free, plug-and-play, and applicable to any diffusion model.

## Architecture Overview

- **Spectral Evolution Analyzer**: Characterize frequency content across timesteps using FFT
- **SEA Filter Designer**: Compute optimal frequency response for each timestep (emphasize low freq early, high freq late)
- **Feature Transformer**: Apply frequency filtering to features before similarity comparison
- **Cache Decision**: Compare filtered features; cache if distance below threshold
- **Runtime Integration**: Intercept feature computation, check cache with SEA filtering

## Implementation

Analyze spectral evolution in diffusion to design optimal filters:

```python
import numpy as np
from scipy import fft

def analyze_spectral_evolution(noisy_trajectory, clean_image):
    """
    Analyze how frequency content changes across diffusion timesteps.
    noisy_trajectory: list of (B, C, H, W) tensors across timesteps
    clean_image: target (B, C, H, W) tensor
    Returns: frequency response per timestep
    """
    num_timesteps = len(noisy_trajectory)
    frequency_responses = []

    for t in range(num_timesteps):
        noisy = noisy_trajectory[t].detach().cpu().numpy()
        residual = clean_image.numpy() - noisy

        # FFT on spatial dimensions
        freq_domain = np.abs(fft.fft2(residual, axes=(2, 3)))

        # Average over batch and channels
        freq_magnitude = freq_domain.mean(axis=(0, 1))

        # Radial frequency average
        freq_radial = []
        h, w = freq_magnitude.shape
        for r in range(min(h, w) // 2):
            y, x = np.ogrid[:h, :w]
            mask = (x - w//2)**2 + (y - h//2)**2 <= (r+1)**2
            mask = mask & ((x - w//2)**2 + (y - h//2)**2 > r**2)
            if mask.sum() > 0:
                freq_radial.append(freq_magnitude[mask].mean())

        frequency_responses.append(freq_radial)

    return np.array(frequency_responses)

def design_sea_filters(frequency_responses, num_frequencies=32):
    """
    Design timestep-dependent frequency filters.
    Early timesteps: emphasize low frequencies
    Late timesteps: include high frequencies
    """
    num_timesteps = frequency_responses.shape[0]
    filters = np.zeros((num_timesteps, num_frequencies))

    for t in range(num_timesteps):
        # Normalized timestep (0=start, 1=end)
        normalized_t = t / max(1, num_timesteps - 1)

        # Create frequency filter: low-pass early, relaxed late
        # Using Gaussian roll-off
        for freq_idx in range(num_frequencies):
            # Normalized frequency (0=DC, 1=Nyquist)
            normalized_freq = freq_idx / num_frequencies

            # Cutoff frequency rises over time
            cutoff = 0.1 + normalized_t * 0.9

            # Gaussian filter centered at cutoff
            response = np.exp(-((normalized_freq - cutoff)**2) / (2 * 0.1**2))
            filters[t, freq_idx] = response

    return filters
```

Implement SEA-filtered feature comparison:

```python
def apply_sea_filter(features, sea_filter):
    """
    Apply frequency filter to features.
    features: (B, C, H, W) tensor
    sea_filter: (num_frequencies,) filter weights
    Returns: filtered (B, C, H, W) tensor
    """
    # FFT
    freq_domain = torch.fft.fft2(features, dim=(2, 3))
    freq_magnitude = torch.abs(freq_domain)

    # Apply filter: weight frequency components
    # Simplified: multiply magnitude by filter
    filtered_freq = freq_magnitude * torch.tensor(
        sea_filter[:freq_magnitude.shape[-1]], device=features.device
    )

    # Inverse FFT (reconstruct spatial domain)
    reconstructed = torch.fft.ifft2(
        torch.sign(freq_domain) * filtered_freq, dim=(2, 3)
    )
    reconstructed = torch.abs(reconstructed)

    return reconstructed

def seacache_similarity(feat_current, feat_cached, t, sea_filters, threshold=0.1):
    """
    Compute similarity using SEA filters.
    Returns True if features are similar enough to use cache.
    """
    # Get filter for current timestep
    sea_filter = sea_filters[t]

    # Apply filter
    feat_current_filtered = apply_sea_filter(feat_current, sea_filter)
    feat_cached_filtered = apply_sea_filter(feat_cached, sea_filter)

    # Normalize features
    feat_current_norm = feat_current_filtered / (feat_current_filtered.norm() + 1e-8)
    feat_cached_norm = feat_cached_filtered / (feat_cached_filtered.norm() + 1e-8)

    # Compute cosine distance
    distance = 1.0 - (feat_current_norm * feat_cached_norm).sum()

    return distance < threshold
```

Integrate into diffusion inference:

```python
class SeaCacheDiffusionModel:
    def __init__(self, base_model, sea_filters):
        self.model = base_model
        self.sea_filters = sea_filters
        self.feature_cache = {}
        self.cache_threshold = 0.15

    def forward(self, x, t):
        """
        Forward pass with SeaCache.
        x: (B, C, H, W) noisy image
        t: timestep scalar or tensor
        """
        # Check cache
        cache_key = (t.item() if hasattr(t, 'item') else t,)

        if cache_key in self.feature_cache:
            cached_features = self.feature_cache[cache_key]

            # Compute features for current input
            features = self._extract_features(x)

            # Compare using SEA similarity
            if self.seacache_similarity(
                features, cached_features, t.item() if hasattr(t, 'item') else t,
                self.sea_filters, self.cache_threshold
            ):
                # Use cached features
                return self._denoise_from_features(cached_features, t)

        # Cache miss: compute features normally
        output = self.model(x, t)

        # Cache features for future timesteps
        features = self._extract_features(x)
        self.feature_cache[cache_key] = features.detach()

        return output

    def _extract_features(self, x):
        """Extract intermediate layer features."""
        # Typically extract from penultimate layer
        return self.model.encoder(x)

    def _denoise_from_features(self, features, t):
        """Denoise using cached features."""
        return self.model.decoder(features, t)
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Cache threshold | 0.15 | Lower (0.10) for aggressive caching; higher (0.25) for conservative |
| Frequency bands | 32 | More bands (64) for fine-grained control; fewer (16) for speed |
| Filter rolloff | Gaussian σ=0.1 | Steeper (σ=0.05) for sharper frequency cutoff |
| Cache memory budget | 1 GB | Track cached features; clear oldest if exceeding budget |

**When to use**: For image and video diffusion models where multiple forward passes are expensive and you can tolerate modest quality trade-offs.

**When not to use**: For high-quality fine-detail generation where every feature computation matters; training-free approach may miss task-specific patterns.

**Common pitfalls**:
- Cache threshold too aggressive, reusing stale features; validate on validation set
- Filters not adapted to specific models; compute frequency analysis on training data for target model
- Memory leaks from unbounded cache; implement cache eviction (LRU) or size limits

## Reference

SeaCache achieves 1.5–2.5× speedup on FLUX-1.Dev, HunyuanVideo, and Wan-2.1 models in testing, with speedups higher on longer sequences. The approach is plug-and-play, training-free, and compatible with all diffusion architectures.
