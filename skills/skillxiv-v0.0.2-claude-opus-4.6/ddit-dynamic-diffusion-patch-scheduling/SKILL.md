---
name: ddit-dynamic-diffusion-patch-scheduling
title: "DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.16968"
keywords: [diffusion models, efficiency optimization, dynamic inference, patch scheduling, image generation]
description: "Accelerate diffusion transformer inference by dynamically adjusting patch granularity during generation based on detail complexity at each timestep. Early denoising steps (establishing low-frequency structure) use coarse patches; later steps (adding high-frequency detail) use fine patches. Achieves 3.52× speedup on FLUX-1.Dev and 3.2× on video models while maintaining quality through variance-based adaptive scheduling."
---

# DDiT: Variance-Aware Adaptive Patch Sizing for Efficient Diffusion

Diffusion transformers achieve state-of-the-art generation quality but process every diffusion timestep with uniform computational cost. However, timesteps serve fundamentally different purposes: early steps establish global structure (low-frequency components), while late steps refine texture and detail (high-frequency). Using fixed patch granularity throughout is computationally wasteful—coarse patches suffice early, but fine patches become necessary only late in generation.

Traditional approaches use fixed patch sizes throughout inference. The challenge is determining when detail-level refinement becomes necessary without human tuning or expensive online measurements.

## Core Concept

DDiT uses the rate of change in the latent manifold as a proxy for detail complexity, enabling automatic scheduling decisions. The approach measures acceleration (third-order finite differences) of the latent representation over timesteps. High acceleration indicates rapid latent evolution and suggests detail generation is active; low acceleration indicates stable structure and allows coarser patches.

The system divides latents into spatial regions and compares acceleration variance against a threshold. Regions with low variance use large patch sizes; high-variance regions use fine patches.

## Architecture Overview

- **Variable Patch Support**: Modify patch embedding to support multiple granularities (p, 2p, 4p) using LoRA-style adaptation branches
- **Latent Tracker**: Monitor latent representations across recent timesteps to compute second and third-order finite differences
- **Variance Measurer**: Compute acceleration variance within spatial patches
- **Scheduler**: Compare variance to threshold; select patch size per region
- **Efficient Router**: Use coarse patches early, progressively refine as variance increases

## Implementation

Implement multi-scale patch support by adding LoRA branches to the patch embedding layer:

```python
class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.base_embedding = nn.Linear(3 * patch_size**2, dim)

        # LoRA branches for 2x and 4x coarser patches
        self.coarse_2x = nn.Sequential(
            nn.Linear(3 * (2 * patch_size)**2, dim // 2),
            nn.Linear(dim // 2, dim)
        )
        self.coarse_4x = nn.Sequential(
            nn.Linear(3 * (4 * patch_size)**2, dim // 2),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x, patch_scale=1):
        """
        x: (B, 3, H, W) image
        patch_scale: 1 (fine), 2 (medium), 4 (coarse)
        """
        if patch_scale == 1:
            # Standard patch embedding
            patches = self._extract_patches(x, self.patch_size)
            return self.base_embedding(patches)
        elif patch_scale == 2:
            patches = self._extract_patches(x, self.patch_size * 2)
            return self.coarse_2x(patches)
        elif patch_scale == 4:
            patches = self._extract_patches(x, self.patch_size * 4)
            return self.coarse_4x(patches)

    def _extract_patches(self, x, patch_size):
        """Extract non-overlapping patches from image."""
        B, C, H, W = x.shape
        patches = x.reshape(
            B, C, H // patch_size, patch_size,
            W // patch_size, patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            B, -1, C * patch_size * patch_size
        )
        return patches
```

Compute acceleration-based scheduling decisions:

```python
def compute_acceleration(latents_history, window_size=3):
    """
    Compute third-order finite differences to estimate acceleration.
    latents_history: list of (B, L, D) latent tensors from recent timesteps
    Returns: acceleration tensor indicating detail generation activity
    """
    if len(latents_history) < window_size:
        return None

    # Use recent window for differentiation
    recent = latents_history[-window_size:]

    # First differences
    delta_1 = recent[1] - recent[0]
    delta_2 = recent[2] - recent[1]

    # Second differences
    delta2_1 = delta_2 - delta_1

    # Third differences (acceleration)
    acceleration = torch.abs(delta2_1).mean(dim=-1)  # (B, L)
    return acceleration

def select_patch_schedule(acceleration, threshold=0.05):
    """
    Select patch scale per spatial region based on acceleration variance.
    acceleration: (B, num_spatial_patches) tensor
    threshold: variance threshold to trigger fine patches
    Returns: (B, num_spatial_patches) patch scale tensor [1, 2, or 4]
    """
    # Compute local variance in 3x3 spatial neighborhoods
    B, num_patches = acceleration.shape
    spatial_size = int(num_patches ** 0.5)

    acceleration_2d = acceleration.reshape(B, spatial_size, spatial_size)

    # Pad and compute local variance
    padded = torch.nn.functional.pad(
        acceleration_2d, (1, 1, 1, 1), mode='constant', value=0
    )

    local_variance = []
    for i in range(1, spatial_size + 1):
        for j in range(1, spatial_size + 1):
            window = padded[:, i - 1 : i + 2, j - 1 : j + 2]
            var = window.var(dim=(1, 2))
            local_variance.append(var)

    local_variance = torch.stack(local_variance, dim=1)

    # Assign patch scales: low variance → coarse patches (4x), high → fine (1x)
    patch_scale = torch.where(
        local_variance < threshold,
        torch.tensor(4, device=acceleration.device),
        torch.where(
            local_variance < 2 * threshold,
            torch.tensor(2, device=acceleration.device),
            torch.tensor(1, device=acceleration.device)
        )
    )

    return patch_scale
```

Integrate into diffusion inference loop:

```python
def efficient_diffusion_forward(
    model, x_noisy, timesteps, threshold=0.05
):
    """
    Run diffusion with dynamic patch scheduling.
    """
    latents_history = []

    for t in timesteps:
        # Embed with adaptive patches
        if len(latents_history) >= 3:
            acceleration = compute_acceleration(latents_history, window_size=3)
            patch_schedule = select_patch_schedule(acceleration, threshold)
        else:
            # Default to coarse patches early
            patch_schedule = torch.full(
                (x_noisy.shape[0], 256), 4, device=x_noisy.device
            )

        # Forward pass with scheduled patches
        latent = model(x_noisy, t, patch_schedule=patch_schedule)
        latents_history.append(latent.detach())

        # Denoise step
        x_noisy = x_noisy + (latent - x_noisy) * 0.1

    return x_noisy
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Acceleration threshold | 0.05 | Increase (0.1) for more aggressive coarsening; decrease (0.02) for quality focus |
| Variance window | 3×3 spatial | Use 2×2 for fast inference; 4×4 for high quality |
| History window | 3 timesteps | Minimum to compute acceleration reliably |
| Patch scales | [1, 2, 4] | Match base model patch size; adjust ratios per model |

**When to use**: For real-time image and video generation where latency is critical and you can afford modest quality trade-offs.

**When not to use**: When consistency across timesteps is paramount or for fine-detail artistic work where quality cannot be compromised.

**Common pitfalls**:
- Starting acceleration measurements too early (timesteps <20); initial chaos makes acceleration unstable
- Using spatially uniform patches; track per-patch variance to enable fine-grained control
- Setting threshold too high, causing all regions to use fine patches; calibrate on validation set

## Reference

DDiT achieves 3.52× speedup on FLUX-1.Dev and 3.2× on video models (HunyuanVideo, Wan-2.1) while maintaining comparable visual quality to full-detail inference. The approach is architecture-agnostic and applicable to any diffusion transformer.
