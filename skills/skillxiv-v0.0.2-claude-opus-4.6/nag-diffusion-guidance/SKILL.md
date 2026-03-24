---
name: nag-diffusion-guidance
title: "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.21179"
keywords: [diffusion models, negative guidance, attention, few-step sampling, image generation]
description: "Apply training-free negative guidance in diffusion models by extrapolating in attention space with L1-based normalization, restoring suppression of unwanted attributes across architectures and modalities."
---

# Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models

## Core Concept

Normalized Attention Guidance (NAG) addresses the breakdown of negative guidance (suppressing unwanted attributes) in diffusion models, particularly in aggressive few-step sampling regimes. While Classifier-Free Guidance (CFG) works well for positive guidance, negative guidance becomes unstable as sampling steps decrease.

NAG introduces a training-free mechanism that performs extrapolation in attention space with L1-based normalization and refinement. By operating at the attention level rather than the latent level, NAG maintains effectiveness across diverse architectures (UNet, DiT), sampling strategies, and modalities (image, video) without requiring model retraining or architecture modifications.

## Architecture Overview

- **Attention-Space Extrapolation**: Extract attention maps from unconditional paths and extrapolate them to suppress unwanted patterns
- **L1-Based Normalization**: Normalize attention extrapolations using L1 distance metrics for stability
- **Refinement Pipeline**: Apply post-processing to smooth and validate guidance signals
- **Plug-in Integration**: Works with any diffusion architecture by intercepting attention computations
- **Multi-Modal Support**: Applies uniformly to image, video, and other modalities
- **Minimal Computational Overhead**: Negligible impact on inference speed compared to standard sampling

## Implementation

The following steps outline how to integrate NAG into a diffusion sampling pipeline:

1. **Extract baseline attention maps** - Run unconditional generation and capture intermediate attention states
2. **Compute guidance direction** - Calculate L1-normalized difference between conditional and unconditional paths
3. **Perform attention extrapolation** - Extrapolate guidance signals in attention space beyond typical guidance scales
4. **Apply refinement** - Smooth and validate the extrapolated attention to prevent artifacts
5. **Integrate into sampling loop** - Inject guidance at appropriate denoising steps
6. **Monitor quality** - Track text alignment and fidelity metrics during generation

```python
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

class AttentionGuidance:
    def __init__(self, guidance_scale: float = 7.5, l1_epsilon: float = 1e-6):
        self.guidance_scale = guidance_scale
        self.l1_epsilon = l1_epsilon

    def extract_attention_maps(self, model_output: Dict) -> Dict[str, torch.Tensor]:
        """Extract attention maps from model intermediate layers."""
        attention_maps = {}
        for name, module in model_output.items():
            if 'attention' in name:
                attention_maps[name] = module.detach()
        return attention_maps

    def compute_l1_normalized_guidance(self, uncond_attn: torch.Tensor,
                                       cond_attn: torch.Tensor) -> torch.Tensor:
        """Compute L1-normalized guidance direction."""
        diff = cond_attn - uncond_attn
        l1_norm = torch.norm(diff, p=1, dim=-1, keepdim=True)
        l1_norm = torch.clamp(l1_norm, min=self.l1_epsilon)
        normalized_guidance = diff / l1_norm
        return normalized_guidance

    def extrapolate_guidance(self, guidance: torch.Tensor) -> torch.Tensor:
        """Extrapolate guidance in attention space."""
        extrapolated = guidance * self.guidance_scale
        return extrapolated

    def refine_attention(self, attention: torch.Tensor, smoothing_window: int = 3) -> torch.Tensor:
        """Refine attention maps to prevent artifacts."""
        # Apply spatial smoothing
        kernel = torch.ones(1, 1, smoothing_window, smoothing_window) / (smoothing_window ** 2)
        kernel = kernel.to(attention.device)
        refined = F.conv2d(attention.unsqueeze(0), kernel, padding=smoothing_window//2)
        refined = refined.squeeze(0)

        # Clip to valid range
        refined = torch.clamp(refined, 0, 1)
        return refined

    def apply_guidance(self, model_latents: torch.Tensor, uncond_attn: Dict[str, torch.Tensor],
                      cond_attn: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply NAG to sampling step."""
        guided_latents = model_latents.clone()

        for name in uncond_attn.keys():
            guidance = self.compute_l1_normalized_guidance(uncond_attn[name], cond_attn[name])
            guidance = self.extrapolate_guidance(guidance)
            guidance = self.refine_attention(guidance)

        return guided_latents
```

## Practical Guidance

**Hyperparameters to tune:**
- **Guidance scale** (3.0-15.0): Controls strength of negative suppression. Higher values suppress more aggressively; 7.5 is a good starting point
- **L1 epsilon** (1e-8 to 1e-4): Numerical stability threshold; prevent division by zero during normalization
- **Refinement smoothing window** (1-5): Kernel size for spatial smoothing; larger windows prevent fine artifacts but may lose detail
- **Extrapolation schedule**: Vary guidance strength across denoising steps; stronger guidance early, lighter late

**When to use:**
- Few-step diffusion sampling (4-8 steps) where negative guidance typically fails
- Generating images with strong negative prompts or attribute suppression
- Video generation where unwanted artifacts need suppression
- Multi-modal generation (image+text) with complex constraints

**When NOT to use:**
- High-step sampling (50+ steps) where standard CFG already works effectively
- Purely generative tasks without negative constraints
- Real-time applications requiring minimal inference overhead
- Tasks where positive guidance alone is sufficient

**Common pitfalls:**
- **Over-guidance**: Guidance scale too high causes color shifts, distortions, or unrelated artifacts
- **Attention mismatch**: Extracting attention from wrong layers or misaligned timesteps reduces effectiveness
- **Insufficient refinement**: Skipping smoothing can introduce checkerboard patterns or local artifacts
- **Step mismatch**: Applying guidance at wrong denoising steps reduces suppression effectiveness

## Reference

NAG is a universally applicable, training-free method that restores effective negative guidance across diverse architectures and sampling strategies. The paper demonstrates improvements in text-image alignment, perceptual quality, and user preference compared to standard negative guidance approaches.

Original paper: "Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models" (arxiv.org/abs/2505.21179)
