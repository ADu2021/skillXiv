---
name: world-models-calibrated-uncertainty
title: "World Models That Know When They Don't Know - Controllable Video Generation with Calibrated Uncertainty"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.05927
keywords: [video generation, uncertainty quantification, calibrated confidence, robot planning, hallucination detection]
description: "Quantify confidence in video generation by estimating latent-space uncertainty and mapping to interpretable heatmaps. Detect untrustworthy regions in generated frames for robot planning and policy evaluation—critical when video hallucinations could cause real-world failures."
---

## Overview

The C3 framework enables video models to estimate confidence levels at subpatch granularity through latent-space uncertainty estimation and interpretable visualization. Rather than pixel-space uncertainty (which is training-unstable), this approach estimates uncertainty in latent space and maps it to high-resolution uncertainty heatmaps.

## When to Use

- Video generation for robotics applications where hallucinations risk failure
- Identifying physically unrealistic video regions before using for planning
- Out-of-distribution detection in video models
- Scenarios requiring confidence scores for downstream task reliability
- Applications needing interpretable uncertainty visualization

## When NOT to Use

- Cases where single-pass generation is sufficient without confidence measurement
- Tasks that don't require understanding failure modes
- Training on limited data where calibration is difficult
- Real-time applications with strict latency requirements

## Core Technique

Latent-space uncertainty with proper calibration and visualization:

```python
# Uncertainty quantification for video generation
class CalibratedVideoUncertainty:
    def __init__(self, vae_model, diffusion_model):
        self.vae = vae_model
        self.diffusion = diffusion_model
        # Uncertainty predictor in latent space
        self.uncertainty_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def generate_with_uncertainty(self, prompt, actions):
        """
        Generate video with per-patch confidence estimates.
        Returns generated frames and uncertainty heatmaps.
        """
        # Generate in latent space
        latent_video = self.diffusion.sample(prompt, actions)

        # Estimate uncertainty for each latent patch
        patch_uncertainty = self.uncertainty_head(latent_video)

        # Train using strictly proper scoring rules
        # ensures calibration across correctness and confidence
        calibrated_scores = self.apply_proper_scoring_loss(
            patch_uncertainty
        )

        # Decode to pixel space
        pixel_video = self.vae.decode(latent_video)

        # Map latent uncertainties to RGB-space heatmaps
        rgb_uncertainty = self.latent_to_pixel_uncertainty(
            patch_uncertainty
        )

        return pixel_video, rgb_uncertainty

    def apply_proper_scoring_loss(self, predictions):
        """
        Train via strictly proper scoring rules for both
        correctness and calibration.
        """
        # Ensures predicted confidence matches actual accuracy
        loss = self.proper_scoring_rule(predictions)
        return loss

    def latent_to_pixel_uncertainty(self, latent_uncertainty):
        """Convert latent uncertainties to RGB heatmaps."""
        # Upsample and visualize uncertainty
        heatmap = torch.nn.functional.interpolate(
            latent_uncertainty,
            scale_factor=self.vae.scale_factor,
            mode='bilinear'
        )
        # Map to RGB colormap
        rgb_heatmap = self.uncertainty_to_rgb(heatmap)
        return rgb_heatmap
```

Validation on robotics datasets (Bridge, DROID) demonstrates in-distribution calibration and out-of-distribution detection capabilities.

## Key Results

- Dense confidence estimation at subpatch level
- High-resolution uncertainty heatmaps identifying untrustworthy regions
- Training stability through latent-space estimation
- Validated on large-scale robotics datasets

## Implementation Notes

- Strictly proper scoring rules ensure calibration
- Latent-space estimation avoids pixel-space training instability
- Per-patch granularity enables fine-grained confidence maps
- Uncertainty scores guide downstream task planning

## References

- Original paper: https://arxiv.org/abs/2512.05927
- Focus: Uncertainty quantification in video generation
- Domain: Generative modeling, robotics, calibration
