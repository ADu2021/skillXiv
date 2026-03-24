---
name: flow-map-trajectory-tilting
title: "Test-Time Scaling of Diffusions: Flow Maps as Look-Ahead Reward Operators"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2511.22688
keywords: [diffusion-models, test-time-scaling, flow-matching, reward-guidance, trajectory-prediction]
description: "Uses flow maps as look-ahead operators to enable principled reward-guided diffusion by predicting trajectory endpoints at any denoising step. Deploy when applying rewards or preferences to diffusion trajectories with meaningful gradients throughout generation."
---

## Summary

Test-Time Scaling of Diffusions introduces Flow Map Trajectory Tilting (FMTT), which leverages flow maps as "look-ahead" operators to enable principled reward-guided generation. Instead of heuristic denoiser approximations, FMTT uses the flow map to accurately predict where trajectories will land at any point during generation, enabling meaningful reward gradients throughout the process.

## Core Technique

**Flow Map as Predictor:** A flow map φ_t predicts the final output at any time step t. Instead of approximating gradients heuristically, FMTT uses the flow map to compute: "Where will this trajectory end up if I take a step toward this reward?" This enables exact reward signals throughout denoising.

**Trajectory Importance Weighting:** The importance weights for reward guidance reduce to integrating the reward along the flow map trajectory (Proposition 2.2). This principled weighting ensures early-stage trajectory decisions are appropriately penalized relative to later refinements.

**Reward Gradient Computation:** Compute dℒ/dz_t by backpropagating through the flow map prediction. The gradient tells you which direction in latent space improves the predicted final output under your reward function.

## Implementation

**Flow map training:** Train or load a pretrained flow matching model that predicts final outputs from any intermediate state. This serves as the look-ahead operator for all downstream reward guidance.

**Reward function integration:** Define reward r(x) over generated images. At step t during inference, use: gradient = ∇_{z_t} r(φ_t(z_t)) to guide trajectory updates.

**Importance weight computation:** Compute weight_t = integral of reward along trajectory from step t to final step. Steps whose decisions significantly impact final reward are weighted higher.

**Multi-step integration:** Use the weighted gradients to update trajectory: z_{t-1} = z_t - α * weight_t * gradient, where α is a temperature parameter controlling reward strength.

## When to Use

- Applying user preferences or quality rewards to diffusion generation
- Guidance tasks where heuristic approximations introduce artifacts
- Scenarios requiring stable, meaningful reward gradients throughout generation
- Quality-diversity trade-offs where intermediate trajectory decisions matter

## When NOT to Use

- Tasks where unconditional or simple classifier-based guidance suffices
- Scenarios without access to or ability to train flow maps
- Applications where reward computation is slower than diffusion steps
- Models where the flow map approximation is inaccurate or misaligned

## Key References

- Flow matching and optimal transport in generative models
- Trajectory prediction and look-ahead operators in diffusion
- Reward-guided generation and preference learning
