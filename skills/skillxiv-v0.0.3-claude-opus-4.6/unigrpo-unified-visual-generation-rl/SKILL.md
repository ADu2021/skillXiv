---
name: unigrpo-unified-visual-generation-rl
title: "UniGRPO: Unified Policy Optimization for Interleaved Text-Image Generation"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.23500"
keywords: [Flow Matching, GRPO, Velocity Regularization, CFG Elimination, Multimodal RL]
description: "Replace classifier-free guidance and KL-based regularization in flow matching with velocity-based MSE regularization and GRPO for joint text-image optimization. Achieves 0.8381 TextAlign and 0.90 GenEval without CFG overhead. Works best for multimodal generation where text and image must stay coherent. Trigger: When optimizing vision-language models with flow matching and need better joint text-image policy."
category: "Component Innovation"
---

## What This Skill Does

Replace two components in multimodal flow-matching models: (1) swap classifier-free guidance (CFG) with direct GRPO optimization during training, and (2) replace KL regularization on latent spaces with MSE regularization on velocity fields. Enables joint text-image policy learning without training-inference mismatch.

## Problems with Prior Approach

**Problem 1: Classifier-Free Guidance Creates Training-Inference Gap**
- CFG requires branching computation during inference (unconditioned + conditioned samples)
- During training with CFG, multiple rollouts per prompt increase computational cost
- Gradient graph complexity from branched rollouts complicates GRPO scaling to multi-turn sequences

**Problem 2: KL Regularization on Latents is Crude**
- Standard KL penalty applies uniform constraints across all noise levels
- High-noise steps (where model has more freedom) and low-noise steps (where precision matters) get same penalty
- Enables reward hacking: model exploits high-noise regions where KL is loose

The paper's insight: Remove CFG entirely during training (use direct GRPO instead), and apply noise-aware regularization directly on velocity fields rather than latent distributions.

## The Swap: Two-Part Component Replacement

**Swap 1: CFG → Direct GRPO**

```python
# Prior approach: Classifier-free guidance during training
def train_with_cfg(prompts, model, num_rollouts=4):
    """
    CFG requires branched rollouts:
    - Unconditional sample (for guidance scale)
    - Conditional sample (actual generation)
    Doubles forward passes; complicates gradient estimation
    """
    unconditional_outputs = []
    conditional_outputs = []
    for prompt in prompts:
        # Unconditional branch
        unc_sample = model.sample(prompt=None)
        unconditional_outputs.append(unc_sample)

        # Conditional branch
        cond_sample = model.sample(prompt=prompt)
        conditional_outputs.append(cond_sample)

    # CFG interpolation (post-hoc, inference-only)
    # During training: gradient computation through two branches complicates GRPO
    return mix_guidance(conditional_outputs, unconditional_outputs, scale=7.5)

# New approach: Direct GRPO without CFG
def train_with_direct_grpo(prompts, model):
    """
    Single forward pass per prompt; GRPO directly optimizes
    conditional distribution without branched computation.
    Linear rollouts enable multi-turn sequence optimization.
    """
    samples = []
    for prompt in prompts:
        sample = model.sample(prompt=prompt)  # One rollout per prompt
        samples.append(sample)

    # GRPO applied directly to conditional policy
    # No branching; gradients flow through single path
    advantages = compute_group_relative_advantages(samples)
    loss = -advantages.mean()
    return loss
```

Key difference: CFG trades two samples for guidance effect; direct GRPO uses single sample with reward signal to optimize conditioned policy. During training, CFG branching complicates gradient graphs; GRPO's linear structure scales better.

**Swap 2: KL Regularization → Velocity-Based MSE**

```python
# Prior: KL penalty on latent distributions (uniform across noise levels)
def kl_regularization(latents_pred, latents_ref, kl_weight=0.001):
    """
    Applies same KL penalty regardless of noise level.
    High noise: KL loose (model has freedom) → reward hacking
    Low noise: KL tight (need precision) → same penalty as high noise
    """
    kl_div = compute_kl(latents_pred, latents_ref)
    return kl_weight * kl_div.mean()

# New: MSE on velocity fields (noise-aware, direct flow)
def velocity_regularization(velocity_pred, velocity_ref, mse_weight=0.001):
    """
    MSE applied to velocity field (dX/dt) instead of latents.
    Velocities at all noise levels treated equally.
    Prevents reward hacking: harder to exploit high-noise regions
    when optimizing flow trajectory directly.
    """
    mse = ((velocity_pred - velocity_ref) ** 2).mean()
    return mse_weight * mse
```

Rationale: Velocity fields are noise-scale-invariant (same metric applies across diffusion timesteps). MSE on velocities provides uniform constraints, preventing the model from concentrating reward signals in high-noise regions.

## Performance Impact

**Baseline (CFG + KL regularization):**
- Text Alignment (TA) score: ~0.81
- GenEval score: ~0.88
- Training efficiency: Multi-rollout CFG increases compute

**With Direct GRPO + Velocity MSE:**
- Text Alignment: 0.8381 (+0.0281, +3.5% relative)
- GenEval: 0.90 (+0.02, +2.3% relative)
- Training efficiency: Linear rollouts, single sample per prompt
- Inference: CFG removed entirely, single sample at inference (no branching overhead)

**Ablation findings:**
- Removing CFG alone: comparable or superior performance (suggests CFG not needed during training)
- Velocity MSE vs KL: MSE prevents reward hacking better in high-noise regions
- Combined effect: Joint optimization of text + image policies in unified MDP outperforms separate optimization

## When to Use

- Multimodal models with interleaved text-image generation (Reasoning + Image flow)
- Flow-matching-based diffusion models for visual generation
- When inference latency is critical (velocity MSE + no CFG = faster sampling)
- Models where joint text-image coherence matters (not independent generation)
- Large-scale training where compute efficiency (single rollouts) matters

## When NOT to Use

- Models requiring strong guidance control (CFG is useful if you need guidance scale tuning at inference)
- Latent-space models where KL is theoretically grounded (e.g., VAEs with principled regularization)
- Single-modality generation (optimization benefit is primarily for multimodal interleaving)
- If your model already has separate text/image pipelines (joint MDP assumes unified architecture)

## Implementation Checklist

To integrate direct GRPO + velocity MSE:

1. **Remove CFG from training**:
   - Replace branched rollouts with single-sample inference
   - Ensure model is trained with condition (prompt) always present
   - Remove unconditional branch from model (frees parameters)

2. **Implement velocity-based regularization**:
   - Predict flow velocity (dX/dt) from noise model
   - Compute MSE between predicted velocity and reference/baseline velocity
   - Replace KL loss with: `loss += mse_weight * velocity_mse`
   - Typical weight: 0.001 (same scale as prior KL)

3. **Formulate as unified MDP**:
   - Combine text-generation and image-generation into single sequence
   - Reward signal reflects joint text-image quality (e.g., CLIP score between reasoning and generated image)
   - GRPO computes advantages over group of samples

4. **Verify improvements**:
   - TextAlign benchmark: target 0.8381+
   - GenEval benchmark: target 0.90+
   - Compare training time per epoch: should be ~1-1.5× baseline with single rollouts

5. **Optional tuning**:
   - MSE weight: 0.0001-0.01 (if reward hacking observed, increase)
   - GRPO group size: 4-8 samples (balance gradient noise vs compute)
   - Flow timesteps: 10-100 (more steps = higher quality, slower)

## Known Issues

- **Training-inference mismatch if CFG used at inference**: If you train without CFG but want CFG at inference, reintroduce it post-training (guidance scale search needed).
- **Velocity instability at high noise**: If model predicts noisy velocities early in diffusion, add velocity smoothing or reduce learning rate.
- **Reward collapse**: Ensure reward signal provides sufficient variation across samples. If all samples get similar rewards, GRPO cannot compute meaningful advantages.
- **Mode coverage**: Direct GRPO may optimize for single mode (best text-image pair). Explore entropy regularization if diversity matters.

## Related Work

Builds on GRPO (Group Relative Policy Optimization) for RL and flow matching (Albergo et al.). Velocity-based regularization adapts ideas from optimal transport (Wasserstein flow). Extends prior CFG work (Ho & Salimans) by showing CFG unnecessary during training with direct policy optimization. Relates to concurrent work on removing inference-time guidance for diffusion.
