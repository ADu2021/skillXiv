---
name: manifold-aware-rl-video
title: "SAGE-GRPO: Manifold-Aware Exploration for Video Generation Reinforcement Learning"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.21872
keywords: [Reinforcement Learning, Video Generation, Diffusion, Manifold Learning]
description: "Constrain video GRPO policy updates to stay within pre-trained model's data manifold using dual-control exploration. Implement precise manifold-aware SDE with logarithmic noise variance correction (captures geometric signal decay standard methods miss). Apply gradient norm equalizer to balance learning across diffusion timesteps (mitigate vanishing/exploding gradients). Use dual trust region combining position control (anchored exploration) and velocity control (KL constraints) for stability-plasticity balance."
---

## Core Mechanism

SAGE-GRPO treats video generation as constrained exploration within the manifold defined by a pre-trained diffusion model.

**Manifold Definition:**
The pre-trained video model defines a valid video data manifold—the set of videos the model considers plausible. Policy optimization must keep updates within this manifold's vicinity, preventing drift to out-of-distribution proposals.

**Exploration Constraints:**
Rather than unconstrained policy gradient descent, apply multi-scale constraints:
1. **Micro-level**: Per-timestep noise variance control
2. **Macro-level**: Multi-step trust region constraints

This balances exploration (improve reward) with stability (stay on manifold).

## Key Components

### Micro-Level: Precise Manifold-Aware SDE

Standard approaches use first-order approximations for noise variance, missing geometric signal decay in diffusion.

**Standard (Linear) Approximation:**
```python
# Assumes linear noise schedule
sigma_t = sqrt(1 - alpha_cumprod_t)  # First-order only
Σ_t = eta^2 * sigma_t^2  # Linear scaling
```

**Precise (Logarithmic) Correction:**
```python
# Geometric signal decay captured via logarithmic term
Σ_t = η²[-(σ_t - σ_{t+1}) + log((1 - σ_{t+1})/(1 - σ_t))]

# Breakdown:
# -(σ_t - σ_{t+1}): Linear noise variance change
# log((1-σ_{t+1})/(1-σ_t)): Logarithmic correction
#   Captures how signal degrades geometrically
#   Missing in standard first-order approximation
```

**Geometric Intuition:**
In diffusion, noise doesn't degrade linearly—signal decays exponentially. The log term captures this exponential decay that linear methods miss. At high noise (t→1), signal is nearly zero; at low noise (t→0), signal is dense. The logarithmic correction accounts for this non-linear relationship.

**Implementation:**
```python
def manifold_aware_sde_variance(sigma_t, sigma_t_next, eta=1.0):
    """
    Compute noise variance for diffusion policy gradient.

    Args:
        sigma_t: noise level at current step
        sigma_t_next: noise level at next step
        eta: temperature parameter (0-1, controls diffusion randomness)

    Returns:
        Σ_t: variance for policy gradient update
    """
    linear_term = sigma_t - sigma_t_next

    # Logarithmic correction for geometric decay
    signal_ratio = (1 - sigma_t_next) / (1 - sigma_t)
    log_term = np.log(signal_ratio)

    variance = eta**2 * (linear_term + log_term)
    return variance
```

This precise variance schedule prevents both over-correction (early diffusion steps with high noise) and under-correction (late steps with low noise).

### Gradient Norm Equalizer

Standard GRPO training has severe gradient imbalance across diffusion timesteps.

**The Problem:**
```
Timestep 0 (t=0, low noise):
  - Model output is refined
  - Loss signal is clear
  - Gradients are large/exploding

Timestep T (t=T, high noise):
  - Model output is diffuse
  - Loss signal is weak
  - Gradients vanish/explode
```

This imbalance causes:
- Early timesteps overfit (gradient updates dominate)
- Late timesteps undertrain (gradients too small)
- Unstable training dynamics

**Solution: Per-Timestep Normalization**

```python
def gradient_norm_equalizer(policy_gradient, timestep, max_timestep):
    """
    Normalize gradient magnitude to ensure balanced learning across timesteps.

    Args:
        policy_gradient: gradient from policy loss
        timestep: current diffusion step (0 to T)
        max_timestep: maximum timestep (T)

    Returns:
        normalized_gradient: balanced gradient
    """
    # Compute baseline gradient norm at this timestep
    baseline_norm = compute_baseline_norm(timestep, max_timestep)

    # Normalize current gradient to baseline
    current_norm = np.linalg.norm(policy_gradient)
    if current_norm > 0:
        normalized = policy_gradient * (baseline_norm / current_norm)
    else:
        normalized = policy_gradient

    return normalized
```

Effect: Each timestep contributes equally to optimization pressure, preventing domination by any single diffusion phase.

### Macro-Level: Dual Trust Region

Prevents long-horizon drift while maintaining plasticity (ability to change).

**Position Control (Recentering Anchor):**
```python
# Every N steps, refresh anchor policy
if step % anchor_refresh_interval == 0:
    anchor_policy = current_policy.clone()
    anchor_checkpoint = save_checkpoint()

# Measure distance from anchor
position_divergence = kl_divergence(current_policy, anchor_policy)

# Constrain how far we drift from anchor
if position_divergence > max_position_drift:
    # Pull back toward anchor
    current_policy = interpolate(current_policy, anchor_policy, alpha=0.5)
```

**Velocity Control (Instantaneous Step Constraint):**
```python
# Limit per-step policy update
# (velocity = instantaneous change in policy)
step_kl = kl_divergence(policy_old, policy_new)

# Reject if single step is too large
if step_kl > max_step_kl:
    # Reduce learning rate or reject this step
    policy_new = policy_old  # Stay put
```

**Dual Purpose:**

1. **Stability:** Position control prevents accumulated drift
   - Regular anchoring prevents long-horizon divergence
   - Early stopping if drift detected

2. **Plasticity:** Velocity control allows meaningful per-step updates
   - Not too conservative (still exploring)
   - Not too aggressive (stable convergence)

This resolves the classic stability-plasticity dilemma: trust region alone too strict, unconstrained updates too chaotic. Dual control provides both.

## Implementation Pattern

**GRPO Loop with Manifold Awareness:**

```python
for episode in range(num_episodes):
    # Collect trajectories
    video = video_prior_sample()  # From pre-trained model

    for t in range(T):
        # Micro-level: manifold-aware variance
        sigma_t = noise_schedule[t]
        sigma_t_next = noise_schedule[t+1]
        variance = manifold_aware_sde_variance(sigma_t, sigma_t_next)

        # Compute policy gradient
        policy_grad = compute_policy_gradient(video, t)

        # Gradient norm equalization
        policy_grad = gradient_norm_equalizer(policy_grad, t, T)

        # Macro-level: dual trust region
        # (position & velocity control applied after gradient step)
        policy_update = optimizer.step(policy_grad)

        # Position control: check drift from anchor
        if step % anchor_refresh == 0:
            anchor_policy = current_policy.detach()

        position_div = kl_divergence(current_policy, anchor_policy)
        if position_div > threshold:
            current_policy = anchor_policy  # Reset to anchor

        # Velocity control: check per-step change
        velocity = kl_divergence(policy_old, current_policy)
        if velocity > max_step_kl:
            current_policy = policy_old  # Reject this step
```

## Conditions of Applicability

**Works well when:**
- Pre-trained video model is high quality (manifold well-defined)
- Reward signal is correlated with video quality (GRPO has signal)
- Multi-hour training feasible (GRPO is expensive)
- Stability more important than raw exploration (manifold constraint prioritized)

**Less optimal when:**
- Pre-trained model is weak (manifold not meaningful)
- Reward is sparse or misleading (hard to optimize)
- Real-time inference required (GRPO training offline)
- Out-of-distribution exploration is beneficial (manifold constraint limiting)

## Integration Points

**Input:**
- Pre-trained video diffusion model (defines manifold)
- Reward function (optimized signal)
- Video seed or prompt (generation input)

**Output:**
- RL-finetuned policy (stays on manifold, maximizes reward)
- Improved video quality (aesthetic, consistency, goal alignment)

**Compatibility:**
- Works with any diffusion-based video model
- Reward function is modular (can swap different rewards)
- Micro/macro constraints orthogonal (can enable/disable individually)
