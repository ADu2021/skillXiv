---
name: meanflow-one-step-generation
title: "Flow Straighter and Faster: Efficient One-Step Generation via MeanFlow with Rectified Trajectories"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2511.23342
keywords: [flow-matching, one-step-generation, trajectory-learning, diffusion-distillation, generative-models]
description: "Trains mean-velocity models on rectified couplings from pretrained flow models to dramatically smooth loss landscape, enabling faster convergence and superior one-step generation quality without additional training data."
---

## Summary

Flow Straighter and Faster (MeanFlow) addresses one-step generation by identifying that learning mean-velocity fields on highly curved generative trajectories creates noisy optimization. The key innovation trains mean-velocity models on rectified couplings (straighter trajectories from pretrained flow models) combined with distance-based truncation heuristics, dramatically smoothing the loss landscape.

## Core Technique

**Trajectory Rectification:** Instead of learning velocity from independent random couplings (which have arbitrary curvature), use a pretrained flow model to generate "straighter" trajectories. The pretrained model has already learned optimal transports, and sampling from its trajectories provides smoother data for the student model.

**Distance-Based Truncation:** Remove the most curved trajectory pairs using a distance metric. If two adjacent samples in the coupling show large ||x_end - x_start|| relative to ||z_end - z_start||, they're likely noisy and removed from training.

**Mean-Velocity Learning:** Train a lightweight model to predict average velocity along trajectories:
```
v_mean = E[dx/dt] along trajectory
```
This is simpler than predicting full trajectories and benefits from rectified data.

## Implementation

**Pretrained flow generation:** Load a trained flow matching model (e.g., stable diffusion flow). Sample trajectories for one-step student training.

**Trajectory selection:** For candidates in the batch, compute:
```python
curvature = ||x_t - x_0|| / (t + ε)  # Distance per unit time
if curvature > threshold:
    skip_sample(trajectory)
```

**Mean-velocity model:** Train a small transformer:
```python
velocity = mlp(concat(x_0, x_T, t))  # Predict average velocity
loss = mse(velocity, (x_T - x_0) / (t_max - t_min))
```

**One-step inference:** At test time, sample z ~ N(0,I) and apply: x = z + velocity(z, t=0), with optional refinement.

## When to Use

- One-step image generation where training data is limited
- Scenarios where distillation from pretrained flow models is available
- Applications requiring both fast inference and good quality
- Tasks where the pretrained model's trajectory space is appropriate

## When NOT to Use

- Scenarios requiring multi-step generation for highest quality
- Tasks without access to good pretrained flow models
- Situations where the original trajectory space is preferable
- Applications where velocity prediction is inaccurate or unstable

## Key References

- Flow matching and optimal transport theory
- Diffusion distillation and student-teacher learning
- Trajectory optimization and curvature analysis
- One-step generative models and sampling efficiency
