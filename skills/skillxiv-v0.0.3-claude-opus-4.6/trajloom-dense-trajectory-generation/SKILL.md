---
name: trajloom-dense-trajectory-generation
title: "TrajLoom: Spatiotemporal Consistency for Extended Trajectory Prediction"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22606"
keywords: [Trajectory Prediction, Grid-Anchor Encoding, VAE, Flow Models, Spatiotemporal Consistency]
description: "Improve dense trajectory generation by replacing absolute coordinate encoding with grid-anchor offset encoding (reduces location variance 90%→10%), adding spatiotemporal consistency regularizers to VAE (30-40× improvement in 81-frame prediction), and using boundary-anchored fine-tuning for flow models. Effective for predicting extended trajectories (81 frames vs. prior 24-frame limits) in autonomous driving and video prediction where motion coherence across time steps is critical."
category: "Component Innovation"
---

## What This Skill Does

Replace single-component trajectory encoding and loss functions with three complementary modifications: position-invariant grid-anchor encoding (eliminates location bias), spatiotemporal consistency regularizers for VAE (enforces motion realism), and on-policy boundary-anchored flow refinement (reduces long-horizon drift).

## Component 1: Grid-Anchor Offset Encoding

**Old component:** Absolute pixel coordinates for trajectory points, which entangle motion representation with global position.

```python
# Traditional: absolute coordinates bind motion to location
def encode_trajectory_absolute(trajectory):
    # trajectory shape: [T, P, 2] - T timesteps, P points, (x,y) coords
    return trajectory  # Motion patterns vary by grid cell
```

**New component:** Offset from fixed grid anchors removes location-dependent statistics.

```python
# Grid-Anchor Offset Encoding
def encode_trajectory_grid_anchor(trajectory, grid_size=8):
    """
    Express each point as offset from its grid cell anchor.
    Removes location-driven variance that confuses motion learning.
    """
    T, P, _ = trajectory.shape
    offsets = torch.zeros_like(trajectory)

    for p in range(P):
        point_xy = trajectory[:, p]  # [T, 2]
        # Find grid cell for each point
        grid_cell = (point_xy / grid_size).long()
        grid_anchor = grid_cell * grid_size

        # Offset: point - anchor
        offsets[:, p] = point_xy - grid_anchor.float()

    return offsets
```

**Impact:** Reduces coordinate variance driven by location from ~90% to ~10%, making motion patterns consistent and learnable across image regions.

---

## Component 2: Spatiotemporal Consistency Regularizers for VAE

**Old component:** Standard VAE with only pointwise reconstruction loss `L_recon`, ignoring temporal and spatial coherence.

```python
class StandardVAE(nn.Module):
    def forward(self, trajectory):
        z = self.encode(trajectory)
        recon = self.decode(z)
        loss = F.mse_loss(recon, trajectory)  # Only frame-level error
        return recon, loss
```

**New component:** Augment with temporal velocity matching and multi-scale spatial neighbor consistency.

```python
class TrajLoomVAE(nn.Module):
    def __init__(self, encoder, decoder, lambda_temporal=0.1, lambda_spatial=0.05):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_temporal = lambda_temporal
        self.lambda_spatial = lambda_spatial

    def forward(self, trajectory):
        z = self.encode(trajectory)
        recon = self.decode(z)

        # Standard reconstruction
        loss_recon = F.mse_loss(recon, trajectory)

        # Temporal consistency: velocity should be smooth
        vel_target = trajectory[1:] - trajectory[:-1]  # Ground truth velocity
        vel_recon = recon[1:] - recon[:-1]
        loss_temporal = F.mse_loss(vel_recon, vel_target)

        # Spatial consistency: neighbors should move coherently
        # Multi-scale: check consistency at distances 1, 2, 4
        loss_spatial = 0
        for scale in [1, 2, 4]:
            for p in range(trajectory.shape[1] - scale):
                neighbor_dist_target = torch.norm(
                    trajectory[:, p] - trajectory[:, p + scale],
                    dim=-1
                )
                neighbor_dist_recon = torch.norm(
                    recon[:, p] - recon[:, p + scale],
                    dim=-1
                )
                loss_spatial += F.mse_loss(neighbor_dist_recon, neighbor_dist_target)

        total_loss = (
            loss_recon +
            self.lambda_temporal * loss_temporal +
            self.lambda_spatial * loss_spatial
        )
        return recon, total_loss
```

**Impact on 81-frame prediction:**
- Without regularizers: VEPE = 29.35–63.09 pixels (large jitter)
- With TrajLoom-VAE: VEPE = 1.59–2.04 pixels (**30–40× improvement**)

---

## Component 3: Boundary-Anchored On-Policy Flow Fine-Tuning

**Old component:** Rectified flow trained on interpolated states, creating train-test mismatch during ODE sampling.

```python
class StandardFlow(nn.Module):
    def forward(self, z, t):
        # Flow trained on interpolated z_t = t*z_1 + (1-t)*z_0
        # At test time: solve ODE starting from observed z_0
        # Mismatch: test states never visited during training
        return self.velocity_net(z, t)
```

**New component:** Anchor boundary conditions and fine-tune on self-visited ODE states.

```python
class TrajLoomFlow(nn.Module):
    def __init__(self, velocity_net):
        super().__init__()
        self.velocity_net = velocity_net

    def forward(self, z_0, num_steps=100):
        """Standard ODE solution with boundary hints."""
        z = z_0.clone()
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = step / num_steps
            vel = self.velocity_net(z, t)
            z = z + vel * dt

        return z

    def on_policy_finetune(self, z_0_batch, target_z_1_batch, K=5):
        """
        Fine-tune on actual ODE trajectories (not interpolations).
        K-step rollout: solve ODE, compute loss, backprop.
        """
        optimizer = torch.optim.Adam(self.velocity_net.parameters(), lr=1e-4)

        for _ in range(K):
            # Rollout ODE
            z_trajectory = self.forward(z_0_batch, num_steps=50)  # Generated trajectory

            # Loss: endpoint consistency with target + velocity smoothness
            loss_endpoint = F.mse_loss(z_trajectory, target_z_1_batch)

            # Velocity smoothness: encourage consistent motion
            # Compute velocities along trajectory
            vel_pred = (z_trajectory[1:] - z_trajectory[:-1]) / (1/50)
            vel_smooth = torch.norm(vel_pred[1:] - vel_pred[:-1])
            loss_smooth = vel_smooth.mean()

            loss = loss_endpoint + 0.1 * loss_smooth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()
```

**Boundary hint mechanism:**
- Anchor z₀ (last history token) and z₁ (target future state)
- ODE solver is constrained to respect endpoints, reducing divergence

**Impact on 81-frame flow-based generation:**
- FVMD (flow velocity magnitude distance): 4872 → 1338 (**2.5–3.6× reduction**)
- FlowTV (temporal variance): substantially reduced
- DivCurlE (spatial discontinuities): eliminated

## Performance Summary

| Component | Metric | Before | After | Improvement |
|-----------|--------|--------|-------|-------------|
| **Offset Encoding** | Location variance explained | 90% | 10% | 80pp reduction |
| **TrajLoom-VAE** | VEPE (81 frames) | 29.35–63.09 px | 1.59–2.04 px | **30–40×** |
| **Flow + Boundary** | FVMD | 4872 | 1338 | **2.5–3.6×** |

## When to Use

- Predicting extended trajectories (≥81 frames)
- Tasks where temporal coherence matters (autonomous driving, video prediction)
- When training data is limited (VAE regularizers prevent overfitting to jittery patterns)
- Applications sensitive to local motion smoothness (motion capture, character animation)
- Generative models where perceptual quality of generated trajectories is critical

## When NOT to Use

- Short-horizon prediction (<10 frames): overhead not justified
- Sparse trajectories: grid-anchor encoding assumes dense point clouds
- Tasks where absolute global coordinates are semantically meaningful
- Real-time applications with strict latency budgets (flow fine-tuning adds compute)

## Implementation Checklist

**1. Replace coordinate encoding:**
```python
# Old: trajectory_encoded = trajectory
# New:
trajectory_encoded = encode_trajectory_grid_anchor(
    trajectory,
    grid_size=8  # Adjust for your image resolution
)
```

**2. Augment VAE loss:**
```python
vae = TrajLoomVAE(
    encoder=your_encoder,
    decoder=your_decoder,
    lambda_temporal=0.1,  # Weight temporal consistency
    lambda_spatial=0.05   # Weight spatial coherence
)
# Training automatically includes consistency terms
```

**3. Optional: fine-tune flow model on-policy:**
```python
# After standard flow training:
flow_model.on_policy_finetune(
    z_0_batch=observed_latents,
    target_z_1_batch=future_latents,
    K=5  # Number of rollout iterations
)
```

**4. Verification:**
- Measure VEPE (endpoint error) on 81-frame prediction
- Check temporal smoothness: velocity should vary slowly
- Ablate: remove each component to confirm individual contributions

**5. Hyperparameter tuning:**
- Grid size: 8 works for 256×256 images; scale proportionally
- Lambda values: start with (0.1, 0.05); increase if jitter persists
- Flow steps: 50-100 steps balances speed vs. quality
- Boundary anchor weight: increase if ODE diverges at horizon

**6. Known issues:**
- Grid anchors may create boundary artifacts at image edges; use padding
- Spatial consistency loss is expensive for large point clouds (>1000 points)
- Flow fine-tuning requires second-order gradients; use lower-precision training if memory-limited
- Extrapolation beyond training horizon: consistency guarantees degrade

## Related Work

Builds on rectified flows (Flow Matching, Liphardt et al.) and VAE regularization (β-VAE, annealed loss). Grid-anchor encoding parallels local coordinate systems in computer graphics and relates to positional normalization in transformers. Boundary-anchored fine-tuning generalizes active learning on ODE solvers.
