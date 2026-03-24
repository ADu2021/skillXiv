---
name: jit-spatial-diffusion-acceleration
title: "Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.10744"
keywords: [Diffusion, Acceleration, Spatial Tokens, Inference, Transformers]
description: "Accelerate diffusion transformers by processing only sparse anchor tokens in early stages, then expanding to full spatial resolution using learnable extrapolation. Combines SAG-ODE velocity field lifting with importance-guided token activation for lossless speedup."
---

# Technique: Spatial Token Approximation via Importance-Guided Anchor Selection

Diffusion transformers generate images by iteratively denoising all spatial tokens across T timesteps. This full-token processing is computationally expensive. Just-in-Time (JiT) accelerates inference by computing only anchor tokens initially, then extrapolating the velocity field to predict evolution of non-anchor tokens, finally expanding to full resolution in later stages.

The key insight is that early diffusion steps contain redundancy: full spatial processing is unnecessary before the model establishes coarse structure. By selecting anchors based on velocity variance, the method preserves high-activity regions while approximating others.

## Core Concept

JiT operates across three complementary mechanisms:

1. **Spatially Approximated Generative ODE (SAG-ODE)**: Uses an augmented lifter operator to extrapolate velocity fields from anchor tokens to full space.

2. **Deterministic Micro-Flow (DMF)**: Ensures smooth transitions when expanding token sets, maintaining consistency between stages.

3. **Importance-Guided Token Activation (ITA)**: Dynamically selects tokens based on local velocity variance rather than fixed patterns.

This enables stage-adaptive token reduction: aggressive sparsity early when structure is coarse, gradual token addition as generation details accumulate.

## Architecture Overview

- **Anchor token selector**: Identifies sparse subset based on velocity variance
- **Lifter operator**: Maps sparse velocity field to full token space
- **DMF handler**: Manages token set expansion with smooth transitions
- **Full transformer layers**: Operate on growing token set over timesteps
- **Lossless design**: No upsampling artifacts or reconstruction losses

## Implementation Steps

### Step 1: Compute Anchor Tokens via ITA

Select tokens based on local velocity variance in the generative ODE.

```python
import torch
import torch.nn.functional as F

def select_anchor_tokens(velocity_field, anchor_ratio=0.25):
    """
    Select anchor tokens based on local velocity variance.

    velocity_field: (batch, seq_len, dim) velocity predictions from transformer
    anchor_ratio: fraction of tokens to retain as anchors
    """
    # Compute local variance of velocity field
    # Use 1D convolution to get neighborhood variance
    kernel_size = 5
    padding = kernel_size // 2

    # Unfold to compute variance across neighborhoods
    unfolded = F.unfold(
        velocity_field.unsqueeze(-1).permute(0, 2, 3, 1),
        kernel_size=(kernel_size, 1),
        padding=(padding, 0)
    )  # (batch, dim*kernel_size, seq_len)

    variance = unfolded.std(dim=1)  # (batch, seq_len)

    # Select top-anchor_ratio as anchors
    num_anchors = int(anchor_ratio * velocity_field.shape[1])
    anchor_indices = torch.topk(variance, num_anchors, dim=1)[1]

    return anchor_indices, variance
```

### Step 2: Implement SAG-ODE Lifter Operator

Extrapolate sparse velocity field to full token space using learned projections.

```python
class LifterOperator(torch.nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.dim = dim
        # Learn to lift sparse observations to full space
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, dim)
        )

    def forward(self, anchor_velocity, anchor_positions, full_seq_len):
        """
        Extrapolate anchor velocity field to full sequence.

        anchor_velocity: (batch, num_anchors, dim)
        anchor_positions: (batch, num_anchors) indices
        full_seq_len: int
        """
        batch_size = anchor_velocity.shape[0]

        # Initialize full velocity field via interpolation
        full_velocity = torch.zeros(
            batch_size,
            full_seq_len,
            self.dim,
            device=anchor_velocity.device
        )

        # Linear interpolation between anchors for structure
        sorted_indices = torch.argsort(anchor_positions, dim=1)
        sorted_positions = torch.gather(anchor_positions, 1, sorted_indices)
        sorted_velocity = torch.gather(
            anchor_velocity,
            1,
            sorted_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        )

        for i in range(sorted_positions.shape[1] - 1):
            pos_i = sorted_positions[:, i]
            pos_j = sorted_positions[:, i + 1]
            vel_i = sorted_velocity[:, i]
            vel_j = sorted_velocity[:, i + 1]

            # Linear interpolation
            for t in range(full_seq_len):
                if pos_i <= t <= pos_j:
                    alpha = (t - pos_i) / (pos_j - pos_i + 1e-8)
                    full_velocity[:, t] = (1 - alpha) * vel_i + alpha * vel_j

        return full_velocity
```

### Step 3: Deterministic Micro-Flow for Token Expansion

Smoothly transition when expanding token set between stages.

```python
def deterministic_micro_flow(
    old_tokens,
    new_token_indices,
    target_noise_level,
    num_micro_steps=4
):
    """
    Evolve new tokens toward target state smoothly.

    old_tokens: (batch, old_seq_len, dim) existing tokens
    new_token_indices: (batch, num_new) indices for tokens to expand
    target_noise_level: noise schedule value for next stage
    """
    batch_size = old_tokens.shape[0]
    dim = old_tokens.shape[2]

    # Initialize new tokens from neighborhood context
    new_tokens = torch.zeros(
        batch_size,
        len(new_token_indices),
        dim,
        device=old_tokens.device
    )

    for i, idx in enumerate(new_token_indices):
        # Use local context (average of neighbors)
        neighbor_start = max(0, idx - 2)
        neighbor_end = min(old_tokens.shape[1], idx + 3)
        context = old_tokens[:, neighbor_start:neighbor_end].mean(dim=1)
        new_tokens[:, i] = context

    # Micro-flow: evolve new tokens to match velocity field
    for step in range(num_micro_steps):
        alpha = (step + 1) / num_micro_steps
        # Blend toward target state with noise level adjustment
        new_tokens = alpha * new_tokens + (1 - alpha) * context

    return new_tokens
```

### Step 4: Stage-Adaptive Token Processing

Gradually increase token set as generation progresses.

```python
def inference_with_jit(
    model,
    latent,
    num_steps=20,
    initial_anchor_ratio=0.25,
    final_anchor_ratio=1.0
):
    """
    Run diffusion with stage-adaptive spatial token processing.
    """
    seq_len = latent.shape[1]

    for t in range(num_steps):
        # Compute anchor ratio: increase gradually
        progress = t / num_steps
        anchor_ratio = (
            initial_anchor_ratio +
            progress * (final_anchor_ratio - initial_anchor_ratio)
        )

        num_anchors = max(1, int(anchor_ratio * seq_len))

        # Select anchors
        anchor_indices, variance = select_anchor_tokens(
            latent,
            anchor_ratio=anchor_ratio
        )

        # Process only anchor tokens through transformer
        anchor_tokens = latent[:, anchor_indices]
        anchor_output = model.transformer_forward(anchor_tokens)

        # Extrapolate to full space using SAG-ODE lifter
        lifter = LifterOperator(latent.shape[-1])
        full_velocity = lifter(anchor_output, anchor_indices, seq_len)

        # Expand token set at later stages
        if t > num_steps // 2 and anchor_ratio < 1.0:
            new_indices = select_indices_for_expansion(seq_len, anchor_indices)
            new_tokens = deterministic_micro_flow(latent, new_indices, t/num_steps)
            latent = combine_tokens(latent, new_tokens, new_indices)
        else:
            latent = full_velocity

    return latent
```

## Practical Guidance

**When to Use:**
- Image generation with diffusion transformers (FLUX, DiT variants)
- Inference where latency is critical
- Long sequence generation where speedup compounds across timesteps
- Production deployments with strict latency SLAs

**When NOT to Use:**
- Training (acceleration applies to inference only)
- CNNs or other non-transformer architectures
- Scenarios requiring guaranteed bit-perfect reproducibility

**Hyperparameter Tuning:**
- **initial_anchor_ratio**: 0.1-0.3 works well; too low causes visible artifacts
- **anchor_ratio schedule**: Linear increase often sufficient; smooth curves can help
- **micro_flow_steps**: 2-4 balances smoothness and latency
- **kernel_size for variance**: 5-7 captures sufficient locality

**Common Pitfalls:**
- Anchor ratio too aggressive early (visual artifacts)
- Insufficient micro-flow steps (token jitter)
- Non-smooth anchor ratio schedule causing abrupt transitions
- Forgetting to scale attention mask during partial token processing

## Reference

[Just-in-Time paper on arXiv](https://arxiv.org/abs/2603.10744)
