---
name: direct-denoising-diffusion
title: "Back to Basics: Let Denoising Generative Models Denoise—Direct x-Prediction in Diffusion"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.13720"
keywords: [Diffusion Models, Direct Denoising, x-Prediction, Generative Modeling, Capacity Efficiency]
description: "Improve diffusion model capacity efficiency by directly predicting clean data instead of noise—leverage the manifold assumption that natural data occupies low-dimensional space while noise spans full dimensionality."
---

# Recover Capacity by Direct Denoising—Predict Clean Data Not Noise

Diffusion models typically predict noise (epsilon) or velocity at each denoising step. This choice seems natural but imposes a hidden cost: noise and velocity distributions span the full data dimensionality, wasting model capacity on high-dimensional structure. Direct x-prediction assumes natural data lies on a low-dimensional manifold—the model only needs to preserve essential information while filtering noise, using capacity more efficiently.

This paper demonstrates that limited-capacity networks can generate high-dimensional data via x-prediction where epsilon/v-prediction fails. The insight is practical: shift the prediction target to align with your data's intrinsic dimensionality, not the ambient space.

## Core Concept

Diffusion models work by iteratively adding and removing noise. At each step, the model must predict either:

1. **Noise (ε-prediction)**: The random noise added; distributed across full dimensionality
2. **Velocity (v-prediction)**: A mixture; also spreads across full space
3. **Clean data (x-prediction)**: The denoised output; concentrated on low-dimensional manifold

The mathematical relationships between these are equivalent—they're coordinate transformations of the same underlying process. However, **they differ fundamentally in capacity requirement**:

- **High-dimensional noise**: Requires network to preserve all directional information; capacity-intensive
- **Low-dimensional data**: Network only needs to capture essential structure; more efficient

This manifold assumption—natural data lives in lower-dimensional space than noise—creates practical capacity gaps observable at scale.

## Architecture Overview

- **Manifold Assumption**: Natural data concentrates on low-dimensional manifold; noise fills full high-dimensional space
- **x-Prediction Head**: Network outputs clean data directly, not noise or velocity
- **Matching Framework**: Use standard diffusion loss (MSE) adapted for x-prediction via mathematical conversion
- **Patch-Based Encoding**: For image/video models, use large patches to reduce dimensionality and align with manifold structure
- **Capacity Scaling**: Lower capacity networks exhibit larger performance gaps; x-prediction shows more benefit at limited budgets

## Implementation Steps

**Step 1: Reformulate Diffusion Process for x-Prediction.** Convert noise schedule to x-prediction formulation.

```python
import torch
import torch.nn as nn

class XPredictionDiffusion:
    """
    Diffusion process using x-prediction (direct clean data prediction).
    """
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps

        # Standard beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Key quantities for x-prediction
        # At step t, noisy image: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_1m_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to clean sample (forward process).
        x_0: clean data (batch, channels, height, width)
        t: timestep (batch,)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas = self.sqrt_alphas_cumprod[t]
        sqrt_1m_alphas = self.sqrt_1m_alphas_cumprod[t]

        # Reshape for broadcasting
        while len(sqrt_alphas.shape) < len(x_0.shape):
            sqrt_alphas = sqrt_alphas.unsqueeze(-1)
            sqrt_1m_alphas = sqrt_1m_alphas.unsqueeze(-1)

        # Noisy sample
        x_t = sqrt_alphas * x_0 + sqrt_1m_alphas * noise

        return x_t, noise

    def predict_x0_from_xt(self, x_t, t, model_prediction):
        """
        In x-prediction, the model directly outputs x_0 estimate.
        This is the reverse process; often called "x-denoising".
        """
        # The model prediction IS the x_0 estimate
        return model_prediction

    def loss_x_prediction(self, x_0, t, model_pred):
        """
        L2 loss for x-prediction: MSE between model output and true x_0.
        Simpler than noise prediction (no noise extraction needed).
        """
        return torch.nn.functional.mse_loss(model_pred, x_0, reduction='mean')
```

**Step 2: Build Network with x-Prediction Head.** Design model to output clean data directly.

```python
class XPredictionModel(nn.Module):
    """
    Diffusion model with x-prediction head.
    Can be Transformer, UNet, etc.; example: Transformer-based.
    """
    def __init__(self, input_channels=3, patch_size=4, hidden_dim=768, num_layers=12):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Patch embedding
        self.patch_embed = nn.Linear(3 * patch_size * patch_size, hidden_dim)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Transformer backbone
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=3072),
            num_layers=num_layers
        )

        # x-Prediction head: output clean data patches
        self.x_pred_head = nn.Linear(hidden_dim, 3 * patch_size * patch_size)

    def forward(self, x_t, t):
        """
        x_t: noisy input (batch, 3, H, W)
        t: timestep (batch,)
        Returns: predicted clean image (batch, 3, H, W)
        """
        batch, channels, height, width = x_t.shape

        # Patchify
        patches = self._patchify(x_t)  # (batch, num_patches, patch_dim)

        # Embed patches
        patch_embeds = self.patch_embed(patches)  # (batch, num_patches, hidden_dim)

        # Time embedding
        time_emb = self.time_embed(t.unsqueeze(-1).float())  # (batch, hidden_dim)
        time_emb = time_emb.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Add time to patches
        patch_embeds = patch_embeds + time_emb  # Broadcasting

        # Transformer
        hidden = self.transformer(patch_embeds)  # (batch, num_patches, hidden_dim)

        # x-Prediction: directly output clean patches
        clean_patches = self.x_pred_head(hidden)  # (batch, num_patches, patch_dim)

        # Unpatchify
        x_0_pred = self._unpatchify(clean_patches, height, width)

        return x_0_pred

    def _patchify(self, x):
        """Convert image to non-overlapping patches."""
        batch, channels, height, width = x.shape
        x = x.reshape(
            batch,
            channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size
        )
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(batch, -1, channels * self.patch_size * self.patch_size)
        return x

    def _unpatchify(self, patches, height, width):
        """Reconstruct image from patches."""
        batch, num_patches, patch_dim = patches.shape
        channels = 3

        patches = patches.reshape(
            batch,
            height // self.patch_size,
            width // self.patch_size,
            channels,
            self.patch_size,
            self.patch_size
        )
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = patches.reshape(batch, channels, height, width)

        return x
```

**Step 3: Training Loop.** Train with x-prediction loss.

```python
def train_x_prediction_diffusion(model, dataloader, num_epochs=100, lr=1e-4):
    """
    Train diffusion model with x-prediction.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    diffusion = XPredictionDiffusion(timesteps=1000)

    for epoch in range(num_epochs):
        total_loss = 0

        for x_0 in dataloader:  # x_0: clean images
            batch_size = x_0.shape[0]

            # Random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,))

            # Forward process: add noise
            x_t, noise = diffusion.add_noise(x_0, t)

            # Model prediction (x-prediction)
            x_0_pred = model(x_t, t)

            # Loss: direct MSE between prediction and clean data
            loss = diffusion.loss_x_prediction(x_0, t, x_0_pred)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")
```

**Step 4: Inference (Denoising).** Generate by iteratively denoising.

```python
@torch.no_grad()
def generate_x_prediction(model, diffusion, shape, num_steps=100):
    """
    Generate samples via iterative x-prediction denoising.
    """
    # Start with noise
    x_t = torch.randn(shape)

    # Iterate denoising (reverse process)
    timesteps = torch.linspace(999, 0, num_steps).long()

    for t in timesteps:
        t_batch = torch.full((x_t.shape[0],), t)

        # Model predicts clean image at this step
        x_0_pred = model(x_t, t_batch)

        # Get alphas for current and next step
        alpha_t = diffusion.alphas_cumprod[t]
        alpha_prev = diffusion.alphas_cumprod[t - 1] if t > 0 else 1.0

        # Update x_t using predicted x_0
        # This is the reverse step in x-prediction formulation
        c1 = torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
        c2 = torch.sqrt(alpha_prev / alpha_t)

        x_t = c2 * x_0_pred + c1 * (x_t - torch.sqrt(1 - alpha_t) * x_0_pred)

        # Add minimal noise if not final step
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            x_t = x_t + sigma_t * noise

    return x_t.clamp(-1, 1)
```

## Practical Guidance

**When to Use:** Training diffusion models on bounded data (images, quantized latents) with limited model capacity. x-Prediction shows largest benefits when model capacity is restricted (< 1B parameters).

**Hyperparameters:**
- Patch size: larger patches (8–16) reduce dimensionality and favor x-prediction; adjust based on image resolution
- Schedule: use same beta schedule as noise-pred; mathematical equivalence enables direct comparison
- Learning rate: typically same as noise-pred; no special adjustments needed

**Pitfalls:**
- **Unbounded predictions**: Models can output values outside data range; apply clipping or use bounded activations
- **Instability at early timesteps**: Very noisy inputs may lead to erratic x-0 predictions; use warm-up or gradient clipping
- **Not always faster**: Wall-clock time may be similar (both models have same architecture); gain is capacity efficiency, not compute
- **Dataset-dependent**: Manifold assumption stronger on natural images; weaker on synthetic/high-dimensional data

**When NOT to Use:** Very large models where capacity is not a bottleneck; data not concentrated on low-dimensional manifold.

**Integration:** Drop-in replacement for noise-prediction models; use same sampling procedures with adapted formulations.

---
Reference: https://arxiv.org/abs/2511.13720
