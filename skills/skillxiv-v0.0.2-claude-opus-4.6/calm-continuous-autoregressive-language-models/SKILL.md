---
name: calm-continuous-autoregressive-language-models
title: "Continuous Autoregressive Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.27688"
keywords: [Continuous Embeddings, Autoregressive Models, Token Compression, Likelihood-Free, Energy Scoring]
description: "Replace discrete token prediction with continuous vector prediction by training a high-fidelity autoencoder to compress K tokens into single latent vectors, enabling K-fold sequence length reduction while maintaining likelihood-free generation through energy-based scoring rules."
---

# Title: Predict Continuous Token-Chunk Vectors for Efficient Language Generation

Traditional autoregressive language models predict one discrete token at a time, requiring billions of forward passes for lengthy generations. CALM (Continuous Autoregressive Language Model) reduces generation steps by a factor of K by compressing K tokens into a single continuous vector, then predicting vectors directly. The key innovation is a **likelihood-free training framework** using energy scoring that enables stable continuous-space learning without explicit probability densities.

The approach combines a high-fidelity variational autoencoder for token compression with an energy-based generative head that uses strictly proper scoring rules for training.

## Core Concept

**Continuous-Space Next-Vector Prediction**:
- Compress K consecutive tokens into a single latent vector via a learned autoencoder
- Train the model to predict next latent vector autoregressively: p(Z) = ∏p(zᵢ|z<i)
- Use energy scoring (likelihood-free) instead of explicit probability distributions
- K-fold reduction in sequence length with >99.9% reconstruction fidelity

This sidesteps diffusion or flow-matching complexity by using energy-based models that evaluate alignment between predictions and observations through sample distances rather than probability densities.

## Architecture Overview

- **Autoencoder**: High-fidelity token compression (VQ-VAE style, ~75M parameters)
- **Variational Regularization**: KL divergence smoothing latent space with posterior collapse prevention
- **Energy Transformer Head**: Residual MLP blocks (10% of model parameters) generating predictions via energy scoring
- **Strictly Proper Scoring**: Energy Score for training, BrierLM for evaluation
- **Two-Stage Training**: Autoencoder pre-training, then CALM model training

## Implementation Steps

**1. Train High-Fidelity Autoencoder**

Build a symmetric encoder-decoder that compresses K=4 tokens into l=128-dimensional latent vectors. The encoder maps token embeddings through feed-forward networks and flattens to a latent vector. Use variational regularization with KL clipping to prevent posterior collapse.

```python
# Autoencoder architecture
class TokenAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, latent_dim=128):
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )
        self.decoder = nn.Linear(latent_dim, vocab_size)

    def encode_tokens(self, token_ids, k=4):
        # Reshape [B, T] -> [B*T//k, k, vocab_dim]
        embeddings = self.embed(token_ids)
        chunks = embeddings.reshape(-1, k, -1)
        # Each chunk -> latent vector
        z_params = self.encoder(chunks.flatten(1))
        z = z_params[:, :self.latent_dim]
        return z

    def decode_chunk(self, z):
        logits = self.decoder(z)
        return logits  # [B, vocab_size]
```

**2. Implement Variational Regularization with KL Clipping**

During autoencoder training, add KL divergence loss with a clipping strategy to ensure all latent dimensions remain informative. This prevents the encoder from collapsing dimensions to the prior.

```python
def compute_autoencoder_loss(z_params, targets, vocab_logits):
    # Reconstruction: cross-entropy on token sequences
    recon_loss = F.cross_entropy(vocab_logits, targets)

    # KL divergence with clipping per dimension
    mean, logvar = z_params[:, :self.latent_dim], z_params[:, self.latent_dim:]
    kl_per_dim = 0.5 * (mean**2 + torch.exp(logvar) - logvar - 1)

    # Clip KL at floor to prevent posterior collapse
    kl_clipped = torch.clamp(kl_per_dim, min=0.5)
    kl_loss = kl_clipped.mean()

    return recon_loss + beta * kl_loss
```

**3. Design Energy-Based Generative Head**

Rather than predicting probability distributions, use an energy transformer that generates predictions via iterative refinement. The head consists of residual MLP blocks that progressively refine an initial noise vector conditioned on model hidden states.

```python
class EnergyGenerativeHead(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_blocks=4):
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + latent_dim, hidden_dim),
                nn.SwiGLU(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, latent_dim)
            )
            for _ in range(num_blocks)
        ])

    def forward(self, hidden_state, noise):
        # Iteratively refine prediction
        z_pred = noise
        for block in self.blocks:
            z_pred = block(torch.cat([hidden_state, z_pred], dim=-1)) + z_pred
        return z_pred
```

**4. Train with Energy Score Loss**

Use the Energy Score, a strictly proper scoring rule that measures alignment through sample distances. Combine multiple target samples from the encoder posterior with model predictions.

```python
def energy_score_loss(z_pred, z_targets_samples, z_model_samples, M=100, N=8):
    # z_targets_samples: M samples from encoder posterior [B, M, latent_dim]
    # z_model_samples: N samples from model [B, N, latent_dim]

    # Compute distances: ||z_pred - z_target||
    target_dists = torch.norm(z_pred.unsqueeze(1) - z_targets_samples, dim=2)
    # Average over target samples
    term1 = target_dists.mean(dim=1)

    # Compute distances between model samples
    pairwise_dists = torch.cdist(z_model_samples, z_model_samples)
    term2 = pairwise_dists.mean()

    return term1.mean() - 0.5 * term2
```

**5. Implement Discrete Grounding Loop**

Despite conceptual elegance, continuous inputs are unstable. Use discrete feedback: predict continuous vector → decode to tokens → embed and compress for next step. This maintains a discrete loop while gaining efficiency benefits.

```python
def generate_continuous(model, initial_tokens, max_steps, k=4):
    current_tokens = initial_tokens
    for step in range(max_steps):
        # Compress current tokens to latent vectors
        z = autoencoder.encode_tokens(current_tokens)

        # Predict next latent vector
        z_next = model(z)  # continuous prediction

        # Decode back to tokens
        logits = autoencoder.decode_chunk(z_next)
        next_tokens = logits.argmax(dim=-1)

        # Append and continue
        current_tokens = torch.cat([current_tokens, next_tokens], dim=1)
    return current_tokens
```

## Practical Guidance

**When to Use**:
- Long-context generation where token-level prediction becomes bottleneck
- Settings requiring controllable sampling with temperature calibration
- Tasks benefiting from K-fold speedup in generation steps (K=4 typical)

**Hyperparameters**:
- K (compression factor): 4 tokens per vector (trade-off between compression and reconstruction)
- latent_dim: 128 for K=4 (scales with compression ratio)
- β (KL weight): 0.001 for balancing reconstruction and regularization
- λ_KL (clipping threshold): 0.5 per dimension

**When NOT to Use**:
- Short-form generation where token-level precision matters more than speed
- Tasks requiring byte-level control or character-level reasoning
- Systems where latency from autoencoder encoding/decoding is unacceptable

**Pitfalls**:
- **Insufficient autoencoder fidelity**: If reconstruction error exceeds 0.1%, downstream generation fails
- **Posterior collapse**: KL clipping is essential; without it, latent dimensions become uninformative
- **Discrete grounding inefficiency**: Encoding tokens at each step can negate speedup gains; batch large chunks

**Key Implementation Detail**: The discrete grounding loop is essential despite seeming inefficient. Direct continuous inputs cause small perturbations to produce completely different token sequences, breaking training stability. The discrete feedback provides stable anchor points.

## Reference

arXiv: https://arxiv.org/abs/2510.27688
