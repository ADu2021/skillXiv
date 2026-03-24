---
name: starflow-latent-flows
title: "STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.06276"
keywords: [generative-models, normalizing-flows, image-synthesis, transformers, latent-space]
description: "Learn to implement Transformer Autoregressive Flows for efficient high-resolution image synthesis using latent space normalization and maximum likelihood training."
---

# STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis

## Core Concept

STARFlow demonstrates the first successful application of normalizing flows at scale for high-resolution image synthesis. Unlike diffusion models that require iterative denoising, normalizing flows learn invertible transformations that map simple distributions to complex image distributions through a single forward pass, enabling end-to-end maximum likelihood training without discretization or approximation.

## Architecture Overview

- **Latent Space Modeling**: Operates in the latent space of pretrained autoencoders rather than directly on pixels, reducing computational requirements
- **Transformer Autoregressive Flow (TARFlow)**: Core component combining normalizing flows with autoregressive Transformers
- **Deep-Shallow Hybrid Design**: Deep Transformer block for representational capacity, shallow blocks for efficiency
- **Conditional Generation**: Supports both class-conditional and text-conditional image generation
- **Novel Sampling Algorithm**: Custom guidance technique that enhances output quality during inference

## Implementation

### Step 1: Set Up Latent Autoencoder

Obtain a pretrained autoencoder (such as VAE-based models) that encodes images to latent vectors:

```python
import torch
import torch.nn as nn

class LatentEncoder(nn.Module):
    def __init__(self, encoder_checkpoint):
        super().__init__()
        self.encoder = load_pretrained_autoencoder(encoder_checkpoint)

    def encode(self, images):
        """Encode images to latent space, shape: (B, C, H, W) -> (B, L, D)"""
        with torch.no_grad():
            latents = self.encoder.encode(images)
        return latents.flatten(1, -1)  # Flatten spatial dims to sequence

class LatentDecoder(nn.Module):
    def __init__(self, decoder_checkpoint):
        super().__init__()
        self.decoder = load_pretrained_autoencoder(decoder_checkpoint)

    def decode(self, latents, shape):
        """Decode latent vectors back to images"""
        return self.decoder.decode(latents.reshape(shape))
```

### Step 2: Implement Transformer Autoregressive Flow

Create the core TARFlow module combining autoregressive dependencies with flow transformations:

```python
class TransformerAutoregressive Flow(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Deep block for capacity
        self.deep_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True),
            num_layers=num_layers
        )

        # Shallow blocks for efficiency
        self.shallow_transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True)
            for _ in range(4)
        ])

        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_prev, condition=None):
        """Compute flow transformation for next token"""
        x = self.embedding(x_prev)

        # Pass through deep transformer
        x = self.deep_transformer(x)

        # Pass through shallow blocks sequentially
        for shallow in self.shallow_transformers:
            x = shallow(x)

        logits = self.output_projection(x)
        return logits
```

### Step 3: Maximum Likelihood Training Loop

Implement the training objective that directly optimizes the flow model's likelihood:

```python
def train_normalizing_flow(model, encoder, decoder, dataloader, optimizer, epochs=100):
    """Train with exact maximum likelihood on continuous latent space"""

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in dataloader:
            # Encode images to latent space
            latents = encoder.encode(images)  # Shape: (B, L)

            # Quantize to discrete tokens for autoregressive training
            tokens = quantize_latents(latents, vocab_size=4096)

            # Compute log probability under flow model
            log_probs = []
            for t in range(tokens.shape[1]):
                prev_tokens = tokens[:, :t]
                target_token = tokens[:, t]

                logits = model(prev_tokens, condition=labels)
                log_prob = torch.log_softmax(logits[:, -1, :], dim=-1)
                log_probs.append(log_prob.gather(1, target_token.unsqueeze(1)))

            # Sum log probabilities across sequence
            loss = -torch.cat(log_probs, dim=1).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
```

### Step 4: Implement Sampling with Custom Guidance

Create the inference-time sampling algorithm for high-quality generation:

```python
def sample_with_guidance(model, encoder, decoder, condition, guidance_scale=7.5,
                         max_length=256, temperature=1.0):
    """Sample from flow model with custom guidance algorithm"""

    batch_size = condition.shape[0]
    tokens = torch.zeros(batch_size, max_length, dtype=torch.long, device=condition.device)

    for t in range(max_length):
        # Get logits for next token
        with torch.no_grad():
            logits = model(tokens[:, :t], condition=condition)

        # Apply temperature scaling
        logits = logits[:, -1, :] / temperature

        # Guidance: blend conditional and unconditional predictions
        with torch.no_grad():
            logits_uncond = model(tokens[:, :t], condition=None)[:, -1, :]

        logits = logits_uncond + guidance_scale * (logits - logits_uncond)

        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        tokens[:, t] = next_tokens

    # Decode tokens back to images
    latents = dequantize_tokens(tokens, vocab_size=4096)
    images = decoder.decode(latents, shape=(batch_size, channels, height, width))

    return images
```

## Practical Guidance

- **Latent Space Selection**: Choose autoencoders with good reconstruction quality and reasonable latent dimension (typically 4-16 dims per spatial patch)
- **Vocabulary Size**: Balance between expressiveness and computational cost; 4096 tokens is a good starting point
- **Architecture Scaling**: Use deeper Transformers for better quality, shallower ones for efficiency; experiment with layer ratios
- **Guidance Strength**: Start with guidance_scale=7.5; higher values increase condition adherence but may reduce diversity
- **Training Data**: Requires large-scale image datasets; consider pretraining on text-image pairs for text-conditional models
- **Inference Speed**: Flows are faster than diffusion models (single forward pass vs. iterative steps), but slower than autoregressive pixel-based models

## Reference

- Flow-based generative modeling provides exact log-likelihood computation without variational bounds
- Autoregressive decomposition enables efficient training of high-dimensional distributions
- Latent space modeling reduces pixel-level complexity while maintaining visual quality
- Transformer architectures provide the flexibility needed for both image and conditional tokens
