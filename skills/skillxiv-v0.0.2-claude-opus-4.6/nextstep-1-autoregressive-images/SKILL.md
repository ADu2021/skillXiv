---
name: nextstep-1-autoregressive-images
title: "NextStep-1: Autoregressive Image Generation with Continuous Tokens"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.10711
keywords: [image-generation, autoregressive-modeling, continuous-tokens, flow-matching, text-to-image]
description: "Train a unified autoregressive model to generate images and text by directly handling continuous image tokens with flow matching, eliminating the need for quantization or separate diffusion models."
---

# NextStep-1: Autoregressive Image Generation with Continuous Tokens

## Core Concept

Most text-to-image models use separate pipelines: text encoders, diffusion models, and discrete image tokenizers. NextStep-1 unifies image and text generation in a single autoregressive architecture that directly processes continuous image tokens using flow matching instead of diffusion.

The key innovation is combining a large autoregressive language model (14B parameters) with a smaller flow matching head (157M parameters) to predict both discrete text tokens and continuous image tokens in a unified next-token prediction framework.

## Architecture Overview

- **Unified Autoregressive Backbone**: Single 14B-parameter transformer predicts both text and image tokens sequentially
- **Flow Matching Head**: 157M-parameter decoder with flow matching objectives for continuous image token prediction
- **Mixed Token Types**: Seamlessly handles discrete tokens (text) and continuous vectors (image features) in one sequence
- **Next-Token Prediction**: Standard language model training objective applied across both modalities
- **Efficient Inference**: Direct autoregressive generation without iterative diffusion sampling loops

## Implementation Steps

### 1. Encode Images to Continuous Tokens

Convert images into continuous token representations using a learned encoder. These are embedded directly into the autoregressive sequence without quantization.

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class ContinuousImageTokenizer(nn.Module):
    """
    Encodes images to continuous token embeddings
    Similar to VQ-VAE but without quantization
    """
    def __init__(self, image_size=256, token_dim=768, num_tokens=1024):
        super().__init__()
        self.image_size = image_size
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        # Vision encoder: convert image to feature map
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Project to continuous tokens
        self.to_tokens = nn.Linear(256 * 32 * 32, num_tokens * token_dim)

    def encode(self, images):
        """
        Args:
            images: [batch, 3, H, W] image tensors

        Returns:
            tokens: [batch, num_tokens, token_dim] continuous embeddings
        """
        features = self.encoder(images)
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)

        tokens = self.to_tokens(features)
        tokens = tokens.view(batch_size, self.num_tokens, self.token_dim)
        return tokens

    def decode(self, tokens):
        """
        Args:
            tokens: [batch, num_tokens, token_dim]

        Returns:
            images: [batch, 3, H, W] reconstructed images
        """
        batch_size = tokens.shape[0]
        flat = tokens.view(batch_size, -1)

        # Reverse projection
        features = self.to_tokens.weight.T @ flat.T  # Approximate inversion
        features = features.view(batch_size, 256, 32, 32)

        # Decoder: upsample back to image size
        decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        images = decoder(features)
        return images
```

### 2. Create Mixed Token Sequences

Combine text tokens and continuous image tokens into single sequences that the autoregressive model can process.

```python
def create_mixed_sequence(text_ids, image_tokens, tokenizer, image_token_start=50257):
    """
    Create interleaved sequence of text and image tokens.
    Format: [text_tokens] [IMAGE_START] [image_tokens] [IMAGE_END]

    Args:
        text_ids: [seq_len] text token IDs
        image_tokens: [num_image_tokens, token_dim] continuous embeddings
        tokenizer: text tokenizer
        image_token_start: special token ID marking image start

    Returns:
        input_ids: [total_seq_len] mixed token sequence (text IDs as scalars)
        embeddings: [total_seq_len, embedding_dim] actual embeddings
    """
    text_embedding_dim = 768
    image_token_dim = 768

    # Text embeddings
    text_embeddings = tokenizer.transformer.wte(text_ids)  # [seq_len, 768]

    # Insert image tokens
    input_ids = torch.cat([
        text_ids,
        torch.tensor([image_token_start], device=text_ids.device),
        torch.arange(50257 + 1, 50257 + image_tokens.shape[0] + 1,
                    device=text_ids.device),
        torch.tensor([image_token_start + 1], device=text_ids.device)
    ])

    # Combine embeddings
    image_start_embed = text_embeddings[-1:] * 0.5  # Learned marker
    embeddings = torch.cat([
        text_embeddings,
        image_start_embed,
        image_tokens,  # ← Continuous embeddings inserted directly
        image_start_embed
    ], dim=0)

    return input_ids, embeddings
```

### 3. Define the Unified Autoregressive Model

Build a transformer that predicts next tokens, treating continuous image tokens like any other embedding.

```python
class UnifiedAutoregressive(nn.Module):
    """
    14B parameter autoregressive model for text + continuous image tokens
    """
    def __init__(self, vocab_size=50257, hidden_size=1024, num_layers=24,
                 num_image_tokens=1024, image_token_dim=768):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_image_tokens = num_image_tokens

        # Text embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer decoder
        self.transformer = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=16,
                dim_feedforward=4096, batch_first=True
            ) for _ in range(num_layers)
        ])

        # Output heads for next token prediction
        self.text_head = nn.Linear(hidden_size, vocab_size)  # Discrete text tokens
        self.image_token_head = nn.Linear(hidden_size, image_token_dim)  # Continuous image

    def forward(self, input_ids, embeddings):
        """
        Args:
            input_ids: [batch, seq_len] (for masking/identifying token types)
            embeddings: [batch, seq_len, hidden_size] mixed embeddings

        Returns:
            text_logits: [batch, seq_len, vocab_size] for text token prediction
            image_logits: [batch, seq_len, image_token_dim] for image token prediction
        """
        hidden = embeddings

        # Forward through transformer layers
        for layer in self.transformer:
            hidden = layer(hidden, hidden)

        # Predict next tokens
        text_logits = self.text_head(hidden)
        image_logits = self.image_token_head(hidden)

        return text_logits, image_logits
```

### 4. Implement Flow Matching Loss for Image Tokens

Use flow matching instead of diffusion to train the continuous image token predictions.

```python
class FlowMatchingHead(nn.Module):
    """
    Flow matching objective for continuous image tokens.
    Simpler than diffusion, more stable than MSE loss.
    """
    def __init__(self, token_dim=768, hidden_size=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, token_dim)
        )

    def forward(self, target_tokens, context_hidden, t=None):
        """
        Flow matching loss: learn to map noise to data
        target: continuous image tokens [batch, num_tokens, token_dim]
        context: model hidden states [batch, seq_len, hidden_size]
        """
        batch_size, num_tokens, token_dim = target_tokens.shape

        # Flow matching: learn velocity field from noise to data
        noise = torch.randn_like(target_tokens)
        t = torch.rand(batch_size, 1, 1, device=target_tokens.device)

        # Interpolate: z_t = (1-t) * noise + t * target
        z_t = (1 - t) * noise + t * target_tokens

        # Predict velocity
        context_repeated = context_hidden[:, -num_tokens:, :].unsqueeze(1)
        velocity_pred = self.mlp(torch.cat([z_t, context_repeated.expand_as(z_t)], dim=-1))

        # Flow matching loss: velocity should match (target - noise)
        target_velocity = target_tokens - noise
        flow_loss = ((velocity_pred - target_velocity) ** 2).mean()

        return flow_loss
```

### 5. Training Loop with Mixed Objectives

Combine text prediction loss (cross-entropy) with flow matching loss for image tokens.

```python
def train_nextstep(model, flow_head, image_tokenizer, dataloader, num_epochs=10):
    """
    Train unified autoregressive model with mixed objectives
    """
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(flow_head.parameters()) +
        list(image_tokenizer.parameters()),
        lr=1e-4
    )

    for epoch in range(num_epochs):
        for batch_idx, (images, captions) in enumerate(dataloader):
            # Encode images to continuous tokens
            image_tokens = image_tokenizer.encode(images)  # [batch, num_tokens, 768]

            # Tokenize text
            text_ids = tokenizer.encode(captions)

            # Create mixed sequences
            input_ids, embeddings = create_mixed_sequence(
                text_ids, image_tokens, tokenizer
            )

            # Forward pass
            text_logits, image_logits = model(input_ids, embeddings)

            # Text prediction loss (standard cross-entropy)
            text_loss = F.cross_entropy(
                text_logits.view(-1, model.vocab_size),
                text_ids.view(-1)
            )

            # Image token prediction with flow matching
            # Extract image token portion of context
            image_context = embeddings[:, :text_ids.shape[1], :]
            flow_loss = flow_head(image_tokens, image_context)

            # Combined loss
            total_loss = text_loss + 0.5 * flow_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {batch_idx}")
                print(f"  Text Loss: {text_loss:.4f}, Flow Loss: {flow_loss:.4f}")
```

### 6. Inference and Image Generation

Use the trained model to generate images autoregressively from text prompts.

```python
@torch.no_grad()
def generate_image(model, flow_head, image_tokenizer, prompt, max_new_tokens=1024):
    """
    Generate image autoregressively from text prompt
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt)

    # Add image start token
    input_ids = torch.cat([
        input_ids,
        torch.tensor([50257], device=input_ids.device)  # IMAGE_START
    ])

    # Autoregressive generation of image tokens
    generated_tokens = []
    hidden_state = None

    for step in range(max_new_tokens):
        embeddings = model.embed_tokens(input_ids.unsqueeze(0))
        text_logits, image_logits = model(input_ids.unsqueeze(0), embeddings)

        # Sample next image token
        next_token_logits = image_logits[0, -1, :]
        next_token = torch.randn_like(next_token_logits) * 0.1 + next_token_logits

        generated_tokens.append(next_token)

        if step % 100 == 0:
            print(f"Generated {step}/{max_new_tokens} image tokens")

    # Stack generated image tokens
    image_tokens = torch.stack(generated_tokens)  # [num_tokens, 768]

    # Decode to image
    image = image_tokenizer.decode(image_tokens.unsqueeze(0))
    return image
```

## Practical Guidance

### Hyperparameters & Configuration

- **Token Dimension**: 768 (align with language model embedding size)
- **Number of Image Tokens**: 1024 (depends on desired spatial resolution)
- **Flow Matching Weight**: 0.5 relative to text loss (balance both objectives)
- **Learning Rate**: 1e-4 (conservative due to mixed objectives)
- **Batch Size**: 64-256 (computational dependent)
- **Gradient Clipping**: max_norm=1.0 (prevent divergence)

### When to Use NextStep-1 Approach

- You want unified text-to-image generation in one model
- You prefer autoregressive generation over iterative sampling
- Computational efficiency (no diffusion loops) is important
- You need both text and image generation capabilities
- You want to avoid quantization artifacts from VQ-VAE

### When NOT to Use NextStep-1

- Image quality is paramount (diffusion models still produce better results)
- You need fine-grained control over image generation (e.g., inpainting)
- Inference latency must be absolute minimum (autoregressive is slower than diffusion)
- You only need image-to-text or don't need text+image in same model
- Your computational resources are very limited

### Common Pitfalls

1. **Imbalanced Loss Weights**: If text loss dominates, image quality suffers. Use 1:0.5 ratio and monitor both.
2. **Insufficient Image Tokens**: Too few tokens (< 256) loses spatial detail. 1024 is reasonable baseline.
3. **Poor Continuous Token Learning**: Without good flow matching, image tokens become noisy. Use proper velocity field training.
4. **Ignoring Text-Image Alignment**: Text and image should be processed with awareness of each other, not independently.
5. **Slow Inference**: Autoregressive generation is 100x slower than diffusion for same image. Use strategies like speculative decoding.

## Reference

NextStep-1 (2508.10711): https://arxiv.org/abs/2508.10711

Unified autoregressive model for text and continuous image tokens with flow matching, eliminating quantization loss and diffusion overhead while enabling efficient image generation and editing.
