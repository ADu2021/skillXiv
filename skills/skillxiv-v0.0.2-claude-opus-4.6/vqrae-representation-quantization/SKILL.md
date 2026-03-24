---
name: vqrae-representation-quantization
title: "VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2511.23386
keywords: [quantization autoencoders, multimodal learning, discrete tokens, vector quantization, unified tokenizer]
description: "Unify multimodal understanding, generation, and reconstruction using high-dimensional codebooks for semantic information. VQRAE achieves 100% codebook utilization at 1536 dimensions—ideal when you need a single tokenizer for vision-language tasks."
---

## Overview

VQRAE combines continuous semantic representations and discrete tokens in a single framework through high-dimensional vector quantization autoencoders. Two-stage training first learns semantic quantization, then jointly optimizes with self-distillation for all tasks.

## When to Use

- Unified multimodal tokenization (understanding + generation + reconstruction)
- Visual semantic understanding and generation
- Need for single tokenizer across multiple tasks
- High-dimensional codebooks for semantic information
- Autoregressive model training on multimodal data

## When NOT to Use

- Task-specific tokenizers already optimal
- Scenarios where separate understanding/generation models work
- Limited codebook dimension resources

## Core Technique

High-dimensional vector quantization for semantic tokens:

```python
# VQRAE: Unified multimodal tokenization
class RepresentationQuantizationAutoencoder:
    def __init__(self, codebook_dim=1536):
        self.encoder = nn.Sequential(
            # Visual feature extraction with ViT
            VisionTransformer(pretrained=True),
            nn.Linear(768, 512)
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, image_size * image_size * 3)
        )

        # High-dimensional codebook
        self.codebook_dim = codebook_dim
        self.codebook = nn.Embedding(
            codebook_dim,
            embedding_dim=512
        )

    def encode_with_quantization(self, image):
        """Encode to discrete tokens."""
        # Continuous features
        features = self.encoder(image)

        # Quantize to discrete tokens
        token_indices = torch.argmin(
            torch.cdist(features.unsqueeze(1), self.codebook.weight),
            dim=2
        ).squeeze(1)

        return token_indices

    def two_stage_training(self, image_dataset):
        """
        Stage 1: Learn semantic codebook with pixel reconstruction
        Stage 2: Joint optimization with self-distillation
        """
        # Stage 1: Semantic codebook learning
        for batch in image_dataset:
            # Encode
            features = self.encoder(batch)

            # Quantize
            token_indices = self.quantize(features)
            quantized_features = self.codebook(token_indices)

            # Reconstruct pixels
            reconstructed = self.decoder(quantized_features)

            # Loss: reconstruction fidelity
            recon_loss = torch.nn.functional.mse_loss(
                reconstructed,
                batch
            )

            recon_loss.backward()
            self.optimizer.step()

        # Stage 2: Joint optimization with self-distillation
        for batch in image_dataset:
            # Encode with teacher (frozen original encoder)
            teacher_features = self.teacher_encoder(batch)

            # Encode with student (quantized)
            student_tokens = self.encode_with_quantization(batch)
            student_features = self.codebook(student_tokens)

            # Distillation loss: align student with teacher
            distill_loss = torch.nn.functional.mse_loss(
                student_features,
                teacher_features.detach()
            )

            # Reconstruction loss
            reconstructed = self.decoder(student_features)
            recon_loss = torch.nn.functional.mse_loss(
                reconstructed,
                batch
            )

            total_loss = distill_loss + recon_loss
            total_loss.backward()
            self.optimizer.step()

    def quantize(self, features):
        """Vector quantization with high-dimensional codebook."""
        # Flatten batch dimension
        flat_features = features.reshape(-1, features.shape[-1])

        # Find nearest codebook entries
        distances = torch.cdist(
            flat_features.unsqueeze(1),
            self.codebook.weight.unsqueeze(0)
        )

        indices = torch.argmin(distances, dim=2).squeeze(1)

        return indices.reshape(features.shape[:-1])

    def compute_codebook_utilization(self, image_batch):
        """Measure codebook usage efficiency."""
        tokens = self.encode_with_quantization(image_batch)
        unique_tokens = len(torch.unique(tokens))

        utilization = unique_tokens / self.codebook_dim

        return utilization
```

## Key Results

- 100% codebook utilization at 1536 dimensions
- Competitive across understanding, generation, reconstruction
- High-dimensional codebooks capture semantic info
- Favorable autoregressive scaling

## References

- Original paper: https://arxiv.org/abs/2511.23386
- Focus: Unified multimodal tokenization
- Domain: Vision-language models, tokenization
