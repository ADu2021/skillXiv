---
name: nepa-next-embedding-prediction
title: "NEPA: Next-Embedding Prediction as Self-Supervised Visual Pretraining for Vision Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.16922
keywords: [self-supervised-learning, vision-transformers, next-embedding-prediction, autoregressive, visual-pretraining]
description: "Train vision transformers through autoregressive next-embedding prediction without pixel reconstruction, tokenizers, or contrastive losses. Apply causal masking and stop-gradient on target embeddings. Achieve 83.8% (ViT-B) and 85.3% (ViT-L) ImageNet-1K accuracy with strong transfer to downstream tasks."
---

## Skill Summary

NEPA introduces Next-Embedding Predictive Autoregression, a self-supervised visual pretraining method training vision transformers to "predict future patch embeddings conditioned on past ones, using causal masking and stop gradient." Rather than learning static representations or reconstructing pixels, the approach adopts generative prediction inspired by language models. Patch sequences are processed autoregressively with normalized cosine similarity training objective, achieving competitive downstream performance without discrete tokenizers or contrastive losses.

## When To Use

- Visual representation learning without pixel-level reconstruction
- Scenarios where autoregressive prediction paradigm suits your downstream tasks
- Projects exploring language-model-inspired vision pre-training
- Research on generative vs. contrastive vs. reconstruction-based approaches

## When NOT To Use

- Domains benefiting specifically from pixel-level detail preservation
- Applications where contrastive learning already works well
- Scenarios where autoregressive inductive bias mismatches task requirements
- Models with strict parameter budgets (may require separate large models)

## Core Technique

The method adopts generative prediction paradigm from language models:

**1. Patch Embedding Sequences**
Divide images into patches and embed them into sequences using standard vision transformer embedding layers. Create input sequences representing spatial information through patch order.

**2. Autoregressive Prediction**
Model predicts next embedding from previous ones in sequence:
- Forward pass with causal masking to enforce directionality
- Use stop-gradient on target embeddings to prevent collapse

Unlike typical contrastive or reconstruction approaches, this embraces next-token prediction as primary learning signal.

**3. Training Objective**
Use normalized cosine similarity without requiring:
- Pixel reconstruction (unlike MAE)
- Discrete tokenizers (unlike VQ-VAE)
- Contrastive losses (unlike SimCLR/BYOL)

This simplified objective reduces computational overhead while maintaining learning signal.

**4. Results**
- ViT-B: 83.8% ImageNet-1K top-1 accuracy
- ViT-L: 85.3% ImageNet-1K top-1 accuracy
- Strong transfer to semantic segmentation (ADE20K) and other downstream tasks

## Implementation Notes

Use standard ViT architecture with causal masking in self-attention. Implement stop-gradient on target patch embeddings. Use cosine similarity with L2 normalization as training objective. Train on ImageNet-scale datasets. Evaluate transfer performance on multiple downstream tasks.

## References

- Original paper: Next-Embedding Prediction (Dec 2025)
- Vision transformers (ViT)
- Self-supervised visual learning
