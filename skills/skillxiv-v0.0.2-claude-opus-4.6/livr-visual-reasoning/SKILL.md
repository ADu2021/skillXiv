---
name: livr-visual-reasoning
title: "Latent Implicit Visual Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.21218
keywords: [vision, latent-space, reasoning, multimodal, implicit-learning]
description: "Enable richer visual reasoning in multimodal models via learnable latent tokens trained with visual bottlenecking. Introduces special tokens that implicitly learn task-relevant visual abstractions without explicit supervision, through attention masking forcing visual information through latents—improving vision-heavy tasks without task-specific annotations."
---

## Overview

LIVR addresses the limitation that multimodal models rely primarily on text-based reasoning despite visual input. Rather than requiring explicit supervision for intermediate abstractions, this method trains latent tokens implicitly through bottleneck masking.

## Core Technique

**Latent Tokens with Visual Bottlenecking:**

```python
class LatentImplicitVisualReasoning:
    def __init__(self, num_latent_tokens=8):
        # Learnable special tokens initialized randomly
        self.latent_tokens = nn.Parameter(torch.randn(num_latent_tokens, hidden_dim))
        self.model = MultimodalModel()

    def forward_with_bottleneck(self, image, text, use_bottleneck=True):
        """
        Force visual information through latent tokens via attention masking.
        """
        all_tokens = [self.latent_tokens, image_tokens, text_tokens]

        if use_bottleneck:
            # Bottleneck masking: answers can't directly see images
            # Must route through latents
            mask = create_bottleneck_mask(
                num_latents=len(self.latent_tokens),
                num_images=image_tokens.shape[0],
                num_text=text_tokens.shape[0]
            )
        else:
            # Standard attention
            mask = None

        output = self.model.forward(all_tokens, attention_mask=mask)
        return output
```

**Two-Stage Training:**

```python
def train_latent_reasoning(model, dataset):
    # Stage 1: Bottleneck learning
    for batch in dataset:
        output = model.forward_with_bottleneck(batch, use_bottleneck=True)
        loss = task_loss(output)
        loss.backward()
        optimizer.step()

    # Stage 2: Remove bottleneck
    # Latents now encode useful information
    for batch in dataset:
        output = model.forward_with_bottleneck(batch, use_bottleneck=False)
        loss = task_loss(output)
        loss.backward()
        optimizer.step()
```

## When to Use

Use when: Vision-heavy tasks, avoiding explicit supervision, cross-task generalization needed.

## References

- Learnable latent tokens for implicit learning
- Visual bottlenecking via attention masking
- End-to-end training without task-specific supervision
