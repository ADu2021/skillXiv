---
name: growing-transformers-modular-expansion
title: "Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07129"
keywords: [Constructive Learning, Layer-wise Training, Parameter Efficiency, LLM Scaling]
description: "Build language models layer-by-layer on frozen embeddings, training new Transformer blocks sequentially while keeping lower layers frozen. Achieves 5% improvement over monolithic baselines on MMLU while fitting 740M trainable parameters per stage on single GPUs, enabling resource-efficient incremental scaling to 2.3B parameters."
---

# Growing Transformers: Progressive Model Construction on Frozen Foundations

Traditional large language model training requires end-to-end gradient flow through billions of parameters, demanding massive compute clusters and suffering from optimization instabilities. Growing Transformers inverts this approach through constructive learning: build models incrementally by stacking Transformer layers on a fixed embedding foundation, training each new layer to convergence before freezing and adding the next. This strategy reduces memory requirements, enables single-GPU training of large models, and paradoxically improves performance by ~5% compared to training entire models from scratch.

The core insight is that semantic understanding emerges from architectural depth and composition rather than sophisticated input embeddings. By freezing random or minimally-trained embeddings, you force each subsequent layer to learn meaningful transformations from the layer below, creating a strong inductive bias toward hierarchical feature extraction.

## Core Concept

Growing Transformers uses constructive training—a two-component strategy:

1. **Layer-wise Sequential Addition**: Start with a frozen embedding layer and trained lower layers. Add one Transformer block at a time, train it to convergence in isolation, then freeze before adding the next.
2. **Holistic Fine-tuning via LoRA**: After building the full stack, apply Low-Rank Adaptation (LoRA) across all layers for final performance gains without retraining from scratch.

Each new layer sees identical training conditions: fixed representations from below and a standard language modeling objective. The model grows incrementally while remaining trainable on single GPUs, then benefits from whole-model fine-tuning after construction completes.

## Architecture Overview

- **Frozen Embedding Layer**: Random or pretrained embeddings (trainable dimension: 0 or 1B parameters) that serve as the non-learnable base
- **Sequential Transformer Blocks**: Standard 2-layer → 3-layer → 4-layer progression, each trained independently until convergence
- **LoRA Fine-tuning Head**: Low-rank weight matrices applied to attention and feed-forward layers across the full stack
- **Training Scheduler**: Per-layer convergence detection (loss plateau) with early stopping to determine when to add next layer
- **Unified Tokenizer**: Shared token vocabulary throughout training (no vocabulary shifts during growth)
- **Checkpoint Management**: Persistent storage of frozen layers during sequential training to enable resumption

## Implementation

The following demonstrates layer-by-layer training with checkpoint management:

```python
import torch
import torch.nn as nn
from typing import Optional, List

class FrozenEmbedding(nn.Module):
    """Non-trainable embedding layer as training foundation."""
    def __init__(self, vocab_size: int, embed_dim: int, initialize_random: bool = True):
        super().__init__()
        if initialize_random:
            self.embed = nn.Embedding(vocab_size, embed_dim)
            # Freeze immediately—no training signal flows through
            for param in self.embed.parameters():
                param.requires_grad = False
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, input_ids):
        return self.embed(input_ids)

class TransformerBlock(nn.Module):
    """Single Transformer layer for independent training."""
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.norm2(x)
        return x

class GrowingTransformer(nn.Module):
    """Model that grows layer-by-layer on frozen embeddings."""
    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.embedding = FrozenEmbedding(vocab_size, hidden_dim, initialize_random=True)
        self.blocks = nn.ModuleList()  # Grows during training
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Initialize with first layer
        self.add_block()

    def add_block(self):
        """Add a new trainable Transformer layer."""
        new_block = TransformerBlock(self.hidden_dim, self.num_heads, self.ff_dim)
        self.blocks.append(new_block)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return x

def train_single_layer(model: GrowingTransformer, layer_idx: int, train_loader,
                       optimizer, criterion, num_epochs: int = 100,
                       patience: int = 5) -> float:
    """Train a single layer while freezing all others."""
    # Freeze all blocks except the one being trained
    for i, block in enumerate(model.blocks):
        for param in block.parameters():
            param.requires_grad = (i == layer_idx)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass through all layers (but only latest layer trains)
            logits = model(input_ids)
            logits = logits.view(-1, model.hidden_dim)  # Flatten for language modeling head
            target = target_ids.view(-1)

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Layer {layer_idx} converged at epoch {epoch} with loss {best_loss:.4f}")
            break

    return best_loss

def sequential_growth(initial_model: GrowingTransformer, train_loader,
                      num_growth_steps: int = 5, epochs_per_layer: int = 100):
    """Grow model by repeatedly adding and training layers."""
    for step in range(num_growth_steps):
        print(f"\n=== Growth Step {step + 1}/{num_growth_steps} ===")

        # Add new layer
        if step > 0:  # First layer already exists
            initial_model.add_block()

        # Train newly added layer
        optimizer = torch.optim.AdamW(initial_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        layer_loss = train_single_layer(
            initial_model,
            layer_idx=step,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=epochs_per_layer,
            patience=5
        )

        # Freeze current layer before proceeding
        for param in initial_model.blocks[step].parameters():
            param.requires_grad = False

        print(f"Layer {step} frozen. Model now has {len(initial_model.blocks)} layers")

    return initial_model

# Example usage
def apply_lora_finetuning(model: GrowingTransformer, train_loader, lr: float = 1e-3, epochs: int = 10):
    """Apply LoRA fine-tuning across all layers after growth."""
    # Add LoRA parameters to attention layers
    for block in model.blocks:
        # Simplified: in practice use peft library for proper LoRA
        block.lora_q = nn.Linear(model.hidden_dim, 8)  # Low-rank projection
        block.lora_v = nn.Linear(8, model.hidden_dim)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    for epoch in range(epochs):
        total_loss = 0
        for input_ids, target_ids in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = nn.CrossEntropyLoss()(logits.view(-1, model.hidden_dim), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"LoRA Fine-tuning Epoch {epoch + 1}: Loss {total_loss / len(train_loader):.4f}")
```

This implementation demonstrates the core constructive learning pipeline: sequentially adding layers, training each in isolation, freezing, and then fine-tuning the complete stack.

## Practical Guidance

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| **Frozen Embedding Initialization** | Random or uniform [-0.1, 0.1] | Confirms non-semantic role |
| **Layers to Add** | 6-12 for 2.3B model | Balance training time vs. final performance |
| **Training per Layer** | 100-200 epochs | Use early stopping on validation plateau |
| **LoRA Rank** | 8-32 | Lower rank for constrained memory |
| **LoRA Alpha** | 16 (or 2× rank) | Scaling factor for LoRA contribution |
| **Batch Size** | 512-2048 | Single-GPU training benefits from large batches |
| **Learning Rate** | 1e-4 (layer training), 1e-3 (LoRA) | Decay during layer training |

### When to Use Growing Transformers

- **Resource-constrained training**: Fitting billion-parameter models on single GPUs
- **Incremental model deployment**: Adding capacity to existing models without retraining from scratch
- **Efficient scaling experiments**: Quickly exploring 1B, 2B, 3B variants with minimal overhead
- **Distributed training pipelines**: Different layers can train on different GPUs without synchronization
- **Research on layer dynamics**: Analyzing how representational capacity evolves during sequential growth

### When NOT to Use

- **State-of-the-art performance**: End-to-end training still achieves slightly higher performance on massive datasets
- **Models <100M parameters**: Overhead of layer management exceeds benefits; use standard training
- **Recurrent or dynamic architectures**: Constructive learning assumes layer independence; doesn't apply to RNNs or adaptive models
- **Multi-task or continual learning** with shared task-specific heads: Frozen embeddings may bottleneck task adaptation
- **Tight inference latency requirements**: Layer management and checkpoint switching add minimal but measurable overhead

### Common Pitfalls

1. **Embedding Dimension Too Small**: Frozen embeddings with insufficient capacity become a bottleneck. Use hidden_dim >= 512 for models >1B parameters.
2. **Training New Layers Too Long**: Overtraining a single layer before adding the next reduces diversity. Use early stopping aggressively.
3. **Skipping LoRA Fine-tuning**: Layer-wise training creates suboptimal inter-layer interactions. Always perform end-to-end LoRA fine-tuning afterward.
4. **Learning Rate Schedule Mismatch**: Using identical LR for all layers causes instability. Freeze layers with larger learning rates to stabilize training.
5. **Ignoring Validation Performance**: Monitor validation loss during layer training to detect when adding the next layer is premature.

## Reference

Ke, Y., Saha, S., et al. (2025). Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate. *arXiv preprint arXiv:2507.07129*.

Available at: https://arxiv.org/abs/2507.07129
