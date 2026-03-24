---
name: weight-decay-plasticity
title: "Weight Decay Improves Language Model Plasticity"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.11137"
keywords: [Language Model Pretraining, Hyperparameter Optimization, Plasticity, Fine-Tuning, Regularization]
description: "Improve downstream task performance by increasing weight decay during pretraining (0.3-1.0 vs. default 0.1), enabling better model plasticity and representation structure."
---

# Weight Decay Improves Language Model Plasticity

## Problem Context

LLM development optimizes pretraining hyperparameters based solely on validation loss, ignoring downstream adaptability. Models with similar pretraining loss may perform differently after fine-tuning—a critical gap between optimizing for pretraining performance versus post-training performance.

## Core Concept

**Higher weight decay during pretraining improves model plasticity**—the ability of pretrained models to adapt effectively to downstream tasks. Models pretrained with higher weight decay (0.3–1.0) demonstrate superior downstream task performance compared to the standard default of 0.1, even when they achieve higher pretraining validation loss.

## Architecture Overview

Three mechanistic effects explain improved plasticity:

- **Representation Structure**: Weight decay encourages linearly separated representations, making internal features more interpretable and transferable
- **Attention Matrix Regularization**: Higher weight decay reduces attention matrix rank, preventing overfitting to high-dimensional pretraining noise
- **Overfitting Reduction**: Increased weight decay reduces train-validation gap, indicating less memorization and greater flexibility

## Implementation

The adjustment is minimal—modify weight decay hyperparameter during pretraining:

```python
# Standard pretraining with default weight decay
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.1  # Default (suboptimal for plasticity)
)

# Improved pretraining with higher weight decay
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.5  # Recommended (0.3-1.0 range)
)

# Training loop remains unchanged
for epoch in range(num_epochs):
    for batch in train_loader:
        logits = model(batch)
        loss = cross_entropy(logits, batch['labels'])
        loss.backward()
        optimizer.step()
```

**Key implementation considerations:**

1. **Sweep weight decay**: Test range [0.1, 0.3, 0.5, 0.7, 1.0] during pretraining
2. **Track both metrics**: Monitor pretraining validation loss AND downstream task performance
3. **Model families**: Effect holds across Llama-2, OLMo-2; likely generalizes
4. **Model sizes**: Tested from 0.5B to 4B parameters; effect consistent across scales
5. **Training regimes**: Effect observed in both 20 and 140 tokens-per-parameter settings

## Practical Guidance

**When to use**: Always consider higher weight decay if pretraining models. The improvement in downstream plasticity outweighs slightly higher pretraining loss for nearly all use cases.

**Hyperparameter selection**: Start with 0.3–0.5; evaluate on held-out downstream tasks. Higher values (0.7+) provide diminishing returns but may improve representation structure.

**Downstream evaluation**: Test on diverse CoT (Chain-of-Thought) tasks (math, reasoning, etc.). Plasticity benefits most visible on complex reasoning tasks.

**Interaction with other hyperparameters**: Weight decay effect is largely orthogonal to learning rate, batch size, and other training parameters. No major conflicts observed.

**Regularization stacking**: If already using other regularization (dropout, layer normalization), weight decay still provides independent benefit.

## Mechanistic Understanding

**Linear separability**: Higher weight decay encourages features to cluster into linearly separable groups. This can be detected via:

```python
def measure_linear_separability(model, dataset):
    """Compute linear probing accuracy as proxy for separability"""
    hidden_states = extract_hidden_states(model, dataset)
    # Train linear classifier on frozen representations
    probe = LogisticRegression()
    accuracy = cross_val_score(probe, hidden_states, dataset['labels'])
    return accuracy
```

**Attention rank reduction**: Weight decay reduces rank of attention matrices:

```python
def measure_attention_rank(model):
    """Compute rank of attention weight matrices"""
    ranks = []
    for layer in model.layers:
        W_attn = layer.attention.W_v.weight
        rank = np.linalg.matrix_rank(W_attn)
        ranks.append(rank)
    return np.mean(ranks)
```

**Train-validation gap**: Monitor overfitting during pretraining:

```python
def measure_overfitting(train_loss, val_loss):
    """Larger weight decay reduces this gap"""
    return train_loss - val_loss
```

## Reference

Experiments span multiple model families (Llama-2, OLMo-2), sizes (0.5B–4B), training regimes (20 and 140 tokens-per-parameter), and evaluation on six downstream CoT tasks. Higher weight decay consistently improves downstream performance by 2–5% even when pretraining validation loss is higher, demonstrating that plasticity is a more important objective than in-distribution pretraining performance for practical LLM development.
