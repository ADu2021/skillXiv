---
name: single-matrix-lora
title: "SingLoRA: Low Rank Adaptation Using a Single Matrix"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.05566"
keywords: [LoRA, Parameter Efficiency, Fine-Tuning, Gradient Stability, Model Adaptation]
description: "Replace LoRA's two-matrix decomposition with a single learnable matrix (AA⊤) to eliminate scale imbalances and improve training stability. Reduces parameters by ~50% while maintaining or exceeding LoRA performance."
---

# SingLoRA: Stable and Efficient Rank Adaptation with Single-Matrix Decomposition

Low-rank adaptation (LoRA) enables efficient fine-tuning by adding a small trainable component while freezing base model weights. However, LoRA's two-matrix factorization (W₀ + BA) suffers from scale disparities during training, causing gradient instability and suboptimal convergence. SingLoRA replaces the asymmetric two-matrix design with a symmetric single-matrix approach (W₀ + AA⊤), which inherently eliminates inter-matrix scaling conflicts while halving trainable parameters and guaranteeing mathematically-grounded training stability.

The core insight is that gradient descent's behavior depends on parameter scaling, but LoRA's two matrices have no principled relationship. Using a symmetric update (AA⊤) removes these pathological scaling interactions and provides transformation-invariance guarantees that standard LoRA lacks.

## Core Concept

SingLoRA replaces LoRA's weight update from W₀ + BA to W₀ + AA⊤, where A is a single low-rank matrix. This seemingly small change provides several key benefits:

1. **Parameter efficiency**: Uses roughly half the parameters of LoRA (single matrix vs. two)
2. **Stable gradients**: Symmetric structure eliminates scale imbalances that plague LoRA
3. **Transformation-invariant optimization**: Equivalent parameterizations produce identical optimizer updates, preventing optimization pathologies
4. **Standard optimizer compatibility**: Works with SGD and Adam without special tuning

The mathematical insight is that LoRA's BA decomposition allows arbitrary scaling (scale A by α, scale B by 1/α), and different scales yield different gradients despite representing the same weight update. SingLoRA's AA⊤ has no such degree of freedom.

## Architecture Overview

- **Single low-rank matrix A**: Learnable m × r matrix where r is the rank
- **Symmetric update**: Weight perturbation is AA⊤, automatically symmetric
- **Non-square extension**: Handles non-square weight matrices via truncation
- **Gradient computation**: Efficiently computed via chain rule without explicit AA⊤ formation
- **Integration with optimizers**: Drop-in replacement for LoRA in fine-tuning frameworks

## Implementation

Initialize and apply SingLoRA to a linear layer:

```python
import torch
import torch.nn as nn

class SingLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Single learnable matrix (initialize small)
        self.A = nn.Parameter(
            torch.randn(out_features, rank) * 0.01
        )

    def forward(self, x):
        # Compute AA^T
        # Weight update: W = W_0 + A @ A^T
        lora_out = x @ self.A @ self.A.T
        return lora_out
```

Apply SingLoRA to a transformer layer by wrapping linear projections:

```python
from transformers import AutoModel
import torch.nn as nn

model = AutoModel.from_pretrained("meta-llama/Llama-2-7b")

def add_singloRA(model, rank=8):
    """Add SingLoRA to all linear layers in model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Wrap with SingLoRA adapter
            original_forward = module.forward

            def create_forward(orig_module, lora):
                def forward(x):
                    return orig_module.weight @ x.T + lora(x)
                return forward

            lora_adapter = SingLoRA(
                module.in_features,
                module.out_features,
                rank=rank
            )

            # Replace forward pass
            module.lora = lora_adapter
            module.forward = lambda x: (
                nn.functional.linear(x, module.weight, module.bias) +
                lora_adapter(x)
            )

add_singloRA(model, rank=16)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
lora_params = sum(
    p.numel() for p in model.parameters() if "lora" in str(p)
)
print(f"LoRA params: {lora_params:,} ({100*lora_params/total_params:.2f}%)")
```

Fine-tune with standard optimizers without modification:

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# Freeze base model, train only SingLoRA
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4  # Standard learning rate; no special tuning needed
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        # SingLoRA gradients are stable; no special handling required
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
```

Handle non-square weight matrices via truncation:

```python
class SingLoRATruncated(nn.Module):
    """SingLoRA for non-square matrices (e.g., out_features > in_features)."""

    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Single matrix truncated to match weight shape
        self.A = nn.Parameter(
            torch.randn(out_features, rank) * 0.01
        )

    def forward(self, x):
        # AA^T produces out_features x out_features
        # Implicitly handles shape mismatch by truncation
        perturbation = self.A @ self.A.T
        # Apply only to first in_features columns
        return x @ perturbation[:, :self.in_features].T
```

## Practical Guidance

### When to Use SingLoRA

Use SingLoRA when:
- Parameter efficiency is critical (mobile, edge deployment)
- Fine-tuning stability is a concern
- You're already considering LoRA but want better performance
- Multiple adaptation layers would explode parameter count
- You need guaranteed convergence properties

### When NOT to Use

Avoid SingLoRA for:
- Tasks where rank is already extremely small (r < 2)
- Scenarios requiring asymmetric adaptation
- Cases where LoRA performance is already sufficient and deployment is frozen
- Applications requiring per-layer scaling flexibility

### Rank Selection Guide

| Model Size | Task | Recommended Rank |
|-----------|------|------------------|
| 7B | Instruction tuning | 8-16 |
| 7B | Domain adaptation | 4-8 |
| 13B | Instruction tuning | 16-32 |
| 70B | Domain specialization | 8-16 |
| 70B | Full fine-tuning | 32-64 |

Start at rank 8; increase if validation loss plateaus early, decrease if training is unstable.

### Performance Characteristics

| Aspect | vs. LoRA |
|--------|---------|
| Parameters | ~50% reduction |
| Training time | Similar or faster |
| Convergence stability | Provably better |
| Final accuracy | Equal or better |
| Inference cost | Negligible difference |

### Critical Hyperparameters

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| Rank (r) | 4-64 | Higher rank = more capacity but slower |
| Learning rate | 1e-4 to 5e-4 | Standard LoRA rates work |
| Initialization scale | 0.001-0.01 | Small initialization stabilizes training |
| Gradient clipping | 1.0 | Recommended even though stable |

### Common Pitfalls

1. **Ignoring the symmetric constraint**: Don't try to regularize A and A^T separately; they're coupled.
2. **Initializing A too large**: Large initialization causes training instability despite mathematical guarantees.
3. **Forgetting to freeze base model**: Only A should be trainable; all base parameters must be frozen.
4. **Misunderstanding stability claims**: Transformation-invariance prevents certain pathologies but doesn't eliminate all optimization challenges.
5. **Rank too small**: Insufficient capacity leads to underfitting. Validate carefully.

### Gradient Flow Analysis

SingLoRA provides gradient stability because:
- Symmetric structure ensures eigenvalue stability
- No inter-matrix scaling degrees of freedom
- Parameter initialization directly controls effective learning rate

This contrasts with LoRA, where scale imbalances can cause vanishing or exploding gradients even with standard initializations.

## Reference

"SingLoRA: Low Rank Adaptation Using a Single Matrix" - [arXiv:2507.05566](https://arxiv.org/abs/2507.05566)
