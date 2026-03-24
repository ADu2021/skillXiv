---
name: muon-optimizer-tail-memory-learning
title: "Muon Optimizer: Selective Parameter Optimization for Associative Memory"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2509.26030
keywords: [optimizer, muon, transformers, associative-memory, efficiency]
description: "Improve LLM training efficiency by selectively applying Muon optimizer to Value-Output attention weights and FFN layers, which function as associative memories. Use when training data exhibits heavy-tailed distributions requiring robust rare-fact learning."
---

# Muon Optimizer: Selective Parameter Optimization for Associative Memory

This research explains why Muon optimizer outperforms Adam through mechanistic analysis. Muon excels at learning transformer components functioning as associative memories—specifically Value-Output attention and FFNs—particularly when training data has heavy-tailed distributions where rare facts appear infrequently.

## Core Architecture

- **Selective application**: Apply Muon to VO+FFN, Adam to other parameters
- **Associative memory insight**: VO weights and FFNs implement content-addressable memory
- **Heavy-tailed robustness**: Muon handles rare but important facts better than Adam
- **Computational efficiency**: Near-full gains with only 40% of parameters using Muon

## Implementation Steps

Apply selective optimizer assignment to transformer layers:

```python
# Create mixed optimizer configuration targeting specific components
from torch.optim import Adam
from muon_optim import Muon

def create_selective_optimizer(model, learning_rate=1e-3):
    """Assign Muon to VO+FFN, Adam to remaining parameters"""

    vo_ffn_params = []
    other_params = []

    for name, param in model.named_parameters():
        if any(x in name for x in ["value_out", "ffn"]):
            vo_ffn_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {
            "params": vo_ffn_params,
            "optimizer": Muon,
            "lr": learning_rate,
            "momentum": 0.95
        },
        {
            "params": other_params,
            "optimizer": Adam,
            "lr": learning_rate,
            "betas": (0.9, 0.999)
        }
    ])

    return optimizer
```

Execute training with mixed optimization:

```python
# Standard training loop with selective optimizer
optimizer = create_selective_optimizer(model)

for step, (x, y) in enumerate(dataloader):
    logits = model(x)
    loss = compute_loss(logits, y)

    # Standard backprop; optimizer handles mixed updates
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Practical Guidance

**When to use selective Muon:**
- Training language models with heavy-tailed knowledge distributions
- Models requiring robust handling of rare facts or long-tail phenomena
- Scenarios where compute budget allows selective optimization
- Tasks combining frequent patterns with rare exceptions

**When NOT to use:**
- Uniform data distributions (Adam sufficient)
- Domains where all parameters equally important
- When full-model Muon training already optimal
- Real-time deployment with strict latency constraints

**Hyperparameters:**
- **Momentum (0.95)**: Standard for Muon; increase to 0.97 for slower convergence, decrease to 0.9 for stability
- **Learning rate**: Keep Muon and Adam learning rates equal initially; adjust Muon upward by 1.5x if underfitting
- **Parameter allocation**: 40% Muon (VO+FFN) optimal across scales; test 30-50% range for different architectures

**Component selection:**
- **VO (Value-Output) weights**: Always include; these implement fact retrieval
- **FFN layers**: Include entirely or FFN-up only depending on model design
- **Attention scores**: Keep with Adam; norm-based updates less suitable
- **Embeddings**: Keep with Adam; Muon offers limited benefit

## Architecture Notes

The VO+FFN pair together form transformer associative memory:
- VO weights map from attention-computed queries to output space
- FFN layers implement non-linear fact association
- Heavy-tailed data (rare facts) benefits from Muon's orthonormality-preserving updates

## Performance Improvements

- **Near-full gains** (~95% of single-Muon performance) with selective optimization
- **Faster training** due to reduced Muon computation on fewer parameters
- **Robust scaling** from 160M to 700M parameter models

## References

Builds on mechanistic interpretability of transformer components and optimizer theory.
