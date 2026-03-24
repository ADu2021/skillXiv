---
name: derf-normalization-free-transformers
title: "Stronger Normalization-Free Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.10938
keywords: [normalization-free, transformers, activation functions, generalization, training stability]
description: "Replace LayerNorm with Derf(x) = erf(αx + s) for improved generalization in transformers. Derf outperforms LayerNorm across vision, speech, and DNA modeling—ideal when normalization-free training provides benefits without architectural complexity."
---

## Overview

The paper demonstrates that simpler point-wise functions can replace traditional normalization while achieving superior generalization. Derf, based on rescaled Gaussian CDF, provides normalization-free transformers with improved performance across diverse domains.

## When to Use

- Training transformers without normalization layers
- Scenarios where generalization is critical
- Vision, speech, and sequence modeling tasks
- Need for improved training stability
- Replacing LayerNorm or RMSNorm

## When NOT to Use

- Applications already achieving good normalization
- Scenarios where norm-dependent features matter
- Models requiring specific normalization properties

## Core Technique

Derf activation function design:

```python
# Normalization-free transformers with Derf
class DerfNormalizationFree(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.s = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Derf(x) = erf(α*x + s)"""
        from scipy.special import erf as scipy_erf

        # Apply Derf activation
        output = torch.erf((self.alpha * x + self.s) / math.sqrt(2.0))

        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        # NO LayerNorm, use Derf instead
        self.derf = DerfNormalizationFree(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            self.derf,
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # Self-attention
        attn_output = self.attn(x, x, x)[0]
        # Apply Derf instead of LayerNorm
        x = x + self.derf(attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)
        # Derf-based combination
        x = x + self.derf(ffn_output)

        return x
```

## Key Results

- Derf outperforms LayerNorm on generalization
- Effective across vision, speech, DNA
- Dynamic Tanh surpassed
- Improved training stability

## References

- Original paper: https://arxiv.org/abs/2512.10938
- Focus: Normalization-free transformer training
- Domain: Transformer architectures, activation functions
