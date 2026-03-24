---
name: cosinegate-dynamic-routing
title: "CosineGate: Semantic Dynamic Routing via Cosine Incompatibility"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.22206
keywords: [dynamic-routing, efficient-inference, residual-networks, gating]
description: "Achieve efficient neural networks via self-supervised dynamic routing using Cosine Incompatibility Ratio (CIR). Ground gating decisions in geometric novelty rather than learned heuristics, enable per-sample/per-block binary routing via Gumbel-softmax, constrain with progressive FLOPs regularization—maintaining accuracy while reducing computation 28.5% on CIFAR-10."
---

## Overview

Dynamic routing based on semantic novelty between identity and residual paths.

## Core Technique

**Cosine Incompatibility Ratio:**

```python
# CIR = 1 - cos(x, F(x))
# Low CIR: identity sufficient (skip block)
# High CIR: block adds novelty (execute)

cir_score = 1.0 - cosine_similarity(input_x, residual_output)
should_skip = cir_score < threshold
```

**Differentiable Gating:**
Gumbel-softmax for training, deterministic for inference.

## Performance

- 91.3% accuracy with 28.5% FLOPs saving
- No auxiliary supervision needed

## References

- Cosine incompatibility ratio for routing
- Geometric grounding of gate decisions
- FLOPs-constrained gating
