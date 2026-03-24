---
name: ttt-e2e-long-context
title: "End-to-End Test-Time Training for Long Context"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.23675
keywords: [long-context, test-time-training, meta-learning, efficiency]
description: "Enable long-context modeling via test-time training with meta-learning. Inner loop continues training on context, compressing information into weights rather than KV cache, outer loop optimizes initialization—maintaining full-attention quality with RNN-like constant inference latency across 8K-128K token contexts."
---

## Overview

Reformulates long-context as continual learning problem solved at test time.

## Core Technique

**Meta-Learning for Test Time:**

```python
# Inner loop: compress context into weights
for token in context:
    gradient = compute_gradient_on_token(model, token)
    model.weights += gradient  # Compress context

# Outer loop: optimize initialization
# Treat inner loop as differentiable step
```

## Performance

- Full-attention quality across context lengths
- 2.7× faster than attention at 128K
- Constant-time inference

## References

- Test-time meta-learning
- Weight compression of context
- End-to-end optimization
