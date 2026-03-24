---
name: peft-rlvr-evaluation
title: "Evaluating Parameter Efficient Methods for RLVR"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.23165
keywords: [parameter-efficient, reinforcement-learning, adapter, peft]
description: "Comprehensively evaluate 12+ parameter-efficient fine-tuning methods for RL with Verifiable Rewards (RLVR). Show DoRA/AdaLoRA outperform LoRA, SVD-based methods fail on RL, extreme reduction creates bottlenecks—providing empirical evidence that geometric-aware adapters align better with RL's off-principal update dynamics."
---

## Overview

Reveals LoRA's limitations in RL contexts through systematic PEFT evaluation.

## Core Technique

**Structural Variant Comparison:**

```python
# Standard LoRA sometimes suboptimal
methods = [LoRA, DoRA, AdaLoRA, PiSSA, VeRA, ...]

# Evaluate on RLVR (sparse binary rewards)
results = benchmark_all_methods(methods, rlvr_training)

# Finding: DoRA/AdaLoRA outperform
```

## When to Use

Use when: RL fine-tuning, parameter efficiency critical, method selection.

## References

- Comprehensive PEFT method evaluation
- RL-specific optimization dynamics
- Geometric-aware adapter superiority
