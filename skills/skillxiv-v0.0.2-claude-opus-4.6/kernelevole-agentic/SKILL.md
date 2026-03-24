---
name: kernelevole-agentic
title: "KernelEvolve: Scaling Agentic Kernel Coding for AI Accelerators"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.23236
keywords: [kernel-optimization, agents, automated-synthesis, hardware-aware]
description: "Automate compute kernel optimization via agentic AI and retrieval-augmented prompting. Unified context-aware transformation function, hardware-specific constraint KB, self-improving state machine exploring kernel variants—achieving 1.25-17× speedups on production recommendation workloads in hours vs weeks of manual effort."
---

## Overview

Autonomous kernel optimization across heterogeneous hardware via agentic synthesis.

## Core Technique

**Context-Aware Transformation:**

```python
# Single universal operator adapts via context
kernel = universal_operator.transform(
    operator_spec,
    hardware_context=gpu_type,
    optimization_history=prior_attempts
)
```

**Retrieval-Augmented Prompting:**
Hardware-specific constraints guide generation.

## Performance

- 1.25-17× speedup on production workloads
- Hours vs weeks of manual optimization

## References

- Context-aware kernel generation
- Hardware constraint knowledge bases
- Self-improving kernel optimization
