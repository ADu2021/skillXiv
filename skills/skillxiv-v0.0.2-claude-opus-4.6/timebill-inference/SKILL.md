---
name: timebill-inference
title: "TimeBill: Time-Budgeted Inference for LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.21859
keywords: [inference, efficiency, time-budget, adaptive, latency]
description: "Guarantee LLM inference completes within time budgets via response length prediction, execution time estimation, and adaptive KV cache eviction. Three-stage pipeline predicts response length, estimates end-to-end time with 1.22% accuracy, adjusts cache eviction ratio—enabling time-critical deployment in robotics and autonomous systems."
---

## Overview

TimeBill solves the problem of unpredictable LLM inference latency in time-critical systems.

## Core Technique

**Three-Stage Pipeline:**

```python
# Stage 1: Response Length Prediction
length_predictor = fine_grained_response_length_model()
predicted_length = length_predictor.predict(input)

# Stage 2: Execution Time Estimation
ete = execution_time_estimator()
worst_case_time = ete.estimate(predicted_length)

# Stage 3: Adaptive KV Cache Eviction
cache_eviction_ratio = calculate_optimal_eviction(worst_case_time, time_budget)
```

## When to Use

Use when: Robotics, autonomous driving, time-critical inference.

## References

- Response length prediction
- FLOPs-based execution time modeling
- Adaptive KV cache management
