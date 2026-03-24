---
name: scale-selective-test-time
title: "SCALE: Selective Resource Allocation for Mathematical Test-Time Scaling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.00466
keywords: [test-time-scaling, resource-allocation, mathematical-reasoning, system-1-2, adaptive-compute]
description: "Decomposes math problems into sequential sub-problems, assesses difficulty, and allocates simple ones to fast System 1 reasoning while directing complex ones to deliberate System 2. Save 33-53% tokens while improving accuracy by up to 13.75 points on AIME."
---

## Summary

SCALE addresses inefficient resource allocation in test-time scaling by implementing selective resource allocation based on sub-problem difficulty. Rather than uniformly distributing computational resources across all steps, the approach decomposes mathematical problems into sequential sub-problems, assesses difficulty, and dynamically assigns simple problems to fast processing while directing complex ones to deliberate reasoning.

## Core Technique

**Problem Decomposition:** Break mathematical problems into smaller steps or sub-problems. For math:
- Step 1: Parse and identify key quantities
- Step 2: Determine solution strategy
- Step 3: Execute calculations
- Step 4: Verify answer

**Difficulty Assessment:** For each sub-problem, estimate complexity via:
- Token count of problem statement
- Number of dependencies on previous steps
- Complexity of mathematical operations involved
- Uncertainty in model confidence

**Dual-System Allocation:**
- **System 1 (Fast):** Single-shot generation with minimal reasoning, optimal for straightforward calculations
- **System 2 (Deliberate):** Multiple sampling, majority voting, extended reasoning for complex steps

## Implementation

**Difficulty scorer:** Train a lightweight classifier:
```python
difficulty = mlp(encode(subproblem))  # 0 (easy) to 1 (hard)
```

**Dynamic allocation:**
```python
if difficulty < 0.3:
    # System 1: single pass
    answer = model.generate(subproblem, max_tokens=50)
else:
    # System 2: deliberate reasoning
    answers = [model.generate(subproblem, max_tokens=500) for _ in range(5)]
    answer = majority_vote(answers)
```

**Token counting:** Track token usage:
```python
total_tokens = sum(difficulty_threshold * tokens_system1 + (1 - difficulty_threshold) * tokens_system2)
```

**Accuracy gains:** Hard sub-problems get more compute, soft problems use minimal resources, overall accuracy improves while token usage decreases.

## When to Use

- Mathematical reasoning tasks with variable problem difficulty
- Applications where token budget is constrained but accuracy is critical
- Scenarios where problem decomposition is natural
- Automated grading systems (AIME, AMC, etc.) with measurable accuracy

## When NOT to Use

- Tasks where uniform reasoning across all steps is necessary
- Scenarios without clear problem decomposition
- Applications where difficulty assessment is unreliable
- Real-time inference where dynamic allocation overhead matters

## Key References

- System 1 and System 2 thinking in cognitive science
- Adaptive computation and dynamic neural networks
- Majority voting and ensemble methods for reasoning
- Difficulty estimation and sampling strategies
