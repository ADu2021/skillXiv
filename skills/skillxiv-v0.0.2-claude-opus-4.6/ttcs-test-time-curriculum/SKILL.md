---
name: ttcs-test-time-curriculum
title: "TTCS: Test-Time Curriculum Synthesis for Self-Evolving"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.22628"
keywords: [Test-Time Training, Curriculum Learning, Self-Evolution, GRPO, Synthetic Data]
description: "Improve model performance at test time by dynamically synthesizing curriculum of problem variants. Co-evolving synthesizer and solver agents create reinforcing feedback for continuous improvement without external labels."
---

# TTCS: Test-Time Curriculum Synthesis

## Problem
Test-time training struggles because raw test questions are often too difficult to yield high-quality pseudo-labels. Direct self-training on hard problems produces weak supervision signals. A curriculum approach that adaptively generates easier intermediate problems enables gradual difficulty increase.

Standard test-time scaling uses independent parallel attempts but misses cross-trajectory learning opportunities.

## Core Concept
TTCS implements two co-evolving agents: a Synthesizer generates problem variants at the solver's capability frontier, and a Solver self-improves using mixed training data. Self-consistency rewards enable curriculum adaptation without external labels.

The Synthesizer targets intermediate difficulty by maximizing 0.5 confidence. The Solver trains on original plus synthesized problems, improving systematically.

## Architecture Overview

- **Synthesizer Agent**: Generates test-guided variants preserving reasoning structure
- **Capability-Adaptive Reward**: Scores variants at solver's frontier using self-consistency
- **Diversity Penalty**: Discourages trivial copying between samples
- **Solver Agent**: Self-consistency training on mixed original + synthesized data
- **Online Filtering**: Retains intermediate difficulty samples during training
- **GRPO Training**: Stable policy optimization under label-free constraints

## Implementation

### Step 1: Initialize Capability-Adaptive Reward
Define reward targeting intermediate difficulty at solver's frontier.

```python
def synthesizer_reward(synthesized_variant, solver, target_confidence=0.5):
    """Compute reward for synthesized problems at capability frontier."""
    responses = [solver.sample_response(synthesized_variant) for _ in range(8)]
    agreement = sum(1 for r in responses if is_correct(synthesized_variant, r)) / len(responses)
    capability_score = 4 * agreement * (1 - agreement)
    return capability_score
```

### Step 2: Synthesizer Generates Variants
Create problem variants preserving core reasoning with varying difficulty.

```python
def synthesize_problem_variants(original_problem, model, num_variants=5):
    """Generate diverse problem variants conditioning on original."""
    variants = []
    for i in range(num_variants):
        prompt = f"Create variant preserving core reasoning: {original_problem}"
        temperature = 0.5 + i * 0.15
        variant = model.generate(prompt, temperature=temperature)
        variants.append(variant)
    return variants
```

### Step 3: Filter by Diversity
Remove near-duplicate variants using semantic similarity threshold.

```python
def filter_by_diversity(variants, original_problem, threshold=0.75):
    """Keep only semantically diverse variants."""
    filtered = []
    for variant in variants:
        is_diverse = True
        if compute_similarity(variant, original_problem) > threshold:
            is_diverse = False
        for existing in filtered:
            if compute_similarity(variant, existing) > threshold:
                is_diverse = False
                break
        if is_diverse:
            filtered.append(variant)
    return filtered
```

### Step 4: Solver Self-Evolution Training
Train solver on mixed curriculum using GRPO optimization.

```python
def solver_self_evolution(model, problems, epochs=2):
    """Self-evolve solver on original + synthesized problems."""
    for epoch in range(epochs):
        batch = np.random.choice(problems, size=32)
        for problem in batch:
            responses = [model.sample_response(problem) for _ in range(8)]
            correct_count = sum(1 for r in responses if is_correct(problem, r))
            reward = correct_count / len(responses)
            loss = -reward * compute_log_prob(model, responses)
            loss.backward()
        optimizer.step()
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target confidence | 0.5 | Intermediate difficulty zone |
| Variants per problem | 3-8 | Balance quality and diversity |
| Diversity threshold | 0.75 | Semantic similarity cutoff |
| Self-consistency samples | 8 | Voting for correctness |
| Training epochs | 2-4 | Convergence iterations |

### When to Use

- Mathematical/reasoning benchmarks with clear correctness
- Test problems harder than training distribution
- Optimizing Pass@k metrics
- Multiple valid solution approaches exist

### When Not to Use

- Easy test sets (ceiling effect)
- Unique, exact solution requirements
- Synthesis that diverges from original domain
- Tasks where synthetic variants degrade validity

### Common Pitfalls

1. Weak reward targeting—explicitly measure current difficulty via self-consistency
2. Insufficient diversity—semantic similarity reduces variant effectiveness
3. Distribution shift—validate synthesized variants match original distribution
4. Solver overfitting to synthetic patterns rather than generalizing

## Reference
TTCS: Test-Time Curriculum Synthesis for Self-Evolving
https://arxiv.org/abs/2601.22628
