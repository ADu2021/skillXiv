---
name: reasoning-via-sampling
title: "Reasoning with Sampling: Your Base Model is Smarter Than You Think"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.14901"
keywords: [Sampling, MCMC, Inference, Reasoning, Base Models]
description: "Improves base model reasoning through iterative sampling without training or fine-tuning. Uses MCMC-inspired sampling to extract latent reasoning from pretrained models, achieving RL-comparable gains on math, coding, and QA tasks while preserving diversity."
---

# Reasoning via Sampling: Unlocking Latent Model Capabilities

Base language models possess latent reasoning capabilities that remain untapped by standard greedy decoding. Fine-tuning and RL are expensive; sampling offers a simpler alternative that leverages model likelihoods directly.

This technique applies MCMC-inspired iterative sampling to extract superior reasoning from any pretrained LLM at inference time, with no training requirements or curated verification datasets.

## Core Concept

The approach iteratively refines samples by:
- Generating multiple candidate solutions through sampling
- Using the model's own likelihood as a quality signal
- Selecting highest-likelihood sequences that solve the problem
- Repeating to find better solutions within the model's capability envelope

Unlike RL, this preserves sample diversity because it doesn't collapse reward toward mode-seeking behavior.

## Architecture Overview

- Multiple sampling rounds with early stopping when solutions are found
- Likelihood-based ranking of valid solutions (no external verifier needed)
- Resampling around promising regions based on conditional probabilities
- Terminal condition when acceptable solution found or iteration limit reached

## Implementation Steps

Initialize the sampling loop to generate multiple rollouts from the base model. The core insight is that higher model likelihood correlates with quality:

```python
def iterative_sample(model, prompt, num_rounds=10, samples_per_round=8):
    best_solution = None
    best_likelihood = float('-inf')

    for round_idx in range(num_rounds):
        # Generate candidates from current prompt state
        rollouts = []
        for _ in range(samples_per_round):
            sequence, likelihood = model.sample_with_likelihood(prompt)
            rollouts.append((sequence, likelihood))

        # Find best valid solution this round
        for sequence, likelihood in rollouts:
            if is_valid_solution(sequence) and likelihood > best_likelihood:
                best_likelihood = likelihood
                best_solution = sequence

        # Early exit if good solution found
        if best_solution and best_likelihood > threshold:
            break

    return best_solution
```

Implement a refinement step that conditions on partial solutions to guide future sampling. Rather than sampling from scratch each round, continue from promising prefixes:

```python
def refinement_sampling(model, prompt, partial_solution, num_samples=8):
    # Use partial solution as context for next round
    refined_prompt = prompt + partial_solution

    candidates = []
    for _ in range(num_samples):
        sequence, likelihood = model.sample_with_likelihood(refined_prompt)
        # Compute likelihood of full solution relative to partial
        relative_likelihood = compute_conditional_likelihood(
            sequence, partial_solution, model
        )
        candidates.append((sequence, relative_likelihood))

    # Return highest-likelihood continuation
    return max(candidates, key=lambda x: x[1])
```

## Practical Guidance

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Rounds per problem | 8-15 | More rounds improve quality, higher compute cost |
| Samples per round | 4-16 | Balance between diversity and speed |
| Likelihood threshold | Top 10% per-round | Task-dependent; adjust based on success rates |
| Temperature | 0.7-1.0 | Higher = more diversity, may hurt quality |

**When to use:**
- Inference-only applications without retraining budget
- Tasks where model itself is capable but needs coaxing
- Scenarios requiring sample diversity (not single best answer)
- Math, coding, and reasoning tasks

**When NOT to use:**
- Very weak base models that lack capability
- Tasks requiring specialized knowledge outside training data
- High-latency constraints (multiple rounds add inference time)

**Common pitfalls:**
- Using greedy decoding instead of sampling (eliminates exploration)
- Setting likelihood threshold too high (no solutions found)
- Insufficient rounds to find good solutions (premature exit)
- Not validating solutions before selecting (garbage-in garbage-out)

Reference: [Reasoning with Sampling on arXiv](https://arxiv.org/abs/2510.14901)
