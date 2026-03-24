---
name: mits-pointwise-mutual-information-tree-search
title: "MITS: Enhanced Tree Search Reasoning for LLMs via Pointwise Mutual Information"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.03632"
keywords: [Tree Search, Reasoning, Pointwise Mutual Information, LLM Decoding, Test-Time Scaling]
description: "Score reasoning paths using PMI to identify question-specific relevant steps without rollout simulations, enabling efficient tree search 12× faster than MCTS while improving reasoning accuracy."
---

# Technique: PMI-Based Efficient Tree Search for LLM Reasoning

Large language models generate multiple reasoning trajectories to solve complex problems, but evaluating which paths are most promising typically requires expensive rollout simulations or external verifiers. MITS uses Pointwise Mutual Information (PMI) to score reasoning steps directly: how much does a particular step increase the likelihood of being correct specifically for this question?

Unlike generic step quality scores, PMI captures question-specific relevance. A mathematical notation definition might be high-quality universally but irrelevant to an algebra problem about finances. By computing PMI(question; step), MITS filters out generic patterns and focuses computation on steps that genuinely inform the specific problem.

## Core Concept

MITS operates on three key innovations:

1. **PMI Scoring**: Each reasoning step S receives a score measuring "how much a reasoning path's plausibility increases because of the specific question," computed as log p(S|q) / p(S).

2. **Incremental Computation**: PMI is calculated as reasoning unfolds, not recomputed from scratch. Each new step contributes its conditional probability given prior steps and the question, enabling efficient computation without look-ahead simulations.

3. **Entropy-Based Adaptive Allocation**: High-uncertainty steps receive more candidate exploration; certain steps require fewer attempts. This optimizes overall search budget automatically.

## Architecture Overview

- **Question Input**: Problem statement for which reasoning is needed
- **LLM Generation**: Generate candidate reasoning steps (chains or trees)
- **PMI Computation**: Score each step using conditional probability estimates
- **Adaptive Sampling**: Concentrate search effort on high-entropy decision points
- **Voting**: Combine top-scored paths using weighted frequency and PMI consensus
- **Final Output**: Aggregated answer from best-performing reasoning paths

## Implementation Steps

Compute the base and conditional probabilities needed for PMI calculation. These can be extracted from LLM logits during generation.

```python
def compute_pmi_score(step_text, question, prior_steps, lm):
    """
    Compute PMI(question; step) measuring question-specific relevance.

    Args:
        step_text: The reasoning step to score
        question: The problem statement
        prior_steps: List of previous steps in the chain
        lm: Language model with log-probability access

    Returns:
        pmi_score: log p(step|question, prior_steps) - log p(step|prior_steps)
    """
    # Conditional probability given question and prior context
    context_with_q = f"Question: {question}\n" + \
                     "\n".join(prior_steps) + "\n"

    log_prob_given_question = lm.get_log_prob(
        context_with_q, step_text
    )

    # Conditional probability given only prior context (no question)
    context_without_q = "\n".join(prior_steps) + "\n"
    log_prob_given_context = lm.get_log_prob(
        context_without_q, step_text
    )

    # PMI = log p(S|Q,prior) - log p(S|prior)
    pmi_score = log_prob_given_question - log_prob_given_context

    return pmi_score
```

Implement incremental PMI calculation during tree search without full rollouts.

```python
def search_with_incremental_pmi(question, lm, max_depth=10,
                                 num_candidates_per_step=5):
    """
    Perform tree search where PMI guides step selection incrementally.

    Args:
        question: Problem statement
        lm: Language model
        max_depth: Maximum reasoning chain length
        num_candidates_per_step: Candidates to generate per step

    Returns:
        paths: List of (path, score) tuples sorted by quality
    """
    paths = [{"steps": [], "pmi_sum": 0.0, "log_probs": []}]

    for depth in range(max_depth):
        new_paths = []

        for path in paths:
            # Generate candidate next steps
            candidates = lm.generate(
                question=question,
                prior_steps=path["steps"],
                num_candidates=num_candidates_per_step,
                return_logprobs=True
            )

            for candidate, logprob in candidates:
                # Compute PMI for this step incrementally
                context_without_q = "\n".join(path["steps"]) + "\n"
                logprob_without_q = lm.get_log_prob(
                    context_without_q, candidate
                )

                pmi = logprob - logprob_without_q

                # Create extended path
                new_path = {
                    "steps": path["steps"] + [candidate],
                    "pmi_sum": path["pmi_sum"] + pmi,
                    "log_probs": path["log_probs"] + [logprob],
                    "last_pmi": pmi
                }
                new_paths.append(new_path)

        # Keep top paths to manage search space
        paths = sorted(new_paths,
                      key=lambda p: p["pmi_sum"],
                      reverse=True)[:20]

    return paths
```

Implement entropy-based adaptive allocation to concentrate effort where uncertainty is high.

```python
def adaptive_sampling_allocation(entropy_scores, budget,
                                  min_samples=2, max_samples=20):
    """
    Allocate search budget based on step entropy.

    Args:
        entropy_scores: Entropy for each candidate step
        budget: Total samples to allocate across steps
        min_samples: Minimum samples per step
        max_samples: Maximum samples per step

    Returns:
        samples_per_step: Number of samples to generate for each step
    """
    # Normalize entropy to [0, 1]
    norm_entropy = entropy_scores / entropy_scores.max()

    # High entropy = more samples (quadratic scaling emphasizes differences)
    proportional_samples = norm_entropy ** 2 * budget

    # Clamp to min/max and normalize to actual budget
    clamped = np.clip(proportional_samples, min_samples, max_samples)
    scaled = clamped / clamped.sum() * budget

    return np.round(scaled).astype(int)
```

Aggregate results from multiple reasoning paths using PMI-weighted voting.

```python
def ensemble_predictions(paths, question, lm, num_paths=5):
    """
    Combine top reasoning paths into final prediction.

    Args:
        paths: List of (path, pmi_score) tuples from search
        question: Original question
        lm: Language model for extraction
        num_paths: Number of top paths to ensemble

    Returns:
        final_answer: Aggregated answer across paths
        confidence: Confidence in the answer
    """
    top_paths = paths[:num_paths]

    predictions = []
    weights = []

    for path in top_paths:
        # Extract answer from final step
        answer = extract_answer_from_steps(path["steps"], question, lm)

        # Weight by PMI score normalized by path length
        # (avoid biasing toward longer paths)
        normalized_pmi = path["pmi_sum"] / len(path["steps"])
        weight = np.exp(normalized_pmi)

        predictions.append(answer)
        weights.append(weight)

    # Frequency-weighted voting
    unique_answers = list(set(predictions))
    answer_scores = {}

    for answer in unique_answers:
        # Weight by both frequency and PMI
        freq_weight = sum(w for pred, w in zip(predictions, weights)
                         if pred == answer)
        freq = sum(1 for p in predictions if p == answer) / len(predictions)

        combined_score = freq_weight * freq
        answer_scores[answer] = combined_score

    final_answer = max(answer_scores, key=answer_scores.get)
    confidence = answer_scores[final_answer] / sum(answer_scores.values())

    return final_answer, confidence
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Entropy threshold | Auto-adaptive | Higher entropy = more samples; quadratic scaling emphasizes uncertainty |
| Path count | 5-20 top paths | Balance ensemble diversity with computational cost |
| Max depth | 10-20 steps | Domain-dependent; deeper for complex reasoning, shallower for simple |
| Rollout frequency | Every N steps or entropy spike | Check entropy to trigger deeper exploration adaptively |
| When to use | Open-ended reasoning tasks | Math, logic puzzles, scientific problem-solving |
| When NOT to use | Few-step or deterministic tasks | Overhead of PMI computation outweighs benefits |
| Common pitfall | Computing PMI with biased log probs | Use consistent probability estimates; avoid mixing models |

### When to Use MITS

- Mathematical reasoning or logic puzzles where multiple solution paths are plausible
- Tasks where reasoning steps vary significantly in relevance
- Scenarios requiring test-time scaling without access to external verifiers
- Combining with other tree search methods (MCTS, beam search) for comparison

### When NOT to Use MITS

- Tasks with deterministic single-path solutions
- Real-time inference where latency is critical
- Simple classification or factual lookup
- Scenarios with stable, well-calibrated LLM probabilities

### Common Pitfalls

- **Probability calibration**: LLM logits may be miscalibrated; validate PMI assumptions
- **Question representation**: PMI depends on clear question representation; ambiguous formulations degrade scoring
- **Path diversity loss**: Early high-PMI pruning can eliminate diverse reasoning strategies; maintain more paths initially
- **Inconsistent tokenization**: PMI scores depend on how steps are tokenized; keep tokenization consistent
- **Depth imbalance**: Very deep paths may accumulate high PMI despite low individual step quality; normalize by path length

## Reference

Paper: https://arxiv.org/abs/2510.03632
