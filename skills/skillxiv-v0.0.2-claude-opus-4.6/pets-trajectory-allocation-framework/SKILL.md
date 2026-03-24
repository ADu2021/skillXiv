---
name: pets-trajectory-allocation-framework
title: "PETS: Principled Framework for Optimal Trajectory Allocation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.16745"
keywords: [self-consistency, trajectory sampling, budget allocation, Bayesian optimization, reasoning]
description: "Optimize sampling budget allocation for self-consistency inference by treating trajectory allocation as a crowdsourcing problem. Introduce self-consistency rate (agreement with infinite-budget consensus) as optimization target. Offline setting uses Optimistic Knowledge Gradient for Bayesian optimization; online setting uses difficulty grid discretization and greedy allocation. Achieves up to 75% budget reduction (offline) and 55% (online) while maintaining accuracy."
---

# PETS: Efficient Sampling via Principled Trajectory Allocation

Self-consistency inference—aggregating outputs from multiple sampled trajectories—improves LLM reasoning accuracy substantially. However, it requires sampling many trajectories per question, creating a trade-off: allocating fixed budgets uniformly wastes resources on easy questions while starving hard ones.

The challenge is discovering which questions need more samples. Easy questions often achieve consensus quickly; hard questions require more samples to reach the same confidence threshold. Allocating more samples to hard questions improves final accuracy without increasing total budget.

## Core Concept

PETS formulates trajectory allocation as a budget optimization problem. The key metric is "self-consistency rate"—the agreement between a finite-budget majority vote and the theoretical consensus from unlimited sampling. This metric measures when a question has been sampled "enough."

PETS provides two algorithms: an offline setting (all questions known upfront) using Bayesian optimization with Knowledge Gradient, and an online setting (sequential question arrivals) using difficulty grid discretization and greedy allocation.

## Architecture Overview

- **Self-Consistency Rate Metric**: Measure agreement between k-sample majority vote and infinite-budget consensus
- **Offline Bayesian Optimization**: Model uncertainty over per-question difficulty and select questions for additional sampling using Optimistic Knowledge Gradient
- **Online Greedy Allocation**: Estimate question difficulty from training distribution; allocate budgets greedily upon arrival
- **Difficulty Estimator**: Predict question difficulty from features (problem length, token count, answer distribution entropy)
- **Budget Solver**: Given difficulty distribution and total budget, compute per-question allocations

## Implementation

Compute self-consistency rate and identify under-sampled questions:

```python
def compute_self_consistency_rate(trajectories_by_question):
    """
    Measure agreement between sampled consensus and infinite-budget consensus.
    trajectories_by_question: dict mapping question_id -> list of (answer, score)
    Returns: dict mapping question_id -> consistency_rate (0.0-1.0)
    """
    consistency_rates = {}

    for qid, trajectories in trajectories_by_question.items():
        answers, scores = zip(*trajectories)

        # Infinite-budget consensus: majority vote on all samples
        from collections import Counter
        answer_counts = Counter(answers)
        infinite_consensus = answer_counts.most_common(1)[0][0]

        # Finite-budget consistency: agreement at various sample counts
        consistency_by_k = {}

        for k in [1, 3, 5, 10, len(trajectories)]:
            if k > len(trajectories):
                k = len(trajectories)

            top_k_answers = answers[:k]
            top_k_majority = Counter(top_k_answers).most_common(1)[0][0]

            is_consistent = (top_k_majority == infinite_consensus)
            consistency_by_k[k] = float(is_consistent)

        # Average consistency rate across sample sizes
        consistency_rates[qid] = sum(consistency_by_k.values()) / len(consistency_by_k)

    return consistency_rates
```

Implement Bayesian optimization for offline budget allocation:

```python
def allocate_budgets_offline(
    questions, initial_trajectories, total_budget, target_consistency=0.95
):
    """
    Offline setting: allocate additional samples to questions that need them.
    Uses Optimistic Knowledge Gradient for Bayesian optimization.
    """
    consistency_rates = compute_self_consistency_rate(initial_trajectories)

    # Questions that haven't reached target consistency
    under_sampled = {
        qid: 1.0 - rate for qid, rate in consistency_rates.items()
        if rate < target_consistency
    }

    if not under_sampled:
        return initial_trajectories  # Done

    # Bayesian optimization: iteratively add samples to highest-value questions
    remaining_budget = total_budget

    while remaining_budget > 0 and under_sampled:
        # Compute Knowledge Gradient value for each under-sampled question
        kg_values = {}

        for qid in under_sampled.keys():
            # KG: expected reduction in regret from sampling this question
            current_inconsistency = under_sampled[qid]
            # Assume sampling reduces inconsistency; KG is proportional to current inconsistency
            kg_values[qid] = current_inconsistency

        # Select question with highest Knowledge Gradient
        best_qid = max(kg_values.items(), key=lambda x: x[1])[0]

        # Sample additional trajectories for best question
        new_trajectory = sample_trajectory(best_qid)
        initial_trajectories[best_qid].append(new_trajectory)

        remaining_budget -= 1

        # Re-compute consistency rate
        consistency_rates = compute_self_consistency_rate(initial_trajectories)
        consistency = consistency_rates.get(best_qid, 0.0)

        if consistency >= target_consistency:
            del under_sampled[best_qid]
        else:
            under_sampled[best_qid] = 1.0 - consistency

    return initial_trajectories
```

Implement online difficulty-based allocation:

```python
def estimate_question_difficulty(question_features):
    """
    Predict question difficulty from features.
    question_features: dict with 'length', 'token_count', 'entropy'
    Returns: difficulty score (0.0-1.0)
    """
    from sklearn.preprocessing import StandardScaler

    # Simple linear model; in practice train on dev set
    features = [
        question_features.get('length', 100) / 500.0,
        question_features.get('token_count', 50) / 200.0,
        question_features.get('entropy', 0.5)  # answer distribution entropy
    ]

    difficulty = (features[0] + features[1] + features[2]) / 3.0
    difficulty = min(1.0, max(0.0, difficulty))

    return difficulty

def allocate_budgets_online(
    incoming_questions, difficulty_estimator, total_budget
):
    """
    Online setting: allocate budgets upon question arrival.
    Uses difficulty grid and greedy solver.
    """
    # Discretize difficulty into bins
    difficulty_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
    bin_allocations = {
        0.0: 1,  # Easy: 1 sample
        0.25: 2,
        0.5: 3,  # Medium: 3 samples
        0.75: 5,
        1.0: 8  # Hard: 8 samples
    }

    allocations = {}
    total_allocated = 0

    for question in incoming_questions:
        # Estimate difficulty
        difficulty = estimate_question_difficulty(question)

        # Find closest bin
        closest_bin = min(difficulty_bins, key=lambda x: abs(x - difficulty))

        # Allocate budget for this difficulty
        budget = bin_allocations[closest_bin]

        # Greedy: if total budget exceeded, reduce
        if total_allocated + budget > total_budget:
            budget = max(1, total_budget - total_allocated)

        allocations[question['id']] = budget
        total_allocated += budget

        if total_allocated >= total_budget:
            break

    return allocations
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Target consistency | 0.95 | Range 0.85–0.99; higher requires more samples |
| Initial budget (offline) | 10% of total | Allocate enough for coarse difficulty estimates |
| Difficulty bins (online) | 5 bins | 3–7 bins balance granularity with simplicity |
| Sample per easy question | 1–3 | Easy questions often consensus with 1–2 samples |
| Sample per hard question | 5–10 | Hard questions often need 8+ samples |

**When to use**: For inference-time optimization where computational budget is constrained and you can afford to spend time allocating resources optimally.

**When not to use**: For real-time systems with tight latency budgets; allocation overhead may exceed benefit. Use when questions are batched.

**Common pitfalls**:
- Under-estimating difficulty; use dev set to calibrate difficulty predictor
- Ignoring variance in sample quality; some trajectories are more informative than others (consider weighting by score)
- Allocating too aggressively to hard questions; balance budget between ensuring all questions are answered vs. maximal consistency on hard ones

## Reference

PETS reduces sampling budget by 75% (offline) and 55% (online) on GPQA while maintaining perfect self-consistency. The approach enables efficient reasoning inference without auxiliary reward models, using only majority-vote structure to guide allocation.
