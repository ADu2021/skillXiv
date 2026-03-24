---
name: dsdr-dual-scale-diversity-regularization
title: "DSDR: Dual-Scale Diversity Regularization for LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.19895"
keywords: [reinforcement learning, reasoning diversity, exploration, LLM training, regularization]
description: "Improve LLM reasoning by promoting diversity at both trajectory and token levels simultaneously. Global (trajectory) scale rewards distinct correct solutions; local (token) scale applies entropy regularization per decision point. Dual-scale approach couples these via diversity-weighted allocation: solutions that are globally more distinctive receive stronger local regularization, focusing exploration where it matters most among underexplored correct modes."
---

# DSDR: Multi-Scale Diversity Promotion for Stable Reasoning Exploration

Language models trained with reinforcement learning on reasoning tasks often collapse onto narrow solution modes, missing diverse valid reasoning paths. Standard approaches either reward all correct solutions equally (missing that some are more valuable) or apply uniform entropy regularization (missing that not all decision points need exploration). The challenge is discovering which correct modes are underexplored and focusing exploration effort there.

Two key insights: (1) not all correct modes are equally valuable for downstream learning, (2) exploration effort should be concentrated at decision points that distinguish underexplored modes. Existing methods treat diversity at single scales, missing this multi-scale structure.

## Core Concept

DSDR operates at two complementary scales:

**Global (Trajectory) Scale**: Promote diversity among correct reasoning paths by assigning higher rewards to distinct solutions. Use semantic diversity (embedding differences) and formula diversity (unique mathematical expressions) to measure distinctness.

**Local (Token) Scale**: Apply length-invariant entropy regularization within correct trajectories to prevent overconfidence at individual decision points.

**Coupling Mechanism**: Link scales by letting global distinctiveness guide local regularization strength. Solutions that are globally more distinctive receive stronger local entropy penalties, directing exploration toward decision points that matter for underexplored modes.

## Architecture Overview

- **Semantic Diversity Scorer**: Embed solutions and compute pairwise cosine distances in semantic space
- **Formula Diversity Tracker**: Count unique mathematical expressions across correct solutions
- **Global Reward Augmenter**: Increase rewards for semantically/formulaically distinct solutions
- **Local Entropy Computer**: Compute per-token entropy within each trajectory
- **Coupling Weight**: Compute per-solution diversity weight (based on global distinctiveness) and scale local regularization accordingly
- **Normalized Loss**: Avoid length bias by averaging per-token, then scaling by diversity weight

## Implementation

Compute semantic and formula diversity among correct solutions:

```python
def compute_solution_diversity(correct_solutions):
    """
    Measure diversity of correct solutions using semantic and formula metrics.
    correct_solutions: list of (response_text, correctness_flag)
    Returns: diversity_scores dict
    """
    from sklearn.metrics.pairwise import cosine_distances
    import re

    # Extract text from solutions
    texts = [sol[0] for sol in correct_solutions if sol[1]]

    # Semantic diversity: embed and compute pairwise distances
    embeddings = embed_texts(texts)  # Shape: (N, 768)
    semantic_distances = cosine_distances(embeddings)
    avg_semantic_diversity = semantic_distances.mean()

    # Formula diversity: count unique mathematical expressions
    formulas = []
    for text in texts:
        # Extract equations/expressions (simple regex)
        found_formulas = re.findall(r'[a-zA-Z0-9_]+\s*[+\-*/]\s*[a-zA-Z0-9_]+', text)
        formulas.extend(found_formulas)

    unique_formulas = len(set(formulas))
    total_formulas = len(formulas)
    formula_diversity = unique_formulas / max(total_formulas, 1)

    return {
        'semantic': avg_semantic_diversity,
        'formula': formula_diversity,
        'combined': (avg_semantic_diversity + formula_diversity) / 2.0
    }
```

Compute global and local diversity metrics during training:

```python
def compute_trajectory_diversity_weight(solution, other_correct_solutions, alpha=0.1):
    """
    Compute per-solution diversity weight for coupling.
    Solutions that are globally more distinctive get higher weight.
    """
    # Embed this solution
    solution_embedding = embed_text(solution)

    # Compute distances to all other correct solutions
    distances = []
    for other in other_correct_solutions:
        if other == solution:
            continue
        other_embedding = embed_text(other)
        dist = cosine_distance(solution_embedding, other_embedding)
        distances.append(dist)

    if not distances:
        return 1.0  # Only solution; neutral weight

    # Average distance (higher = more distinctive)
    avg_distance = np.mean(distances)

    # Scale to [0.5, 2.0] range for weighting
    diversity_weight = 0.5 + 1.5 * min(avg_distance / 0.5, 1.0)

    return diversity_weight

def dsdr_loss(
    logits, targets, correctness_mask, correct_trajectories,
    alpha_global=0.5, alpha_local=0.1
):
    """
    Compute DSDR loss combining global and local diversity.
    logits: (B, T, V) model predictions
    targets: (B, T) ground truth tokens
    correctness_mask: (B,) binary correctness
    correct_trajectories: list of correct solution texts
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Baseline policy gradient loss
    pg_loss = -selected_logprobs

    # Compute per-trajectory diversity weight
    diversity_weights = []
    for b in range(logits.shape[0]):
        if correctness_mask[b] == 1:  # Only for correct solutions
            weight = compute_trajectory_diversity_weight(
                correct_trajectories[b], correct_trajectories
            )
        else:
            weight = 1.0  # Neutral for incorrect

        diversity_weights.append(weight)

    diversity_weights = torch.tensor(diversity_weights, device=logits.device)

    # Local entropy regularization (token-level)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (B, T)
    entropy_reg = -entropy  # Higher entropy = more exploration

    # Couple local regularization to global diversity weight
    # Globally diverse solutions get stronger entropy penalty
    coupled_entropy_reg = entropy_reg * diversity_weights.unsqueeze(1)

    # Normalize by sequence length to avoid length bias
    seq_lengths = (targets != 0).sum(dim=1).float().unsqueeze(1)
    normalized_entropy = coupled_entropy_reg.sum(dim=1) / seq_lengths.sum(dim=1)

    # Combine global and local objectives
    total_loss = pg_loss.sum(dim=1) + alpha_local * normalized_entropy

    # Scale by diversity weight (encourage exploration on distinctive solutions)
    total_loss = total_loss * diversity_weights

    return total_loss.mean()
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Alpha global (diversity bonus) | 0.5 | Range 0.2–1.0; higher rewards distinctness more aggressively |
| Alpha local (entropy weight) | 0.1 | Range 0.05–0.3; higher penalizes overconfidence more strongly |
| Coupling strength | 1.0 | Scales how much global diversity influences local regularization |
| Diversity score window | 10 solutions | Compare against recent batch; use sliding window to stabilize |

**When to use**: For reasoning tasks (math, code, QA) where multiple correct approaches exist and you want to explore diverse solution strategies.

**When not to use**: For tasks with unique solutions or where all correct approaches are equally valuable; overhead of diversity tracking is wasted.

**Common pitfalls**:
- Computing diversity only on small batches; use larger windows (10–20 solutions) for stable similarity estimates
- Forgetting to normalize entropy regularization by sequence length; longer sequences shouldn't receive systematically different learning signals
- Setting coupling strength too high, causing excessive exploration; calibrate on validation set to find stability threshold

## Reference

DSDR achieves improved reasoning accuracy and diversity by combining global trajectory-level rewards with coupled local token-level regularization. The dual-scale approach focuses exploration on decision points that distinguish underexplored correct modes, improving both accuracy and solution diversity on mathematical reasoning benchmarks.
