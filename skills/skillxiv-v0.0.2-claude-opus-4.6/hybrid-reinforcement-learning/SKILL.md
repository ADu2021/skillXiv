---
name: hybrid-reinforcement-learning
title: "Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dense"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.07242"
keywords: [Reward Modeling, Sparse Rewards, Dense Rewards, Reinforcement Learning, Hybrid Approach]
description: "Combine sparse verifier rewards with dense reward model scores using stratified normalization to overcome limitations of either approach alone."
---

# Technique: Stratified Hybrid Reward Combinations

Reinforcement learning faces a fundamental trade-off: sparse rewards (verifiers with high precision but low coverage) lack training signal, while dense rewards (reward models with richer feedback) suffer from potential misalignment. Hybrid Reinforcement Learning (HERO) resolves this by strategically combining both signals.

The key insight is that sparse verifiers define reliable correctness boundaries, while dense reward models distinguish quality within those boundaries. By grouping outputs by verifier status and normalizing reward model scores within each group, the system gains the stability of verification while leveraging the granularity of reward models.

## Core Concept

HERO operates through two mechanisms:

1. **Stratified Normalization**: Partition outputs by verifier result (correct/incorrect), then normalize dense scores within each group. This preserves verifier-guaranteed boundaries while refining quality judgments.

2. **Variance-Aware Weighting**: Allocate training emphasis toward ambiguous examples where dense signals matter most—hard-to-verify tasks where reward model discrimination is valuable.

## Architecture Overview

- **Verifier**: Binary (correct/incorrect) or categorizes by confidence
- **Reward Model**: Produces dense scores for all outputs
- **Stratification**: Partition outputs by verifier result
- **Normalization**: Normalize dense scores within strata
- **Weighting**: Emphasize examples where disagreement is high
- **Training**: Update policy using hybrid signals

## Implementation Steps

Implement the verifier (binary or rule-based).

```python
def verify_output(output, ground_truth, verification_type='exact_match'):
    """
    Apply verification rule to output.

    Args:
        output: Generated output
        ground_truth: Ground truth answer
        verification_type: 'exact_match', 'contains', 'program_execution'

    Returns:
        verdict: Boolean correctness
    """

    if verification_type == 'exact_match':
        # Exact string matching (e.g., math answers)
        return output.strip() == ground_truth.strip()

    elif verification_type == 'contains':
        # Substring matching (e.g., fact-based QA)
        return ground_truth.lower() in output.lower()

    elif verification_type == 'program_execution':
        # Code execution verification
        try:
            exec_result = execute_code(output)
            return exec_result == ground_truth
        except:
            return False

    else:
        raise ValueError(f"Unknown verification type: {verification_type}")


def batch_verify(outputs, ground_truths, verification_type='exact_match'):
    """
    Verify a batch of outputs.

    Args:
        outputs: List of generated outputs
        ground_truths: List of ground truths
        verification_type: Verification method

    Returns:
        verdicts: Boolean array of correctness
    """

    verdicts = []
    for output, truth in zip(outputs, ground_truths):
        verdict = verify_output(output, truth, verification_type)
        verdicts.append(verdict)

    return verdicts
```

Implement stratified normalization combining both reward signals.

```python
def stratified_normalization(outputs, verdicts, reward_scores,
                            stratify_by='verifier'):
    """
    Normalize reward scores within verifier strata.

    Args:
        outputs: List of generated outputs
        verdicts: Boolean array from verifier
        reward_scores: Float array of reward model scores
        stratify_by: 'verifier' for binary, or verifier categories

    Returns:
        normalized_scores: Stratified-normalized reward scores
    """

    import numpy as np

    verdicts = np.array(verdicts)
    reward_scores = np.array(reward_scores)

    normalized_scores = np.zeros_like(reward_scores)

    # Separate correct and incorrect groups
    correct_mask = verdicts == True
    incorrect_mask = verdicts == False

    # Normalize within correct group
    if correct_mask.sum() > 0:
        correct_rewards = reward_scores[correct_mask]
        min_correct = correct_rewards.min()
        max_correct = correct_rewards.max()

        if max_correct > min_correct:
            normalized_scores[correct_mask] = \
                (correct_rewards - min_correct) / (max_correct - min_correct) * 0.9 + 0.1

    # Normalize within incorrect group
    if incorrect_mask.sum() > 0:
        incorrect_rewards = reward_scores[incorrect_mask]
        min_incorrect = incorrect_rewards.min()
        max_incorrect = incorrect_rewards.max()

        if max_incorrect > min_incorrect:
            normalized_scores[incorrect_mask] = \
                (incorrect_rewards - min_incorrect) / (max_incorrect - min_incorrect) * 0.4

    return normalized_scores
```

Implement variance-aware weighting.

```python
def compute_variance_aware_weights(reward_scores, verdicts,
                                   variance_threshold=0.1):
    """
    Weight examples based on reward model-verifier disagreement.

    Args:
        reward_scores: Float array of reward model scores
        verdicts: Boolean verifier results
        variance_threshold: Disagreement threshold for emphasis

    Returns:
        weights: Per-example importance weights
    """

    import numpy as np

    reward_scores = np.array(reward_scores)
    verdicts = np.array(verdicts)

    # Compute disagreement: high-scoring incorrect outputs or low-scoring correct
    correct_mask = verdicts == True
    incorrect_mask = verdicts == False

    disagreement = np.zeros_like(reward_scores)

    # Incorrect outputs with high rewards: disagreement
    if incorrect_mask.sum() > 0:
        incorrect_rewards = reward_scores[incorrect_mask]
        max_incorrect = incorrect_rewards.max()
        disagreement[incorrect_mask] = incorrect_rewards / max_incorrect

    # Correct outputs with low rewards: disagreement
    if correct_mask.sum() > 0:
        correct_rewards = reward_scores[correct_mask]
        max_correct = correct_rewards.max()
        disagreement[correct_mask] = (max_correct - correct_rewards) / max_correct

    # Convert disagreement to weights: emphasize ambiguous examples
    weights = 1.0 + disagreement * 2.0  # Range [1.0, 3.0]

    return weights
```

Implement training with hybrid rewards.

```python
def train_with_hybrid_rewards(policy_model, optimizer, batch_outputs,
                             batch_truths, reward_model, verifier_type='exact_match',
                             num_epochs=3):
    """
    Train policy using hybrid sparse-dense rewards.

    Args:
        policy_model: Language model policy
        optimizer: PyTorch optimizer
        batch_outputs: Generated outputs
        batch_truths: Ground truths
        reward_model: Trained reward model
        verifier_type: Type of verifier to use
        num_epochs: Training epochs

    Returns:
        losses: Training loss curve
    """

    import torch
    import torch.nn.functional as F

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Verify outputs
        verdicts = batch_verify(batch_outputs, batch_truths, verifier_type)

        # Get reward model scores
        reward_scores = reward_model.score_batch(batch_outputs)

        # Apply stratified normalization
        hybrid_rewards = stratified_normalization(
            batch_outputs, verdicts, reward_scores
        )

        # Compute variance-aware weights
        weights = compute_variance_aware_weights(reward_scores, verdicts)

        # Training step
        for output, hybrid_reward, weight in \
                zip(batch_outputs, hybrid_rewards, weights):

            # Get log probability
            log_prob = policy_model.get_log_prob(output)

            # Weighted loss: maximize log_prob * reward * weight
            loss = -(log_prob * hybrid_reward * weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(batch_outputs))

    return losses
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Verifier precision | High precision preferred | Rule-based verifiers less prone to drift |
| Reward model quality | Train on diverse examples | Poor RM quality degrades hybrid signals |
| Strata balance | Ideally 1:1 correct/incorrect | Imbalanced strata reduce normalization effectiveness |
| Weight range | 1.0-3.0 | Controls emphasis on ambiguous examples |
| When to use | Hard-to-verify tasks | Where neither sparse nor dense alone suffices |
| When NOT to use | Easily verifiable tasks | Pure verifier sufficient if high coverage |
| Common pitfall | Reward model overfitting | Use validation set to track RM calibration |

### When to Use HERO

- Tasks with verifiable correctness but sparse success (e.g., reasoning where final answer is checkable but path is unclear)
- Scenarios combining symbolic verification with learned value judgments
- Settings where reward model has known limitations
- Training where both quality refinement and correctness guarantees matter

### When NOT to Use HERO

- Tasks where verifier has perfect coverage (pure sparse reward sufficient)
- Domains where verifiers are unreliable (pure dense RM better)
- Real-time training where stratification overhead is problematic

### Common Pitfalls

- **Imbalanced strata**: If one class dominates, normalization becomes unstable; oversample minority
- **Reward model drift**: RM quality degrades over time; periodically retrain on validation data
- **Verifier brittleness**: Rule-based verifiers may be overly strict; consider multi-level verdicts
- **Weight explosion**: Extreme disagreement cases can create unstable weights; clip to reasonable range

## Reference

Paper: https://arxiv.org/abs/2510.07242
