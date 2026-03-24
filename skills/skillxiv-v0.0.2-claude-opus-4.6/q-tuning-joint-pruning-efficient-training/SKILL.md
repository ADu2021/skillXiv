---
name: q-tuning-joint-pruning-efficient-training
title: "Winning the Pruning Gamble: A Unified Approach to Joint Sample and Token Pruning for Efficient Supervised Fine-Tuning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.23873"
keywords: [data-pruning, token-pruning, training-efficiency, data-selection, fine-tuning]
description: "Dramatically reduce training data requirements (to 12.5% of original) while improving model performance using joint sample and token pruning guided by Error-Uncertainty plane diagnostics. Asymmetric pruning preserves calibration signals while removing redundant tokens from misconception examples."
---

# Q-Tuning: Joint Sample and Token Pruning via Error-Uncertainty Analysis

Training LLMs on massive datasets is expensive, but much of that data is redundant. The challenge is identifying which examples and which tokens matter. Prior work handles either sample-level pruning (which examples to keep?) or token-level pruning (which tokens to truncate?), but not both jointly. This mismatch means either discarding valuable signals or wasting compute on irrelevant tokens.

Q-Tuning introduces the Error-Uncertainty (EU) plane: a diagnostic tool that simultaneously characterizes data quality at sample and token levels. Using this, you can apply **asymmetric pruning**—aggressive token removal for misconception examples (where tokens don't matter much) while preserving calibration samples entirely (where every token matters).

## Core Concept

The Error-Uncertainty plane is a 2D diagnostic that plots each training sample by two metrics:

1. **Error**: Does the model get this example right or wrong?
2. **Uncertainty**: How confident is the model on this example?

This creates four quadrants:

- **High Error, High Uncertainty**: Misconceptions (model confidently wrong). Prune aggressively.
- **High Error, Low Uncertainty**: Calibration signals (model knows it's uncertain). Keep entirely.
- **Low Error, High Uncertainty**: Confusing examples (not representative). Prune moderately.
- **Low Error, Low Uncertainty**: Easy examples (redundant). Prune moderately.

For each quadrant, apply different token-pruning ratios: keep informative misconceptions as calibration signals, but prune their less-important tokens.

## Architecture Overview

- **Error detector**: Binary classification (correct/incorrect) on each example
- **Uncertainty quantifier**: Compute confidence scores (probability, variance, entropy)
- **EU plane mapper**: Plot samples, categorize by quadrant
- **Token scorer**: Rank token importance within each sample
- **Asymmetric pruner**: Apply quadrant-specific pruning ratios
- **Training loop**: Standard fine-tuning with pruned data

## Implementation Steps

First, compute error and uncertainty scores for all training data:

```python
import torch
import numpy as np
from collections import defaultdict

class ErrorUncertaintyAnalyzer:
    """
    Analyze training data via Error-Uncertainty plane.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute_eu_scores(self, dataset, batch_size=32):
        """
        Score all examples on Error-Uncertainty plane.

        Args:
            dataset: Training dataset (input, target pairs)
            batch_size: Batch size for inference

        Returns:
            scores: List of {"idx": idx, "error": 0/1, "uncertainty": 0-1}
        """
        scores = []

        for batch_idx in range(0, len(dataset), batch_size):
            batch = dataset[batch_idx : batch_idx + batch_size]
            inputs = self.tokenizer(
                [ex["input"] for ex in batch],
                padding=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get logits for each example
            logits = outputs.logits

            for i, example in enumerate(batch):
                # Compute correctness
                predicted_token_id = logits[i, -1, :].argmax()
                target_token = self.tokenizer.encode(example["target"])[0]
                is_correct = predicted_token_id == target_token

                # Compute uncertainty as entropy of output distribution
                probs = torch.softmax(logits[i, -1, :], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

                # Normalize entropy to [0, 1] range
                max_entropy = np.log(logits.shape[-1])
                normalized_entropy = entropy / max_entropy

                scores.append({
                    "idx": batch_idx + i,
                    "error": 0 if is_correct else 1,  # Error: 1 if wrong
                    "uncertainty": normalized_entropy
                })

        return scores
```

Next, categorize examples into EU quadrants and apply asymmetric pruning policies:

```python
def categorize_and_prune(scores, dataset, tokenizer):
    """
    Map to EU plane, determine pruning strategy per quadrant.

    Args:
        scores: Error-uncertainty scores
        dataset: Original training data
        tokenizer: Tokenizer for token-level operations

    Returns:
        pruned_dataset: Filtered and truncated examples
    """
    # Define quadrant boundaries (using medians as simple thresholds)
    errors = np.array([s["error"] for s in scores])
    uncertainties = np.array([s["uncertainty"] for s in scores])

    error_threshold = np.median(errors)
    uncertainty_threshold = np.median(uncertainties)

    # Categorize examples
    quadrants = defaultdict(list)
    for score in scores:
        error_hi = score["error"] > error_threshold
        unc_hi = score["uncertainty"] > uncertainty_threshold

        if error_hi and unc_hi:
            quadrant = "misconception"  # High error, high uncertainty
        elif error_hi and not unc_hi:
            quadrant = "calibration"  # High error, low uncertainty
        elif not error_hi and unc_hi:
            quadrant = "confusing"  # Low error, high uncertainty
        else:
            quadrant = "easy"  # Low error, low uncertainty

        quadrants[quadrant].append(score["idx"])

    # Define asymmetric pruning ratios
    pruning_ratios = {
        "misconception": 0.5,  # Keep 50% of tokens
        "calibration": 1.0,   # Keep 100% of tokens (valuable)
        "confusing": 0.7,     # Keep 70% of tokens
        "easy": 0.3           # Keep 30% of tokens (redundant)
    }

    # Apply pruning
    pruned_dataset = []
    for example_idx, example in enumerate(dataset):
        # Find quadrant for this example
        score_entry = next(s for s in scores if s["idx"] == example_idx)
        error_hi = score_entry["error"] > error_threshold
        unc_hi = score_entry["uncertainty"] > uncertainty_threshold

        if error_hi and unc_hi:
            quadrant = "misconception"
        elif error_hi and not unc_hi:
            quadrant = "calibration"
        elif not error_hi and unc_hi:
            quadrant = "confusing"
        else:
            quadrant = "easy"

        # Get pruning ratio for this quadrant
        keep_ratio = pruning_ratios[quadrant]

        # Apply token-level pruning to input
        tokens = tokenizer.encode(example["input"])
        num_keep = max(1, int(len(tokens) * keep_ratio))
        pruned_tokens = tokens[:num_keep]

        # Keep full target (labels aren't pruned)
        pruned_input = tokenizer.decode(pruned_tokens)

        pruned_dataset.append({
            "input": pruned_input,
            "target": example["target"],
            "quadrant": quadrant
        })

    return pruned_dataset
```

Now integrate EU-guided pruning into your fine-tuning loop:

```python
def finetune_with_eu_pruning(model, dataset, num_epochs=3):
    """
    Fine-tune model using EU-guided pruned data.

    Args:
        model: Model to fine-tune
        dataset: Full training data
        num_epochs: Number of training epochs

    Returns:
        model: Fine-tuned model
    """
    analyzer = ErrorUncertaintyAnalyzer(model, tokenizer)

    # Step 1: Compute EU scores on full dataset
    print("Computing Error-Uncertainty scores...")
    scores = analyzer.compute_eu_scores(dataset)

    # Step 2: Apply joint pruning based on EU quadrants
    print("Applying joint sample and token pruning...")
    pruned_dataset = categorize_and_prune(scores, dataset, tokenizer)

    print(f"Reduced dataset from {len(dataset)} to {len(pruned_dataset)} examples")
    print(f"Reduction ratio: {len(pruned_dataset) / len(dataset) * 100:.1f}%")

    # Step 3: Fine-tune on pruned data
    print("Fine-tuning on pruned dataset...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in create_batches(pruned_dataset, batch_size=32):
            inputs = tokenizer(
                [ex["input"] for ex in batch],
                padding=True,
                return_tensors="pt"
            )
            targets = tokenizer(
                [ex["target"] for ex in batch],
                padding=True,
                return_tensors="pt"
            )

            outputs = model(**inputs)
            loss = compute_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {total_loss / len(pruned_dataset):.4f}")

    return model
```

## Practical Guidance

**When to use Q-Tuning:**
- Training on redundant or noisy datasets (common in web-scale data)
- Reducing compute costs for large model fine-tuning
- Improving model calibration (removing easy examples reduces overconfidence)
- Identifying which examples matter most for your task

**When NOT to use:**
- Small curated datasets (pruning risks losing signal)
- Tasks where every example is carefully selected
- Settings where model uncertainty is unreliable (frozen early layers)
- Fine-tuning with very few examples (<1000)

**Empirical results (SmolLM2-1.7B baseline):**

| Data Retention | Improvement Over Full Data | Time Saved |
|---|---|---|
| 100% | Baseline | 0% |
| 50% | -2% (slight regression) | 50% |
| 25% | +15% to +20% | 75% |
| 12.5% | +25% to +38% (peak) | 87.5% |

**Key insight**: Extreme pruning (87.5% reduction) *improves* performance because you're removing easy, redundant examples and keeping hard, calibration-critical examples.

**Hyperparameter tuning:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| error_threshold | median | Higher = fewer "hard" examples in misconception quadrant |
| uncertainty_threshold | median | Higher = requires higher entropy to be "uncertain" |
| misconception_ratio | 0.5 | Higher = preserve more misconception signals |
| calibration_ratio | 1.0 | Always 1.0 (don't prune calibration) |
| easy_ratio | 0.3 | Lower = more aggressive redundancy removal |

**Common pitfalls:**
- **Unreliable uncertainty scores**: If your model's entropy doesn't correlate with confidence, EU categorization fails. Validate on a small held-out set first.
- **Over-aggressive easy-example pruning**: Easy examples aren't *useless*; they provide coverage. If easy_ratio=0.1 causes divergence, increase to 0.3-0.5.
- **Ignoring calibration examples**: These are rare and precious. Always keep 100% of low-error, low-uncertainty examples.
- **Single pass limitation**: EU scores change as the model trains. For iterative fine-tuning, recompute EU scores every 2-3 epochs.

**Integration checklist:**
- [ ] Compute EU scores on full dataset and validate distribution (should have balanced quadrants)
- [ ] Visualize EU plane scatter plot to inspect quadrant composition
- [ ] Validate pruning ratios don't eliminate entire quadrants
- [ ] Fine-tune on pruned data and compare loss curves to full-data baseline
- [ ] Evaluate on validation set to confirm pruning improves or maintains performance
- [ ] Monitor calibration metrics (confidence vs. accuracy) separately

Reference: https://arxiv.org/abs/2509.23873
