---
name: surgical-post-training-error-correction
title: "Surgical Post-Training: Cutting Errors, Keeping Knowledge"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.01683"
keywords: [Post-Training, Error Correction, Data Rectification, Binary Classification, Reasoning Alignment]
description: "Correct reasoning errors with minimal data collection by using an oracle to surgically fix only erroneous steps in existing trajectories. Use binary classification loss on rectified pairs with implicit KL regularization to prevent knowledge forgetting."
---

# Surgical Post-Training: Surgical Error Correction with Knowledge Preservation

Standard post-training data collection requires extensive high-quality annotation. Surgical Post-Training (SPoT) takes a different approach: starting with existing model-generated trajectories, use a stronger oracle model to surgically correct only the erroneous reasoning steps while preserving the original trajectory structure and style. This minimal-edit approach dramatically reduces annotation burden while maintaining knowledge from pretraining.

The core insight is that error correction should be **surgical**, not wholesale. Most of a trajectory is correct; fixing only the broken parts minimizes distribution shift. Combined with binary classification objectives and implicit KL regularization through reward-based loss, SPoT achieves consistent improvements with 4,000-10,000 rectified pairs instead of the 100,000+ examples typical approaches require.

## Core Concept

SPoT operates through three coordinated mechanisms:

1. **Data Rectification Pipeline**: Generate erroneous responses from the model, use an oracle to fix only the steps that fail, keep everything else unchanged
2. **Trajectory Similarity Enforcement**: Filter pairs to ensure they differ only minimally (>60% token overlap), preventing major distribution shifts
3. **Binary Classification Objective**: Treat correct vs. incorrect reasoning as a classification problem, then regularize via a KL constraint acting as an elastic tether to prevent catastrophic forgetting

## Architecture Overview

- **Input**: Erroneous trajectories from base model, oracle model for correction
- **Rectification**: Oracle identifies step-level errors, proposes minimal fixes
- **Similarity Filtering**: Keep only pairs with high token-overlap ratio (LCS-based)
- **Loss Design**: Binary classification loss with KL-based regularization
- **Output**: Fine-tuned model with improved reasoning accuracy

## Implementation Steps

**Step 1: Generate candidate erroneous and correct pairs**

Sample trajectories from the model, identify failures, and use oracle for correction.

```python
def generate_rectified_pairs(base_model, oracle_model, prompt_batch,
                            num_samples_per_prompt=2):
    """Generate erroneous model outputs and oracle corrections."""
    rectified_pairs = []

    for prompt in prompt_batch:
        # Sample multiple model outputs (some will likely be incorrect)
        candidates = [base_model.generate(prompt) for _ in range(num_samples_per_prompt)]

        for candidate in candidates:
            # Check if candidate is correct via oracle evaluation
            is_correct, error_steps = oracle_model.evaluate(candidate)

            if not is_correct and len(error_steps) > 0:
                # Oracle corrects only the identified error steps
                corrected = oracle_model.fix_steps(candidate, error_steps)

                rectified_pairs.append({
                    'incorrect': candidate,
                    'correct': corrected,
                    'error_steps': error_steps
                })

    return rectified_pairs
```

**Step 2: Filter pairs by trajectory similarity**

Ensure rectified pairs differ only minimally to avoid distribution shift. Use Longest Common Subsequence (LCS) to measure overlap.

```python
def lcs_ratio(s1, s2):
    """Compute longest common subsequence length / max(len(s1), len(s2))."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    return lcs_length / max(m, n)

# Filter pairs: keep only those with >60% token-level overlap
min_similarity_ratio = 0.60
filtered_pairs = []

for pair in rectified_pairs:
    tokens_incorrect = pair['incorrect'].split()
    tokens_correct = pair['correct'].split()

    similarity = lcs_ratio(tokens_incorrect, tokens_correct)

    if similarity >= min_similarity_ratio:
        filtered_pairs.append(pair)
    # else: too much divergence; skip to avoid distribution shift

print(f"Kept {len(filtered_pairs)} pairs after similarity filtering")
```

**Step 3: Tokenize and prepare for training**

Convert trajectories to token sequences for model training.

```python
def prepare_training_batch(filtered_pairs, tokenizer, max_length=512):
    """Tokenize rectified pairs for training."""
    incorrect_ids = []
    correct_ids = []
    attention_masks = []

    for pair in filtered_pairs:
        # Tokenize both incorrect and correct versions
        incorrect_tokens = tokenizer.encode(
            pair['incorrect'],
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        correct_tokens = tokenizer.encode(
            pair['correct'],
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        incorrect_ids.append(incorrect_tokens['input_ids'])
        correct_ids.append(correct_tokens['input_ids'])
        attention_masks.append(correct_tokens['attention_mask'])

    return {
        'incorrect_ids': torch.cat(incorrect_ids),
        'correct_ids': torch.cat(correct_ids),
        'attention_mask': torch.cat(attention_masks)
    }
```

**Step 4: Design binary classification loss with KL regularization**

The core loss treats correct vs. incorrect as classification, regularized by KL to reference model.

```python
def compute_spt_loss(model, reference_model, batch, temperature=1.0):
    """
    SPoT loss = Binary Classification + KL Regularization

    Classification: maximize likelihood of correct responses, minimize incorrect
    KL: prevent divergence from reference model via elastic tether
    """
    incorrect_ids = batch['incorrect_ids']
    correct_ids = batch['correct_ids']
    attention_mask = batch['attention_mask']

    # Forward pass: get log probabilities for both sequences
    with torch.no_grad():
        ref_logits_correct = reference_model(correct_ids)['logits']
        ref_logits_incorrect = reference_model(incorrect_ids)['logits']

    logits_correct = model(correct_ids, attention_mask=attention_mask)['logits']
    logits_incorrect = model(incorrect_ids, attention_mask=attention_mask)['logits']

    # Binary classification objective:
    # - Maximize likelihood of correct trajectory
    # - Minimize likelihood of incorrect trajectory
    loss_correct = -torch.nn.functional.log_softmax(logits_correct, dim=-1).mean()
    loss_incorrect = torch.nn.functional.log_softmax(logits_incorrect, dim=-1).mean()

    classification_loss = loss_correct - loss_incorrect  # Rank correct above incorrect

    # KL regularization: constrain divergence from reference model
    # This acts as an "elastic tether" preventing catastrophic forgetting
    kl_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(logits_correct / temperature, dim=-1),
        torch.nn.functional.softmax(ref_logits_correct / temperature, dim=-1),
        reduction='batchmean'
    )

    # Combined loss with KL weight
    beta = 0.1  # KL regularization strength
    total_loss = classification_loss + beta * kl_loss

    return total_loss
```

**Step 5: Training loop with convergence monitoring**

Train the model with rectified pairs, monitoring both accuracy improvement and knowledge retention.

```python
def train_spt(model, reference_model, train_dataloader,
              epochs=3, learning_rate=2e-5):
    """Fine-tune model with Surgical Post-Training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            optimizer.zero_grad()

            # Compute loss
            loss = compute_spt_loss(model, reference_model, batch)

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

        # Early stopping: if loss plateaus, reduce learning rate
        if epoch > 0 and abs(avg_loss - prev_loss) < 1e-4:
            learning_rate *= 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        prev_loss = avg_loss

    return model
```

**Step 6: Evaluation and knowledge retention check**

Verify that fine-tuned model improves on error correction while retaining general knowledge.

```python
def evaluate_spt(model, test_prompts, oracle_model, baseline_model):
    """Evaluate SPoT: accuracy improvement and knowledge retention."""
    correct_before = 0
    correct_after = 0
    total = len(test_prompts)

    for prompt in test_prompts:
        # Baseline model output
        response_baseline = baseline_model.generate(prompt)
        is_correct_before, _ = oracle_model.evaluate(response_baseline)

        # Fine-tuned model output
        response_ft = model.generate(prompt)
        is_correct_after, _ = oracle_model.evaluate(response_ft)

        if is_correct_before:
            correct_before += 1
        if is_correct_after:
            correct_after += 1

    improvement = (correct_after - correct_before) / total * 100
    accuracy_after = correct_after / total * 100

    print(f"Accuracy improvement: {improvement:+.1f}%")
    print(f"Final accuracy: {accuracy_after:.1f}%")

    return improvement, accuracy_after
```

## Practical Guidance

**Hyperparameter Selection:**
- **Similarity ratio threshold**: 0.6 (60% LCS overlap). Lower = more change tolerance; higher = stricter conservation.
- **KL weight β**: 0.05-0.2. Higher = stronger forgetting prevention; too high can prevent improvement.
- **Temperature for KL**: 1.0 standard; increase to 2-3 for softer regularization on divergent tasks.
- **Training epochs**: 2-5. More epochs risk overfitting to rectified data; SPoT gains saturate quickly.
- **Learning rate**: 2e-5 (conservative). Lower rates preserve pretraining better.

**When to Use:**
- Post-training with limited annotation budget (has oracle but few labeled examples)
- Correcting specific reasoning errors in code generation, math, planning
- Fine-tuning after pretraining when preserving general knowledge is critical

**When NOT to Use:**
- Complete retraining (use standard SFT instead)
- Tasks where oracle corrections require major trajectory changes
- Settings with abundant high-quality labeled data (standard approaches are more direct)

**Common Pitfalls:**
- **Broken rectification pipeline**: Oracle corrections must be faithful and minimal. Poor oracle → noise in training.
- **Over-filtering by similarity**: Too strict LCS threshold discards useful corrections. Start at 0.6, lower gradually.
- **KL regularization too strong**: β > 0.3 can prevent improvement. Monitor validation accuracy; increase β if it drops.
- **Insufficient rectified data**: <1000 pairs generally insufficient for stable training. Collect 4,000-10,000 for robust gains.

## Reference

arXiv: https://arxiv.org/abs/2603.01683
