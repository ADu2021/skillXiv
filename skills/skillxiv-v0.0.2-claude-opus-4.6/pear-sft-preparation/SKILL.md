---
name: pear-sft-preparation
title: "Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.01058"
keywords: [SFT, Reinforcement Learning, Importance Sampling, Loss Reweighting, AIME]
description: "Improve post-RL performance by reweighting SFT loss using importance sampling. Prioritize training examples that match the target policy distribution, not the behavior policy. Achieves 14.6% Pass@8 gains on AIME."
---

# PEAR: SFT Preparation for Reinforcement Learning

## Problem
Standard supervised fine-tuning (SFT) optimizes for behavior policy distribution—the policy that generated training data. But downstream RL operates under the target policy, creating distributional mismatch.

Models trained with uniform SFT may not prepare well for RL. The mismatch between offline SFT optimization and online RL sampling creates sub-optimal initialization.

## Core Concept
PEAR (Policy Evaluation-inspired) reweights SFT loss using importance sampling ratios between target and behavior policies. Rather than training uniformly, concentrate on tokens whose continuations remain plausible under the target policy.

This ensures offline updates focus on trajectories the RL stage will actually revisit, improving final post-RL performance.

## Architecture Overview

- **Importance Sampling Weights**: Compute target/behavior policy likelihood ratios
- **Token-Level Reweighting**: Apply importance weights to individual token losses
- **Suffix Ratio Computation**: Use future continuations to estimate target likelihood
- **Block-Level Stabilization**: Partition sequences for gradient stability
- **Sequence-Level Granularity**: Uniform trajectory-wide weights for simplicity

## Implementation

### Step 1: Compute Importance Sampling Weights
Calculate policy likelihood ratios for reweighting.

```python
def compute_importance_weights(batch, behavior_model, target_model):
    """Compute importance sampling weights from behavior to target policy."""
    weights = []

    for sequence in batch:
        tokens = tokenize(sequence)
        sequence_weight = 1.0

        for token_idx in range(len(tokens)):
            # Compute probability under behavior policy
            behavior_logits = behavior_model.get_logits(tokens[:token_idx])
            behavior_probs = F.softmax(behavior_logits, dim=-1)
            behavior_prob = behavior_probs[tokens[token_idx]].item()

            # Compute probability under target policy (being optimized)
            target_logits = target_model.get_logits(tokens[:token_idx])
            target_probs = F.softmax(target_logits, dim=-1)
            target_prob = target_probs[tokens[token_idx]].item()

            # Importance ratio
            ratio = target_prob / (behavior_prob + 1e-6)
            sequence_weight *= ratio

        weights.append(sequence_weight)

    return torch.tensor(weights)
```

### Step 2: Token-Level Reweighting
Apply importance weights to individual token losses.

```python
def token_level_reweighting(model, batch, behavior_model, target_model, learning_rate=1e-4):
    """Reweight SFT loss using token-level importance sampling."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for sequence in batch:
        tokens = tokenize(sequence)

        # Compute suffix ratios for token-level weights
        suffix_weights = []
        cumulative_ratio = 1.0

        for token_idx in range(len(tokens) - 1, -1, -1):
            # Suffix ratio: probability of tokens[token_idx:] under target vs behavior
            suffix_target = compute_suffix_prob(target_model, tokens, token_idx)
            suffix_behavior = compute_suffix_prob(behavior_model, tokens, token_idx)

            suffix_ratio = suffix_target / (suffix_behavior + 1e-6)
            suffix_weights.insert(0, suffix_ratio)
            cumulative_ratio *= suffix_ratio

        # Apply weights to loss
        for token_idx in range(len(tokens)):
            target_token = tokens[token_idx]
            context_tokens = tokens[:token_idx]

            # Forward pass
            logits = model.get_logits(context_tokens)
            loss = F.cross_entropy(logits, target_token, reduction='none')

            # Weight loss by importance ratio
            weighted_loss = loss * suffix_weights[token_idx]
            weighted_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
```

### Step 3: Block-Level Stabilization
Partition sequences for more stable gradient updates.

```python
def block_level_reweighting(model, batch, importance_weights, block_size=64, learning_rate=1e-4):
    """Apply block-level importance weighting for gradient stability."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for sequence, weight in zip(batch, importance_weights):
        tokens = tokenize(sequence)

        # Partition into blocks
        for block_start in range(0, len(tokens), block_size):
            block_end = min(block_start + block_size, len(tokens))
            block_tokens = tokens[block_start:block_end]

            # Compute loss for block
            block_loss = 0
            for token_idx, token in enumerate(block_tokens):
                context = tokens[:block_start + token_idx]
                logits = model.get_logits(context)
                loss = F.cross_entropy(logits, token)
                block_loss += loss

            # Apply weight to entire block
            weighted_block_loss = (block_loss / len(block_tokens)) * weight

            weighted_block_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return model
```

### Step 4: Training Loop with PEAR
Integrated SFT training with importance reweighting.

```python
def train_sft_with_pear(model, training_data, behavior_model, num_epochs=3, learning_rate=1e-4):
    """Train model with importance-sampled loss for RL preparation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Batch processing
        for batch in create_batches(training_data, batch_size=32):
            # Compute importance weights
            weights = compute_importance_weights(batch, behavior_model, model)

            # Token-level reweighting
            for sequence, weight in zip(batch, weights):
                tokens = tokenize(sequence)

                for token_idx in range(len(tokens)):
                    context = tokens[:token_idx]
                    target = tokens[token_idx]

                    logits = model.get_logits(context)
                    loss = F.cross_entropy(logits, target)

                    # Weight by importance ratio
                    weighted_loss = loss * weight.item()
                    weighted_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    return model
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Importance weight clipping | 0.01-100 | Prevent extreme weights |
| Block size (if block-level) | 32-128 | Balance granularity and stability |
| Suffix ratio smoothing | 0.1e-6 | Numerical stability |
| Learning rate | 1e-4 to 5e-5 | Typical SFT rates |
| Training epochs | 2-4 | Standard convergence |

### When to Use

- Preparing models for post-training with RL
- Improving AIME/competition math performance via subsequent GRPO
- Scenarios where behavior data distribution differs from target policy
- Maximizing final performance after expensive RL tuning
- Any SFT→RL pipeline where offline initialization matters

### When Not to Use

- When RL is not planned (standard SFT sufficient)
- Behavior and target policies are very similar
- Extreme importance weights due to distribution shift
- Online learning scenarios (PEAR is offline-only)
- Data from uniform random behavior (weights meaningless)

### Common Pitfalls

1. **Extreme importance weights**: Distribution mismatch causes unbounded weights. Clip or use variance reduction.
2. **Weight instability**: Early epochs with poor target model estimation. Start with behavior model as target.
3. **Data quality sensitivity**: Importance sampling amplifies noise in low-probability examples. Filter outliers.
4. **Insufficient samples**: Token-level ratios need stable estimates. Ensure large behavior dataset.

## Reference
Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning
https://arxiv.org/abs/2602.01058
