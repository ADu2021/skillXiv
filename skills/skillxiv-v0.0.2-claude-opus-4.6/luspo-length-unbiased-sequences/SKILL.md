---
name: luspo-length-unbiased-sequences
title: "Length-Unbiased Sequence Policy Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.05261"
keywords: [Policy Optimization, Sequence Rewards, Length Bias, GRPO Stability, Training]
description: "Fix length bias in sequence-level policy optimization by scaling each sequence's loss by its token count, eliminating gradient imbalances that cause models to shorten responses during training."
---

# Length-Unbiased Sequence Policy Optimization

## Problem Context

Current GRPO and GSPO implementations exhibit systematic length bias where shorter correct sequences receive disproportionately large gradient updates while longer incorrect sequences receive proportionally less penalty. This destabilizes training, causing models to gradually shorten responses to optimize loss rather than improve reasoning quality. The problem is particularly acute in sequence-level optimization where clipping mechanisms amplify length-dependent bias.

## Core Concept

LUSPO introduces a trivial but critical fix: [length-weighted loss, sequence normalization, unbiased gradients] that scales each sequence's loss contribution by its token count. This eliminates the structural bias without adding hyperparameters or computational cost, enabling stable training across response lengths.

## Architecture Overview

- **Problem diagnosis**: Analysis of loss structure in GRPO/GSPO showing length-dependent gradient imbalance
- **Solution**: Per-sequence loss scaling L_seq = (length / mean_length) * L_raw
- **Integration**: Drop-in modification to existing optimizers (GRPO, GSPO, DAPO)
- **Validation**: Comprehensive experiments showing improvements across dense, MoE, and multimodal models
- **Result**: Stable training without length bias, consistent improvements (up to 6.9% on AIME24)

## Implementation

### Step 1: Diagnose length bias in your training setup

Analyze gradient contributions to understand the bias structure before applying the fix.

```python
# Analyze length bias
def analyze_length_bias(rewards, sequence_lengths, log_probs):
    """
    Diagnose length bias: compare gradient contributions by length.
    """
    # Normalize rewards to standard scale
    normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Compute raw losses
    raw_losses = -log_probs * normalized_rewards

    # Group by length percentile
    length_percentiles = torch.quantile(
        sequence_lengths.float(),
        torch.tensor([0.25, 0.5, 0.75, 1.0])
    )

    groups = {
        'short': sequence_lengths <= length_percentiles[0],
        'medium': (sequence_lengths > length_percentiles[0]) &
                  (sequence_lengths <= length_percentiles[2]),
        'long': sequence_lengths > length_percentiles[2]
    }

    print("Length bias analysis:")
    for group_name, mask in groups.items():
        if mask.sum() > 0:
            avg_loss = raw_losses[mask].mean().item()
            avg_length = sequence_lengths[mask].mean().item()
            print(f"  {group_name}: avg_loss={avg_loss:.4f}, avg_length={avg_length:.1f}")

    return {
        'groups': groups,
        'raw_losses': raw_losses,
        'sequence_lengths': sequence_lengths
    }
```

### Step 2: Compute length-weighted normalization

Calculate per-sequence length weights for fair loss scaling.

```python
# Length weighting
def compute_length_weights(sequence_lengths, method='linear'):
    """
    Compute per-sequence weights to normalize by length.

    Methods:
    - 'linear': weight = length / mean_length
    - 'log': weight = log(length) / mean_log_length
    - 'inverse': weight = mean_length / length (inverse normalization)
    """
    if method == 'linear':
        mean_length = sequence_lengths.float().mean()
        weights = sequence_lengths.float() / (mean_length + 1e-8)

    elif method == 'log':
        log_lengths = torch.log(sequence_lengths.float() + 1.0)
        mean_log_length = log_lengths.mean()
        weights = log_lengths / (mean_log_length + 1e-8)

    elif method == 'inverse':
        mean_length = sequence_lengths.float().mean()
        weights = (mean_length + 1e-8) / sequence_lengths.float()

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return weights / weights.mean()  # Normalize to preserve overall scale
```

### Step 3: Implement LUSPO in GRPO

Integrate length weighting into the standard GRPO loss computation. The modification is straightforward: multiply loss by length weight.

```python
# LUSPO: GRPO with length weighting
class LUSPO:
    def __init__(self, model, optimizer, group_size=8, gamma=1.0):
        self.model = model
        self.optimizer = optimizer
        self.group_size = group_size
        self.gamma = gamma  # Clipping parameter

    def compute_loss(self, batch):
        """
        Compute LUSPO loss: length-weighted GRPO.
        """
        prompts = batch['prompts']
        responses = batch['responses']
        rewards = batch['rewards']
        log_probs = batch['log_probs']
        sequence_lengths = batch['sequence_lengths']

        # Compute length weights
        length_weights = compute_length_weights(
            sequence_lengths, method='linear'
        )

        # Standard GRPO advantages
        batch_size = len(rewards)
        num_groups = batch_size // self.group_size

        advantages = []
        for group_idx in range(num_groups):
            group_start = group_idx * self.group_size
            group_end = (group_idx + 1) * self.group_size

            group_rewards = rewards[group_start:group_end]
            group_lengths = sequence_lengths[group_start:group_end]

            # Per-token advantage (GRPO)
            mean_reward = group_rewards.mean()

            for i in range(self.group_size):
                # Standard GRPO advantage
                adv = group_rewards[i] - mean_reward
                # LUSPO: scale by length weight
                weighted_adv = adv * length_weights[group_start + i]
                advantages.append(weighted_adv)

        advantages = torch.stack(advantages)

        # GRPO clipping objective with length weighting
        log_prob_ratio = log_probs - log_probs.detach()
        clipped_ratio = torch.clamp(
            torch.exp(log_prob_ratio),
            1 - self.gamma,
            1 + self.gamma
        )

        # Core difference: scale loss by length weight
        unweighted_loss = -torch.min(
            log_prob_ratio * advantages,
            clipped_ratio * advantages
        )

        # Length-weighted loss
        loss = (unweighted_loss * length_weights).mean()

        return loss, advantages
```

### Step 4: Apply in standard training loop

Integrate LUSPO into existing training, replacing standard GRPO loss computation.

```python
# Training with LUSPO
def train_luspo(
    model, train_loader, verifier, optimizer,
    num_epochs=3, group_size=8, device='cuda'
):
    """
    Training loop using LUSPO for length-unbiased optimization.
    """
    luspo = LUSPO(model, optimizer, group_size=group_size)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass: generate responses
            prompts = batch['prompts']
            responses = []
            log_probs_list = []
            sequence_lengths = []

            for prompt in prompts:
                response, log_prob = model.generate_with_logprobs(
                    prompt, max_tokens=200
                )
                responses.append(response)
                log_probs_list.append(log_prob)
                sequence_lengths.append(len(response.split()))

            # Compute rewards
            rewards = torch.tensor([
                verifier(r) for r in responses
            ], device=device)

            # Prepare batch for LUSPO
            batch['responses'] = responses
            batch['rewards'] = rewards
            batch['log_probs'] = torch.stack(log_probs_list)
            batch['sequence_lengths'] = torch.tensor(
                sequence_lengths, device=device
            )

            # Compute LUSPO loss
            loss, advantages = luspo.compute_loss(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                avg_length = batch['sequence_lengths'].float().mean().item()
                print(f"  Batch {batch_idx + 1}: loss={avg_loss:.4f}, "
                      f"avg_length={avg_length:.1f}")

        print(f"Epoch {epoch + 1}: Loss={total_loss / num_batches:.4f}")
```

### Step 5: Evaluate across response lengths

Verify that LUSPO eliminates length bias by measuring performance across response length ranges.

```python
# Evaluate length-balanced performance
def evaluate_length_performance(
    model, verifier, test_prompts, num_samples=100
):
    """
    Evaluate performance separately for short and long responses.
    Verify LUSPO prevents length-biased optimization.
    """
    results_by_length = {
        'short': [],
        'medium': [],
        'long': []
    }

    all_lengths = []
    all_rewards = []

    for prompt in test_prompts:
        for _ in range(num_samples // len(test_prompts)):
            response = model.generate(prompt, max_tokens=200)
            reward = verifier(response)
            length = len(response.split())

            all_lengths.append(length)
            all_rewards.append(reward)

    all_lengths = torch.tensor(all_lengths)
    all_rewards = torch.tensor(all_rewards)

    # Split by length percentile
    median_length = all_lengths.median().item()
    q25 = all_lengths.quantile(0.25).item()
    q75 = all_lengths.quantile(0.75).item()

    results_by_length['short'] = all_rewards[all_lengths <= q25].float().mean().item()
    results_by_length['medium'] = all_rewards[
        (all_lengths > q25) & (all_lengths <= q75)
    ].float().mean().item()
    results_by_length['long'] = all_rewards[all_lengths > q75].float().mean().item()

    return results_by_length, {
        'short_length': q25,
        'medium_length': (q25 + q75) / 2,
        'long_length': q75
    }
```

## Practical Guidance

**When to use**: Any sequence-level optimization (GRPO, GSPO, DAPO) where response length varies. Especially critical for reasoning tasks where longer reasoning chains may be beneficial.

**Hyperparameters**:
- **Length weighting method**: 'linear' (default) works well; 'log' for heavily skewed length distributions
- **No new hyperparameters**: LUSPO uses same hyperparameters as GRPO

**Key empirical findings**:
- Improvements: 2-6.9% across AIME24, MATH500, GSM8K
- No quality regression: models maintain or improve pass@1
- Works across: dense models, MoE, multimodal (image + text)

**Common pitfalls**:
- Forgetting to normalize length weights (divide by mean) → scales loss magnitude
- Using raw length instead of normalized → may overcorrect
- Applying to token-level optimization where length is already handled → redundant

**Scaling**: Zero computational overhead; simple multiplication operation. Scales perfectly to large models and datasets.

## Reference

Paper: https://arxiv.org/abs/2602.05261
Code: Available at author's repository (VeRL integration)
Related work: GRPO, GSPO, sequence-level rewards
Benchmarks: AIME24, MATH500, GSM8K, code generation tasks
