---
name: remit-rl-guided-mid-training
title: "ReMiT: RL-Guided Mid-Training for Iterative LLM Evolution"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.06663"
keywords: [Mid-Training, Token Reweighting, RL Feedback, Curriculum Learning, Reasoning]
description: "Improve LLM reasoning by reweighting pre-training data during mid-training based on discrepancies between RL-tuned and base models, boosting reasoning performance without external teachers or extra data."
---

# ReMiT: RL-Guided Mid-Training for Iterative LLM Evolution

## Problem Context

Standard LLM training follows a unidirectional pipeline: pre-training → post-training (RL). This pipeline ignores a key opportunity: insights from RL-tuned models could retroactively improve the pre-trained foundation. Currently, once pre-training ends, high-quality data curated for reasoning is frozen, missing the chance to rebalance based on what the RL model learns matters most.

## Core Concept

ReMiT introduces [RL-guided token reweighting, mid-training phase, soft modulation] to create a self-reinforcing loop. By measuring discrepancies between base and RL model confidence on each token, ReMiT dynamically reweights tokens during mid-training (the final pre-training stage with curated high-quality data). This prioritizes tokens pivotal for reasoning without requiring external teachers.

## Architecture Overview

- **Data signal**: Compute weight updates from RL model tuning
- **Reweighting strategy**: Soft scaling via sigmoid function with clipping bounds
- **Semantic preservation**: Avoid hard token removal; use soft modulation
- **Mid-training timing**: Apply during final pre-training stage before full RL post-training
- **Iterative benefit**: 3% average improvement sustained through post-training

## Implementation

### Step 1: Compute RL model discrepancy signals

Measure where the RL-tuned model and base model disagree on token predictions.

```python
# Compute RL-base model discrepancy
def compute_model_discrepancy(
    base_model, rl_model, data_batch,
    window_size=1  # Context window for attention
):
    """
    Compute token-level discrepancy between base and RL models.
    High discrepancy indicates important tokens for reasoning.
    """
    input_ids = data_batch['input_ids']
    attention_mask = data_batch.get('attention_mask')

    # Forward both models
    with torch.no_grad():
        base_outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        base_logits = base_outputs.logits

        rl_outputs = rl_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        rl_logits = rl_outputs.logits

    # Compute per-token discrepancy
    base_probs = F.softmax(base_logits, dim=-1)
    rl_probs = F.softmax(rl_logits, dim=-1)

    # Discrepancy: L2 distance between probability distributions
    discrepancy = torch.norm(base_probs - rl_probs, p=2, dim=-1)

    # Alternative: Jensen-Shannon divergence
    # m_probs = 0.5 * (base_probs + rl_probs)
    # js_div = 0.5 * (F.kl_div(torch.log(base_probs), m_probs) +
    #                 F.kl_div(torch.log(rl_probs), m_probs))

    return discrepancy
```

### Step 2: Convert discrepancy to soft token weights

Transform discrepancy into soft token reweighting via sigmoid modulation.

```python
# Soft reweighting from discrepancy
def compute_soft_token_weights(
    discrepancy,
    scaling_factor=2.0,
    clip_min=0.5,
    clip_max=2.0
):
    """
    Convert discrepancy to soft token weights.

    Uses sigmoid for smooth modulation: avoids hard removal while
    emphasizing important tokens.

    Args:
        discrepancy: Per-token discrepancy scores, shape (batch_size, seq_len)
        scaling_factor: Sigmoid steepness (higher = sharper transitions)
        clip_min, clip_max: Weight bounds to preserve training stability
    """
    # Normalize discrepancy to [0, 1]
    discrepancy_min = discrepancy.min(dim=-1, keepdim=True)[0]
    discrepancy_max = discrepancy.max(dim=-1, keepdim=True)[0]

    discrepancy_norm = (discrepancy - discrepancy_min) / (
        discrepancy_max - discrepancy_min + 1e-8
    )

    # Sigmoid mapping: high discrepancy → weight > 1.0
    # low discrepancy → weight < 1.0
    raw_weights = torch.sigmoid(scaling_factor * (discrepancy_norm - 0.5))

    # Clip to preserve stability
    token_weights = torch.clamp(raw_weights, clip_min, clip_max)

    return token_weights
```

### Step 3: Apply weights during mid-training

Integrate token weights into the pre-training loss computation.

```python
# Mid-training with token reweighting
class ReMiTPreTrainer:
    def __init__(self, model, base_model, rl_model):
        self.model = model  # Model being trained (usually base_model)
        self.base_model = base_model.eval()  # Frozen reference
        self.rl_model = rl_model.eval()  # Frozen RL reference

    def compute_weighted_loss(
        self, input_ids, labels, attention_mask=None,
        discrepancy_weight=1.0,
        scaling_factor=2.0
    ):
        """
        Compute mid-training loss with RL-guided token reweighting.
        """
        # Standard language modeling loss
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        # Compute per-token loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1)
        )
        token_loss = token_loss.view(shift_labels.shape)

        # Compute RL discrepancy
        discrepancy = compute_model_discrepancy(
            self.base_model, self.rl_model,
            {'input_ids': input_ids, 'attention_mask': attention_mask}
        )

        # Convert to soft weights
        token_weights = compute_soft_token_weights(
            discrepancy, scaling_factor=scaling_factor
        )

        # Apply weights (soft modulation)
        weighted_loss = token_loss * token_weights[..., :-1]

        # Return mean loss
        if attention_mask is not None:
            mask = attention_mask[..., 1:].float()
            weighted_loss = (weighted_loss * mask).sum() / mask.sum()
        else:
            weighted_loss = weighted_loss.mean()

        return weighted_loss
```

### Step 4: Implement full mid-training loop

Integrate ReMiT into the standard training pipeline.

```python
# Full mid-training with ReMiT
def train_remit_mid_training(
    model, base_model, rl_model, train_loader,
    optimizer, num_epochs=1,
    discrepancy_weight=1.0,
    device='cuda'
):
    """
    Mid-training phase with RL-guided reweighting.
    Run this after initial pre-training, before full post-training RL.
    """
    trainer = ReMiTPreTrainer(model, base_model, rl_model)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            input_ids = batch['input_ids']
            labels = batch.get('labels', input_ids.clone())
            attention_mask = batch.get('attention_mask')

            # Compute weighted loss with ReMiT
            loss = trainer.compute_weighted_loss(
                input_ids, labels, attention_mask,
                discrepancy_weight=discrepancy_weight,
                scaling_factor=2.0
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}: Loss={avg_loss:.4f}")

        epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}")

    return model
```

### Step 5: Evaluate improvements through post-training

Measure performance gains persisting through subsequent RL training.

```python
# Evaluate ReMiT effectiveness
def evaluate_remit_benefit(
    base_model, remit_model, rl_model, test_benchmarks,
    benchmark_fn=None, device='cuda'
):
    """
    Compare reasoning performance across:
    - Base model (pre-training only)
    - ReMiT model (pre-training + ReMiT mid-training)
    - Both after RL post-training

    Verify that ReMiT benefits persist through RL.
    """
    results = {}

    for benchmark_name, test_data in test_benchmarks.items():
        print(f"\nBenchmark: {benchmark_name}")

        base_scores = []
        remit_scores = []

        for prompt, reference in test_data:
            # Base model performance
            with torch.no_grad():
                base_output = base_model.generate(
                    prompt, max_tokens=200, device=device
                )
                base_score = benchmark_fn(base_output, reference)
                base_scores.append(base_score)

            # ReMiT model performance
            with torch.no_grad():
                remit_output = remit_model.generate(
                    prompt, max_tokens=200, device=device
                )
                remit_score = benchmark_fn(remit_output, reference)
                remit_scores.append(remit_score)

        base_avg = sum(base_scores) / len(base_scores)
        remit_avg = sum(remit_scores) / len(remit_scores)
        improvement = ((remit_avg - base_avg) / base_avg) * 100

        results[benchmark_name] = {
            'base': base_avg,
            'remit': remit_avg,
            'improvement_pct': improvement
        }

        print(f"  Base: {base_avg:.2%}, ReMiT: {remit_avg:.2%}, "
              f"Improvement: {improvement:+.2f}%")

    return results
```

## Practical Guidance

**When to use**: Reasoning-intensive tasks (math, code, logic) where hidden representations from RL training are informative. Apply during mid-training (final pre-training stage with high-quality curated data).

**Hyperparameters**:
- **scaling_factor**: 1.0-3.0 (controls sigmoid steepness)
  - 1.0: gentle reweighting
  - 2.0: moderate (recommended)
  - 3.0+: aggressive emphasis on high-discrepancy tokens
- **clip_min, clip_max**: (0.5, 2.0) typical range
  - Prevents extreme weight imbalance
- **discrepancy_weight**: 1.0 (standard); 0.5-1.5 range

**Key empirical findings**:
- Average 3% improvement across 10 reasoning benchmarks
- Benefits persist through post-training (2%+ sustained improvement)
- No external teacher required; reuses in-pipeline RL model
- Scales to models 1B-3B; behavior unknown for larger models

**Common pitfalls**:
- Running ReMiT too late in training → limited data to reweight
- Using only base model discrepancy → misses RL insights
- Hard token removal instead of soft weights → removes signal
- Not tuning scaling_factor → suboptimal discrepancy mapping
- Applying to all tokens equally → ignores positional importance

**Throughput impact**: ~43% reduction in training throughput per epoch, but 3.5x faster convergence overall → net speedup.

**Scaling considerations**: Tested on 1B-3B models. Scaling to 7B+ models unclear; recommend testing on target size.

## Reference

Paper: https://arxiv.org/abs/2602.06663
Code: Available at author's repository
Related work: Curriculum learning, token reweighting, iterative training
Benchmarks: GSM8K, MATH, code generation, reasoning tasks
