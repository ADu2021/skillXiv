---
name: weak-driven-learning-logit-mixing
title: "Weak-Driven Learning: How Weak Agents Make Strong Agents Stronger"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.08222"
keywords: [Knowledge Distillation, Logit Mixing, Fine-Tuning, Hard Negatives, Training Saturation]
description: "Break through supervised fine-tuning saturation by mixing logits from weaker model checkpoints into strong model training targets. Amplifies hard negatives that strong models have already suppressed, enabling continued learning after standard training plateaus."
---

# Weak-Driven Learning: Leveraging Weak Models to Improve Strong Models

Standard supervised fine-tuning for language models hits saturation: target logits plateau while non-target logits stop declining. The strong model has already learned simple patterns and no longer benefits from standard training signals. Weak-driven learning inverts traditional knowledge distillation—instead of learning from a superior teacher, the strong model learns from its own earlier, weaker checkpoints.

Weak models assign non-negligible probability to plausible-but-incorrect alternatives that strong models have already suppressed. By mixing weak model logits into training targets, you reintroduce uncertainty and force the strong model to maintain harder distinctions even after standard training stagnates.

## Core Concept

Standard fine-tuning: min ||logits(x) - logits_correct||^2. This saturates because target logits converge to large values, non-target gradients vanish.

Weak-driven learning: mix weak and strong logits as target:
```
logits_mix(x) = λ·logits_strong(x) + (1-λ)·logits_weak(x)
```

Train strong model to match this mixture. The weak model's high probability on incorrect alternatives creates "hard negatives" the strong model must actively suppress.

## Architecture Overview

- **Weak Reference Model**: Earlier strong checkpoint or smaller variant
- **Logit Mixing**: Combine weak and strong logits with tunable weight λ
- **Curriculum Learning**: Select harder examples (where weak and strong disagree most)
- **Joint Training**: Update strong model with mixed-logit targets
- **Calibration**: Mixture weight λ balances signal strength and training stability

## Implementation

Implement logit mixing for training:

```python
import torch
import torch.nn.functional as F

def compute_mixed_logits(strong_logits, weak_logits, lambda_weight=0.3):
    """Mix weak and strong logits to create training targets."""
    mixed = lambda_weight * strong_logits + (1 - lambda_weight) * weak_logits
    return mixed

def compute_weak_driven_loss(strong_logits, weak_logits, labels, lambda_weight=0.3, temperature=1.0):
    """Compute loss using mixed logits as targets."""
    # Compute mixed-logit targets
    mixed_logits = compute_mixed_logits(strong_logits, weak_logits, lambda_weight)

    # Temperature scaling for smoothing
    strong_logits_scaled = strong_logits / temperature
    mixed_logits_scaled = mixed_logits.detach() / temperature

    # KL divergence: strong model matches mixed-logit distribution
    loss = F.kl_div(
        F.log_softmax(strong_logits_scaled, dim=-1),
        F.softmax(mixed_logits_scaled, dim=-1),
        reduction='batchmean'
    )

    return loss

# Example training step
def weak_driven_training_step(strong_model, weak_model, batch, optimizer, lambda_weight=0.3):
    """Single weak-driven training step."""
    inputs, labels = batch

    # Get logits from both models
    with torch.no_grad():
        weak_logits = weak_model(inputs).logits

    strong_logits = strong_model(inputs).logits

    # Compute weak-driven loss
    loss = compute_weak_driven_loss(strong_logits, weak_logits, labels, lambda_weight)

    # Update strong model only
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

Implement curriculum-enhanced data selection:

```python
def select_hard_examples(strong_model, weak_model, dataset, top_k=100):
    """Select examples where weak and strong models most disagree."""
    disagreements = []

    for i, example in enumerate(dataset):
        with torch.no_grad():
            weak_logits = weak_model(example)
            strong_logits = strong_model(example)

        # Measure disagreement as KL divergence
        weak_probs = F.softmax(weak_logits, dim=-1)
        strong_probs = F.softmax(strong_logits, dim=-1)
        kl_divergence = (weak_probs * (torch.log(weak_probs) - torch.log(strong_probs))).sum()

        disagreements.append((i, kl_divergence.item()))

    # Sort by disagreement and select top-k
    disagreements.sort(key=lambda x: x[1], reverse=True)
    hard_indices = [idx for idx, _ in disagreements[:top_k]]

    return [dataset[i] for i in hard_indices]
```

Integrate into full training loop:

```python
def weak_driven_training_loop(strong_model, weak_model, train_dataloader, num_epochs=10, lambda_weight=0.3):
    """Full weak-driven training."""
    optimizer = torch.optim.AdamW(strong_model.parameters(), lr=2e-5)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_dataloader):
            loss = weak_driven_training_step(strong_model, weak_model, batch, optimizer, lambda_weight)
            total_loss += loss

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch}, Batch {batch_idx + 1}: Loss = {avg_loss:.4f}")

        # Curriculum: every K epochs, refresh hard example selection
        if (epoch + 1) % 5 == 0:
            hard_examples = select_hard_examples(strong_model, weak_model, train_dataloader, top_k=1000)
            # Re-weigh dataloader to emphasize hard examples
            train_dataloader = create_curriculum_dataloader(hard_examples, batch_size=32)

    return strong_model
```

## Practical Guidance

| Parameter | Recommendation | Notes |
|-----------|-----------------|-------|
| λ (mixing weight) | 0.2-0.4 | Higher λ emphasizes strong model; lower emphasizes weak alternative signals. |
| Weak model source | Previous checkpoint or smaller variant | Either works; ensure weak model is sufficiently different from strong. |
| Temperature | 1.0-2.0 | Higher temperature softens probability distributions for learning. |
| Curriculum refresh | Every 5-10 epochs | Periodically select new hard examples to maintain challenge. |
| Learning rate | 2e-5 (LLMs) | Conservative rate; mixing introduces stability. |

**When to Use**
- Fine-tuning has hit plateau (loss stagnates despite training)
- You have access to a weaker reference model
- Math reasoning, code generation, or other domains with long-tail difficulty
- You want to improve discrimination on hard negatives

**When NOT to Use**
- Initial training stages (supervised signal still informative)
- Tasks without clear right/wrong answers
- When weak model quality is very poor

**Common Pitfalls**
- λ too high makes mixture indistinguishable from standard training; try lower values
- Using weak model from earlier in training when still suboptimal; use converged checkpoint
- Not adjusting temperature; helps stabilize when mixing distributions
- Forgetting to keep weak model frozen; never update it during training

## Reference

See https://arxiv.org/abs/2602.08222 for empirical validation on mathematical reasoning (MATH, AIME) and code generation (HumanEval), including analysis of gradient amplification and saturation breakthrough mechanisms.
