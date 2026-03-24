---
name: lk-losses-speculative-decoding
title: "LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.23881"
keywords: [Speculative Decoding, Inference Optimization, Training Objectives, LLM Speed, Draft Model]
description: "LK Losses replace KL divergence with direct acceptance rate optimization for speculative decoding, achieving 8-10% improvements without computational overhead."
---

# Technique: Direct Acceptance Rate Optimization for Draft Models

Speculative decoding accelerates LLM inference by using a small draft model to generate token candidates, then verifying them with the larger model. The bottleneck: draft models are trained with KL divergence loss to match the main model's distribution, but KL divergence doesn't directly optimize the metric that matters—acceptance rate. This mismatch is especially severe for small draft models with limited capacity, causing them to converge to suboptimal solutions despite minimizing KL divergence.

LK Losses solve this by directly optimizing the acceptance rate during draft model training. Rather than hoping KL divergence training correlates with acceptance rate, these specialized loss functions target the actual objective: maximizing how many draft tokens the main model accepts.

## Core Concept

The core insight: theoretical equivalence between KL divergence and acceptance rate optimization only holds in the limit of infinite model capacity. In practice, small draft models plateau at solutions that minimize KL divergence but leave acceptance rate suboptimal.

LK Losses define loss functions that directly measure acceptance likelihood. For each draft token, the loss captures whether the main model is likely to accept it—accounting for both token probability and verification dynamics.

This makes training objectives alignment with inference metrics, enabling draft models to learn the behaviors that actually speed up inference.

## Architecture Overview

- **Acceptance Rate Metric**: Computed during training by simulating verification step
- **Loss Function**: Direct differentiable approximation to acceptance probability
- **Drop-in Integration**: Replaces standard KL loss in draft model training
- **No Extra Computation**: Uses existing forward passes; minimal overhead
- **Universal Compatibility**: Works with any verifier architecture (standard LLM or restricted)

## Implementation Steps

LK Losses replace KL divergence objectives in draft model training. Here's how to implement and integrate them:

Define the acceptance rate loss function that directly measures draft token viability:

```python
import torch
import torch.nn.functional as F

def lk_loss(
    draft_logits,           # [batch, vocab_size]
    verifier_logits,        # [batch, vocab_size]
    targets,                # [batch]
    temperature=1.0,
    eps=1e-8
):
    """
    Direct acceptance rate loss for speculative decoding.
    Optimizes the probability that verifier accepts draft tokens.
    """
    batch_size = draft_logits.shape[0]

    # Compute log-probabilities for both models
    draft_log_probs = F.log_softmax(draft_logits / temperature, dim=-1)
    verifier_log_probs = F.log_softmax(verifier_logits / temperature, dim=-1)

    # Extract log-probs for target tokens
    draft_target_log_probs = draft_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    verifier_target_log_probs = verifier_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Acceptance likelihood: min(1, verifier_prob / draft_prob)
    # High acceptance when draft underestimates (prob < verifier)
    # Low penalty when draft overestimates (prob > verifier)
    # This directly models the rejection sampling acceptance condition
    log_acceptance_ratio = verifier_target_log_probs - draft_target_log_probs
    acceptance_prob = torch.clamp(torch.exp(log_acceptance_ratio), max=1.0)

    # Loss: negative log-acceptance (minimize rejection probability)
    acceptance_loss = -torch.log(acceptance_prob + eps)

    return acceptance_loss.mean()
```

Integrate LK Losses into your draft model training loop by replacing KL loss:

```python
from torch.optim import AdamW

class DraftModelTrainer:
    def __init__(self, draft_model, verifier_model, learning_rate=1e-4):
        self.draft_model = draft_model
        self.verifier_model = verifier_model
        self.optimizer = AdamW(draft_model.parameters(), lr=learning_rate)
        self.verifier_model.eval()  # Keep verifier frozen during training

    def train_step(self, batch_input_ids, batch_target_ids):
        """
        Single training step using LK Loss instead of KL divergence.
        """
        # Forward through draft model
        draft_outputs = self.draft_model(batch_input_ids)
        draft_logits = draft_outputs.logits

        # Forward through verifier (no gradients)
        with torch.no_grad():
            verifier_outputs = self.verifier_model(batch_input_ids)
            verifier_logits = verifier_outputs.logits

        # Compute acceptance rate loss
        loss = lk_loss(draft_logits, verifier_logits, batch_target_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataloader, num_epochs=3):
        """Train draft model for multiple epochs."""
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                loss = self.train_step(batch['input_ids'], batch['target_ids'])
                total_loss += loss
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```

## Practical Guidance

**When to Use:**
- Draft models in speculative decoding pipelines
- When draft model capacity is significantly smaller than verifier
- When you observe draft models stuck at high KL divergence but low acceptance rates
- For long-sequence generation where token acceptance rate matters most

**When NOT to Use:**
- When draft model already achieves >95% acceptance rates
- When draft and verifier models have similar capacities (KL loss works fine)
- Real-time systems requiring model retraining (requires offline training)

**Hyperparameters:**
- `temperature`: Controls smoothness of probability distributions (0.5–1.5 typical)
- `eps`: Numerical stability constant (1e-8 typical)
- Can combine with existing draft model optimization techniques

**Implementation Notes:**
- Keep verifier model frozen; it's only used for loss computation
- Works with any verifier architecture; compute its logits during training
- Use same tokenizer/vocabulary for both draft and verifier
- LK Loss is differentiable; use standard backpropagation

**Integration into Inference:**
Once trained, replace KL-divergence-trained draft model with LK-Loss-trained variant. No changes to speculative decoding verification logic needed.

**Performance:**
- Typical improvement: 8–10% relative increase in average acceptance length
- Consistent gains across model sizes (8B to 685B tested)
- No computational overhead; same training time as KL objective
- Scales to multiple model families

---

**Reference:** [LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](https://arxiv.org/abs/2602.23881)
