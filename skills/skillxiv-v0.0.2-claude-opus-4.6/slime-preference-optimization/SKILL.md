---
name: slime-preference-optimization
title: "SLIME: Stabilized Likelihood Implicit Margin Enforcement for Preference Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.02383"
keywords: [Preference Optimization, Reference-Free, Margin-Based, Stability, LLM Alignment]
description: "Optimize model preferences by decoupling preference learning from generation quality. Explicitly maximize chosen response likelihood while using token-level stabilization to prevent quality degradation from over-suppressing rejected responses."
---

# SLIME: Stabilized Preference Optimization

## Problem
Margin-based preference optimization methods like DPO and SimPO can degrade the quality of preferred responses while optimizing margins. Over-aggressive suppression of rejected tokens removes valid syntax and reasoning patterns.

A reference-free method that preserves preferred response quality while improving alignment is needed.

## Core Concept
SLIME decouples preference learning from generation quality through three components: explicit likelihood anchoring of chosen responses, token-level softplus stabilization preventing probability collapse, and dual-margin optimization with soft and hard constraints.

This ensures the model maintains high probability for preferred continuations while treating rejected responses with measured regularization rather than complete suppression.

## Architecture Overview

- **Likelihood Anchoring**: Maximize log-probability of chosen responses explicitly
- **Token-Level Stabilization**: Softplus penalty prevents rejected probability collapse
- **Hard Margin**: Defines victory condition where loss becomes zero
- **Soft Margin**: Continued gradient signal beyond hard margin
- **Reference-Free**: No need for reference model like in DPO
- **Dual-Margin Design**: Balances margin satisfaction with output quality

## Implementation

### Step 1: Define SLIME Loss Function
Implement anchored likelihood with stabilized margins.

```python
import torch
import torch.nn.functional as F

def slime_loss(chosen_logits, rejected_logits, hard_margin=1.0, soft_margin=0.5):
    """Compute SLIME loss combining likelihood and margin objectives."""
    # Get probabilities
    chosen_probs = F.softmax(chosen_logits, dim=-1)
    rejected_probs = F.softmax(rejected_logits, dim=-1)

    # Component 1: Likelihood anchoring—maximize chosen probability
    # Higher is better, so negative for loss
    chosen_log_prob = torch.log(chosen_probs + 1e-8)
    likelihood_loss = -chosen_log_prob.mean()

    # Component 2: Margin constraint with dual margins
    margin = chosen_probs - rejected_probs

    # Hard margin: loss = 0 when margin >= hard_margin
    hard_margin_loss = F.relu(hard_margin - margin).pow(2).mean()

    # Soft margin: additional penalty beyond hard margin (continued learning)
    soft_margin_penalty = F.softplus(soft_margin - margin).mean()

    # Component 3: Token-level stabilization—prevent rejected collapse
    # Softplus prevents log(rejected_prob) from exploding negatively
    rejected_stability = F.softplus(-torch.log(rejected_probs + 1e-8)).mean()

    # Total loss: balance all components
    total_loss = (0.5 * likelihood_loss +
                  0.3 * hard_margin_loss +
                  0.1 * soft_margin_penalty +
                  0.1 * rejected_stability)

    return total_loss
```

### Step 2: Batch Preference Training
Train on preference pairs using SLIME objective.

```python
def train_with_slime(model, preference_pairs, num_epochs=3, learning_rate=1e-4):
    """Train model on preference pairs using SLIME loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in create_batches(preference_pairs, batch_size=32):
            losses = []

            for chosen, rejected in batch:
                # Tokenize inputs
                chosen_tokens = model.tokenize(chosen)
                rejected_tokens = model.tokenize(rejected)

                # Get logits
                chosen_logits = model.get_logits(chosen_tokens[:-1])
                rejected_logits = model.get_logits(rejected_tokens[:-1])

                # Target tokens for next position
                chosen_targets = chosen_tokens[1:]
                rejected_targets = rejected_tokens[1:]

                # Extract logits for target positions
                chosen_target_logits = chosen_logits[torch.arange(len(chosen_targets)), chosen_targets]
                rejected_target_logits = rejected_logits[torch.arange(len(rejected_targets)), rejected_targets]

                # Compute SLIME loss
                loss = slime_loss(chosen_target_logits, rejected_target_logits)
                losses.append(loss)

            # Batch gradient update
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += batch_loss.item()

        print(f"Epoch {epoch}, Average Loss: {total_loss / len(batch)}")

    return model
```

### Step 3: Token-Level Stabilization Module
Implement softplus-based stabilization for rejected responses.

```python
class StabilizedMarginOptimizer:
    def __init__(self, hard_margin=1.0, soft_margin=0.5, stability_strength=0.1):
        self.hard_margin = hard_margin
        self.soft_margin = soft_margin
        self.stability_strength = stability_strength

    def compute_stabilization_penalty(self, rejected_logits):
        """Prevent rejected probability from collapsing."""
        rejected_probs = F.softmax(rejected_logits, dim=-1)

        # Log probabilities
        log_rejected = torch.log(rejected_probs + 1e-8)

        # Softplus prevents extreme negative values
        stability_penalty = F.softplus(-log_rejected)

        # Weighted penalty
        total_penalty = self.stability_strength * stability_penalty.mean()

        return total_penalty

    def compute_margin_loss(self, chosen_logits, rejected_logits):
        """Compute dual-margin loss."""
        chosen_probs = F.softmax(chosen_logits, dim=-1)
        rejected_probs = F.softmax(rejected_logits, dim=-1)

        margin = chosen_probs - rejected_probs

        # Hard margin satisfaction
        hard_loss = F.relu(self.hard_margin - margin).pow(2).mean()

        # Soft margin for continued learning
        soft_loss = F.softplus(self.soft_margin - margin).mean()

        return hard_loss + 0.5 * soft_loss
```

### Step 4: Comparison with Baselines
Validate SLIME quality preservation against DPO/SimPO.

```python
def compare_preference_methods(preference_pairs, test_prompts):
    """Compare SLIME against DPO and SimPO."""
    results = {}

    for method in ['slime', 'dpo', 'simpo']:
        model = initialize_model()

        if method == 'slime':
            model = train_with_slime(model, preference_pairs)
        elif method == 'dpo':
            model = train_with_dpo(model, preference_pairs)
        else:
            model = train_with_simpo(model, preference_pairs)

        # Evaluate quality preservation
        preferred_quality = evaluate_response_quality(model, test_prompts)
        alignment = evaluate_preference_alignment(model, preference_pairs)

        results[method] = {
            'quality': preferred_quality,
            'alignment': alignment
        }

    return results
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hard margin | 0.8-1.2 | Victory condition threshold |
| Soft margin | 0.3-0.5 | Below hard; continued learning |
| Stability strength | 0.05-0.2 | Prevent probability collapse |
| Likelihood weight | 0.4-0.6 | Balance quality preservation |
| Learning rate | 1e-4 to 5e-4 | Standard preference learning |

### When to Use

- Aligning LLMs with human preferences while preserving capability
- Tasks where quality of preferred response is critical
- Scenarios where rejected responses contain valid patterns (code syntax, reasoning steps)
- Reference-free alignment (no external reference model available)
- Multi-model comparison on alignment leaderboards (MT-Bench, Arena)

### When Not to Use

- When preferred responses are inherently low-quality (optimization cannot fix)
- Tasks where strong margin separation is sole objective
- Models already well-aligned (further optimization not needed)
- Computational constraints requiring simpler methods
- Scenarios where reference models are accurate and available

### Common Pitfalls

1. **Margin miscalibration**: Margins too tight create instability; too loose lose learning signal. Start with default, monitor.
2. **Stability-alignment trade-off**: High stability strength may prevent good margin learning. Balance via validation set.
3. **Quality degradation**: SLIME prevents it better than baselines, but monitor preferred response outputs for subtle decline.
4. **Batch size sensitivity**: Small batches may increase variance. Use batch size 16+.

## Reference
SLIME: Stabilized Likelihood Implicit Margin Enforcement for Preference Optimization
https://arxiv.org/abs/2602.02383
