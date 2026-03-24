---
name: self-eval-t2i
title: "Self-Evaluation Unlocks Any-Step T2I Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.22374
keywords: [text-to-image, diffusion, training, few-step, self-supervised]
description: "Train text-to-image models from scratch for any inference step count via self-evaluation mechanism. Model evaluates its own generated samples using current score estimates as dynamic self-teacher, enabling global distribution matching without external teachers—achieving few-step quality equivalent to many-step models at all budgets."
---

## Overview

Self-E addresses a fundamental limitation in text-to-image generation: achieving high quality at any inference budget requires either many steps or external teacher models. This framework enables from-scratch training where models teach themselves through self-evaluation, eliminating teacher dependence.

## Core Technique

The key insight is that models can evaluate their own samples by leveraging internal score estimation during training.

**Self-Evaluation Mechanism:**
The model generates samples and evaluates them using its current learned score function.

```python
# Self-evaluating text-to-image generation
class SelfEvaluatingGenerator:
    def __init__(self, flow_model):
        self.model = flow_model
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def training_step_with_self_evaluation(self, x_t, x_0, prompt):
        """
        Two complementary training objectives:
        1. Flow matching: local supervision via data
        2. Self-evaluation: global supervision from own scores
        """
        # Objective 1: Flow matching (standard)
        # Teaches model how samples should move toward data
        flow_loss = self.model.flow_matching_loss(x_t, x_0, prompt)

        # Objective 2: Self-evaluation (novel)
        # Use model's own score estimates for additional supervision
        self.eval()  # Freeze for score computation
        with torch.no_grad():
            # Generate samples during training
            x_gen_1 = self.model.sample(x_T=torch.randn_like(x_0), t=1.0)
            x_gen_2 = self.model.sample(x_T=torch.randn_like(x_0), t=0.5)

            # Compute score estimates for generated samples
            # Using Tweedie's formula: score ≈ (x_0_pred - x) / σ²
            score_1 = self.model.score_estimate(x_gen_1, prompt)
            score_2 = self.model.score_estimate(x_gen_2, prompt)

        self.train()  # Unfreeze for training

        # Self-evaluation loss: encourage generated samples toward data
        # This uses model's own score as learning signal
        self_eval_loss = self.classifier_free_guidance_loss(
            x_gen_1, x_gen_2, score_1, score_2, prompt
        )

        # Combined loss
        total_loss = flow_loss + 0.1 * self_eval_loss

        total_loss.backward()
        self.optimizer.step()

        return {'flow_loss': flow_loss, 'self_eval_loss': self_eval_loss}
```

**Classifier-Free Guidance via Self-Scores:**
Use model's own conditional/unconditional score predictions as learning signal.

```python
def classifier_free_guidance_loss(model, x_gen_conditional, x_gen_unconditional):
    """
    Guidance signal comes from model's own score estimates.
    Encourages generated samples toward conditional distribution.
    """
    # Score estimate for conditional (with prompt)
    score_conditional = model.score_function(
        x_gen_conditional, prompt=prompt
    )

    # Score estimate for unconditional (without prompt)
    score_unconditional = model.score_function(
        x_gen_conditional, prompt=None
    )

    # Guidance: conditional score - unconditional score
    # Tells us which direction makes sample more aligned with prompt
    guidance_vector = score_conditional - score_unconditional

    # Loss: encourage following guidance
    guidance_loss = -torch.dot(guidance_vector, x_gen_conditional)

    return guidance_loss
```

**Two-Pass Forward Strategy:**
Computing self-evaluation requires careful stop-gradient operations.

```python
def two_pass_forward_for_self_eval(model, x_t, prompt):
    """
    Forward pass 1: Generate sample (frozen model)
    Forward pass 2: Evaluate sample and compute gradient
    """
    model.eval()

    # Pass 1: Generate with frozen model
    with torch.no_grad():
        x_generated = model.sample(x_T=x_t, steps=num_steps)

    model.train()

    # Pass 2: Evaluate generated sample
    # Model now processes generated sample in training mode
    score_for_generated = model.score_estimate(x_generated, prompt)

    # Loss from evaluation
    evaluation_loss = mse(score_for_generated, target_score)

    return evaluation_loss
```

## When to Use This Technique

Use Self-E when:
- Training text-to-image models from scratch
- Few-step inference is critical
- No access to external teacher models
- Computational resources are constrained

## When NOT to Use This Technique

Avoid this approach if:
- Teacher distillation already works well
- Training from scratch unnecessary
- High-quality teacher models readily available
- Extreme few-step performance (<4 steps) required

## Implementation Notes

The framework requires:
- Flow matching base loss implementation
- Score function estimation (Tweedie's formula)
- Classifier-free guidance computation
- Careful stop-gradient masking for self-teacher
- Training loop managing eval/train mode switches

## Key Performance

- State-of-the-art at all inference budgets (1-50+ steps)
- Particularly strong in few-step regime
- No distillation or external teachers required
- From-scratch training competitive with teacher-based methods

## References

- Flow matching with local data supervision
- Self-evaluation via model score estimates
- Classifier-free guidance from internal scores
- Tweedie's formula for score approximation
- Global distribution matching without teachers
