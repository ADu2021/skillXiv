---
name: pretrain-zero-active-pretraining
title: "PretrainZero: Reinforcement Active Pretraining with Bilevel Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.03442
keywords: [pretraining, active-learning, reinforcement-learning, curriculum-learning, data-selection]
description: "Bilevel min-max optimization where mask generator selects informative spans from pretraining data and mask predictor recovers them via chain-of-thought, enabling effective RL pretraining on noisy corpora without supervised fine-tuning."
---

## Summary

PretrainZero introduces a reinforcement learning framework combining two coupled tasks—a mask generation policy that actively selects informative spans and a mask prediction policy using chain-of-thought reasoning to recover them. This bilevel optimization enables effective RL-based pretraining directly on noisy real-world corpora like Wikipedia without requiring supervised fine-tuning or external reward models.

## Core Technique

**Bilevel Optimization:** Formulate as a min-max game:
- **Max (Generator):** Select challenging, learnable spans that improve predictor
- **Min (Predictor):** Recover selected spans through reasoning

**Mask Generation Policy:** Learns to identify spans that are:
- Informative (removing them makes prediction harder)
- Learnable (not random noise)
- Beneficial (predictor improves on them)

**Mask Prediction Policy:** Uses chain-of-thought reasoning to predict masked spans based on context.

## Implementation

**Generator network:** Selects which spans to mask:
```python
class MaskGenerator(nn.Module):
    def forward(self, text_tokens, context_embeddings):
        # Score each span for masking
        span_scores = self.scorer(context_embeddings)
        # Select top-k spans
        mask_idx = topk(span_scores, k=num_masks)
        return mask_idx

# Generate masks during pretraining
mask_indices = generator.forward(tokens, context)
masked_text = apply_masks(text, mask_indices)
```

**Predictor with chain-of-thought:**
```python
class MaskPredictor(nn.Module):
    def forward(self, masked_text, mask_positions):
        # Generate reasoning about context
        reasoning = self.cot_generator(masked_text, mask_positions)
        # Predict masked spans
        predictions = self.span_predictor(concat(masked_text, reasoning))
        return predictions

# Supervised loss on masked spans
loss = cross_entropy(predictions, original_spans)
```

**Bilevel optimization:** Alternate between:
1. Optimize generator to select spans that predictor struggles on
2. Optimize predictor to recover selected spans

```python
for epoch in range(num_epochs):
    # Step 1: Generator optimization
    # Select spans that maximize predictor loss
    loss_generator = -predictor_loss(generator_selection)
    generator_optimizer.step(loss_generator)

    # Step 2: Predictor optimization
    # Learn to predict generator-selected spans
    loss_predictor = compute_prediction_loss()
    predictor_optimizer.step(loss_predictor)
```

## When to Use

- Large-scale pretraining on noisy unlabeled data
- Scenarios where data quality is variable
- Tasks requiring active curriculum learning
- Pretraining before fine-tuning on downstream tasks

## When NOT to Use

- Scenarios with clean, high-quality labeled data
- Tasks where supervised pretraining is simpler
- Real-time pretraining where alternating optimization is slow
- Applications without chain-of-thought reasoning benefit

## Key References

- Bilevel optimization and min-max games
- Active learning and curriculum design
- Chain-of-thought reasoning for understanding
- Pretraining and self-supervised learning
