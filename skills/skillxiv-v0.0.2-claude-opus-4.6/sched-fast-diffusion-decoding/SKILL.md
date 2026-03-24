---
name: sched-fast-diffusion-decoding
title: "Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.02892
keywords: [diffusion language models, early exit, confidence scheduling, inference speedup, training-free]
description: "Accelerate diffusion LLM decoding by halting when confidence stabilizes using progress-dependent thresholds. SchED achieves 3.8-4.0× speedup while retaining 99.8-100% quality—training-free and model-agnostic for any dLLM."
---

## Overview

SchED is a training-free algorithm that aggregates full-span logit margins to accelerate decoding in diffusion language models. It applies smooth, progress-dependent thresholds to determine when generation quality stabilizes sufficiently for early termination.

## When to Use

- Diffusion language model inference speedup
- Long-form generation with early completion
- Training-free acceleration (no retraining required)
- Model-agnostic application across dLLM families
- Robust performance on instruction-tuned models

## When NOT to Use

- Fixed short-length generations
- Tasks requiring full model outputs
- Scenarios where quality must be perfect

## Core Technique

Progress-aware confidence scheduling with logit margin aggregation:

```python
# SchED: Training-free early exit for dLLMs
class SchEDEarlyExit:
    def __init__(self, model):
        self.model = model

    def compute_confidence_schedule(self, progress):
        """Progress-dependent confidence threshold."""
        # Smooth function: lower threshold as progress increases
        base_threshold = 0.7
        progress_factor = progress ** 0.5  # Sqrt schedule
        threshold = base_threshold + 0.2 * progress_factor
        return threshold

    def aggregate_logit_margins(self, logits):
        """Full-span margin aggregation for confidence."""
        # Get top-2 logits
        top_logits, _ = torch.topk(logits, k=2, dim=-1)
        margin = top_logits[:, 0] - top_logits[:, 1]

        # Aggregate across all positions
        avg_margin = margin.mean()
        return avg_margin

    def decode_with_sched(self, prompt, max_tokens):
        """Inference with early stopping."""
        output = []

        for step in range(max_tokens):
            progress = step / max_tokens

            # Generate next token
            token, logits = self.model.generate_with_logits(prompt)
            output.append(token)

            # Compute confidence
            confidence = self.aggregate_logit_margins(logits)

            # Check threshold
            threshold = self.compute_confidence_schedule(progress)

            if confidence > threshold:
                # Early exit: quality stabilized
                break

        return output
```

## Key Results

- 3.8-4.0× speedup across dLLM families
- 99.8-100% quality retention
- Training-free (no retraining required)
- Model-agnostic application

## References

- Original paper: https://arxiv.org/abs/2512.02892
- Focus: Efficient diffusion LLM inference
- Domain: Language models, early stopping
