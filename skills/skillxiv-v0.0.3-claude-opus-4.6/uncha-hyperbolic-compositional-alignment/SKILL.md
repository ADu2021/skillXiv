---
name: uncha-hyperbolic-compositional-alignment
title: "UNCHA: Uncertainty-Guided Compositional Alignment in Hyperbolic VLMs"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22042"
keywords: [Hyperbolic VLM, Entailment Loss, Uncertainty Calibration, Compositional Alignment, Part-to-Whole Semantics]
description: "Swap fixed entailment loss with uncertainty-guided piecewise-continuous formulation to improve part-to-whole compositional alignment in hyperbolic vision-language models by +3.0-3.1% on zero-shot classification. Use when training hyperbolic VLMs on part-object relationships and need better multi-object compositional understanding."
category: "Component Innovation"
---

## What This Skill Does

Replace the fixed entailment loss in hyperbolic vision-language models with an uncertainty-calibrated piecewise-continuous formulation that adaptively modulates contrastive temperatures. This improves compositional alignment by +3.0 percentage points on ImageNet and +7.74 pp on multi-object representation tasks.

## The Component Swap

The old HyCoCLIP entailment loss becomes zero after achieving alignment, preventing further refinement:

```python
# Old approach: fixed entailment loss saturates
loss_old = max(0, phi(part, whole) - eta * omega(part))
```

The new UNCHA approach replaces this with three coordinated modifications. First, add a piecewise-continuous component that prevents saturation:

```python
# New: piecewise-continuous entailment with adaptive temperature
loss_ent_base = max(0, phi(part, whole) - eta * omega(part))
loss_ent_refined = loss_ent_base + alpha * phi(part, whole)  # Non-zero gradient after alignment

# Uncertainty calibration modulates loss weight by estimated uncertainty
u_part = model.predict_uncertainty(part)
loss_ent_calibrated = loss_ent_refined * torch.exp(-u_part) + u_part
```

Second, add an entropy regularization term to prevent degenerate uncertainty estimates:

```python
# Entropy regularization on uncertainty predictions
u_normalized = torch.softmax(uncertainties, dim=0)
entropy_reg = -torch.sum(u_normalized * torch.log(u_normalized + 1e-8))
loss_entropy = entropy_reg  # Encourages non-trivial uncertainty
```

Third, modulate the contrastive temperature by uncertainty to soften supervision for uncertain pairs:

```python
# Adaptive temperature based on uncertainty (for softmax in contrastive loss)
temperature_default = 0.07
temperature_adaptive = torch.exp(uncertainty / 2.0) * temperature_default
```

## Performance Impact

**Zero-shot classification (ViT-B/16):**
- ImageNet: +3.0 percentage points (45.8% → 48.8%)
- CIFAR-100: +3.1 pp (60.1% → 63.2%)

**Multi-object compositional representation (3 objects):**
- +7.74 percentage points on part-to-whole alignment (73.22% → 80.96%)

**Trade-off:**
- Fine-grained recognition (CUB): -1.6 pp (16.4% → 14.8%) — slightly worse on fine-grained categories due to increased part-level supervision

## When to Use

- Training hyperbolic vision-language models with part-level annotations
- Optimizing compositional alignment in multi-object recognition tasks
- When you have access to part-level bounding boxes or segmentation masks
- Models using hyperbolic embeddings (e.g., HyCoCLIP or descendants)

## When NOT to Use

- Datasets without part-level annotations
- Fine-grained recognition tasks where part-to-whole trade-offs may harm accuracy
- Vision-language models using Euclidean embeddings (requires hyperbolic geometry)
- Scenarios where computational cost of uncertainty prediction is prohibitive

## Implementation Checklist

To adopt this component swap:

1. **Verify your model has hyperbolic embeddings:**
   ```python
   # Check that your model uses hyperbolic distance
   assert hasattr(model, 'curvature') or 'hyperbolic' in str(type(model))
   ```

2. **Add uncertainty predictor head:**
   - Small MLP mapping embeddings to scalar uncertainty: `[d_model] → [hidden] → [1]`
   - Initialize with small values (log-variance ~-5) for stable training

3. **Replace entailment loss:**
   - Replace `max(0, φ(p,q) − η·ω(p))` with piecewise form
   - Add entropy regularization with weight ~0.01-0.1

4. **Adaptive temperature in contrastive loss:**
   - Change `logits = embeddings @ embeddings.T / tau` to `logits = embeddings @ embeddings.T / tau_adaptive`

5. **Verify improvements:**
   - Measure on same zero-shot classification benchmark
   - Compare ImageNet top-1 before/after
   - Check that part-level alignment improves (ComCo metric if available)

6. **Hyperparameter tuning if needed:**
   - `alpha`: weight of the piecewise term (default: 0.1-0.5)
   - `entropy_weight`: entropy regularization strength (default: 0.01-0.1)
   - `uncertainty_scale`: scaling factor for temperature modulation (default: 1.0)

## Related Work

This builds on HyCoCLIP (hyperbolic VLMs) and relates to uncertainty calibration methods in vision-language models. The entropy regularization pattern resembles label smoothing in supervised learning, but applied to learned uncertainty estimates.
