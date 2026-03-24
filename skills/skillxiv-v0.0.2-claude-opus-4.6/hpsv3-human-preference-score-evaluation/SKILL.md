---
name: hpsv3-human-preference-score-evaluation
title: HPSv3 - Human Preference Score Evaluation for Images
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03789
keywords: [image-quality-assessment, preference-learning, vision-language-models, human-alignment]
description: "A VLM-based preference scoring system trained on 1.17M annotated comparisons to evaluate text-to-image generation quality at scale. Uses uncertainty-aware ranking loss for fine-grained assessment across diverse images and supports iterative quality improvement through chain-of-human-preference sampling."
---

# HPSv3: Human Preference Score Evaluation

## Core Concept

HPSv3 is a comprehensive human preference scoring framework for evaluating text-to-image generation quality. Rather than designing hand-crafted metrics, it learns human preferences directly from large-scale annotated data, enabling robust assessment across diverse image types and quality levels. The approach combines dataset construction, preference modeling via vision-language models, and iterative refinement to provide both an evaluation metric and a method for quality improvement.

## Architecture Overview

- **HPDv3 Dataset**: Curates 1.08M text-image pairs with 1.17M annotated pairwise comparisons spanning state-of-the-art generative models and real-world images
- **Preference Model**: Vision-language model (VLM) backbone trained with uncertainty-aware ranking loss to predict relative image quality
- **Chain-of-Human-Preference (CoHP)**: Iterative refinement mechanism that selects high-quality images at each generation step without requiring additional data
- **Quality Estimation**: Produces normalized preference scores reflecting human consensus on image quality

## Implementation Steps

### Step 1: Prepare Data and Preference Annotations

Construct a diverse dataset of image-text pairs with human preference judgments. The key is capturing comparisons across different quality distributions and model architectures.

```python
# Pseudo-code for preference annotation preparation
import json

def prepare_preference_data(image_pairs, annotations):
    """
    Prepare pairwise preference data from human annotations.

    Args:
        image_pairs: List of (image_a, image_b, text_prompt) tuples
        annotations: List of preference labels (0=tie, 1=a_better, -1=b_better)

    Returns:
        Formatted preference dataset for training
    """
    preference_dataset = []
    for (img_a, img_b, prompt), pref_label in zip(image_pairs, annotations):
        preference_dataset.append({
            "image_a": img_a,
            "image_b": img_b,
            "prompt": prompt,
            "preference": pref_label,
            "confidence": estimate_annotation_confidence(pref_label)
        })
    return preference_dataset
```

Collect annotations from multiple human raters and compute inter-rater agreement to identify high-confidence preference pairs.

### Step 2: Train the Preference Model with Uncertainty-Aware Loss

Use a pre-trained vision-language model and fine-tune it on preference rankings. The uncertainty-aware ranking loss weights training examples by confidence in the preference judgment.

```python
def uncertainty_aware_ranking_loss(model_scores, preference_labels, confidence_weights):
    """
    Compute ranking loss weighted by confidence in preference annotations.

    model_scores: [batch_size, 2] predicted scores for image pairs
    preference_labels: [batch_size] human preference annotations
    confidence_weights: [batch_size] confidence scores for each annotation

    Returns:
        Weighted ranking loss value
    """
    # Margin ranking loss with confidence weighting
    score_a = model_scores[:, 0]
    score_b = model_scores[:, 1]

    # Larger margin for high-confidence preferences
    margin = 0.5
    loss = torch.clamp(margin - preference_labels * (score_a - score_b), min=0)
    weighted_loss = (loss * confidence_weights).mean()

    return weighted_loss
```

Train the model to minimize this loss across the preference dataset. The confidence weighting ensures that noisy or uncertain annotations contribute less to gradient updates.

### Step 3: Implement Chain-of-Human-Preference (CoHP) for Iterative Refinement

Use the trained preference model to iteratively select and regenerate higher-quality images without requiring new training data.

```python
def chain_of_human_preference(prompt, num_iterations=3, candidates_per_step=4):
    """
    Iteratively improve image quality by selecting best candidates each step.

    Args:
        prompt: Text description for image generation
        num_iterations: Number of refinement iterations
        candidates_per_step: Number of images to generate per iteration

    Returns:
        Selected best image after iterative refinement
    """
    current_best = None
    best_score = -float('inf')

    for iteration in range(num_iterations):
        # Generate candidates at current iteration
        candidates = generate_candidates(prompt, num_candidates=candidates_per_step)

        # Score each candidate using trained preference model
        scores = []
        for candidate in candidates:
            # If we have a previous best, compare pairwise
            if current_best is not None:
                comparison_score = preference_model(candidate, current_best, prompt)
            else:
                # Use absolute quality score on first iteration
                comparison_score = preference_model.score_single(candidate, prompt)
            scores.append(comparison_score)

        # Select best image
        best_idx = np.argmax(scores)
        if scores[best_idx] > best_score:
            current_best = candidates[best_idx]
            best_score = scores[best_idx]

    return current_best, best_score
```

This approach enables quality improvement without additional training data by leveraging the preference model's learned criteria.

## Practical Guidance

### When to Use HPSv3

- **Model evaluation**: Assessing quality of text-to-image generation outputs across diverse prompts
- **Quality improvement**: Selecting best images from multiple generations without retraining
- **Comparative analysis**: Benchmarking different generative models using human-aligned scoring
- **Dataset curation**: Filtering high-quality examples during data collection

### When NOT to Use HPSv3

- Real-time single-image scoring: CoHP requires multiple generation steps; use direct scoring for latency-sensitive applications
- Domain shift scenarios: HPSv3 trained on general text-to-image pairs may not align well with highly specialized domains (e.g., medical imaging)
- Computational constraints: VLM inference for pairwise comparisons requires significant memory and computation

### Hyperparameter Recommendations

- **Confidence threshold**: Use 0.7+ for filtering high-confidence preference pairs during training
- **Margin in ranking loss**: Set to 0.3-0.7 depending on preference label distribution
- **CoHP candidates per step**: 4-8 candidates balances quality improvements against computational cost
- **CoHP iterations**: 2-4 iterations typically sufficient; diminishing returns beyond 4 steps

### Key Insights

The core innovation is combining large-scale preference annotation with uncertainty weighting. By explicitly modeling which annotations are reliable, the preference model learns more robust quality criteria. The CoHP approach demonstrates that human preference patterns can drive iterative refinement without ground-truth labels.

## Reference

**HPSv3: Towards Wide-Spectrum Human Preference Score** (arXiv:2508.03789)

The paper introduces HPDv3 dataset with 1.08M images and 1.17M preference annotations, a VLM-based preference model trained with uncertainty-aware loss, and Chain-of-Human-Preference mechanism for iterative quality improvement in text-to-image generation.
