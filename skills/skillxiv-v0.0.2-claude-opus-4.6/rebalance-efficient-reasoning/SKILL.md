---
name: rebalance-efficient-reasoning
title: "ReBalance: Efficient Reasoning with Balanced Thinking"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.12372"
keywords: [Reasoning Models, Computational Efficiency, Confidence-Guided Steering, Training-Free Optimization]
description: "Diagnose and correct reasoning inefficiencies (overthinking and underthinking) in large reasoning models using confidence-based steering vectors, without retraining. Enables optimal reasoning budgets across model scales."
---

# ReBalance: Efficient Reasoning with Balanced Thinking

Large reasoning models often face a critical tradeoff: they either "overthink" simple problems (wasting computation) or "underthink" difficult ones (producing weak results). Existing approaches require either retraining or task-specific tuning, making them impractical for deployment. ReBalance solves this through confidence monitoring and dynamic steering—a training-free framework that adapts reasoning behavior in real-time.

The core insight is elegant: confidence signals encode reasoning dynamics. High variance in confidence indicates overthinking (the model keeps revising answers), while sustained overconfidence signals underthinking (the model moves forward without adequate exploration). By monitoring these patterns and steering the model's hidden states, ReBalance balances computational efficiency with reasoning quality.

## Core Concept

ReBalance operates via confidence-conditioned steering vectors derived from small-scale reference datasets. The approach diagnoses two pathological reasoning modes and corrects them dynamically:

- **Overthinking Detection**: Identified through coefficient of variation in confidence scores across reasoning steps
- **Underthinking Detection**: Recognized via consistent but misaligned confidence (high confidence, low correctness)
- **Dynamic Steering**: Real-time activation magnitude and direction modulation based on detected mode

The framework aggregates hidden states from reference examples to create "reasoning mode prototypes"—compact representations of optimal reasoning trajectories. These prototypes generate steering vectors that guide the model's internal representations without modifying weights.

## Architecture Overview

- **Reference Dataset Processing**: Small auxiliary dataset (100-500 examples) to extract reasoning mode prototypes
- **Confidence Tracking**: Real-time monitoring of prediction uncertainty across reasoning steps
- **Hidden State Aggregation**: Collecting and averaging activations from reference examples at each layer
- **Dynamic Steering Vector**: Computed from layer-wise prototype differences, scaled by confidence signals
- **Inference-Time Modulation**: Apply steering amplified by confidence variance (overthinking) or dampened (underthinking)

## Implementation Steps

### Step 1: Build Reasoning Mode Prototypes

Extract hidden state activations from a reference dataset for correct and incorrect reasoning trajectories. For each layer, compute the mean hidden states across successful completions.

```python
import torch
import numpy as np

def extract_prototypes(model, reference_dataset, num_layers=24):
    """Extract reasoning mode prototypes from reference examples."""
    prototypes = {layer_idx: [] for layer_idx in range(num_layers)}

    for example in reference_dataset:
        prompt = example['prompt']
        correct_output = example['correct_reasoning']

        # Forward pass with hidden state extraction
        with torch.no_grad():
            outputs = model.generate(
                prompt,
                max_new_tokens=2000,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            hidden_states = outputs.hidden_states

        # Collect layer-wise activations
        for layer_idx in range(num_layers):
            layer_hidden = hidden_states[layer_idx]
            prototypes[layer_idx].append(layer_hidden)

    # Average across examples
    for layer_idx in range(num_layers):
        prototypes[layer_idx] = torch.mean(
            torch.stack(prototypes[layer_idx]), dim=0
        )

    return prototypes
```

### Step 2: Compute Confidence Signals

Track prediction confidence throughout generation to detect reasoning pathologies.

```python
def compute_confidence_trajectory(logits_sequence):
    """
    Compute confidence metrics across reasoning steps.
    Returns: (mean_confidence, confidence_variance, consistency_score)
    """
    # Softmax probabilities for each token
    probs = torch.softmax(logits_sequence, dim=-1)

    # Maximum probability = model confidence
    token_confidence = torch.max(probs, dim=-1).values.cpu().numpy()

    # Statistical measures
    mean_conf = np.mean(token_confidence)
    conf_variance = np.var(token_confidence)

    # Coefficient of variation (normalized measure)
    conf_cv = np.std(token_confidence) / (np.mean(token_confidence) + 1e-8)

    return mean_conf, conf_variance, conf_cv
```

### Step 3: Generate Steering Vectors

Compute difference between optimal and current representations, scaled by confidence dynamics.

```python
def compute_steering_vector(current_hidden, prototype, confidence_cv, mode='overthinking'):
    """
    Compute steering adjustment to redirect reasoning.
    mode: 'overthinking' (high variance) or 'underthinking' (high confidence, low performance)
    """
    # Base steering direction
    steering = prototype - current_hidden

    if mode == 'overthinking':
        # For high-variance reasoning, amplify steering to reduce exploration
        # Strength scales with variance
        scaling_factor = min(confidence_cv * 0.5, 1.0)  # Cap at 1.0
    else:  # underthinking
        # For overconfident but wrong reasoning, amplify to force reconsideration
        scaling_factor = max(1.0 - confidence_cv, 0.1)  # Inverse relationship

    return steering * scaling_factor
```

### Step 4: Apply Steering During Inference

Inject steering vectors into the forward pass at key layers to guide reasoning trajectory.

```python
def rebalance_inference(model, prompt, prototypes, threshold_overthinking=0.8,
                        threshold_underthinking=0.3):
    """
    Run inference with real-time confidence-guided steering.
    """
    generated_tokens = []
    confidence_trajectory = []

    for step in range(max_steps):
        # Standard forward pass
        logits = model(prompt + ''.join(generated_tokens))

        # Track confidence
        mean_conf, _, conf_cv = compute_confidence_trajectory(logits)
        confidence_trajectory.append((mean_conf, conf_cv))

        # Diagnose reasoning mode
        if conf_cv > threshold_overthinking:
            mode = 'overthinking'
        elif mean_conf > threshold_underthinking and validation_score_low:
            mode = 'underthinking'
        else:
            mode = None

        # Apply steering if needed
        if mode:
            with torch.no_grad():
                current_hidden = model.get_hidden_states(logits)
                steering_vec = compute_steering_vector(
                    current_hidden,
                    prototypes[current_layer],
                    conf_cv,
                    mode
                )
                # Modify logits via steering
                logits = logits + steering_coefficient * steering_vec

        # Standard token sampling
        next_token = torch.argmax(logits, dim=-1)
        generated_tokens.append(next_token)

    return ''.join(generated_tokens)
```

## Practical Guidance

**Hyperparameters:**
- Reference dataset size: 100-500 examples (task-dependent; more examples = better prototypes)
- Steering magnitude: 0.1-0.5 (controls how aggressively to correct reasoning)
- Overthinking threshold (CV): 0.7-1.0 (higher = more sensitive to variance)
- Underthinking threshold (confidence): 0.2-0.5 (task-dependent, tune on validation set)

**When to Use:**
- Reasoning tasks where latency or compute budget is constrained
- Multi-scale model deployments (same steering works across model sizes)
- When you have access to model internals (hidden states) via library that exposes them
- Tasks requiring consistent quality across difficulty ranges (math, QA, coding)

**When NOT to Use:**
- Black-box APIs without hidden state access
- Real-time streaming scenarios where collecting reference data is impractical
- Tasks with highly variable "correct" reasoning patterns (some right answers require exploration)

**Pitfalls:**
- Reference dataset must represent the same task distribution as inference
- Prototypes from wrong-answer examples can hurt performance; curate carefully
- Steering vectors are layer-specific; applying to wrong layers reduces effectiveness
- Extreme steering magnitudes can cause incoherent outputs; start conservative and increase

## Reference

Paper: [arxiv.org/abs/2603.12372](https://arxiv.org/abs/2603.12372)
