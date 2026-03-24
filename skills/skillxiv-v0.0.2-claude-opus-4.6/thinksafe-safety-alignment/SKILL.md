---
name: thinksafe-safety-alignment
title: "THINKSAFE: Self-Generated Safety Alignment for Reasoning Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.23143"
keywords: [Safety Alignment, Self-Generated Data, Reasoning Models, Refusal Learning]
description: "Align reasoning models to be safe without external supervision by prepending refusal instructions to harmful queries and training on self-generated safe responses. Unlocks latent safety knowledge within the model's native distribution."
---

# THINKSAFE: Self-Generated Safety Alignment

## Problem
Reasoning models often exhibit safety-capability trade-offs when trained with external teacher models. Traditional distillation approaches suffer from distributional mismatch—the model learns from teacher-generated responses rather than maintaining its own reasoning distribution. This leads to degraded reasoning ability or incomplete safety coverage.

RLHF approaches like GRPO are computationally expensive for safety training. A lightweight method that preserves reasoning capability while improving safety is needed.

## Core Concept
THINKSAFE unlocks latent safety knowledge by prepending refusal-oriented instructions to harmful queries. Instead of teaching new safety concepts, it redirects the model's generation probability toward safety-aligned reasoning paths that already exist within its learned distribution.

The model generates its own training data (avoiding distributional mismatch), and a lightweight safety filter ensures only verified-safe responses are retained. This self-supervised approach requires no external teacher.

## Architecture Overview

- **Refusal-Oriented Instructions**: Prepend prompt like "The following prompt is harmful. You should refuse to answer" before harmful queries
- **Dual Sampling**: Apply refusal steering to harmful prompts; sample directly for benign prompts to maintain helpfulness
- **Self-Generated Data**: Student model generates all training responses within its native distribution
- **Safety Filtering**: Validate responses using a safety guard model (e.g., Llama-Guard-3-8B)
- **Supervised Fine-Tuning**: Train on filtered self-generated dataset using standard cross-entropy loss

## Implementation

### Step 1: Prepare Harmful and Benign Prompt Sets
Assemble two curated sets representing different risk profiles.

```python
def prepare_prompt_sets(harmful_source, benign_source):
    """Load and structure harmful and benign prompts."""
    harmful_prompts = load_harmful_queries(harmful_source)
    benign_prompts = load_benign_queries(benign_source)

    # Ensure both sets are balanced
    min_size = min(len(harmful_prompts), len(benign_prompts))

    return {
        'harmful': harmful_prompts[:min_size],
        'benign': benign_prompts[:min_size]
    }
```

### Step 2: Generate Refusal-Steered Responses
Apply refusal instructions to harmful prompts and collect generated responses.

```python
def generate_refusal_steered_responses(harmful_prompts, model, num_samples=3):
    """Generate safe refusals using refusal steering."""
    refusal_instruction = "The following prompt is harmful. You should refuse to answer it."
    training_data = []

    for prompt in harmful_prompts:
        steered_prompt = f"{refusal_instruction}\n\nHarmful prompt: {prompt}"
        for _ in range(num_samples):
            response = model.generate(steered_prompt, temperature=0.7)
            training_data.append({'prompt': prompt, 'response': response})

    return training_data
```

### Step 3: Apply Safety Filtering
Validate all responses using a safety classifier before including in training set.

```python
def apply_safety_filtering(training_data, safety_classifier):
    """Filter responses through safety guard model."""
    filtered_data = []
    for example in training_data:
        safety_score = safety_classifier.classify(example['response'])
        if safety_score in ['safe', 'acceptable']:
            filtered_data.append(example)
    return filtered_data
```

### Step 4: Fine-Tune on Filtered Data
Train the model using standard supervised fine-tuning on the curated dataset.

```python
def finetune_on_safety_data(model, filtered_data, learning_rate=1e-5, epochs=3):
    """Fine-tune model on self-generated safety training data."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for batch in create_batches(filtered_data, batch_size=32):
            inputs = model.tokenizer(batch, padding=True, return_tensors='pt')
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    return model
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Refusal instruction | "...harmful. Should refuse" | Explicit directiveness |
| Samples per prompt | 3-5 | Balance size with diversity |
| Benign/Harmful ratio | 1:1 | Balanced training |
| Safety classifier | Llama-Guard-3-8B | Established safety model |
| Learning rate | 1e-5 to 5e-5 | Standard SFT rates |

### When to Use

- Aligning reasoning models while preserving chain-of-thought capability
- Lightweight safety tuning without RLHF infrastructure
- Domain-specific safety pattern learning
- Iterative safety behavior refinement

### When Not to Use

- Highly misaligned models lacking latent safety knowledge
- Sophisticated context-dependent attacks
- When perfect benign performance is critical
- Safety issues outside model's native reasoning capability

### Common Pitfalls

1. Weak refusal instructions lack directiveness
2. Imbalanced datasets increase false refusals
3. Poor safety classifier quality corrupts training data
4. Insufficient filtering introduces conflicting signals

## Reference
THINKSAFE: Self-Generated Safety Alignment for Reasoning Models
https://arxiv.org/abs/2601.23143
