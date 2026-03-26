---
name: llm-diverse-creative-writing
title: "Modifying Large Language Model Post-Training for Diverse Creative Writing"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.17126"
keywords: [LLM, Creative Writing, Diversity, DPO, ORPO, Preference Optimization]
description: "Learn to enhance LLM post-training for diverse creative outputs by weighting training pairs using deviation metrics (semantic and style diversity). Applies to models where standard alignment reduces diversity, enabling competitive quality with higher output variety."
---

## Core Concept

Standard LLM post-training improves output quality but often reduces diversity—a critical limitation for creative writing. This skill introduces **Diversified Preference Optimization (DDPO and DORPO)**, which weights training samples by their deviation from peers sharing the same prompt. Deviation measures how unique and diverse a training example is, prioritizing rare high-quality instances to maintain both quality and diversity.

## Architecture Overview

- **Deviation Calculation**: Computes mean pairwise distance between a training sample and all others with identical prompt using semantic embeddings
- **Dual Diversity Metrics**: Combines semantic diversity (via Jina embeddings) and style diversity (via specialized embeddings)
- **Loss Weighting**: Scales DPO/ORPO loss terms by winning response deviation to emphasize diverse examples
- **Reward Model**: Trained on external signals (e.g., Reddit upvotes) for quality evaluation
- **Preference Pair Creation**: Transforms score-based training data into preference pairs for DPO/ORPO training

## Implementation Steps

### Step 1: Calculate Semantic Diversity via Embeddings

This step converts training responses into embeddings and computes deviation as mean pairwise distance within prompt groups.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize embedding model
embedding_model = SentenceTransformer('jinaai/jina-embeddings-v3')

def calculate_deviation(responses_for_prompt):
    """
    Calculate deviation (mean pairwise distance) for a list of responses
    sharing the same prompt. Lower values indicate similarity;
    higher values indicate uniqueness.
    """
    embeddings = embedding_model.encode(responses_for_prompt)

    n = len(embeddings)
    if n <= 1:
        return 0.0

    # Compute pairwise distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(dist)

    return np.mean(distances) if distances else 0.0
```

### Step 2: Compute Style Diversity

Style diversity captures writing patterns beyond semantic content, using style-specific embeddings that focus on sentence structure, vocabulary choice, and tone.

```python
def compute_style_features(text):
    """
    Extract style features: sentence length variance,
    vocabulary richness, and punctuation patterns.
    """
    sentences = text.split('.')
    sentence_lengths = [len(s.split()) for s in sentences if s.strip()]

    vocab_size = len(set(text.lower().split()))
    total_words = len(text.split())
    vocab_richness = vocab_size / max(total_words, 1)

    punctuation_count = sum(1 for c in text if c in '!?,;:')

    features = {
        'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
        'sentence_length_variance': np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0,
        'vocab_richness': vocab_richness,
        'punctuation_density': punctuation_count / max(len(text), 1)
    }
    return features

def style_distance(style_features_1, style_features_2):
    """
    Compute Euclidean distance between style feature vectors.
    """
    feat_vec_1 = np.array(list(style_features_1.values()))
    feat_vec_2 = np.array(list(style_features_2.values()))
    return np.linalg.norm(feat_vec_1 - feat_vec_2)
```

### Step 3: Create Preference Pairs with Deviation Weighting

Convert score-based training data into preference pairs, annotating each with semantic and style deviation values.

```python
def create_preference_pairs(training_data):
    """
    Transform score-based data (dict with prompt, responses, scores)
    into preference pairs with deviation weights. Assumes responses
    are sorted by quality score (higher is better).
    """
    preference_pairs = []

    for example in training_data:
        prompt = example['prompt']
        responses = example['responses']  # sorted by score
        scores = example['scores']

        semantic_dev = calculate_deviation(responses)

        style_features = [compute_style_features(r) for r in responses]
        style_dev = np.mean([
            style_distance(style_features[i], style_features[j])
            for i in range(len(style_features))
            for j in range(i + 1, len(style_features))
        ]) if len(style_features) > 1 else 0

        # Pair highest-scoring response with lower-scoring ones
        for i in range(len(responses) - 1):
            win_response = responses[-1]  # Highest scored
            lose_response = responses[i]  # Lower scored

            # Deviation of winning response
            win_dev = (semantic_dev + style_dev) / 2

            pair = {
                'prompt': prompt,
                'winning_response': win_response,
                'losing_response': lose_response,
                'deviation_weight': win_dev,
                'semantic_deviation': semantic_dev,
                'style_deviation': style_dev
            }
            preference_pairs.append(pair)

    return preference_pairs
```

### Step 4: Implement Diversified DPO (DDPO)

Extend standard DPO loss by scaling with the deviation weight of the winning response, emphasizing unique high-quality examples.

```python
import torch
import torch.nn.functional as F

def diversified_dpo_loss(model, batch, beta=0.5):
    """
    Diversified DPO loss scales standard DPO by deviation weight.
    beta: temperature parameter for preference modeling.
    Deviation weight emphasizes rare, high-quality responses.
    """
    prompts = batch['prompts']
    win_responses = batch['winning_responses']
    lose_responses = batch['losing_responses']
    deviation_weights = batch['deviation_weights']

    # Forward pass for winning responses
    win_logits = model.compute_logits(prompts, win_responses)
    win_log_probs = F.log_softmax(win_logits, dim=-1).sum(dim=-1)

    # Forward pass for losing responses
    lose_logits = model.compute_logits(prompts, lose_responses)
    lose_log_probs = F.log_softmax(lose_logits, dim=-1).sum(dim=-1)

    # Standard DPO loss
    log_odds = win_log_probs - lose_log_probs
    dpo_loss = -F.logsigmoid(beta * log_odds)

    # Scale by deviation weight (higher deviation = higher loss weight)
    weighted_loss = dpo_loss * deviation_weights

    return weighted_loss.mean()
```

### Step 5: Train with Iterative Refinement

Use the weighted loss in a training loop with a standard optimizer, optionally iterating with data refresh.

```python
def train_diversified_model(
    model,
    preference_pairs,
    num_epochs=3,
    batch_size=8,
    learning_rate=1e-5
):
    """
    Train model using diversified preference pairs.
    Iterates through epochs of preference-based training.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Shuffle and batch preference pairs
        import random
        random.shuffle(preference_pairs)

        for i in range(0, len(preference_pairs), batch_size):
            batch_pairs = preference_pairs[i:i + batch_size]

            batch = {
                'prompts': [p['prompt'] for p in batch_pairs],
                'winning_responses': [p['winning_response'] for p in batch_pairs],
                'losing_responses': [p['losing_response'] for p in batch_pairs],
                'deviation_weights': torch.tensor(
                    [p['deviation_weight'] for p in batch_pairs],
                    dtype=torch.float32
                )
            }

            loss = diversified_dpo_loss(model, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

    return model
```

## Practical Guidance

**When to Use:**
- Fine-tuning creative writing models (fiction, poetry, dialogue generation)
- Balancing quality improvements with output diversity
- Working with limited preference data where sample quality varies significantly
- Models that show quality improvement but excessive repetitive output patterns

**When NOT to Use:**
- Tasks requiring strict consistency and minimal variation (factual extraction, code generation)
- Scenarios with insufficient training data to compute meaningful deviation metrics
- When semantic/style embedding models aren't well-calibrated for your domain

**Hyperparameter Tuning:**
- **Beta (temperature)**: 0.5-1.0 ranges from conservative to aggressive preference modeling. Start at 0.5 and increase if model becomes too risk-averse.
- **Deviation weighting**: Normalize deviation values to [0, 1] range to ensure stable loss scaling across batches.
- **Embedding model choice**: Jina embeddings v3 work well for general text; consider domain-specific models (e.g., SciBERT) for specialized writing.

**Common Pitfalls:**
- Unstable deviation metrics on small prompt groups (fewer than 3 responses per prompt)
- Over-weighting high-deviation examples can lead to quality degradation; use regularization with base DPO loss
- Style features may need domain customization; generic features work but domain-specific ones perform better

## References

- arXiv:2503.17126 - Full paper on Diversified DPO and DORPO
- Rafailov et al. (2023) - Direct Preference Optimization (DPO)
- Hang et al. (2023) - Odds Ratio Preference Optimization (ORPO)
- Jina AI Embeddings - https://jina.ai/embeddings/
