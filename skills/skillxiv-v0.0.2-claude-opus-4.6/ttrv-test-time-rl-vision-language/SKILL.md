---
name: ttrv-test-time-rl-vision-language
title: "TTRV: Test-Time Reinforcement Learning for Vision Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.06783"
keywords: [Test-Time Scaling, Vision-Language Models, Reinforcement Learning, Adaptation, Unlabeled Data]
description: "Adapt vision-language models at test time without labels by extracting implicit reward signals (prediction frequency and entropy) and optimizing via GRPO."
---

# Technique: Test-Time RL for Unlabeled Vision-Language Tasks

Standard VLM fine-tuning requires labeled data and model updates, which isn't always practical at deployment. TTRV enables adaptation during inference on unlabeled test data by extracting implicit reward signals from model outputs themselves: consistency across multiple generations and prediction confidence. This allows models to self-improve on specific tasks without any supervision.

The key insight is that agreement-based rewards (frequency of answers) and entropy-based diversity regularization provide sufficient signal for model improvement during test time. By combining these signals through Group Relative Policy Optimization, VLMs can refine their predictions for the task at hand.

## Core Concept

TTRV operates through two complementary mechanisms:

1. **Frequency-Based Reward**: Generate multiple responses to the same input. Reward answers that appear frequently (consensus signal), encouraging consistent predictions.

2. **Diversity Regularizer**: Entropy-based regularization prevents mode collapse to a single answer, maintaining exploration and robustness.

These rewards feed into GRPO, which adjusts model parameters to favor high-reward predictions while remaining frozen otherwise—enabling efficient test-time adaptation.

## Architecture Overview

- **Input**: An unlabeled test image and query
- **Multi-Response Generation**: Generate K responses using the VLM
- **Reward Computation**: Score responses via frequency and entropy
- **GRPO Optimization**: Update model to maximize predicted rewards
- **Inference**: Use adapted model for remaining test examples

## Implementation Steps

Implement the frequency-based reward that encourages consensus.

```python
def compute_frequency_reward(responses, response_field='answer'):
    """
    Reward responses based on prediction consensus.

    Args:
        responses: List of generated responses (dicts with 'answer' field)
        response_field: Which field to measure frequency on

    Returns:
        rewards: Array of reward values [0, 1]
    """
    import numpy as np
    from collections import Counter

    # Extract the field to measure (e.g., class label)
    predictions = [r.get(response_field, '') for r in responses]

    # Count prediction frequency
    pred_counts = Counter(predictions)
    max_count = max(pred_counts.values()) if pred_counts else 1

    # Reward based on frequency: popular answers get higher reward
    rewards = np.array([
        pred_counts[pred] / max_count for pred in predictions
    ])

    return rewards
```

Implement entropy-based diversity regularization.

```python
def compute_entropy_regularizer(logits, temperature=1.0):
    """
    Entropy-based diversity reward preventing mode collapse.

    Args:
        logits: Model output logits for each response
        temperature: Softmax temperature

    Returns:
        entropy_term: Regularization term [0, 1]
    """
    import torch
    import torch.nn.functional as F

    # Convert logits to probabilities
    probs = F.softmax(logits / temperature, dim=-1)

    # Shannon entropy: H(p) = -sum(p * log(p))
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    # Normalize to [0, 1]
    max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32))
    normalized_entropy = entropy / max_entropy

    return normalized_entropy
```

Combine rewards with diversity control.

```python
def combine_rewards(frequency_rewards, entropy_term, diversity_weight=0.1):
    """
    Combine frequency-based and entropy-based rewards.

    Args:
        frequency_rewards: Consensus signal rewards
        entropy_term: Diversity regularization term
        diversity_weight: Balancing weight for diversity

    Returns:
        combined_rewards: Final reward signal
    """
    import numpy as np

    # Frequency provides main signal; entropy prevents collapse
    combined = frequency_rewards + diversity_weight * entropy_term

    return combined
```

Implement GRPO update for test-time adaptation.

```python
def grpo_update_step(model, image, prompt, num_candidates=5,
                     optimizer=None, learning_rate=1e-4):
    """
    Single GRPO step for test-time adaptation.

    Args:
        model: Vision-language model
        image: Input image
        prompt: Query prompt
        num_candidates: Number of responses to generate
        optimizer: AdamW optimizer
        learning_rate: Learning rate for this update

    Returns:
        adapted_model: Updated model
    """
    import torch
    import torch.nn.functional as F

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Generate multiple responses
    responses = []
    response_logits = []

    for _ in range(num_candidates):
        output = model.generate(
            image, prompt,
            return_logits=True,
            temperature=0.7
        )
        responses.append(output['text'])
        response_logits.append(output['logits'])

    # Compute rewards
    freq_rewards = compute_frequency_reward(
        [{'answer': r} for r in responses]
    )
    entropy_rewards = torch.stack([
        compute_entropy_regularizer(logits) for logits in response_logits
    ])

    rewards = combine_rewards(
        freq_rewards, entropy_rewards, diversity_weight=0.1
    )

    # GRPO optimization: update to maximize expected rewards
    # Generate again to compute gradients
    output = model.generate(
        image, prompt,
        return_logits=True,
        return_log_probs=True
    )

    log_probs = output['log_probs']
    rewards_tensor = torch.tensor(rewards).float()

    # Advantage: how much better than baseline (mean reward)
    advantages = rewards_tensor - rewards_tensor.mean()

    # Policy gradient loss: maximize log_prob * advantage
    loss = -(log_probs * advantages).mean()

    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model
```

Full test-time adaptation loop.

```python
def test_time_adaptation_loop(model, test_images, prompts, num_updates=5):
    """
    Adapt model at test time on unlabeled data.

    Args:
        model: Vision-language model
        test_images: List of test images
        prompts: Corresponding prompts
        num_updates: Number of adaptation updates per image

    Returns:
        adapted_model: Model fine-tuned on test distribution
    """
    import torch

    # Initialize optimizer for test-time updates
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Adapt on each test image
    for image, prompt in zip(test_images, prompts):
        for update_step in range(num_updates):
            model = grpo_update_step(
                model, image, prompt,
                num_candidates=5,
                optimizer=optimizer,
                learning_rate=1e-4
            )

    return model
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Number of candidates | 3-10 per update | More candidates improve reward estimation; balance with compute |
| Update steps per image | 1-10 | Often 1-3 steps sufficient; more for distribution shift |
| Diversity weight | 0.05-0.2 | Prevent mode collapse while keeping frequency signal dominant |
| Learning rate | 1e-4 to 1e-5 | Conservative to avoid instability; test-time budget is limited |
| When to use | VLM tasks with distribution shift at test time | Classification, VQA on new domains |
| When NOT to use | First-pass inference where latency critical | Adaptation adds computational overhead |
| Common pitfall | Overfitting to single test example | Use validation set from test distribution to monitor |

### When to Use TTRV

- Vision tasks with domain shift between training and test (e.g., different photo styles, lighting)
- Deployment scenarios where model improvement without retraining is valuable
- Few-shot or single-example adaptation needed

### When NOT to Use TTRV

- Real-time systems requiring single-pass inference
- Tasks where test distribution is identical to training
- Scenarios where updating model parameters is not permitted

### Common Pitfalls

- **Reward signal instability**: Small batch of candidates may have noisy frequency estimates; increase K
- **Mode collapse**: Entropy weight too low; tune diversity weight
- **Overfitting to test example**: Track validation performance separately
- **Learning rate sensitivity**: Start conservative and increase gradually

## Reference

Paper: https://arxiv.org/abs/2510.06783
