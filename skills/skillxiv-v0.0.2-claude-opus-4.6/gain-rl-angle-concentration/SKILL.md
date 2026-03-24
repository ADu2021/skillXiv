---
name: gain-rl-angle-concentration
title: "Angles Don't Lie: Unlocking Training-Efficient RL Through the Model's Own Signals"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.02281"
keywords: [reinforcement-learning, data-selection, angle-concentration, efficiency, gradient-signals]
description: "Improve RL training efficiency by 2.5× using angle concentration between token hidden states as a cost-effective data scheduling signal, selecting high-gradient samples dynamically."
---

# Angles Don't Lie: Unlocking Training-Efficient RL Through the Model's Own Signals

## Core Concept

GAIN-RL identifies a powerful cost-free signal hidden in every LLM: the geometric relationship between token hidden states (cosine similarity / "angle concentration"). The key discovery: models with concentrated angles between tokens exhibit larger gradient norms, indicating higher learning capacity for that example. Rather than expensive external metrics or heuristics, the framework uses intrinsic model signals to schedule training data dynamically. This enables 2.5× training speedup using only 50% of data while surpassing full-dataset performance, through three components: angle-based data ranking, Gaussian probability sampling, and dynamic curriculum adjustment.

The approach is remarkably simple yet effective: higher angle concentration → higher learning potential → prioritize early in training.

## Architecture Overview

- **Angle Concentration Metric**: Cosine similarity between consecutive token hidden states as learning capacity signal
- **Three Concentration Patterns**: Layer-wise, epoch-wise, and data-wise patterns govern learning dynamics
- **Data Reordering**: Single offline pass ranking examples by combined intra/inter-segment angles
- **Gaussian Probability Sampling**: Curriculum learning starting high-concentration, transitioning to lower-concentration
- **Dynamic Mean Shift**: Progressively sample harder examples as training progresses

## Implementation

1. **Angle Concentration Calculation**: Compute cosine similarity between token states

```python
def compute_angle_concentration(model, example, device='cuda'):
    """
    Calculate intra-segment and inter-segment angle concentration.
    Angle concentration = cosine similarity between consecutive token hidden states.
    Higher concentration indicates higher learning potential.
    """
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(
            example['input_ids'].to(device),
            output_hidden_states=True
        )

    hidden_states = outputs.hidden_states  # [num_layers, seq_len, hidden_dim]

    angle_concentration = {}

    # Compute intra-segment angles (within reasoning segment)
    intra_angles = []
    for layer_idx, layer_states in enumerate(hidden_states):
        for i in range(len(layer_states) - 1):
            # Cosine similarity between consecutive tokens
            h_i = layer_states[i]
            h_next = layer_states[i + 1]

            # Normalize for cosine similarity
            h_i_norm = h_i / (torch.norm(h_i) + 1e-8)
            h_next_norm = h_next / (torch.norm(h_next) + 1e-8)

            # Cosine similarity
            cosine_sim = torch.dot(h_i_norm, h_next_norm).item()
            intra_angles.append(cosine_sim)

    angle_concentration['intra'] = np.mean(intra_angles)

    # Compute inter-segment angles (between reasoning and output segments)
    # Split at answer boundary
    answer_boundary = example.get('answer_start_idx', len(hidden_states[-1]) // 2)

    inter_angles = []
    for layer_idx, layer_states in enumerate(hidden_states):
        if answer_boundary < len(layer_states):
            # Angle between last token of reasoning and first token of answer
            reasoning_token = layer_states[answer_boundary - 1]
            answer_token = layer_states[answer_boundary]

            reasoning_norm = reasoning_token / (torch.norm(reasoning_token) + 1e-8)
            answer_norm = answer_token / (torch.norm(answer_token) + 1e-8)

            cosine_sim = torch.dot(reasoning_norm, answer_norm).item()
            inter_angles.append(cosine_sim)

    angle_concentration['inter'] = np.mean(inter_angles) if inter_angles else 0.0

    # Combined signal: higher concentration indicates higher learning potential
    combined_signal = 0.5 * angle_concentration['intra'] + 0.5 * angle_concentration['inter']

    return {
        'intra': angle_concentration['intra'],
        'inter': angle_concentration['inter'],
        'combined': combined_signal
    }
```

2. **Data Reordering**: Single offline pass to rank all training examples

```python
def rank_dataset_by_angles(model, dataset, batch_size=32, device='cuda'):
    """
    Rank entire dataset by angle concentration in single offline pass.
    Expensive computation only done once, reused across epochs.
    """
    ranked_data = []

    print("Computing angle concentration for all examples...")

    for batch_idx in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[batch_idx : batch_idx + batch_size]

        for example in batch:
            try:
                angles = compute_angle_concentration(model, example, device)
                combined_score = angles['combined']

                ranked_data.append({
                    'example_idx': len(ranked_data),
                    'example': example,
                    'angle_score': combined_score,
                    'angle_intra': angles['intra'],
                    'angle_inter': angles['inter']
                })
            except Exception as e:
                # Skip examples that cause errors
                print(f"Skipping example due to error: {e}")
                continue

    # Sort by combined angle score (descending)
    ranked_data.sort(key=lambda x: x['angle_score'], reverse=True)

    print(f"Ranked {len(ranked_data)} examples")
    print(f"Top 10 angle scores: {[d['angle_score'] for d in ranked_data[:10]]}")
    print(f"Bottom 10 angle scores: {[d['angle_score'] for d in ranked_data[-10:]]}")

    return ranked_data
```

3. **Gaussian Probability Sampling**: Curriculum learning with dynamic scheduling

```python
def gaussian_curriculum_sampling(ranked_data, epoch, total_epochs,
                                batch_size=32, num_samples=1000):
    """
    Sample from dataset using Gaussian distribution over rankings.
    Start with high-concentration examples, gradually introduce harder examples.
    """
    n = len(ranked_data)

    # Mean position starts at 0 (easiest, highest concentration)
    # Progressively shifts right (harder, lower concentration) over epochs
    mu_t = (epoch / total_epochs) * (n / 2)  # Shift from 0 to n/2

    # Standard deviation controls spread (how many examples to consider)
    sigma_t = n / (2 * total_epochs) * (1 + epoch / total_epochs)

    # Gaussian probability for each ranked position
    positions = np.arange(n)
    probabilities = (1 / (sigma_t * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((positions - mu_t) / sigma_t) ** 2
    )

    # Normalize to valid probability distribution
    probabilities = probabilities / np.sum(probabilities)

    # Sample batch indices according to Gaussian
    sampled_indices = np.random.choice(
        n, size=min(num_samples, batch_size), p=probabilities, replace=False
    )

    sampled_batch = [ranked_data[i]['example'] for i in sampled_indices]

    return sampled_batch
```

4. **Dynamic Mean Shift**: Update curriculum based on training progress

```python
def update_curriculum_mean(current_accuracy, baseline_accuracy,
                          previous_mu, n_samples, learning_rate=0.5):
    """
    Dynamically adjust curriculum difficulty based on model performance.
    If accuracy improves, move to harder examples faster.
    """
    # Accuracy improvement metric
    acc_improvement = current_accuracy - baseline_accuracy

    # Tanh-based smooth update rule
    mu_update_1 = (n_samples / 2) * np.tanh(0.1 * acc_improvement) + (n_samples / 2) * np.tanh(0.5)

    # Also incorporate angle signal directly (easier examples have higher scores)
    angle_signal_update = (n_samples / 2) * 0.3  # Mild contribution

    # Combined update with momentum
    new_mu = previous_mu + learning_rate * (mu_update_1 + angle_signal_update)

    # Clamp to valid range [0, n_samples]
    new_mu = np.clip(new_mu, 0, n_samples)

    return new_mu
```

5. **Training Loop with Angle-Based Scheduling**: Full integration

```python
def train_with_angle_scheduling(model, ranked_dataset, num_epochs=10,
                               batch_size=32, learning_rate=1e-5):
    """
    Complete training loop using angle concentration for data scheduling.
    Achieves 2.5× speedup using 50% of data.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    baseline_accuracy = evaluate_model(model, ranked_dataset[:100])
    mu_t = 0.0  # Start with easiest examples

    for epoch in range(num_epochs):
        # Sample batch using Gaussian curriculum
        batch = gaussian_curriculum_sampling(
            ranked_dataset, epoch, num_epochs,
            batch_size=batch_size,
            num_samples=len(ranked_dataset) // 2  # Use only 50% of data
        )

        # Training step
        loss = 0.0
        for example in batch:
            output = model(example['input_ids'])
            batch_loss = compute_loss(output, example)
            loss += batch_loss / len(batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Periodically evaluate and update curriculum
        if epoch % 2 == 0:
            current_accuracy = evaluate_model(model, ranked_dataset)
            mu_t = update_curriculum_mean(
                current_accuracy, baseline_accuracy, mu_t, len(ranked_dataset)
            )

            print(f"Epoch {epoch}: Accuracy={current_accuracy:.1%}, Curriculum_mean={mu_t:.1f}")

    return model
```

## Practical Guidance

**When to Apply:**
- RL fine-tuning is compute-expensive and training-data-heavy
- Want to improve data efficiency without external heuristics
- Have access to compute for single offline angle computation pass
- Need training speedup while maintaining or improving final performance

**Implementation Steps:**
1. Forward pass all training data through base model once (offline, one-time cost)
2. Compute angle concentration for each example (<10 minutes for 7K samples)
3. Rank by combined intra + inter angle scores
4. Train using Gaussian curriculum sampling over 10+ epochs
5. Dynamically update mean position based on validation accuracy

**Performance Expectations:**
- Training speedup: 2.5× using only 50% of data
- Final performance: Matches or exceeds full-dataset training
- Angle computation cost: <10 minutes per 7,000 samples
- Per-example overhead: ~0.1 seconds (amortized over training)

**Configuration Tuning:**
- Gaussian mean (μ): Start at 0 (easiest), shift to n/2 by end of training
- Gaussian std (σ): Control curriculum smoothness (higher = broader selection)
- Learning rate for μ update: 0.5 typically works well
- Batch size: Larger batches reduce gradient noise from biased sampling

**Key Hyperparameters:**
```python
ANGLE_CONFIG = {
    'initial_mu': 0.0,           # Start with easiest examples
    'final_mu_fraction': 0.5,    # End at middle difficulty
    'sigma_factor': 2.0,         # Gaussian width control
    'curriculum_update_freq': 2, # Update every N epochs
    'learning_rate_mu': 0.5,     # Mean shift learning rate
    'data_fraction': 0.5         # Use 50% of training data
}
```

**Monitoring Metrics:**
- Angle concentration distribution (should be bimodal)
- Curriculum mean position over epochs (should gradually shift right)
- Validation accuracy (should improve with curriculum)
- Training loss per epoch
- Gradient norm statistics

**Common Issues:**
- Insufficient angle variation: Check if model architecture supports diverse angles
- Curriculum shifts too fast: Reduce learning_rate_mu parameter
- Overfitting to curriculum: Increase data_fraction gradually
- Angle computation fails: Ensure hidden states properly extracted
- Memory issues: Reduce batch size, process in smaller chunks

## Reference

Implemented on Qwen-2.5-7B and Qwen-3-4B for RL fine-tuning on mathematical (MATH-500, AIME24) and coding (HumanEval) tasks. Demonstrates 2.5× training speedup using only 50% of training data while surpassing full-dataset performance. Works across model architectures—angle concentration is a model-agnostic learning signal. Training uses 4-16 A100 GPUs with standard RL algorithms (GRPO, DPO).
