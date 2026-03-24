---
name: tldr-thinking-reweighting
title: "TL;DR: Too Long, Do Re-weighting for Efficient LLM Reasoning Compression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.02678"
keywords: [reasoning-compression, token-efficiency, data-reweighting, chain-of-thought, system1-system2]
description: "Compress reasoning models by dynamically re-weighting short-CoT (System-1) and long-CoT (System-2) training data, achieving 40% token reduction while maintaining accuracy."
---

# TL;DR: Too Long, Do Re-weighting for Efficient LLM Reasoning Compression

## Core Concept

TL;DR addresses the efficiency problem in reasoning models: long chain-of-thought generates excessive tokens on simple problems that don't need detailed reasoning, while short-CoT lacks precision on complex problems. Rather than static data mixing, the framework dynamically re-weights System-1 data (concise solutions on easy problems) and System-2 data (detailed reasoning on hard problems) during training. The key insight: short-CoT examples generalize across difficulty levels, while long-CoT from hard problems preserves performance better than from easy ones. This iterative rebalancing achieves ~40% token reduction while maintaining reasoning accuracy without manual data construction overhead.

Results show that with proper re-weighting, the combination of short and long reasoning examples creates models that are both efficient and accurate.

## Architecture Overview

- **System-1/System-2 Paradigm**: Split training data by reasoning intensity (short vs. long CoT)
- **Dynamic Re-weighting**: Exponential update of mixture weights based on benefit metrics
- **Boundary Estimation**: Establish performance baselines for efficiency and accuracy targets
- **Generalization Discovery**: Short-CoT works across problem difficulties; long-CoT from hard problems preferable
- **Minimal Manual Work**: Automatic re-weighting requires no additional data construction

## Implementation

1. **Data Preparation and Categorization**: Organize training data by reasoning type

```python
def prepare_mixed_reasoning_data(dataset):
    """
    Split dataset into System-1 (short) and System-2 (long) reasoning.
    Easy problems: use short CoT
    Hard problems: use long CoT
    """
    short_cot_data = []
    long_cot_data = []

    for example in dataset:
        # Determine difficulty: problem-solving accuracy metric
        difficulty = estimate_problem_difficulty(example)

        # Generate short CoT: direct solution path
        short_reasoning = generate_short_cot(example, max_tokens=100)
        short_solution = extract_final_answer(short_reasoning)

        # Generate long CoT: detailed exploration
        long_reasoning = generate_long_cot(example, max_tokens=1000)
        long_solution = extract_final_answer(long_reasoning)

        if is_correct(short_solution, example['answer']):
            # Short CoT works → System-1 data
            short_cot_data.append({
                'question': example['question'],
                'reasoning': short_reasoning,
                'answer': short_solution,
                'difficulty': difficulty
            })

        if is_correct(long_solution, example['answer']):
            # Long CoT needed for hard problems → System-2 data
            long_cot_data.append({
                'question': example['question'],
                'reasoning': long_reasoning,
                'answer': long_solution,
                'difficulty': difficulty
            })

    return short_cot_data, long_cot_data
```

2. **Boundary Estimation**: Establish efficiency and accuracy targets

```python
def estimate_training_boundaries(short_model, long_model, val_set):
    """
    Measure baseline performance for short and long CoT approaches.
    These establish efficiency and accuracy bounds for re-weighting.
    """
    # Baseline 1: Short CoT efficiency
    short_accs, short_tokens = [], []
    for example in val_set:
        output = short_model.generate(example['question'], max_tokens=200)
        acc = is_correct(output, example['answer'])
        tokens = len(output.split())
        short_accs.append(acc)
        short_tokens.append(tokens)

    efficiency_baseline = np.mean(short_tokens)  # Average tokens for short CoT
    accuracy_floor = np.mean(short_accs)         # Minimum acceptable accuracy

    # Baseline 2: Long CoT accuracy
    long_accs = []
    for example in val_set:
        output = long_model.generate(example['question'], max_tokens=2000)
        acc = is_correct(output, example['answer'])
        long_accs.append(acc)

    accuracy_ceiling = np.mean(long_accs)  # Maximum achievable accuracy

    return {
        'efficiency_baseline': efficiency_baseline,
        'accuracy_floor': accuracy_floor,
        'accuracy_ceiling': accuracy_ceiling
    }
```

3. **Dynamic Re-weighting Algorithm**: Iteratively adjust mixture proportions

```python
def dynamic_reweight_training(short_cot_data, long_cot_data, model, boundaries,
                              num_epochs=2000, validation_interval=32):
    """
    Dynamically adjust mixture of short and long CoT during training.
    Re-weights based on benefit metrics from both data types.
    """
    num_short = len(short_cot_data)
    num_long = len(long_cot_data)

    # Initial mixture weights: equal
    alpha = np.array([1.0, 1.0])  # [weight_short, weight_long]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        # Sample from both datasets according to current weights
        short_samples = np.random.choice(num_short, size=int(32 * alpha[0]), replace=True)
        long_samples = np.random.choice(num_long, size=int(32 * alpha[1]), replace=True)

        # Training step on mixed batch
        loss = 0.0
        for idx in short_samples:
            example = short_cot_data[idx]
            example_loss = model.compute_loss(example)
            loss += example_loss

        for idx in long_samples:
            example = long_cot_data[idx]
            example_loss = model.compute_loss(example)
            loss += example_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Periodically re-weight
        if epoch % validation_interval == 0:
            # Evaluate on validation set
            short_acc = evaluate(model, short_cot_data[:100], boundaries['accuracy_floor'])
            long_acc = evaluate(model, long_cot_data[:100], boundaries['accuracy_ceiling'])

            # Compute benefits
            short_benefit = short_acc - boundaries['accuracy_floor']  # Compression gain
            long_benefit = long_acc - boundaries['accuracy_floor']    # Accuracy gain

            # Update weights using exponential smoothing
            # Prefer data that provides the most benefit
            c = 0.1  # Smoothing parameter
            u = np.array([short_benefit, long_benefit])

            alpha_t = (1 - c) * alpha / np.sum(alpha) + c * u
            alpha = np.array([max(1, x) for x in alpha_t])  # Ensure positive weights

    return model
```

4. **Validation and Checkpoint Selection**: Choose best trade-off point

```python
def select_best_checkpoint(trained_models, boundaries, preference='balanced'):
    """
    Select checkpoint balancing accuracy and compression.
    preference: 'speed' (emphasize tokens), 'quality' (emphasize accuracy)
    """
    best_model = None
    best_score = -float('inf')

    for model, epoch in trained_models:
        # Evaluate on test set
        accuracy = eval_accuracy(model)
        avg_tokens = eval_token_length(model)

        # Normalize to 0-1 range
        accuracy_normalized = accuracy / boundaries['accuracy_ceiling']
        compression_ratio = avg_tokens / boundaries['efficiency_baseline']

        if preference == 'balanced':
            # Equal weight to both metrics
            score = 0.5 * accuracy_normalized + 0.5 * (1 - compression_ratio)
        elif preference == 'speed':
            # Prioritize compression
            score = 0.3 * accuracy_normalized + 0.7 * (1 - compression_ratio)
        else:  # 'quality'
            # Prioritize accuracy within 30% of baseline
            if accuracy < boundaries['accuracy_ceiling'] * 0.7:
                score = -float('inf')  # Reject low accuracy
            else:
                score = accuracy_normalized

        if score > best_score:
            best_score = score
            best_model = model

    return best_model
```

5. **Training Configuration**: Recommended hyperparameters

```python
# Configuration for TL;DR training
TLDR_CONFIG = {
    'base_model': 'deepseek-r1-distill-qwen-7b',
    'num_epochs': 2000,
    'validation_interval': 32,
    'initial_alpha': [1.0, 1.0],  # Equal weights initially
    'smoothing_param_c': 0.1,
    'batch_size': 32,
    'learning_rate': 1e-5,
    'accuracy_target': 0.95,  # 95% of full reasoning model
    'compression_target': 0.60,  # 60% of original token count
    'benchmarks': ['MATH-500', 'AIME24', 'GPQA', 'LiveCodeBench']
}
```

## Practical Guidance

**When to Apply:**
- Need to compress reasoning models while maintaining accuracy
- Have mixed-difficulty problem datasets
- Want to reduce inference tokens by 30-40% with minimal accuracy loss
- Don't want to manually construct optimal data ratios

**Setup Requirements:**
- Base reasoning model (DeepSeek-R1-Distill, similar distilled model)
- Mixed-difficulty problem dataset
- Validation set with diverse problem types
- 2-4 GPU training with gradient accumulation

**Data Preparation:**
1. Separate dataset by problem difficulty (use accuracy-based metric)
2. Generate short CoT for all problems (direct solution)
3. Generate long CoT for problems where short CoT fails
4. Validate success rates: short works on ~60%, long works on ~90%

**Performance Targets:**
- Token reduction: 30-40% of original long-CoT model
- Accuracy maintenance: ≥95% of full reasoning model performance
- Fast-thinking activation: 30-40% of easy problems use short CoT only
- Inference speedup: 20-30% wall-clock improvement

**Key Hyperparameters:**
- Smoothing parameter (c): Controls re-weighting aggressiveness
  - c=0.05: Conservative, gradual changes
  - c=0.1: Balanced (recommended)
  - c=0.2: Aggressive, rapid adjustments
- Initial alpha values: Start at [1.0, 1.0] for neutral initialization
- Validation interval: Every 32 steps typically works well

**Monitoring During Training:**
- Track short and long CoT accuracies separately
- Plot weight evolution over epochs (should converge)
- Monitor average token length per batch
- Validate on held-out test set periodically

**Common Pitfalls:**
- Smoothing parameter too high: Oscillating weights, unstable training
- Insufficient validation data: Noisy weight updates
- Initial data imbalance: Pre-balance short and long datasets
- Target accuracy too aggressive: Set to 0.90-0.95, not 1.0

## Reference

Implemented on DeepSeek-R1-Distill-Qwen (7B/14B variants), evaluated on MATH-500, AIME24, GPQA Diamond, and MMLU-Pro benchmarks. Training typically 2,000 steps with validation every 32 steps. Achieves ~40% token reduction while maintaining 95%+ of baseline reasoning accuracy. No additional data construction required beyond standard dataset splits.
