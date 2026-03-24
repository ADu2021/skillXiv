---
name: sample-more-think-less-gfpo
title: "Sample More Think Less: Group Filtered Policy Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.09726
keywords: [policy-optimization, rl-training, token-efficiency, length-penalization, inference-optimization]
description: "Group Filtered Policy Optimization (GFPO) reduces inference-time computation by sampling larger groups during training and filtering responses based on length and token efficiency to teach models efficient reasoning."
---

# Sample More Think Less: Group Filtered Policy Optimization

## Core Concept

Group Filtered Policy Optimization (GFPO) addresses a critical problem in RL-trained language models: length inflation, where models generate unnecessarily verbose outputs during inference. Rather than optimizing purely for correctness, GFPO trades training-time computation for inference efficiency by sampling multiple candidate responses and filtering them based on efficiency metrics.

The key insight is that larger sample groups at training time enable models to learn more efficient reasoning patterns, reducing computation needed at test time.

## Architecture Overview

- **Group Sampling**: Sample multiple candidate responses (groups) per problem during RL training
- **Dual Filtering Criteria**: Evaluate each response on two metrics: (1) output length, (2) reward-per-token efficiency ratio
- **Efficiency-Aware Selection**: Train only on responses that achieve good performance without excessive verbosity
- **Adaptive Variant**: Dynamically allocate more samples to harder problems for targeted efficiency gains
- **Inference Speedup**: Resulting models require fewer tokens at test time while maintaining accuracy

## Implementation Steps

### 1. Prepare RL Training Setup

Start with a base language model and define your reward function (e.g., correctness on reasoning tasks). Set up PPO or similar RL algorithm infrastructure.

```python
# Hugging Face Transformers + TRL style setup
from trl import PPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

def compute_reward(response, reference_answer):
    """Reward function: 1.0 for correct, 0.0 for incorrect"""
    is_correct = check_correctness(response, reference_answer)
    return 1.0 if is_correct else 0.0
```

### 2. Sample Groups and Compute Metrics

For each training problem, generate multiple candidate responses. Compute both correctness reward and efficiency metrics for each.

```python
# Group sampling during training
def sample_group(prompt, problem_id, group_size=8):
    """Sample multiple responses per problem"""
    responses = []
    metrics = []

    for _ in range(group_size):
        response = base_model.generate(
            prompt,
            max_length=2048,
            temperature=0.7,
            do_sample=True
        )
        response_text = tokenizer.decode(response)

        # Compute metrics
        correctness_reward = compute_reward(response_text, gold_answers[problem_id])
        num_tokens = len(response)
        efficiency = correctness_reward / (num_tokens + 1)  # Reward-per-token

        responses.append(response_text)
        metrics.append({
            'correctness': correctness_reward,
            'length': num_tokens,
            'efficiency': efficiency,
            'response': response_text
        })

    return responses, metrics
```

### 3. Apply Dual Filtering

Filter the group using both length and efficiency criteria. Select responses that are both correct and relatively concise.

```python
def filter_responses(metrics, length_threshold=None, efficiency_threshold=0.5):
    """
    Filter responses based on two metrics:
    - Must exceed efficiency threshold (reward per token)
    - Preferentially selects shorter responses among valid ones
    """
    # First pass: filter by efficiency (correctness-weighted by length)
    efficient_responses = [
        m for m in metrics
        if m['efficiency'] > efficiency_threshold
    ]

    if not efficient_responses:
        # Fallback: take highest efficiency even if below threshold
        efficient_responses = sorted(metrics, key=lambda x: x['efficiency'],
                                    reverse=True)[:1]

    # Second pass: among efficient ones, prefer shorter ones
    if length_threshold is not None:
        filtered = [m for m in efficient_responses if m['length'] <= length_threshold]
        if filtered:
            return filtered

    # Return top efficiency responses
    return sorted(efficient_responses, key=lambda x: x['efficiency'],
                  reverse=True)[:2]
```

### 4. Compute Length Penalty in Training Objective

Integrate length awareness into the RL loss. Add a penalty term that discourages verbose responses.

```python
def compute_training_loss(response, reward, num_tokens, length_weight=0.1):
    """
    Combined RL objective with efficiency component:
    L = -reward + length_weight * log(num_tokens)
    """
    base_reward = reward
    length_penalty = length_weight * torch.log(torch.tensor(num_tokens + 1))

    # Negative because we're optimizing (policy gradient uses -log_prob * advantage)
    advantage = base_reward - length_penalty
    return advantage
```

### 5. Update Policy with Efficient Samples

Use the filtered, efficient responses to update the policy. Only train on responses that pass the efficiency filter.

```python
# PPO training loop with GFPO filtering
for batch_idx, problem_batch in enumerate(training_problems):
    group_responses = []
    group_metrics = []

    for problem_id, prompt in problem_batch:
        responses, metrics = sample_group(prompt, problem_id, group_size=8)
        filtered = filter_responses(metrics, efficiency_threshold=0.6)

        group_responses.extend(filtered)
        group_metrics.extend([m['response'] for m in filtered])

    # Train PPO on filtered responses
    ppo_trainer.train({
        'prompts': [p for p, _ in problem_batch],
        'responses': group_responses,
        'rewards': [m['correctness'] for m in group_metrics]
    })
```

### 6. Implement Adaptive Difficulty (Optional Enhancement)

Dynamically increase sampling for harder problems where the efficiency filter rejects more candidates.

```python
def adaptive_group_size(problem_id, difficulty_scores):
    """
    Allocate more samples to harder problems
    difficulty_scores[problem_id] should be between 0 and 1
    """
    base_group_size = 8
    difficulty = difficulty_scores.get(problem_id, 0.5)

    # Harder problems get larger groups
    adaptive_size = int(base_group_size * (1 + difficulty))
    return min(adaptive_size, 32)  # Cap at 32

# In training loop:
for problem_id, prompt in problem_batch:
    group_size = adaptive_group_size(problem_id, problem_difficulties)
    responses, metrics = sample_group(prompt, problem_id, group_size=group_size)
    filtered = filter_responses(metrics)
```

## Practical Guidance

### Hyperparameters & Configuration

- **Group Size**: 8-16 for standard training, 16-32 for adaptive difficulty (more samples = more training time)
- **Length Threshold**: Set to 70-80th percentile of gold-answer lengths to avoid over-penalizing longer-but-correct responses
- **Efficiency Threshold**: 0.5-0.8 depending on reward scale; higher values enforce stricter efficiency
- **Length Weight**: 0.05-0.2 in combined loss; balance between correctness and conciseness
- **Reward-per-Token Ratio**: Use (correctness_reward + 0.1) / (num_tokens + 1) to avoid division by zero

### When to Use GFPO

- Your RL-trained model exhibits length inflation or verbosity
- You have access to multiple candidate generations during training
- Inference efficiency is important (faster response time, lower token consumption)
- You can define clear correctness signals that can be computed at training time
- You have computational budget for sampling multiple candidates per problem

### When NOT to Use GFPO

- Your task requires generating long, detailed explanations (essay writing, thorough analysis)
- You cannot access reward signals during training (black-box evaluation only)
- Your base model is already concise and doesn't exhibit length inflation
- Computational cost of multi-sample training is prohibitive
- Correctness and length are fundamentally orthogonal in your task

### Common Pitfalls

1. **Too Aggressive Length Penalization**: Over-penalizing length causes models to underblow on complex problems. Calibrate against gold-answer lengths.
2. **Insufficient Group Sizes**: Sampling only 2-3 candidates per problem reduces diversity. Use at least 8 candidates.
3. **Stale Difficulty Scores**: Adaptive difficulty becomes ineffective if problem difficulties aren't updated. Recompute after policy updates.
4. **Ignoring Correctness**: Over-optimizing efficiency at expense of accuracy defeats the purpose. Ensure efficiency filter preserves correct responses.
5. **No Inference Evaluation**: Train with efficiency but validate on actual test-time efficiency (actual token counts), not just training metrics.

## Reference

Sample More Think Less (2508.09726): https://arxiv.org/abs/2508.09726

GFPO reduces inference-time token consumption by 46-71% on reasoning tasks while maintaining accuracy by training on length-filtered, efficiency-optimized responses sampled in larger groups.
