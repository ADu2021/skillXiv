---
name: vcrl-variance-curriculum-rl
title: "VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.19803"
keywords: [curriculum learning, reinforcement learning, reward variance, LLM training, sample difficulty, policy gradient]
description: "VCRL improves LLM mathematical reasoning by dynamically adjusting training sample difficulty based on group reward variance. Uses variance-based sampling with memory replay to focus on moderately-difficult samples where models succeed ~50% of the time, achieving 4.67-point improvement over GRPO on 8B models."
---

# Variance-based Curriculum RL: Focus training on samples with optimal difficulty

## The Problem with Standard RL Training

Rollout-based reinforcement learning methods like GRPO have a fundamental issue: they treat all samples equally during training, regardless of difficulty. The model might waste effort on trivial problems it already solves perfectly, or struggle without improvement on impossibly hard ones. Neither scenario drives learning forward. The sweet spot for learning exists where the model succeeds about 50% of the time—where uncertainty is maximal and every success or failure provides genuine signal.

The challenge is identifying which samples occupy that sweet spot without explicit labels. VCRL solves this by observing an implicit difficulty signal already present in training data: the variance of rewards across multiple rollouts of the same prompt.

## Core Concept

VCRL recognizes a key insight from statistical learning: when you generate multiple rollout attempts on a single prompt, the distribution of successes and failures reveals the prompt's difficulty for the current model. Easy prompts succeed almost every time (low variance). Hard prompts fail almost every time (low variance). But moderately difficult prompts—where success is about 50/50—show high variance across rollouts.

The algorithm leverages this variance signal in two mechanisms. First, it continuously samples rewards from rollout groups and calculates normalized variance scores for each prompt. Prompts with high variance scores are deemed valuable for training. Second, it maintains a replay buffer (memory bank) of high-variance prompts and substitutes them into training batches whenever low-variance samples appear, ensuring the model always learns from optimally challenging examples.

## Architecture Overview

- **Variance Calculator**: Computes normalized group reward variance p = σ²/σ²_max for each training sample based on k successes out of G rollouts
- **Threshold Filter**: Compares variance score p against threshold κ to identify low-value samples for replacement
- **Memory Bank**: Priority queue storing high-variance samples with their variance scores, keyed by prompts
- **Batch Sampler**: During training, replaces low-variance samples in batches with highest-variance samples from memory
- **Policy Optimizer**: Standard AdamW optimization over selected batch, unchanged from baseline

## Implementation

### Step 1: Initialize Memory Bank and Set Hyperparameters

The memory bank is a priority queue that accumulates high-value samples during training. Initialize it at the start of training and configure the key hyperparameters that control curriculum behavior.

```python
from collections import defaultdict
from heapq import heappush, heappop
import numpy as np

class VCRLMemoryBank:
    """Priority queue for storing high-variance training samples."""

    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.samples = []  # heap of (variance_score, prompt_text, rollouts)
        self.prompt_to_score = {}  # track best score for each unique prompt

    def add(self, prompt, variance_score, rollouts):
        """Add sample to memory bank if variance is high enough."""
        if variance_score > self.prompt_to_score.get(prompt, 0):
            self.prompt_to_score[prompt] = variance_score
            heappush(self.samples,
                    (variance_score, prompt, rollouts))

            # Keep memory bounded
            if len(self.samples) > self.max_size:
                _, old_prompt, _ = heappop(self.samples)
                if old_prompt in self.prompt_to_score:
                    del self.prompt_to_score[old_prompt]

    def sample(self, k=1):
        """Sample k highest-variance samples from memory."""
        if not self.samples:
            return []
        # Return top-k without removing (samples stay in memory)
        return sorted(self.samples, reverse=True)[:k]

# Initialize memory bank
memory_bank = VCRLMemoryBank(max_size=5000)

# VCRL hyperparameters
VARIANCE_THRESHOLD = 0.3  # κ - samples below this replaced
GROUP_SIZE = 8            # G - number of rollouts per prompt
BATCH_SIZE = 32           # training batch size
MEMORY_SAMPLE_RATE = 0.5  # fraction of low-variance samples to replace
```

### Step 2: Calculate Normalized Group Reward Variance

During rollout collection, compute variance scores for each prompt based on success/failure patterns. For binary rewards (success=1, failure=0), variance depends on how many of G rollouts succeeded.

```python
def calculate_group_variance(num_successes, group_size):
    """
    Calculate unbiased group reward variance using binomial statistics.

    When k out of G rollouts succeed, variance σ² = k(G-k)/(G(G-1))
    with theoretical maximum when k = G/2 of σ²_max = G/(4(G-1))
    """
    if group_size <= 1:
        return 0.0

    k = num_successes
    g = group_size

    # Unbiased variance estimator
    variance = (k * (g - k)) / (g * (g - 1))

    # Theoretical maximum (50/50 split)
    max_variance = g / (4 * (g - 1))

    # Normalized score: 0 to 1
    normalized_variance = variance / max_variance if max_variance > 0 else 0.0

    return min(normalized_variance, 1.0)

def process_rollout_group(prompt, rollout_results, group_size=8):
    """
    Process a group of rollout results for one prompt.

    rollout_results: list of (success, output) tuples
    Returns: (variance_score, sample_data)
    """
    num_successes = sum(1 for success, _ in rollout_results if success)
    variance_score = calculate_group_variance(num_successes, group_size)

    sample_data = {
        'prompt': prompt,
        'variance_score': variance_score,
        'num_successes': num_successes,
        'rollouts': rollout_results,
        'difficulty_level': 'easy' if num_successes > 6 else
                           'hard' if num_successes < 2 else 'moderate'
    }

    return variance_score, sample_data

# Example usage during data collection
prompt = "Solve: What is 2^100 mod 13?"
rollout_results = [
    (True, "Using Fermat's little theorem..."),
    (False, "Incorrect calculation"),
    (True, "Correct approach and answer"),
    (False, "Wrong method"),
    (True, "Right answer"),
    (True, "Correct reasoning"),
    (False, "Sign error"),
    (True, "Verified solution")
]

variance, sample = process_rollout_group(prompt, rollout_results, group_size=8)
memory_bank.add(prompt, variance, rollout_results)
```

### Step 3: Build Training Batch with Curriculum Replacement

During training iteration, construct the batch by filtering out low-variance samples and replacing them with high-variance ones from the memory bank. This ensures every training step focuses on moderately-difficult problems.

```python
def build_curriculum_batch(dataset_batch, memory_bank,
                          variance_threshold=0.3,
                          replacement_rate=0.5):
    """
    Filter low-variance samples and replace with memory bank samples.

    This is the core curriculum mechanism: dynamically adjust batch composition
    to focus on samples with learning potential (high variance = moderate difficulty).
    """
    curriculum_batch = []
    samples_to_replace = []

    # First pass: keep high-variance samples, mark low-variance for replacement
    for sample in dataset_batch:
        variance_score = sample.get('variance_score', 0)

        if variance_score >= variance_threshold:
            # Keep this sample - it's in the learning sweet spot
            curriculum_batch.append(sample)
        else:
            # Mark for replacement - too easy or too hard
            samples_to_replace.append(sample)

    # Second pass: replace low-variance samples with high-variance from memory
    num_to_replace = int(len(samples_to_replace) * replacement_rate)

    if memory_bank.samples and num_to_replace > 0:
        memory_samples = memory_bank.sample(num_to_replace)

        for i, memory_sample in enumerate(memory_samples):
            if i < len(samples_to_replace):
                _, prompt, rollouts = memory_sample
                curriculum_batch.append({
                    'prompt': prompt,
                    'rollouts': rollouts,
                    'variance_score': memory_sample[0],
                    'source': 'memory_bank'
                })

    # If we don't have enough memory samples, use remaining original low-variance
    while len(curriculum_batch) < len(dataset_batch):
        curriculum_batch.append(samples_to_replace[
            len(curriculum_batch) - len(dataset_batch) + len(samples_to_replace)
        ])

    return curriculum_batch

# Training loop integration
for epoch in range(num_epochs):
    for dataset_batch in train_dataloader:
        # Apply curriculum filtering
        curriculum_batch = build_curriculum_batch(
            dataset_batch,
            memory_bank,
            variance_threshold=VARIANCE_THRESHOLD,
            replacement_rate=MEMORY_SAMPLE_RATE
        )

        # Standard policy gradient optimization
        logits = model(curriculum_batch)
        loss = compute_policy_loss(logits, curriculum_batch)
        loss.backward()
        optimizer.step()
```

### Step 4: Integrate with Training Loop

Wire curriculum sampling into your existing RL training framework, updating the memory bank as you collect new variance signals and using it to shape batch composition.

```python
import torch
from torch.optim import AdamW

class VCRLTrainer:
    """Curriculum RL trainer using variance-based sample selection."""

    def __init__(self, model, dataset, memory_bank, learning_rate=1e-6):
        self.model = model
        self.dataset = dataset
        self.memory_bank = memory_bank
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

    def collect_rollouts(self, prompts, num_rollouts=8):
        """Generate multiple rollouts for each prompt and compute variance."""
        all_variance_scores = []
        all_samples = []

        for prompt in prompts:
            rollout_results = []

            # Generate rollouts with model
            for _ in range(num_rollouts):
                response = self.model.generate(prompt)
                is_correct = self.evaluate_response(prompt, response)
                rollout_results.append((is_correct, response))

            # Compute variance signal
            variance_score, sample_data = process_rollout_group(
                prompt, rollout_results, num_rollouts
            )

            # Update memory with high-variance samples
            if variance_score > 0.4:  # Store top quartile
                self.memory_bank.add(prompt, variance_score, rollout_results)

            all_variance_scores.append(variance_score)
            all_samples.append(sample_data)

        return all_samples, all_variance_scores

    def train_step(self, batch_size=32):
        """Single training iteration with curriculum sampling."""
        # Sample from dataset
        dataset_batch = self.dataset.sample_batch(batch_size)

        # Enrich with variance scores from recent rollouts
        prompts = [s['prompt'] for s in dataset_batch]
        samples, variance_scores = self.collect_rollouts(prompts)

        for i, sample in enumerate(samples):
            dataset_batch[i]['variance_score'] = variance_scores[i]

        # Apply curriculum filter
        curriculum_batch = build_curriculum_batch(
            dataset_batch,
            self.memory_bank,
            variance_threshold=0.3,
            replacement_rate=0.5
        )

        # Convert to tensors and optimize
        batch_tensor = self._prepare_batch(curriculum_batch)
        logits = self.model(batch_tensor)

        # Compute policy loss with rewards
        rewards = torch.tensor([
            s['num_successes'] for s in curriculum_batch
        ], dtype=torch.float32)

        loss = -torch.mean(logits * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_response(self, prompt, response):
        """Check if model response is correct (dataset-specific)."""
        # Implement based on your task (e.g., exact match, equivalence check)
        pass

    def _prepare_batch(self, curriculum_batch):
        """Convert batch to model input format."""
        # Implement tokenization and batching
        pass

# Usage
trainer = VCRLTrainer(model, dataset, memory_bank)

for step in range(10000):
    loss = trainer.train_step(batch_size=32)
    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss:.4f}")
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Variance Threshold (κ) | 0.30-0.40 | Lower = more curriculum, higher = stricter filtering |
| Group Size (G) | 6-8 | Minimum 4, more gives better variance estimate |
| Memory Bank Size | 3000-5000 | Balance between diversity and computational cost |
| Replacement Rate | 0.3-0.5 | Fraction of low-variance samples to replace per batch |
| Learning Rate | 1×10⁻⁶ | Use constant rate with curriculum (no decay needed) |
| Batch Size | 32-64 | Standard RL batch sizes work well |

### When to Use VCRL

- Training LLMs on well-defined evaluation tasks (math, code, QA) where you can cheaply assess correctness
- Models 8B-70B with sufficient compute for multiple rollouts per sample
- Long-horizon reasoning tasks where difficulty variance is pronounced
- When baseline methods plateau despite abundant training data
- Problems with clear ground truth (not open-ended generation)

### When NOT to Use VCRL

- Open-ended generation or creative tasks without clear correctness metrics
- Extremely small models (≤1B) where rollout cost is prohibitive
- Very limited compute budgets (cannot afford multiple rollouts per sample)
- Tasks where all samples are equally difficult
- Real-time inference requirements (training is slower due to variance computation)
- Scenarios requiring diversity over optimization (preference learning where variety matters)

### Common Pitfalls

**Low variance overall**: If most prompts have variance scores near 0 or 1, your rollout group size may be too large. Reduce G to 4-6 to increase variance signal.

**Memory bank starvation**: Early in training, insufficient high-variance samples populate memory. Start with higher threshold (0.5) and gradually lower to 0.3 after first epoch.

**Convergence instability**: If loss curves are noisy, reduce batch replacement rate to 0.3 and increase memory bank size. The model needs diverse gradients.

**Skewed evaluation**: If your correctness metric is biased (e.g., penalizes unconventional but valid reasoning), variance estimates become unreliable. Verify your evaluation function first.

---

Reference: [VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2509.19803)
