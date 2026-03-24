---
name: coba-rl-budget-allocation
title: "CoBA-RL: Capability-Oriented Budget Allocation for Reinforcement Learning in LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03048"
keywords: [Budget Allocation, Reinforcement Learning, Curriculum Learning, Sample Selection, Training Efficiency]
description: "Dynamically allocate training budget across samples using a capability-oriented value function that measures per-sample training importance based on model capability evolution. Reduces training time via greedy heap-based allocation optimizing exploration-exploitation tradeoff."
---

# CoBA-RL: Capability-Aware Budget Allocation

Standard RL training allocates compute uniformly across all training samples, but samples have asymmetric value as the model evolves. Early training requires diverse exploration; later stages benefit from focused improvement on difficult problems. CoBA-RL automatically balances this by dynamically allocating computational budget—training iterations, gradient updates, or rollouts—to high-value samples.

The approach uses a capability-oriented value function that scores samples based on the model's current capability level, then uses efficient greedy allocation to assign budget where it matters most. This trades per-sample training cost for dramatically faster overall convergence.

## Core Concept

CoBA-RL decomposes budget allocation into two components:

1. **Capability-Oriented Value Function**: Maps (sample, model_capability) to training importance using a Beta distribution. Samples marked as failing become high-priority when the model's global failure rate is high (exploration phase); successful samples gain priority when failure rate drops (exploitation phase).

2. **Greedy Heap-Based Allocation**: Instead of solving expensive dynamic programming (O(M·B·range)), use a max-heap to iteratively select the B highest-value samples in O(B·log M) time.

This separates the conceptual problem (what should we prioritize?) from the computational method (how do we find it efficiently?).

## Architecture Overview

- **Capability Monitor**: Tracks global task failure rate across the training set
- **Value Function**: Beta distribution computing per-sample importance given model capability
- **Heap Allocator**: Maintains priority queue of samples, returns top-B samples efficiently
- **Training Loop**: Standard GRPO/PPO with variable iteration counts per sample
- **Feedback Cycle**: Updated failure rate from each epoch refines next allocation

## Implementation

### Step 1: Define the Capability-Oriented Value Function

Create a function that scores sample importance based on current model capability.

```python
# Capability-oriented value function
import numpy as np
from scipy.stats import beta

class CapabilityValueFunction:
    def __init__(self, alpha=2.0, beta_param=5.0):
        """
        Value function using Beta distribution.
        alpha: shape parameter for failure weight
        beta_param: shape parameter for capability weighting
        """
        self.alpha = alpha
        self.beta_param = beta_param

    def compute_value(self, sample_failed: bool,
                     global_failure_rate: float) -> float:
        """
        Compute training value for a sample given current capability.

        Args:
            sample_failed: Whether this sample failed in last eval
            global_failure_rate: Fraction of all samples failing

        Returns:
            Importance weight [0, 1]
        """
        # During exploration (high failure rate): prioritize failing samples
        # During exploitation (low failure rate): prioritize successful samples
        if sample_failed:
            # Failing samples important during exploration
            return beta.pdf(global_failure_rate, self.alpha, self.beta_param)
        else:
            # Successful samples important during exploitation
            return 1.0 - beta.pdf(global_failure_rate, self.alpha, self.beta_param)

class SampleValueQueue:
    def __init__(self, samples: List[dict], value_fn: CapabilityValueFunction):
        self.samples = samples
        self.value_fn = value_fn
        self.values = [0.0] * len(samples)

    def update_values(self, global_failure_rate: float, last_results: dict):
        """Update value for each sample based on latest capability."""
        for i, sample in enumerate(self.samples):
            sample_id = sample['id']
            sample_failed = last_results.get(sample_id, True)
            self.values[i] = self.value_fn.compute_value(
                sample_failed,
                global_failure_rate
            )

    def get_topk(self, k: int) -> List[dict]:
        """Return k highest-value samples using heap."""
        # Use negative values for max-heap
        top_indices = np.argsort(-np.array(self.values))[:k]
        return [self.samples[i] for i in top_indices]
```

### Step 2: Implement Greedy Budget Allocation

Create a heap-based allocator that assigns training iterations to samples.

```python
# Greedy budget allocation
class GreedyBudgetAllocator:
    def __init__(self, total_budget: int, min_budget: int = 1,
                 max_budget: int = 128):
        """
        Allocate total_budget training iterations across samples.

        Args:
            total_budget: Total iterations to allocate (sum of B values)
            min_budget: Minimum iterations per sample
            max_budget: Maximum iterations per sample
        """
        self.total_budget = total_budget
        self.min_budget = min_budget
        self.max_budget = max_budget

    def allocate(self, sample_values: np.ndarray) -> np.ndarray:
        """
        Greedy allocation: repeatedly assign budget to highest-value sample.
        """
        allocation = np.ones(len(sample_values), dtype=int) * self.min_budget
        remaining = self.total_budget - self.min_budget * len(sample_values)

        # Normalize values to weights
        weights = sample_values / (sample_values.sum() + 1e-8)

        # Greedily allocate remaining budget
        while remaining > 0:
            # Find highest-value sample below max_budget
            candidates = [(i, weights[i]) for i in range(len(sample_values))
                         if allocation[i] < self.max_budget]
            if not candidates:
                break

            # Assign one iteration to highest-value candidate
            best_idx = max(candidates, key=lambda x: x[1])[0]
            allocation[best_idx] += 1
            remaining -= 1

        return allocation
```

### Step 3: Integrate with GRPO Training Loop

Modify standard GRPO training to use variable budget per sample.

```python
# GRPO training with budget allocation
def train_with_budget_allocation(
    model,
    dataset: List[dict],
    value_fn: CapabilityValueFunction,
    allocator: GreedyBudgetAllocator,
    num_epochs: int = 10,
    batch_size: int = 16
):
    """Train model using CoBA-RL budget allocation."""

    # Initialize queue and last results
    queue = SampleValueQueue(dataset, value_fn)
    last_results = {sample['id']: False for sample in dataset}

    for epoch in range(num_epochs):
        # Compute global failure rate
        failures = sum(1 for v in last_results.values() if v)
        global_failure_rate = failures / len(last_results)

        # Update values based on current capability
        queue.update_values(global_failure_rate, last_results)

        # Allocate budget: compute iterations per sample
        budget_allocation = allocator.allocate(np.array(queue.values))

        # Train with allocated budget
        new_results = {}
        for sample, iterations in zip(dataset, budget_allocation):
            # Run GRPO for 'iterations' steps on this sample
            for _ in range(iterations):
                # Standard GRPO step
                logits = model(sample['prompt'])
                loss = compute_grpo_loss(logits, sample['target'])
                loss.backward()
                optimizer.step()

            # Evaluate sample
            output = model.generate(sample['prompt'])
            new_results[sample['id']] = evaluate(output, sample['target'])

        # Update results for next epoch
        last_results.update(new_results)

        # Log metrics
        success_rate = sum(new_results.values()) / len(new_results)
        print(f"Epoch {epoch}: {success_rate:.2%} success, "
              f"failure_rate={global_failure_rate:.2%}")

    return model
```

### Step 4: Tune Allocation Parameters

Use empirical observation to optimize allocation settings.

```python
# Parameter tuning
def tune_allocation_params(
    model,
    dataset: List[dict],
    total_iterations: int
):
    """Tune allocation parameters using quick validation."""

    best_config = None
    best_loss = float('inf')

    # Grid search over key parameters
    for alpha in [1.0, 2.0, 3.0]:
        for beta_val in [3.0, 5.0, 8.0]:
            for max_budget in [64, 128, 256]:
                value_fn = CapabilityValueFunction(alpha=alpha, beta_param=beta_val)
                allocator = GreedyBudgetAllocator(
                    total_budget=total_iterations,
                    max_budget=max_budget
                )

                # Quick validation on subset
                val_loss = validate_config(
                    model,
                    dataset[:len(dataset)//4],  # 25% subset
                    value_fn,
                    allocator
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_config = (alpha, beta_val, max_budget)

    print(f"Best config: alpha={best_config[0]}, beta={best_config[1]}, "
          f"max_budget={best_config[2]}")
    return best_config
```

## Practical Guidance

**When to use CoBA-RL:**
- Long training runs (100+ epochs) where convergence speed matters
- Heterogeneous datasets with variable sample difficulty
- RL-based fine-tuning where computational budget is constrained
- Multi-stage learning where capability evolves significantly

**When not to use:**
- Short training runs (<20 epochs) where overhead dominates
- Datasets with uniform sample difficulty
- Batch-based methods that require fixed sample counts
- Real-time inference where allocation overhead is problematic

**Common Pitfalls:**
- Value function miscalibration: Alpha/beta parameters must match your data distribution
- Allocation imbalance: Max_budget too low starves difficult samples; too high wastes iterations
- Stale capability metrics: Update global failure rate per epoch, not per-batch
- Sample shuffling: Allocator assumes samples are fixed; shuffle before each epoch

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning Strategy |
|-----------|-------|-----------------|
| alpha (Beta shape) | 1.0-5.0 | Higher = more exploration; tune on early-stage convergence |
| beta (Beta shape) | 3.0-10.0 | Lower = sharper phase transition; tune on mid-stage diversity |
| max_budget | 64-256 | Set to 2-4× dataset size; higher = more focus on hard samples |
| min_budget | 1-4 | Set to 1 for exploration; increase if underfitting on easy samples |
| Budget total | dataset_size × 5-20 | Lower bound for convergence; empirically determine via ablation |

## Reference

See the full paper at: https://arxiv.org/abs/2602.03048

Key results: 928× speedup over dynamic programming; demonstrated on GRPO training loops; open-source code available. Integrates seamlessly into existing LLM post-training pipelines.
