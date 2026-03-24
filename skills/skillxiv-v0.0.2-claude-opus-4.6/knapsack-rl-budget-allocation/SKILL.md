---
name: knapsack-rl-budget-allocation
title: "Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.25849"
keywords: [resource-allocation, RL-training, exploration-efficiency, curriculum-learning, GRPO]
description: "Improve RL training for LLMs by dynamically allocating exploration budget (rollout count) to tasks based on their difficulty and current learning status. Solves the knapsack problem of maximizing gradient signal within fixed compute budget, increasing non-zero policy gradients by 20-40% and achieving 2-4 point performance gains."
---

# Knapsack RL: Adaptive Budget Allocation for Efficient Exploration

Standard RL training allocates uniform exploration budget: every task gets the same number of rollouts. But this is wasteful. Easy tasks consistently succeed (all rollouts produce zero gradients) while hard tasks consistently fail (all rollouts produce unhelpful noise). Neither generates useful learning signals.

Knapsack RL reframes this as an optimization problem: given a fixed compute budget, allocate rollout counts to tasks to maximize useful gradients. Easy tasks need fewer rollouts; hard tasks need more. This simple insight increases training efficiency significantly.

## Core Concept

For each task, define a cost (number of rollouts) and value (expected policy gradient magnitude). Tasks fall into regimes:

- **Easy regime**: Consistently succeed regardless of policy. Zero gradient. Should reduce budget.
- **Learning regime**: Sometimes fail, sometimes succeed. High gradient. Should increase budget.
- **Hard regime**: Consistently fail. Near-zero gradient. Should reduce budget.

Knapsack RL identifies which regime each task is in, then solves:

**Maximize:** Σ (gradient_quality_i × rollout_count_i)
**Subject to:** Σ (rollout_count_i) ≤ total_budget

This is a variant of the knapsack problem (hence the name), solvable with dynamic programming or greedy algorithms.

## Architecture Overview

- **Regime detector**: Classifies tasks (easy/learning/hard) based on success rate
- **Gradient estimator**: Estimates expected gradient magnitude per task
- **Value function**: Combines regime + gradient to define task value
- **Budget allocator**: Solves knapsack problem to assign rollout counts
- **Training loop**: Standard GRPO with adaptive per-task budgets

## Implementation Steps

First, implement task regime detection:

```python
import numpy as np
from collections import defaultdict

class RegimeDetector:
    """
    Detect which learning regime each task is in.
    """
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.task_history = defaultdict(list)  # task_id -> [success/failure bool]

    def record_outcome(self, task_id, success):
        """Record whether a rollout succeeded."""
        self.task_history[task_id].append(success)

        # Keep only recent outcomes
        if len(self.task_history[task_id]) > self.window_size:
            self.task_history[task_id].pop(0)

    def get_regime(self, task_id):
        """
        Classify task's current learning regime.

        Args:
            task_id: Task identifier

        Returns:
            regime: "easy", "learning", or "hard"
            success_rate: Fraction of recent successes
        """
        if task_id not in self.task_history or not self.task_history[task_id]:
            return "unknown", 0.5  # Assume learning if no history

        outcomes = self.task_history[task_id]
        success_rate = sum(outcomes) / len(outcomes)

        if success_rate > 0.8:
            regime = "easy"
        elif success_rate < 0.2:
            regime = "hard"
        else:
            regime = "learning"

        return regime, success_rate
```

Now implement gradient quality estimation:

```python
class GradientQualityEstimator:
    """
    Estimate the usefulness of gradients from a task.
    """
    def __init__(self):
        self.gradient_history = defaultdict(list)

    def record_gradient_magnitude(self, task_id, gradient_magnitude):
        """Record gradient magnitude from a rollout."""
        self.gradient_history[task_id].append(gradient_magnitude)

    def estimate_expected_gradient(self, task_id, regime):
        """
        Estimate gradient magnitude for this task.

        Args:
            task_id: Task identifier
            regime: Current regime ("easy", "learning", "hard")

        Returns:
            expected_gradient: Expected magnitude of policy gradients
        """
        if task_id not in self.gradient_history or not self.gradient_history[task_id]:
            # Default estimates based on regime
            defaults = {"easy": 0.1, "learning": 1.0, "hard": 0.1}
            return defaults.get(regime, 0.5)

        # Average recent gradient magnitudes
        recent_gradients = self.gradient_history[task_id][-10:]
        return np.mean(recent_gradients)

    def estimate_variance(self, task_id):
        """
        Estimate variance of gradients (uncertainty in quality).

        Args:
            task_id: Task identifier

        Returns:
            variance: Estimated gradient variance
        """
        if task_id not in self.gradient_history or len(self.gradient_history[task_id]) < 5:
            return 0.5  # High uncertainty if few samples

        recent = self.gradient_history[task_id][-10:]
        return np.var(recent)
```

Implement the knapsack solver for budget allocation:

```python
class BudgetAllocator:
    """
    Solve knapsack problem to allocate rollout budgets.
    """
    def __init__(self, regime_detector, gradient_estimator):
        self.regime_detector = regime_detector
        self.gradient_estimator = gradient_estimator

    def compute_task_value(self, task_id, regime, expected_gradient, variance):
        """
        Compute value (expected utility) of a rollout for this task.

        Args:
            task_id: Task identifier
            regime: Learning regime
            expected_gradient: Estimated gradient magnitude
            variance: Gradient variance

        Returns:
            value: Expected gradient * uncertainty discount
        """
        # Base value: how good are gradients from this task?
        base_value = expected_gradient

        # Adjust for variance: high variance = risky
        # Reduce value if we're uncertain about gradient quality
        uncertainty_discount = 1.0 / (1.0 + variance)

        # Regime-specific multiplier
        if regime == "learning":
            multiplier = 1.0  # Full value for learning regime
        elif regime == "easy":
            multiplier = 0.1  # Low value for easy (mostly zero gradients)
        else:  # hard
            multiplier = 0.2  # Low value for hard (mostly noise)

        value = base_value * uncertainty_discount * multiplier
        return value

    def allocate_budgets(self, tasks, total_budget, min_rollouts=1, max_rollouts=16):
        """
        Solve knapsack to allocate budgets to tasks.

        Args:
            tasks: List of task IDs
            total_budget: Total rollouts available
            min_rollouts: Minimum rollouts per task
            max_rollouts: Maximum rollouts per task

        Returns:
            allocations: Dict mapping task_id -> num_rollouts
        """
        # Compute value for each task
        task_values = {}
        for task_id in tasks:
            regime, success_rate = self.regime_detector.get_regime(task_id)
            expected_grad = self.gradient_estimator.estimate_expected_gradient(
                task_id, regime
            )
            variance = self.gradient_estimator.estimate_variance(task_id)

            value = self.compute_task_value(task_id, regime, expected_grad, variance)
            task_values[task_id] = value

        # Greedy knapsack: sort by value, allocate greedily
        sorted_tasks = sorted(task_values.items(), key=lambda x: x[1], reverse=True)

        allocations = {}
        remaining_budget = total_budget

        for task_id, value in sorted_tasks:
            # Allocate minimum first
            allocations[task_id] = min_rollouts
            remaining_budget -= min_rollouts

            if remaining_budget <= 0:
                break

        # Allocate remaining budget to high-value tasks
        for task_id, value in sorted_tasks:
            if remaining_budget <= 0:
                break

            # How many extra rollouts to allocate to this task?
            current = allocations[task_id]
            additional = min(remaining_budget, max_rollouts - current)

            allocations[task_id] += additional
            remaining_budget -= additional

        return allocations
```

Finally, integrate into GRPO training loop:

```python
def grpo_with_knapsack_allocation(
    model,
    tasks,
    num_training_iterations=100,
    total_budget_per_iteration=64,
    reallocation_frequency=5
):
    """
    Train model using GRPO with adaptive knapsack-based budget allocation.

    Args:
        model: Policy model to train
        tasks: List of training tasks
        num_training_iterations: Total iterations
        total_budget_per_iteration: Total rollouts per iteration
        reallocation_frequency: Reallocate budget every N iterations

    Returns:
        model: Trained model
    """
    regime_detector = RegimeDetector()
    gradient_estimator = GradientQualityEstimator()
    allocator = BudgetAllocator(regime_detector, gradient_estimator)

    for iteration in range(num_training_iterations):
        # Reallocate budgets periodically
        if iteration % reallocation_frequency == 0:
            allocations = allocator.allocate_budgets(
                tasks,
                total_budget_per_iteration,
                min_rollouts=1,
                max_rollouts=16
            )

            print(f"Iteration {iteration}: Budget allocation")
            for task_id, num_rollouts in sorted(allocations.items()):
                regime, _ = regime_detector.get_regime(task_id)
                print(f"  Task {task_id} ({regime}): {num_rollouts} rollouts")

        # Execute rollouts with allocated budgets
        all_rewards = []
        all_logprobs = []

        for task_id, num_rollouts in allocations.items():
            task = tasks[task_id]

            # Generate rollouts
            rollouts = []
            for _ in range(num_rollouts):
                rollout = model.generate(task)
                success = verify_task(task, rollout)
                rollouts.append((rollout, float(success)))

                # Record outcome for regime detection
                regime_detector.record_outcome(task_id, success)

            # Compute advantages (standard GRPO)
            rewards = torch.tensor([r[1] for r in rollouts])
            logprobs = torch.tensor([
                model.get_logprob(r[0], task) for r in rollouts
            ])

            # Normalize advantages within group
            advantage = rewards - rewards.mean()

            all_rewards.append(advantage)
            all_logprobs.append(logprobs)

            # Record gradient magnitudes for quality estimation
            for logprob, reward in zip(logprobs, rewards):
                # Gradient magnitude ∝ |log(pi) * advantage|
                grad_magnitude = abs(logprob * advantage).mean().item()
                gradient_estimator.record_gradient_magnitude(task_id, grad_magnitude)

        # Compute policy loss and update
        all_rewards = torch.cat(all_rewards)
        all_logprobs = torch.cat(all_logprobs)

        loss = -(all_logprobs * all_rewards.detach()).mean()

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: loss={loss.item():.4f}")

    return model
```

## Practical Guidance

**When to use Knapsack RL:**
- Heterogeneous task difficulty (mix of easy, medium, hard tasks)
- Limited compute budgets where efficiency matters
- Curriculum learning scenarios (naturally allocates more budget to learning tasks)
- Training benchmarks with 20+ diverse tasks

**When NOT to use:**
- Homogeneous task sets (all same difficulty — uniform allocation is fine)
- Unlimited compute budget (efficiency less important)
- Single-task training (no allocation decision to make)
- Real-time training where reallocation overhead is significant

**Efficiency gains:**

| Regime Mix | Uniform Allocation | Knapsack RL | Improvement |
|---|---|---|---|
| 50% easy, 50% hard | 48% accuracy | 52% accuracy | +4% |
| 25% easy, 50% learning, 25% hard | 52% accuracy | 56% accuracy | +4% |
| Highly imbalanced | 45% accuracy | 54% accuracy | +9% |

**Gradient signal improvement:**

| Metric | Baseline | Knapsack | Gain |
|---|---|---|---|
| Non-zero gradients | 30% | 50% | +67% |
| Avg gradient magnitude | 0.5 | 0.65 | +30% |
| Training efficiency | 1x | 1.3-1.4x | +30-40% |

**Hyperparameter tuning:**

| Parameter | Default | Tuning Notes |
|-----------|---------|--------------|
| min_rollouts | 1 | Higher prevents starvation (e.g., 2 for safety) |
| max_rollouts | 16 | Cap budget per task to maintain diversity |
| reallocation_freq | 5 | More frequent = responsive but noisy; less frequent = stable |
| window_size | 20 | More history = stable estimates; less history = adaptive |

**Common pitfalls:**
- **Starvation**: If a task gets 0 rollouts in early iterations, it may never recover. Use min_rollouts > 0.
- **Noisy regime detection**: If success rate flutters near boundary (e.g., 0.5), regime oscillates. Use moving average or hysteresis (easy: >0.8, hard: <0.2, learning: rest).
- **Gradient variance**: Early training has high gradient variance (unreliable estimates). Use conservative multipliers until iteration 10+.
- **Over-concentration**: If one high-value task dominates, others starve. Cap any task's share to 50% of budget.

**Integration checklist:**
- [ ] Verify uniform allocation baseline on your task set
- [ ] Implement regime detector; validate on 50 tasks (ensure reasonable regime distributions)
- [ ] Collect gradient statistics on 100 rollouts per task; verify estimates are reasonable
- [ ] Run knapsack allocation; inspect allocations (easy tasks < learning < hard tasks)
- [ ] Train with knapsack allocation; compare to uniform (measure performance gain)
- [ ] Monitor task-wise success rates (should improve over time)
- [ ] Periodically inspect allocations; reallocate frequency should match task changes

Reference: https://arxiv.org/abs/2509.25849
