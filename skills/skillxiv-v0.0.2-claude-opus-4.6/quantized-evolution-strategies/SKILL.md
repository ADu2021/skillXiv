---
name: quantized-evolution-strategies
title: "Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03120"
keywords: [Quantization, Fine-Tuning, Evolution Strategies, Gradient Stagnation, Memory Efficiency]
description: "Fine-tune quantized LLMs directly in low-precision discrete parameter space using evolution strategies with accumulated error feedback. Overcome gradient stagnation in quantized models by accumulating fractional updates using Delta-Sigma modulation, achieving significant improvements in INT4 quantized models without full-precision gradients."
---

# Quantized Evolution Strategies: High-precision Fine-tuning of Quantized LLMs

## Problem Context

Quantized language models (INT4, INT8) reduce memory but create a fundamental optimization problem: gradient updates often fall below quantization thresholds, causing training to stagnate. Standard backprop doesn't work because discrete rounding eliminates infinitesimal signals. Quantized Evolution Strategies (QES) solves this using population-based perturbations and accumulated error feedback inspired by signal processing.

## Core Concept

QES operates in three phases: (1) estimate gradients using population-based perturbations without backprop, (2) accumulate error residuals at each iteration (Delta-Sigma modulation), (3) make discrete updates only when accumulated error crosses rounding threshold. This transforms infinitesimal signals into discrete changes within quantized parameter space.

## Architecture Overview

- **Population-based gradients**: Estimate gradients via finite differences on random perturbations
- **Error accumulation**: Track fractional updates that don't cross quantization boundaries
- **Delta-Sigma feedback**: Accumulated residuals trigger discrete parameter changes
- **Stateless replay**: Reconstruct optimization history from seeds, not full state
- **Direct quantization optimization**: No full-precision shadow parameters

## Implementation

### Step 1: Population-based gradient estimation

```python
import torch
import numpy as np
from typing import Tuple

class PopulationGradientEstimator:
    """Estimate gradients from function evaluations (no backprop)."""

    def __init__(
        self,
        pop_size: int = 32,
        sigma: float = 0.01,
        learning_rate: float = 1e-3
    ):
        self.pop_size = pop_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def estimate_gradients(
        self,
        model: torch.nn.Module,
        reward_fn,
        batch: dict
    ) -> dict:
        """
        Estimate gradients using population sampling.

        Args:
            model: Quantized language model
            reward_fn: Function returning scalar reward
            batch: Input batch

        Returns:
            estimated_grads: Per-parameter gradient estimates
        """
        # Get parameter shapes
        params = {name: p for name, p in model.named_parameters()}
        param_names = list(params.keys())

        # Baseline reward
        baseline_reward = reward_fn(model, batch)

        # Sample perturbations
        perturbations = {}
        rewards = []

        for i in range(self.pop_size):
            # Random Gaussian perturbation
            pert_dict = {}
            for name in param_names:
                pert = self.sigma * torch.randn_like(params[name])
                pert_dict[name] = pert

            # Apply perturbation
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.data.add_(pert_dict[name])

            # Evaluate
            reward = reward_fn(model, batch)
            rewards.append(reward)

            # Restore
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.data.sub_(pert_dict[name])

            perturbations[i] = pert_dict

        rewards = np.array(rewards)

        # Estimate gradient: (R - baseline) * perturbation / sigma^2
        estimated_grads = {}
        for name in param_names:
            grad_estimate = torch.zeros_like(params[name])

            for i in range(self.pop_size):
                advantage = rewards[i] - baseline_reward
                pert = perturbations[i][name]
                grad_estimate.add_(advantage * pert)

            grad_estimate.div_(self.pop_size * self.sigma ** 2)
            estimated_grads[name] = grad_estimate

        return estimated_grads
```

### Step 2: Accumulated error feedback (Delta-Sigma modulation)

```python
class AccumulatedErrorFeedback:
    """Accumulate fractional updates until crossing quantization threshold."""

    def __init__(self, quantization_bits: int = 4):
        self.quantization_bits = quantization_bits
        self.quantization_step = 1.0 / (2 ** quantization_bits)
        self.error_buffer = {}

    def apply_accumulated_update(
        self,
        param_tensor: torch.Tensor,
        param_name: str,
        gradient: torch.Tensor,
        learning_rate: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Apply update with error accumulation.

        Strategy:
        - Compute update: delta = lr * gradient
        - Add to error buffer: error[name] += delta
        - When |error| > quantization_step: make discrete update
        """
        if param_name not in self.error_buffer:
            self.error_buffer[param_name] = torch.zeros_like(param_tensor)

        # Proposed update
        proposed_update = learning_rate * gradient

        # Accumulate error
        self.error_buffer[param_name].add_(proposed_update)

        # Check for quantization threshold crossing
        error = self.error_buffer[param_name]

        # Discrete step: sign of error, magnitude 1 quant step
        discrete_update = torch.sign(error) * self.quantization_step
        crosses_threshold = torch.abs(error) >= self.quantization_step

        # Apply discrete update where threshold crossed
        actual_update = discrete_update.clone()
        actual_update[~crosses_threshold] = 0

        # Update parameters
        param_tensor.add_(actual_update)

        # Reset error for positions that updated
        self.error_buffer[param_name][crosses_threshold] = 0

        # Track stats
        num_updates = crosses_threshold.sum().item()

        return param_tensor, num_updates
```

### Step 3: Stateless seed replay optimization

```python
class StatelessSeedReplay:
    """Reconstruct optimization state from random seeds without memory overhead."""

    def __init__(self):
        self.seed_history = []
        self.reward_history = []

    def save_iteration_state(self, seed: int, reward: float):
        """Save only seed and reward, not full gradient state."""
        self.seed_history.append(seed)
        self.reward_history.append(reward)

    def reconstruct_gradients(
        self,
        model: torch.nn.Module,
        iteration: int,
        sigma: float = 0.01
    ) -> dict:
        """
        Reconstruct gradients at specific iteration from saved seeds.

        Avoids storing full gradient history.
        """
        # Replay random perturbations from saved seed
        torch.manual_seed(self.seed_history[iteration])

        params = {name: p for name, p in model.named_parameters()}
        estimated_grads = {}

        baseline_reward = self.reward_history[iteration]

        # Regenerate perturbations
        for i in range(32):  # pop_size
            pert_dict = {}
            for name in params.keys():
                pert = sigma * torch.randn_like(params[name])
                pert_dict[name] = pert

            # Use stored reward
            if i < len(self.reward_history):
                reward = self.reward_history[i + iteration * 32]
            else:
                reward = baseline_reward

            # Accumulate gradient
            for name, pert in pert_dict.items():
                advantage = reward - baseline_reward
                if name not in estimated_grads:
                    estimated_grads[name] = torch.zeros_like(params[name])
                estimated_grads[name].add_(advantage * pert)

        # Normalize
        for name in estimated_grads:
            estimated_grads[name].div_(32 * sigma ** 2)

        return estimated_grads
```

### Step 4: QES training step

```python
class QuantizedEvolutionStrategies:
    """Full QES optimizer for quantized model fine-tuning."""

    def __init__(
        self,
        model: torch.nn.Module,
        reward_fn,
        quantization_bits: int = 4,
        pop_size: int = 32,
        learning_rate: float = 1e-3,
        use_seed_replay: bool = True
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.pop_size = pop_size
        self.learning_rate = learning_rate

        self.grad_estimator = PopulationGradientEstimator(
            pop_size=pop_size, sigma=0.01, learning_rate=learning_rate
        )
        self.error_feedback = AccumulatedErrorFeedback(quantization_bits)
        self.seed_replay = StatelessSeedReplay() if use_seed_replay else None

    def training_step(self, batch: dict) -> dict:
        """Single QES training step."""
        # Estimate gradients
        estimated_grads = self.grad_estimator.estimate_gradients(
            self.model, self.reward_fn, batch
        )

        # Apply updates with error accumulation
        total_updates = 0
        for name, param in self.model.named_parameters():
            if name in estimated_grads:
                grad = estimated_grads[name]
                _, num_updates = self.error_feedback.apply_accumulated_update(
                    param.data, name, grad, self.learning_rate
                )
                total_updates += num_updates

        # Evaluate updated model
        reward = self.reward_fn(self.model, batch)

        return {
            'reward': reward,
            'num_updates': total_updates,
            'update_ratio': total_updates / sum(p.numel() for p in self.model.parameters())
        }
```

### Step 5: Fine-tuning loop

```python
def fine_tune_quantized_llm(
    model: torch.nn.Module,
    train_loader,
    verifier,
    num_steps: int = 1000,
    quantization_bits: int = 4,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
):
    """
    Fine-tune quantized LLM using QES.

    Args:
        model: Quantized language model (INT4 or INT8)
        train_loader: Training data
        verifier: Reward function (task accuracy or loss)
        num_steps: Optimization steps
        quantization_bits: Bit-width of quantization
        learning_rate: ES learning rate
    """
    def reward_fn(model, batch):
        """Compute reward on batch."""
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            return -loss.item()  # Negative loss = reward

    qes = QuantizedEvolutionStrategies(
        model, reward_fn,
        quantization_bits=quantization_bits,
        pop_size=32,
        learning_rate=learning_rate
    )

    for step in range(num_steps):
        batch = next(iter(train_loader))
        batch = {k: v.to(device) for k, v in batch.items()}

        metrics = qes.training_step(batch)

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: "
                  f"Reward={metrics['reward']:.4f}, "
                  f"Updates={metrics['num_updates']}, "
                  f"UpdateRatio={metrics['update_ratio']:.4f}")

    return model
```

## Practical Guidance

**When to use**: Fine-tuning quantized models (INT4, INT8) where gradient-based methods fail due to discrete rounding

**Hyperparameters**:
- **pop_size**: 16-64 (larger = better gradient estimates but slower)
- **sigma**: 0.005-0.02 (perturbation scale)
- **learning_rate**: 1e-4 to 1e-2 (depends on pop_size)
- **quantization_bits**: 4 (INT4) or 8 (INT8)

**Key advantages**:
- Overcomes gradient stagnation in quantized models
- No full-precision shadow parameters (memory efficient)
- Population-based approach robust to noise
- Error accumulation ensures progress despite discrete constraints

**Common pitfalls**:
- pop_size too small → noisy gradients
- learning_rate too high → thrashing in discrete space
- Not validating that updates actually improve loss
- Using on non-quantized models (wastes computation)

**Scaling**: Linear in pop_size and model size. Efficient for INT4 models.

## Reference

Paper: https://arxiv.org/abs/2602.03120
Related work: Evolution strategies, quantization, fine-tuning, Delta-Sigma modulation
Benchmarks: Arithmetic reasoning on INT4 quantized models
