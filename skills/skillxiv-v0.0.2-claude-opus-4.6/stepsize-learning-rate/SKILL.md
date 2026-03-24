---
name: stepsize-learning-rate
title: "Stepsize anything: A unified learning rate schedule for budgeted-iteration training"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24452"
keywords: [learning rate schedule, optimization, convergence, budget-aware, unified framework]
description: "Improve training efficiency under iteration budgets using the Unified Budget-Aware (UBA) schedule, a theoretically grounded learning rate approach governed by a single hyperparameter φ that balances adaptability across network architectures."
---

# Stepsize Anything: Unified Learning Rate Schedule for Budgeted-Iteration Training

## Core Concept

Stepsize Anything introduces the Unified Budget-Aware (UBA) learning rate schedule, addressing a practical gap in modern deep learning: how to set learning rates when training is constrained by a fixed iteration budget rather than convergence criteria.

Traditional learning rate schedules assume unlimited computational resources and optimize for eventual convergence. In contrast, UBA is theoretically grounded in landscape curvature properties and explicitly accounts for predetermined computational constraints. The key innovation is expressing the schedule through a single interpretable hyperparameter φ that relates to the problem's condition number, enabling consistent performance across diverse architectures without extensive hyperparameter tuning.

## Architecture Overview

- **Condition Number Connection**: Theoretical link between hyperparameter φ and landscape curvature (condition number)
- **Budget-Aware Framework**: Explicitly incorporates iteration budget into schedule design rather than treating it as secondary
- **Convergence Guarantees**: Proven convergence properties for different φ values
- **Unified Applicability**: Single schedule works across CNNs, Transformers, and other architectures
- **Adaptivity**: Automatically adjusts strategy based on available budget and problem properties
- **Minimal Tuning**: Single hyperparameter φ replaces extensive per-task hyperparameter search

## Implementation

The following steps outline how to implement and use the UBA schedule:

1. **Estimate condition number** - Assess landscape curvature characteristics of your problem
2. **Set hyperparameter φ** - Choose φ value based on condition number estimate and risk tolerance
3. **Initialize learning rate schedule** - Compute initial lr based on budget and φ
4. **Apply schedule during training** - Update learning rate according to UBA formula at each step
5. **Monitor convergence** - Track loss and validation metrics to validate choice of φ
6. **Adapt if needed** - Adjust φ dynamically if budget changes or convergence is suboptimal

```python
import torch
import torch.optim as optim
from typing import Callable
import math

class UBAScheduler:
    def __init__(self, optimizer: optim.Optimizer, total_steps: int,
                 phi: float = 0.5, base_lr: float = 0.1, condition_number: float = 1.0):
        """
        Initialize Unified Budget-Aware learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer
            total_steps: Total training iterations (budget)
            phi: Hyperparameter balancing adaptivity (0.0-1.0).
                 ~0.3: Conservative, gradual decay
                 ~0.5: Moderate, balanced
                 ~0.8: Aggressive, fast initial decay
            base_lr: Initial learning rate
            condition_number: Landscape condition number estimate (default ~1.0 for well-conditioned)
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.phi = phi
        self.base_lr = base_lr
        self.condition_number = condition_number
        self.current_step = 0

    def get_learning_rate(self) -> float:
        """Compute learning rate for current step using UBA formula."""
        if self.total_steps <= 0:
            return self.base_lr

        # Normalize step to [0, 1]
        progress = self.current_step / self.total_steps

        # UBA schedule: combines budget awareness with condition-number adaptation
        # Decay stronger early (φ controls speed), gentler late
        schedule_value = (1 - progress * self.phi) ** 2

        # Adjust by condition number (ill-conditioned problems need smaller learning rates)
        adaptive_factor = 1.0 / math.sqrt(self.condition_number)

        lr = self.base_lr * schedule_value * adaptive_factor
        return lr

    def step(self):
        """Advance scheduler and update optimizer learning rates."""
        lr = self.get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1

    def get_schedule_values(self, steps: int = None) -> list:
        """Visualize schedule for debugging."""
        if steps is None:
            steps = self.total_steps
        values = []
        for step in range(steps):
            self.current_step = step
            values.append(self.get_learning_rate())
        return values


class BudgetAwareTrainer:
    def __init__(self, model: torch.nn.Module, train_loader, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.device = device

    def train_with_budget(self, epochs: int, total_budget: int, phi: float = 0.5,
                         base_lr: float = 0.1, condition_number: float = 1.0):
        """Train model with fixed iteration budget and UBA schedule."""
        optimizer = optim.SGD(self.model.parameters(), lr=base_lr)
        scheduler = UBAScheduler(optimizer, total_budget, phi=phi,
                                base_lr=base_lr, condition_number=condition_number)
        loss_fn = torch.nn.CrossEntropyLoss()

        total_steps = 0
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if total_steps >= total_budget:
                    print(f"Budget exhausted at step {total_steps}")
                    return

                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                output = self.model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_steps += 1
                if total_steps % 100 == 0:
                    lr = scheduler.get_learning_rate()
                    print(f"Step {total_steps}: Loss={loss.item():.4f}, LR={lr:.6f}")

        return total_steps
```

## Practical Guidance

**Hyperparameter φ selection:**
- **φ = 0.3**: Conservative decay, slower learning rate decrease. Use for well-initialized models or large budgets.
- **φ = 0.5**: Moderate decay, recommended default. Balances early exploration with late refinement.
- **φ = 0.8**: Aggressive decay, rapid initial learning rate drop. Use for noisy data or small budgets.

**Condition number estimation:**
- Well-conditioned problems (condition number ~1-10): Use default factor
- Moderately ill-conditioned (10-100): Apply sqrt adjustment
- Severely ill-conditioned (>100): Use preconditioning or reduce learning rate further

**When to use:**
- Training with fixed computational budgets (e.g., hourly slots, limited GPU time)
- Comparing performance across models with different problem scales
- Automated hyperparameter tuning where single parameter is preferable
- Training diverse architectures (CNNs, Transformers, RNNs) with consistent strategy

**When NOT to use:**
- Training until convergence with unlimited budget
- Adaptive methods (Adam, RMSprop) where built-in schedule suffices
- Problems with highly unknown landscape characteristics (condition number unknowable)
- Distributed training where synchronization overhead dominates

**Common pitfalls:**
- **Underestimating condition number**: Too large effective learning rate causes divergence; start conservative
- **φ out of range**: Values outside [0, 1] invalidate theoretical guarantees; stay in recommended range
- **Fixed schedule across problem scales**: Rescale base_lr by problem scale for consistency
- **Ignoring batch size effects**: Effective learning rate scales with batch size; adjust base_lr proportionally

## Reference

The UBA schedule demonstrates consistent improvements across diverse scenarios including ResNet for computer vision and OLMo for language modeling, with varying training budgets. The single-parameter design simplifies hyperparameter tuning and enables principled adaptation across problem domains.

Original paper: "Stepsize anything: A unified learning rate schedule for budgeted-iteration training" (arxiv.org/abs/2505.24452)
