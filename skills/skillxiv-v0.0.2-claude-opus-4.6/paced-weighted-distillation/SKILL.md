---
name: paced-weighted-distillation
title: "PACED: Distillation at the Frontier of Student Competence"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.11178"
keywords: [Knowledge Distillation, Curriculum Learning, Student Competence, Beta Distribution]
description: "Weight distillation loss by student pass rate using Beta kernel: suppress mastered (high pass rate) and intractable (low pass rate) problems, prioritize intermediate zone where learning signal is richest."
---

# Technique: Beta-Weighted Pass-Rate Distillation

Knowledge distillation suffers from inefficient learning signals: problems the student already solves contribute gradients with vanishing signal, while unsolvable problems produce noisy gradients. PACED focuses learning in the **zone of proximal development** (ZPD)—where the student has partial competence.

The approach uses a simple Beta kernel weighting function w(p) = p^α(1-p)^β (typically α=β=1) to weight each problem's loss by its student pass rate. This creates a bell-shaped curve peaking at intermediate competence (p≈0.5), theoretically motivated by signal-to-noise ratio analysis.

## Core Concept

The key observation is that gradient signal-to-noise ratio (SNR) provably vanishes at both pass-rate boundaries: when p→0 (intractable) and p→1 (mastered). The Beta kernel emerges as the optimal weight family from leading-order SNR analysis.

Implementation is trivial—just scale each problem's loss by w(p)—yet achieves substantial improvements from more efficient gradient allocation.

## Architecture Overview

- **Student model**: Target model being trained
- **Teacher model**: Reference providing distillation targets
- **Pass rate estimator**: Evaluates student performance per problem
- **Beta kernel weighter**: Computes per-problem weights
- **Loss scaler**: Applies weights to distillation loss

## Implementation Steps

### Step 1: Compute Student Pass Rates

Estimate pass rate for each problem in the training set.

```python
import torch
import numpy as np
from collections import defaultdict

class PassRateEstimator:
    def __init__(self, num_rollouts=8):
        self.num_rollouts = num_rollouts
        self.pass_rates = {}

    def estimate_pass_rates(self, student_model, problems, batch_size=32):
        """
        For each problem, estimate fraction of correct rollouts.

        problems: list of problem identifiers
        student_model: model to evaluate
        """
        problem_pass_counts = defaultdict(int)
        problem_rollout_counts = defaultdict(int)

        student_model.eval()

        for problem_idx, problem in enumerate(problems):
            # Generate K rollouts for this problem
            for rollout_idx in range(self.num_rollouts):
                # Run student on problem
                output = student_model.solve(problem)

                # Check correctness
                is_correct = self.check_correctness(output, problem)

                if is_correct:
                    problem_pass_counts[problem_idx] += 1

                problem_rollout_counts[problem_idx] += 1

        # Compute pass rates
        for problem_idx in range(len(problems)):
            if problem_rollout_counts[problem_idx] > 0:
                pass_rate = (
                    problem_pass_counts[problem_idx] /
                    problem_rollout_counts[problem_idx]
                )
            else:
                pass_rate = 0.0

            self.pass_rates[problem_idx] = pass_rate

        return self.pass_rates

    def check_correctness(self, output, problem):
        """External verification of correctness."""
        # Use problem's ground truth or external evaluator
        return self._verify(output, problem)

    def _verify(self, output, problem):
        # Placeholder: implement based on task
        return True
```

### Step 2: Compute Beta Kernel Weights

For each problem, compute w(p) = p^α(1-p)^β.

```python
class BetaKernelWeighter:
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Beta kernel parameters.

        alpha=beta=1: w(p) = p(1-p), peaks at p=0.5
        """
        self.alpha = alpha
        self.beta = beta

    def compute_weights(self, pass_rates):
        """
        Compute weights from pass rates.

        pass_rates: dict {problem_idx: pass_rate}
        returns: dict {problem_idx: weight}
        """
        weights = {}

        for problem_idx, pass_rate in pass_rates.items():
            # Clamp to [0, 1]
            p = np.clip(pass_rate, 0, 1)

            # Beta kernel: p^α * (1-p)^β
            weight = (p ** self.alpha) * ((1 - p) ** self.beta)

            weights[problem_idx] = weight

        return weights

    def visualize_kernel(self, alpha=1.0, beta=1.0):
        """Visualize beta kernel shape."""
        p_values = np.linspace(0, 1, 100)
        weights = p_values ** alpha * (1 - p_values) ** beta

        return p_values, weights
```

### Step 3: Weighted Distillation Loss

Apply Beta kernel weights to distillation training.

```python
class WeightedDistillationTrainer:
    def __init__(self, student_model, teacher_model, pass_rate_estimator, weighter):
        self.student = student_model
        self.teacher = teacher_model
        self.pass_rate_estimator = pass_rate_estimator
        self.weighter = weighter

    def train_step(
        self,
        problems,
        teacher_targets,
        optimizer,
        alpha=1.0,
        beta=1.0,
        temperature=1.0
    ):
        """
        Single training step with Beta-weighted distillation.

        problems: list of problem instances
        teacher_targets: teacher predictions for each problem
        optimizer: torch optimizer
        """
        batch_size = len(problems)

        # Compute pass rates (can cache across steps)
        pass_rates = self.pass_rate_estimator.estimate_pass_rates(
            self.student,
            problems,
            num_rollouts=4  # Faster evaluation for training
        )

        # Compute Beta kernel weights
        weighter = BetaKernelWeighter(alpha=alpha, beta=beta)
        weights = weighter.compute_weights(pass_rates)

        # Forward pass
        student_logits = self.student(problems)

        # Distillation loss with weights
        total_weighted_loss = 0

        for idx in range(batch_size):
            problem_idx = idx

            # Standard KL divergence for this problem
            student_log_probs = torch.log_softmax(
                student_logits[idx] / temperature,
                dim=-1
            )
            teacher_probs = torch.softmax(
                teacher_targets[idx] / temperature,
                dim=-1
            )

            kl_div = torch.nn.functional.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            )

            # Weight by Beta kernel
            weight = weights.get(problem_idx, 0.5)
            weighted_loss = weight * kl_div

            total_weighted_loss += weighted_loss

        avg_loss = total_weighted_loss / batch_size

        # Backward pass
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        return {
            'loss': avg_loss.item(),
            'pass_rates_mean': np.mean(list(pass_rates.values())),
            'weights_mean': np.mean(list(weights.values()))
        }
```

### Step 4: Curriculum Training Loop

Train with periodic pass rate re-estimation to adapt weights.

```python
def train_with_paced_curriculum(
    student_model,
    teacher_model,
    train_problems,
    teacher_targets,
    num_epochs=10,
    alpha=1.0,
    beta=1.0,
    reestimate_every_steps=100
):
    """
    Full training loop with PACED weighting.
    """
    pass_rate_estimator = PassRateEstimator(num_rollouts=8)
    weighter = BetaKernelWeighter(alpha=alpha, beta=beta)
    trainer = WeightedDistillationTrainer(
        student_model,
        teacher_model,
        pass_rate_estimator,
        weighter
    )

    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_idx in range(0, len(train_problems), batch_size=32):
            batch_problems = train_problems[batch_idx:batch_idx + 32]
            batch_targets = teacher_targets[batch_idx:batch_idx + 32]

            # Training step
            metrics = trainer.train_step(
                batch_problems,
                batch_targets,
                optimizer,
                alpha=alpha,
                beta=beta
            )

            epoch_loss += metrics['loss']
            global_step += 1

            # Periodically re-estimate pass rates
            if global_step % reestimate_every_steps == 0:
                print(f"Step {global_step}: Loss={metrics['loss']:.4f}, "
                      f"Avg Pass Rate={metrics['pass_rates_mean']:.2f}, "
                      f"Avg Weight={metrics['weights_mean']:.4f}")

        print(f"Epoch {epoch + 1}: Avg Loss={epoch_loss / (len(train_problems) // 32):.4f}")

    return student_model
```

### Step 5: Theoretical Justification

Demonstrate the SNR analysis motivating Beta weighting.

```python
class DistillationSNRAnalysis:
    """Analyze gradient SNR for distillation loss."""

    @staticmethod
    def gradient_snr_at_pass_rate(pass_rate, model_scale=1.0):
        """
        Estimate signal-to-noise ratio of gradients at given pass rate.

        Theory: SNR ∝ p(1-p) (provably maximal at p=0.5)
        """
        # Signal: gradient from high-quality predictions (pass rate > 0.5)
        signal = pass_rate * (1 - pass_rate)

        # Noise: gradient variance, higher at extremes
        noise = 1.0 / (1 + model_scale * signal)

        snr = signal / noise

        return snr

    @staticmethod
    def plot_snr_curve():
        """Visualize why Beta weighting is optimal."""
        pass_rates = np.linspace(0, 1, 100)
        snr_values = [
            DistillationSNRAnalysis.gradient_snr_at_pass_rate(p)
            for p in pass_rates
        ]

        # This should resemble p(1-p)
        return pass_rates, snr_values
```

## Practical Guidance

**When to Use:**
- Knowledge distillation with heterogeneous problem difficulty
- Curriculum learning where students need staged difficulty progression
- Scenarios where "mastery" is problem-dependent
- Multi-task learning with varying task difficulty

**When NOT to Use:**
- Uniform problem difficulty (no benefit from weighting)
- Extreme computational constraints (requires pass rate estimation)
- Single-task settings where curriculum unnecessary

**Hyperparameter Tuning:**
- **α, β**: Default α=β=1 creates p(1-p); increase α/β for sharper/wider curve
- **num_rollouts for pass rate estimation**: 4-8 typical; more accurate but slower
- **reestimate_every_steps**: 100-500 steps; balance adaptation vs overhead
- **temperature**: 1.0-2.0 for softmax sharpening

**Common Pitfalls:**
- Stale pass rates (re-estimate frequently, especially early training)
- Kernel shape mismatched to problem distribution (visualize and tune)
- Too aggressive weighting (can eliminate learning from hard problems)
- Forgetting that ZPD varies per student (re-estimate as model improves)

## Reference

[PACED paper on arXiv](https://arxiv.org/abs/2603.11178)
