---
name: t3d-trajectory-distillation
title: "T3D: Few-Step Diffusion LMs via Trajectory Self-Distillation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.12262"
keywords: [Diffusion Language Models, Distillation, Few-Step Generation, Self-Supervision, Mode-Seeking]
description: "Accelerate diffusion language models from many to few generation steps using trajectory self-distillation. Collect clean-noisy state pairs along teacher trajectories, apply reverse-KL mode-seeking distillation, and weight losses by decoding order to prioritize early predictions where cascading errors compound."
---

# T3D: Few-Step Diffusion LMs via Trajectory Self-Distillation

## Problem Context

Diffusion language models require many sampling steps (25-50+) for high-quality generation. Few-step variants are needed for deployment, but naive training on random corruption schedules causes train-test mismatch: inference uses confidence-based schedules while training sees uniform random masking. T3D solves this by collecting trajectory pairs from teacher's actual inference schedule and using mode-seeking distillation.

## Core Concept

T3D operates in two phases: (1) collect (clean, intermediate) state pairs along teacher trajectories during inference, (2) train student via reverse-KL objective that promotes mode-seeking (concentrating on high-probability outputs) with path-consistency weighting. This addresses the multimodal posterior problem where forward-KL spreads mass across modes.

## Architecture Overview

- **Trajectory collection**: Run full teacher steps, save clean (x₀) and intermediate (xₜ) states
- **Reverse-KL distillation**: Use discriminative KL divergence promoting mode concentration
- **Path-consistency weighting**: Weight token losses by decoding order (early tokens > later tokens)
- **Mode-seeking loss**: Prevent mode-covering behavior in highly multimodal posteriors
- **Few-step inference**: Student generates quality outputs in 2-4 steps

## Implementation

### Step 1: Collect trajectory pairs from teacher

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

class TrajectoryCollector:
    """Collect (clean, corrupted) state pairs from teacher inference."""

    def __init__(
        self,
        teacher_model,
        num_collection_steps: int = 50,
        corruption_schedule: str = 'linear'
    ):
        self.teacher_model = teacher_model
        self.num_collection_steps = num_collection_steps
        self.corruption_schedule = corruption_schedule

    def get_corruption_schedule(self, num_steps: int) -> torch.Tensor:
        """
        Get corruption level schedule for diffusion.

        Args:
            num_steps: Number of denoising steps

        Returns:
            sigma_t: Noise level at each step [num_steps]
        """
        if self.corruption_schedule == 'linear':
            # Linear schedule from max to min
            return torch.linspace(1.0, 0.0, num_steps)
        elif self.corruption_schedule == 'cosine':
            # Cosine schedule
            t = torch.linspace(0, 1, num_steps)
            return torch.cos(t * torch.pi / 2)
        else:
            raise ValueError(f"Unknown schedule: {self.corruption_schedule}")

    def corrupt_tokens(
        self,
        clean_tokens: torch.Tensor,  # [seq_len]
        sigma: float,
        mask_token_id: int
    ) -> torch.Tensor:
        """
        Add noise by masking tokens.

        Args:
            clean_tokens: Original tokens
            sigma: Noise level (0-1, where 1=all masked)
            mask_token_id: Mask token ID

        Returns:
            corrupted: Noisy version with sigma fraction masked
        """
        num_mask = int(sigma * len(clean_tokens))
        mask_positions = torch.randperm(len(clean_tokens))[:num_mask]

        corrupted = clean_tokens.clone()
        corrupted[mask_positions] = mask_token_id

        return corrupted

    def collect_trajectories(
        self,
        prompt: str,
        num_trajectories: int = 4
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate trajectories and collect (clean, intermediate) pairs.

        Args:
            prompt: Input prompt
            num_trajectories: Number of trajectories to collect

        Returns:
            trajectories: List of {clean: Tensor, intermediates: List[Tensor]}
        """
        trajectories = []
        corruption_schedule = self.get_corruption_schedule(self.num_collection_steps)

        for traj_idx in range(num_trajectories):
            # Generate clean sample with teacher
            clean_tokens = self.teacher_model.generate(prompt, num_steps=50)

            # Collect intermediate corrupted versions
            intermediates = []
            for step_idx, sigma in enumerate(corruption_schedule):
                corrupted = self.corrupt_tokens(
                    clean_tokens,
                    sigma.item(),
                    mask_token_id=self.teacher_model.mask_token_id
                )
                intermediates.append(corrupted)

            trajectories.append({
                'clean': clean_tokens,
                'intermediates': intermediates,
                'schedule': corruption_schedule
            })

        return trajectories
```

### Step 2: Implement reverse-KL distillation objective

```python
class ReverseKLDistillation:
    """Reverse-KL (mode-seeking) distillation for few-step training."""

    def __init__(
        self,
        teacher_model,
        student_model,
        temperature: float = 1.0
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature

    def compute_teacher_logits(
        self,
        corrupted_tokens: torch.Tensor,
        clean_tokens: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """Get teacher's predicted distribution for clean tokens."""
        with torch.no_grad():
            teacher_logits = self.teacher_model.forward(
                corrupted_tokens,
                step=sigma,
                return_logits=True
            )
        return teacher_logits

    def compute_reverse_kl_loss(
        self,
        student_logits: torch.Tensor,   # [seq_len, vocab_size]
        teacher_logits: torch.Tensor,   # [seq_len, vocab_size]
        clean_tokens: torch.Tensor      # [seq_len]
    ) -> torch.Tensor:
        """
        Compute reverse-KL loss: E_q[log q - log p].

        Promotes mode concentration on teacher's high-probability outputs.
        q = student (learner), p = teacher (reference)
        """
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)

        # Sample from student
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        student_samples = torch.multinomial(
            student_probs.view(-1, student_probs.shape[-1]),
            num_samples=1
        ).squeeze(-1)

        # Compute KL: E_q[log q - log p]
        # In practice, approximate with cross-entropy from student samples
        kl_loss = (
            student_log_probs[range(len(clean_tokens)), student_samples] -
            teacher_log_probs[range(len(clean_tokens)), student_samples]
        ).mean()

        return kl_loss

    def compute_mode_seeking_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        clean_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Mode-seeking objective: penalize student spreading mass.

        Uses reverse-KL which avoids mode-covering behavior.
        """
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Reverse-KL: sum over positions
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # For each position, compute KL(student || teacher)
        # = sum_v student_probs[v] * (log student_probs[v] - log teacher_probs[v])
        kl_per_position = (
            student_probs *
            (student_log_probs - F.log_softmax(teacher_logits, dim=-1))
        ).sum(dim=-1)

        return kl_per_position.mean()
```

### Step 3: Path-consistency weighting

```python
class PathConsistencyWeighting:
    """Weight losses by token decoding order."""

    @staticmethod
    def compute_order_weights(
        seq_len: int,
        schedule: torch.Tensor,
        order: str = 'confidence'
    ) -> torch.Tensor:
        """
        Compute per-token loss weights based on decoding order.

        Early tokens (where cascading errors start) get higher weight.

        Args:
            seq_len: Sequence length
            schedule: Corruption schedule
            order: 'confidence' (teacher confidence order) or 'sequential'

        Returns:
            weights: [seq_len] weight for each position
        """
        if order == 'sequential':
            # Linear decay: first token gets highest weight
            weights = torch.linspace(1.0, 0.3, seq_len)
        elif order == 'confidence':
            # Weights derived from teacher's confidence ranking
            # Higher confidence tokens earlier; they cascade errors more
            confidence_ranks = torch.arange(seq_len, dtype=torch.float32)
            weights = (seq_len - confidence_ranks) / seq_len
        else:
            weights = torch.ones(seq_len)

        return weights

    @staticmethod
    def apply_path_consistency_loss(
        loss_per_position: torch.Tensor,  # [seq_len]
        weights: torch.Tensor              # [seq_len]
    ) -> torch.Tensor:
        """Weight loss by position importance."""
        weighted_loss = (loss_per_position * weights).mean()
        return weighted_loss
```

### Step 4: Full training step with distillation

```python
class T3DTrainer:
    """Trajectory-based few-step distillation trainer."""

    def __init__(
        self,
        teacher_model,
        student_model,
        optimizer,
        num_student_steps: int = 4
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.optimizer = optimizer
        self.num_student_steps = num_student_steps

        self.collector = TrajectoryCollector(teacher_model)
        self.kl_distiller = ReverseKLDistillation(teacher_model, student_model)
        self.path_weighting = PathConsistencyWeighting()

    def training_step(
        self,
        prompt: str,
        num_trajectories: int = 4
    ) -> Dict[str, float]:
        """
        Single training step on trajectories.

        Args:
            prompt: Input prompt
            num_trajectories: Trajectories to collect and train on

        Returns:
            metrics: Loss and auxiliary metrics
        """
        # Step 1: Collect trajectories
        trajectories = self.collector.collect_trajectories(
            prompt, num_trajectories=num_trajectories
        )

        total_loss = 0.0
        num_samples = 0

        for traj in trajectories:
            clean_tokens = traj['clean']
            intermediates = traj['intermediates']
            schedule = traj['schedule']

            # Compute path-consistency weights
            path_weights = self.path_weighting.compute_order_weights(
                len(clean_tokens), schedule, order='confidence'
            )

            # Sample corruption levels from schedule
            num_train_steps = len(schedule) // max(1, len(schedule) // self.num_student_steps)

            for step_idx in range(0, len(schedule), num_train_steps):
                corrupted = intermediates[step_idx]
                sigma = schedule[step_idx]

                # Forward pass: student predicts denoising
                student_logits = self.student_model.forward(
                    corrupted, step=sigma.item(), return_logits=True
                )

                # Teacher prediction for mode-seeking target
                teacher_logits = self.kl_distiller.compute_teacher_logits(
                    corrupted, clean_tokens, sigma.item()
                )

                # Reverse-KL mode-seeking loss
                kl_loss = self.kl_distiller.compute_mode_seeking_loss(
                    student_logits, teacher_logits, clean_tokens
                )

                # Apply path-consistency weighting
                weighted_loss = self.path_weighting.apply_path_consistency_loss(
                    kl_loss.view(-1), path_weights
                )

                # Backward pass
                self.optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), 1.0
                )
                self.optimizer.step()

                total_loss += weighted_loss.item()
                num_samples += 1

        avg_loss = total_loss / max(1, num_samples)

        return {'loss': avg_loss}
```

### Step 5: Few-step inference and evaluation

```python
def generate_few_step(
    student_model,
    prompt: str,
    num_steps: int = 4,
    schedule: str = 'cosine'
) -> torch.Tensor:
    """
    Generate with few-step student model.

    Args:
        student_model: Distilled few-step model
        prompt: Input prompt
        num_steps: Number of generation steps (2-4)
        schedule: Corruption schedule

    Returns:
        generated_tokens: Final prediction
    """
    # Get schedule
    if schedule == 'cosine':
        sigmas = torch.linspace(1.0, 0.0, num_steps)
    else:
        sigmas = torch.linspace(1.0, 0.0, num_steps)

    # Initialize with full masking
    seq_len = 128  # Default
    corrupted = torch.full((seq_len,), student_model.mask_token_id)

    for step_idx, sigma in enumerate(sigmas):
        # Denoise at current corruption level
        logits = student_model.forward(corrupted, step=sigma.item(), return_logits=True)
        probs = F.softmax(logits, dim=-1)
        corrupted = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return corrupted


def evaluate_few_step_quality(
    student_model,
    teacher_model,
    test_prompts: List[str],
    verifier,
    num_steps: int = 4
) -> Dict[str, float]:
    """Benchmark few-step vs full student performance."""
    few_step_scores = []
    full_step_scores = []

    for prompt in test_prompts:
        # Few-step
        few_tokens = generate_few_step(student_model, prompt, num_steps=num_steps)
        few_text = student_model.decode(few_tokens)
        few_score = verifier(few_text)

        # Full-step (teacher)
        full_tokens = teacher_model.generate(prompt, num_steps=50)
        full_text = teacher_model.decode(full_tokens)
        full_score = verifier(full_text)

        few_step_scores.append(few_score)
        full_step_scores.append(full_score)

    return {
        'few_step_accuracy': sum(few_step_scores) / len(few_step_scores),
        'full_step_accuracy': sum(full_step_scores) / len(full_step_scores),
        'speedup': 50 / num_steps  # Approximate steps reduction
    }
```

## Practical Guidance

**When to use**: Diffusion language models needing deployment-ready few-step variants; trading some quality for 10-15× speedup

**Hyperparameters**:
- **num_student_steps**: 2-4 (speedup-quality tradeoff)
- **temperature**: 1.0-1.5 (higher = softer distributions)
- **path_consistency_order**: 'confidence' (early predictions matter more)
- **num_collection_trajectories**: 4-8 per batch

**Key advantages**:
- Addresses train-test mismatch via trajectory collection
- Mode-seeking prevents spread across multiple outputs
- Path-consistency weighting prioritizes early predictions
- Maintains quality under aggressive step reduction

**Common pitfalls**:
- Reverse-KL can underflow on very multimodal tasks
- Path weights too extreme → ignores later predictions
- Too few trajectories → limited training signal
- Not validating that early predictions actually cascade errors

**Scaling**: Linear in number of trajectories collected.

## Reference

Paper: https://arxiv.org/abs/2602.12262
Related work: Diffusion models, distillation, few-step inference, mode-seeking losses
Benchmarks: Text generation, code synthesis, reasoning tasks
