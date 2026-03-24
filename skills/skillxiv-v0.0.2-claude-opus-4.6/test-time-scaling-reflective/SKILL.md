---
name: test-time-scaling-reflective
title: "Test-Time Scaling with Reflective Generative Model"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.01951"
keywords: [Test-Time Compute, Process Reward Models, Self-Supervised Learning, Reasoning]
description: "Scale model performance at test time by generating multiple reasoning trajectories and selecting the best using a self-supervised process reward model. MetaStone-S1 achieves 32B-equivalent performance using only 32B parameters and 53M for trajectory scoring, learning process rewards from outcome labels alone without process annotations."
---

# Test-Time Scaling with Reflective Generative Models: More Compute at Inference

Inference-time compute is often cheaper than training-time compute, yet most models use fixed generation sequences. Test-time scaling generates multiple candidate trajectories and selects the highest-confidence path, dramatically improving reasoning performance without additional training. The challenge is scoring trajectories—process reward models typically require expensive per-step annotations. Reflective Generative Models solve this with a unified backbone that shares parameters between generation and evaluation, plus a self-supervised process reward model that learns trajectory scoring from only final-answer correctness.

The key insight is that a single shared backbone can simultaneously generate reasoning steps and evaluate them. By training a process reward model to distinguish correct from incorrect reasoning using only outcome labels, you eliminate annotation bottlenecks while achieving performance parity with o3-mini using 32B parameters instead of much larger models.

## Core Concept

Test-time scaling combines three components:

1. **Reflective Generation**: A single backbone generates reasoning trajectories while simultaneously serving as a process evaluator via additional scoring heads
2. **Self-Supervised Process Reward Model (SPRM)**: Learns to score reasoning steps using only binary outcome supervision (correct/incorrect final answer)
3. **Multi-Trajectory Selection**: Generate k trajectories at test time (k=2, 8, 32 for low/medium/high compute), select highest-scoring path using SPRM

The model reaches an "Aha Moment" during training where it transitions from treating all reasoning patterns identically to meaningfully discriminating good trajectories from bad ones, using only final answer signals.

## Architecture Overview

- **Shared Backbone**: LLM generating both reasoning tokens and trajectory scores (e.g., Llama-based 32B)
- **Generation Head**: Standard language modeling head for producing reasoning text
- **Scoring Head**: Additional output layer for step-level trajectory confidence (53M parameters)
- **Token-level Evaluator**: Processes trajectory tokens and outputs continuous scores per step
- **Self-Supervised Loss**: Combines binary cross-entropy (correct/incorrect outcome) with confidence weighting
- **Dynamic Filtering**: Removes noisy training samples during SPRM training based on score variance

## Implementation

The following demonstrates the unified reflective generation and self-supervised scoring:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ReflectiveGenerationHead(nn.Module):
    """Standard language generation head for reasoning trajectories."""
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_dim)
        logits = self.proj(hidden_states)
        return logits  # (batch, seq_len, vocab_size)

class ProcessRewardHead(nn.Module):
    """Lightweight trajectory scoring head (53M params)."""
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else 256
            out_dim = 256
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.LayerNorm(out_dim),
                nn.Dropout(0.1)
            ])

        # Final scalar output (trajectory score)
        layers.append(nn.Linear(256, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_dim)
        # Output step-level scores
        scores = self.mlp(hidden_states)  # (batch, seq_len, 1)
        return scores.squeeze(-1)  # (batch, seq_len)

class ReflectiveGenerativeModel(nn.Module):
    """Unified backbone for generation and evaluation."""
    def __init__(self, model_name: str = "meta-llama/Llama-2-32b",
                 hidden_dim: int = 4096, vocab_size: int = 32000):
        super().__init__()

        # Base LLM backbone (frozen or fine-tuned during training)
        self.backbone = None  # In practice, load from model_name
        self.hidden_dim = hidden_dim

        # Lightweight scoring head (53M parameters)
        self.process_reward_head = ProcessRewardHead(hidden_dim, num_layers=3)

        # Generation head
        self.generation_head = ReflectiveGenerationHead(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)
        # Returns: generation logits and trajectory scores

        # Backbone forward pass (e.g., from Llama)
        hidden_states = self.backbone(input_ids, attention_mask=attention_mask).hidden_states

        # Generation output
        generation_logits = self.generation_head(hidden_states)

        # Trajectory scoring output
        trajectory_scores = self.process_reward_head(hidden_states)

        return generation_logits, trajectory_scores

class SelfSupervisedProcessRewardModel:
    """Train process reward model from outcome labels alone (no per-step annotations)."""

    @staticmethod
    def compute_sprm_loss(trajectory_scores: torch.Tensor,
                          is_correct: torch.Tensor,
                          temperature: float = 1.0,
                          confidence_threshold: float = 0.2) -> torch.Tensor:
        """
        Self-supervised loss: learn trajectory quality from outcome correctness.

        Args:
            trajectory_scores: (batch, seq_len) predicted step-level scores
            is_correct: (batch,) binary labels (1=correct final answer, 0=incorrect)
            temperature: softmax temperature for score normalization
            confidence_threshold: filter low-confidence samples
        """
        batch_size, seq_len = trajectory_scores.shape

        # Trajectory-level score: average over sequence
        trajectory_level_scores = trajectory_scores.mean(dim=1)  # (batch,)

        # Confidence-based weighting: upweight high-variance samples
        score_variance = trajectory_scores.var(dim=1)  # (batch,)
        confidence_weights = torch.sigmoid(score_variance / temperature)

        # Only train on high-confidence samples
        mask = score_variance > confidence_threshold
        if mask.sum() == 0:
            mask = torch.ones_like(mask)  # Fallback: use all

        # Binary cross-entropy with confidence weighting
        bce_loss = F.binary_cross_entropy_with_logits(
            trajectory_level_scores[mask],
            is_correct[mask].float(),
            reduction='none'
        )

        # Weight by confidence
        weighted_loss = (bce_loss * confidence_weights[mask]).mean()

        return weighted_loss

def generate_multiple_trajectories(model: ReflectiveGenerativeModel,
                                   input_ids: torch.Tensor,
                                   num_trajectories: int = 8,
                                   max_length: int = 512,
                                   temperature: float = 1.0) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate k reasoning trajectories and score them.

    Args:
        model: Reflective generative model
        input_ids: Problem prompt (batch_size, seq_len)
        num_trajectories: Number of trajectories to generate per input
        max_length: Maximum trajectory length
        temperature: Sampling temperature (higher = more diverse)

    Returns:
        trajectories: List of (batch, seq_len) token sequences
        trajectory_scores: List of (batch,) scalar scores
    """
    trajectories = []
    trajectory_scores_list = []

    with torch.no_grad():
        for traj_idx in range(num_trajectories):
            current_ids = input_ids.clone()
            step_scores = []

            # Autoregressive generation with scoring
            for step in range(max_length):
                generation_logits, trajectory_scores = model(current_ids)

                # Sample next token (with temperature)
                next_token_logits = generation_logits[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)

                # Append token and its score
                current_ids = torch.cat([current_ids, next_token], dim=1)
                step_scores.append(trajectory_scores[:, -1].cpu())

            trajectories.append(current_ids)
            trajectory_scores_list.append(torch.stack(step_scores, dim=1))

    return trajectories, trajectory_scores_list

def select_best_trajectory(trajectories: List[torch.Tensor],
                          trajectory_scores: List[torch.Tensor]) -> torch.Tensor:
    """Select highest-scoring trajectory for each input in batch."""
    # trajectory_scores: List of (batch, seq_len)
    # Compute trajectory-level score (mean across steps)

    trajectory_level_scores = [scores.mean(dim=1) for scores in trajectory_scores]
    # Shape: (num_trajectories, batch)

    # Stack and find best trajectory per batch element
    all_scores = torch.stack(trajectory_level_scores, dim=0)  # (num_trajectories, batch)
    best_trajectory_indices = all_scores.argmax(dim=0)  # (batch,)

    # Select best trajectory for each batch element
    best_trajectories = []
    for batch_idx in range(trajectories[0].shape[0]):
        best_traj_idx = best_trajectory_indices[batch_idx].item()
        best_trajectories.append(trajectories[best_traj_idx][batch_idx])

    return torch.stack(best_trajectories, dim=0)

def train_reflective_model_step(model: ReflectiveGenerativeModel,
                               input_ids: torch.Tensor,
                               target_ids: torch.Tensor,
                               is_correct: torch.Tensor,
                               optimizer: torch.optim.Optimizer,
                               alpha: float = 0.3) -> Tuple[float, float]:
    """
    Single training step combining generation and process reward learning.

    Args:
        model: Reflective generative model
        input_ids: Problem prompts (batch, seq_len)
        target_ids: Correct reasoning trajectories (batch, target_len)
        is_correct: Outcome correctness (batch,)
        optimizer: Training optimizer
        alpha: Weighting of SPRM loss vs generation loss
    """
    optimizer.zero_grad()

    # Forward pass
    generation_logits, trajectory_scores = model(input_ids)

    # Generation loss (standard language modeling)
    gen_loss = F.cross_entropy(
        generation_logits[:, :-1].reshape(-1, generation_logits.shape[-1]),
        target_ids[:, 1:].reshape(-1)
    )

    # Self-supervised process reward loss
    sprm_loss = SelfSupervisedProcessRewardModel.compute_sprm_loss(
        trajectory_scores, is_correct
    )

    # Combined loss
    total_loss = gen_loss + alpha * sprm_loss

    total_loss.backward()
    optimizer.step()

    return gen_loss.item(), sprm_loss.item()
```

This implementation shows the core reflective architecture: shared backbone for generation and scoring, plus self-supervised trajectory learning from outcome labels.

## Practical Guidance

| Aspect | Recommended Value | Notes |
|--------|------------------|-------|
| **Trajectory Count (k)** | 2 (low), 8 (medium), 32 (high) | 8× compute multiplier per increment |
| **Temperature** | 0.8-1.2 for generation | Higher = more diverse trajectories |
| **SPRM Loss Weight (α)** | 0.3-0.5 | Balance generation vs. trajectory discrimination |
| **Confidence Threshold** | 0.2-0.4 (score variance) | Filter noisy samples during SPRM training |
| **Aha Moment Detection** | Monitor SPRM accuracy | Should jump 30-50% at transition phase |
| **Batch Size** | 128-256 during trajectory generation | Memory-intensive due to k×batch size tokens |

### When to Use Test-Time Scaling

- **Latency-tolerant inference**: Can afford 2-32× longer generation (seconds, not milliseconds)
- **Complex reasoning tasks**: Math, code generation, multi-step planning improve significantly
- **Cost-conscious scaling**: Better cost/accuracy tradeoff than training larger models
- **Adaptive compute**: Select k=2 for simple queries, k=32 for complex ones
- **Ensemble learning**: Multiple trajectories provide uncertainty estimates
- **Reward hacking defense**: Process rewards reduce vulnerability to superficial outputs

### When NOT to Use

- **Real-time interactive systems**: 32× generation slowdown unacceptable (>5 second latency)
- **Models already optimized for reasoning**: O1/O3 families already use advanced test-time scaling
- **Supervised process rewards available**: If you have per-step annotations, use supervised process rewards (higher quality)
- **Vocabulary <5K or >100K**: Self-supervised SPRM assumes diverse token distributions; extreme vocabularies cause training instability
- **Very short reasoning tasks** (1-2 steps): Trajectory diversity negligible; additional compute doesn't help

### Common Pitfalls

1. **Temperature Too Low**: Setting temperature <0.5 creates near-identical trajectories. Increase to 1.0-1.5 for diversity.
2. **Aha Moment Missed**: If SPRM loss doesn't show sharp accuracy transition, learning rate is too high (causes oscillation) or too low (model doesn't learn). Use lr=1e-4 with 10% warmup.
3. **Confidence Weighting Ineffective**: If you filter >50% of samples, threshold is too aggressive. Reduce confidence_threshold to 0.1.
4. **Trajectory Score Collapse**: All trajectories score similarly = model hasn't learned discrimination. Add auxiliary losses (e.g., contrastive learning between correct/incorrect pairs).
5. **Ignoring Output Token Limits**: Long-sequence generation with k=32 rapidly exhausts memory. Implement sliding-window scoring or token budget constraints.

## Reference

Ying, L., Wang, R., et al. (2025). Test-Time Scaling with Reflective Generative Model. *arXiv preprint arXiv:2507.01951*.

Available at: https://arxiv.org/abs/2507.01951
