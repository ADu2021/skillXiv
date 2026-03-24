---
name: looped-elastic-depth-transformers
title: "LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.11451"
keywords: [Transformer Architecture, Elastic Depth, Looped Processing, Budget-Conditioned Reasoning, Self-Distillation]
description: "Enable budget-conditioned reasoning by repeatedly applying a shared transformer block stack with trajectory-based conditioning on time and step size. Train via shortcut-consistency loss to align shorter and full-length trajectories, enabling variable-depth inference without retraining."
---

# LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning

## Problem Context

Transformers with fixed depth have fixed computation. Extended reasoning requires more depth, but retraining is expensive. LoopFormer enables variable-depth inference by repeatedly applying the same transformer blocks, with trajectories conditioned on normalized time (0 to 1 over steps) and step size. Shorter trajectories remain informative while longer ones refine.

## Core Concept

LoopFormer uses: (1) a shared stack of K transformer blocks applied repeatedly, (2) trajectory-based conditioning via time t ∈ [0,1] and step size Δt indicating position in loop, (3) shortcut-consistency training that aligns trajectories of different lengths, (4) AdaLN-style modulation of attention/FFN residual strengths.

## Implementation

### Step 1: Trajectory-based conditioning

```python
import torch
import torch.nn as nn
import math
from typing import Tuple

class TrajectoryConditioner:
    """Encode time and step size for loop iteration."""

    def __init__(self, dim: int = 768):
        self.dim = dim

    def encode_trajectory_position(
        self,
        iteration_idx: int,
        total_iterations: int,
        step_size: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode normalized time t and step size Δt.

        Args:
            iteration_idx: Current loop iteration (0-based)
            total_iterations: Total number of iterations
            step_size: Size of current step (fraction of sequence)

        Returns:
            (time_embedding, step_embedding): [dim] tensors
        """
        # Normalized time: 0 to 1 over course of all iterations
        t = iteration_idx / max(1, total_iterations - 1)

        # Step size (default: uniform 1/total_iterations)
        if step_size is None:
            step_size = 1.0 / total_iterations

        # Sine-cosine positional encoding
        time_embedding = self._sinusoidal_encode(t)
        step_embedding = self._sinusoidal_encode(step_size)

        return time_embedding, step_embedding

    def _sinusoidal_encode(self, value: float) -> torch.Tensor:
        """Sine-cosine encoding of scalar value."""
        embedding = torch.zeros(self.dim)

        for i in range(0, self.dim, 2):
            omega = 1 / (10000 ** (i / self.dim))
            if i < self.dim:
                embedding[i] = math.sin(value * omega)
            if i + 1 < self.dim:
                embedding[i + 1] = math.cos(value * omega)

        return embedding

    def condition_via_adaLN(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, dim]
        time_embedding: torch.Tensor,
        step_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply trajectory conditioning via learned modulation (AdaLN-style).

        Learns: scale and bias that depend on time and step.
        """
        combined = time_embedding + step_embedding  # [dim]

        # Learned projection: embedding -> (scale, bias)
        # Would be part of model parameters
        scale = 1.0 + 0.1 * combined.mean()
        bias = 0.05 * combined.mean()

        return hidden_states * scale + bias
```

### Step 2: Looped transformer with shared blocks

```python
class LoopedTransformer(nn.Module):
    """Transformer with shared block stack applied repeatedly."""

    def __init__(
        self,
        dim: int = 768,
        num_blocks: int = 6,
        num_heads: int = 12,
        max_loops: int = 16
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.max_loops = max_loops

        # Shared transformer blocks (reused across loops)
        self.transformer_blocks = nn.ModuleList([
            self._create_transformer_block(dim, num_heads)
            for _ in range(num_blocks)
        ])

        self.conditioner = TrajectoryConditioner(dim)

    def _create_transformer_block(
        self,
        dim: int,
        num_heads: int
    ) -> nn.Module:
        """Create single transformer block with gating."""
        class GatedTransformerBlock(nn.Module):
            def __init__(self, dim, num_heads):
                super().__init__()
                self.attn = nn.MultiheadAttention(dim, num_heads)
                self.ffn = nn.Sequential(
                    nn.Linear(dim, 4 * dim),
                    nn.GELU(),
                    nn.Linear(4 * dim, dim)
                )
                # Learned gates for residual connections
                self.attn_gate = nn.Parameter(torch.ones(1))
                self.ffn_gate = nn.Parameter(torch.ones(1))

            def forward(self, x):
                # Gated residuals
                attn_out, _ = self.attn(x, x, x)
                x = x + self.attn_gate * attn_out
                ffn_out = self.ffn(x)
                x = x + self.ffn_gate * ffn_out
                return x

        return GatedTransformerBlock(dim, num_heads)

    def forward_looped(
        self,
        embeddings: torch.Tensor,  # [batch, seq_len, dim]
        num_loops: int,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, list]:
        """
        Apply transformer blocks repeatedly in loops.

        Args:
            embeddings: Input embeddings
            num_loops: Number of times to apply block stack
            return_intermediate: Return all intermediate states

        Returns:
            (final_embeddings, intermediate_states)
        """
        current_state = embeddings
        intermediates = [current_state.clone()]

        for loop_idx in range(num_loops):
            # Trajectory conditioning
            time_emb, step_emb = self.conditioner.encode_trajectory_position(
                loop_idx, num_loops
            )

            # Apply all blocks in stack
            for block_idx, block in enumerate(self.transformer_blocks):
                current_state = block(current_state)

            # Apply trajectory conditioning (modulate residuals for this loop)
            current_state = self.conditioner.condition_via_adaLN(
                current_state, time_emb, step_emb
            )

            if return_intermediate:
                intermediates.append(current_state.clone())

        return current_state, intermediates if return_intermediate else []
```

### Step 3: Shortcut and consistency loss

```python
class ShortcutConsistencyTraining:
    """Train via multi-length trajectories with consistency."""

    def __init__(self, model: LoopedTransformer):
        self.model = model

    def compute_consistency_loss(
        self,
        embeddings: torch.Tensor,
        full_loops: int = 16,
        shortcut_loops: int = 4
    ) -> torch.Tensor:
        """
        Compute loss aligning short and full trajectories.

        Strategy: Full trajectory as teacher, short trajectory as student.
        """
        # Full trajectory (teacher)
        full_output, _ = self.model.forward_looped(embeddings, full_loops)

        # Shortcut trajectory (student, fewer loops)
        short_output, _ = self.model.forward_looped(embeddings, shortcut_loops)

        # Consistency loss: MSE between outputs
        consistency_loss = torch.nn.functional.mse_loss(short_output, full_output)

        return consistency_loss

    def compute_multi_trajectory_loss(
        self,
        embeddings: torch.Tensor,
        target: torch.Tensor,
        max_loops: int = 16,
        sample_loops: list = None
    ) -> torch.Tensor:
        """
        Train on multiple trajectory lengths simultaneously.

        Args:
            sample_loops: Which loop lengths to train on
                         (default: [4, 8, 12, 16])
        """
        if sample_loops is None:
            sample_loops = [4, 8, 12, 16]

        total_loss = torch.tensor(0.0, requires_grad=True)

        # Main LM loss on full trajectory
        full_output, _ = self.model.forward_looped(embeddings, max_loops)
        lm_loss = torch.nn.functional.cross_entropy(
            full_output.view(-1, full_output.size(-1)),
            target.view(-1)
        )
        total_loss = total_loss + lm_loss

        # Shortcut loss: train shorter trajectories to match full
        for num_loops in sample_loops:
            if num_loops >= max_loops:
                continue

            short_output, _ = self.model.forward_looped(embeddings, num_loops)

            # Stop-gradient on full output (don't update via short trajectory)
            shortcut_loss = torch.nn.functional.mse_loss(
                short_output,
                full_output.detach()
            )
            total_loss = total_loss + 0.1 * shortcut_loss

        return total_loss
```

### Step 4: Inference with elastic depth

```python
class ElasticDepthInference:
    """Perform inference with variable computational budget."""

    def __init__(self, model: LoopedTransformer):
        self.model = model

    def generate_with_budget(
        self,
        prompt_embeddings: torch.Tensor,
        budget_loops: int,
        schedule: str = 'linear'
    ) -> torch.Tensor:
        """
        Generate with specified loop budget (variable depth).

        Args:
            budget_loops: Number of loops available
            schedule: How to allocate loops ('linear', 'exponential')

        Returns:
            output: Generated embeddings
        """
        if schedule == 'linear':
            num_loops = budget_loops
        elif schedule == 'exponential':
            # Early layers get fewer loops; later layers refine
            num_loops = min(budget_loops, int(math.sqrt(budget_loops)) * 4)
        else:
            num_loops = budget_loops

        output, _ = self.model.forward_looped(prompt_embeddings, num_loops)

        return output

    def benchmark_depth_vs_quality(
        self,
        test_embeddings: torch.Tensor,
        target: torch.Tensor,
        max_budget: int = 16
    ) -> dict:
        """Profile quality vs. computational budget."""
        results = {}

        for budget in range(1, max_budget + 1):
            output = self.generate_with_budget(test_embeddings, budget)
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.size(-1)),
                target.view(-1)
            )
            results[budget] = {
                'loss': loss.item(),
                'computation': budget * len(self.model.transformer_blocks)
            }

        return results
```

### Step 5: Full training loop

```python
def train_loopformer(
    model: LoopedTransformer,
    train_loader,
    optimizer,
    num_epochs: int = 10,
    full_loops: int = 16,
    shortcut_loops_list: list = None,
    device: str = 'cuda'
):
    """
    Train LoopFormer with shortcut-consistency.

    Args:
        shortcut_loops_list: Which shortcut depths to train on
    """
    if shortcut_loops_list is None:
        shortcut_loops_list = [4, 8, 12]

    consistency_trainer = ShortcutConsistencyTraining(model)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (embeddings, target) in enumerate(train_loader):
            embeddings = embeddings.to(device)
            target = target.to(device)

            # Multi-trajectory loss
            loss = consistency_trainer.compute_multi_trajectory_loss(
                embeddings, target,
                max_loops=full_loops,
                sample_loops=shortcut_loops_list
            )

            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")

        # Validate at different depths
        if (epoch + 1) % 5 == 0:
            print("  Depth vs Quality:")
            inference = ElasticDepthInference(model)
            results = inference.benchmark_depth_vs_quality(
                embeddings, target, max_budget=full_loops
            )
            for budget, metrics in results.items():
                print(f"    {budget} loops: loss={metrics['loss']:.4f}")

    return model
```

## Practical Guidance

**When to use**: Variable-length reasoning; inference with heterogeneous compute budgets; amortized reasoning

**Hyperparameters**:
- **num_blocks**: 4-8 (shared blocks)
- **max_loops**: 12-20 (maximum depth)
- **shortcut_loops**: [4, 8, 12] (training depths)
- **shortcut_loss_weight**: 0.05-0.2 (balance main vs. consistency)
- **num_heads**: 8-16

**Key advantages**:
- Variable-depth inference without retraining
- Shorter trajectories remain informative
- Shared parameters reduce memory
- Smooth quality scaling with loops

**Common pitfalls**:
- shortcut_loss too strong → longer trajectories don't improve
- Max loops too large → diminishing returns, instability
- Not validating intermediate trajectories actually work
- Skip-consistency not tight enough

**Scaling**: Linear in num_loops; shared blocks reduce memory vs. fixed-depth stacks.

## Reference

Paper: https://arxiv.org/abs/2602.11451
Related work: Elastic networks, adaptive computation, trajectory-based conditioning
Benchmarks: Long reasoning sequences, language modeling
