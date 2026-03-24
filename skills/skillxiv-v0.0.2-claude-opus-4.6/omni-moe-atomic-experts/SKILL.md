---
name: omni-moe-atomic-experts
title: "OmniMoE: Efficient MoE by Orchestrating Atomic Experts"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.05711"
keywords: [Mixture-of-Experts, Atomic Experts, Routing, System-Algorithm Codesign, Efficiency]
description: "Scale mixture-of-experts models efficiently by decomposing experts into atomic vector pairs with Cartesian product routing and expert-centric scheduling. Achieves 10.9× speedup and 50% fewer parameters versus fine-grained baselines through system-algorithm codesign that converts scattered memory access into contiguous batched operations."
---

# OmniMoE: Atomic Expert Orchestration for Efficient Scaling

Scaling mixture-of-experts (MoE) systems to millions of experts requires solving two orthogonal challenges: routing complexity and hardware efficiency. Standard MoE scales linearly with expert count, requiring O(N) per-token routing computations for N experts. Additionally, dynamic expert selection creates memory-bound random access patterns that underutilize modern accelerators designed for dense matrix operations.

OmniMoE solves both problems through system-algorithm codesign. Instead of storing full expert networks, experts reduce to minimal vector pairs (atomic form). Routing decomposes the 1D expert space into 2D grids, reducing complexity from O(N) to O(√N). Critically, execution reorders computation from token-centric (fetch different experts per token) to expert-centric (group requests targeting the same experts), converting scattered lookups into contiguous memory access and high-throughput matrix operations.

## Core Concept

Traditional MoE for each token computes routing scores for all N experts, selects top-k, then fetches their parameters. This creates bottlenecks:

1. **Routing bottleneck**: O(N) score computations per token
2. **Memory bottleneck**: Each token fetches scattered expert parameters, underutilizing hardware

OmniMoE addresses both:

**Atomic Experts**: Experts reduce to pairs of vectors (U, V) with dimensions [d_hidden, r] and [r, d_out]. A token selects experts and combines their contributions:
output = Σ(α_i · U_i · V_i^T · input)

This slashes parameter count—1M atomic experts require less storage than 10K dense experts.

**Cartesian Product Routing**: The expert index space decomposes: i ∈ [0, N) becomes (i₁, i₂) where i₁ ∈ [0, √N), i₂ ∈ [0, √N). Routing requires computing √N dimensional projections separately, reducing scoring to O(√N) per token.

**Expert-Centric Scheduling**: Instead of processing tokens independently, the system groups all tokens requesting expert (i₁, i₂) and computes them in a single batched GEMM. This inverts execution order and converts random access into sequential, cache-friendly operations.

## Architecture Overview

- **Atomic Expert Decomposition**: Store experts as vector pairs (U ∈ ℝ^[d×r], V ∈ ℝ^[r×d]) rather than full dense matrices
- **Factorized Routing**: Decompose N experts into √N × √N grid; compute two separate √N-dimensional projections
- **Two-stage Routing**: Stage 1 selects top-k₁ experts from first dimension, Stage 2 selects top-k₂ from second, yielding up to k₁·k₂ experts per token
- **Expert-Centric Execution**: Batch tokens by target expert and execute as fused GEMM operations
- **Dense Backbone**: Maintain a shared dense MLP backbone alongside expert pool for universal computation

## Implementation

The implementation requires three components: expert parameter storage, routing computation, and expert-centric scheduling.

First, initialize and store atomic experts efficiently:

```python
import torch
import torch.nn as nn

class AtomicExpertPool(nn.Module):
    """Stores millions of experts as U·V^T vector pairs."""

    def __init__(self, num_experts, d_hidden, d_out, rank=64):
        super().__init__()
        self.num_experts = num_experts
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.rank = rank

        # Store experts as vector pairs: shape [num_experts, d_hidden, rank] and [num_experts, rank, d_out]
        self.U = nn.Parameter(torch.randn(num_experts, d_hidden, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(num_experts, rank, d_out) * 0.01)

    def forward(self, expert_ids, input_tensor):
        """Apply selected experts to input via matrix multiplications."""
        # expert_ids: [batch, k] selected expert indices
        # input_tensor: [batch, d_hidden]
        batch_size, k = expert_ids.shape

        # Gather expert parameters
        U_selected = self.U[expert_ids]  # [batch, k, d_hidden, rank]
        V_selected = self.V[expert_ids]  # [batch, k, rank, d_out]

        # Compute expert output: U·V^T·input
        # intermediate: [batch, k, rank]
        intermediate = torch.einsum('bkdr,bd->bkr', U_selected, input_tensor)

        # output: [batch, k, d_out]
        output = torch.einsum('bkro,bkr->bko', V_selected, intermediate)

        return output
```

Next, implement Cartesian product routing:

```python
class CartesianProductRouter(nn.Module):
    """Routes to experts via 2D Cartesian product decomposition."""

    def __init__(self, d_hidden, sqrt_num_experts, rank=256):
        super().__init__()
        self.sqrt_num_experts = sqrt_num_experts

        # Two separate projections for decomposed routing
        self.router_1 = nn.Linear(d_hidden, sqrt_num_experts)
        self.router_2 = nn.Linear(d_hidden, sqrt_num_experts)

    def forward(self, hidden_states, k=2):
        """
        Route tokens to experts via 2D grid.
        Args:
            hidden_states: [batch, d_hidden]
            k: select top-k experts from each dimension (yields k^2 experts)
        Returns:
            expert_ids: [batch, k^2] expert indices
            routing_weights: [batch, k^2] routing weights
        """
        # Compute routing scores
        scores_1 = self.router_1(hidden_states)  # [batch, sqrt_N]
        scores_2 = self.router_2(hidden_states)  # [batch, sqrt_N]

        # Select top-k from each dimension
        weights_1, indices_1 = torch.topk(scores_1, k, dim=-1)  # [batch, k]
        weights_2, indices_2 = torch.topk(scores_2, k, dim=-1)  # [batch, k]

        # Normalize weights
        weights_1 = torch.softmax(weights_1, dim=-1)
        weights_2 = torch.softmax(weights_2, dim=-1)

        # Compute 2D Cartesian product of selected experts
        # expert_id = i1 * sqrt_N + i2
        expert_ids = indices_1.unsqueeze(2) * self.sqrt_num_experts + indices_2.unsqueeze(1)
        expert_ids = expert_ids.reshape(hidden_states.shape[0], -1)  # [batch, k^2]

        # Combine weights multiplicatively
        routing_weights = (weights_1.unsqueeze(2) * weights_2.unsqueeze(1)).reshape(hidden_states.shape[0], -1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        return expert_ids, routing_weights
```

Finally, integrate expert-centric scheduling into your forward pass:

```python
class OmniMoELayer(nn.Module):
    """MoE layer with expert-centric scheduling."""

    def __init__(self, d_hidden, num_experts, rank=64):
        super().__init__()
        self.d_hidden = d_hidden
        self.sqrt_num_experts = int(num_experts ** 0.5)

        self.router = CartesianProductRouter(d_hidden, self.sqrt_num_experts)
        self.expert_pool = AtomicExpertPool(num_experts, d_hidden, d_hidden, rank)
        self.dense_backbone = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 4),
            nn.ReLU(),
            nn.Linear(d_hidden * 4, d_hidden)
        )

    def forward(self, hidden_states):
        """
        Forward pass with expert-centric scheduling.
        Groups tokens by target expert for efficient batching.
        """
        batch_size = hidden_states.shape[0]

        # Route: get expert assignments
        expert_ids, weights = self.router(hidden_states, k=2)  # [batch, k^2]

        # Expert-centric scheduling: group tokens by target expert
        output = torch.zeros_like(hidden_states)

        # Iterate over expert groups and process in batches
        for expert_id in range(self.sqrt_num_experts ** 2):
            mask = (expert_ids == expert_id).any(dim=-1)
            if not mask.any():
                continue

            # Gather tokens targeting this expert
            token_indices = torch.where(mask)[0]
            expert_input = hidden_states[token_indices]

            # Apply expert (single GEMM with batch of tokens)
            expert_out = torch.matmul(expert_input, self.expert_pool.U[expert_id])
            expert_out = torch.matmul(expert_out, self.expert_pool.V[expert_id])

            # Weight by routing probability
            expert_weight = weights[token_indices, (expert_ids[token_indices] == expert_id).nonzero(as_tuple=True)[1]]
            expert_out = expert_out * expert_weight.unsqueeze(-1)

            output[token_indices] += expert_out

        # Add dense backbone contribution
        output = output + self.dense_backbone(hidden_states)

        return output
```

## Practical Guidance

| Component | Recommendation | Notes |
|-----------|-----------------|-------|
| Expert rank | 64-256 | Lower rank reduces parameters; higher rank increases expressiveness. Tune per model size. |
| Cartesian grid size | √N where N=1M-10M | Balances routing complexity (O(√N)) with expert granularity. |
| Top-k per dimension | 2-4 | Higher k reduces sparsity but increases routing cost. k=2 yields 4-16 total experts per token. |
| Dense backbone ratio | 10-30% of MoE capacity | Shared dense layer provides universal computation; rest comes from sparse experts. |

**When to Use**
- Scaling to 1M+ experts on consumer hardware (GPUs/TPUs with < 100GB memory)
- Tasks requiring extreme parameter efficiency without sacrificing model capacity
- Language modeling or multimodal architectures where sparsity matters
- Training efficiency is a primary constraint

**When NOT to Use**
- Models with < 1M total parameters where dense approaches suffice
- Tasks requiring dense computation throughout (fully dense preferred)
- If atomic decomposition (rank < d_out) causes unacceptable expressiveness loss

**Common Pitfalls**
- Rank too low causes expert bottleneck—verify expressiveness on validation set
- Forgetting dense backbone removes universal computation path, hurts performance
- Not batching expert-centric execution defeats the purpose; always group by expert

## Reference

See https://arxiv.org/abs/2602.05711 for the full system-algorithm codesign paper, which includes detailed complexity analysis, memory profiling on real hardware, and benchmarks on standard language modeling and multimodal tasks.
