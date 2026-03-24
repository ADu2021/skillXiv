---
name: kinetics-test-time-scaling
title: "Kinetics: Rethinking Test-Time Scaling Laws"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.05333"
keywords: [scaling-laws, inference-optimization, attention-cost, sparse-attention, reasoning-efficiency]
description: "Reveals that memory bandwidth—not computation—dominates test-time costs; proposes eFLOPs metric incorporating both computation and memory, showing 14B+ minimum threshold for reasoning value."
---

# Kinetics: Memory-Aware Test-Time Scaling

## Core Concept

Conventional test-time scaling analysis focuses on computational FLOPs, but actual inference bottlenecks are memory bandwidth constraints. Kinetics introduces eFLOPs (effective FLOPs) that combine computational and memory access costs, revealing fundamental truths: smaller models are less valuable than assumed, a ~14B parameter minimum threshold exists before scaling strategies help, and attention—not parameter count—dominates cost in extended reasoning. Sparse attention emerges as the key complementary approach, enabling 60+ point gains on mathematical reasoning without expensive dense attention.

## Architecture Overview

- **eFLOPs Metric**: Cost model incorporating both computation (FLOPs) and memory bandwidth constraints
- **Memory-Aware Analysis**: Shows KV cache access costs dominate parameter costs during generation
- **Scaling Law Revision**: Demonstrates smaller models (<14B) rarely benefit from test-time scaling
- **Sparse Attention Paradigm**: Reduces quadratic attention cost via block-level top-k sparsity
- **Block Top-k Implementation**: Practical sparse attention variant achieving 11-26× speedup on H200 GPUs
- **Cost Threshold Framework**: Guides model selection and scaling strategy for given compute budgets

## Implementation

The following code demonstrates eFLOPs cost analysis and sparse attention:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np

class EFLOPsCostModel:
    """
    Effective FLOPs cost model incorporating both computation and memory.
    """
    def __init__(self, gpu_peak_flops: float = 1.5e15,  # H200
                 memory_bandwidth_gb_s: float = 4.8):    # H200
        self.peak_flops = gpu_peak_flops
        self.bandwidth_bytes_s = memory_bandwidth_gb_s * (1024 ** 3)

    def compute_dense_attention_cost(self, seq_len: int, hidden_dim: int,
                                    num_heads: int = 32) -> Tuple[float, float, float]:
        """
        Compute computation and memory costs for dense attention.

        Returns: (computation_flops, memory_bytes, total_eflops)
        """
        # Attention computation: Q @ K^T -> (seq_len, seq_len) -> @ V
        # FLOPs: 2 * seq_len^2 * hidden_dim + seq_len^2 operations
        compute_flops = 2 * seq_len * seq_len * hidden_dim

        # Memory access: load Q, K, V, store output + intermediate (attention matrix)
        # Each is seq_len * hidden_dim
        q_bytes = seq_len * hidden_dim * 4
        k_bytes = seq_len * hidden_dim * 4
        v_bytes = seq_len * hidden_dim * 4
        attn_matrix_bytes = seq_len * seq_len * 4  # Attention scores
        output_bytes = seq_len * hidden_dim * 4

        total_memory_bytes = q_bytes + k_bytes + v_bytes + attn_matrix_bytes + output_bytes

        # Convert to eFLOPs: treat memory access as FLOPs
        # ~1 FLOP per byte accessed (conservative estimate)
        memory_flops = total_memory_bytes

        total_eflops = compute_flops + memory_flops

        return compute_flops, total_memory_bytes, total_eflops

    def compute_sparse_attention_cost(self, seq_len: int, hidden_dim: int,
                                     sparsity_ratio: float = 0.1) -> Tuple[float, float, float]:
        """
        Compute cost for sparse (block top-k) attention.

        sparsity_ratio: fraction of attention matrix retained (e.g., 0.1 = 10% sparse)
        """
        # Computation: still need to compute full attention scores to select top-k
        # Then only compute values with selected entries
        compute_flops = 2 * seq_len * seq_len * hidden_dim * sparsity_ratio

        # Memory: reduced by sparsity
        q_bytes = seq_len * hidden_dim * 4
        k_bytes = seq_len * hidden_dim * 4
        v_bytes = seq_len * hidden_dim * 4
        attn_matrix_bytes = seq_len * seq_len * 4 * sparsity_ratio
        output_bytes = seq_len * hidden_dim * 4

        total_memory_bytes = q_bytes + k_bytes + v_bytes + attn_matrix_bytes + output_bytes

        memory_flops = total_memory_bytes
        total_eflops = compute_flops + memory_flops

        return compute_flops, total_memory_bytes, total_eflops

    def estimate_generation_latency(self, seq_len: int, hidden_dim: int,
                                   num_layers: int, dense: bool = True) -> float:
        """Estimate generation latency given hardware characteristics."""
        if dense:
            _, _, eflops = self.compute_dense_attention_cost(seq_len, hidden_dim)
        else:
            _, _, eflops = self.compute_sparse_attention_cost(seq_len, hidden_dim)

        # Assume attention dominates; scale by number of layers
        total_eflops = eflops * num_layers

        # Latency = total_eflops / peak_throughput
        # Assume memory-bound: bandwidth limited
        estimated_latency = total_eflops / self.bandwidth_bytes_s

        return estimated_latency

    def scaling_law_threshold(self, compute_budget_flops: float) -> int:
        """
        Compute minimum model size where test-time scaling becomes worthwhile.

        Based on empirical finding: ~14B minimum threshold.
        """
        # Rule of thumb from paper: ~14B is the knee in scaling law
        # Adjust based on compute budget
        return max(7, int(14 * (compute_budget_flops / 1e14)))


class BlockTopKAttention(nn.Module):
    """
    Sparse attention using block-level top-k selection.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 32, block_size: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, top_k_ratio: float = 0.1) -> torch.Tensor:
        """
        Sparse attention with block top-k selection.

        x: (batch, seq_len, hidden_dim)
        top_k_ratio: fraction of attention matrix to keep (e.g., 0.1 = 10%)
        """
        batch_size, seq_len, _ = x.shape

        # Standard projection
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Block-level top-k selection
        # Divide sequence into blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        block_scores = scores.view(batch_size, self.num_heads, num_blocks,
                                   self.block_size, seq_len)

        # Compute mean score per block
        block_means = block_scores.mean(dim=3)  # (batch, num_heads, num_blocks, seq_len)

        # Select top-k blocks
        k = max(1, int(seq_len * top_k_ratio))
        _, top_block_indices = torch.topk(block_means, k=k, dim=3)

        # Create sparse mask
        sparse_mask = torch.zeros_like(scores)
        for b in range(batch_size):
            for h in range(self.num_heads):
                for query_block in range(num_blocks):
                    query_start = query_block * self.block_size
                    query_end = min(query_start + self.block_size, seq_len)

                    for i in range(query_start, query_end):
                        top_indices = top_block_indices[b, h, query_block]
                        for top_idx in top_indices:
                            sparse_mask[b, h, i, top_idx] = 1.0

        # Apply sparse mask
        scores = scores * sparse_mask + (1 - sparse_mask) * (-1e9)

        # Attention
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)  # (batch, num_heads, seq_len, head_dim)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        return output


class KineticsScalingOptimizer:
    """
    Guides model selection and scaling strategy based on compute budget.
    """
    def __init__(self, cost_model: EFLOPsCostModel):
        self.cost_model = cost_model

    def recommend_model_size(self, compute_budget_flops: float) -> int:
        """Recommend model size given compute budget."""
        threshold = self.cost_model.scaling_law_threshold(compute_budget_flops)
        return threshold

    def should_use_sparse_attention(self, seq_len: int, model_size: int,
                                   cost_budget_flops: float) -> bool:
        """Determine if sparse attention is worthwhile."""
        # Sparse attention becomes valuable when:
        # 1. Attention is significant cost (long sequences)
        # 2. Model is small enough to fit with sparse KV

        dense_cost = self.cost_model.compute_dense_attention_cost(seq_len, 4096)[2]
        sparse_cost = self.cost_model.compute_sparse_attention_cost(seq_len, 4096, 0.1)[2]

        # Use sparse if it saves 50%+ cost
        return (dense_cost - sparse_cost) / dense_cost > 0.5

    def estimate_reasoning_gains(self, model_size: int,
                                num_reasoning_tokens: int,
                                sparse_attention: bool = False) -> float:
        """Estimate accuracy improvement from extended reasoning."""
        # Empirical: 60+ points gain on MATH with sparse attention
        if not sparse_attention or model_size < 14:
            return 0.0

        # Scaling curve: diminishing returns with more tokens
        log_tokens = np.log(num_reasoning_tokens + 1)
        gains = 60 * (1 - np.exp(-log_tokens / 10))  # Saturation curve

        return gains
```

## Practical Guidance

**eFLOPs Computation**: Always account for memory bandwidth in cost analysis. Modern GPUs are memory-bound for sequence operations; dense attention on H200 achieves <10% peak FLOP utilization due to memory limitations.

**Model Size Threshold**: The 14B minimum threshold is empirical but generalizable. Below 14B, test-time scaling provides minimal benefit relative to inference cost. Budget-conscious deployments should use sparse attention instead.

**Sparse Attention Tuning**: Block size of 64 tokens provides good granularity. Top-k ratio of 0.1 (keeping 10% of attention matrix) is typical; adjust based on accuracy-latency tradeoff.

**Sequence Length Scheduling**: Sparse attention becomes increasingly valuable as sequences grow. Below 4K tokens, dense may be comparable; above 16K, sparse is strongly preferred.

**Benchmarking Strategy**: Test both dense and sparse variants on your specific hardware. H200 benefits more from sparse due to higher memory-computation ratio.

**Hybrid Approaches**: Consider dense attention for early layers (building representations) and sparse for later layers (refinement). This balances accuracy with efficiency.

## Reference

Kinetics achieves strong efficiency improvements:
- **Block Top-k**: 11-26× speedup on H200 GPUs
- **Sparse attention gains**: 60+ points on MATH-related tasks
- **Memory-aware insight**: Bandwidth, not computation, is the primary bottleneck
- **Scaling threshold**: Minimum 14B parameter models for effective test-time scaling

This framework is particularly valuable for reasoning-intensive applications where extended generation is necessary but computational budgets are constrained. The memory-aware perspective corrects widespread misconceptions about scaling benefits for smaller models.
