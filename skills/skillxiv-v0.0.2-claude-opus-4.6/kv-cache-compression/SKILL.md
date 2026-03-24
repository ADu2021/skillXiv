---
name: kv-cache-compression
title: "Inference-Time Hyper-Scaling with KV Cache Compression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.05345"
keywords: [inference-optimization, memory-efficiency, kv-cache, scaling, reasoning]
description: "Enables 8x KV cache compression with minimal training overhead to improve reasoning accuracy by allowing more token generation within computational budgets."
---

# Inference-Time Hyper-Scaling with KV Cache Compression

## Core Concept

In transformer inference, the KV cache—storing key and value vectors for all previous tokens—becomes the primary memory bottleneck, often limiting reasoning performance more than computational capacity. Rather than generating tokens until memory exhaustion, KV cache compression allows strategic memory reduction to generate additional tokens within the same budget. Dynamic Memory Sparsification (DMS) achieves 8× compression with only 1,000 training steps, enabling inference-time hyper-scaling where extra tokens directly translate to improved reasoning accuracy.

## Architecture Overview

- **Dynamic Memory Sparsification (DMS)**: Compression technique that delays eviction of cached tokens, implicitly merging representations
- **Training Efficiency**: Requires minimal training overhead (1,000 steps) making adaptation practical
- **Inference-Time Strategy**: Uses freed memory budget to generate additional reasoning tokens
- **Memory-Bandwidth Focus**: Optimizes for actual inference bottleneck (memory, not computation)
- **Multi-Model Compatibility**: Applies across various LLM families without architectural changes
- **Reasoning Task Optimization**: Particularly effective for complex reasoning requiring extended token generation

## Implementation

The following code demonstrates the DMS algorithm and hyper-scaling strategy:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class DynamicMemorySparsification:
    """
    Dynamic Memory Sparsification for KV cache compression.
    """
    def __init__(self, compression_ratio: float = 0.125, merge_threshold: float = 0.5):
        self.compression_ratio = compression_ratio
        self.merge_threshold = merge_threshold

    def compute_token_importance(self, key_vectors: torch.Tensor,
                                query_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for cached tokens based on attention patterns.

        key_vectors: (seq_len, hidden_dim) cached KV vectors
        query_vectors: (num_heads, hidden_dim) current query
        Returns: (seq_len,) importance scores in [0, 1]
        """
        # Normalize vectors for cosine similarity
        keys_norm = F.normalize(key_vectors, p=2, dim=1)
        query_norm = F.normalize(query_vectors, p=2, dim=1)

        # Compute attention similarity as importance
        importance = torch.matmul(keys_norm, query_norm.t())  # (seq_len, num_heads)
        importance = importance.max(dim=1).values  # (seq_len,)

        # Normalize to [0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

        return importance

    def merge_kv_tokens(self, key_cache: torch.Tensor,
                       value_cache: torch.Tensor,
                       importance_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge low-importance tokens by delayed eviction and averaging.

        key_cache, value_cache: (seq_len, hidden_dim)
        importance_scores: (seq_len,) importance of each token
        Returns: compressed (key_cache, value_cache)
        """
        seq_len = key_cache.shape[0]
        target_len = max(1, int(seq_len * self.compression_ratio))

        # Identify tokens to keep (high importance) vs merge (low importance)
        _, keep_indices = torch.topk(importance_scores, target_len)
        keep_indices = torch.sort(keep_indices)[0]  # Maintain order

        # Create compression mapping: which tokens to merge into each kept token
        compress_to = torch.zeros(seq_len, dtype=torch.long, device=key_cache.device)

        # Assign each discarded token to nearest kept token
        for i in range(seq_len):
            if i in keep_indices:
                compress_to[i] = i
            else:
                # Find nearest kept token
                nearest = torch.argmin(torch.abs(keep_indices - i))
                compress_to[i] = keep_indices[nearest].item()

        # Merge values: average tokens that compress to same position
        key_merged = torch.zeros(target_len, key_cache.shape[1],
                                device=key_cache.device, dtype=key_cache.dtype)
        value_merged = torch.zeros(target_len, value_cache.shape[1],
                                  device=value_cache.device, dtype=value_cache.dtype)
        counts = torch.zeros(target_len, device=key_cache.device)

        for orig_idx, merged_idx in enumerate(compress_to):
            key_merged[merged_idx] += key_cache[orig_idx]
            value_merged[merged_idx] += value_cache[orig_idx]
            counts[merged_idx] += 1

        # Average merged tokens
        key_merged /= counts.unsqueeze(1).clamp(min=1)
        value_merged /= counts.unsqueeze(1).clamp(min=1)

        return key_merged, value_merged


class CompressedKVCacheLM(nn.Module):
    """
    Language model with compressed KV cache for extended generation.
    """
    def __init__(self, model_dim: int = 4096, num_heads: int = 32,
                 compression_ratio: float = 0.125):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Attention projections
        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        self.wo = nn.Linear(model_dim, model_dim)

        self.dms = DynamicMemorySparsification(compression_ratio=compression_ratio)

    def forward_attention(self, query: torch.Tensor,
                         key_cache: torch.Tensor,
                         value_cache: torch.Tensor,
                         compress: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Attention with optional KV cache compression.

        query: (batch, 1, model_dim) current token query
        key_cache, value_cache: (seq_len, model_dim) cached KV
        Returns: (output, compressed_key_cache, compressed_value_cache)
        """
        # Project current query
        Q = self.wq(query)  # (batch, 1, model_dim)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch, num_heads, 1, head_dim)

        # Project cached keys and values
        K = self.wk(key_cache)  # (seq_len, model_dim)
        V = self.wv(value_cache)

        K = K.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, seq_len, head_dim)
        V = V.view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        # Compute attention
        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, V)  # (batch, num_heads, 1, head_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.shape[0], output.shape[1], self.model_dim)
        output = self.wo(output)

        # Optionally compress cache
        if compress and key_cache.shape[0] > 512:
            importance = self.dms.compute_token_importance(key_cache,
                                                           self.wq(query).squeeze(1))
            key_cache, value_cache = self.dms.merge_kv_tokens(key_cache, value_cache, importance)

        return output, key_cache, value_cache


class InferenceTimeHyperScaling:
    """
    Strategy for leveraging freed KV cache memory for extended generation.
    """
    def __init__(self, model: nn.Module, total_budget_gb: float = 80.0):
        self.model = model
        self.total_budget_gb = total_budget_gb
        self.bytes_per_token = 2 * 4 * 4096  # fp32, 2 (K,V), model_dim

    def estimate_tokens_available(self, compression_ratio: float = 0.125) -> int:
        """Estimate how many tokens can be generated within memory budget."""
        available_bytes = self.total_budget_gb * (1024 ** 3)
        tokens_uncompressed = available_bytes / self.bytes_per_token

        # With compression, we can fit more
        tokens_compressed = tokens_uncompressed / compression_ratio

        return int(tokens_compressed)

    def generate_with_extended_horizon(self, prompt_ids: List[int],
                                       initial_max_tokens: int = 256,
                                       compression_enabled: bool = True) -> List[int]:
        """
        Generate tokens with extended horizon using freed KV cache memory.
        """
        tokens = prompt_ids.copy()

        # With compression, we can extend generation
        max_new_tokens = initial_max_tokens
        if compression_enabled:
            available = self.estimate_tokens_available()
            max_new_tokens = min(available, initial_max_tokens * 3)  # 3x extension typical

        # Simulate generation loop
        for _ in range(max_new_tokens):
            # In real implementation: forward pass, sample next token
            # For now, just show structure
            next_token = 1  # Placeholder
            tokens.append(next_token)

        return tokens
```

## Practical Guidance

**Training Schedule**: DMS requires only 1,000 training steps, making it practical to apply to frozen pre-trained models. Use a learning rate of 1e-4 to 1e-5 for stability.

**Compression Ratio Selection**: Start with 0.125 (8× compression). For accuracy-critical tasks, use 0.25 (4× compression); for speed-critical tasks, try 0.06 (16× compression).

**Token Importance Scoring**: Use the provided attention-based scoring, but consider task-specific variants: for code, prioritize recent tokens; for reasoning, prioritize intermediate results.

**Merged Token Representation**: When averaging K and V vectors during merging, use weighted averaging based on importance scores rather than uniform averaging for better fidelity.

**Memory Budget Planning**: Calculate total GPU VRAM available, subtract model weights and activations, then allocate remaining budget to KV cache. This determines max-achievable sequence length.

**Benchmark on Target Tasks**: DMS shows 12+ point improvements on AIME and GPQA. Benchmark on your specific reasoning tasks to validate improvements.

## Reference

DMS achieves strong empirical results on reasoning benchmarks:
- **Qwen-R1 32B on AIME 24**: +12.0 points improvement
- **GPQA**: +8.6 points
- **LiveCodeBench**: +9.7 points
- **Training Efficiency**: 1,000 steps on single GPU

The method is particularly valuable for reasoning-heavy applications where extended generation directly correlates with accuracy. By converting memory freed through compression into additional reasoning tokens, hyper-scaling enables better problem-solving within fixed computational budgets.
