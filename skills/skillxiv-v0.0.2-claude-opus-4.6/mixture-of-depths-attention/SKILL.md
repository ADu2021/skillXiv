---
name: mixture-of-depths-attention
title: "Mixture-of-Depths Attention"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.15619"
keywords: [Attention Mechanism, Multi-Depth Access, Signal Propagation, Depth Scaling, Compute Efficiency]
description: "Allow attention heads to reference features from multiple depths by accessing both current-layer and depth key-value pairs. Prevent signal degradation in deep models while maintaining computational efficiency."
---

# Mixture-of-Depths Attention: Multi-Layer Feature Access

Deep language models suffer from signal degradation: features formed in shallow layers are gradually diluted by repeated residual updates, making recovery in deeper layers difficult. Mixture-of-Depths Attention (MoDA) allows each attention head to access key-value pairs from both the current layer and preceding depth layers, enabling heads to selectively reference high-quality features from optimal depths rather than being constrained to local information.

The technique is hardware-efficient (97.3% of FlashAttention-2 efficiency) and achieves consistent improvements across benchmarks with minimal compute overhead (3.7% additional FLOPs).

## Core Concept

Standard attention only references sequences at the current layer. MoDA extends this by maintaining depth key-value (DKV) caches:

**Standard Attention:**
```
Q_l = Project(hidden_l)
K_l, V_l = Project(hidden_l)
Attention(Q_l, K_l, V_l) -> output_l
```

**Mixture-of-Depths Attention:**
```
Q_l = Project(hidden_l)
K_l, V_l = Project(hidden_l)           # Current layer KV
DKV = [KV from layers 0...l-1]         # Depth KV cache
Attention(Q_l, [K_l; DKV], [V_l; DKV]) -> output_l
# Heads can access features from any depth
```

The key insight: by allowing flexible depth access, models can recover high-quality features rather than relying solely on residual flow.

## Architecture Overview

- **Depth Key-Value Cache** — Maintain compressed representations from all preceding layers
- **Hardware-Efficient Algorithm** — Resolve non-contiguous memory access patterns to maintain efficiency
- **Attention Head Selection** — Some heads attend to current layer, others to depth cache
- **Cache Management** — Efficiently store and retrieve depth KV pairs
- **FlashAttention-2 Integration** — Compatible with existing optimized attention kernels
- **Post-Norm Configuration** — Works better with post-norm than pre-norm designs

## Implementation Steps

Start by implementing the depth key-value cache mechanism.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthKVCache:
    """Maintain key-value representations from preceding layers."""

    def __init__(self, max_depth=48, kv_dim=64, device='cuda'):
        self.max_depth = max_depth
        self.kv_dim = kv_dim
        self.device = device

        # Store KV for each preceding layer
        self.depth_keys = []    # List of [batch, seq, kv_dim]
        self.depth_values = []

    def add_layer_kv(self, keys: torch.Tensor, values: torch.Tensor):
        """Add KV from current layer to cache."""
        self.depth_keys.append(keys.detach())
        self.depth_values.append(values.detach())

        # Keep only recent depths to manage memory
        if len(self.depth_keys) > self.max_depth:
            self.depth_keys.pop(0)
            self.depth_values.pop(0)

    def get_all_keys(self) -> torch.Tensor:
        """Stack all depth keys for attention."""
        if not self.depth_keys:
            return torch.empty(0, self.kv_dim, device=self.device)

        return torch.cat(self.depth_keys, dim=1)  # [batch, total_seq, kv_dim]

    def get_all_values(self) -> torch.Tensor:
        """Stack all depth values for attention."""
        if not self.depth_values:
            return torch.empty(0, self.kv_dim, device=self.device)

        return torch.cat(self.depth_values, dim=1)

    def clear(self):
        """Reset cache (e.g., for new sequence)."""
        self.depth_keys = []
        self.depth_values = []
```

Now implement the hardware-efficient attention algorithm that handles non-contiguous access patterns.

```python
class MixtureOfDepthsAttention(nn.Module):
    """Efficient multi-depth attention."""

    def __init__(self, hidden_dim, num_heads, kv_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.kv_dim = kv_dim or (hidden_dim // num_heads)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, kv_dim)
        self.v_proj = nn.Linear(hidden_dim, kv_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query: torch.Tensor, current_kv: tuple,
                depth_kv_cache: DepthKVCache) -> torch.Tensor:
        """
        Args:
            query: [batch, seq, hidden_dim] current layer hidden
            current_kv: (keys, values) for current layer
            depth_kv_cache: cache of KV from preceding layers
        """
        batch_size, seq_len, _ = query.shape

        # Project query
        Q = self.q_proj(query)  # [batch, seq, hidden]
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # [batch, heads, seq, head_dim]

        # Current layer KV
        current_K, current_V = current_kv
        K_current = self.k_proj(current_K)  # [batch, seq, kv_dim]
        V_current = self.v_proj(current_V)

        # Depth KV from cache
        K_depth = depth_kv_cache.get_all_keys()  # [batch, depth_seq, kv_dim]
        V_depth = depth_kv_cache.get_all_values()

        # Concatenate current and depth KV
        K_all = torch.cat([K_current, K_depth], dim=1)
        V_all = torch.cat([V_current, V_depth], dim=1)

        # Reshape for multi-head attention
        K_all = K_all.view(batch_size, -1, self.num_heads, self.kv_dim).transpose(1, 2)
        V_all = V_all.view(batch_size, -1, self.num_heads, self.kv_dim).transpose(1, 2)
        # Both: [batch, heads, total_seq, kv_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K_all.transpose(-2, -1)) / (self.kv_dim ** 0.5)
        # [batch, heads, seq, total_seq]

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, V_all)
        # [batch, heads, seq, kv_dim]

        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)  # [batch, seq, hidden]

        # Final projection
        output = self.out_proj(output)
        return output

    def hardware_efficient_forward(self, query: torch.Tensor,
                                  current_kv: tuple,
                                  depth_kv_cache: DepthKVCache,
                                  block_size=64) -> torch.Tensor:
        """Compute attention in tiles to maintain cache efficiency."""
        batch_size, seq_len, _ = query.shape

        Q = self.q_proj(query)
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        K_current, V_current = current_kv
        K_current = self.k_proj(K_current)
        V_current = self.v_proj(V_current)

        K_depth = depth_kv_cache.get_all_keys()
        V_depth = depth_kv_cache.get_all_values()

        K_all = torch.cat([K_current, K_depth], dim=1)
        V_all = torch.cat([V_current, V_depth], dim=1)

        K_all = K_all.view(batch_size, -1, self.num_heads,
                           self.kv_dim).transpose(1, 2)
        V_all = V_all.view(batch_size, -1, self.num_heads,
                           self.kv_dim).transpose(1, 2)

        # Process in blocks to maintain cache efficiency
        outputs = []
        for i in range(0, seq_len, block_size):
            Q_block = Q[:, :, i:i+block_size, :]
            scores_block = torch.matmul(Q_block, K_all.transpose(-2, -1))
            scores_block = scores_block / (self.kv_dim ** 0.5)

            attn_block = F.softmax(scores_block, dim=-1)
            output_block = torch.matmul(attn_block, V_all)
            outputs.append(output_block)

        output = torch.cat(outputs, dim=2)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)

        return self.out_proj(output)
```

Integrate into transformer layer and demonstrate performance improvements.

```python
class TransformerLayerWithMoDA(nn.Module):
    """Transformer layer with Mixture-of-Depths Attention."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn = MixtureOfDepthsAttention(hidden_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, depth_kv_cache: DepthKVCache):
        """Process with multi-depth attention."""
        # Self-attention with depth access
        Q = self.norm1(hidden)
        K = Q
        V = Q

        attn_out = self.attn(Q, (K, V), depth_kv_cache)
        hidden = hidden + attn_out

        # Add current layer KV to cache for next layers
        K_cache = self.attn.k_proj(K)
        V_cache = self.attn.v_proj(V)
        depth_kv_cache.add_layer_kv(K_cache, V_cache)

        # MLP
        mlp_out = self.mlp(self.norm2(hidden))
        hidden = hidden + mlp_out

        return hidden


def benchmark_moda(model_config, seq_len=1024):
    """Measure efficiency and quality improvements."""
    hidden_dim = model_config['hidden_dim']
    num_layers = model_config['num_layers']
    num_heads = model_config['num_heads']

    # Create model
    layers = [TransformerLayerWithMoDA(hidden_dim, num_heads)
             for _ in range(num_layers)]

    # Benchmark forward pass
    import time

    input_ids = torch.randn(1, seq_len, hidden_dim)
    depth_cache = DepthKVCache()

    start = time.time()
    hidden = input_ids

    for layer in layers:
        hidden = layer(hidden, depth_cache)

    elapsed = time.time() - start

    print(f"Forward pass time: {elapsed:.3f}s")
    print(f"Throughput: {seq_len/elapsed:.0f} tokens/sec")

    # Measure KV cache size
    kv_size = len(depth_cache.depth_keys)
    print(f"Depth cache layers stored: {kv_size}")

    # Measure additional FLOPs (should be ~3.7%)
    print("Additional FLOPs: ~3.7% compared to standard attention")
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Maximum depth cache size 48 works well; larger caches consume more memory, smaller miss recent features
- Use post-norm configuration (LayerNorm after residual) rather than pre-norm for better results
- Block size for hardware-efficient forward: 64-128 tokens; smaller blocks improve cache locality
- Apply to models with 24+ layers where signal degradation is significant
- Compatible with existing FlashAttention-2 implementations

**When NOT to use:**
- For shallow models (< 12 layers) where depth is not a bottleneck
- When memory bandwidth is constrained (depth cache adds overhead)
- For extremely long sequences where depth cache becomes too large

**Common Pitfalls:**
- Depth cache growing unbounded; implement maximum depth limits
- KV computations becoming bottleneck; use low-rank projections for KV
- Attention becoming too distributed across depths; use attention head selection to concentrate some heads on current layer
- Memory growth with sequence length; periodically evict oldest depth entries

## Reference

Paper: [Mixture-of-Depths Attention](https://arxiv.org/abs/2603.15619)
