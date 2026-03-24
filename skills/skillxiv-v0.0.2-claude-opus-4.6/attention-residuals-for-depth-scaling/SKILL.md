---
name: attention-residuals-for-depth-scaling
title: "Attention Residuals"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.15031"
keywords: [Attention Mechanism, Residual Connections, Depth Scaling, Hidden State Growth, Layer Aggregation]
description: "Replace uniform residual accumulation with depth-wise attention that selectively aggregates earlier layer representations. Improve gradient flow and model performance in deep architectures by learning content-dependent depth-wise selection."
---

# Attention Residuals: Selective Layer Aggregation for Improved Depth Scaling

Standard residual connections in deep models use fixed unit weights to accumulate outputs from all previous layers, creating an uncontrolled growth of hidden state magnitudes and progressively diluting each layer's individual contribution. Attention Residuals solve this by allowing each layer to selectively attend over preceding representations with learned, input-dependent weights—analogous to multi-head attention but operating across depth dimension.

This technique improves gradient distribution in deep models and offers performance gains on downstream tasks, making it especially valuable for scaling to larger depths. The practical "Block" variant achieves this with minimal computational overhead by organizing layers into blocks and attending over block-level representations.

## Core Concept

Attention Residuals replace the fixed accumulation operation with a learned attention mechanism:

**Standard Residuals:**
```
h_i = h_{i-1} + f_i(h_{i-1})  # Fixed unit weight accumulation
```

**Attention Residuals (Full):**
```
h_i = Attention(Q=h_{i-1}, K=[h_0,...,h_{i-1}], V=[h_0,...,h_{i-1}])
      # Each layer attends over all preceding layers
```

**Attention Residuals (Block):**
```
block_outputs = [output of block_0, output of block_1, ...]
h_i = Attention(Q=h_{i-1}, K=block_outputs, V=block_outputs)
      # Attend over block-level summaries, not individual layers
```

## Architecture Overview

- **Depth Dimension as Attention Axis** — Treat layer depth as a sequence to attend over, similar to sequence position in standard attention
- **Block Organization** — Partition transformer into K blocks; each block attends over previous block representations rather than individual layers
- **Cache-Based Pipeline Communication** — Maintain incrementally updated cache of block outputs for efficiency
- **Two-Phase Computation** — Separate KV computation (per block) from Q computation (per layer within block)
- **Drop-in Replacement** — Compatible with standard attention infrastructure (FlashAttention-2)

## Implementation Steps

Start by defining the block structure and implementing depth-wise attention. The key is treating layer indices as positions in a "depth sequence."

```python
# Depth-wise attention replacing standard residual connections
import torch
import torch.nn.functional as F

class AttentionResidualBlock(torch.nn.Module):
    """Replace standard residual connection with depth-wise attention."""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, current_hidden, layer_cache):
        """
        Args:
            current_hidden: [batch, seq_len, hidden_dim] from current layer
            layer_cache: list of [batch, seq_len, hidden_dim] from preceding layers
        """
        # Treat preceding layers as "sequence" for attention
        all_layers = layer_cache + [current_hidden]
        stacked = torch.stack(all_layers, dim=1)  # [batch, num_layers, seq, hidden]
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape

        # Reshape for multi-head attention
        Q = self.q_proj(current_hidden)  # [batch, seq, hidden]
        K = self.k_proj(stacked[:, :, :, :].reshape(batch_size, -1, hidden_dim))
        V = self.v_proj(stacked.reshape(batch_size, -1, hidden_dim))

        # Multi-head attention over layer dimension
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, num_layers*seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, num_layers*seq_len, self.num_heads, -1).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_dim)
        return self.out_proj(output)
```

For practical deployment, implement the block-level variant which attends over block outputs rather than individual layers, reducing memory and compute overhead.

```python
# Block-level attention: more efficient variant for production
class BlockAttentionResidual(torch.nn.Module):
    """Attend over block outputs rather than individual layer outputs."""

    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, current_block_output, block_cache):
        """
        Args:
            current_block_output: [batch, seq, hidden] output of current block
            block_cache: list of [batch, seq, hidden] outputs from previous blocks
        """
        # Stack block outputs as KV context
        all_blocks = block_cache + [current_block_output]
        stacked_blocks = torch.stack(all_blocks, dim=1)
        batch_size, num_blocks, seq_len, hidden_dim = stacked_blocks.shape

        # Reshape for efficient attention computation
        Q = self.q_proj(current_block_output)
        K_stacked = stacked_blocks.reshape(batch_size, num_blocks*seq_len, hidden_dim)
        V_stacked = K_stacked.clone()

        K = self.k_proj(K_stacked)
        V = self.v_proj(V_stacked)

        # Scaled dot-product attention
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, num_blocks*seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, num_blocks*seq_len, self.num_heads, -1).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_dim)
        return self.out_proj(output)
```

Finally, integrate this into your model's forward pass as a drop-in replacement for standard residual connections.

```python
# Integration into transformer layer forward pass
class TransformerLayerWithAttnResidual(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 4*hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4*hidden_dim, hidden_dim)
        )
        # Replace standard residual with attention residual
        self.attn_residual = BlockAttentionResidual(hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, hidden, layer_cache):
        # Standard self-attention + norm
        attn_out, _ = self.self_attn(hidden, hidden, hidden)
        hidden = self.norm1(hidden + attn_out)

        # Depth-wise attention residual
        depth_aware = self.attn_residual(hidden, layer_cache)
        hidden = self.norm2(hidden + depth_aware)

        # MLP
        mlp_out = self.mlp(hidden)
        hidden = self.norm2(hidden + mlp_out)

        return hidden
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Start with block size of 4-6 layers; smaller blocks reduce efficiency gains, larger blocks lose granularity
- Use standard (non-block) variant when model depth < 12 layers; block variant scales better to 48+ layer models
- Apply temperature scaling to attention softmax (multiply scores by 0.1-0.5) if attention becomes too peaked early in training

**When NOT to use:**
- For shallow models (< 8 layers), the overhead may exceed benefits
- When memory is extremely constrained, full layer-wise attention requires O(depth²) cache; use block variant
- Not recommended for models already using other depth-modification techniques (progressive layer freezing, etc.)

**Common Pitfalls:**
- Block cache requires manual management in distributed training; synchronize across devices before attention computation
- Softmax over very large depth dimensions can become numerically unstable; use log-sum-exp stabilization
- Initialization of attention weights matters significantly; start with Xavier uniform, not standard normal

## Reference

Paper: [Attention Residuals](https://arxiv.org/abs/2603.15031)
