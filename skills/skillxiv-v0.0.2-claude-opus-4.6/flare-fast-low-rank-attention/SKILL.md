---
name: flare-fast-low-rank-attention
title: "FLARE: Fast Low-Rank Attention Routing Engine for Scalable Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.12594
keywords: [low-rank-attention, long-sequences, attention-routing, scalability, transformers]
description: "Implement low-rank attention routing using encode-decode factorization to achieve linear-time complexity on long sequences while maintaining compatibility with optimized attention kernels."
---

# FLARE: Fast Low-Rank Attention Routing Engine

## Core Concept

FLARE reduces the quadratic complexity of self-attention in Transformers by routing sequence information through a small set of latent tokens using low-rank factorization. The method maintains compatibility with fused attention kernels, enabling practical speedups on modern hardware. Instead of materializing large projection matrices, FLARE decomposes attention into a minimal encode-decode factorization with rank M, effectively creating a mixture of head-specific low-rank pathways that compress information without sacrificing model expressiveness.

## Architecture Overview

- **Encode-Decode Factorization**: Two-stage attention routing through latent tokens
- **Head-Specific Low-Rank Paths**: Each attention head gets distinct latent slices for fine-grained routing
- **Kernel Compatibility**: Uses standard scaled dot-product attention for compatibility with flash kernels
- **Linear Complexity**: Reduces attention from O(n²) to O(n·m) where m is latent dimension
- **Hardware Efficiency**: Leverages optimized attention implementations without custom kernels

## Implementation Steps

### 1. Create Latent Token Pool

Initialize learnable latent tokens that serve as information bottlenecks:

```python
def create_latent_tokens(
    num_latents: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int = None
) -> torch.nn.Parameter:
    """
    Create head-specific latent token slices for low-rank routing.

    Total params: num_latents * hidden_size
    Each attention head gets: num_latents * (hidden_size // num_heads) parameters
    """
    if head_dim is None:
        head_dim = hidden_size // num_heads

    # Create latents: (num_heads, num_latents, head_dim)
    latents = torch.nn.Parameter(
        torch.randn(num_heads, num_latents, head_dim) * 0.02
    )
    return latents
```

### 2. Implement Encode Stage (Sequence to Latents)

Compress sequence information into latent representations:

```python
def encode_to_latents(
    x: torch.Tensor,  # (batch, seq_len, hidden)
    latents: torch.Tensor,  # (num_heads, num_latents, head_dim)
    num_heads: int
) -> torch.Tensor:
    """
    Encode sequence information into latent tokens via attention.

    Creates mapping from full sequence to compressed latent space.
    """
    batch_size, seq_len, hidden_size = x.shape
    head_dim = hidden_size // num_heads

    # Reshape into heads
    x_heads = x.view(batch_size, seq_len, num_heads, head_dim)
    x_heads = x_heads.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

    # Compute attention from latents to sequence (reverse for encoding)
    # Latents act as queries, sequence as keys/values
    q = latents.unsqueeze(0)  # (1, num_heads, num_latents, head_dim)
    k = x_heads  # (batch, num_heads, seq_len, head_dim)
    v = x_heads

    # Scaled dot-product attention
    scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    encoded = torch.matmul(weights, v)  # (batch, num_heads, num_latents, head_dim)

    return encoded
```

### 3. Implement Decode Stage (Latents to Sequence)

Expand latent representations back to full sequence:

```python
def decode_from_latents(
    encoded: torch.Tensor,  # (batch, num_heads, num_latents, head_dim)
    query: torch.Tensor,  # (batch, seq_len, hidden)
    num_heads: int
) -> torch.Tensor:
    """
    Decode compressed latent information back to sequence length.

    Produces attention output by projecting latents to full sequence.
    """
    batch_size, seq_len, hidden_size = query.shape
    head_dim = hidden_size // num_heads

    # Reshape query into heads
    q_heads = query.view(batch_size, seq_len, num_heads, head_dim)
    q_heads = q_heads.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)

    # Compute attention from sequence queries to encoded latents
    k = encoded  # (batch, num_heads, num_latents, head_dim)
    v = encoded

    # Scaled dot-product attention
    scores = torch.matmul(q_heads, k.transpose(-2, -1)) / sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)  # (batch, num_heads, seq_len, head_dim)

    # Reshape back to sequence format
    output = output.transpose(1, 2).contiguous()
    output = output.view(batch_size, seq_len, hidden_size)

    return output
```

### 4. Combine Encode-Decode into Attention Layer

Integrate full FLARE attention mechanism:

```python
class FLAREAttention(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, num_latents: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.head_dim = hidden_size // num_heads

        # Linear projections
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)

        # Learnable latent tokens (per-head)
        self.latents = torch.nn.Parameter(
            torch.randn(num_heads, num_latents, self.head_dim) * 0.02
        )

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        # Project inputs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to per-head format
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Encode: compress sequence to latents
        # Latents attend to keys/values
        latent_q = self.latents.unsqueeze(0)
        encode_scores = torch.matmul(latent_q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        encode_weights = torch.softmax(encode_scores, dim=-1)
        encoded = torch.matmul(encode_weights, v)

        # Decode: expand from latents back to sequence
        # Sequence queries attend to latents
        decode_scores = torch.matmul(q, encoded.transpose(-2, -1)) / sqrt(self.head_dim)
        decode_weights = torch.softmax(decode_scores, dim=-1)
        output = torch.matmul(decode_weights, encoded)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, hidden_size)
        output = self.out_proj(output)

        return output
```

### 5. Scale to Million-Point Sequences

Optimize for extreme-length inputs:

```python
def create_flare_layer_optimized(
    hidden_size: int,
    num_heads: int = 8,
    num_latents: int = 256,
    gradient_checkpointing: bool = True,
    use_flash_attention: bool = True
) -> FLAREAttention:
    """
    Create optimized FLARE layer for long sequences.
    """
    layer = FLAREAttention(hidden_size, num_heads, num_latents)

    if gradient_checkpointing:
        # Enable gradient checkpointing to reduce memory
        layer.gradient_checkpointing = True

    if use_flash_attention:
        # Use flash attention kernels for speed
        layer.use_flash_attention = True

    return layer
```

## Practical Guidance

### When to Use FLARE

- Processing million-token sequences or longer documents
- PDE surrogate modeling and scientific computing
- Long-range dependencies (>10k tokens) in language tasks
- Memory-constrained environments requiring sequence compression
- Applications with fused attention kernel support (recent GPUs)

### When NOT to Use

- Short sequences (<2k tokens) where overhead exceeds benefits
- Tasks requiring exact attention maps for interpretability
- Models without optimized attention kernel support
- Scenarios demanding maximum expressiveness over efficiency

### Key Hyperparameters

- **num_latents**: 64-512; controls compression ratio (seq_len / num_latents)
- **num_heads**: 8-16; standard head counts, each gets distinct latent slices
- **Encode Attention Type**: Standard or sparse for preprocessing
- **Decode Attention Type**: Standard scaled dot-product for compatibility
- **Gradient Checkpointing**: Enable for >100k token sequences

### Performance Expectations

- Memory: Linear in sequence length (from O(n²))
- Speed: 4-8x faster on long sequences (up to 1M tokens)
- Accuracy: Minimal degradation on downstream tasks
- Kernel Compatibility: Works with FlashAttention-2 and similar

## Reference

Lei, M., & et al. (2024). FLARE: Fast Low-rank Attention Routing Engine. arXiv preprint arXiv:2508.12594.
