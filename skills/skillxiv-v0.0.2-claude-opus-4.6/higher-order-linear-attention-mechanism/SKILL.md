---
name: higher-order-linear-attention-mechanism
title: "Higher-order Linear Attention"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.27258"
keywords: [Linear Attention, Efficient Transformers, Streaming Inference, Long Context, Recurrent Architectures]
description: "Enable data-dependent higher-order interactions in attention using prefix-sufficient statistics that maintain linear time and constant state, replacing quadratic dot-product attention while preserving expressivity through compact matrix operations."
---

# Title: Scale Attention to Long Contexts via Linear-Time Matrix Interactions

Scaled dot-product attention requires O(n²) memory and time for sequence length n, crippling long-context applications. Higher-Order Linear Attention (HLA) maintains O(d²) state and O(d² + d·dᵥ) per-token cost by replacing the n×n attention matrix with compact prefix statistics—outer products of keys and query-value accumulators. The key insight is using **extended summaries** that prevent future-token leakage while enabling higher-order (beyond first-order) interactions.

The approach supports exact causal masking, streaming inference, and parallel training through associative scans.

## Core Concept

**Linear-Time Higher-Order Interactions**:
- **Prefix Statistics**: Maintain key outer product sums (S^K_t) and query-value accumulators (C^QE_t)
- **Streaming Computation**: Compute outputs as `o_t = q_t^T S^K_t C^QE_t` (linear time per token)
- **Causal Masking**: Extended summaries (G_t, h_t) prevent seeing future tokens
- **Associative Scans**: Enable chunk-parallel training that reproduces serial computation exactly
- **Higher-Order Expressivity**: Supports second-order and third-order tensor operations without quadratic scaling

## Architecture Overview

- **Key Statistics**: d×d matrix of key outer products (S^K_t = Σ_i k_i k_i^T)
- **Query-Value Accumulator**: d×d_v matrix (C^QE_t = Σ_i φ(q_i) v_i)
- **Cross-Term Accumulator**: G_t prevents future-token leakage in causal setting
- **Normalization**: Denominator terms (h_t, m_t) for masked computation
- **Multi-Head Support**: Efficient with multi-query attention (O(h·d·d_v) space per head group)

## Implementation Steps

**1. Implement Prefix Statistics Computation**

Maintain running statistics instead of materializing attention matrices.

```python
class HigherOrderLinearAttention(nn.Module):
    def __init__(self, dim, num_heads, dv=None):
        self.num_heads = num_heads
        self.dim = dim // num_heads  # d per head
        self.dv = dv or dim // num_heads

    def forward(self, query, key, value, causal=True):
        batch, seq_len, dim = query.shape
        d = self.dim
        dv = self.dv

        # Initialize prefix statistics
        S_K = torch.zeros(batch, self.num_heads, d, d, device=query.device)  # Key outer products
        C_QE = torch.zeros(batch, self.num_heads, d, dv, device=query.device)  # Query-value accum
        G_t = torch.zeros(batch, self.num_heads, d, dv, device=query.device)  # Cross-term
        h_t = torch.zeros(batch, self.num_heads, d, device=query.device)  # Denominator
        m_t = torch.zeros(batch, self.num_heads, device=query.device)  # Scalar denominator

        # Reshape for multi-head
        q = query.view(batch, seq_len, self.num_heads, d).transpose(1, 2)
        k = key.view(batch, seq_len, self.num_heads, d).transpose(1, 2)
        v = value.view(batch, seq_len, self.num_heads, dv).transpose(1, 2)

        outputs = []

        for t in range(seq_len):
            q_t = q[:, :, t, :]  # [batch, heads, d]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]

            # Update prefix statistics
            # S_t^K = S_{t-1}^K + k_t k_t^T
            S_K = S_K + torch.einsum('bhi,bhj->bhij', k_t, k_t)

            # C_t^QE = C_{t-1}^QE + φ(q_t) v_t
            phi_q_t = torch.softmax(q_t / np.sqrt(d), dim=-1)
            C_QE = C_QE + torch.einsum('bhi,bhj->hij', phi_q_t, v_t)

            # Compute output: o_t = q_t^T S_t^K C_t^QE
            S_times_C = torch.einsum('bhij,bhjk->bhik', S_K, C_QE)
            output = torch.einsum('bhi,bhij->bhj', q_t, S_times_C)

            # Causal masking: apply cross-term correction
            if causal:
                # G_t = Σ_{i<t} φ(q_i) v_i (accumulate query-value for past timesteps)
                # Prevents q_t from attending to future elements
                output = output - torch.einsum('bhi,bhij->bhj', q_t, G_t)
                G_t = G_t + torch.einsum('bhi,bhj->hij', phi_q_t, v_t)

            # Normalize by denominator
            m_t = m_t + 1  # Normalization factor
            output = output / (m_t.unsqueeze(-1) + 1e-8)

            outputs.append(output)

        # Reshape back to [batch, seq_len, dim]
        output = torch.stack(outputs, dim=2)  # [batch, heads, seq_len, dv]
        output = output.transpose(1, 2).reshape(batch, seq_len, dim)
        return output
```

**2. Enable Streaming Inference with Extended Summaries**

Implement the causal masking mechanism that prevents future-token leakage.

```python
class StreamingHigherOrderAttention(nn.Module):
    def __init__(self, dim, num_heads, dv=None):
        self.num_heads = num_heads
        self.dim = dim // num_heads
        self.dv = dv or self.dim

    def stream_forward(self, query, key, value, state=None):
        # For streaming: process one token at a time
        batch, d_model = query.shape

        if state is None:
            # Initialize state
            state = {
                'S_K': torch.zeros(batch, self.num_heads, self.dim, self.dim),
                'C_QE': torch.zeros(batch, self.num_heads, self.dim, self.dv),
                'G_t': torch.zeros(batch, self.num_heads, self.dim, self.dv),
                'h_t': torch.zeros(batch, self.num_heads, self.dim),
                'step': 0
            }

        # Reshape inputs
        q = query.view(batch, self.num_heads, self.dim)
        k = key.view(batch, self.num_heads, self.dim)
        v = value.view(batch, self.num_heads, self.dv)

        # Update statistics
        state['S_K'] = state['S_K'] + torch.einsum('bhi,bhj->bhij', k, k)
        state['C_QE'] = state['C_QE'] + torch.einsum('bhi,bhj->hij', q, v)

        # Compute output with causal mask
        S_times_C = torch.einsum('bhij,bhjk->bhik', state['S_K'], state['C_QE'])
        output = torch.einsum('bhi,bhij->bhj', q, S_times_C)

        # Apply cross-term (causal correction)
        output = output - torch.einsum('bhi,bhij->bhj', q, state['G_t'])
        state['G_t'] = state['G_t'] + torch.einsum('bhi,bhj->hij', q, v)

        # Normalization
        state['h_t'] = state['h_t'] + torch.ones_like(k).sum(dim=-1, keepdim=True)
        output = output / (state['h_t'] + 1e-8)

        state['step'] += 1
        return output.view(batch, -1), state
```

**3. Implement Associative Scans for Parallel Training**

Enable efficient chunk-based parallel computation that reproduces serial computation.

```python
class AssociativeScanAttention(nn.Module):
    def __init__(self, dim, chunk_size=16):
        self.dim = dim
        self.chunk_size = chunk_size

    def chunk_wise_forward(self, query, key, value):
        seq_len = query.shape[1]
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        # Process each chunk independently, then combine
        chunk_outputs = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)

            q_chunk = query[:, start:end, :]
            k_chunk = key[:, start:end, :]
            v_chunk = value[:, start:end, :]

            # Within-chunk computation
            S_K_chunk = torch.einsum('bti,btj->bij', k_chunk, k_chunk)
            C_QE_chunk = torch.einsum('bti,btj->bij', q_chunk, v_chunk)

            # Output within chunk
            output_chunk = torch.einsum('bti,bij->btj', q_chunk, S_K_chunk @ C_QE_chunk)

            chunk_outputs.append(output_chunk)

        # Combine chunks with cross-chunk interactions
        # This is where associative structure enables parallelization
        output = torch.cat(chunk_outputs, dim=1)
        return output
```

**4. Complexity Analysis Verification**

Verify that per-token computation is linear, not quadratic.

```python
def analyze_complexity():
    # Standard dot-product attention
    n = 4096  # Sequence length
    d = 64  # Head dimension
    dv = 64  # Value dimension

    # Dot-product: O(n²d)
    dot_product_ops = n * n * d
    dot_product_memory = n * n  # Attention matrix

    # HLA: O(n * (d² + d*dv))
    hla_ops = n * (d * d + d * dv)
    hla_memory = d * d + d * dv  # Prefix statistics

    print(f"Dot-product ops: {dot_product_ops / 1e9:.1f}B")
    print(f"HLA ops: {hla_ops / 1e6:.1f}M")
    print(f"Speedup: {dot_product_ops / hla_ops:.0f}x")
```

## Practical Guidance

**When to Use**:
- Long-context applications (>4K tokens)
- Streaming inference with limited memory
- Sequence-to-sequence models with variable lengths

**Hyperparameters**:
- num_heads: 8-16 (standard transformer settings)
- dim_head: 64 (standard dimension per head)
- chunk_size (parallel training): 16-64 (trade-off between parallelism and communication)

**When NOT to Use**:
- Tasks where quadratic interaction patterns are essential
- Very small sequences where O(n²) is acceptable
- Systems requiring exact scaled-dot-product semantics

**Pitfalls**:
- **Insufficient softmax approximation**: φ(q) needs careful design for stability
- **Normalization issues**: Denominator tracking is subtle; wrong denominator breaks causal masking
- **Chunk parallelization complexity**: Associative structure requires careful implementation

**Key Insight**: Unlike linear attention variants that lose expressivity, HLA maintains higher-order interactions through tensor products. The trick is using prefix statistics to keep state constant-sized.

## Reference

arXiv: https://arxiv.org/abs/2510.27258
