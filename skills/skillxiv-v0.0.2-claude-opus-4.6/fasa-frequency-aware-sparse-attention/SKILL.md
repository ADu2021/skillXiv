---
name: fasa-frequency-aware-sparse-attention
title: "FASA: Frequency-Aware Sparse Attention"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03152"
keywords: [Sparse Attention, KV Cache Compression, Rotary Embeddings, Training-Free, Frequency Analysis]
description: "Identify dominant frequency components in RoPE embeddings to determine which attention dimensions are essential, reducing KV cache by 81% while maintaining performance. Training-free approach integrates with existing compression methods for 2.56x speedup on long-context reasoning."
---

# FASA: Functional Sparsity in Attention via Frequency Analysis

Rotary Position Embeddings (RoPE) use multiple frequency components to encode position, but not all frequencies are equally important for each task. FASA discovers which frequency dimensions actually contribute to model performance, enabling aggressive KV cache reduction by computing attention only on critical frequencies. This training-free approach complements other compression methods and requires only one-time offline calibration.

The key insight is that high-frequency components (capturing fine-grained positional distinctions) often matter less than low-frequency components (capturing coarse relationships) for understanding context. By identifying task-dependent "dominant frequency chunks," FASA achieves near-oracle compression without task-specific training.

## Core Concept

FASA operates in two stages:

1. **Token Importance Prediction (TIP)**: Use pre-identified dominant frequency chunks to efficiently score token importance without full attention computation.

2. **Focused Attention Computation (FAC)**: Perform full-precision attention only on selected critical tokens and dimensions, dramatically reducing KV cache memory requirements.

Unlike uniform sparsity, this approach exploits the structure of rotary embeddings to identify task-relevant dimensions.

## Architecture Overview

- **Frequency Analyzer**: Identifies dominant frequency components via one-time calibration
- **Dimension Selector**: Maps tasks to critical frequency chunks
- **Token Scorer**: Efficient importance prediction using only dominant frequencies
- **Attention Engine**: Computes attention on selected dimensions/tokens
- **Cache Manager**: Stores only critical KV pairs in reduced format
- **Integration Layer**: Works with other compression methods (quantization, eviction)

## Implementation

### Step 1: Analyze RoPE Frequency Components

Create tools to identify which frequency dimensions matter for a given model/task.

```python
# Frequency analysis for RoPE
import numpy as np
from typing import List, Tuple

class RoPEFrequencyAnalyzer:
    def __init__(self, hidden_dim: int, rope_theta: float = 10000.0):
        """
        Analyze frequency importance in RoPE embeddings.

        Args:
            hidden_dim: Dimension of embeddings (must be even)
            rope_theta: Base frequency (standard: 10000)
        """
        self.hidden_dim = hidden_dim
        self.rope_theta = rope_theta

        # Compute frequency components
        self.frequencies = self._compute_frequencies()

    def _compute_frequencies(self) -> np.ndarray:
        """Compute RoPE frequency components."""
        # inv_freq = 1.0 / (rope_theta ** (2i / d))
        inv_freq = 1.0 / (self.rope_theta ** (
            np.arange(0, self.hidden_dim, 2) / self.hidden_dim
        ))
        return inv_freq

    def rank_frequencies_by_importance(
        self,
        activations: np.ndarray,  # [batch, seq_len, hidden_dim]
        top_k: int = 64
    ) -> Tuple[List[int], List[float]]:
        """
        Rank frequency components by importance using activation variance.

        Args:
            activations: Model activations for calibration set
            top_k: Number of top frequencies to return

        Returns:
            indices: Frequency indices (0 to hidden_dim//2)
            scores: Importance scores for each frequency
        """
        # Split activations into frequency pairs
        # RoPE uses sin/cos pairs for each frequency
        freq_activations = []

        for freq_idx in range(self.hidden_dim // 2):
            # Get pair of dimensions for this frequency
            dim1 = 2 * freq_idx
            dim2 = 2 * freq_idx + 1

            pair_act = activations[:, :, dim1:dim2+1]
            # Importance = variance across positions
            importance = np.var(pair_act)
            freq_activations.append((freq_idx, importance))

        # Sort by importance (descending)
        sorted_freqs = sorted(freq_activations, key=lambda x: x[1], reverse=True)

        # Return top-K
        indices = [f[0] for f in sorted_freqs[:top_k]]
        scores = [f[1] for f in sorted_freqs[:top_k]]

        return indices, scores

def identify_dominant_frequency_chunks(
    model,
    calibration_dataset,
    num_chunks: int = 8,
    compression_ratio: float = 0.25
) -> List[List[int]]:
    """
    Identify dominant frequency chunks for a model via calibration.

    Args:
        model: Language model
        calibration_dataset: Validation set for analysis
        num_chunks: Number of frequency chunks to create
        compression_ratio: Target fraction of dimensions to keep

    Returns:
        List of frequency chunk lists, each containing dimension indices
    """
    analyzer = RoPEFrequencyAnalyzer(model.config.hidden_size)

    # Collect activations
    all_activations = []
    for batch in calibration_dataset:
        with torch.no_grad():
            hidden = model.encoder(batch['input_ids'])
        all_activations.append(hidden.cpu().numpy())

    activations = np.concatenate(all_activations, axis=0)

    # Rank frequencies
    ranked_indices, scores = analyzer.rank_frequencies_by_importance(activations)

    # Create chunks: distribute frequencies
    target_freq_count = int(
        (model.config.hidden_size // 2) * compression_ratio
    )

    # Create chunks of consecutive frequencies
    chunk_size = max(1, target_freq_count // num_chunks)
    chunks = []

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(ranked_indices))
        chunk_freqs = ranked_indices[start_idx:end_idx]
        chunks.append(chunk_freqs)

    return chunks
```

### Step 2: Implement Token Importance Prediction (TIP)

Create efficient token scoring using only dominant frequencies.

```python
# Token Importance Prediction
class TokenImportancePredictor:
    def __init__(self, dominant_freqs: List[int], hidden_dim: int):
        """
        Predict token importance using dominant frequencies only.

        Args:
            dominant_freqs: Indices of important frequency components
            hidden_dim: Total hidden dimension
        """
        self.dominant_freqs = dominant_freqs
        self.hidden_dim = hidden_dim

        # Map frequency indices to dimension indices
        # Each frequency contributes 2 dimensions (sin/cos)
        self.dimension_indices = []
        for freq_idx in dominant_freqs:
            self.dimension_indices.append(2 * freq_idx)
            self.dimension_indices.append(2 * freq_idx + 1)

    def score_tokens(
        self,
        queries: torch.Tensor,  # [batch, seq_len, hidden_dim]
        keys: torch.Tensor,     # [batch, seq_len, hidden_dim]
        values: torch.Tensor    # [batch, seq_len, hidden_dim]
    ) -> torch.Tensor:
        """
        Predict importance of each token using dominant dimensions only.

        Args:
            queries, keys, values: Attention inputs

        Returns:
            scores: [batch, seq_len] importance scores
        """
        # Extract dominant dimensions only
        dim_idx = torch.tensor(
            self.dimension_indices,
            device=queries.device
        )

        q_dominant = queries[..., dim_idx]
        k_dominant = keys[..., dim_idx]
        v_dominant = values[..., dim_idx]

        # Compute query-key alignment on dominant dims only
        # This is fast because we compute on 25% of dimensions
        attn_scores = torch.matmul(
            q_dominant,
            k_dominant.transpose(-2, -1)
        ) / math.sqrt(q_dominant.shape[-1])

        # Sum over query dimension to get token importance
        # Tokens that match many queries are important
        scores = torch.softmax(attn_scores, dim=-1).sum(dim=-2)

        return scores

    def select_important_tokens(self, scores: torch.Tensor,
                               top_k: int) -> torch.Tensor:
        """Select top-K tokens by importance."""
        _, indices = torch.topk(scores, k=min(top_k, scores.shape[-1]), dim=-1)
        return indices
```

### Step 3: Implement Focused Attention Computation (FAC)

Create attention module that uses sparse frequencies and selected tokens.

```python
# Focused Attention Computation
class FocusedAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int,
                 dominant_freqs: List[int], cache_k: int = 512):
        """
        Sparse attention using dominant frequencies and selected tokens.

        Args:
            hidden_dim: Model hidden dimension
            num_heads: Number of attention heads
            dominant_freqs: Dominant frequency indices for this attention layer
            cache_k: Number of KV tokens to cache
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.cache_k = cache_k

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scorer = TokenImportancePredictor(dominant_freqs, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Focused attention: sparse on selected tokens.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            use_cache: Whether to cache KV pairs

        Returns:
            output: [batch, seq_len, hidden_dim]
            cache: Cached KV pairs if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Score tokens using important frequencies
        token_importance = self.scorer.score_tokens(queries, keys, values)

        # Select top-K tokens
        selected_indices = self.scorer.select_important_tokens(
            token_importance,
            top_k=self.cache_k
        )

        # Gather selected K and V for caching
        batch_idx = torch.arange(batch_size, device=keys.device)[:, None]
        cached_keys = keys[batch_idx, selected_indices]
        cached_values = values[batch_idx, selected_indices]

        # Full attention on current query with selected KV pairs
        # Reshape for multi-head attention
        q_heads = queries.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_heads = cached_keys.view(batch_size, self.cache_k, self.num_heads, -1).transpose(1, 2)
        v_heads = cached_values.view(batch_size, self.cache_k, self.num_heads, -1).transpose(1, 2)

        # Attention with selected tokens
        attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.hidden_dim // self.num_heads)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v_heads)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(attn_output)

        # Return cache if needed
        cache = (cached_keys, cached_values) if use_cache else None

        return output, cache
```

### Step 4: Integrate FASA into Full Model

Modify model to use frequency-aware sparse attention.

```python
# Full FASA integration
class FASSAModel(nn.Module):
    def __init__(self, base_model: nn.Module, dominant_chunks: List[List[int]]):
        """
        Wrap base model with FASA sparse attention.

        Args:
            base_model: Original language model
            dominant_chunks: Frequency chunks for each attention layer
        """
        super().__init__()
        self.base_model = base_model
        self.dominant_chunks = dominant_chunks

        # Replace attention layers with sparse variants
        for layer_idx, layer in enumerate(base_model.layers):
            if hasattr(layer, 'self_attn'):
                # Get dominant frequencies for this layer
                freqs = dominant_chunks[layer_idx % len(dominant_chunks)]

                layer.self_attn = FocusedAttention(
                    hidden_dim=base_model.config.hidden_size,
                    num_heads=base_model.config.num_attention_heads,
                    dominant_freqs=freqs,
                    cache_k=512
                )

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass using sparse attention."""
        return self.base_model(input_ids, **kwargs)

def apply_fasa_to_model(
    model: nn.Module,
    calibration_dataset,
    compression_ratio: float = 0.25
) -> nn.Module:
    """
    Apply FASA to a model via calibration.

    Args:
        model: Base language model
        calibration_dataset: Data for importance calibration
        compression_ratio: Target KV cache reduction

    Returns:
        Model with FASA-enabled attention
    """
    # Identify dominant frequencies
    chunks = identify_dominant_frequency_chunks(
        model,
        calibration_dataset,
        compression_ratio=compression_ratio
    )

    # Apply FASA
    fassa_model = FASSAModel(model, chunks)

    return fassa_model
```

## Practical Guidance

**When to use FASA:**
- Long-context inference (32K+ tokens) where KV cache dominates memory
- Scenarios accepting 0.7% accuracy loss for 2.5x+ speedup
- Multi-domain models where frequency importance varies
- Complementing other compression methods (quantization, eviction)

**When not to use:**
- Short contexts (<8K) where KV cache isn't bottleneck
- Tasks requiring maximum precision on positional reasoning
- Real-time systems where calibration overhead is unacceptable
- Scenarios without offline calibration budget

**Common Pitfalls:**
- Calibration set mismatch: If calibration domain differs from deployment, frequency importance changes
- Too aggressive compression: 81% reduction works for some tasks; test empirically
- Neglecting interaction with other optimizations: FASA + quantization requires separate tuning
- Uniform frequency chunks: Chunk importance varies per task; per-layer adaptation helps

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| compression_ratio | 0.18-0.35 | 0.25 baseline; higher for speed, lower for accuracy |
| num_chunks | 4-16 | Balance granularity and memory; 8 typical |
| cache_k (tokens) | 256-1024 | Higher = less compression; tune with accuracy |
| Calibration size | 100-500 examples | More data improves frequency estimates |

## Reference

See the full paper at: https://arxiv.org/abs/2602.03152

Key results: 2.56× speedup with 18.9% KV cache retention; near-oracle accuracy with only 25% of attention dimensions. Training-free; composes with other compression methods. Code available at https://github.com/AMAP-ML/FASA-ICLR2026
