---
name: token-sparse-attention-long-context
title: "Token Sparse Attention: Efficient Long-Context Inference"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03216"
keywords: [Sparse Attention, Long-Context, Token Selection, KV Cache Efficiency, Inference Optimization]
description: "Dynamically select important tokens at the attention head level, performing dense attention only on selected tokens and scattering results back. Achieves 3.23x attention speedup at 128K context with 1% accuracy loss through layer-wise representation stability analysis."
---

# Token Sparse Attention: Head-Wise Selective Attention

Attention computation dominates inference latency for long contexts. Instead of attending to all tokens, Token Sparse Attention selectively attends to the most important tokens identified via a lightweight scoring mechanism. Unlike token eviction methods that permanently remove tokens, this approach preserves tokens through residual connections, allowing layer-wise reconsideration.

The key insight is that different attention heads have different token importance patterns. By allowing each head to independently select its critical tokens and gather/scatter them efficiently, the system maintains representation quality while reducing computation quadratically.

## Core Concept

Token Sparse Attention operates as a two-stage process:

1. **Compression**: Each attention head scores all tokens, selects the top-K important ones, and performs dense attention on this reduced set.

2. **Decompression**: The attention output is scattered back to the original sequence dimension via gather/scatter operations, preserving full sequence shape for subsequent layers.

This compress-decompress design allows tokens to be reconsidered in later layers when their importance changes, avoiding the irreversibility of token eviction.

## Architecture Overview

- **Token Scoring Layer**: Lightweight mechanism computing per-head importance scores without full attention cost
- **Selection Threshold**: Determines K (number of tokens to retain) per head
- **Gather Operation**: Selects top-K tokens into dense attention block
- **Dense Attention**: Standard attention computation on reduced token set
- **Scatter Operation**: Reconstructs full-sequence attention output
- **Stability Analysis**: Identifies layers where representation drift indicates token importance changes

## Implementation

### Step 1: Implement Token Scoring Mechanism

Create an efficient scoring function that identifies important tokens without full quadratic computation.

```python
# Token scoring for importance
class TokenScorer:
    def __init__(self, hidden_dim: int, method: str = "entropy"):
        """
        Score token importance efficiently.

        Args:
            hidden_dim: Hidden dimension of tokens
            method: "entropy" (query-key alignment) or "magnitude"
        """
        self.hidden_dim = hidden_dim
        self.method = method
        if method == "learned":
            self.score_proj = nn.Linear(hidden_dim, 1)

    def score_tokens(self, queries: torch.Tensor,
                    keys: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for tokens.

        Args:
            queries: [batch, seq_len, hidden_dim]
            keys: [batch, seq_len, hidden_dim]

        Returns:
            scores: [batch, seq_len] importance scores
        """
        if self.method == "entropy":
            # Compute query-key alignment as importance
            # Query entropy: how much info each query seeks
            qk = torch.matmul(queries, keys.transpose(-2, -1))
            qk = qk / math.sqrt(self.hidden_dim)

            # Sum over query dimension to get token importance
            scores = torch.softmax(qk, dim=-1).sum(dim=-2)
            return scores

        elif self.method == "magnitude":
            # Token norm as proxy for importance
            query_norm = torch.norm(queries, dim=-1)
            key_norm = torch.norm(keys, dim=-1)
            return (query_norm + key_norm) / 2

        elif self.method == "learned":
            # Learned scoring function
            return self.score_proj(queries).squeeze(-1)

    def select_top_k(self, scores: torch.Tensor,
                    k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-K tokens by score.

        Args:
            scores: [batch, seq_len]
            k: Number of tokens to select

        Returns:
            selected_scores: [batch, k]
            selected_indices: [batch, k]
        """
        selected_scores, selected_indices = torch.topk(scores, k, dim=-1)
        return selected_scores, selected_indices
```

### Step 2: Implement Gather and Scatter Operations

Create efficient operations to select and restore tokens.

```python
# Gather and scatter for sparse attention
def gather_tokens(
    tokens: torch.Tensor,  # [batch, seq_len, hidden_dim]
    indices: torch.Tensor   # [batch, k]
) -> torch.Tensor:
    """
    Gather selected tokens via indexing.

    Returns:
        gathered: [batch, k, hidden_dim]
    """
    batch_size = tokens.shape[0]
    batch_idx = torch.arange(batch_size, device=tokens.device)[:, None]
    return tokens[batch_idx, indices]

def scatter_attention_output(
    attention_output: torch.Tensor,  # [batch, k, hidden_dim]
    indices: torch.Tensor,           # [batch, k]
    seq_len: int
) -> torch.Tensor:
    """
    Scatter attention output back to original sequence.

    Returns:
        scattered: [batch, seq_len, hidden_dim]
    """
    batch_size = attention_output.shape[0]
    hidden_dim = attention_output.shape[-1]

    # Initialize output with zeros
    scattered = torch.zeros(
        batch_size, seq_len, hidden_dim,
        device=attention_output.device,
        dtype=attention_output.dtype
    )

    # Scatter selected positions
    batch_idx = torch.arange(batch_size, device=indices.device)[:, None]
    scattered[batch_idx, indices] = attention_output

    return scattered

def scatter_add_attention_output(
    full_output: torch.Tensor,      # [batch, seq_len, hidden_dim]
    attention_output: torch.Tensor, # [batch, k, hidden_dim]
    indices: torch.Tensor           # [batch, k]
) -> torch.Tensor:
    """
    Add scattered attention to existing output (for residual connections).

    Returns:
        updated: [batch, seq_len, hidden_dim]
    """
    batch_size = attention_output.shape[0]
    batch_idx = torch.arange(batch_size, device=indices.device)[:, None]
    full_output[batch_idx, indices] += attention_output
    return full_output
```

### Step 3: Create Sparse Attention Head

Combine scoring, gathering, and sparse computation.

```python
# Sparse attention head module
class SparseAttentionHead(nn.Module):
    def __init__(self, hidden_dim: int, k: int = 512):
        """
        Sparse attention operating on top-K tokens.

        Args:
            hidden_dim: Dimension per head
            k: Number of tokens to attend to
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = k

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scorer = TokenScorer(hidden_dim, method="entropy")

    def forward(self, hidden_states: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sparse attention forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: Optional mask

        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Score and select tokens
        scores = self.scorer.score_tokens(queries, keys)
        _, indices = self.scorer.select_top_k(
            scores,
            k=min(self.k, seq_len)
        )

        # Gather selected tokens
        selected_queries = gather_tokens(queries, indices)
        selected_keys = gather_tokens(keys, indices)
        selected_values = gather_tokens(values, indices)

        # Sparse attention on selected tokens
        attn_weights = torch.matmul(
            selected_queries,
            selected_keys.transpose(-2, -1)
        ) / math.sqrt(self.hidden_dim)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, selected_values)

        # Project output
        attn_output = self.out_proj(attn_output)

        # Scatter back to original sequence
        output = scatter_attention_output(attn_output, indices, seq_len)

        # Residual connection: add unselected tokens
        batch_idx = torch.arange(batch_size, device=indices.device)[:, None]
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool,
                         device=indices.device)
        mask[batch_idx, indices] = False
        unselected_output = hidden_states.clone()
        unselected_output[~mask] = 0
        output = output + unselected_output

        return output
```

### Step 4: Analyze Layer Stability for Pruning

Identify which layers have stable representations to determine pruning points.

```python
# Layer stability analysis
class LayerStabilityAnalyzer:
    def __init__(self, window_size: int = 3):
        """Analyze representation drift across layers."""
        self.window_size = window_size
        self.drift_history = []

    def compute_representation_drift(
        self,
        hidden_before: torch.Tensor,  # [batch, seq_len, hidden_dim]
        hidden_after: torch.Tensor
    ) -> float:
        """
        Compute representation change as drift metric.

        Uses cosine distance between normalized hidden states.
        """
        # Normalize
        h_before_norm = torch.nn.functional.normalize(hidden_before, dim=-1)
        h_after_norm = torch.nn.functional.normalize(hidden_after, dim=-1)

        # Compute cosine similarity
        similarity = torch.sum(h_before_norm * h_after_norm, dim=-1)
        drift = 1.0 - similarity.mean().item()

        return drift

    def is_layer_stable(self, drift: float, threshold: float = 0.1) -> bool:
        """Check if layer has stable representations."""
        self.drift_history.append(drift)

        if len(self.drift_history) < self.window_size:
            return False

        recent_drift = self.drift_history[-self.window_size:]
        avg_drift = sum(recent_drift) / len(recent_drift)
        return avg_drift < threshold

def identify_prunable_layers(model: nn.Module,
                            test_batch: torch.Tensor) -> List[int]:
    """
    Identify layers where token selection can be aggressive.

    Returns:
        Layer indices where representation is stable
    """
    prunable_layers = []
    analyzer = LayerStabilityAnalyzer()

    hidden = test_batch
    for layer_idx, layer in enumerate(model.layers):
        hidden_before = hidden.clone()
        hidden = layer(hidden)

        drift = analyzer.compute_representation_drift(
            hidden_before,
            hidden
        )

        if analyzer.is_layer_stable(drift):
            prunable_layers.append(layer_idx)

    return prunable_layers
```

### Step 5: End-to-End Inference with Token Sparsity

Integrate sparse attention into full inference pipeline.

```python
# Full sparse inference
def sparse_attention_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    sparsity_k: int = 512
) -> torch.Tensor:
    """
    Run inference with token sparse attention.

    Args:
        model: Language model
        input_ids: [batch, seq_len]
        max_new_tokens: Tokens to generate
        sparsity_k: Number of tokens per head

    Returns:
        generated_ids: [batch, seq_len + max_new_tokens]
    """
    # Identify prunable layers
    prunable_layers = identify_prunable_layers(model, input_ids)

    # Enable sparsity in selected heads
    for layer_idx in prunable_layers:
        if hasattr(model.layers[layer_idx].self_attn, 'to_sparse'):
            model.layers[layer_idx].self_attn.to_sparse(k=sparsity_k)

    # Standard generation with sparse attention active
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )

    return output_ids
```

## Practical Guidance

**When to use Token Sparse Attention:**
- Long-context inference (32K+ tokens) where attention is bottleneck
- Scenarios where <1% accuracy loss is acceptable for 3x speedup
- Systems with GPU/TPU supporting efficient gather/scatter operations
- Batch inference where latency matters more than throughput

**When not to use:**
- Short contexts (<4K tokens) where sparsity overhead dominates benefit
- Tasks requiring precise token interaction (e.g., code analysis)
- Systems without efficient sparse operations (CPUs)
- Real-time applications needing predictable latency

**Common Pitfalls:**
- K too small: Prunes important context, causing accuracy loss
- K too large: Eliminates sparsity benefits
- Instability in early layers: May prune tokens needed by later layers
- Residual connection bugs: Unselected tokens must be preserved

**Hyperparameter Guidelines:**

| Parameter | Recommended | Tuning Strategy |
|-----------|------------|-----------------|
| sparsity_k | 512-1024 | Higher = less sparsity; tune on accuracy-speed tradeoff |
| scoring_method | entropy | Learned scoring if accuracy critical; entropy for speed |
| stability_threshold | 0.05-0.1 | Higher = more layers eligible for sparsity |
| prunable_layers | Last 50% | Compress representation is stable in later layers |

## Reference

See the full paper at: https://arxiv.org/abs/2602.03216

Key results: 3.23× attention speedup at 128K context with <1% accuracy degradation. Triton-based kernel implementation; FlashAttention compatible. Code available on GitHub.
