---
name: less-is-more-sparse-attention
title: LessIsMore - Training-Free Sparse Attention for Reasoning Models
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.07101
keywords: [sparse-attention, inference-optimization, reasoning-models, token-selection]
description: "Improves inference efficiency through training-free sparse attention using global token selection patterns aggregated from local attention heads for unified cross-head ranking."
---

## LessIsMore: Training-Free Sparse Attention for Reasoning Models

### Core Concept

LessIsMore enhances inference efficiency by using global attention patterns to select which tokens are essential for processing. Rather than maintaining separate token subsets per attention head, it aggregates signals across heads to create unified token rankings, enabling faster decoding while preserving reasoning quality.

### Architecture Overview

- **Local Head Analysis**: Examine token importance within individual attention heads
- **Global Aggregation**: Combine importance signals across all heads using contextual information
- **Unified Token Ranking**: Generate single importance score per token for all subsequent layers
- **Dynamic Selection**: Prune low-importance tokens adaptively based on thresholds

### Implementation Steps

**Step 1: Extract Local Attention Patterns**

Analyze individual attention head token importance:

```python
# Pseudocode for local attention analysis
class LocalAttentionAnalyzer:
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def analyze_head_importance(self, attention_weights, hidden_states):
        """
        Compute per-head token importance from attention patterns.

        Args:
            attention_weights: (batch, num_heads, seq_len, seq_len)
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            head_importance: (batch, num_heads, seq_len)
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        head_importance = []
        for head_idx in range(num_heads):
            # Attention entropy: lower entropy = more focused attention
            head_attn = attention_weights[:, head_idx, :, :]
            entropy = -torch.sum(
                head_attn * torch.log(head_attn + 1e-10),
                dim=-1
            )

            # Inverse entropy: lower entropy = higher importance
            importance = 1.0 / (entropy + 1e-8)
            head_importance.append(importance)

        return torch.stack(head_importance, dim=1)

    def get_head_token_scores(self, attention_weights):
        """
        Compute importance score for each token per head.
        """
        # Sum incoming attention to each token
        token_in_degree = torch.sum(attention_weights, dim=2)  # Sum over source positions
        return token_in_degree
```

**Step 2: Implement Global Aggregation**

Combine local signals with contextual information:

```python
# Pseudocode for global aggregation
class GlobalAggregator(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Aggregation network
        self.aggregation_layer = nn.Sequential(
            nn.Linear(num_heads + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def aggregate_head_signals(self, head_scores, hidden_states):
        """
        Aggregate per-head importance scores with contextual cues.

        Args:
            head_scores: (batch, num_heads, seq_len)
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            global_importance: (batch, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        global_importance = []
        for pos in range(seq_len):
            # Gather head scores for this position
            pos_head_scores = head_scores[:, :, pos]  # (batch, num_heads)

            # Get contextual information from hidden state
            pos_context = hidden_states[:, pos, :]  # (batch, hidden_dim)

            # Combine head signals and context
            combined = torch.cat([pos_head_scores, pos_context], dim=-1)
            global_score = self.aggregation_layer(combined)

            global_importance.append(global_score)

        return torch.stack(global_importance, dim=1).squeeze(-1)
```

**Step 3: Implement Unified Token Selection**

Select tokens with unified ranking across layers:

```python
# Pseudocode for unified token selection
class UnifiedTokenSelector:
    def __init__(self, sparsity_ratio=0.3):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        self.selected_positions = {}

    def select_tokens(self, importance_scores, recent_context=None):
        """
        Select tokens to keep based on unified importance scores.

        Args:
            importance_scores: (batch, seq_len)
            recent_context: Recently processed tokens for ranking adjustment

        Returns:
            selected_mask: (batch, seq_len) boolean mask
        """
        batch_size, seq_len = importance_scores.shape

        # Boost importance of recent tokens for contextual relevance
        if recent_context is not None:
            recent_boost = torch.zeros_like(importance_scores)
            num_recent = min(16, seq_len)
            recent_boost[:, -num_recent:] = 0.2

            adjusted_scores = importance_scores + recent_boost
        else:
            adjusted_scores = importance_scores

        # Select top-k tokens
        num_keep = max(1, int(seq_len * (1 - self.sparsity_ratio)))
        _, top_indices = torch.topk(adjusted_scores, num_keep, dim=1)

        # Create mask
        selected_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
        for batch_idx in range(batch_size):
            selected_mask[batch_idx, top_indices[batch_idx]] = True

        return selected_mask

    def apply_sparse_attention(self, attention_weights, selected_mask):
        """
        Apply token selection to attention computation.
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Mask out non-selected positions
        for pos in range(seq_len):
            if not selected_mask[0, pos]:
                attention_weights[:, :, pos, :] = 0
                attention_weights[:, :, :, pos] = 0

        # Renormalize attention
        attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return attention_weights
```

**Step 4: Integrate into Decoding Loop**

Apply sparse attention during inference:

```python
# Pseudocode for sparse attention in decoding
def sparse_decoding(model, input_ids, num_tokens_to_generate=100, sparsity_ratio=0.3):
    """
    Generate tokens using sparse attention.
    """
    selector = UnifiedTokenSelector(sparsity_ratio=sparsity_ratio)
    analyzer = LocalAttentionAnalyzer(model.num_heads)
    aggregator = GlobalAggregator(model.hidden_dim, model.num_heads)

    generated_tokens = input_ids.clone()
    kv_cache = None

    for step in range(num_tokens_to_generate):
        with torch.no_grad():
            # Forward pass with attention tracking
            outputs = model(
                generated_tokens,
                return_attention=True,
                kv_cache=kv_cache
            )

            # Analyze attention patterns
            head_scores = analyzer.analyze_head_importance(
                outputs.attention_weights,
                outputs.hidden_states
            )

            # Aggregate globally
            global_importance = aggregator.aggregate_head_signals(
                head_scores,
                outputs.hidden_states
            )

            # Select tokens for next layer
            selected_mask = selector.select_tokens(global_importance)

            # Get next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Append selected tokens only
            if selected_mask[:, -1].all():
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(-1)], dim=1)
                kv_cache = outputs.kv_cache

    return generated_tokens
```

### Practical Guidance

**Hyperparameters and Configuration**:
- Sparsity ratio: 0.2-0.4 (keep 60-80% of tokens)
- Recent context window: 16 tokens (always keep recent for coherence)
- Importance aggregation attention: 0.1-0.3 boost for recent tokens
- Head analysis entropy threshold: Adaptive per model

**When to Use LessIsMore**:
- Long-sequence reasoning tasks where efficiency matters
- Inference systems with latency constraints
- Reasoning models that generate lengthy intermediate steps
- Tasks where token importance is well-correlated with attention patterns

**When NOT to Use**:
- Very short sequences where sparsity provides minimal benefit
- Tasks requiring comprehensive context (rare, highly connected information)
- Scenarios where all tokens are equally important
- When token-level interpretability is critical

**Implementation Notes**:
- The method is completely training-free, no retraining required
- Recent tokens are always kept to maintain local coherence
- Sparsity can be adjusted per-layer or globally
- Monitor accuracy degradation with different sparsity ratios
- Head analysis can be cached across similar inputs for efficiency

### Reference

Paper: Less Is More: Training-Free Sparse Attention for Efficient Reasoning
ArXiv: 2508.07101
Performance: 1.1× decoding speedup, 2× fewer attended tokens, 1.13× end-to-end speedup vs existing methods
