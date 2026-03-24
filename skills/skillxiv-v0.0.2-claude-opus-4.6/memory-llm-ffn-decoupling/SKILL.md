---
name: memory-llm-ffn-decoupling
title: "MemoryLLM: Plug-n-Play Interpretable Feed-Forward Memory for Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.00398"
keywords: [FFN Architecture, Memory Mechanism, Interpretability, Static Lookup, Transformer Components]
description: "Decouple feed-forward networks from self-attention by training FFNs on context-free token embeddings instead of residual streams. Enables pre-computation of FFN outputs as static lookup tables for inference efficiency and improved interpretability."
---

# MemoryLLM: Decoupled Feed-Forward Memory

Standard Transformers intertwine attention outputs with FFN inputs through the residual stream, making each token's FFN output context-dependent. MemoryLLM decouples this by training FFNs independently on token-indexed embeddings rather than context-aware representations. This transforms FFNs into interpretable lookup tables that can be pre-computed and stored on-device, enabling faster inference and clearer understanding of what FFNs learn.

The key insight is that FFN parameters can be viewed as a retrieval table where tokens are keys and hidden states are values. By separating FFN training from attention, we get both efficiency and interpretability.

## Core Concept

MemoryLLM operates on architectural decoupling:

1. **Standard Transformer**: `Attention → FFN`, where FFN sees attention-modified residuals
2. **MemoryLLM**: Attention and FFN work independently; FFN receives only token embeddings (context-free)

This enables:
- **Static Lookup**: FFN outputs pre-computed and cached
- **Interpretability**: FFN behavior per-token is visible
- **Flexibility**: FFN outputs can be selectively disabled or modified

## Architecture Overview

- **Independent FFN Training**: FFNs trained only on token embeddings, not residual streams
- **Token-Key-Value Framework**: Think of FFN as (up_proj: key, down_proj: value)
- **Parallel Processing**: Attention and FFN execute independently
- **Static Lookup Layer**: FFN outputs cached as token-indexed lookup tables
- **Flexible Integration**: Optionally use context-aware variant for better performance

## Implementation

### Step 1: Design Context-Free FFN

Create FFN that operates only on token identity, not context.

```python
# Context-free FFN implementation
class MemoryFFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 vocab_size: int):
        """
        FFN operating on token embeddings (context-free).

        Args:
            hidden_dim: Hidden dimension
            intermediate_dim: Intermediate expansion dimension
            vocab_size: Vocabulary size for static lookup
        """
        super().__init__()

        # Up-projection: embed token -> hidden space
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=True)

        # Activation
        self.activation = nn.GELU()

        # Down-projection: hidden -> output
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=True)

        # Optional: Pre-compute lookup table for static inference
        self.static_lookup = None
        self.vocab_size = vocab_size

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply FFN to token embeddings (context-free).

        Args:
            token_embeddings: [batch, seq_len, hidden_dim]

        Returns:
            ffn_output: [batch, seq_len, hidden_dim]
        """
        x = self.up_proj(token_embeddings)
        x = self.activation(x)
        x = self.down_proj(x)
        return x

    def precompute_static_lookup(self, embedding_layer: nn.Module):
        """
        Pre-compute FFN outputs for all vocabulary tokens.

        Args:
            embedding_layer: Model's token embedding layer
        """
        token_ids = torch.arange(self.vocab_size)
        token_embeddings = embedding_layer(token_ids)

        with torch.no_grad():
            ffn_outputs = self.forward(token_embeddings)

        # Store as static lookup: [vocab_size, hidden_dim]
        self.static_lookup = ffn_outputs.detach()
        self.register_buffer('_static_lookup_buffer', self.static_lookup)

    def lookup_static(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Retrieve pre-computed FFN outputs (for inference)."""
        if self.static_lookup is None:
            return None

        return self.static_lookup[token_ids]
```

### Step 2: Create Context-Aware Variant (Optional)

For better performance, support hybrid mode mixing context-free and context-aware.

```python
# Flexible FFN variant
class FlexMemoryFFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 vocab_size: int, context_weight: float = 0.5):
        """
        Flexible FFN: mix context-free and context-aware components.

        Args:
            hidden_dim: Hidden dimension
            intermediate_dim: Intermediate expansion
            vocab_size: Vocabulary size
            context_weight: Balance between context-free (0) and context-aware (1)
        """
        super().__init__()
        self.context_weight = context_weight

        # Context-free component
        self.memory_ffn = MemoryFFN(hidden_dim, intermediate_dim, vocab_size)

        # Context-aware component (if needed)
        if context_weight > 0:
            self.context_up = nn.Linear(hidden_dim, intermediate_dim // 2, bias=True)
            self.context_down = nn.Linear(intermediate_dim // 2, hidden_dim, bias=True)
            self.activation = nn.GELU()

    def forward(self, token_embeddings: torch.Tensor,
               context_residuals: torch.Tensor) -> torch.Tensor:
        """
        Combine context-free and context-aware outputs.

        Args:
            token_embeddings: [batch, seq_len, hidden_dim]
            context_residuals: [batch, seq_len, hidden_dim] (from attention)

        Returns:
            ffn_output: [batch, seq_len, hidden_dim]
        """
        # Context-free component
        context_free_out = self.memory_ffn(token_embeddings)

        if self.context_weight == 0:
            return context_free_out

        # Context-aware component
        context_aware_out = self.context_up(context_residuals)
        context_aware_out = self.activation(context_aware_out)
        context_aware_out = self.context_down(context_aware_out)

        # Blend
        output = ((1 - self.context_weight) * context_free_out +
                 self.context_weight * context_aware_out)

        return output
```

### Step 3: Integrate into Transformer Layer

Modify standard transformer layer to use decoupled FFN.

```python
# Memory-augmented transformer layer
class MemoryTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int,
                 intermediate_dim: int, vocab_size: int):
        """
        Transformer layer with decoupled memory FFN.

        Args:
            hidden_dim: Model dimension
            num_heads: Number of attention heads
            intermediate_dim: FFN intermediate dimension
            vocab_size: Vocabulary size
        """
        super().__init__()

        # Standard attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )

        # Decoupled memory FFN
        self.memory_ffn = MemoryFFN(hidden_dim, intermediate_dim, vocab_size)

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor,
               token_embeddings: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward with independent attention and memory FFN.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] (from residual)
            token_embeddings: [batch, seq_len, hidden_dim] (token identities)
            attention_mask: Optional

        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        # Attention block
        attn_input = self.norm1(hidden_states)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input,
                                        attn_mask=attention_mask)
        hidden_states = hidden_states + attn_output

        # Memory FFN block (operates on token embeddings)
        ffn_input = self.norm2(hidden_states)
        ffn_output = self.memory_ffn(token_embeddings)
        output = hidden_states + ffn_output

        return output
```

### Step 4: Training with Decoupled FFN

Modify training to properly supervise context-free FFN.

```python
# Training with memory FFN
def train_memory_transformer(
    model: nn.Module,
    train_loader,
    embedding_layer: nn.Module,
    num_epochs: int = 10
):
    """
    Train transformer with decoupled memory FFN.

    Args:
        model: Transformer model using MemoryFFN
        train_loader: Training data
        embedding_layer: Token embedding layer
        num_epochs: Training epochs
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            # Get token embeddings for context-free FFN
            token_embeddings = embedding_layer(input_ids)

            # Forward pass
            logits = model(input_ids, token_embeddings)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")

    # Pre-compute static lookups
    for layer in model.layers:
        if hasattr(layer, 'memory_ffn'):
            layer.memory_ffn.precompute_static_lookup(embedding_layer)

    return model
```

### Step 5: Inference with Static Lookups

Use pre-computed FFN outputs for efficiency.

```python
# Inference using static lookups
class MemoryTransformerInference(nn.Module):
    def __init__(self, model: nn.Module):
        """
        Inference wrapper using static FFN lookups.

        Args:
            model: Trained MemoryTransformer
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor,
               use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass using pre-computed FFN lookups.

        Args:
            input_ids: [batch, seq_len]
            use_cache: Whether to cache KV pairs

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        hidden_states = self.model.embed_tokens(input_ids)

        # Shortcut: retrieve pre-computed FFN outputs
        cache = None
        if use_cache:
            cache = []

        for layer in self.model.layers:
            # Attention
            attn_input = layer.norm1(hidden_states)
            if cache:
                attn_output, cache_entry = layer.self_attn(
                    attn_input, use_cache=True
                )
                cache.append(cache_entry)
            else:
                attn_output = layer.self_attn(attn_input)

            hidden_states = hidden_states + attn_output

            # FFN via static lookup
            if hasattr(layer, 'memory_ffn'):
                ffn_lookup = layer.memory_ffn.lookup_static(input_ids)

                if ffn_lookup is not None:
                    # Use pre-computed output directly
                    hidden_states = hidden_states + ffn_lookup
                else:
                    # Fallback to regular FFN
                    ffn_output = layer.memory_ffn(
                        self.model.embed_tokens(input_ids)
                    )
                    hidden_states = hidden_states + ffn_output

        # Output projection
        logits = self.model.lm_head(hidden_states)

        return logits

    def generate(self, input_ids: torch.Tensor,
                max_new_tokens: int = 128) -> torch.Tensor:
        """Generate using memory model."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
```

## Practical Guidance

**When to use MemoryLLM:**
- Scenarios where interpretability matters (understanding what FFNs learn)
- On-device inference with memory constraints
- Models where FFN computation is bottleneck
- Research on separable components in language models

**When not to use:**
- Maximum performance is critical (context-aware FFNs better)
- Models with complex token interdependencies
- Real-time systems where pre-computation overhead matters
- Scenarios with very large vocabularies (static lookup memory)

**Common Pitfalls:**
- Static lookup overhead: Pre-computing for 100K vocabulary requires substantial memory
- Performance regression: Context-free FFN loses modeling capacity; may need larger hidden dims
- Vocabulary mismatch: Re-generating static lookup if vocabulary changes
- Training instability: Context-free FFN may need careful initialization

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| context_weight | 0.0-1.0 | 0.0 = full memory; 0.5+ for better quality |
| intermediate_dim | hidden_dim × 2-4 | Standard; larger for context-free compensation |
| Pre-compute budget | Model allows | Static lookup must fit in available memory |

## Reference

See the full paper at: https://arxiv.org/abs/2602.00398

Key results: Improved interpretability with comparable performance on 250M-1B models. Static lookups enable efficient on-device inference. Flex-MemoryLLM variant bridges performance gap with standard Transformers.
