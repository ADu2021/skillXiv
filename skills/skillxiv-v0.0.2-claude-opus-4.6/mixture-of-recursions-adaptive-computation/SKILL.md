---
name: mixture-of-recursions-adaptive-computation
title: "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.10524"
keywords: [Adaptive Computation, Parameter Sharing, Recursive Depth, Token-Level Routing, Efficient LLMs]
description: "Build parameter-efficient models that assign different computation depths per token via learned routing, combining weight sharing with dynamic computation complexity. Use when you need to maximize model capacity within compute budgets or create models that allocate compute adaptively based on token complexity."
---

# Mixture-of-Recursions: Token-Aware Adaptive Computation via Dynamic Recursion Depths

Standard Transformers allocate identical computation to all tokens, wasting capacity on simple tokens and limiting it for complex ones. Mixture-of-Recursions (MoR) combines two efficiency gains: parameter sharing through weight recycling and adaptive computation through token-aware routing. The system assigns each token a recursion depth (how many times shared layers are applied), enabling some tokens to skip deep computation entirely while others receive multiple reasoning passes.

The key insight is that recursion depth acts as an "expert dimension" in the MoE framework: instead of routing tokens to different parameter sets, route them to different recursion sequences. Selective key-value caching strategies further reduce memory overhead, making the architecture practical.

## Core Concept

MoR reduces parameters via weight sharing while maintaining expressivity through recursive application. Unlike standard parameter sharing that applies fixed weights a constant number of times, MoR uses lightweight routers to assign each token a custom recursion depth at the start of the computation graph. This allows simple tokens (e.g., punctuation) to complete early, while complex tokens (e.g., key concepts) receive deeper processing.

Two routing strategies govern this: expert-choice (recursion depths select top-k tokens to process deeper) and token-choice (each token gets a fixed routing decision). Two KV caching strategies reduce memory: recursion-wise (store KV only for tokens routed to that depth) and recursive-sharing (reuse first-layer KV across all depths).

## Architecture Overview

- **Shared Parameter Blocks**: 4 parameter-sharing strategies (Cycle, Sequence, Middle-Cycle, Middle-Sequence) that reuse weights across recursion depths while preserving first/last layer uniqueness
- **Routing Routers**: Linear or MLP projections computing scalar scores for recursion depth assignment per token
- **Expert-Choice Routing**: Each recursion depth selects top-k tokens; only selected tokens proceed to next depth via hierarchical filtering
- **Token-Choice Routing**: Single routing decision per token at the start; token commits to fixed recursion sequence via top-1 gating
- **Selective KV Caching**: Store key-value pairs only at assigned recursion depths (recursion-wise) or exclusively at first depth and reuse (recursive-sharing)

## Implementation

### Parameter Sharing Strategies

Four strategies for reusing weights across recursion depths, with Middle-Cycle identified as optimal.

```python
import torch
import torch.nn as nn

class SharedTransformerLayer(nn.Module):
    """Base layer with weight sharing for recursion-based reuse."""

    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, kv_cache=None):
        # Self-attention with optional caching
        attn_out, kv = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, kv

class MoRArchitecture(nn.Module):
    """Mixture-of-Recursions: applies shared layers variable times per token."""

    def __init__(self, hidden_dim, num_heads, mlp_dim, num_recursions=3, sharing_strategy='middle-cycle'):
        super().__init__()
        self.num_recursions = num_recursions
        self.sharing_strategy = sharing_strategy
        self.hidden_dim = hidden_dim

        if sharing_strategy == 'cycle':
            # Single shared layer applied cyclically
            self.shared_layers = nn.ModuleList(
                [SharedTransformerLayer(hidden_dim, num_heads, mlp_dim) for _ in range(1)]
            )
            self.num_unique = 1

        elif sharing_strategy == 'sequence':
            # Shared sequence applied cyclically
            self.shared_layers = nn.ModuleList(
                [SharedTransformerLayer(hidden_dim, num_heads, mlp_dim) for _ in range(3)]
            )
            self.num_unique = 3

        elif sharing_strategy == 'middle-cycle':
            # First and last layers unique, middle layers shared cyclically
            self.first_layer = SharedTransformerLayer(hidden_dim, num_heads, mlp_dim)
            self.middle_layers = nn.ModuleList(
                [SharedTransformerLayer(hidden_dim, num_heads, mlp_dim) for _ in range(2)]
            )
            self.last_layer = SharedTransformerLayer(hidden_dim, num_heads, mlp_dim)
            self.num_unique = 4

        elif sharing_strategy == 'middle-sequence':
            # First and last unique, middle applies in sequence cyclically
            self.first_layer = SharedTransformerLayer(hidden_dim, num_heads, mlp_dim)
            self.middle_sequence = nn.ModuleList(
                [SharedTransformerLayer(hidden_dim, num_heads, mlp_dim) for _ in range(3)]
            )
            self.last_layer = SharedTransformerLayer(hidden_dim, num_heads, mlp_dim)
            self.num_unique = 5

    def apply_with_sharing(self, x, recursion_depth, kv_cache=None):
        """Apply shared layers recursion_depth times."""

        if self.sharing_strategy == 'middle-cycle':
            # First layer
            x, kv = self.first_layer(x, kv_cache)

            # Apply middle layers recursion_depth times
            for _ in range(recursion_depth):
                layer_idx = len(self.middle_layers) % self.num_unique
                x, kv = self.middle_layers[layer_idx](x, kv)

            # Last layer
            x, kv = self.last_layer(x, kv_cache)

        elif self.sharing_strategy == 'cycle':
            # Single layer repeated recursion_depth times
            for _ in range(recursion_depth):
                x, kv = self.shared_layers[0](x, kv_cache)

        return x, kv
```

### Token-Choice Routing

Assign each token a fixed recursion depth decided at the start of computation.

```python
class TokenChoiceRouter(nn.Module):
    """Route each token to a specific recursion depth via top-1 gating."""

    def __init__(self, hidden_dim, num_recursions):
        super().__init__()
        self.num_recursions = num_recursions

        # Linear router predicting recursion depth for each token
        self.router = nn.Linear(hidden_dim, num_recursions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            recursion_depths: (batch, seq_len) integer depths per token
            routing_probs: (batch, seq_len, num_recursions) soft routing probabilities
        """
        # Compute routing scores
        scores = self.router(hidden_states)  # (batch, seq_len, num_recursions)

        # Soft probabilities for training
        routing_probs = self.softmax(scores)

        # Hard decisions (top-1 depth per token)
        recursion_depths = torch.argmax(routing_probs, dim=-1)  # (batch, seq_len)

        # Adjust depths: 1 = base, 2 = +1 recursion, 3 = +2 recursions, etc.
        recursion_depths = recursion_depths + 1

        return recursion_depths, routing_probs

class TokenChoiceMoR(nn.Module):
    """Apply MoR with token-choice routing."""

    def __init__(self, hidden_dim, num_heads, mlp_dim, num_recursions=3):
        super().__init__()
        self.router = TokenChoiceRouter(hidden_dim, num_recursions)
        self.mor_layer = MoRArchitecture(hidden_dim, num_heads, mlp_dim, num_recursions)
        self.num_recursions = num_recursions

    def forward(self, hidden_states, kv_cache=None):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Route tokens to recursion depths
        recursion_depths, routing_probs = self.router(hidden_states)

        # Process each token with its assigned depth
        output = torch.zeros_like(hidden_states)

        for seq_idx in range(seq_len):
            token_depth = recursion_depths[0, seq_idx].item()  # Batch 0, token seq_idx
            token_hidden = hidden_states[:, seq_idx:seq_idx+1, :]

            # Apply recursion layers
            processed, _ = self.mor_layer.apply_with_sharing(token_hidden, token_depth, kv_cache)

            output[:, seq_idx:seq_idx+1, :] = processed

        return output, routing_probs
```

### Expert-Choice Routing with Hierarchical Filtering

Each recursion depth selects top-k tokens; only selected tokens proceed deeper.

```python
class ExpertChoiceRouter(nn.Module):
    """Each recursion depth (expert) selects top-k tokens to process deeper."""

    def __init__(self, hidden_dim, num_recursions, top_k_ratio=0.5):
        super().__init__()
        self.num_recursions = num_recursions
        self.top_k_ratio = top_k_ratio

        # Per-depth routers
        self.routers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_recursions)]
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            routing_masks: List of (batch, seq_len) binary masks per depth
            routing_probs: (batch, seq_len, num_recursions) soft probabilities
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        routing_masks = []
        routing_probs = []

        current_hidden = hidden_states
        active_token_indices = torch.arange(seq_len)

        for depth in range(self.num_recursions):
            # Score tokens at this depth
            scores = self.routers[depth](current_hidden)  # (active_count, 1)

            # Select top-k tokens to proceed
            k = max(1, int(len(active_token_indices) * self.top_k_ratio))
            _, top_k_indices = torch.topk(scores.squeeze(-1), k)

            # Create mask for this depth
            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[active_token_indices[top_k_indices]] = True
            routing_masks.append(mask)

            # Update for next depth (only process selected tokens)
            current_hidden = hidden_states[mask]
            active_token_indices = active_token_indices[mask]

        return routing_masks

class ExpertChoiceMoR(nn.Module):
    """Apply MoR with expert-choice routing and hierarchical filtering."""

    def __init__(self, hidden_dim, num_heads, mlp_dim, num_recursions=3, top_k_ratio=0.5):
        super().__init__()
        self.router = ExpertChoiceRouter(hidden_dim, num_recursions, top_k_ratio)
        self.mor_layer = MoRArchitecture(hidden_dim, num_heads, mlp_dim, num_recursions)

    def forward(self, hidden_states, kv_cache=None):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get routing masks for each depth
        routing_masks = self.router(hidden_states)

        # Apply recursion with hierarchical filtering
        output = hidden_states.clone()

        for depth, mask in enumerate(routing_masks):
            selected_tokens = hidden_states[:, mask, :]

            # Process only selected tokens
            processed, _ = self.mor_layer.apply_with_sharing(selected_tokens, 1, kv_cache)

            output[:, mask, :] = processed

        return output
```

### Selective KV Caching: Recursion-Wise Strategy

Store key-value pairs only at assigned recursion depths to reduce memory.

```python
class RecursionWiseKVCache:
    """
    Cache key-value pairs only for tokens routed to specific recursion depths.
    Saves memory by ~(Nr+1)/2Nr factor where Nr is number of recursions.
    """

    def __init__(self, num_recursions, seq_len, hidden_dim):
        self.num_recursions = num_recursions
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Separate KV cache per recursion depth
        self.kv_caches = [
            {'key': None, 'value': None}
            for _ in range(num_recursions)
        ]

    def cache_kv(self, depth, key, value, token_indices):
        """
        Cache key-value for specific tokens at specific recursion depth.

        Args:
            depth: Recursion depth (0 to num_recursions-1)
            key: (batch, active_tokens, hidden_dim)
            value: (batch, active_tokens, hidden_dim)
            token_indices: Indices of active tokens in full sequence
        """
        if self.kv_caches[depth]['key'] is None:
            # Initialize cache for this depth
            self.kv_caches[depth]['key'] = torch.zeros(
                1, self.seq_len, self.hidden_dim, device=key.device
            )
            self.kv_caches[depth]['value'] = torch.zeros(
                1, self.seq_len, self.hidden_dim, device=value.device
            )

        # Store only for active tokens
        self.kv_caches[depth]['key'][:, token_indices, :] = key
        self.kv_caches[depth]['value'][:, token_indices, :] = value

    def retrieve_kv(self, depth, token_indices):
        """Retrieve cached KV for specific tokens at depth."""
        if self.kv_caches[depth]['key'] is None:
            return None, None

        key = self.kv_caches[depth]['key'][:, token_indices, :]
        value = self.kv_caches[depth]['value'][:, token_indices, :]

        return key, value

    def memory_savings(self):
        """Compute memory saved vs. standard full KV cache."""
        # Standard: all tokens, all depths
        standard_size = self.num_recursions * self.seq_len

        # Recursion-wise: tokens stored only once per routing
        # Approximate: seq_len tokens + (seq_len * top_k_ratio) for next depth
        saved_size = self.seq_len + (self.seq_len * 0.5)

        return (standard_size - saved_size) / standard_size
```

### Recursive Sharing KV Cache

Cache KV pairs only at first recursion block and reuse across depths.

```python
class RecursiveSharingKVCache:
    """
    Store KV exclusively at first recursion block, reuse across all deeper recursions.
    Achieves maximum memory savings at potential inference speed cost.
    """

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.kv_cache = {'key': None, 'value': None}

    def cache_kv_first(self, key, value):
        """Store KV from first recursion block."""
        self.kv_cache['key'] = key
        self.kv_cache['value'] = value

    def reuse_kv_all_depths(self):
        """Return same KV cache for all deeper recursions."""
        return self.kv_cache['key'], self.kv_cache['value']

    def memory_savings(self):
        """Maximal savings: 1/Nr where Nr is number of recursions."""
        # Only first layer KV stored, reused
        return 1.0  # 100% memory savings for KV cache
```

## Practical Guidance

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Recursion Depths (Nr) | 2-4 | Tested range; 3 is standard |
| Parameter Sharing | Middle-Cycle | Optimal strategy (4-5 unique layers, rest shared) |
| Model Scales | 135M-1.7B | Effective for efficient models |
| Top-k Ratio (expert-choice) | 0.5 | Keep 50% of tokens per depth; adjust per domain |
| Router Type | Linear | MLP routers show no benefit; simpler is better |
| Training Data | FineWeb-Edu | Deduplicated high-quality data |
| Training Budget | 20B tokens | Standard LLM pretraining quantity |
| Activation Functions | Sigmoid (expert-choice), Softmax (token-choice) | As specified per routing type |

### When to Use

- Training efficient models with limited compute budgets (mobile, edge)
- Creating models that adaptively allocate reasoning depth per token complexity
- Scenarios where parameter count matters more than latency (constrained training)
- Building multi-task systems where simple tasks need less computation
- Pretraining from scratch with parameter-efficiency as primary goal

### When NOT to Use

- Fine-tuning scenarios (routing patterns learned during pretraining don't transfer)
- Real-time low-latency systems (variable recursion depths cause execution irregularity)
- Streaming/on-device inference (hierarchical filtering requires knowing all token difficulties upfront)
- Models where uniform expressivity is critical (some tokens inevitably get less computation)

### Common Pitfalls

- **Mismatched sharing strategies**: Cycle works for tiny models; Middle-Cycle is necessary for 1B+; test on your target scale
- **Top-k ratio too aggressive**: Using 0.25 causes information bottleneck; stay at 0.5+ for early experiments
- **Ignoring auxiliary losses**: Routing needs load-balancing losses to prevent all tokens going to one depth; include layer auxiliary loss
- **Incompatible KV caching strategies**: Recursive-sharing incompatible with expert-choice (depths select different tokens); use recursion-wise instead
- **Over-optimizing for memory**: Recursive-sharing saves memory but hurts inference speed; measure both latency and memory for your hardware
- **Forgetting validation monitoring**: Routing patterns can degrade early in training; track routing entropy and active token ratios

## Reference

Lin, J., Wang, Y., Zhu, S., et al. (2024). Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. arXiv preprint arXiv:2507.10524.
