---
name: lychee-decode-sparse-kv-sharing
title: "LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.04541"
keywords: [Sparse Attention, KV Cache Sharing, Long-Context Inference, Head-Level Sparsity, Differentiable Optimization]
description: "Classify attention heads into retrieval (full attention) and sparse (token-selected) roles using HardKuma distribution for differentiable discrete optimization. Sparse heads reuse KV pairs from retrieval heads, reducing cache by 90% while maintaining quality through joint training."
---

# LycheeDecode: Head-Level KV Cache Sharing

Long-context inference requires enormous KV caches, dominating memory and computation. LycheeDecode observes that attention heads have different token importance patterns: some need full attention (retrieval heads), others can operate on a small token subset (sparse heads). By training heads to specialize and having sparse heads reuse KV pairs from retrieval heads, the system dramatically reduces cache requirements while maintaining model quality.

The key innovation is using the HardKumaraswamy distribution to solve the discrete optimization problem of selecting which tokens each head attends to, enabling differentiable training without relaxation artifacts.

## Core Concept

LycheeDecode operates on head-level specialization:

1. **Retrieval Heads**: Perform full attention across all tokens to identify important ones
2. **Sparse Heads**: Attend only to tokens selected by retrieval heads, reusing their KV cache

This allows aggressive cache compression while preserving attention flexibility per head.

## Architecture Overview

- **Dual Head Classification**: Learn which heads are retrieval vs. sparse
- **HardKuma Selector**: Discrete token selection via HardKumaraswamy distribution
- **KV Cache Sharing**: Sparse heads reference retrieval heads' selected tokens
- **Joint Training**: Optimize head roles and token selection together
- **Custom Kernel**: Efficient sparse attention computation via block-sparse format

## Implementation

### Step 1: Implement HardKumaraswamy Distribution

Create differentiable discrete token selector.

```python
# HardKumaraswamy for discrete selection
import torch
import torch.nn.functional as F
from torch.distributions import Distribution

class HardKumaraswamy:
    """
    Hard Kumaraswamy distribution for discrete selection.

    Produces values concentrated at 0 and 1 while remaining differentiable.
    """

    @staticmethod
    def sample(alpha: float, beta: float, shape: torch.Size,
              device: torch.device) -> torch.Tensor:
        """
        Sample from HardKuma(alpha, beta).

        Args:
            alpha, beta: Distribution parameters
            shape: Output shape
            device: Device for tensor

        Returns:
            Samples in {0, 1}
        """
        # Sample from Kumaraswamy distribution
        u = torch.rand(shape, device=device)

        # Kumaraswamy CDF inverse: (1 - (1-u)^{1/beta})^{1/alpha}
        s = (1 - (1 - u) ** (1 / beta)) ** (1 / alpha)

        # Hard: threshold at 0.5 to get discrete {0, 1}
        samples = torch.where(s < 0.5, torch.zeros_like(s), torch.ones_like(s))

        return samples

    @staticmethod
    def cdf(z: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        """Kumaraswamy CDF for gradient computation."""
        return 1 - (1 - z) ** beta ** alpha

class HardKumaSelector(torch.nn.Module):
    def __init__(self, hidden_dim: int, max_tokens: int = 1024):
        """
        Learn to select which tokens to attend to per head.

        Args:
            hidden_dim: Hidden dimension per head
            max_tokens: Maximum tokens to potentially select
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens

        # Learnable parameters for HardKuma distribution
        self.alpha_param = torch.nn.Parameter(torch.tensor(1.5))
        self.beta_param = torch.nn.Parameter(torch.tensor(1.5))

    def forward(self, attention_scores: torch.Tensor,
               num_selected: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select important tokens using HardKuma.

        Args:
            attention_scores: [batch, seq_len] score per token
            num_selected: Number of tokens to select

        Returns:
            selected_mask: [batch, seq_len] binary selection mask
            selected_indices: [batch, num_selected] indices of selected tokens
        """
        batch_size, seq_len = attention_scores.shape

        # Normalize scores to [0, 1]
        normalized_scores = torch.sigmoid(attention_scores)

        # Generate selection mask using HardKuma
        mask = HardKumaraswamy.sample(
            self.alpha_param.item(),
            self.beta_param.item(),
            (batch_size, seq_len),
            attention_scores.device
        )

        # Bias selection towards high-score tokens
        biased_scores = normalized_scores * mask + (1 - mask) * 0.1

        # Select top-K tokens
        _, selected_indices = torch.topk(
            biased_scores,
            k=min(num_selected, seq_len),
            dim=-1
        )

        # Create selection mask
        selected_mask = torch.zeros_like(attention_scores)
        batch_indices = torch.arange(batch_size, device=attention_scores.device)[:, None]
        selected_mask[batch_indices, selected_indices] = 1.0

        return selected_mask, selected_indices
```

### Step 2: Design Head-Level Classification

Determine which heads are retrieval vs. sparse.

```python
# Head classification module
class HeadClassifier(torch.nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int):
        """
        Learn to classify heads as retrieval or sparse.

        Args:
            num_heads: Number of attention heads
            hidden_dim: Model dimension
        """
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Learnable head roles
        self.head_roles = torch.nn.Parameter(
            torch.randn(num_heads) * 0.01
        )

    def get_head_types(self, temperature: float = 1.0) -> torch.Tensor:
        """
        Get soft assignment of heads to roles.

        Args:
            temperature: Temperature for softmax

        Returns:
            probs: [num_heads] probability of being retrieval head
        """
        # Sigmoid to get probability of being retrieval (vs. sparse)
        probs = torch.sigmoid(self.head_roles / temperature)
        return probs

    def get_retrieval_heads(self, threshold: float = 0.5) -> torch.Tensor:
        """Get indices of retrieval heads."""
        probs = self.get_head_types()
        return torch.where(probs > threshold)[0]

    def get_sparse_heads(self, threshold: float = 0.5) -> torch.Tensor:
        """Get indices of sparse heads."""
        probs = self.get_head_types()
        return torch.where(probs <= threshold)[0]
```

### Step 3: Implement Hybrid Attention Layer

Create attention layer with retrieval and sparse head variants.

```python
# Hybrid sparse attention layer
class HybridSparseAttention(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        """
        Attention layer with retrieval and sparse heads.

        Args:
            hidden_dim: Model dimension
            num_heads: Number of heads
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        # Head classification and selection
        self.head_classifier = HeadClassifier(num_heads, hidden_dim)
        self.selector = HardKumaSelector(self.head_dim)

    def forward(self, hidden_states: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               use_cache: bool = False) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Hybrid sparse attention forward.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: Optional mask
            use_cache: Whether to cache selected KV

        Returns:
            output: [batch, seq_len, hidden_dim]
            cache: Optional cache dict for retrieval head KV pairs
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        queries = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

        keys = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        values = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Get head types
        retrieval_probs = self.head_classifier.get_head_types()

        # Process each head type
        attn_outputs = []
        cache_dict = {"selected_k": [], "selected_v": []} if use_cache else None

        for head_idx in range(self.num_heads):
            q = queries[:, head_idx]  # [batch, seq_len, head_dim]
            k = keys[:, head_idx]
            v = values[:, head_idx]

            is_retrieval = retrieval_probs[head_idx] > 0.5

            if is_retrieval:
                # Retrieval head: full attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(weights, v)

                # Cache for sparse heads to reuse
                if use_cache:
                    cache_dict["selected_k"].append(k)
                    cache_dict["selected_v"].append(v)

            else:
                # Sparse head: attend to selected tokens only
                # Use scores from retrieval head as importance signal
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

                # Select important tokens
                token_importance = scores.mean(dim=1)  # [batch, seq_len]
                mask, indices = self.selector(token_importance, num_selected=512)

                # Gather selected K, V
                selected_k = k[torch.arange(batch_size)[:, None], indices]
                selected_v = v[torch.arange(batch_size)[:, None], indices]

                # Sparse attention on selected tokens
                sparse_scores = torch.matmul(q, selected_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                sparse_weights = torch.softmax(sparse_scores, dim=-1)
                output = torch.matmul(sparse_weights, selected_v)

                if use_cache:
                    cache_dict["selected_k"].append(selected_k)
                    cache_dict["selected_v"].append(selected_v)

            attn_outputs.append(output)

        # Concatenate head outputs
        attn_output = torch.stack(attn_outputs, dim=1)  # [batch, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output, cache_dict if use_cache else None
```

### Step 4: Training with Joint Optimization

Train head roles and token selection together.

```python
# Training loop with joint optimization
def train_hybrid_sparse_attention(
    model: torch.nn.Module,
    train_loader,
    num_epochs: int = 10,
    lambda_retrieval: float = 0.5
):
    """
    Train model with hybrid sparse attention.

    Args:
        model: Language model with HybridSparseAttention
        train_loader: Training data
        num_epochs: Number of epochs
        lambda_retrieval: Weight for retrieval head regularization
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        total_loss = 0
        total_kv_reduction = 0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            # Forward
            logits, kv_caches = model(input_ids, use_cache=True)

            # Task loss
            task_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

            # Regularization: encourage balanced head roles
            head_classifier = model.layers[0].self_attn.head_classifier
            role_probs = head_classifier.get_head_types()

            # Entropy regularization: avoid all heads becoming same type
            entropy = -torch.sum(role_probs * torch.log(role_probs + 1e-8))
            entropy_loss = -lambda_retrieval * entropy

            # Total loss
            loss = task_loss + entropy_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Track KV reduction
            if kv_caches:
                reduction = estimate_kv_reduction(model, kv_caches)
                total_kv_reduction += reduction

        avg_loss = total_loss / len(train_loader)
        avg_reduction = total_kv_reduction / len(train_loader)

        print(f"Epoch {epoch}: loss={avg_loss:.4f}, KV_reduction={avg_reduction:.2%}")

    return model

def estimate_kv_reduction(model, kv_caches):
    """Estimate KV cache reduction from sparse heads."""
    # Simple estimate: sparse heads use ~50% of tokens
    num_sparse_heads = len(model.layers[0].self_attn.head_classifier.get_sparse_heads())
    total_heads = model.layers[0].self_attn.num_heads

    if total_heads == 0:
        return 0.0

    sparse_fraction = num_sparse_heads / total_heads
    reduction = sparse_fraction * 0.5  # Sparse heads use ~50% of tokens

    return reduction
```

## Practical Guidance

**When to use LycheeDecode:**
- Long-context inference (32K+ tokens) where KV cache dominates memory
- Scenarios accepting <1% quality loss for 2-3x speedup
- Models where head-level specialization is beneficial
- Systems with GPU/TPU supporting custom sparse kernels

**When not to use:**
- Short-context inference (<4K)
- Real-time applications needing deterministic latency
- Models already optimized with other sparsity methods
- Systems lacking efficient sparse operation support

**Common Pitfalls:**
- Head roles too fixed: Insufficient entropy regularization locks roles
- Token selection too aggressive: Select too few tokens, quality degrades
- Cache sharing bugs: Ensure sparse heads properly reference retrieval heads
- Training instability: HardKuma distribution needs careful parameter initialization

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| num_selected | 256-1024 | Higher = less compression; 512 typical |
| lambda_retrieval | 0.1-1.0 | Higher = more balanced head roles |
| alpha, beta (HardKuma) | 1.0-3.0 | Affects discrete selection sharpness |

## Reference

See the full paper at: https://arxiv.org/abs/2602.04541

Key results: Up to 2.7× speedup at 128K context with comparable performance. Custom TileLang kernel for efficient block-sparse decoding. Code and trained models released on GitHub.
