---
name: sparse-sparse-attention
title: "SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.20102"
keywords: [Efficient Attention, Sparse Transformers, Attention Gap, Training Optimization]
description: "Overcome the attention gap in sparse transformers by training with both full and sparse attention simultaneously, aligned through bidirectional losses that encourage naturally sparser distributions while maintaining learning capability, enabling efficient inference without capability degradation."
---

# SSA: Sparse Sparse Attention

Sparse attention mechanisms reduce computational costs but suffer from two problems: the attention gap (learned distributions don't match sparse patterns) and the capability gap (sparse-only training underperforms). This skill demonstrates how to overcome both through dual-stream training that alternates between full and sparse attention, aligned via bidirectional losses that encourage naturally sparse distributions.

The core insight is that full and sparse attention should learn together—full attention guides learning while sparse attention optimizes for efficiency.

## Core Concept

Sparse Sparse Attention (SSA) implements:

1. **Dual-Stream Training**: Randomly alternate between full and sparse attention during training
2. **Bidirectional Alignment**: Sparsity loss encourages full→sparse matching; commitment loss keeps sparse→full aligned
3. **Natural Sparsity Emergence**: Full attention gradually becomes sparse through training, reducing gap
4. **Seamless Inference**: Can run with either sparse (fast) or full (capable) mode without retraining

## Architecture Overview

- **Full Attention Stream**: Provides complete gradient signals for all tokens
- **Sparse Attention Stream**: Operates on selected tokens for efficiency
- **Token Selection Mechanism**: Learnable selection of important tokens
- **Sparsity Loss**: Encourages full-attention outputs to match sparse-attention patterns
- **Commitment Loss**: Maintains sparse-attention stability
- **Alignment Computation**: Bidirectional matching between streams

## Implementation Steps

The system trains through alternating attention streams with alignment.

**1. Implement Token Selection Mechanism**

Learn which tokens are important for sparse attention.

```python
class LearnableTokenSelector(torch.nn.Module):
    """
    Learns which tokens are important, enabling sparse attention selection.
    Produces both hard selections (for forward) and soft weights (for gradient flow).
    """
    def __init__(self, hidden_dim=768, sparsity_level=0.5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.sparsity_level = sparsity_level

        # Score network: predict importance of each token
        self.score_network = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

        # Temperature for soft selection
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states):
        """
        Select tokens for sparse attention.
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            selected_mask: (batch, seq_len) binary selection [0, 1]
            soft_weights: (batch, seq_len) soft weights for gradient flow
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute importance scores
        scores = self.score_network(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Determine how many tokens to keep (based on sparsity level)
        num_to_keep = max(1, int(seq_len * (1 - self.sparsity_level)))

        # Hard selection: top-k tokens
        _, top_k_indices = torch.topk(scores, k=num_to_keep, dim=-1)

        # Create hard mask
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, top_k_indices, 1.0)

        # Soft selection: temperature-scaled softmax for gradient flow
        soft_weights = torch.softmax(scores / self.temperature, dim=-1) * seq_len

        # Straight-through estimator: forward with hard, backward with soft
        selected_mask = hard_mask - soft_weights.detach() + soft_weights

        return selected_mask, soft_weights
```

**2. Implement Dual-Stream Attention**

Build both full and sparse attention paths with unified interface.

```python
class DualStreamAttention(torch.nn.Module):
    """
    Implements both full and sparse attention paths.
    Can switch between them dynamically during training.
    """
    def __init__(self, hidden_dim=768, num_heads=8, sparsity_level=0.5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.sparsity_level = sparsity_level

        # Token selector
        self.selector = LearnableTokenSelector(hidden_dim, sparsity_level)

        # Shared QKV projections
        self.qkv_proj = torch.nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def full_attention(self, hidden_states):
        """
        Standard full attention: all tokens attend to all tokens.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(hidden_dim, dim=-1)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Attention: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_dim // self.num_heads)

        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to V
        context = torch.matmul(attention_weights, v)

        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, hidden_dim)

        # Output projection
        output = self.out_proj(context)

        return output, attention_weights

    def sparse_attention(self, hidden_states):
        """
        Sparse attention: only selected tokens participate.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Select tokens
        selection_mask, soft_weights = self.selector(hidden_states)

        # Keep only selected tokens
        selected_indices = torch.nonzero(selection_mask.sum(dim=0) > 0.5, as_tuple=True)[0]
        selected_hidden = hidden_states[:, selected_indices]

        # Project to Q, K, V
        qkv = self.qkv_proj(selected_hidden)
        q, k, v = qkv.split(hidden_dim, dim=-1)

        # Reshape for multi-head
        q = q.reshape(batch_size, -1, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, -1).transpose(1, 2)

        # Sparse attention (on subset)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.hidden_dim // self.num_heads)
        attention_weights = torch.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, -1, hidden_dim)

        # Expand back to full sequence (fill unselected with zeros)
        full_context = torch.zeros(batch_size, seq_len, hidden_dim, device=hidden_states.device)
        full_context[:, selected_indices] = context

        output = self.out_proj(full_context)

        return output, attention_weights

    def forward(self, hidden_states, use_sparse=None):
        """
        Forward pass with random stream selection during training.
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            use_sparse: None (random), True (sparse only), False (full only)
        Returns:
            output: (batch, seq_len, hidden_dim)
            metadata: Dict with attention info
        """
        if use_sparse is None:
            # Random selection during training
            use_sparse = torch.rand(1).item() > 0.5

        if use_sparse:
            output, attention_weights = self.sparse_attention(hidden_states)
            stream_type = 'sparse'
        else:
            output, attention_weights = self.full_attention(hidden_states)
            stream_type = 'full'

        return output, {'stream_type': stream_type, 'attention_weights': attention_weights}
```

**3. Implement Bidirectional Alignment Losses**

Create losses encouraging full-sparse matching.

```python
def compute_alignment_losses(
    full_attention_output,
    sparse_attention_output,
    full_attention_weights,
    sparse_attention_weights,
    lambda_sparsity=1.0,
    lambda_commitment=0.25
):
    """
    Compute bidirectional alignment losses between full and sparse attention.
    Args:
        full_attention_output: (batch, seq_len, hidden_dim)
        sparse_attention_output: (batch, seq_len, hidden_dim)
        full_attention_weights: (batch, num_heads, seq_len, seq_len)
        sparse_attention_weights: (batch, num_heads, sparse_seq_len, sparse_seq_len)
        lambda_sparsity: Weight of sparsity loss
        lambda_commitment: Weight of commitment loss
    Returns:
        total_loss: Combined alignment loss
    """
    # Sparsity loss: encourage full output to match sparse output
    # This encourages full attention to naturally become sparser
    sparsity_loss = torch.nn.functional.mse_loss(
        full_attention_output, sparse_attention_output
    )

    # Commitment loss: keep sparse attention committed to helping generation
    # Sparse attention should be meaningful, not just zero padding
    # Measure by entropy of sparse attention weights
    sparse_entropy = -(sparse_attention_weights * torch.log(sparse_attention_weights + 1e-8)).sum()
    commitment_loss = -sparse_entropy  # Penalize low entropy (too certain)

    # Combined loss
    total_loss = lambda_sparsity * sparsity_loss + lambda_commitment * commitment_loss

    return total_loss, {'sparsity': sparsity_loss, 'commitment': commitment_loss}
```

**4. Build SSA Transformer Layer**

Integrate dual-stream attention into transformer layer.

```python
class SSATransformerLayer(torch.nn.Module):
    """
    Transformer layer with Sparse Sparse Attention.
    Handles alternating full/sparse streams during training.
    """
    def __init__(self, hidden_dim=768, num_heads=8, ffn_dim=3072, sparsity_level=0.5):
        super().__init__()

        # Dual-stream attention
        self.attention = DualStreamAttention(hidden_dim, num_heads, sparsity_level)

        # Layer norms
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ffn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_dim, hidden_dim)
        )

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, hidden_states, training_mode=True):
        """
        Forward pass with SSA.
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            training_mode: Whether to use dual-stream or single inference mode
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # Attention with residual
        normed = self.norm1(hidden_states)
        attention_output, metadata = self.attention(normed, use_sparse=None if training_mode else False)

        # If training, compute alignment losses
        if training_mode and metadata['stream_type'] == 'sparse':
            # Alternate forward again with full attention for alignment
            full_output, _ = self.attention(normed, use_sparse=False)
            alignment_loss, _ = compute_alignment_losses(
                full_output, attention_output, _, metadata['attention_weights']
            )
        else:
            alignment_loss = 0.0

        hidden_states = hidden_states + self.dropout(attention_output)

        # FFN with residual
        normed = self.norm2(hidden_states)
        ffn_output = self.ffn(normed)
        hidden_states = hidden_states + self.dropout(ffn_output)

        return hidden_states, alignment_loss
```

**5. Training Loop with Alternating Streams**

Implement training that switches between full/sparse attention.

```python
def train_ssa_transformer(
    model,
    train_dataloader,
    optimizer,
    num_epochs=10,
    sparsity_schedule='constant'
):
    """
    Training loop for SSA-enabled transformer.
    Alternates between full and sparse attention streams.
    Args:
        model: SSA-enabled transformer
        train_dataloader: Training data iterator
        optimizer: PyTorch optimizer
        num_epochs: Number of training epochs
        sparsity_schedule: How sparsity changes over training (constant, increasing, etc.)
    Returns:
        loss_history: Training loss per step
    """
    loss_history = []

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids, labels = batch

            # Forward pass
            logits = model(input_ids, training_mode=True)

            # Task loss (e.g., next-token prediction)
            task_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1)
            )

            # Alignment loss (encouraging sparse efficiency)
            alignment_loss = 0.0
            for layer in model.layers:
                _, layer_alignment = layer(input_ids, training_mode=True)
                alignment_loss += layer_alignment

            alignment_loss = alignment_loss / len(model.layers)

            # Combined loss
            total_loss = task_loss + 0.1 * alignment_loss

            # Backward and update
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_history.append(total_loss.item())

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={total_loss:.4f}")

    return loss_history
```

**6. Inference with Sparse or Full Mode**

Use trained model with flexible attention mode selection.

```python
def inference_with_ssa(model, input_ids, max_length=512, use_sparse=True):
    """
    Generate text using SSA transformer.
    Can switch between sparse (fast) or full (capable) attention at inference.
    Args:
        model: Trained SSA transformer
        input_ids: (batch, seq_len) input token IDs
        max_length: Maximum generation length
        use_sparse: Whether to use sparse attention (True=fast, False=capable)
    Returns:
        generated_ids: (batch, max_length) generated token IDs
    """
    model.eval()

    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass with specified attention mode
            for layer in model.layers:
                normed = layer.norm1(hidden_states)
                attention_output, _ = layer.attention(normed, use_sparse=use_sparse)
                hidden_states = hidden_states + attention_output
                # ... FFN ...

            # Predict next token
            next_logits = model.output_head(hidden_states[:, -1])
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
```

## Practical Guidance

**When to Use SSA:**
- Transformer models where inference latency is critical
- Long-sequence tasks (>2K tokens) where sparse attention provides significant speedup
- Scenarios requiring both efficiency and quality without retraining

**When NOT to Use:**
- Short sequences (<512 tokens) where sparsity overhead exceeds benefits
- Tasks where attention pattern interpretability is essential
- Models already heavily optimized for full attention

**Key Hyperparameters:**
- `sparsity_level`: Fraction of tokens to prune (0.3-0.7 typical)
- `lambda_sparsity`: Weight of sparsity loss (0.5-2.0)
- `lambda_commitment`: Weight of commitment loss (0.1-0.5)
- `temperature`: Softness of token selection (0.5-2.0)

**Performance Impact:**
- Inference speedup: 2-3× with 50% sparsity
- Training overhead: ~17% additional cost (dual-stream forward/backward)
- Memory reduction: ~30-40% with sparse inference

**Integration Pattern:**
Drop-in replacement for standard transformer layers. Set `use_sparse=False` during training (gets alignment losses), then `use_sparse=True` during inference for efficiency.

## Reference

Research paper: https://arxiv.org/abs/2511.20102
