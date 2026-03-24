---
name: infllm-v2-dense-sparse-switchable-attention
title: "InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.24663"
keywords: [sparse attention, long-context llm, parameter-free adaptation, hardware-efficient transformers, switchable mechanisms]
description: Implement dense-sparse switchable attention enabling LLMs to scale from short to long sequences with 4× speedup and 98-99.7% performance retention, requiring no extra parameters by reusing pretrained attention weights through trainable sparse pattern selection.
---

# InfLLM-V2: Dense-Sparse Switchable Attention

## Outcome

Enable language models pretrained on short sequences (4k tokens) to seamlessly adapt to long-context processing (32k+ tokens) with **4× computational speedup**, **98.1% long-context performance retention**, and **zero additional parameters** through a trainable sparse attention framework that dynamically switches attention patterns based on sequence length.

## Problem Context

Standard Transformer self-attention exhibits quadratic complexity O(n²) in sequence length, creating severe computational and memory bottlenecks when processing long documents. Existing sparse attention approaches introduce substantial new parameters (NSA), disrupt the conventional pretrain-short-finetune-long training workflow, and cause slow convergence. Models need to adapt from short-sequence pretraining (where attention is computationally feasible) to long-sequence inference (where dense attention becomes prohibitive) without architectural mismatch or retraining overhead.

## Core Concept

InfLLM-V2 implements a **parameter-free dense-sparse switchable attention mechanism** that reuses all pretrained dense attention parameters while introducing trainable sparse pattern selection. The framework automatically selects dense attention for short sequences and sparse attention for long sequences, eliminating the parameter explosion of competing methods. Key insight: rather than learning new KV projections for sparse heads, the method extracts sparse tokens from the same pretrained dense attention output space, preserving learned representations while reducing computation.

The mechanism integrates three sparse pattern types—Selected Attention (learns which tokens matter most), Sliding Attention (maintains local context), and Compressed Attention (hierarchical token aggregation)—into a unified sparse pathway controlled by learned routing coefficients. This unification removes redundant output projections and ensures compatibility with standard dense attention training.

## Architecture Overview

**Switchable Attention Framework:**

- **Dense Pathway**: Full self-attention for sequences below a learnable threshold length, using original pretrained weights
- **Sparse Pathway**: Multi-pattern selection combining selected, sliding, and compressed attention for longer sequences
- **Pattern Router**: Trainable coefficients determining contribution weight of each sparse pattern without adding KV projection parameters
- **Hardware Kernel**: Two-pass CUDA implementation with LSE approximation fusing head-group summation into FlashAttention loop, reducing GPU memory transfers

**Three Core Innovations:**

- **Parameter-Free Adaptation**: No new KV projection weights; sparse patterns reuse dense attention's pretrained output space
- **Unified Sparse Patterns**: Consolidated Selected + Sliding + Compressed attention into single module removing redundant pathways
- **Sequence-Length Switching**: Automatic mode selection based on input length with no training instability or architectural mismatch

## Implementation

### Step 1: Dense Attention Layer Foundation

Define the base dense attention layer reusing pretrained parameters. This serves as the initialization for both dense and sparse pathways.

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class DenseAttention(nn.Module):
    """
    Standard dense self-attention layer with pretrained parameters.
    Used directly for short sequences, parameter source for sparse variants.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, 1, seq_len, seq_len) or None

        Returns:
            output: (batch_size, seq_len, hidden_size)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)

        output = self.out_proj(context)

        return output, attn_weights
```

### Step 2: Sparse Pattern Selection Module

Implement the three sparse attention patterns that reuse the pretrained KV space without additional projection parameters.

```python
class SparseAttentionPatterns(nn.Module):
    """
    Multi-pattern sparse attention that selects relevant tokens without new parameters.
    Reuses K and V from dense pathway, applies pattern-specific masking.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 64,
        topk_ratio: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.topk_ratio = topk_ratio
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def selected_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token selection: attend to top-k most relevant tokens per position.
        q, k, v shape: (batch_size, num_heads, seq_len, head_dim)
        Returns: (batch_size, num_heads, seq_len, head_dim)
        """
        seq_len = q.shape[2]
        k_tokens = max(1, int(seq_len * self.topk_ratio))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Select top-k tokens for each query position
        topk_scores, topk_indices = torch.topk(scores, k=k_tokens, dim=-1)

        # Create sparse attention mask
        attn_weights = torch.full_like(scores, float('-inf'))
        attn_weights.scatter_(-1, topk_indices, topk_scores)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = torch.where(torch.isinf(attn_weights), 0.0, attn_weights)

        context = torch.matmul(attn_weights, v)
        return context

    def sliding_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Local attention: attend to surrounding tokens within window.
        q, k, v shape: (batch_size, num_heads, seq_len, head_dim)
        Returns: (batch_size, num_heads, seq_len, head_dim)
        """
        seq_len = q.shape[2]
        window_size = self.block_size

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create sliding window mask
        mask = torch.ones_like(scores, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[:, :, i, :start] = False
            mask[:, :, i, end:] = False

        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.where(torch.isinf(attn_weights), 0.0, attn_weights)

        context = torch.matmul(attn_weights, v)
        return context

    def compressed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hierarchical compression: attend to block-averaged tokens.
        q, k, v shape: (batch_size, num_heads, seq_len, head_dim)
        Returns: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.block_size

        # Compress k and v by averaging blocks
        num_blocks = (seq_len + block_size - 1) // block_size

        # Pad sequences to multiple of block_size
        padded_len = num_blocks * block_size
        pad_len = padded_len - seq_len

        if pad_len > 0:
            k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad_len), value=0.0)
            v_padded = torch.nn.functional.pad(v, (0, 0, 0, pad_len), value=0.0)
        else:
            k_padded = k
            v_padded = v

        # Reshape to blocks and average
        k_blocks = k_padded.view(batch_size, num_heads, num_blocks, block_size, head_dim)
        v_blocks = v_padded.view(batch_size, num_heads, num_blocks, block_size, head_dim)

        k_compressed = k_blocks.mean(dim=3)  # (batch, heads, blocks, head_dim)
        v_compressed = v_blocks.mean(dim=3)

        # Attend from full sequence to compressed tokens
        scores = torch.matmul(q, k_compressed.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v_compressed)

        # Expand back to original sequence length
        context = context.unsqueeze(3).expand(-1, -1, -1, block_size, -1)
        context = context.contiguous().view(batch_size, num_heads, padded_len, head_dim)

        if pad_len > 0:
            context = context[:, :, :seq_len, :]

        return context

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all three sparse patterns.
        Returns combined outputs without additional parameters.
        """
        selected = self.selected_attention(q, k, v)
        sliding = self.sliding_attention(q, k, v)
        compressed = self.compressed_attention(q, k, v)

        return selected, sliding, compressed
```

### Step 3: Switchable Attention with Pattern Router

Implement the core switchable mechanism with trainable pattern coefficients and automatic dense/sparse selection.

```python
class SwitchableAttention(nn.Module):
    """
    Core InfLLM-V2 mechanism: dense for short sequences, sparse for long.
    Reuses all dense attention parameters via trainable pattern routing.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        dense_threshold: int = 2048,
        block_size: int = 64,
        topk_ratio: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dense_threshold = dense_threshold

        # Shared pretrained attention layer (no extra KV projections)
        self.dense_attn = DenseAttention(hidden_size, num_heads, dropout)

        # Sparse patterns reusing K, V from dense pathway
        self.sparse_patterns = SparseAttentionPatterns(
            hidden_size, num_heads, block_size, topk_ratio
        )

        # Trainable pattern router: weights for selected, sliding, compressed
        # These are the ONLY new parameters introduced
        self.pattern_weights = nn.Parameter(
            torch.ones(3) / 3.0  # Initialize equally
        )

        self.head_dim = hidden_size // num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Automatic switching between dense and sparse attention.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: optional attention mask
            use_cache: whether to return attention weights

        Returns:
            output: (batch_size, seq_len, hidden_size)
            attention_weights: debug output or cached weights
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute dense attention (reusing pretrained parameters)
        dense_output, dense_weights = self.dense_attn(hidden_states, attention_mask)

        # For short sequences, use dense attention directly
        if seq_len <= self.dense_threshold:
            return dense_output, dense_weights

        # For long sequences, compute sparse patterns
        # Extract projections from dense pathway
        q = self.dense_attn.q_proj(hidden_states)
        k = self.dense_attn.k_proj(hidden_states)
        v = self.dense_attn.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute sparse patterns without new parameters
        selected, sliding, compressed = self.sparse_patterns(q, k, v)

        # Route patterns using trainable coefficients
        pattern_weights = torch.softmax(self.pattern_weights, dim=0)
        sparse_context = (
            pattern_weights[0] * selected +
            pattern_weights[1] * sliding +
            pattern_weights[2] * compressed
        )

        # Reshape back to sequence format
        sparse_context = sparse_context.transpose(1, 2).contiguous()
        sparse_context = sparse_context.view(batch_size, seq_len, self.hidden_size)

        # Apply output projection (shared, no extra parameters)
        sparse_output = self.dense_attn.out_proj(sparse_context)

        # Optional: linear blend between dense and sparse for stability
        # During training, gradually shift toward sparse patterns
        return sparse_output, None
```

### Step 4: Integration into Transformer Block

Integrate switchable attention into a standard Transformer layer for seamless adoption.

```python
class TransformerBlockWithSwitchableAttention(nn.Module):
    """
    Standard Transformer block with InfLLM-V2 switchable attention.
    Drops into existing models with minimal changes.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        dense_threshold: int = 2048,
    ):
        super().__init__()

        self.attention = SwitchableAttention(
            hidden_size,
            num_heads,
            dropout,
            dense_threshold,
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Standard FFN layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pre-norm transformer block with switchable attention.
        """
        # Attention with residual
        norm_hidden = self.norm1(hidden_states)
        attn_output, _ = self.attention(norm_hidden, attention_mask)
        hidden_states = hidden_states + attn_output

        # FFN with residual
        norm_hidden = self.norm2(hidden_states)
        mlp_output = self.mlp(norm_hidden)
        hidden_states = hidden_states + mlp_output

        return hidden_states
```

### Step 5: Training and Fine-tuning Procedure

Configure training to enable seamless short-to-long adaptation without architectural mismatch.

```python
def create_switchable_model(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    vocab_size: int,
    dense_threshold: int = 2048,
) -> nn.Module:
    """
    Construct full language model with switchable attention throughout.
    Compatible with standard training pipelines.
    """
    layers = nn.ModuleList([
        TransformerBlockWithSwitchableAttention(
            hidden_size,
            num_heads,
            intermediate_size,
            dense_threshold=dense_threshold,
        )
        for _ in range(num_layers)
    ])

    embedding = nn.Embedding(vocab_size, hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size)

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = embedding
            self.layers = layers
            self.norm = nn.LayerNorm(hidden_size)
            self.lm_head = lm_head

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            hidden_states = self.embedding(input_ids)

            for layer in self.layers:
                hidden_states = layer(hidden_states)

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)

            return logits

    return LanguageModel()


def finetune_for_long_context(
    model: nn.Module,
    train_dataloader,
    optimizer,
    num_epochs: int = 2,
    device: str = "cuda",
    gradient_accumulation_steps: int = 4,
):
    """
    Fine-tune pretrained short-context model for long-context capability.

    Key procedure:
    1. Model starts with pretrained dense attention parameters
    2. Pattern router initialized equally, then learns during fine-tuning
    3. No architectural changes, no new KV projections
    4. Gradual activation of sparse patterns as sequence length increases
    """
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
            )

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_dataloader):.4f}")
```

## Practical Guidance

### Hyperparameters and Configuration

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `dense_threshold` | 2048-4096 | Sequence length below which dense attention is used. Adjust based on GPU memory. |
| `block_size` | 64-128 | Window size for sliding attention and block size for compression. |
| `topk_ratio` | 0.05-0.2 | Fraction of tokens selected in selected-attention pattern. Lower = faster but potentially lower quality. |
| `pattern_initialization` | [0.33, 0.33, 0.33] | Initialize pattern weights equally; training adapts automatically. |
| `num_fine_tune_epochs` | 1-3 | Few epochs sufficient since base parameters frozen. |
| `learning_rate_sparse_router` | 1e-4 to 5e-4 | Only pattern_weights need tuning; keep LR conservative. |

### When to Use

- Deploying pretrained LLMs that require long-context capability without retraining from scratch
- Inference scenarios with variable sequence lengths requiring dynamic efficiency
- Models with computational constraints (mobile, edge devices with GPU)
- Fine-tuning budgets limited to a few epochs after pretraining
- Applications requiring seamless transition from short to long contexts without latency spikes
- Multi-modal systems where context length varies significantly per sample

### When NOT to Use

- Very long sequences (100k+ tokens) where even sparse patterns become expensive; consider pure sparse methods or retrieval-augmented approaches
- Pretraining from scratch; use standard dense attention or alternative sparse methods without the switching overhead
- Models where maintaining exact dense attention behavior is critical (e.g., exact replication of reference outputs); sparse approximation introduces mathematical differences
- Scenarios requiring causal attention without forward peeking; ensure sliding window respects causality (only attend to past tokens)
- Hardware without efficient sparse attention kernels (CPUs, older GPUs); performance gains diminish without optimized CUDA implementation
- Tasks requiring all-to-all token interaction (e.g., some music or video tasks); sparsity may hurt performance

### Common Pitfalls

**Threshold too low**: Setting `dense_threshold` too low forces sparse patterns for short sequences where dense is already efficient. Set threshold where GPU memory becomes the bottleneck, typically 2-4k tokens on A100.

**Unbalanced pattern weights**: If one sparse pattern dominates, others contribute little. Monitor pattern weight evolution; add entropy regularization if weights collapse to single pattern.

**Ignoring position encodings**: Sparse attention may not properly extend pretrained position embeddings beyond their training range. Use position interpolation or ALiBi if extending significantly beyond pretraining length.

**Gradient flow to frozen parameters**: Dense attention parameters are frozen during fine-tuning to preserve learned representations. Ensure gradients only flow through pattern_weights, not through q_proj, k_proj, v_proj if using frozen parameter setup.

**Memory bottleneck in pattern computation**: Selected attention requires computing full similarity matrix before top-k selection. For very long sequences, compute top-k in blocks or use approximate methods like LSH to reduce memory.

**Training instability during transition**: Some models show training loss spikes when fine-tuning switches between dense and sparse pathways. Use gradient warmup or gradual threshold decay: start with high threshold, gradually lower during training to activate sparsity.

## Reference

**Paper**: InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation
**arXiv**: https://arxiv.org/abs/2509.24663
**Authors**: OpenBMB
**Reference Implementation**: https://huggingface.co/openbmb/InfLLM-V2-Long-Sparse-Base
**Code Repository**: https://github.com/OpenBMB/infllmv2_cuda_impl
