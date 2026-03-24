---
name: avey-recurrence-free-ranker
title: "Don't Pay Attention"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.11305"
keywords: [attention-free, long-context, recurrence-free, ranking, selective-processing]
description: "Avey architecture pairs a ranker with autoregressive processor to select relevant tokens, decoupling context window from sequence length for efficient long-range processing."
---

# Don't Pay Attention

## Core Concept

Avey is a recurrence-free and attention-free architecture that decouples sequence length from context width, enabling processing of arbitrarily long sequences. It pairs a ranker module that identifies top-k relevant token splits with a neural processor that contextualizes only selected tokens. This weighted-selective-split interaction mechanism preserves semantic information regardless of sequence position while avoiding the quadratic scaling of full attention.

## Architecture Overview

- **Ranker Module**: Partitions sequences into splits and identifies top-k relevant splits using MaxSim scoring with normalized relevance weights
- **Neural Processor**: Three integrated units—enricher (expands embeddings), contextualizer (enables inter-embedding interactions), fuser (combines contextualized and bypassed features)
- **Decoupled Context Window**: Context width independent of total sequence length, enabling 64k generalization from 512-token training
- **Weighted Selection**: Soft scoring mechanism that weights splits by relevance rather than hard selection
- **Composition**: Ranker identifies splits → Processor contextualizes → Fuser combines results

## Implementation

### Step 1: Implement the Ranker Module

Create a module that identifies top-k relevant token splits using MaxSim scoring:

```python
import torch
import torch.nn as nn

class RankerModule(nn.Module):
    """
    Identifies top-k relevant token splits from full sequence.
    Uses MaxSim scoring: max similarity between query and candidate splits.
    """
    def __init__(self, embedding_dim, split_size=16):
        super().__init__()
        self.split_size = split_size
        self.embedding_dim = embedding_dim

        # Query projection for relevance computation
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings, k=32):
        """
        Args:
            embeddings: [batch_size, seq_len, embedding_dim]
            k: number of top splits to select

        Returns:
            selected_splits: [batch_size, k, split_size, embedding_dim]
            relevance_scores: [batch_size, k] normalized weights
        """
        batch_size, seq_len, dim = embeddings.shape

        # Partition sequence into splits
        num_splits = (seq_len + self.split_size - 1) // self.split_size
        padded_len = num_splits * self.split_size

        # Pad if necessary
        if padded_len > seq_len:
            padding = torch.zeros(
                batch_size, padded_len - seq_len, dim,
                device=embeddings.device
            )
            padded_embeddings = torch.cat([embeddings, padding], dim=1)
        else:
            padded_embeddings = embeddings

        # Reshape into splits
        splits = padded_embeddings.reshape(
            batch_size, num_splits, self.split_size, dim
        )

        # Compute split-level representations (average pooling)
        split_reps = splits.mean(dim=2)  # [batch, num_splits, dim]

        # Compute query (from first token or learnable)
        query = self.query_proj(embeddings[:, 0])  # [batch, dim]
        query = query.unsqueeze(1)  # [batch, 1, dim]

        # MaxSim: max similarity of query to each split
        similarities = torch.matmul(query, split_reps.transpose(1, 2))
        # [batch, 1, num_splits]
        similarities = similarities.squeeze(1)  # [batch, num_splits]

        # Select top-k splits
        top_k = min(k, num_splits)
        top_scores, top_indices = torch.topk(similarities, top_k, dim=1)

        # Normalize scores to weights
        relevance_weights = torch.softmax(top_scores, dim=1)

        # Gather selected splits
        selected_splits = splits[
            torch.arange(batch_size).unsqueeze(1),
            top_indices
        ]  # [batch, k, split_size, dim]

        return selected_splits, relevance_weights, top_indices
```

### Step 2: Implement the Enricher Unit

Expand token embeddings via position-wise networks:

```python
class EnricherUnit(nn.Module):
    """
    Expands embeddings using position-wise feed-forward network.
    Projects to higher dimension then back down.
    """
    def __init__(self, embedding_dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = int(embedding_dim * expansion_ratio)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, embedding_dim] or [batch, k, split_size, embedding_dim]
        """
        # Flatten and process
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])

        enriched = self.feed_forward(x_flat)
        output = self.norm(enriched + x_flat)

        return output.reshape(shape)
```

### Step 3: Implement the Contextualizer Unit

Enable inter-embedding interactions within selected splits:

```python
class ContextualizerUnit(nn.Module):
    """
    Enables interactions between selected embeddings.
    Uses self-attention or simplified interaction mechanism.
    """
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, selected_splits):
        """
        Args:
            selected_splits: [batch, k, split_size, embedding_dim]
        """
        batch, k, split_size, dim = selected_splits.shape

        # Flatten k and split_size into sequence dimension
        x = selected_splits.reshape(batch, k * split_size, dim)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)

        # Residual + norm
        contextualized = self.norm(attn_out + x)

        return contextualized.reshape(batch, k, split_size, dim)
```

### Step 4: Implement the Fuser Unit

Combine contextualized and original features:

```python
class FuserUnit(nn.Module):
    """
    Combines contextualized features with bypassed original features.
    Learns optimal blending weights.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, contextualized, original_splits):
        """
        Args:
            contextualized: [batch, k, split_size, embedding_dim]
            original_splits: [batch, k, split_size, embedding_dim]
        """
        # Concatenate features
        combined = torch.cat([contextualized, original_splits], dim=-1)

        # Learn gating weights
        shape = combined.shape[:-1]
        combined_flat = combined.reshape(-1, combined.shape[-1])

        gate = self.gate(combined_flat).reshape(*shape, -1)

        # Blend: gate * contextualized + (1 - gate) * original
        fused = gate * contextualized + (1 - gate) * original_splits

        return self.norm(fused)
```

### Step 5: Integrate Ranker and Processor

Combine modules into end-to-end architecture:

```python
class AveyProcessor(nn.Module):
    """
    Complete Avey architecture: Ranker + Processor (Enricher + Contextualizer + Fuser).
    """
    def __init__(self, embedding_dim, split_size=16, k=32):
        super().__init__()
        self.ranker = RankerModule(embedding_dim, split_size)
        self.enricher = EnricherUnit(embedding_dim)
        self.contextualizer = ContextualizerUnit(embedding_dim)
        self.fuser = FuserUnit(embedding_dim)
        self.k = k

    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, seq_len, embedding_dim]

        Returns:
            output: [batch, seq_len] processed representations
        """
        # Step 1: Rank and select splits
        selected_splits, relevance_weights, indices = self.ranker(
            embeddings, k=self.k
        )

        # Step 2: Enrich selected embeddings
        enriched = self.enricher(selected_splits)

        # Step 3: Contextualize within splits
        contextualized = self.contextualizer(enriched)

        # Step 4: Fuse with original splits
        fused = self.fuser(contextualized, selected_splits)

        # Step 5: Weight by relevance and reconstruct
        batch, k, split_size, dim = fused.shape
        weighted_fused = fused * relevance_weights.reshape(batch, k, 1, 1)

        # Aggregate back to sequence
        output = weighted_fused.reshape(batch, -1, dim)

        return output[:, :embeddings.shape[1]]
```

## Practical Guidance

- **Split Size**: Experiment with 8-32 tokens per split; larger splits reduce computation but may lose granularity
- **Top-K Selection**: Use k=32-64 as starting point; adjust based on sequence length and computational budget
- **Relevance Scoring**: MaxSim can be replaced with other scoring mechanisms (cross-entropy, dot product)
- **Training Stability**: Start with small k and increase gradually; monitor gradient flow through ranker
- **Long-Context Evaluation**: Test on Needle-In-A-Haystack and synthetic retrieval tasks to validate extrapolation
- **Comparison Baseline**: Compare against full attention and sparse attention variants on standard benchmarks

## Reference

Paper: arXiv:2506.11305
Key metrics: 64k generalization from 512-token training, superior long-context vs. Mamba/RWKV
Related work: Sparse attention, mixture-of-experts, efficient transformers, retrieval-augmented generation
