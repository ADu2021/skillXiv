---
name: every-token-counts-16m-context
title: "Every Token Counts: Ultra-Long Context Modeling with Hierarchical Sparse Attention"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2511.23319
keywords: [long-context, sparse-attention, hierarchical-retrieval, moe-models, length-generalization]
description: "Chunk-based landmark-guided sparse attention enabling 16M-token context windows with 90%+ retrieval accuracy on in-context tasks. Use when processing ultra-long documents where full attention is prohibitive but context selection must be dynamic."
---

## Summary

Every Token Counts introduces Hierarchical Sparse Attention (HSA), which enables efficient ultra-long context modeling by partitioning sequences into fixed-length chunks and using learned landmark representations to retrieve top-k most relevant past chunks for each token. This approach satisfies sparsity, random-access flexibility, and length generalization while an 8B-parameter MoE model achieves 90%+ accuracy on 16M-token retrieval tasks.

## Core Technique

**Chunk-Based Retrieval:** Divide input sequence into non-overlapping chunks of fixed length. Create a learnable landmark representation for each chunk capturing its semantic content. For each query position, score all landmarks to select top-k most relevant chunks.

**Landmark-Guided Attention:** Landmarks are learned dense vectors computed from chunk content (via a small encoder). During inference, compute similarity scores between query and all landmarks in O(n/chunk_size) time, then attend to selected chunks in O(k * chunk_size) time—yielding O(k * chunk_size + n/chunk_size) complexity versus O(n²) for full attention.

**Length Generalization:** The architecture extrapolates beyond training context lengths because the retrieval mechanism doesn't depend on absolute position—just relative landmark similarity. Attention fuses results using retrieval scores as weights.

## Implementation

**Landmark computation:** For each chunk, compute landmark vector via: landmark_i = encode(mean_pooling(chunk_i_tokens)). Use a learned transformer layer with cross-attention to refine landmarks based on query context.

**Top-k retrieval:** For query at position q, compute: scores = similarity(query_embedding, all_landmarks), then k_indices = argsort(scores)[-k:]. Retrieve the corresponding chunks.

**Attention fusion:** Attend to each selected chunk with standard multi-head attention, then fuse results: output = sum(retrieval_score_i * attention(q, chunk_i) for i in k_indices).

## When to Use

- Processing documents longer than 100K tokens where full attention is infeasible
- Retrieval-based tasks requiring dynamic context selection
- Long-context understanding of books, code repositories, chat histories
- Applications needing both sparsity and position-independent generalization

## When NOT to Use

- Tasks with relatively short context (under 4K tokens where full attention is fast)
- Applications requiring all historical tokens equally in every decision
- Scenarios where landmark computation overhead dominates (very short chunks)
- Tasks sensitive to hierarchical approximation errors in attention patterns

## Key References

- Retrieval-augmented attention mechanisms for long contexts
- Landmark-based sparse attention design patterns
- MoE architectures for scaling ultra-long context models
