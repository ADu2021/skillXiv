---
name: sparsemm-visual-attention
title: "SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.05344"
keywords: [multimodal-models, attention-sparsity, kv-cache-optimization, visual-language, efficiency]
description: "Discovers that <5% of attention heads process visual information in MLLMs; introduces SparseMM for asymmetric KV-cache allocation achieving 1.38x acceleration and 52% memory reduction."
---

# SparseMM: Head Sparsity in Multimodal LLMs

## Core Concept

Multimodal Large Language Models (MLLMs) allocate uniform computational resources across all attention heads, despite empirical evidence that the vast majority are linguistically-focused. Only approximately 5% of heads actively engage with visual content. SparseMM discovers and exploits this sparsity by assigning asymmetric KV-cache budgets: visual heads receive full cache while text-only heads share a smaller baseline, achieving significant acceleration without accuracy loss. The identification is training-free, using OCR-based spatial grounding to quantify head-level visual relevance.

## Architecture Overview

- **Visual Head Discovery**: OCR-based method identifying <5% of attention heads as "visual heads" across diverse MLLM architectures
- **Training-Free Identification**: No model retraining required; scores based on task-specific spatial grounding patterns
- **Asymmetric Cache Allocation**: Three-part strategy combining local windows, baseline allocation, and score-based priority distribution
- **KV-Cache Optimization Framework**: Prioritizes visually-important heads while maintaining performance guarantees
- **Comprehensive Evaluation**: Tested across DocVQA, OCRBench, TextVQA, MMBench benchmarks
- **Real-Time Acceleration**: Achieves 1.38× speedup with 52% memory reduction

## Implementation

The following code demonstrates visual head discovery and SparseMM cache allocation:

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

class VisualHeadDiscovery:
    """
    Identifies visual heads in MLLMs using OCR-based spatial grounding.
    """
    def __init__(self, model: nn.Module, num_heads: int = 32):
        self.model = model
        self.num_heads = num_heads
        self.visual_scores = None

    def compute_visual_relevance(self, image: torch.Tensor,
                                text_input: str,
                                ocr_boxes: Dict[str, List[Tuple[int, int, int, int]]]) -> Dict[int, float]:
        """
        Compute visual relevance score for each attention head.

        image: (H, W, 3) input image
        text_input: input text/question
        ocr_boxes: dict mapping words to bounding boxes in image
        Returns: dict mapping head_idx to visual_score in [0, 1]
        """
        visual_scores = {}

        # Extract attention patterns from model
        with torch.no_grad():
            # Get attention weights for each head
            attention_patterns = self._extract_attention_patterns(image, text_input)

        # For each head, measure how much it attends to image regions
        for head_idx in range(self.num_heads):
            head_attention = attention_patterns[head_idx]

            # Compute spatial grounding: how well does attention align with OCR regions?
            visual_score = self._compute_grounding_score(head_attention, ocr_boxes, image.shape)

            visual_scores[head_idx] = visual_score

        return visual_scores

    def _extract_attention_patterns(self, image: torch.Tensor,
                                   text_input: str) -> torch.Tensor:
        """Extract attention patterns from model's attention layers."""
        # Simplified: in practice, hook into model's attention layers
        batch_size = 1
        attention_patterns = torch.randn(self.num_heads, 256, 256)  # (num_heads, seq_len, seq_len)
        return attention_patterns

    def _compute_grounding_score(self, attention_weights: torch.Tensor,
                                ocr_boxes: Dict[str, List[Tuple[int, int, int, int]]],
                                image_shape: Tuple[int, int]) -> float:
        """
        Score how well attention aligns with OCR-identified text regions.
        High score = head focuses on image regions with text.
        """
        if not ocr_boxes:
            return 0.0

        h, w = image_shape[:2]

        # Create mask of OCR regions
        ocr_mask = torch.zeros(h, w, dtype=torch.float32)
        for word, boxes in ocr_boxes.items():
            for x1, y1, x2, y2 in boxes:
                ocr_mask[y1:y2, x1:x2] = 1.0

        # Normalize attention to spatial dimensions
        seq_len = attention_weights.shape[0]
        patch_size = h // int(np.sqrt(seq_len))

        # Upsample attention to image dimensions
        spatial_attention = torch.zeros(h, w)
        for i in range(seq_len):
            y = (i // int(np.sqrt(seq_len))) * patch_size
            x = (i % int(np.sqrt(seq_len))) * patch_size
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)

            spatial_attention[y:y_end, x:x_end] = attention_weights[i].mean()

        # Compute overlap with OCR regions
        overlap = (spatial_attention * ocr_mask).sum() / (ocr_mask.sum() + 1e-8)

        return float(overlap.clamp(0, 1))


class SparseMM:
    """
    Sparse Multimodal attention with asymmetric KV-cache allocation.
    """
    def __init__(self, visual_scores: Dict[int, float], num_heads: int,
                 cache_budget_gb: float = 80.0, model_dim: int = 4096):
        self.visual_scores = visual_scores
        self.num_heads = num_heads
        self.cache_budget_gb = cache_budget_gb
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads

        # Rank heads by visual importance
        self.ranked_heads = sorted(visual_scores.items(),
                                  key=lambda x: x[1], reverse=True)

    def allocate_cache(self, sequence_length: int) -> Dict[int, int]:
        """
        Allocate KV-cache tokens per head using three-part strategy:
        1. Local window (all heads)
        2. Uniform baseline (all heads)
        3. Score-based priority (visual heads)
        """
        total_bytes = self.cache_budget_gb * (1024 ** 3)
        bytes_per_token_per_head = 2 * 4 * self.head_dim  # 2 for K,V; 4 bytes for fp32

        # Part 1: Local window (e.g., last 64 tokens)
        window_size = 64
        local_window_budget = window_size * self.num_heads * bytes_per_token_per_head

        # Part 2: Uniform baseline
        remaining_budget = total_bytes - local_window_budget
        baseline_per_head = remaining_budget / (self.num_heads * bytes_per_token_per_head)
        baseline_tokens = int(baseline_per_head)

        # Part 3: Score-based priority for visual heads
        visual_budget = remaining_budget * 0.5  # Allocate 50% of remaining to visual heads
        visual_tokens_per_head = int(visual_budget / (bytes_per_token_per_head * sum(
            score for _, score in self.ranked_heads[:5]  # Assume top 5% are visual
        )))

        # Compute final allocations
        cache_allocation = {}
        for head_idx, visual_score in self.visual_scores.items():
            # Base: window + baseline
            cache_allocation[head_idx] = window_size + baseline_tokens

            # Bonus: if head is visual, add extra cache
            if visual_score > 0.1:  # Visual head threshold
                cache_allocation[head_idx] += int(visual_score * visual_tokens_per_head)

        return cache_allocation

    def apply_sparse_attention(self, query: torch.Tensor,
                             key_cache: torch.Tensor,
                             value_cache: torch.Tensor,
                             cache_allocation: Dict[int, int],
                             head_idx: int) -> torch.Tensor:
        """
        Apply sparse attention using allocated cache for this head.

        query: (batch, seq_len, head_dim)
        key_cache, value_cache: (cache_size, head_dim)
        head_idx: which head is this
        """
        allocated_tokens = cache_allocation.get(head_idx, key_cache.shape[0])

        # Truncate cache to allocated size
        k_sparse = key_cache[-allocated_tokens:] if key_cache.shape[0] > allocated_tokens else key_cache
        v_sparse = value_cache[-allocated_tokens:] if value_cache.shape[0] > allocated_tokens else value_cache

        # Compute attention with sparse cache
        scores = torch.matmul(query, k_sparse.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v_sparse)

        return output


class MLLMWithSparseMM(nn.Module):
    """
    MLLM with SparseMM acceleration integrated.
    """
    def __init__(self, model: nn.Module, num_heads: int = 32):
        super().__init__()
        self.model = model
        self.num_heads = num_heads

        # Discover visual heads
        self.discovery = VisualHeadDiscovery(model, num_heads)
        self.sparse_mm = None
        self.cache_allocation = None

    def discover_and_optimize(self, sample_image: torch.Tensor,
                            sample_text: str,
                            ocr_boxes: Dict[str, List[Tuple[int, int, int, int]]]):
        """
        Discover visual heads and compute cache allocation.
        Call once during initialization.
        """
        visual_scores = self.discovery.compute_visual_relevance(
            sample_image, sample_text, ocr_boxes
        )

        self.sparse_mm = SparseMM(visual_scores, self.num_heads)
        self.cache_allocation = self.sparse_mm.allocate_cache(sequence_length=2048)

    def forward(self, image: torch.Tensor, text: str) -> torch.Tensor:
        """
        Forward pass with sparse attention enabled.
        """
        if self.sparse_mm is None:
            # Fallback to standard attention
            return self.model(image, text)

        # Use sparse attention with optimized cache allocation
        output = self.model(image, text,
                          attention_fn=lambda q, k, v, h: self.sparse_mm.apply_sparse_attention(
                              q, k, v, self.cache_allocation, h
                          ))

        return output
```

## Practical Guidance

**Visual Head Threshold**: Heads with visual score > 0.1 are reliably visual across different MLLMs. Use this threshold for identifying visual heads without tuning.

**OCR-Based Scoring**: Ensure OCR quality before computing visual scores. Poor OCR results in noisy head rankings. Consider using multiple OCR engines and averaging their confidence.

**Cache Budget Allocation**: The three-part strategy (window + baseline + bonus) should maintain at least baseline tokens for all heads to prevent degenerate attention patterns.

**Window Size Selection**: Local window of 64 tokens captures recent context effectively. Increase to 128 for tasks requiring long-range dependencies within recent history.

**Model Variants**: The method works across different MLLM architectures (LLaVA, Qwen-VL, etc.) because visual head sparsity is universal. Test on your target model.

**Deployment Integration**: Apply SparseMM at inference time only; no retraining needed. Hook into attention computation to apply sparse cache lookup by head.

## Reference

SparseMM achieves strong efficiency-accuracy tradeoffs:
- **1.38× real-time acceleration** with sparse cache
- **52% memory reduction** compared to full cache
- **Consistent performance** across DocVQA, OCRBench, TextVQA, MMBench

The discovery that attention heads in MLLMs exhibit extreme sparsity for visual information enables principled optimization without retraining. This makes it immediately applicable to existing deployed models.
