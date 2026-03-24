---
name: infinitevl-linear-sparse-attention
title: "InfiniteVL: Synergizing Linear and Sparse Attention for Highly-Efficient, Unlimited-Input Vision-Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.08829
keywords: [vision-language models, linear attention, sparse attention, long sequence, efficient inference]
description: "Merge sliding window and linear attention (Gated DeltaNet) for unlimited VLM inputs with 3.6× speedup. InfiniteVL handles video understanding at 24 FPS with constant memory—ideal when context length must scale without quadratic overhead."
---

## Overview

InfiniteVL combines sparse window-based attention for local detail with linear attention mechanisms for global efficiency, enabling unlimited input handling without quadratic complexity growth or expanding KV cache issues.

## When to Use

- Vision-language models processing variable-length inputs
- Long video understanding requiring stable 24 FPS performance
- OCR and information-intensive vision tasks
- Unlimited input sequences without memory explosion
- Need for 3.6× speedup over transformer baselines

## When NOT to Use

- Short-context tasks benefiting from standard transformers
- Scenarios where window size is sufficient
- Information-intensive tasks preferring dense attention

## Core Technique

Hybrid attention architecture synergizing sparse and linear mechanisms:

```python
# InfiniteVL: Hybrid linear + sparse attention
class InfiniteVLAttention(nn.Module):
    def __init__(self, dim, window_size=1024):
        super().__init__()
        self.window_attn = SlidingWindowAttention(window_size)
        self.linear_attn = GatedDeltaNetAttention(dim)

    def forward(self, query, key, value, seq_len):
        """Synergize sparse and linear attention."""
        # Sparse attention: sliding window for local context
        local_output = self.window_attn(query, key, value)

        # Linear attention: global efficiency via Gated DeltaNet
        global_output = self.linear_attn(query, key, value)

        # Combine: local detail + global structure
        output = local_output + 0.3 * global_output

        return output
```

## Key Results

- 3.6× inference speedup
- Constant latency and memory with sequence length
- 24 FPS video streaming performance
- Matches leading transformer-based VLMs

## References

- Original paper: https://arxiv.org/abs/2512.08829
- Focus: Efficient long-sequence VLMs
- Domain: Vision-language models, attention mechanisms
