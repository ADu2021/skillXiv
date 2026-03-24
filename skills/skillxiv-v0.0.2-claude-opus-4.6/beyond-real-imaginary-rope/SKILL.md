---
name: beyond-real-imaginary-rope
title: "Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.07525
keywords: [position embeddings, long-context, rotary embeddings, RoPE, extended attention]
description: "Improve long-context performance by incorporating imaginary components discarded in standard RoPE implementations. Use phase information from complex-valued attention for richer positional encoding—especially valuable as context length increases beyond normal ranges."
---

## Overview

Standard RoPE implementations discard the imaginary component of complex-valued dot products during attention calculations. This work leverages that discarded phase information to create improved position encodings, capturing additional positional details essential for modeling long-range dependencies in language models.

## When to Use

- Long-context language models where standard RoPE underperforms
- Scenarios requiring strong performance as context length increases
- Models needing richer positional information for distant dependencies
- Applications with variable-length inputs exceeding typical ranges
- Improving existing RoPE-based models without architectural changes

## When NOT to Use

- Standard-length context where RoPE works adequately
- Models using other positional encoding schemes (ALiBi, relative biases)
- Scenarios where computational overhead is critical
- Applications already achieving satisfactory long-context performance

## Core Technique

Dual-component attention leveraging complex-valued representations:

```python
# Imaginary extension of RoPE
class ExtendedRotaryEmbedding:
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def compute_frequencies(self):
        """
        Compute frequency components for rotation matrices.
        Uses exponential scaling for long-context handling.
        """
        # Frequency scaling for each dimension
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        return inv_freq

    def apply_rope(self, x, seq_positions):
        """
        Apply rotary embeddings with both real and imaginary components.
        Preserves phase information discarded in standard RoPE.
        """
        batch_size, seq_len, dim = x.shape
        inv_freq = self.compute_frequencies()

        # Compute position-dependent rotation angles
        # t is position index, dimension-specific frequency inv_freq[i]
        angles = torch.einsum('i, j -> ij', seq_positions, inv_freq)

        # Create complex rotation matrices
        # Standard RoPE uses: cos(angles) - i*sin(angles)
        real_part = torch.cos(angles)
        imag_part = torch.sin(angles)

        # Create complex representation
        complex_rotation = torch.complex(real_part, imag_part)

        # Apply to input (treating as complex values)
        # Split input into real and imaginary parts
        x_real = x[..., :dim//2]
        x_imag = x[..., dim//2:]
        x_complex = torch.complex(x_real, x_imag)

        # Multiply: (a+bi)(cos+i*sin) preserves phase information
        rotated_complex = x_complex * complex_rotation.unsqueeze(0)

        # Extract both components - don't discard imaginary!
        output = torch.cat([
            rotated_complex.real,
            rotated_complex.imag
        ], dim=-1)

        return output

    def compute_dual_component_attention(self, q, k, v):
        """
        Dual-component attention using both real and imaginary parts.
        Captures phase information for enhanced positional awareness.
        """
        # Apply rotary embeddings preserving imaginary component
        q_rotated = self.apply_rope(q, positions=torch.arange(q.shape[1]))
        k_rotated = self.apply_rope(k, positions=torch.arange(k.shape[1]))

        # Compute attention with full complex representation
        # Standard attention: (Q @ K^T) / sqrt(d)
        # Enhanced: uses both magnitude and phase of Q @ K^T
        attention_scores = torch.einsum('bqd, bkd -> bqk', q_rotated, k_rotated)
        attention_scores = attention_scores / math.sqrt(q.shape[-1])

        # Softmax along sequence dimension
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply to values
        output = torch.einsum('bqk, bkd -> bqd', attention_weights, v)

        return output
```

The approach reconstructs complete complex-valued attention, theoretically providing richer positional encoding for extended sequences.

## Key Results

- Consistent performance improvements across long-context benchmarks
- Gains intensify as context length increases
- Preserves computational efficiency of RoPE
- Drop-in replacement for standard RoPE implementations
- Code available at authors' GitHub repository

## Implementation Notes

- Works as direct replacement for standard RoPE
- Preserves phase information discarded in standard implementations
- Enhanced positional detail especially valuable for long contexts
- Maintains efficient O(n) computational complexity
- Compatible with existing attention mechanisms

## References

- Original paper: https://arxiv.org/abs/2512.07525
- Focus: Long-context language modeling
- Domain: Position embeddings, transformer architecture
