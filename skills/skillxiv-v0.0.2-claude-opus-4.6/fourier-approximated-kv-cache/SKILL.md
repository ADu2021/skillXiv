---
name: fourier-approximated-kv-cache
title: "Beyond Homogeneous Attention: Memory-Efficient LLMs via Fourier-Approximated KV Cache"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.11886"
keywords: [attention, memory-efficiency, KV-cache, transformers, long-context]
description: "Training-free framework compressing KV caches using Fourier basis functions, exploiting heterogeneous transformer head roles for memory-efficient long-context LLMs."
---

# Beyond Homogeneous Attention: Memory-Efficient LLMs via Fourier-Approximated KV Cache

## Core Concept

FourierAttention addresses memory bottlenecks from Key-Value caches by discovering that transformer head dimensions have heterogeneous roles: lower dimensions handle local context while upper dimensions capture long-range dependencies. The method compresses context-insensitive dimensions using orthogonal Fourier basis functions, maintaining only dominant spectral coefficients while preserving long-context performance.

## Architecture Overview

- **Dimension Analysis Stage**: Identifies which head dimensions require full preservation versus compression through noise perturbation experiments
- **Fourier Compression**: Uses translated Fourier transform during prefilling to convert sequences into fixed-length spectral representations (k ≪ L coefficients)
- **Selective Preservation**: Maintains initial tokens, recent local tokens, and long-context-sensitive dimensions uncompressed, while compressing middle-range tokens in less critical dimensions
- **Custom Triton Kernel**: Optimizes memory efficiency during decoding without sacrificing performance

## Implementation

### Step 1: Analyze Attention Head Dimensions

Perform dimension-level sensitivity analysis to identify which head dimensions are context-insensitive and thus compressible:

```python
def analyze_head_dimensions(attention_module, sequence_length):
    """
    Identifies which dimensions can be compressed using noise perturbation.
    Dimensions with low sensitivity to noise are candidates for compression.
    """
    import torch

    num_heads = attention_module.num_heads
    head_dim = attention_module.head_dim
    sensitivity_scores = torch.zeros(head_dim)

    # Perturb each dimension and measure impact on output
    for dim in range(head_dim):
        original_output = attention_module(queries, keys, values)

        # Add noise to this dimension
        noisy_values = values.clone()
        noise = torch.randn_like(noisy_values[:, :, dim:dim+1]) * 0.1
        noisy_values[:, :, dim:dim+1] += noise

        noisy_output = attention_module(queries, keys, noisy_values)
        sensitivity_scores[dim] = torch.norm(original_output - noisy_output)

    # Lower sensitivity dims are compressible
    return sensitivity_scores
```

### Step 2: Apply Fourier Basis Compression

Project KV cache sequences onto fixed-length Fourier basis functions:

```python
def fourier_compress_kv_cache(kv_sequence, num_coefficients):
    """
    Compresses KV cache using Fourier basis projection.
    Exploits shift-invariance for efficient single-pass computation.

    Args:
        kv_sequence: [batch, seq_len, head_dim] or [batch, seq_len, 2, num_heads, head_dim]
        num_coefficients: k, where k << seq_len
    """
    import torch
    import torch.fft as fft

    # Apply FFT along sequence dimension
    freq_domain = fft.rfft(kv_sequence, dim=1)

    # Retain only k dominant coefficients
    compressed = freq_domain[:, :num_coefficients]

    # Store metadata: original sequence length and coefficient indices
    return {
        'coefficients': compressed,
        'original_length': kv_sequence.shape[1],
        'num_coefficients': num_coefficients
    }
```

### Step 3: Selective Token Preservation

Maintain certain tokens uncompressed based on importance analysis:

```python
def selective_preserve_tokens(kv_cache, preserve_initial=32,
                              preserve_recent=64, compress_middle=True):
    """
    Preserves initial tokens, recent tokens, and important dimensions.
    Compresses middle-range tokens in less critical dimensions.

    Returns mixed representation with full and compressed tokens.
    """
    batch_size, seq_len, head_dim = kv_cache.shape

    # Full representations for important ranges
    initial_tokens = kv_cache[:, :preserve_initial]
    recent_tokens = kv_cache[:, -preserve_recent:]

    # Compress middle tokens
    if compress_middle and seq_len > (preserve_initial + preserve_recent):
        middle_start = preserve_initial
        middle_end = seq_len - preserve_recent
        middle_tokens = kv_cache[:, middle_start:middle_end]

        # Apply Fourier compression with lower coefficient count
        compressed_middle = fourier_compress_kv_cache(
            middle_tokens,
            num_coefficients=int(middle_tokens.shape[1] * 0.3)
        )

        return {
            'initial': initial_tokens,
            'recent': recent_tokens,
            'middle_compressed': compressed_middle
        }

    return {'initial': initial_tokens, 'recent': recent_tokens}
```

### Step 4: Reconstruct During Decoding

Expand compressed representations back to full dimension during forward pass:

```python
def reconstruct_from_fourier(compressed_cache, original_length):
    """
    Reconstructs full KV cache from Fourier coefficients during decoding.
    """
    import torch
    import torch.fft as fft

    coefficients = compressed_cache['coefficients']

    # Pad coefficients to original length
    padded = torch.zeros(
        (coefficients.shape[0], original_length, coefficients.shape[2]),
        device=coefficients.device,
        dtype=coefficients.dtype
    )
    padded[:, :coefficients.shape[1]] = coefficients

    # Inverse FFT to reconstruct sequence
    reconstructed = fft.irfft(padded, n=original_length, dim=1)

    return reconstructed
```

## Practical Guidance

- **Threshold Tuning**: Adjust `num_coefficients` based on target compression ratio (76% tested) and sequence length
- **Dimension Assignment**: Use sensitivity scores to partition head dimensions into full vs. compressed groups
- **Memory-Compute Trade-off**: Fourier reconstruction is fast; main savings come from reduced KV cache size
- **Benchmarking**: Validate on long-context tasks (LongBench, Needle-In-A-Haystack) to ensure quality preservation
- **Integration**: Deploy custom Triton kernel for efficient memory-bound operations during prefilling and decoding

## Reference

Paper: arXiv:2506.11886
Key metrics: 76% KV cache compression, superior performance on long-context benchmarks
Related work: KV cache quantization, sparse attention, position interpolation
