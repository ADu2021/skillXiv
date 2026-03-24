---
name: flash-sampling-efficient-decoding
title: "FlashSampling: Fast and Memory-Efficient Exact Sampling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.15854"
keywords: [Sampling Efficiency, Exact Sampling, Gumbel Trick, Bandwidth Optimization, Token Generation]
description: "Fuse categorical sampling directly into LM-head matrix multiplication to eliminate logits materialization. Use Gumbel noise during computation and hierarchical reduction to achieve 19% token-level latency reduction."
---

# FlashSampling: Fusing Sampling into Matrix Multiplication

Sampling from the output logits is a hidden performance bottleneck in language model inference. Current approaches materialize large logits tensors in high-bandwidth memory (HBM), consuming significant memory bandwidth and requiring multiple GPU kernels after LM-head computation. FlashSampling solves this by fusing sampling directly into the LM-head matrix multiplication: logits are computed tile-by-tile on-chip, Gumbel noise is applied during computation, and only the maximum per-tile is tracked—enabling exact categorical sampling without materializing full logits. This achieves up to 19% reduction in per-token latency across modern GPUs.

The approach remains mathematically exact because argmax decomposes over partitions (max of maximums equals global maximum).

## Core Concept

FlashSampling exploits the mathematical property that sampling from a categorical distribution can be reformulated as:

**Standard Sampling:**
```
logits = LM_head(hidden)                    # [vocab_size]
probs = softmax(logits)
token = categorical_sample(probs)
```

**FlashSampling:**
```
# Compute logits tile-by-tile, maintaining only running maximum
max_logit = -inf
max_idx = -1

for tile_idx in range(num_tiles):
    tile_logits = LM_head_partial(hidden, tile_idx)
    tile_logits += Gumbel_noise
    tile_max_idx = argmax(tile_logits)
    tile_max = tile_logits[tile_max_idx]

    if tile_max > max_logit:
        max_logit = tile_max
        max_idx = tile_idx * tile_size + tile_max_idx

# Token with highest Gumbel-perturbed logit is sampled token
token = max_idx
```

Because argmax commutes with Gumbel perturbation, this is mathematically equivalent to exact sampling.

## Architecture Overview

- **LM-Head Tiling** — Decompose matrix multiplication into independent tiles
- **On-Chip Computation** — Keep tiles in fast cache, avoid HBM spills
- **Gumbel Noise Injection** — Apply noise during tile computation
- **Running Maximization** — Track (value, index) of best tile element
- **Grouped Variant** — Hierarchical reduction for tensor parallelism
- **Kernel Fusion** — Single GPU kernel replaces compute + memory stages
- **Batch Processing** — Maintain efficiency across batch dimensions

## Implementation Steps

Start by implementing the basic tile-based sampling logic.

```python
import torch
import torch.nn.functional as F
import numpy as np

class FlashSampler:
    """Efficient sampling via tile-based computation."""

    def __init__(self, vocab_size, hidden_dim, tile_size=256):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.tile_size = tile_size
        self.num_tiles = (vocab_size + tile_size - 1) // tile_size

    @staticmethod
    def gumbel_sample(logits: torch.Tensor, temperature: float = 1.0):
        """Sample using Gumbel-max trick with temperature scaling."""
        # Generate Gumbel noise
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)

        # Apply noise and temperature
        perturbed = (logits + gumbel_noise) / temperature

        # Argmax gives sample
        return torch.argmax(perturbed, dim=-1)

    def sample_tiled(self, hidden: torch.Tensor, lm_head_weight: torch.Tensor,
                    temperature: float = 1.0) -> torch.Tensor:
        """
        Sample from categorical distribution using tile-based computation.

        Args:
            hidden: [batch_size, hidden_dim]
            lm_head_weight: [vocab_size, hidden_dim]
            temperature: sampling temperature
        """
        batch_size = hidden.size(0)
        best_indices = torch.zeros(batch_size, dtype=torch.long)
        best_scores = torch.full((batch_size,), float('-inf'))

        # Iterate over vocabulary tiles
        for tile_idx in range(self.num_tiles):
            start_vocab = tile_idx * self.tile_size
            end_vocab = min(start_vocab + self.tile_size, self.vocab_size)

            # Get tile of LM-head weights
            tile_weights = lm_head_weight[start_vocab:end_vocab, :]
            # [tile_size, hidden_dim]

            # Compute logits for this tile
            tile_logits = torch.matmul(hidden, tile_weights.t())
            # [batch_size, tile_size]

            # Generate Gumbel noise
            u = torch.rand_like(tile_logits)
            gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)

            # Apply noise and temperature
            tile_scores = (tile_logits + gumbel) / temperature
            # [batch_size, tile_size]

            # Find best in this tile
            tile_best_scores, tile_best_local_idx = torch.max(tile_scores, dim=1)

            # Update global best
            improved = tile_best_scores > best_scores
            best_scores[improved] = tile_best_scores[improved]
            best_indices[improved] = (
                start_vocab + tile_best_local_idx[improved]
            )

        return best_indices

    def sample_grouped(self, hidden: torch.Tensor,
                      lm_head_weight: torch.Tensor,
                      temperature: float = 1.0,
                      group_size: int = 4) -> torch.Tensor:
        """
        Grouped variant for tensor-parallel settings.
        Each device handles vocab_size // group_size tokens.
        """
        # Hierarchical reduction across groups
        group_vocab_size = self.vocab_size // group_size

        best_indices = torch.zeros(hidden.size(0), dtype=torch.long)
        best_scores = torch.full((hidden.size(0),), float('-inf'))

        for group_idx in range(group_size):
            # Get weights for this group
            group_start = group_idx * group_vocab_size
            group_end = (group_idx + 1) * group_vocab_size
            group_weights = lm_head_weight[group_start:group_end, :]

            # Tile within group
            for tile_in_group in range((group_vocab_size + self.tile_size - 1)
                                      // self.tile_size):
                tile_start = group_start + tile_in_group * self.tile_size
                tile_end = min(tile_start + self.tile_size, group_end)

                tile_weights = lm_head_weight[tile_start:tile_end, :]
                tile_logits = torch.matmul(hidden, tile_weights.t())

                # Gumbel noise
                u = torch.rand_like(tile_logits)
                gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
                tile_scores = (tile_logits + gumbel) / temperature

                # Update best
                tile_best_scores, tile_best_idx = torch.max(tile_scores, dim=1)
                improved = tile_best_scores > best_scores
                best_scores[improved] = tile_best_scores[improved]
                best_indices[improved] = tile_start + tile_best_idx[improved]

        return best_indices
```

Now implement the kernel-fused version that operates at the hardware level.

```python
class FusedFlashSamplingKernel:
    """Fused kernel combining LM-head and sampling."""

    def __init__(self, vocab_size, hidden_dim, tile_size=256, device='cuda'):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.tile_size = tile_size
        self.device = device

        # Pre-allocate buffers for tile computation
        self.tile_buffer = torch.zeros((tile_size, hidden_dim),
                                      device=device)
        self.scratch_space = torch.zeros((tile_size,), device=device)

    def fused_lm_head_sample(self, hidden: torch.Tensor,
                           lm_head_weight: torch.Tensor,
                           lm_head_bias: torch.Tensor = None,
                           temperature: float = 1.0) -> torch.Tensor:
        """
        Single fused kernel: LM-head matrix multiplication + sampling.
        This represents what would be implemented in CUDA/Triton.
        """
        batch_size = hidden.size(0)
        best_indices = torch.zeros(batch_size, dtype=torch.long,
                                  device=self.device)
        best_scores = torch.full((batch_size,), float('-inf'),
                                device=self.device)

        # Simulate on-chip tile processing
        for tile_idx in range((self.vocab_size + self.tile_size - 1)
                             // self.tile_size):
            start_vocab = tile_idx * self.tile_size
            end_vocab = min(start_vocab + self.tile_size, self.vocab_size)
            tile_vocab_size = end_vocab - start_vocab

            # In actual kernel: all computation happens on-chip
            # Simulating: fetch weight tile, compute tile
            tile_weights = lm_head_weight[start_vocab:end_vocab, :]

            # Matrix multiply
            tile_logits = torch.matmul(hidden, tile_weights.t())

            if lm_head_bias is not None:
                tile_bias = lm_head_bias[start_vocab:end_vocab]
                tile_logits = tile_logits + tile_bias

            # Generate Gumbel noise
            u = torch.rand_like(tile_logits)
            gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)

            # Perturb and scale
            tile_scores = (tile_logits + gumbel) / temperature

            # Find tile maximum
            tile_max_scores, tile_max_indices = torch.max(tile_scores, dim=1)

            # Update global maximum
            improved = tile_max_scores > best_scores
            best_scores[improved] = tile_max_scores[improved]
            best_indices[improved] = start_vocab + tile_max_indices[improved]

        return best_indices

    def benchmark_vs_standard(self, batch_size=32, hidden_dim=4096,
                             vocab_size=128000, num_iterations=100):
        """Compare latency with standard sampling."""
        import time

        hidden = torch.randn(batch_size, hidden_dim, device=self.device)
        lm_head_weight = torch.randn(vocab_size, hidden_dim,
                                     device=self.device)

        # Warm up
        for _ in range(5):
            _ = self.fused_lm_head_sample(hidden, lm_head_weight)

        # FlashSampling timing
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_iterations):
            _ = self.fused_lm_head_sample(hidden, lm_head_weight)

        torch.cuda.synchronize()
        flash_time = time.time() - start

        # Standard sampling timing
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(num_iterations):
            logits = torch.matmul(hidden, lm_head_weight.t())
            u = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            _ = torch.argmax(logits + gumbel, dim=-1)

        torch.cuda.synchronize()
        standard_time = time.time() - start

        speedup = standard_time / flash_time
        reduction_percent = (1 - 1/speedup) * 100

        print(f"FlashSampling: {flash_time:.3f}s")
        print(f"Standard: {standard_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x ({reduction_percent:.1f}% reduction)")

        return speedup
```

Finally, integrate into inference pipeline and demonstrate usage.

```python
class VLLMWithFlashSampling:
    """vLLM-compatible interface with FlashSampling."""

    def __init__(self, model_name, use_flash_sampling=True):
        self.model = load_model(model_name)
        self.tokenizer = load_tokenizer(model_name)
        self.use_flash_sampling = use_flash_sampling

        if use_flash_sampling:
            vocab_size = len(self.tokenizer)
            hidden_dim = self.model.config.hidden_size
            self.sampler = FusedFlashSamplingKernel(vocab_size, hidden_dim)

    def generate(self, prompt: str, max_tokens=100, temperature=0.7):
        """Generate with optional FlashSampling."""
        input_ids = self.tokenizer.encode(prompt)
        generated = input_ids.copy()

        for _ in range(max_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(torch.tensor([generated]))
                hidden = outputs.hidden_states[-1][:, -1, :]  # Last token
                lm_head_weight = self.model.lm_head.weight

            # Sample next token
            if self.use_flash_sampling:
                next_token = self.sampler.fused_lm_head_sample(
                    hidden, lm_head_weight, temperature=temperature
                )
            else:
                # Standard sampling
                logits = torch.matmul(hidden, lm_head_weight.t())
                u = torch.rand_like(logits)
                gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
                next_token = torch.argmax(logits + gumbel, dim=-1)

            generated.append(next_token.item())

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated)
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Tile size 256-512 works well; smaller tiles improve cache locality, larger reduce kernel launch overhead
- Temperature 0.7-1.0 for typical generation; lower temperatures sharpen distribution
- Use when performing high-throughput inference on modern GPUs (H100, B200, etc.)
- Particularly effective for large vocabulary models (100K+ tokens)
- Benefit amplifies with larger batch sizes (less kernel launch overhead amortization)

**When NOT to use:**
- For CPU inference (no CUDA optimization benefit)
- For very small vocabulary models (< 10K) where logits materialization is already fast
- When using older GPU architectures without good on-chip memory

**Common Pitfalls:**
- Gumbel noise generation becoming bottleneck; batch random number generation across tiles
- Tile boundaries causing alignment issues; use aligned tile sizes
- Numerical instability in log-probability computation; use log-sum-exp tricks
- Not accounting for biases; apply bias during tile computation

## Reference

Paper: [FlashSampling: Fast and Memory-Efficient Exact Sampling](https://arxiv.org/abs/2603.15854)
