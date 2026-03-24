---
name: lite-attention-temporal-sparse
title: "LiteAttention: Temporal Sparse Attention for Diffusion Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.11062"
keywords: [Sparse Attention, Diffusion, Video Generation, Temporal Coherence, Efficiency]
description: "Accelerate video diffusion generation by exploiting temporal attention sparsity—skip redundant attention tiles across denoising steps using persistent skip masks, achieving 40% speedup with quality retention."
---

# Accelerate Diffusion Video Generation with Temporal Sparse Attention

Video diffusion transformers spend significant computational budget computing self-attention across redundant spatial regions across denoising timesteps. LiteAttention exploits temporal coherence: tiles deemed non-essential at denoising step t remain non-essential at t+δ. By maintaining a persistent skip-mask and propagating skip decisions forward, LiteAttention reduces self-attention cost without dynamic recomputation at each step.

The key insight is that attention sparsity patterns are stable across adjacent denoising steps. Rather than recomputing which regions to skip (dynamic, expensive) or using fixed patterns throughout (inflexible), LiteAttention propagates early sparsity decisions, combining adaptivity with efficiency.

## Core Concept

Video diffusion models iterate through T denoising steps, each computing full self-attention on spatial tiles (flattened image regions). Most tiles have negligible attention weights across most steps—they can be safely skipped. However, dynamically identifying skippable tiles at every step multiplies computational overhead.

LiteAttention maintains a **Skip-Mask** tracking which tiles are non-essential. The mask is computed once early (step 0 or 1) using **QK-Skip**: a simple condition on query-key dot products. Tiles with max local QK score significantly smaller than cumulative maximum are flagged as skippable. This mask propagates across timesteps, and only if attention patterns diverge significantly does the mask update.

## Architecture Overview

- **Skip-Mask Data Structure**: Bitmask (or Skip-List for high sparsity) encoding which attention tiles can be skipped per timestep
- **QK-Skip Algorithm**: Fast condition checking max(QK) vs cumulative max to identify low-attention tiles
- **Accumulated-Error Calibration**: Assign variable error budgets to timesteps (early steps tolerate less error)
- **Hardware Integration**: Built atop FlashAttention3 for H100 GPUs; skip logic integrates into softmax pipeline
- **Persistent Propagation**: Skip decisions carry forward unless attention distribution shifts significantly

## Implementation Steps

**Step 1: Sparsity Detection.** Identify tiles with negligible QK scores early in denoising.

```python
def compute_qk_skip_mask(queries, keys, threshold_ratio=0.1):
    """
    Compute skip mask using QK-Skip condition.
    Tiles with max QK << cumulative_max are marked as skippable.
    queries, keys: (batch, num_tiles, head_dim)
    threshold_ratio: tile is skipped if max_qk < threshold_ratio * cumulative_max
    """
    # Compute attention scores
    qk_scores = queries @ keys.transpose(-1, -2)  # (batch, num_tiles, num_tiles)

    skip_mask = []
    for b in range(qk_scores.shape[0]):
        local_maxs = qk_scores[b].max(dim=-1).values  # max per tile
        cumulative_max = local_maxs.max()
        skip_threshold = threshold_ratio * cumulative_max

        # Skip tile if its local max is below threshold
        skippable = local_maxs < skip_threshold
        skip_mask.append(skippable)

    return torch.stack(skip_mask)
```

**Step 2: Mask Propagation.** Maintain skip mask across denoising steps; update selectively.

```python
class TemporalSkipAttention(nn.Module):
    def __init__(self, num_tiles, update_frequency=5):
        super().__init__()
        self.num_tiles = num_tiles
        self.skip_mask = None
        self.update_frequency = update_frequency  # recompute every N steps

    def forward(self, queries, keys, values, timestep):
        """
        Compute attention with persistent skip mask.
        timestep: current denoising step (0 to T)
        """
        # Recompute mask periodically or at first step
        if self.skip_mask is None or timestep % self.update_frequency == 0:
            self.skip_mask = compute_qk_skip_mask(queries, keys)

        # Compute full attention scores
        qk = queries @ keys.transpose(-1, -2)
        qk = qk / math.sqrt(queries.shape[-1])

        # Apply skip mask: set masked positions to -inf
        qk[self.skip_mask] = float('-inf')

        # Softmax and output
        attn_weights = torch.softmax(qk, dim=-1)
        attn_weights[self.skip_mask] = 0  # zero-out masked attention
        output = attn_weights @ values

        return output
```

**Step 3: Error Calibration.** Assign variable tolerance to timesteps based on impact on final output.

```python
def calibrate_error_budgets(num_timesteps, early_step_ratio=0.3):
    """
    Early denoising steps affect final output more; allocate smaller error budgets there.
    Returns per-timestep error tolerance multiplier.
    """
    error_budgets = []
    for t in range(num_timesteps):
        # Earlier steps get tighter error budgets (lower multiplier = smaller tolerance)
        if t < int(num_timesteps * early_step_ratio):
            # Scale from 0.5 to 1.0 across early steps
            budget = 0.5 + 0.5 * (t / int(num_timesteps * early_step_ratio))
        else:
            # Later steps tolerate more error
            budget = 1.0

        error_budgets.append(budget)

    return error_budgets
```

## Practical Guidance

**When to Use:** Video diffusion generation where temporal consistency and computational cost matter. Works particularly well for hour-long videos or high-resolution frames where attention is prohibitive.

**Hyperparameters:**
- QK threshold ratio: 0.05–0.2; lower = more aggressive skipping, higher = more conservative
- Mask update frequency: every 5–10 timesteps for video; every 1–3 for stochastic sampling
- Early-step error budget multiplier: 0.3–0.5 works well across diverse models

**Pitfalls:**
- Too-aggressive skipping (low threshold) can skip important spatial dependencies; monitor output quality
- Mask update frequency too long can miss distribution shifts; too frequent recomputes overhead gains
- Accumulated errors across T steps can interact nonlinearly; validate end-to-end quality on diverse content

**When NOT to Use:** Single-image generation (T=1, no temporal redundancy to exploit); inherently sparse-attention designs (local attention) already benefit from sparsity.

**Integration:** Compatible with any diffusion transformer; integrates into softmax pipeline without architecture changes. Pairs well with hierarchical attention and token merging for additional speedup.

---
Reference: https://arxiv.org/abs/2511.11062
