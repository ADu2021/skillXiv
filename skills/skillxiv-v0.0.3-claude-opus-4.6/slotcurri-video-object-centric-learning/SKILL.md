---
name: slotcurri-video-object-centric-learning
title: "SlotCurri: Reconstruction-Guided Slot Curriculum for Video Object-Centric Learning"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22758"
keywords: [Slot Attention, Curriculum Learning, Video Object-Centric Learning, Over-Fragmentation, Progressive Allocation]
description: "Replace fixed full-capacity slot initialization with progressive curriculum-based slot expansion to reduce over-fragmentation in video object-centric learning. Improves FG-ARI by +6.8 on YouTube-VIS and +8.3 on MOVi-C. Use when training slot attention models on videos with variable object counts and sizes."
category: "Component Innovation"
---

## What This Skill Does

Replace the standard practice of allocating all slot attention slots upfront with a progressive curriculum that starts with minimal slots and expands them reconstruction-guided. This eliminates over-fragmentation where single objects split across multiple slots, improving object discovery by +6.8 to +8.3 FG-ARI on real-world video datasets.

## The Component Swap

The old SlotContrast baseline initializes all K slots at training start, forcing the model to use all slots regardless of whether the scene contains K objects:

```python
# Old approach: allocate all slots upfront
K = 10  # Fixed capacity
slots = nn.Parameter(torch.randn(1, K, slot_dim))
# Model pressured to fragment objects across slots to minimize loss
```

The new SlotCurri approach starts with minimal slots and progressively expands them through curriculum stages. Slots are spawned reconstruction-guided from high-error regions:

```python
# New: progressive slot curriculum
K_init = 2  # Start small
K_max = 10  # Final capacity
num_stages = 3

# Stage-dependent slot count: K^(m) = K_init + m*sigma + 3m(m-1)/2
def slot_count_schedule(stage_m, sigma=1):
    return K_init + stage_m * sigma + 3 * stage_m * (stage_m - 1) // 2

# Slot spawning: duplicate from high-reconstruction-error parents
reconstruction_error = ||video_recon - video_real||_2  # Per-slot error
high_error_slots = torch.argsort(reconstruction_error)[:top_k]
new_slots = slots[high_error_slots] + noise_perturbation(scale=0.1)

slots = torch.cat([slots, new_slots], dim=1)  # Expand capacity
```

Augment the reconstruction loss with structure-aware SSIM to better capture object boundaries:

```python
# Old loss: pixel-level MSE only
loss_old = F.mse_loss(video_recon, video_real)

# New loss: MSE + structure-aware SSIM
loss_mse = F.mse_loss(video_recon, video_real)
loss_ssim = 1.0 - ssim(video_recon, video_real, window_size=11)
loss_new = loss_mse + 0.3 * loss_ssim  # Weight SSIM component
```

## Performance Impact

**Object discovery (FG-ARI metric):**
- YouTube-VIS: 38.0 → 44.8 ± 1.2 = **+6.8 percentage points** (18% relative)
- MOVi-C: 69.3 → 77.6 ± 0.9 = **+8.3 pp** (12% relative)

**Boundary quality (mBO metric):**
- YouTube-VIS: 33.7 → 35.5 ± 2.2 = +1.8 pp (modest improvement)

**Trade-off on synthetic datasets:**
- MOVi-E (densely packed objects): +0.8 pp only (curriculum less beneficial for dense object scenarios)

## When to Use

- Video object-centric learning with real-world datasets (variable object counts/sizes)
- Slot attention models showing over-fragmentation (objects split across multiple slots)
- Datasets where objects have varying spatial extents and depths
- Tasks where discovery quality (FG-ARI) is more important than boundary precision

## When NOT to Use

- Dense object scenes with fixed, regular layouts (MOVi-E shows minimal gains)
- Datasets with fixed object counts where fragmentation isn't a problem
- Scenarios where fixed slot allocation is theoretically justified
- Tasks requiring strict per-slot semantics rather than object-level semantics

## Implementation Checklist

To adopt this component swap:

1. **Implement curriculum schedule:**
   ```python
   # Define stage-dependent slot counts
   stage = current_epoch // (total_epochs / num_curriculum_stages)
   K_current = K_init + stage * sigma + 3 * stage * (stage - 1) // 2
   ```

2. **Add slot spawning logic:**
   ```python
   # Every curriculum stage, spawn new slots from high-error parents
   if should_expand_slots(stage):
       reconstruction_errors = compute_per_slot_error(video_recon, video_real)
       parent_slots = select_high_error_slots(reconstruction_errors, num_new=sigma)
       noise = torch.randn_like(parent_slots) * noise_scale
       new_slots = parent_slots + noise
       slots = torch.cat([slots, new_slots], dim=0)
   ```

3. **Replace reconstruction loss:**
   ```python
   # Add structure-aware SSIM to capture boundaries
   loss_ssim = structural_similarity_loss(video_recon, video_real)
   loss_reconstruction = F.mse_loss(...) + 0.3 * loss_ssim
   ```

4. **Adjust slot attention mechanism for variable K:**
   ```python
   # Slot attention must handle variable slot count across stages
   # Typically achieved by padding/masking unused slots
   slots_active = slots[:, :K_current, :]  # Only use current stage capacity
   ```

5. **Verify improvements:**
   - Measure FG-ARI on YouTube-VIS validation set
   - Compare before (K=10) vs after (curriculum K=2→10) on same split
   - Check that per-object consistency improves (fewer splits per ground-truth object)

6. **Hyperparameter tuning:**
   - `K_init`: starting slot count (default: 2, range 1-4)
   - `sigma`: slots added per stage (default: 1, range 1-3)
   - `num_curriculum_stages`: number of expansion stages (default: 3, range 2-5)
   - `ssim_weight`: importance of structure loss (default: 0.3, range 0.1-0.5)
   - `noise_scale`: perturbation magnitude when spawning (default: 0.1, range 0.05-0.2)

7. **Known issues:**
   - Curriculum schedule must align with training epochs; hardcoding stage transitions can cause instability
   - Slot spawning initialization is sensitive to noise scale; too high → divergence, too low → redundancy
   - MOVi-E and dense scenes show minimal gains; curriculum may not help when over-fragmentation isn't the bottleneck

## Related Work

This builds on slot attention methods and curriculum learning. Progressive allocation relates to dynamic architecture approaches, but applies the curriculum principle specifically to object discovery in video.
