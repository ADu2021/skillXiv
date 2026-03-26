---
name: look-where-it-matters
title: "Look Where It Matters: Spatial-On-Demand Vision via High-Res Crop Retrieval"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.16932
keywords: [VLM Architecture, Efficiency, Spatial Attention, Token Reduction, Tool Use]
description: "Add tool-calling interface for on-demand high-resolution crop retrieval, enabling VLMs to first observe low-resolution global view then selectively request detailed crops. Achieves 80.3% of full-res performance (vs 80.46%) with only 36% of visual tokens, reducing wall-clock latency 4.4× (from 2.71s to 0.61s) through KV-cache reuse without architectural changes."
---

## Component Identification

**Old Design (VLM Baseline)**
- Dense processing of full resolution image
- All visual tokens processed for every query
- Complete visual feature extraction regardless of task requirements
- Single-turn inference for complex spatial reasoning

**New Design (AwaRes)**
Tool-calling interface enabling coupled-decision policy: decide whether detail is needed AND which crops to retrieve from predefined set.

## Motivation & Problem Statement

High-resolution image understanding requires substantial compute, yet many queries only need detail from specific regions. The challenge: enable VLMs to make *adaptive* decisions about which details matter, rather than processing entire high-res images uniformly. This reduces redundant token processing while preserving accuracy.

## The Modification

AwaRes implements spatial-on-demand perception through a coupled-decision policy:

```python
# Turn 1: Low-resolution global perception
image_lr = resize_to_low_res(image)  # e.g., 336x336 tokens
response_1 = vlm(image_lr, query)

# Coupled decision via tool-calling
# Decide: (1) do we need high-res crops? (2) which regions?
if needs_detail(response_1):
    crop_indices = [predefined_crops]  # quadrants, halves, center, full
    crops_hr = [image[crop] for crop in crop_indices]

    # Turn 2: High-resolution crop processing with KV-cache reuse
    # Keys/values from Turn 1 remain cached and extended
    response_2 = vlm(
        kv_cache=response_1.kv_cache,  # Reuse Turn 1 computation
        crops=crops_hr,
        query=query
    )
else:
    return response_1
```

The key insight: **Multi-turn KV-cache reuse** allows computation from the initial low-resolution turn to be extended rather than recomputed, eliminating architectural changes while preserving compatibility.

## Efficiency Gains

### Token Efficiency
- Full high-res performance: 80.46%
- AwaRes (36% tokens): 80.3% accuracy
- Effective token reduction: 64% fewer visual tokens processed

### Latency Reduction
Against VisionThink baseline:
- Full latency: 2.71 seconds
- AwaRes latency: 0.61 seconds
- **4.4× speedup** (wall-clock time)

### Memory Implications
Token reduction directly translates to proportional reduction in:
- KV-cache memory (stored attention keys/values)
- Computational operations in attention layers
- Peak memory during inference

## Conditions of Applicability

**Works well when:**
- Spatial detail is non-uniformly distributed (some regions more relevant than others)
- Queries benefit from both global context and local detail
- Inference latency is critical (KV-cache reuse amortizes computation)
- Predefined crop set covers task regions (quadrants, halves, full image sufficient)

**Less optimal when:**
- Uniform detail matters equally across entire image (document reading, dense text)
- Query requires simultaneous detail from distant regions
- Model needs to dynamically compute optimal crop locations (predefined set may not align)

## Drop-In Replacement Checklist

- [x] No architectural changes to VLM core (only adds tool-calling interface)
- [x] Compatible with existing KV-cache mechanisms
- [x] Maintains single-turn interface (tool use is transparent to caller)
- [x] Works with any VLM supporting tool-calling
- [x] Graceful degradation (falls back to low-res when detail unnecessary)
- [x] Token efficiency improves with task-appropriate crop decisions
