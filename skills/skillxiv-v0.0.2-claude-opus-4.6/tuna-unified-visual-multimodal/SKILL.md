---
name: tuna-unified-visual-multimodal
title: "TUNA: Unified Visual Representations for Native Multimodal Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.02014
keywords: [multimodal-learning, vision-language, unified-representations, flow-matching, understanding-generation]
description: "Cascaded VAE+SigLIP encoders creating single continuous representation space supporting both vision understanding and generation, trained jointly on both tasks without format mismatches. Deploy for unified multimodal models where understanding and generation enhance each other."
---

## Summary

TUNA introduces a unified visual representation approach for native unified multimodal models by cascading a VAE encoder with a representation encoder (SigLIP 2) to create a single continuous representation space supporting both understanding and generation tasks. Joint training on these unified representations with both understanding and generation data demonstrates mutual enhancement rather than interference.

## Core Technique

**Cascaded Encoding Pipeline:** First encode images through a VAE encoder to capture semantic structure, then pass through a representation encoder (SigLIP 2) fine-tuned jointly. This two-stage encoding avoids "representation format mismatches" where separate encoders create incompatible spaces for understanding versus generation.

**Unified Representation Space:** Unlike decoupled designs with separate image encoders for encoding and generation, TUNA uses a single learned space that naturally supports both operations. Train jointly by:
1. Encoding images to unified representations
2. Decoding via LLM + flow matching for generation
3. Using same representations for vision-language understanding

**Mutual Enhancement:** Both understanding and generation objectives improve jointly rather than competing. Understanding helps the model learn robust semantic representations, while generation encourages detailed spatial encodings.

## Implementation

**VAE encoding stage:** Use pretrained VAE (e.g., VQGAN or diffusion VAE) to encode images to semantic latents. These latents act as intermediate semantic bottleneck.

**Representation encoding:** Train representation encoder with:
```
rep = encode_siplip(vae_latent)
```
This learns projection from VAE space to unified space, optimized for both tasks.

**Dual head training:** Attach two heads to unified representations:
- **Understanding head:** Token classifier predicting image patches for language understanding
- **Generation head:** Flow matching decoder predicting pixel distributions

**Joint loss:** ℒ_total = λ_u * ℒ_understanding + λ_g * ℒ_generation, where λ weights are balanced (often 0.5/0.5).

## When to Use

- Building unified multimodal models handling both understanding and generation
- Vision-language tasks where separation of encode/decode is problematic
- Applications needing a single representation space for efficiency
- Scenarios where generation and understanding are both important

## When NOT to Use

- Specialized models focusing only on one modality or task
- Scenarios where decoupled encoders are simpler or more modular
- Applications preferring separate understanding and generation pipelines
- Tasks where the unified space creates bottlenecks for specialized architectures

## Key References

- VAE-based image compression and semantic encoding
- CLIP/SigLIP vision-language alignment models
- Flow matching for generative modeling
- Joint multi-task learning in multimodal systems
