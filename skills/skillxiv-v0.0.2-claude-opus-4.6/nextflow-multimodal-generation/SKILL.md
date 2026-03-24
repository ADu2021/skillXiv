---
name: nextflow-multimodal-generation
title: "NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.02204"
keywords: [Multimodal Models, Sequential Modeling, Image Generation, Video Generation, Unified Architecture]
description: "Build unified decoder-only transformers for multimodal tasks using 6 trillion interleaved text-image tokens with next-scale prediction for visual content—enabling fast 1024x1024 image generation (5 seconds), image editing, and video generation while rivaling specialized diffusion models."
---

## Overview

NextFlow is a unified, decoder-only autoregressive transformer trained on massive interleaved text-image discrete token sequences. Unlike specialized models (text models, diffusion models, video models), NextFlow handles all modalities through a single consistent architecture.

**Core Innovation:** Recognize that text and images have fundamentally different structure—text is strictly sequential, images are hierarchical. Use next-token prediction for text but next-scale prediction for visual generation, achieving both speed and quality.

## Architecture: Unified Decoder-Only Design

### Modality Integration

**Single Transformer:**
- Processes both text and image tokens
- Shares parameters across modalities
- Unified training procedure
- Single inference engine

**Token Representation:**
```
Input: "A sunset over the ocean" + [IMAGE_TOKENS]
             ↓
         Tokenize
             ↓
text_tokens = [A, sunset, over, ocean]
image_tokens = [discrete_token_0, ..., discrete_token_n]
             ↓
Combined: [A, sunset, over, ocean, tok_0, tok_1, ...]
             ↓
Transformer processing
```

### Training on Interleaved Data

**6 Trillion Tokens:**
- Mixed text-image sequences from diverse sources
- Interleaved layout (text and images together)
- Natural multimodal reasoning emerges
- Consistent objective function

**Example Training Sequence:**
```
[text: "A cat..."] [image_tokens: descr of cat photo]
[text: "...sitting on a desk"] [image_tokens: desk photo]
```

## Next-Token vs. Next-Scale Prediction

### Critical Design Insight

**Text: Next-Token Prediction**
- Natural ordering (sequential)
- Standard autoregressive approach
- Token-by-token generation

**Images: Next-Scale Prediction**
- Images are inherently hierarchical
- Generate from coarse to fine
- Multi-scale token prediction

### Next-Scale Image Generation

Generate images hierarchically from low to high resolution:

```python
def next_scale_image_generation(model, prompt: str, max_resolution=1024):
    """Generate image at progressively finer scales."""

    # Encode text prompt
    text_tokens = tokenizer.encode(prompt)

    # Initialize with low-resolution tokens (coarse structure)
    image_tokens = []

    scales = [64, 128, 256, 512, 1024]  # Progressive scales
    for scale in scales:
        # Generate tokens for current scale
        # Condition on: prompt + previous coarser scale tokens
        scale_tokens = model.generate(
            text_tokens + image_tokens,
            num_new_tokens=tokens_per_scale(scale),
            temperature=0.8,
        )
        image_tokens.extend(scale_tokens)

    # Decode tokens to image
    image = decode_image_tokens(image_tokens)
    return image
```

**Speed Advantage:**
- 1024x1024 image in ~5 seconds
- Comparable speed to diffusion (5-10 steps)
- Much faster than diffusion models (25-100 steps)
- Orders of magnitude faster than autoregressive pixel models

### Speed Comparison

| Method | Resolution | Time |
|--------|-----------|------|
| NextFlow | 1024x1024 | 5 sec |
| DALL-E 2 | 1024x1024 | 30 sec |
| Stable Diffusion | 1024x1024 | 10-30 sec |
| Autoregressive (pixel) | 256x256 | 60+ sec |

## Multimodal Capabilities

### Image Understanding
- Direct processing of image tokens
- No separate vision encoder needed
- Unified semantic space for text and images

### Image Editing
- Specify region and description
- Regenerate tokens for masked region
- Coherent integration with surrounding content

### Video Generation
- Generate frame sequences as interleaved image token sequences
- Natural temporal coherence from sequential modeling
- Variable frame rate and duration

### Unified Generation
```python
def multimodal_generation_task(model, task: str, inputs):
    """Handle diverse multimodal tasks with single model."""

    if task == "image_generation":
        prompt = inputs["prompt"]
        return generate_image(model, prompt)

    elif task == "image_editing":
        image = inputs["image"]
        edit_region = inputs["mask"]
        edit_prompt = inputs["prompt"]
        return edit_image(model, image, edit_region, edit_prompt)

    elif task == "video_generation":
        prompt = inputs["prompt"]
        duration = inputs["duration"]  # in frames
        return generate_video(model, prompt, duration)

    elif task == "multimodal_understanding":
        image = inputs["image"]
        question = inputs["question"]
        # Answer question about image with reasoning
        answer = model.generate(
            image_tokens + question_tokens,
            max_tokens=100,
        )
        return answer
```

## Benchmark Performance

**Image Generation (Proprietary Benchmark):**
- Visual quality competitive with specialized diffusion models
- Faster generation (5 sec vs. 10-30 sec)
- State-of-the-art among unified models

**Video Generation:**
- Temporal consistency from sequential modeling
- Coherent frame transitions
- Competitive with video-specialized models

**Multimodal Understanding:**
- Strong performance on VQA and image captioning
- Unified training enables transfer between tasks
- Single model for multiple capabilities

## Training Stability

**Multi-Scale Generation Challenges:**
- Token prediction can be unstable (early tokens affect later scales)
- Distribution mismatch between scales

**Robust Training Recipe:**
- Careful initialization of scale-specific layers
- Curriculum: train coarse scales first, then fine scales
- Mixed precision training for stability
- Gradient clipping and layer-wise learning rate scaling

## When to Use NextFlow

**Use when:**
- Building unified multimodal systems
- Need fast image/video generation (5 sec acceptable)
- Want single model for multiple tasks (generation, editing, understanding)
- Prefer efficient inference over specialized models
- Value unified training and inference

**When NOT to use:**
- Extreme quality requirements (specialized diffusion models higher quality)
- Sub-second inference required (fixed-cost approach)
- Specialized tasks with dedicated SOTA models
- Applications with very large image resolution needs (16K+)

## Architectural Comparisons

**vs. Separate Models (Vision + Language):**
- Single model reduces memory and parameter count
- Unified training enables better multimodal reasoning
- Simpler deployment and maintenance

**vs. Diffusion Models:**
- Faster inference (5 sec vs. 25+ steps)
- Similar quality in many cases
- No diffusion process overhead
- Unified architecture enables other tasks

**vs. Vision Transformers + LLMs:**
- No separate vision encoder
- Unified token space enables better reasoning
- Simpler architecture with fewer components

## Implementation Considerations

**Discrete Tokenization:**
- Use learned discrete token codebook for images
- Typically 256-1024 tokens per 256x256 image
- Enables variable-resolution generation through scale approach

**Training Data:**
- 6 trillion tokens is substantial
- Requires diverse multimodal datasets
- Careful curation for quality and balance

**Inference Optimization:**
- KV caching for sequential generation
- Batch processing across requests
- Layer fusion for efficient computation

## Research Contributions

- **Unified Multimodal Architecture:** Single model for text and images
- **Next-Scale Prediction:** Hierarchical approach to visual generation
- **Fast Image Generation:** 5-second 1024x1024 generation
- **Comprehensive Evaluation:** Demonstration across understanding and generation

## Code Availability

Project page available with model checkpoints and training code.

## References

- NextFlow trained on 6 trillion interleaved text-image tokens
- State-of-the-art among unified models on visual quality
- 5-second 1024x1024 image generation (5-10x faster than diffusion)
- Unified architecture enables image editing and video generation
- Demonstrates viability of next-scale prediction for visual content
