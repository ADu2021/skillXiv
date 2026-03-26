---
name: mineru-diffusion-ocr-inverse-rendering
title: "MinerU-Diffusion: Document OCR via Diffusion Decoding"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22458"
keywords: [OCR, Diffusion Models, Parallel Decoding, Block-Wise Attention, Curriculum Learning]
description: "Replace autoregressive token-by-token OCR decoding with block-wise diffusion decoding to achieve 3.2× speedup while maintaining 99.9% accuracy. Works best for document layout parsing where parallel generation is feasible. Trigger: When optimizing OCR systems and want faster inference without accuracy loss."
category: "Component Innovation"
---

## What This Skill Does

Swap autoregressive sequential decoding with block-wise diffusion decoding in document OCR to improve inference speed by 3.2× while maintaining competitive accuracy on structured document parsing tasks.

## Problem with Autoregressive Decoding

Autoregressive models generate OCR tokens sequentially (left-to-right), treating token ordering as an intrinsic property of the task. This forces O(n) sequential steps where each token depends on all prior tokens. For document OCR, this sequential bottleneck is unnecessary: document structure allows parallel token generation once block-level dependencies are satisfied.

The paper's insight: Document OCR is fundamentally an inverse rendering problem where content can be refined in parallel within blocks, not strictly left-to-right.

## The Swap: Diffusion Decoder with Block-Wise Attention

Replace sequential autoregressive generation with iterative diffusion refinement using factored attention:

```python
# Autoregressive (sequential, causal attention)
def autoregressive_decode(tokens, hidden_states):
    """Generate tokens one by one, each attending to all prior tokens"""
    for i in range(len(tokens)):
        next_token = model(tokens[:i+1], hidden_states)  # O(n) steps
        tokens[i] = next_token

# Diffusion with block-wise attention (parallel within blocks)
def block_diffusion_decode(noisy_tokens, hidden_states, blocks=4):
    """
    Iteratively refine all tokens in parallel.
    Attention is factored:
    - Within block: bidirectional (all tokens see each other)
    - Across blocks: causal (tokens attend to preceding blocks)
    - Reduces complexity from O(L²) to O(B·L'²) where L' = L/B
    """
    for step in range(num_diffusion_steps):
        # Process each block, tokens attend within-block bidirectionally
        for block_idx in range(blocks):
            start = block_idx * (len(tokens) // blocks)
            end = (block_idx + 1) * (len(tokens) // blocks)
            # Within-block bidirectional attention + causal to prior blocks
            refined = model(noisy_tokens, hidden_states,
                           block_range=(start, end),
                           attend_to_prior_blocks=True)
            noisy_tokens[start:end] = refined[start:end]
    return noisy_tokens
```

Key difference: Autoregressive requires L sequential forward passes; diffusion refines all L tokens in parallel via iterative denoising over T steps (typically T << L).

## Performance Impact

**Baseline (Autoregressive):**
- OmniDocBench v1.5 score: ~93.37
- Inference latency: baseline (1.0×)

**With Block-Wise Diffusion:**
- OmniDocBench v1.5 score: 93.37 (99.9% relative accuracy maintained)
- Speedup: 2.12× at 99.9% accuracy; up to 3.01× at 98.8% accuracy
- Latency reduced from L sequential steps to ~10-20 iterative refinement steps

**Robustness evidence:** Semantic Shuffle benchmark shows diffusion decoders remain robust when semantic coherence is disrupted (random token permutation), while autoregressive models degrade significantly—indicating diffusion's inherent parallel nature is more semantically stable.

## When to Use

- Document OCR and layout parsing (structured data where spatial relationships matter)
- When inference speed is critical and you can tolerate slight accuracy trade-offs (2-3%)
- Models with bidirectional context (like LayoutLM, DiT) that can leverage parallel refinement
- Tasks where token dependencies are localized (within blocks)

## When NOT to Use

- Streaming/online generation where you need tokens immediately (diffusion requires multiple passes)
- Tasks requiring strict left-to-right coherence (e.g., causal language generation)
- Models already optimized for autoregressive decoding with cached attention
- If you need 100% accuracy preservation (diffusion trades 0.1-2% accuracy for speed)

## Implementation Checklist

To swap autoregressive with block-wise diffusion:

1. **Prepare diffusion infrastructure**: Implement iterative denoising loop with diffusion scheduler (e.g., cosine schedule for T=10-20 steps)

2. **Implement block-wise attention factorization**:
   - Within-block: bidirectional self-attention
   - Across blocks: causal mask (current block attends to prior blocks, not future)
   - Initialize noisy tokens from noise schedule at step 0

3. **Train with two-stage curriculum**:
   - Stage I: Diversity-driven learning on broad, easier document data
   - Stage II: Uncertainty-driven boundary refinement, targeting hard cases where OCR typically fails (tables, figures) via inference consistency scoring

4. **Measure speedup**:
   - Baseline: Time for autoregressive generation
   - New: Time for diffusion + denoising iterations
   - Target: 2-3× improvement; verify accuracy loss < 1%

5. **Optional tuning**:
   - Block size: 4-8 tokens per block (balance parallelism vs stability)
   - Diffusion steps T: 10-20 (more steps = higher accuracy, slower inference)
   - Noise schedule: Cosine schedule typically works best

## Known Issues

- **Accuracy-speed tradeoff**: At 3× speedup, expect 1-2% accuracy drop. Adjust block size or diffusion steps to find the sweet spot.
- **Incompatible with streaming**: Diffusion requires seeing all positions to refine; cannot output tokens until full pass completes.
- **Block boundary artifacts**: If block size is too small, ensure overlapping context or causal attention to prior blocks prevents coherence breaks.

## Related Work

This builds on diffusion-based generation (DDPM, latent diffusion) and adapts it for structured sequence modeling. Relates to parallel decoding approaches (SpecInfer, non-autoregressive MT) but adds curriculum learning to stabilize diffusion training for OCR tasks.
