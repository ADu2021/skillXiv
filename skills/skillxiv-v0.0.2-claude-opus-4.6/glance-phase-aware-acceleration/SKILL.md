---
name: glance-phase-aware-acceleration
title: "Glance: Accelerating Diffusion Models with One Sample via Phase-Aware Tuning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.02899
keywords: [diffusion-models, acceleration, distillation, lora-adaptation, single-image-training]
description: "Phase-aware acceleration using two lightweight LoRA adapters (Slow-LoRA for semantic reconstruction, Fast-LoRA for texture refinement) trained on a single image in one GPU hour, achieving 5× speedup via smart per-phase acceleration rather than uniform speedup."
---

## Summary

Glance introduces a phase-aware acceleration strategy for diffusion model distillation that applies different speedup rates to distinct denoising phases. Rather than uniformly accelerating all timesteps, the method deploys two lightweight LoRA adapters—Slow-LoRA for early semantic reconstruction and Fast-LoRA for later texture refinement. This enables efficient training on single images without massive datasets.

## Core Technique

**Phase Identification:** Diffusion follows two conceptual phases:
- **Early Phase (high noise):** Reconstructs semantic structure; requires careful denoising
- **Late Phase (low noise):** Refines texture and detail; less critical for content

**Phase-Specific LoRA Adapters:** Train two lightweight adapters:
- **Slow-LoRA:** Low-rank matrix for early timesteps, provides conservative updates
- **Fast-LoRA:** Low-rank matrix for late timesteps, enables aggressive acceleration

**Speedup Strategy:** Slow down early phase (maintain quality), speed up late phase (minimal quality loss):
```
step_multiplier = {
    t > 0.7 * T: 1.0  # Early phase: no acceleration
    t <= 0.7 * T: 5.0  # Late phase: 5× acceleration
}
```

## Implementation

**LoRA architecture:** For each phase, add low-rank adaptation:
```python
# Standard linear layer: y = Wx
# LoRA: y = Wx + (A @ B)x where A, B are low-rank
# Rank r << hidden_dim (e.g., r=8, hidden_dim=768)

lora_slow = LoRA(in_dim=768, out_dim=768, rank=8)  # Early phases
lora_fast = LoRA(in_dim=768, out_dim=768, rank=8)  # Late phases
```

**Phase-aware forward pass:**
```python
def forward(x, t):
    # Compute standard diffusion output
    out = diffusion_model(x, t)

    # Apply phase-specific LoRA
    phase = t / max_timesteps
    if phase > 0.7:  # Early phase
        out = out + lora_slow(x)
    else:  # Late phase
        out = out + lora_fast(x)

    return out
```

**Single-image training:** Use the same image repeatedly with augmentations:
```python
for epoch in range(100):
    x = augment(image)  # Random crop, flip, rotate
    noise = randn_like(x)
    t = randint(0, max_timesteps)
    loss = mse(diffusion(x + noise, t), noise)
    loss.backward()
    optimizer.step()
```

## When to Use

- Efficient diffusion model acceleration with minimal data
- Scenarios where training must complete in hours, not days
- Applications needing per-phase control over speed-quality trade-offs
- Tasks where single-image distillation is sufficient for your use case

## When NOT to Use

- Scenarios requiring multi-image or large-scale training
- Tasks where uniform acceleration is preferable
- Applications needing maximum-quality output at any speed
- Real-time generation where distillation overhead matters

## Key References

- Diffusion models and denoising process phases
- LoRA: Low-Rank Adaptation for fine-tuning
- Model distillation and acceleration techniques
- Single-image synthesis and adaptation
