---
name: learnable-multipliers-lm-scaling
title: "Learnable Multipliers: Freeing the Scale of Language Model Matrix Layers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.04890"
keywords: [Language Model Training, Optimization, Scaling, Weight Initialization]
description: "Adapt language model weight matrix scales automatically during training by adding learnable scalar and vector multipliers to each layer. Breaks the noise-weight-decay equilibrium that constrains scales based on hyperparameters rather than data, enabling data-driven scaling adaptation without inference cost or extensive tuning overhead."
---

## When to Use This Skill
- Training large language models with Adam or Muon optimizers
- Scenarios where weight decay creates suboptimal equilibrium norms
- Models where μP multiplier tuning is burdensome (35+ manual values)
- Improving reasoning task performance (BBH, MATH) without increasing model size
- Attention, MLP, and SSM-based architectures

## When NOT to Use This Skill
- Inference-critical applications requiring minimal parameter overhead (overhead is negligible but exists during training)
- Models already achieving saturation performance (marginal gains may not justify complexity)
- Scenarios with fixed weight matrices (non-learnable components)

## Problem Summary
Weight decay in language model training creates a noise-WD equilibrium that constrains matrix layer scales based on optimization hyperparameters (η, λ) rather than data properties. This fixes weight scales as √(η/λ) regardless of task requirements, preventing data-driven scale adaptation. Current methods like μP require extensive manual tuning (35+ multiplier values) across model sizes.

## Solution: Learnable Multipliers (LRM)

Introduce trainable scalar and vector multipliers that adapt layer scales during training without experiencing noise-driven expansion.

```python
# Learnable multiplier parameterization
class LearnableMultiplier:
    def __init__(self, weight_matrix):
        # Scalar multiplier: applies uniform scale
        self.s = nn.Parameter(torch.ones(1))

        # Vector multipliers: per-row and per-column scales (selective placement)
        self.r = nn.Parameter(torch.ones(weight_matrix.shape[0]))  # Row scales
        self.c = nn.Parameter(torch.ones(weight_matrix.shape[1]))  # Column scales

        self.W = weight_matrix

    def forward(self):
        # Combine original weights with learned scales
        # W_scaled = s * diag(r) @ W @ diag(c)
        return self.s * (self.r[:, None] * self.W * self.c[None, :])
```

**Key insight**: Multipliers don't experience gradient noise-driven expansion because gradient averaging across rows/columns reduces noise levels below the noise-WD equilibrium threshold.

## Key Implementation Details

**Placement Strategy:**
Avoid redundancy by not applying both row and column multipliers everywhere:
- Attention blocks: Use carefully selected multiplier placement
- MLP layers: Apply selective row/column multipliers
- SSM blocks: Adapt multiplier types to block architecture

Reference appendix for optimal placement patterns per architecture.

**Critical Training Stabilization:**
- Apply small weight decay to multipliers themselves: λ_lrm = 2×10⁻³
- Exclude multiplier gradients from global gradient clipping norm calculations
- This prevents aggressive clipping that would suppress matrix parameter updates

```python
# Stabilization during backward pass
def stabilized_backward():
    loss.backward()

    # Exclude multiplier grads from global norm for clipping
    matrix_grads = [p.grad for p in model.matrix_params if p.requires_grad]
    matrix_grad_norm = torch.nn.utils.clip_grad_norm_(matrix_grads, max_norm=1.0)

    # Multiplier gradients accumulate independently
    # without suppression from large matrix gradient magnitudes
```

**Width Scaling Property:**
Learnable multipliers automatically adjust to maintain stable activation magnitudes as model width increases—eliminating manual width-scaling rules.

## Performance Gains

On diverse downstream tasks:
- **Adam + LRM**: +1.21% average improvement
- **Muon + LRM**: +1.10% average improvement
- **Reasoning-focused tasks** (BBH, MATH): Larger gains than knowledge tasks (Hellaswag, MMLU, GSM8K)

## Practical Advantages

- **Zero inference cost**: Multipliers merge into weights during deployment
- **Reduces tuning burden**: Eliminates 35+ manually tuned μP values
- **Optimizer agnostic**: Works with both Adam and Muon
- **Architecture flexible**: Applicable to attention, MLP, and SSM blocks
- **Training stability**: Prevents convergence failures from suboptimal equilibrium scaling

## Open Questions for Researchers
- Complete width-scaling rules across different architectures
- Optimal learning rates for multipliers at different model scales
- Whether learnable multipliers unlock specific capability types disproportionately
- Interaction with mixed-precision training and quantization

## Implementation Checklist
1. Identify matrix layers in your architecture
2. Determine optimal multiplier placement from appendix patterns
3. Initialize multipliers to identity (s=1, r=1, c=1)
4. Apply λ_lrm = 2×10⁻³ weight decay to multiplier parameters
5. Exclude multiplier gradients from global clipping norm
6. Monitor activation distributions to confirm stable scaling
7. Merge multipliers into weights for deployment
