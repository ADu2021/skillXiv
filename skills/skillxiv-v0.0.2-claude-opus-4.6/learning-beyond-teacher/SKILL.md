---
name: learning-beyond-teacher
title: "Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.12125"
keywords: [Knowledge Distillation, Reward Scaling, Student-Teacher Learning, Multi-Teacher Merging, RL Finetuning]
description: "Train student models to exceed teacher performance by extrapolating rewards during distillation using a scaling factor λ1. Enables unified students to surpass all individual domain teachers through controlled reward amplification."
---

# Learning beyond Teacher: Reward Extrapolation in Distillation

## Problem Context

Traditional knowledge distillation constrains student models to match teacher outputs, creating a performance ceiling where students can at best equal teachers. When combining multiple domain expert teachers, standard distillation produces weak unified students that underperform all specialists. The core constraint is symmetric weighting between reward and KL regularization terms in the training objective.

## Core Concept

Generalized On-Policy Distillation (G-OPD) with reward extrapolation (ExOPD) breaks the teacher performance ceiling by **setting the reward-KL weight λ > 1**. This asymmetry encourages students to "go beyond matching teacher distributions" by fitting an additional shift term, effectively learning to extrapolate rather than merely imitate.

The key insight: by amplifying reward signals relative to KL constraints, students learn not just what teachers do but can infer what teachers would do under different conditions, surpassing single-teacher performance.

## Architecture Overview

- **Flexible Reference Model**: Supports different models for computing implicit rewards (not just the original teacher)
- **Reward Scaling Factor (λ)**: Controls relative weight between reward fitting and KL regularization
- **Standard OPD Baseline**: λ=1 recovers traditional distillation as special case
- **Extrapolation Regime**: λ>1 enables learning beyond teacher demonstration
- **Optional Reward Correction**: Pre-RL teacher model can further enhance student performance

## Implementation

The key modification to On-Policy Distillation involves adjusting the training objective:

```python
def generalized_opd_loss(student_logprobs, teacher_logprobs,
                         rewards, lambda_scale=1.25):
    """
    Compute G-OPD loss with reward extrapolation.
    Standard OPD uses lambda=1.0; extrapolation uses lambda>1.0
    """
    # KL regularization: constrain student to match teacher
    kl_div = student_logprobs - teacher_logprobs

    # Reward fitting: encourage student to fit rewards
    # With lambda>1, reward signal is amplified relative to KL
    loss = kl_div - lambda_scale * rewards

    return loss.mean()
```

Multi-teacher merging with ExOPD:

```python
def merge_teachers_with_extrapolation(teacher_models,
                                      student_model,
                                      lambda_scale=1.25,
                                      correction_model=None):
    """
    Merge multiple domain expert teachers into unified student
    using reward extrapolation for surpassing individual teachers.
    """
    losses = []

    for teacher in teacher_models:
        # Get teacher outputs for batch
        teacher_logprobs = teacher.get_logprobs(batch)

        # Compute rewards
        if correction_model:
            # Optional: use pre-RL teacher model for reward signal
            rewards = compute_rewards_with_correction(
                teacher, correction_model, batch)
        else:
            rewards = compute_rewards(teacher, batch)

        # Apply G-OPD with extrapolation
        loss = generalized_opd_loss(
            student_model.get_logprobs(batch),
            teacher_logprobs,
            rewards,
            lambda_scale
        )
        losses.append(loss)

    return sum(losses) / len(losses)
```

## Practical Guidance

**When to use**:
- Need to merge multiple domain expert teachers
- Want student to exceed any single teacher performance
- Have access to reward signals (from RL finetuning or preference data)
- Training multimodal or multi-task models

**Tuning the extrapolation factor**:
- Standard OPD baseline: λ = 1.0 (student matches teacher)
- Conservative extrapolation: λ = 1.15-1.25 (gradual beyond teacher)
- Aggressive extrapolation: λ = 1.5+ (substantial amplification, requires careful tuning)
- Papers show λ = 1.25 delivers consistent improvements across benchmarks

**Optional reward correction**:
- Use pre-RL teacher model for reward computation
- Particularly helpful for strong-to-weak distillation (large-to-small models)
- Can provide 5-10% additional improvements

**Expected improvements**:
- Single teacher: 10-15% improvement over standard OPD
- Multi-teacher merging: 15-25% improvement, with unified student exceeding all individual teachers
- Code generation and math reasoning benchmarks show most consistent gains

## Reference

Reward extrapolation in distillation enables learning to surpass teachers by breaking the symmetric constraint between imitation and reward fitting. The approach integrates naturally with multi-teacher scenarios common in modern model merging and ensemble systems.
