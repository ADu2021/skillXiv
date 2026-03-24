---
name: othink-r1-fast-slow-thinking
title: "OThink-R1: Intrinsic Fast/Slow Thinking Mode Switching for Over-Reasoning Mitigation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.02397"
keywords: [reasoning, efficiency, adaptive-thinking, token-reduction, LLM-optimization]
description: "Enable reasoning models to adaptively switch between fast direct responses and slow detailed reasoning, reducing token consumption by 15-40% while maintaining accuracy through dual-mode fine-tuning."
---

# OThink-R1: Intrinsic Fast/Slow Thinking Mode Switching

## Core Concept

OThink-R1 solves a critical efficiency problem in large reasoning models: not all problems require intensive step-by-step reasoning. The framework enables models to "know when to think"—automatically deciding between fast direct responses (System 1) and slow detailed reasoning chains (System 2). This mimics human cognition where simple queries get instant answers while complex problems receive deliberate analysis.

The innovation eliminates the uniform reasoning tax across all problems, achieving 15-40% token reduction while maintaining competitive accuracy on QA and mathematical benchmarks.

## Architecture Overview

- **Thinking Paradigm Identification**: Analyzes 200+ reasoning trajectories to extract patterns distinguishing essential reasoning (preventing misunderstandings, identifying key concepts) from redundant patterns (repeated validation, defensive assumptions)
- **Dual KL-Divergence Constraints**: Prevents model collapse by anchoring to both reference reasoning model (LRM) and base language model (LLM) during fine-tuning
- **Hybrid Dataset Construction**: Removes reasoning where both fast and slow paths succeed; preserves complete reasoning only for genuinely complex tasks
- **Selective Activation**: Fast thinking activates 36-40% of the time on test sets while preserving model quality

## Implementation

1. **Pattern Classification**: Use GPT-4o as judge to classify each problem as requiring "fast" or "slow" thinking based on essential vs. redundant reasoning patterns
   - Essential: Keyword identification, misunderstanding prevention, premise tracking
   - Redundant: Self-validation loops, defensive assumptions, multi-solution exploration

2. **Dataset Preparation**: Split training data into two tracks
   - Track A: Remove reasoning chains where both models succeed but slow thinking adds no value
   - Track B: Keep complete reasoning for problems where slow thinking improves accuracy

3. **Dual Fine-tuning**: Apply hybrid KL-divergence loss anchoring to both reference models

```python
# Pseudo-code for dual KL-divergence constraint
def compute_training_loss(model_output, labels, reference_lrm, reference_llm):
    """
    Balance model between reasoning and non-reasoning behaviors.
    KL losses prevent collapse toward single behavior.
    """
    ce_loss = cross_entropy(model_output, labels)
    kl_lrm = kl_divergence(model_output, reference_lrm)  # Full reasoning model
    kl_llm = kl_divergence(model_output, reference_llm)  # Base model

    # Weighted combination maintains both reference behaviors
    total_loss = ce_loss + alpha * kl_lrm + beta * kl_llm
    return total_loss
```

4. **Inference Control**: During deployment, the model naturally activates slow thinking for complex queries while using fast paths for simple ones. Monitor fast-thinking activation rates (target: 30-40% on validation sets).

## Practical Guidance

**When to Apply:**
- Your reasoning models spend excessive tokens on simple queries that don't require explicit reasoning
- You need to maintain accuracy while reducing inference latency/cost
- Your dataset has mixed difficulty levels with varying reasoning requirements

**Key Hyperparameters:**
- KL divergence weights (α, β): Control the balance between fast and slow behaviors—typically 0.1-0.5
- Fine-tuning epochs: 4 epochs on dual datasets typically sufficient
- Base model selection: Works best with distilled reasoning models (7B-14B range)

**Expected Outcomes:**
- Fast-thinking activation: 30-40% of queries
- Token reduction: 15-40% depending on dataset composition
- Accuracy maintenance: Competitive with full reasoning models
- Inference speedup: 20-30% wall-clock time improvement

**Common Pitfalls:**
- Insufficient KL weighting causes collapse to single behavior
- Overly aggressive pruning of reasoning patterns degrades accuracy
- Pattern classification needs careful validation on actual problem distribution

## Reference

Implementation demonstrated on DeepSeek-R1-Distill-Qwen variants (7B/14B), evaluated across OpenBookQA, CommonsenseQA, ASDIV, and GSM8K. Training uses 8 NVIDIA A100 GPUs with grid-searched hyperparameters across 4 epochs.
