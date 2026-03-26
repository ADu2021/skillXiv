---
name: longcat-flash-prover
title: "LongCat-Flash-Prover: Hierarchical Importance Sampling for Formal Reasoning"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.21065
keywords: [Formal Reasoning, Reinforcement Learning, Policy Optimization, Theorem Proving]
description: "Integrate agentic tool interaction (Lean4 compiler, syntax checkers) with curriculum-based RL for formal reasoning. Replace standard importance sampling with Hierarchical Importance Sampling Policy Optimization (HisPO): sequence-level masking removes train-inference discrepancies, token-level masking filters inconsistent tokens, staleness control manages policy drift. Achieves 97.1% auto-formalization (vs 83% baseline), 95.5% MiniF2F-Test (72 attempts vs 1,024+), 70.8% ProverBench."
---

## Component Identification

**Old Design (Standard Supervised Finetuning)**
- No tool interaction (no Lean verification)
- Single-turn generation without feedback
- No curriculum on interaction complexity
- Standard supervised loss without importance weighting

**New Design (Tool-Integrated GRPO with HisPO)**
- Direct integration with Lean4 compiler and syntax checkers
- Curriculum progression: single-turn → multi-turn tool sequences
- Hierarchical importance sampling addressing train-inference discrepancies
- Legality detection preventing reward hacking

## Motivation & Problem Statement

Formal reasoning requires multi-turn interaction with verification tools (compilers, type checkers) to generate correct proofs. Standard supervised learning fails because:

1. **Distribution mismatch**: Single-turn generation doesn't match multi-turn interaction patterns
2. **Long-horizon instability**: Standard importance sampling destabilizes over long proof sequences
3. **Train-inference gap**: Greedy decoding diverges from training distribution in low-confidence regions

HisPO directly addresses these through geometric averaging of token-level importance ratios.

## The Modification

**Hierarchical Importance Sampling Policy Optimization (HisPO)**

HisPO decomposes importance sampling into two controlled components:

```python
# Standard importance sampling (problematic for long sequences):
# IS_standard = p_new(τ) / p_old(τ) = Π(p_new(a_t) / p_old(a_t))
# Problem: multiplication of ratios across long sequences causes exponential variance

# HisPO solution: Sequence-level and token-level masking
# Step 1: Token-level ratio computation
token_ratios = [p_new(a_t) / p_old(a_t) for t in range(seq_len)]

# Step 2: Detect train-inference discrepancies
# Geometric average (more stable than arithmetic)
geometric_ratio = exp(mean(log(token_ratios)))

# Step 3: Sequence-level masking
# Remove sequences where geometric_ratio indicates large discrepancy
if geometric_ratio < threshold or geometric_ratio > inv_threshold:
    mask_sequence = 0  # Don't use this trajectory
else:
    mask_sequence = 1

# Step 4: Token-level masking (fine-grained control)
token_mask = [
    0 if |log(token_ratio)| > token_threshold else 1
    for token_ratio in token_ratios
]

# Step 5: Staleness control
# Manage policy updates for tokens affected by asynchronous training
staleness_mask = compute_staleness(token_timestamps, policy_update_time)

# Combined loss
loss = (mask_sequence * token_mask * staleness_mask) * policy_gradient(τ)
```

**Curriculum Learning Component:**

Decompose formal reasoning into three capabilities with progressive tool interaction:

1. **Auto-formalization**: Convert informal problem statement to Lean syntax
   - Tool: Syntax checker (one-turn validation)
2. **Sketching**: Outline proof structure
   - Tool: Type checker (mid-turn feedback)
3. **Proving**: Fill proof details with goal-driven interaction
   - Tool: Lean4 compiler (full multi-turn loop)

Each expert learns to dynamically select appropriate strategies based on problem difficulty.

**Legality Detection (Reward Hacking Prevention):**

Formal correctness doesn't guarantee semantic correctness:
```lean
-- Example: syntactically valid but semantically wrong proof
theorem foo : ∀ n, n = 0 := by
  intro n
  sorry  -- Admits goal without proving
```

Implement semantic validators on top of syntax checking to reject formally valid but logically invalid proofs.

## Ablation Results with Exact Numbers

### Auto-Formalization
- HisPO (tool-integrated): **97.1%** on MiniF2F-Test
- Baseline (no tools): **83%**
- **Delta: +14.1 percentage points**

### Theorem Proving (MiniF2F-Test)
- HisPO: **95.5%** with 72 attempts
- Competitors: 92.2% requiring 1,024+ attempts
- **Efficiency: 14.2× fewer attempts**

### Complex Reasoning (ProverBench)
- HisPO: **70.8%** within 220 attempts

### General Domain (PutnamBench)
- HisPO: **41.5%** within 220 attempts

## Conditions of Applicability

**Works well when:**
- Formal verification tools available (Lean, Coq, Isabelle)
- Multi-turn interaction is essential (proof search is inherently iterative)
- Train-inference distribution mismatch is significant (greedy decoding diverges from sampling)
- Sequence length is long (importance sampling variance becomes critical)

**Requires:**
- Curriculum structure matching reasoning stages
- Legality oracle to detect semantic errors
- Sufficient compute for multi-turn trajectory collection

**Less suitable when:**
- Single-turn generation is sufficient
- Verification tools unavailable
- Distribution shift is minimal
- Short sequences (standard IS works adequately)

## Drop-In Replacement Checklist

- [x] Compatible with any formal reasoning domain (Lean4, Coq, etc.)
- [x] No changes to proof executor (tool interface unchanged)
- [x] Curriculum-based training is orthogonal to base model
- [x] HisPO can be applied to any policy gradient method
- [x] Masking mechanism is agnostic to loss function
- [x] Legality detection can wrap any verification tool
