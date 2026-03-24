---
name: stapo-spurious-token-silencing
title: "STAPO: Stabilizing RL for LLMs by Silencing Rare Spurious Tokens"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.15620"
keywords: [reinforcement learning, LLM training stability, gradient normalization, policy optimization, mathematical reasoning]
description: "Stabilize RL training in LLMs by detecting and masking gradient contributions from spurious tokens that comprise 0.01% of output but cause disproportionate instability. Identifies tokens with low probability, low entropy, and positive advantage, then suppresses their gradients during optimization to maintain stable policy entropy and improve reasoning performance by 7%+ across model scales."
---

# STAPO: Suppressing Spurious Token Gradients for Stable LLM Policy Learning

Large language models trained with reinforcement learning on reasoning tasks often experience training instability characterized by erratic policy entropy and performance collapse. This instability stems from spurious tokens—statistically rare outputs that the model assigns high advantage values to, causing them to dominate gradient updates despite representing <0.01% of tokens. These overconfident mistakes can reverse optimization progress and prevent the model from discovering diverse, correct reasoning paths.

The challenge lies in identifying which tokens genuinely contribute harmful signal versus which represent valid exploratory behavior. Spurious tokens share three properties simultaneously: very low generation probability, very low entropy (the model is overconfident), and positive estimated advantage (incorrectly valued as beneficial). Standard RL algorithms treat all non-optimal tokens identically, allowing this small spurious fraction to monopolize optimization.

## Core Concept

STAPO introduces the Silencing Spurious Tokens (S2T) mechanism, which operates as a selective gradient masking layer during policy optimization. Rather than redesigning the learning algorithm, S2T surgically removes gradient perturbations originating from problematic tokens, allowing the optimization process to focus on signal-bearing transitions.

The mechanism works by identifying tokens matching three criteria:
- Probability below threshold (typically τ_p = 0.002)
- Entropy below dynamic quantile (typically τ_h = 80th percentile)
- Positive advantage value

Only tokens satisfying all three conditions are masked, preserving legitimate exploratory signals.

## Architecture Overview

- **Detection Stage**: For each generated token in a batch, compute generation probability p(y|x), entropy H(π), and advantage estimate A(y|x) from the training batch
- **Masking Decision**: Flag token for masking if p(y|x) < τ_p AND H(π) < quantile(H, τ_h) AND A > 0
- **Loss Recalibration**: Suppress gradient flow for flagged tokens by zeroing their loss contributions
- **Normalization Adjustment**: Recompute loss normalization over only non-masked tokens to avoid numerical instability
- **Monitoring**: Track percentage of masked tokens per batch; expect 0.01%-0.1% in healthy training

## Implementation

The detection logic integrates into the loss computation phase of any policy gradient algorithm. For each training batch of (state, action, advantage) tuples, compute a binary mask before backpropagation:

```python
def detect_spurious_tokens(logits, targets, advantages, prob_threshold=0.002):
    """
    Identify tokens that are low-prob, low-entropy, and high-advantage.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    advantages: (batch, seq_len)
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Compute entropy per position
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    entropy_quantile = torch.quantile(entropy, 0.80)

    # Three-way mask
    low_prob = token_probs < prob_threshold
    low_entropy = entropy < entropy_quantile
    high_advantage = advantages > 0

    spurious_mask = low_prob & low_entropy & high_advantage
    return spurious_mask
```

Apply this mask during loss computation by zeroing the contributions of flagged tokens:

```python
def masked_policy_loss(logits, targets, advantages, spurious_mask):
    """
    Compute policy gradient loss with spurious token masking.
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Standard policy gradient
    pg_loss = -token_log_probs * advantages

    # Mask spurious tokens
    pg_loss = pg_loss * (~spurious_mask).float()

    # Normalize by non-masked tokens only
    num_valid = (~spurious_mask).sum()
    return pg_loss.sum() / (num_valid + 1e-8)
```

## Practical Guidance

| Hyperparameter | Default | Guidance |
|---|---|---|
| `prob_threshold` | 0.002 | Adjust based on vocabulary size; for 100k vocab, ranges 0.0005–0.005 |
| `entropy_quantile` | 0.80 | Lower (0.70) masks fewer tokens; higher (0.90) more aggressive |
| `max_masked_ratio` | 0.001 | Warn if >1% of tokens masked; indicates instability elsewhere |

**When to use**: Apply STAPO when training LLMs on reasoning tasks (math, code generation) where policy entropy oscillates or performance plateaus despite correct reward signals.

**When not to use**: Skip for supervised fine-tuning or tasks with sparse, noisy rewards where low-probability exploration is critical.

**Common pitfalls**:
- Setting thresholds too aggressively masks valid exploration; monitor masked token percentage
- Recomputing entropy quantiles per batch adds variance; compute once per training epoch
- Combining with entropy regularization requires tuning regularization strength to avoid double-suppression

## Reference

STAPO improves mathematical reasoning accuracy by 7.13% on average across model scales (1.7B, 8B, 14B) while maintaining stable policy entropy throughout training, demonstrating the technique's robustness to model capacity variations.
