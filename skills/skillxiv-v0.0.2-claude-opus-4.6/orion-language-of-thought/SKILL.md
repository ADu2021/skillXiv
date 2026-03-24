---
name: orion-language-of-thought
title: "ORION: Teaching LMs to Reason Efficiently via Language of Thought"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2511.22891
keywords: [reasoning-efficiency, language-models, mentalese, reinforcement-learning, structured-reasoning]
description: "Compact Mentalese symbolic format trained via SFT, then refined with SLPO (Shorter Length Preference Optimization) to reward concise correct solutions without over-penalizing necessarily long reasoning. Compress reasoning while maintaining accuracy."
---

## Summary

ORION introduces Mentalese, a compact symbolic reasoning format inspired by cognitive science, paired with Shorter Length Preference Optimization (SLPO), an RL method that adaptively rewards concise solutions. The two-stage approach first aligns models to structured reasoning traces via supervised fine-tuning, then uses RLVR with SLPO to recover accuracy while maintaining compression.

## Core Technique

**Mentalese Format:** A symbolic reasoning language more compact than natural language. For example, instead of reasoning in English prose, express reasoning as a series of symbolic operations:
```
[assign:x=5] [op:multiply:x:2:->y] [verify:y==10] [conclude:true]
```
This compact format reduces token count while maintaining reasoning structure.

**Shorter Length Preference Optimization (SLPO):** Standard reward models might over-penalize long but necessary solutions. SLPO uses an adaptive formulation:
```
reward = accuracy_bonus - λ(length) * penalty
```
where λ varies based on solution validity. Only penalize length for correct solutions.

**Two-Stage Training:**
1. SFT: Teach models to output Mentalese reasoning traces
2. RLVR + SLPO: Apply verifier rewards + length penalties, adapting penalties to avoid failing valid long solutions

## Implementation

**Mentalese template design:** Define symbolic operators for your domain:
- Assignment/computation: [op:action:args:->var]
- Verification: [verify:condition]
- Reasoning: [assume:statement]
- Conclusion: [conclude:result]

**SFT alignment:** Collect or generate reasoning traces in Mentalese format. Train with standard language modeling loss on these traces.

**SLPO formulation:** Define reward:
```python
def slpo_reward(mentalese_trace, correctness, length):
    acc_bonus = 1.0 if correctness else -0.5
    if correctness:
        length_penalty = -0.1 * log(length)  # Gentle penalty for correct long solutions
    else:
        length_penalty = 0  # No penalty for incorrect (already penalized)
    return acc_bonus + length_penalty
```

**Verifier-guided RL:** Train a verifier to check Mentalese trace validity, then use verifier signals as part of the reward.

## When to Use

- Reasoning tasks where token efficiency matters (cost, latency, or bandwidth)
- Applications benefiting from structured symbolic reasoning
- Scenarios where reasoning correctness is more important than brevity
- Models needing interpretable reasoning traces for auditing or debugging

## When NOT to Use

- Natural language reasoning where prose is preferred
- Tasks where symbolic reasoning is difficult to formalize
- Scenarios requiring freeform reasoning without structure
- Applications where longer explanations provide better user experience

## Key References

- Mentalese and symbolic reasoning languages
- Reinforcement learning for reasoning and verification
- Length-controlled generation and adaptive penalties
- Chain-of-thought prompting and reasoning optimization
