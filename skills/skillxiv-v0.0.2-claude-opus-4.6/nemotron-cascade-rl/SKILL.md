---
name: nemotron-cascade-rl
title: "Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.13607
keywords: [reinforcement-learning, cascaded-training, reasoning, language-models, multi-domain-rl]
description: "Train language models through sequential, domain-wise RL stages (RLHF → Instruction-Following → Math → Code → SWE) without catastrophic forgetting. Exploit policy-dependent training data distribution where previous behaviors persist when reward-relevant. 14B model surpasses DeepSeek-R1-0528 (671B) on LiveCodeBench."
---

## Skill Summary

Nemotron-Cascade introduces Cascade RL, departing from conventional mixed-domain RL by orchestrating sequential, domain-wise training stages. The approach applies reinforcement learning in progression—RLHF → Instruction-Following → Math → Code → SWE—each building upon previous stages without catastrophic forgetting. The key insight leverages policy-dependent training where previously learned behaviors persist when still reward-relevant, achieving state-of-the-art results where a 14B model surpasses DeepSeek-R1-0528 (671B) on LiveCodeBench.

## When To Use

- Training language models on multi-domain reasoning requiring progressive specialization
- Projects where sequential RL stages prevent catastrophic forgetting across domains
- Scenarios aiming for small unified models outperforming large domain-specific models
- Research exploring cascaded RL as alternative to simultaneous multi-domain training

## When NOT To Use

- Single-domain reasoning tasks where cascaded complexity adds overhead without benefit
- Scenarios requiring training all domains simultaneously for real-time adaptation
- Projects with limited compute where progressive stages increase total training time
- Domains with strong interference patterns between stages

## Core Technique

The core innovation is Cascade RL, applying sequential domain-wise RL:

**1. Sequential Domain-wise RL**
Apply reinforcement learning stages in cascading progression: RLHF → Instruction-Following → Math → Code → SWE. Each stage builds upon the previous without substantial performance degradation, leveraging domain ordering to enable progressive specialization.

**2. Policy-Dependent Training Data Distribution**
Key structural insight distinguishes RL from supervised learning: "the training data distribution is policy-dependent; the LLM generates its own experience." This means previously learned behaviors persist when still reward-relevant, enabling positive transfer between stages.

**3. Benefits from Earlier Stages**
Notably, RLHF substantially improves overall response quality in ways that benefit subsequent specialized reasoning tasks. Each stage enhances later stages through learned behavioral patterns.

**4. Results**
Achieve state-of-the-art: 14B thinking model surpasses DeepSeek-R1-0528 (671B) on LiveCodeBench and achieves silver-medal performance at IOI 2025, demonstrating smaller unified models can match dedicated reasoning models through careful cascaded training.

## Implementation Notes

Design sequential RL stages ordered by domain complexity/interference. Start with RLHF for general alignment. Progress through instruction-following, mathematical, coding, and engineering domains. Monitor for catastrophic forgetting and adjust rewards if performance regresses. Leverage positive transfer from earlier stages to enhance later specialization.

## References

- Original paper: Nemotron-Cascade (Dec 2025)
- Cascade reinforcement learning frameworks
- Multi-domain language model training
