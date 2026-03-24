---
name: stable-rl-with-llms
title: "Stabilizing Reinforcement Learning with LLMs: Token-Level Objectives and Routing Replay"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.01374
keywords: [reinforcement-learning, llm-training, token-level-optimization, moe-routing, training-stability]
description: "Justifies token-level optimization objectives as first-order approximation to sequence rewards via Routing Replay, which fixes MoE routers during policy optimization to reduce training-inference discrepancy. Use when scaling RL training on large MoE language models."
---

## Summary

This paper provides a novel formulation for reinforcement learning with LLMs that justifies using token-level optimization objectives to optimize sequence-level rewards. The key insight is that this approximation holds when training-inference numerical discrepancies and policy staleness are minimized. The primary technique, Routing Replay, stabilizes MoE model training by fixing routed experts during policy optimization.

## Core Technique

**First-Order Approximation Theory:** Traditional RL optimizes sequence-level rewards (final answer correctness), but LLMs optimize token-level losses. The paper shows that token-level objectives are a valid first-order approximation when:
- Training and inference use identical router decisions (no router staleness)
- Numerical precision remains consistent across rollouts (no discrepancy)

**Routing Replay:** During policy optimization, freeze the expert router assignments computed during data collection. This ensures that gradient updates through generators don't change which experts are active, maintaining consistency between training and inference.

**Token-Level Objective:** Optimize the standard RL loss at each token: ℒ = -log p(a_t | s_t) * advantage_t, where advantage is computed from the sequence-level reward signal.

## Implementation

**Router freezing:** In MoE models, compute router decisions r = Router(hidden_states) once per sequence during data collection. Store these assignments. During gradient computation, use fixed r rather than recomputing them on updated hidden states.

**Routing consistency check:** Before training, verify that fixing routers doesn't significantly change model outputs on a validation set. Large divergence indicates the approximation may be invalid.

**Token-level reward computation:** Use sequence-level rewards to compute advantages for each token. Standard approaches: TD-lambda, GAE, or final-answer-only reward signal at the last token.

## When to Use

- Training MoE language models with reinforcement learning at scale
- Scenarios where token-level optimization is more tractable than sequence-level
- Applications requiring stability when training very large models
- Tasks combining RL with mixture-of-experts routing

## When NOT to Use

- Non-MoE models where router sticking is not an issue
- Supervised fine-tuning or non-RL training scenarios
- Applications where routing flexibility during training is important
- Scenarios where first-order approximation breaks down (very stale policies)

## Key References

- Policy gradient methods and advantage estimation in RL
- Mixture-of-experts routing and load balancing
- Training-inference discrepancy in large language models
- Token-level versus sequence-level objective formulations
