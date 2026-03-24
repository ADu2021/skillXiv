---
name: versatileffn-adaptive-ffn
title: "VersatileFFN: Adaptive Wide-and-Deep Reuse in Language Models via Difficulty-Aware Computation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.14531
keywords: [feed-forward-networks, parameter-efficiency, mixture-of-experts, adaptive-computation, dynamic-scaling]
description: "Enable parameter-efficient computation through dual-pathway feed-forward networks. Create virtual experts via hidden subspace slicing (width-versatile) and recursive weight application (depth-versatile). Use difficulty-aware gating to balance pathways, adding computation not memory to fixed parameter budget."
---

## Skill Summary

VersatileFFN introduces a parameter-efficient feed-forward architecture that reuses parameters across two complementary computational pathways: (1) width-versatile pathway creating virtual experts through hidden subspace slicing, and (2) depth-versatile pathway recursively applying FFN weights for iterative refinement. A difficulty-aware gating mechanism dynamically balances pathways, routing easy tokens through efficient width processing and allocating deeper computation to hard tokens—inspired by dual-process cognition theory.

## When To Use

- Building language models with flexible computation-parameter trade-offs
- Scenarios where parameter budgets are fixed but computational budgets are flexible
- Projects inspired by dual-process cognition models
- Research exploring efficient alternatives to standard mixture-of-experts

## When NOT To Use

- Latency-sensitive applications where variable-depth computation adds unpredictability
- Scenarios with strict computational budgets but flexible parameter budgets
- Domains already using standard FFNs that work well
- Applications requiring uniform computational cost across all tokens

## Core Technique

Two complementary computation pathways share parameters:

**1. Width-Versatile Pathway**
Create multiple virtual experts by slicing a shared base FFN into non-overlapping hidden subspaces, mimicking mixture-of-experts routing without increasing parameters. Route different tokens to different slices.

**2. Depth-Versatile Pathway**
Recursively apply the same FFN weights multiple times, allowing tokens to undergo iterative refinement with a token-specific iteration count predicted via Gumbel-Softmax. Easy tokens exit early; hard tokens receive multiple passes.

**3. Difficulty-Aware Fusion**
A gating mechanism dynamically balances the two pathways based on expected computational depth. Route "easy" tokens through the efficient width path and allocate deeper processing to "hard" tokens. As authors state: "both pathways reuse the same parameters, so all additional capacity comes from computation rather than memory."

## Implementation Notes

Start with standard FFN architecture. Split hidden state into multiple non-overlapping subspaces for width versatility. Implement recursive FFN application with token-specific iteration prediction. Design gating mechanism to balance pathways based on token difficulty. This approach inspired by dual-process cognition enables flexible computation allocation within fixed memory.

## References

- Original paper: VersatileFFN (Dec 2025)
- Mixture-of-experts architectures
- Dual-process cognition theory
