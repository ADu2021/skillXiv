---
name: qwenlong-long-context-reasoning
title: "QwenLong-L1.5: Post-training for Long-Context Reasoning and Memory-Augmented Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.12967
keywords: [long-context, reasoning, memory-augmented, reinforcement-learning, multi-hop-grounding]
description: "Systematically post-train models for long-context reasoning through multi-hop data synthesis, stabilized RL with adaptive entropy control, and memory-augmented architecture supporting 4M+ token sequences. Achieves performance comparable to GPT-5 and Gemini-2.5-Pro on long-context benchmarks."
---

## Skill Summary

QwenLong-L1.5 introduces a comprehensive post-training system combining three technical contributions: (1) long-context data synthesis generating challenging reasoning tasks requiring multi-hop evidence grounding, (2) stabilized RL methods including task-balanced sampling and adaptive entropy-controlled policy optimization, and (3) memory-augmented architecture handling ultra-long sequences through iterative memory-based processing for sequences exceeding 4M tokens.

## When To Use

- Building models that reason over extremely long documents (4M+ tokens)
- Projects requiring multi-hop reasoning across globally distributed evidence
- Scenarios where retrieval-augmented generation would be too slow or complex
- Research exploring memory-augmented LLM architectures for long-context tasks

## When NOT To Use

- Inference scenarios with strict latency requirements (long-context processing has inherent delays)
- Short-context tasks where the overhead of memory architecture doesn't pay off
- Projects without computational resources to fine-tune large models with RL
- Domains where simpler retrieval systems already provide sufficient performance

## Core Technique

Three interconnected innovations enable long-context reasoning at scale:

**1. Long-Context Data Synthesis Pipeline**
Develops systematic framework generating challenging reasoning tasks. Deconstructs documents into atomic facts and relationships, then programmatically composes verifiable questions. Includes three specialized methods: knowledge-graph-guided multi-hop reasoning, structural tabular data engines for numerical problems, and multi-agent self-evolving framework for general tasks.

**2. Stabilized Reinforcement Learning Methods**
Overcome training instability in long-context RL through:
- Task-balanced sampling with task-specific advantage estimation mitigating reward distribution bias
- Negative gradient clipping targeting high-entropy tokens in failed responses
- Adaptive Entropy-Controlled Policy Optimization (AEPO) "dynamically regulating exploration-exploitation trade-offs" by masking negative-advantage sequences when entropy exceeds target thresholds

**3. Memory-Augmented Architecture**
For sequences exceeding 4M tokens, implement memory agent framework combining single-pass reasoning within 256K context window with iterative memory-based processing through multi-stage fusion RL training.

## Implementation Notes

Implement multi-hop data synthesis pipeline starting with document decomposition. Set up stabilized RL training with task-balanced sampling and adaptive entropy control. For ultra-long sequences, implement memory agent framework with iterative processing. Apply progressive training from shorter to longer context windows.

## References

- Original paper: QwenLong-L1.5 (Dec 2025)
- Long-context language model training techniques
- Adaptive entropy-controlled policy optimization
