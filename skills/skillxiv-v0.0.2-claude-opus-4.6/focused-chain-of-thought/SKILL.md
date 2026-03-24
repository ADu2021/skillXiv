---
name: focused-chain-of-thought
title: "Focused Chain-of-Thought: Structured Information Extraction for Efficient Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2511.22176
keywords: [prompt-engineering, reasoning-efficiency, information-extraction, structured-formatting, token-reduction]
description: "Training-free prompting strategy that pre-organizes query information into compact structured format, reducing generation tokens by 2-3× while maintaining reasoning accuracy. Apply when reasoning performance is bottlenecked by verbose input formatting."
---

## Summary

Focused Chain-of-Thought (F-CoT) is a training-free prompting strategy that improves inference efficiency by separating information extraction from reasoning. The method first organizes relevant information from a query into a structured, compact format, then guides the model to reason exclusively over this organized context rather than the original verbose question.

## Core Technique

The approach operates in two stages:

**Stage 1 - Information Organization:** Parse the original query and extract relevant facts, constraints, and context. Format these into a clean, structured representation (tables, lists, key-value pairs). This extraction removes verbal noise and redundancy.

**Stage 2 - Focused Reasoning:** Present the organized information to the model alongside a query, asking it to reason exclusively from the structured context. The model no longer needs to parse or filter through verbose original text.

This two-stage separation enables the model to focus computation on reasoning rather than information comprehension.

## Implementation

**Extraction template:** Design a format for your domain. Examples:
- For math: "Given: [list constraints], Find: [objective], Constraints: [list conditions]"
- For code review: "Files changed: [list], Key patterns: [list], Testing requirements: [list]"
- For analysis: "Data points: [structured table], Hypothesis: [statement], Constraints: [list]"

**Structured representation:** Convert original text into tables, bullet points, or key-value pairs. Each element should be a complete, self-contained fact.

**Focused prompt:** Chain-of-thought prompt operating only on structured context:
```
Context: [structured information]
Task: [reasoning goal]
Reasoning:
```

## When to Use

- Tasks with naturally verbose inputs (documents, specifications, chat histories)
- Scenarios where token budget is constrained (long-context applications)
- Reasoning tasks benefiting from organized context representation
- Applications where input clarity directly impacts reasoning quality

## When NOT to Use

- Tasks where implicit context hidden in prose is important to reasoning
- Scenarios where information extraction itself is complex or error-prone
- Applications with highly unstructured or ambiguous inputs
- Tasks benefiting from full original context for semantic understanding

## Key References

- Chain-of-thought prompting and reasoning strategies
- Information extraction and structured formatting techniques
- Token efficiency and inference cost optimization
