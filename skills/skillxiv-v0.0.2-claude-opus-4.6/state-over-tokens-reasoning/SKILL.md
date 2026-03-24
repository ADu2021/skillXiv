---
name: state-over-tokens-reasoning
title: "State Over Tokens: Understanding Reasoning Tokens as Computational State, Not Explanation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.12777
keywords: [interpretability, reasoning-tokens, language-models, computational-state, ontological-divergence]
description: "Reframe reasoning tokens as externalized computational state—the sole persistent information carrier across stateless generation cycles—not human-readable explanations. Model LLM generation as recursive token prediction where state diverges from human semantic interpretation."
---

## Skill Summary

This paper introduces the State over Tokens (SoT) framework, reframing how we understand reasoning tokens in language models. Rather than interpreting reasoning tokens as human-readable explanations of model reasoning, the framework treats them as "an externalized computational state—the sole persistent information carrier across the model's stateless generation cycles." The work resolves critical misconceptions about reasoning token completeness and semantic meaning.

## When To Use

- Research on LLM interpretability and reasoning token analysis
- Projects exploring mechanistic understanding of how models use intermediate representations
- Scenarios where you need to understand what reasoning tokens actually encode
- Studies on the gap between model-internal computation and human-readable explanations

## When NOT To Use

- Engineering applications focused solely on improving output quality (interpretability research, not practical method)
- Scenarios assuming reasoning tokens are fully human-interpretable explanations
- Projects where the computational overhead of token analysis isn't justified by insights
- Domains where treating tokens as explanations is working well enough

## Core Technique

The framework models LLM generation as recursive application of a function ℳ(·) where each cycle produces one token that becomes part of the input for the next cycle. Key insights:

**1. Externalized Computational State**
Reasoning tokens serve as "the sole persistent information carrier across the model's stateless generation cycles." Unlike human explanations, these tokens encode functional information needed for the next generation step.

**2. Misconception of Completeness**
Reasoning tokens don't represent the full computation, only "what is functionally necessary for the next cycle." Human attempts to fully interpret reasoning tokens fundamentally misunderstand their purpose.

**3. Misconception of Shared Meaning**
The model's internal interpretation of tokens diverges significantly from human semantic understanding, potentially involving arbitrary encoding schemes. The same token sequence functions simultaneously as natural language text (readable to humans) and computational substrate (functional to the model).

**4. Ontological Divergence**
A single artifact (token sequence) represents fundamentally incompatible interpretive modes: natural language form (human-readable) and computational substrate (functional to model).

## Implementation Notes

When analyzing reasoning tokens, shift perspective from "what explanation does this provide?" to "what computational state is this encoding?" Recognize that token interpretability at face value may be misleading. Model generation as recursive state evolution where tokens carry functional rather than semantic information. Use this framework to guide interpretability research toward understanding model-internal computation rather than expecting human-readable explanations.

## References

- Original paper: State Over Tokens (Dec 2025)
- LLM mechanistic interpretability literature
- Token embedding analysis and representation learning
