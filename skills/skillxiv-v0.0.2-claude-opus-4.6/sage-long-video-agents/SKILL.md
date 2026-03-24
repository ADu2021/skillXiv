---
name: sage-long-video-agents
title: "SAGE: Smart Any-Horizon Agents with Reinforcement Learning for Long-Context Video Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.13874
keywords: [video-understanding, agent-reasoning, reinforcement-learning, long-context, multi-turn-reasoning]
description: "Enable agents to reason over long videos through multi-turn reasoning for complex questions and single-turn answering for simpler queries. Equip orchestrator VLM with multiple tools (web search, speech transcription, temporal grounding). Use synthetic data generation and multi-reward RL achieving 6.1% improvements on open-ended tasks."
---

## Skill Summary

SAGE introduces an agent-based system for long video reasoning operating in two paradigms: multi-turn reasoning for complex questions and single-turn answering for simpler queries. The orchestrator VLM (SAGE-MM) is equipped with multiple tools including web search, speech transcription, and temporal grounding. Synthetic data generation leverages Gemini-2.5-Flash achieving nearly 100× cost and 10× time savings. Multi-reward RL approach combines step-level rewards with LLM-as-judge evaluation, achieving 6.1% improvements on open-ended tasks and 8.2% gains for videos exceeding 10 minutes.

## When To Use

- Building systems for long-duration video understanding and reasoning
- Scenarios requiring adaptive reasoning strategies based on question complexity
- Projects where temporal grounding and external tools enhance reasoning
- Research on multi-turn video agents with RL optimization

## When NOT To Use

- Short-form video tasks not benefiting from multi-turn complexity
- Applications without access to external tools or context-augmentation
- Scenarios where synthetic data quality is problematic
- Domains where simpler single-pass models suffice

## Core Technique

Multiple innovations enable effective long video reasoning:

**1. System Design**
Use orchestrator VLM (SAGE-MM) equipped with multiple tools rather than relying solely on temporal grounding:
- Web search: external context
- Speech transcription: audio understanding
- Temporal grounding: video-specific localization

Implement two paradigms:
- Multi-turn reasoning: complex questions requiring multiple reasoning steps
- Single-turn answering: simpler queries with direct answers

**2. Synthetic Data Generation**
Leverage Gemini-2.5-Flash to generate high-quality QnA pairs efficiently, achieving "nearly 100× cost and 10× time savings compared to human annotation and subclip processing pipelines."

**3. RL Post-Training Recipe**
Implement multi-reward approach combining:
- Step-level rewards: format, reasonable tool use, argument validity
- LLM-as-judge evaluation: for open-ended problems avoiding string-matching metrics

**4. SAGE-Bench Evaluation**
Curated benchmark with "1744 manually verified samples spanning diverse durations" emphasizing open-ended questions from entertainment videos.

## Results

- Open-ended tasks: 6.1% improvement
- Videos exceeding 10 minutes: 8.2% improvement
- Validates effectiveness of adaptive reasoning for variable-duration content

## Implementation Notes

Design orchestrator VLM with tool integration. Implement multi-turn vs. single-turn reasoning routing. Use synthetic data generation for training pairs. Implement multi-reward RL with step-level and judge-based evaluation. Evaluate on videos of varying durations and question complexities.

## References

- Original paper: SAGE (Dec 2025)
- Agent-based video understanding
- Long-context video reasoning
