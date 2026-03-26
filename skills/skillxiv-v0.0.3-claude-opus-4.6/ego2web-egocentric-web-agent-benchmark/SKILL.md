---
name: ego2web-egocentric-web-agent-benchmark
title: "Ego2Web: Egocentric Video to Web Agent Execution Benchmark"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22529"
keywords: [Web Agent, Egocentric Video, Multimodal Grounding, Physical-Digital Integration, Automatic Evaluation]
description: "Evaluate web agents on tasks grounded in first-person video perception with Ego2Web benchmark. Bridges egocentric video understanding and web agent execution across e-commerce and knowledge lookup domains. Includes Ego2WebJudge LLM-based evaluator achieving 84% human agreement, replacing unreliable domain-specific metrics. Reveals weak agent performance on physically-grounded tasks, highlighting the necessity of accurate multimodal visual understanding."
category: "Evaluation Infrastructure"
---

## Problem Statement

Existing web agent benchmarks treat task specification as abstract text instructions, disconnecting from real-world scenarios where humans perceive task context through visual observation. Ego2Web addresses this gap by grounding agent tasks in egocentric (first-person) video: agents must understand real-world visual context, then execute digital tasks on web interfaces. This physical-digital integration better captures realistic agent deployment where visual context informs task interpretation.

## Dataset Construction

The benchmark pairs first-person video recordings with web task execution requirements across multiple domains:

**Source Material**: Egocentric video recordings from human agents performing real-world activities (e.g., shopping, research).

**Task Specification**: Each video grounds abstract task instructions in concrete first-person visual context. Tasks span:
- E-commerce domains: finding products based on visual preferences observed in video
- Knowledge lookup: answering questions by integrating video context with web search
- Cross-domain: tasks requiring both video understanding and web navigation coordination

**Annotation Protocol**: Expert annotators verify that tasks are solvable from video context alone and that web solutions achieve stated objectives.

**Split Strategy**: Train/test splits prevent agents from memorizing task-response pairs while maintaining domain diversity across splits.

## Evaluation Methodology: Ego2WebJudge

Task evaluation in physically-grounded scenarios presents a challenge: domain-specific metrics (task success, final screenshot comparison) often fail to recognize valid solutions or penalize reasonable intermediate failures.

**Ego2WebJudge Protocol**: Uses a fine-tuned LLM evaluator that assesses whether agent behavior matches task intent:

- **Input context**: Video frames establishing task context, original task description, target success state, and recorded agent interaction sequence
- **Evaluation criterion**: LLM judges whether agent actions coherently pursue the specified goal given video context, scoring success/partial/failure
- **Calibration**: Human evaluators assess identical tasks, with Ego2WebJudge achieving 84% agreement with human judgment

This approach replaces brittle domain-specific success criteria with nuanced task-semantics judgment, enabling fair comparison across heterogeneous task types.

## Benchmark Analysis & Findings

Evaluation of state-of-the-art agents reveals substantial headroom and weak performance across task categories:

- **Multimodal necessity**: Ablation studies demonstrate that accurate video understanding is essential; agents failing on visual grounding fail downstream web tasks
- **Domain variance**: Some task domains show higher agent success than others, suggesting systematic capability gaps
- **Failure analysis**: Agent struggles cluster around translating visual preferences into web search queries and navigating unfamiliar interface layouts

## When to Use

Use this benchmark when developing or evaluating web agents for real-world deployment scenarios where task context comes from visual observation. This is critical for:
- Assistive agents helping users with visual limitations
- Mobile-first agents that must coordinate phone camera perception with web navigation
- Research on embodied AI systems that integrate perception and digital action

## When the Text-Only Model Still Applies

Text-only task specifications remain valuable for studying pure web navigation capability isolation. Use Ego2Web when multimodal grounding is central to your research question; use text-only benchmarks when studying web interface understanding independent of perception.

## Validation Checklist

- [ ] Record agent video interactions on tasks, not just success/failure
- [ ] Run Ego2WebJudge evaluation alongside domain-specific metrics to verify consistency
- [ ] Analyze failure modes: are they vision-based or navigation-based?
- [ ] Test agents on video-grounded tasks with varied visual contexts (cluttered backgrounds, partial occlusion)
- [ ] Verify multimodal necessity through ablation: performance drops when removing video context
