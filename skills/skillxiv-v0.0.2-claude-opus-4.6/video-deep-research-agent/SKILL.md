---
name: video-deep-research-agent
title: "Watching, Reasoning, and Searching: A Video Deep Research Benchmark on Open Web for Agentic Video Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.06943"
keywords: ["Video", "Deep Research", "Web Search", "Reasoning", "Agent", "Multi-hop", "Benchmark"]
description: "Implements video deep research for multi-hop reasoning combining video analysis, web search, and evidence synthesis. Evaluates workflow vs agentic paradigms with 100-sample benchmark across 6 semantic domains, revealing goal drift and long-horizon consistency as core bottlenecks."
---

## Overview

This skill implements the VideoDR methodology for evaluating video-grounded deep research agents. The paper presents the first benchmark combining video analysis with multi-hop web search and reasoning. Two execution paradigms are compared: Workflow (two-stage with explicit visual extraction) and Agentic (end-to-end with raw video access).

## Core Task Definition

Given a video V, question Q, and search tool S, produce factual answer A where:
- Models must extract multi-frame visual anchors from video
- Translate anchors into web search queries
- Perform multi-hop retrieval and verification
- Answer must depend on BOTH video and web evidence

## Key Findings

**Paradigm Performance (Agentic vs Workflow):**
- Gemini-3-pro-preview: 76% vs 69% (+7% Agentic)
- GPT-5.2: 69% vs 69% (tied)
- GPT-4o: 43% vs 42% (+1% Agentic)
- MiniCPM-V 4.5: 16% vs 25% (-9% Agentic penalty)

**Critical Insight:** Agentic is NOT consistently superior. Its advantage depends on model's ability to maintain initial video anchors across long search chains without re-watching.

## Bottlenecks Discovered

1. **Goal Drift** (Agentic exclusive): Search queries deviate from initial visual anchors when models lack strong state retention
2. **Long-Horizon Consistency**: Multi-round search without video re-watch causes drift, especially on videos >10min
3. **Numerical Reliability**: All models show persistent weakness on numerical answers (6-12 errors), independent of paradigm

## Workflow Paradigm (Two-Stage)

**Stage 1 - Visual Extraction:**
Input: Raw video + question → Output: Structured text of cross-frame visual anchors

**Stage 2 - Web Search & Reasoning:**
Input: Anchor text + question → Tools: search, think → Output: Answer

**Advantages:**
- Repeated access to visual summary prevents goal drift
- Better for spatial tasks (Geography: Gemini 70% Workflow vs 50% Agentic)
- Stable on long videos (Gemini 70% Workflow vs 50% Agentic on >10min)
- Supports weak models (MiniCPM-V 4.5: 25% vs 16%)

**When to use:** Long videos, spatial reasoning, weak perception models, consistent target maintenance

## Agentic Paradigm (End-to-End)

Input: Raw video + question → Single loop with search, think tools → Output: Answer

**Advantages:**
- Preserves full visual details
- Better for precision tasks (Technology: Gemini 64.29% Workflow vs 85.71% Agentic)
- No information loss from visual summarization

**Disadvantages:**
- Cannot re-watch after initial viewing
- Vulnerable to goal drift on long chains
- Fails with weak models (MiniCPM-V 4.5: drops from 25% to 16%)

**When to use:** Short videos, fine-grained visual reasoning, strong multimodal models

## Data Annotation & Quality Control

**Annotation Process (3 steps):**
1. Candidate Pool: Stratified sampling by source, domain, duration
2. Initial Filtering: Remove videos lacking prominent visual anchors
3. Question Design: Enforce multi-frame and multi-hop constraints

**Quality Verification (2 stages):**
- Web-only ablation: Question must NOT be answerable from web alone
- Video-only ablation: Question must NOT be answerable from video alone
- Human testing: 5 independent participants solve blindly

**Difficulty Stratification (by human success rate):**
- Low: 4-5 humans succeed (mean 90%)
- Mid: 2-3 humans succeed (mean 50.6%)
- High: 0-1 humans succeed (mean 10.6%)

## Performance Analysis

**By Difficulty Level:**
All models degrade consistently from Low→Mid→High. Agentic gains emerge on Mid/High for strong models (Gemini, GPT-5.2) but cause backslash on High for weak models (MiniCPM, GPT-4o).

**By Video Duration:**
- Short (<5 min): Paradigm effect minimal
- Long (>10 min): Agentic shows strong polarization
  - Strong models leverage details (Gemini: 50%→70%)
  - Weak models amplify drift (MiniCPM: 30%→10%)

**By Domain:**
- Technology: Agentic +21pp (Gemini 64%→86%)
- Geography: Workflow +20pp (Gemini 70%→50%)
- Daily Life: Balanced performance

## Tool Use Effectiveness

Table from paper shows tool count does NOT predict accuracy:

```
Model                Think  Search  Runtime(s)  Accuracy
Gemini-Agentic       2.89   2.52    449         76%
GPT-5.2-Agentic      3.06   2.74    1376        69%
Qwen3-Agentic        1.80   1.21    367         37%
MiniCPM-Agentic      1.97   2.07    139         16%
```

**Key insight:** Effectiveness is converting searches into high-quality evidence chains, not tool frequency.

## Error Breakdown

**Categorical Error** (dominant failure mode): 5-36 cases per model
- Increases in Agentic for weak models (inability to re-localize)
- Root cause: Initial perceptual error propagates through search chain

**Numerical Error** (persistent weakness): 6-12 cases per model
- Consistent across paradigms, all models
- Suggests fundamental LLM limitation on numerical reasoning

**Reasoning Error** (nearly absent): 0-1 cases
- Models perform adequately when given correct anchors

## Implementation Recommendations

Choose paradigm based on:

1. **Video Duration:**
   - <5 min: Either paradigm works
   - >10 min: Workflow strongly preferred unless using state-strong models (Gemini-class)

2. **Task Type:**
   - Spatial/geographic: Workflow (stable anchors essential)
   - Precision-critical (technology): Agentic (visual details matter)
   - General QA: Workflow (more robust)

3. **Model Capability:**
   - Weak perception (e.g., MiniCPM-V 4.5): Workflow mandatory
   - Strong reasoning models (Gemini, GPT-5.2): Agentic acceptable
   - Standard models (GPT-4o): Workflow recommended

4. **Evidence Requirements:**
   - Ambiguous web space: Workflow (consistent anchors guide search)
   - Clear web space: Either paradigm works

## Benchmark Composition

- **Size:** 100 annotated samples
- **Domains:** Daily Life (33%), Economics (16%), Technology (15%), Culture (15%), History (11%), Geography (10%)
- **Questions:** Average 25.54 tokens, 95th percentile 54 tokens
- **Video Duration:** Long-tailed distribution, mostly <5min, tail >10min
- **Human Baseline:** 50.4% average (90% Low, 50.6% Mid, 10.6% High)

## When NOT to use

- Closed-domain video QA (evidence entirely within video)
- Sub-second latency requirements (search operations cause delays)
- Numerical precision critical (models show systematic weakness)
- Agentic paradigm for tasks requiring spatial consistency (geographic reasoning)
- Agentic with weak perception models (significant performance degradation)
- Tasks where visual anchors are unclear or distributed (goal drift risk)

## References

- Paper: https://arxiv.org/abs/2601.06943
- HTML: https://arxiv.org/html/2601.06943
- Code: https://github.com/QuantaAlpha/VideoDR-Benchmark
