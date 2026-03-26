---
name: agentslr-automated-literature-reviews
title: "AgentSLR: Automating Systematic Literature Reviews"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22327"
keywords: [Systematic Literature Reviews, Epidemiology, Agentic AI, Evidence Synthesis, Domain Automation]
description: "Automate systematic literature reviews in epidemiology using agentic AI pipelines. Achieves 58x speed-up (7 weeks to 20 hours) by automating article retrieval, screening, data extraction, and report synthesis. Demonstrates that review quality depends on model capabilities rather than scale. Use when conducting evidence-based reviews in specialized domains, need to validate against human expertise, or require cost-effective evidence synthesis at scale."
category: "Application Transfer"
---

## Domain Problem

Systematic literature reviews (SLRs) in epidemiology are critical for evidence-based policy but present a severe bottleneck: the manual review process takes approximately 7 weeks of expert time per project. This limits how frequently health authorities can synthesize emerging evidence on priority pathogens, vaccines, and interventions. WHO and health agencies need rapid, reliable evidence synthesis to inform policy decisions during disease outbreaks and healthcare planning.

## The Gap

Traditional SLR workflows require expert epidemiologists to manually screen articles, extract data, and synthesize findings. This is slow, expensive, and limits the frequency of evidence updates. Naive application of general-purpose LLMs fails because:
- Epidemiological screening requires domain-specific judgment about study design quality and relevance
- Data extraction from heterogeneous research papers demands understanding of study protocols, statistical methods, and outcome definitions
- Report synthesis must integrate evidence from disparate sources into actionable policy recommendations

Simply prompting a language model to "find relevant papers" produces high false-positive rates and misses domain-specific inclusion criteria.

## Source Technique

The foundation is multi-step agentic AI workflows where an LLM orchestrates a pipeline of specialized tasks. Rather than a single forward pass, the agent iterates through retrieve-screen-extract-synthesize steps, maintaining context and refining decisions based on intermediate results. This mimics the human SLR process: humans don't decide inclusion once; they refine criteria as they encounter papers.

## Adaptation Recipe

AgentSLR bridges the gap between general LLM capability and epidemiological SLR requirements through:

1. **Workflow Decomposition**: Break the monolithic "conduct a literature review" task into substeps (retrieve articles, screen by title/abstract, screen by full text, extract structured data, synthesize findings). Each step can be validated independently against expert judgment.

2. **Domain-Specific Prompting**: Provide the LLM with explicit inclusion/exclusion criteria, outcome definitions, and quality metrics from the research protocol before executing retrieval. This anchors the agent to domain constraints rather than letting it invent criteria.

3. **Human-in-the-Loop Validation**: Identify failure modes through comparison with expert-curated benchmarks. Rather than trusting the agent fully, use expert validation to detect when the agent diverges from domain norms (e.g., missing important study designs or applying criteria inconsistently).

4. **Multi-Model Capability Matching**: Test across frontier models and select based on task-specific performance, not raw scale. The paper found that "performance is driven less by model size or inference cost than by each model's distinctive capabilities." Some models excel at screening, others at data extraction—match the model to the task.

5. **Evidence Integration**: Structure the agent to accumulate extracted data into a synthesis report, maintaining traceability from raw articles to final evidence statements. This enables auditing and builds confidence in policy recommendations.

## Deployment Guide

**Data Pipeline**: Obtain structured research protocols defining inclusion criteria, population, interventions, and outcomes. Format as system prompts or structured JSON. Retrieve candidate articles from PubMed, WHO databases, or institutional repositories. The agent will screen and extract in stages, not all at once.

**Validation Strategy**: Benchmark the agent against 5-10 manually-reviewed articles to calibrate performance. Identify systematic biases (e.g., over-inclusive, over-exclusive, misunderstanding study design). Adjust prompts or switch models based on performance on these calibration cases.

**Failure Modes**: Watch for three key failure patterns: (1) False positives in screening—missing nuance in study eligibility, (2) Data extraction errors—misreporting effect sizes or study arms, (3) Misaligned synthesis—drawing conclusions not supported by extracted data. Use expert spot-checks to detect these early.

**Deployment Considerations**: For production use, implement human review of agent-screened articles at boundaries (borderline inclusions/exclusions) and all extracted data before it feeds into policy recommendations. The 58x speed-up comes from automating clear-cut screening and extraction, not eliminating expert oversight. Budget 20% expert time for validation.

**Monitoring**: Track agreement rates between the agent and expert reviewers over time. If drift occurs (e.g., inclusion criteria interpretation shifts), retrain prompts or switch models. Maintain a log of disagreements to improve prompting.

## When to Use This Skill

Use AgentSLR-style automation when conducting evidence synthesis in specialized domains (epidemiology, medical policy, environmental health) where you need to rapidly synthesize dispersed literature and have access to domain experts for validation. This approach works best with well-defined protocols and clear inclusion criteria.

## When NOT to Use

Do not use this approach if your review scope is poorly defined, domain criteria are ambiguous, or you lack access to expert validators. AgentSLR requires both clear protocols and human oversight—it's not a replacement for expert review, but an accelerant of it. Also avoid if the domain requires interpretation of complex statistical methods the agent may misunderstand.
