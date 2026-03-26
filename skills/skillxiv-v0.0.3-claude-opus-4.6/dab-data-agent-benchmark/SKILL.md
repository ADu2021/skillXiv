---
name: dab-data-agent-benchmark
title: "DAB: Data Agent Benchmark"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.20576"
keywords: [Data Agent, Multi-Database Querying, Benchmark Design, Agent Evaluation, Structured Reasoning]
description: "Evaluate data agents on realistic multi-database queries across 54 tasks spanning 12 datasets, 9 domains, and 4 DBMS systems. Reveals frontier models achieve only 38% pass@1 accuracy, with 85% of failures from incorrect planning rather than data selection. Benchmark captures critical challenges: heterogeneous database integration, identifier reconciliation, unstructured text extraction, and domain knowledge. Identifies optimal exploration ratio (~20% of operations) and cost-efficiency differences across models."
category: "Evaluation Infrastructure"
---

## Benchmark Scope & Design

DAB evaluates data agents on realistic query tasks that require multi-step reasoning across heterogeneous structured data sources. The benchmark comprises 54 queries distributed across:

- **12 Diverse Datasets**: Patents, academic publications, financial records, product catalogs, public health databases, geographic data
- **9 Domains**: Healthcare, finance, technology, e-commerce, law, science, demographics, infrastructure, government
- **4 Database Systems**: SQL (PostgreSQL, MySQL), NoSQL (MongoDB), spreadsheet formats, requiring agents to handle different schema interfaces

Query complexity ranges from single-table lookups to multi-stage joins across incompatible databases with data cleaning steps. This heterogeneity ensures benchmarking real-world agent challenges rather than uniform database navigation.

## Task Definition & Evaluation Metrics

**Query Format**: Natural language instruction plus optional context. Example: "Find patents filed by XYZ company in biotechnology sector in the last 5 years, ranked by citation count." Agent must interpret intent, plan exploration, locate relevant tables, construct queries, and extract/aggregate results.

**Evaluation Criteria**:
- **pass@1**: First-attempt correctness; best model achieves only 38% accuracy
- **pass@50**: Accuracy with up to 50 retry attempts; performance caps at 69%, indicating fundamental limitations not just sampling variance
- **Unsolved problems**: Patents dataset remains 0% solvable across all models, suggesting domain-specific knowledge barriers

**Success Definition**: Returned data matches ground-truth answer set exactly. Partial matches or near-correct queries count as failures, reflecting real-world requirements where approximate answers are unacceptable.

## Error Analysis: Root Causes

The benchmark reveals systematic failure modes across all frontier models:

**Planning Failures (85% of errors)**: Agents construct incorrect query plans even when data is discoverable. Mistakes cluster in:
- **Multi-database integration**: Failing to identify which tables store required data when information is scattered across systems
- **Join strategy errors**: Using incorrect join columns or join types when identifiers don't match cleanly
- **Aggregation mistakes**: Incorrect GROUP BY clauses, SUM vs COUNT confusion, or wrong filtering order

**Data Selection Failures (15% of errors)**: Correctly planned queries access wrong data sources or miss relevant tables entirely.

**Text Extraction Bottleneck**: No agent attempts NLP-based or LLM-based text extraction from unstructured fields; all use regex patterns that fail on variable formatting.

## Optimal Exploration Strategy

Data exploration (inspecting table schemas, column definitions, sample rows) is essential but over-exploration wastes operations:

- **Minimal exploration (<10%)**: Agents proceed without understanding schema, missing relevant tables
- **Optimal exploration (~20%)**: Agents allocating roughly one-fifth of operations to schema discovery and sample inspection perform best
- **Excessive exploration (>30%)**: Agents get lost exploring deep schema trees, exhausting token budgets before executing queries

This reveals a critical tradeoff: agents need sufficient schema context to plan correctly but must reserve operations for execution.

## Cost-Efficiency Insights

Model efficiency varies dramatically:

- **GPT-5-mini**: Best cost-to-accuracy ratio despite lower absolute performance than flagship models
- **Expensive models**: Gemini-3-Pro achieves highest accuracy (38% pass@1) but high token costs dominate budgets
- **Production systems**: PromptQL improved baseline by 7 points through semantic layer construction, though entirely fails on text extraction

## When to Use

Use DAB when evaluating agent capability on structured reasoning tasks requiring plan decomposition, multi-step exploration, and integration of information across heterogeneous sources. This benchmark is essential for:
- Data analysis tools targeting business intelligence use cases
- SQL generation and agent systems
- Research on agent planning and long-horizon reasoning
- Cost-accuracy tradeoff analysis in model selection

## Practical Implications for Deployment

- **Plan validation**: Implement intermediate checking of query plans before execution to catch planning errors early
- **Text extraction**: Augment agents with specialized text extraction modules; regex-only approaches fail systematically
- **Exploration budgets**: Calibrate operation budgets for initial exploration (20%) and execution (80%)
- **Domain adaptation**: Models require significant fine-tuning on domain-specific data; generic training leaves 38% pass rate far below production requirements
- **Parallelization potential**: Unused opportunity for concurrent database queries; agent plans often serialize operations unnecessarily

## Validation Checklist

- [ ] Test on heterogeneous database setups, not single-system benchmarks
- [ ] Analyze error distribution: planning vs selection vs extraction
- [ ] Measure exploration-to-execution operation ratio
- [ ] Evaluate text extraction capability separately from query planning
- [ ] Compare cost-per-correct-answer across candidate models, not just accuracy
