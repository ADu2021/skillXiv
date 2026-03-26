---
name: llm-agent-workflow-optimization-survey
title: "From Static Templates to Dynamic Runtime Graphs: A Survey of LLM Agent Workflow Optimization"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22386"
keywords: [LLM Agents, Workflow Optimization, Agentic Computation Graphs, Dynamic Planning, Structure-Aware Evaluation]
description: "Navigate LLM agent workflow design by understanding the taxonomy of static vs dynamic methods and the dimensions that organize them. Agentic Computation Graphs (ACGs) framework distinguishes when structure is determined (before vs during execution), which components optimize, and what signals guide optimization. Provides structure-aware evaluation criteria combining downstream metrics with graph properties and robustness. Use when designing or optimizing agent workflows, choosing between fixed templates and dynamic adaptation, or evaluating workflow efficiency."
category: "Survey & Synthesis"
---

## Field Overview

LLM agent workflows—the orchestration of language model reasoning, tool calling, and feedback loops—have become a central research area. However, the field lacks a common taxonomy for describing and comparing workflows. Papers discuss "planning," "prompting," "tool selection," and "re-planning" but with inconsistent terminology. This survey unifies the landscape around Agentic Computation Graphs (ACGs): abstract structures representing how an agent decides to call tools, in what order, and with what parameters.

## Taxonomy: The ACG Framework

The survey organizes workflows along three orthogonal dimensions:

**Dimension 1: When Structure Is Determined**
- **Static workflows**: Fix a reusable scaffold before deployment. Examples: ReAct-style fixed chains, predefined tool sequences. Advantage: reproducible, easy to optimize. Disadvantage: rigid, doesn't adapt to task variation.
- **Dynamic workflows**: Select, generate, or revise the workflow for a particular run. Happens before execution (planning) or during execution (reactive). Advantage: adapts to task; Disadvantage: higher latency, harder to optimize, less reproducible.

**Dimension 2: Which Component Optimizes**
Each workflow stage (reasoning, tool selection, tool sequencing, execution feedback) is an optimization point:
- Reasoning optimization: Better prompts, better in-context examples, chain-of-thought structures
- Tool selection: Which tools are available for this task? Should this task call tools at all?
- Tool sequencing: In what order should tools be called? Parallel or serial?
- Execution feedback: How does the agent learn from tool results?

**Dimension 3: What Guides Optimization**
Four signal sources tell the agent whether the workflow is working:
- **Task metrics**: Final answer accuracy, downstream performance
- **Verifier signals**: Does an external model think the answer is correct?
- **Preference signals**: Human feedback on which workflows are better
- **Trace feedback**: Properties of the workflow itself (execution cost, token usage, intermediate reasoning quality)

## Key Distinctions the Survey Makes

The survey clarifies three often-confused concepts:

**Workflow Template** vs **Realized Graph** vs **Execution Trace**

- A **template** is the reusable blueprint (e.g., "if task is math, use calculator; if task is reading, use search"). Shared across many problem instances.
- A **realized graph** is the specific workflow instantiated for one problem (e.g., "for this math problem, call calculator once then reason"). Generated from the template.
- An **execution trace** is what actually happened when running the realized graph (e.g., "calculator call took 50ms, returned value X, agent then took 2 seconds to reason"). Observed after execution.

Many papers conflate these levels. Understanding the distinction is crucial for evaluating workflows fairly.

## Method Comparison: Static vs Dynamic Trade-offs

| Approach | When to Use | Pros | Cons | Computation Cost |
|---|---|---|---|---|
| **Static Template** | Known, homogeneous task distribution | Reproducible, fast, easy to debug | Suboptimal on task variation, no adaptation | Low (single template reused) |
| **Dynamic Planning** (decide before execution) | Diverse tasks, heterogeneous problem types | Adapts workflow per task, near-optimal | Higher latency, harder to optimize, varies cost | Medium (planning overhead + execution) |
| **Reactive Revision** (adapt during execution) | Unpredictable tasks, long horizons | Recovers from errors, truly adaptive | Highest latency, unpredictable cost, error-prone learning | High (real-time adaptation) |

**Decision Criteria**: Choose static if your task distribution is tight (similar problems, similar workflows). Choose dynamic planning if tasks vary but you can batch (the planning overhead is paid once). Choose reactive only if you need to recover from live failures (e.g., real-time robotics).

## Structure-Aware Evaluation Framework

The survey advocates evaluating agent workflows on four dimensions simultaneously, not just downstream accuracy:

1. **Downstream Task Performance**: Does the agent solve the original task? (Accuracy, F1, whatever metric the task demands)

2. **Graph Properties**: Is the workflow itself reasonable? Examples:
   - **Execution cost**: How many tool calls? How many tokens? How much compute?
   - **Stability**: Does the same workflow reliably converge for similar problems, or is it chaotic?
   - **Interpretability**: Can humans understand why this workflow was chosen?

3. **Robustness**: How does the workflow perform on adversarial inputs, distribution shift, or noisy tool results?
   - Does the agent gracefully recover from tool failures?
   - Does it abandon bad workflows or get stuck?

4. **Structural Variation Analysis**: For the same task, does the agent explore multiple workflows or fixate on one? Is this appropriate?
   - Good: Flexible, adaptive
   - Bad: Unstable, inconsistent
   - Context-dependent: Sometimes you want stability, sometimes flexibility

## Literature Navigation: Key Conceptual Tracks

The survey identifies distinct research threads within LLM agent workflows:

**Track 1: Prompt Engineering Track**
Foundation: Few-shot examples, chain-of-thought, system prompts guide reasoning
Key papers: CoT, few-shot prompting, instruction tuning
For practitioners: Start here if you want quick wins without complex infrastructure

**Track 2: Tool Integration Track**
Foundation: How to expose tools, describe their capabilities, handle errors
Key papers: ReAct, Gorilla (tool selection), ToolBench
For practitioners: Read if building practical agents with real APIs/tools

**Track 3: Planning Track**
Foundation: How to decompose complex goals into sub-tasks
Key papers: Tree-of-Thoughts, Graph-of-Thoughts, hierarchical planning
For practitioners: Read if solving long-horizon or multi-step problems

**Track 4: Feedback & Optimization Track**
Foundation: How agents learn from mistakes and improve workflows
Key papers: Self-correction, DPO on trajectory feedback, RLHF on workflows
For practitioners: Read if building continuously-improving systems

**Track 5: Efficiency Track**
Foundation: Minimizing cost while maintaining performance
Key papers: Speculative execution, early stopping, tool caching, parallel execution
For practitioners: Read if latency or cost is constrained

## Open Questions & Future Directions

The survey identifies gaps the field must address:

1. **Workflow Compositionality**: How do you combine learned workflows? Can you take a workflow that solves sub-task A and sub-task B and compose them to solve A+B? Current approaches treat the combined problem monolithically.

2. **Cost-Performance Pareto Frontier**: For a given task, what is the Pareto frontier of (accuracy, latency, cost)? How to choose operating points? Currently, papers optimize one axis without mapping the frontier.

3. **Robustness Under Tool Failure**: Real tools fail. How do agents gracefully degrade? Current benchmarks assume tools work perfectly.

4. **Generalization Across Domains**: A workflow optimized for math tasks doesn't transfer to reading comprehension. How to design workflows that generalize?

5. **Human-in-the-Loop Workflows**: How do agents decide when to ask humans for help vs. solving autonomously? Who is responsible when agents fail?

6. **Scalable Evaluation Infrastructure**: Evaluating workflows is expensive. We need standardized, reusable benchmarks (like the survey argues for) rather than one-off evaluations.

## When to Use This Skill

Use this field guide when designing new agent systems (should I use fixed prompts or dynamic planning?), comparing proposed agent architectures, or evaluating agent research papers (what dimensions matter?). Also useful when troubleshooting existing agents (is the problem in reasoning, tool selection, or feedback loops?).

## When NOT to Use

This skill is a navigation tool, not a implementation guide. It won't tell you exact prompts or code. If you need specific algorithm details, refer to papers in the relevant track (Planning, Feedback, etc.). Also, this survey's value is in understanding the landscape, not in showing you the single best approach—because no single approach is best for all tasks.
