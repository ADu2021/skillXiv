---
name: revere-reflective-research-agent
title: "REVERE: Reflective Evolving Research Engineer for Scientific Workflows"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.20667
keywords: [Agentic AI, Self-Adaptation, Research Code, Prompt Optimization, Reflective Learning]
description: "Enable LLM agents to autonomously improve on research-code tasks through reflective learning from execution trajectories. Distill recurring failure patterns into actionable heuristics applied via targeted prompt edits, improving performance 3.51%-4.89% while maintaining 10x better cost-efficiency."
---

## Capability Gap Addressed
Existing prompt-optimization techniques fail at research-coding tasks because they rely on local signals (single-batch feedback) and full-prompt rewrites, causing poor generalization and knowledge loss. Research-coding workflows demand handling of heterogeneous repositories, underspecified environments, and weak feedback—conditions where static prompts and multi-agent systems struggle.

## Core Abstractions

**1. Execution Trajectory Reflection** - Record failure patterns across repositories: dependency resolution issues, environment configuration errors, data formatting mismatches, incomplete library specifications.

**2. Heuristic Distillation** - Convert recurring failure patterns into reusable heuristics (e.g., "always check Python version compatibility before importing," "validate data encoding before processing").

**3. Prompt Field Edits** - Apply heuristics as surgical Python programs rather than full-prompt regeneration, targeting three configurable fields:
   - System prompt (agent role and capabilities)
   - Task-prompt template (structured task specification)
   - Cumulative cheatsheet (repository-specific and cross-repository remedies)

## Design Decisions

**Global Training Context** - Maintains across adaptation iterations:
- Reflection history (what failures were observed and how they were addressed)
- Auxiliary context (environment specs, common libraries, known issues)
- Cumulative cheatsheet (growing repository of solutions)

This enables learning beyond immediate batch feedback and knowledge retention across iterations.

**Code-Based Field Updates** - Generates Python programs that apply surgical edits to prompts rather than regenerating prompts entirely. Benefits:
- Prevents semantic drift from full rewrites
- Enables precise, targeted modifications
- Preserves coherent system-state understanding

**Unified Reflector Module** - Single agent diagnoses failures and performs edits rather than splitting responsibilities across multiple agents. Advantages:
- Maintains coherent understanding of system state
- Reduces inter-agent communication overhead
- Enables holistic learning across failure types

## Self-Improvement Patterns
REVERE autonomously identifies and accumulates:
1. **Infrastructure issues** - Dependency resolution, environment setup, library version mismatches
2. **Code-specific patterns** - Common error patterns in particular codebases, optimization opportunities
3. **Cross-repository remedies** - General solutions applicable across multiple projects

The paper demonstrates that "agents equipped with mechanisms for continual learning...can meaningfully evolve their capabilities," with improvements persisting across new repositories through the evolving cheatsheet.

## Integration Patterns

**Continuous Adaptation Loop**:
1. Execute agent on batch of research-code tasks
2. Record execution trajectories and identify failures
3. Reflector analyzes patterns and generates prompt edits
4. Apply edits via code-based modification
5. Re-execute on same tasks to validate improvements
6. Evolving cheatsheet transfers to new repositories

**Generalization Mechanism** - The cumulative cheatsheet acts as transferable knowledge, allowing improvements on Repo A to benefit performance on Repo B without explicit transfer learning.

## Performance Profile
- 3.51%-4.89% improvement over expert-crafted baseline instructions
- 10x better cost-efficiency than alternative adaptation methods (fewer API calls, fewer full rewrites)
- Improvements measured across three research-coding benchmarks
- Knowledge retention: heuristics persist and apply to new repositories
