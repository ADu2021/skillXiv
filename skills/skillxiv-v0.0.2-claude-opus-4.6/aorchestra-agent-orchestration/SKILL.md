---
name: aorchestra-agent-orchestration
title: "AOrchestra: Automating Sub-Agent Creation for Agentic Orchestration"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03786"
keywords: [Agent Orchestration, Multi-Agent Systems, Task Decomposition, Sub-Agent Creation, Framework]
description: "Automate sub-agent creation by treating agents as dynamically creatable executors defined by four-tuple abstraction (Instruction, Context, Tools, Model), enabling flexible delegation and cost-aware routing for complex multi-step tasks."
---

# AOrchestra: Automating Sub-Agent Creation via Dynamic Four-Tuple Abstraction

The orchestration bottleneck in multi-agent systems stems from treating agents as fixed roles rather than dynamic executors. AOrchestra solves this by decoupling orchestration from execution through a unified four-tuple abstraction: each sub-agent is defined by its task instruction, relevant context, available tools, and reasoning model. This enables an orchestrator to spawn task-specific agents on-the-fly rather than managing pre-defined roles, reducing unnecessary context and improving cost efficiency across diverse benchmarks.

## Core Concept

AOrchestra models multi-agent systems as two layers: a master orchestrator that only delegates and concludes, and dynamically-created sub-agents that execute specific subtasks. The four-tuple (Instruction, Context, Tools, Model) fully specifies each agent, allowing the orchestrator to optimize per-subtask while maintaining framework-agnostic compatibility with any sub-agent implementation.

## Architecture Overview

- **Orchestrator Layer**: Operates exclusively through two actions—delegating subtasks to spawned agents and returning final answers
- **Sub-Agent Instantiation**: Each subtask spawns a new agent configured with its task instruction, task-specific context window, required tools, and assigned reasoning model
- **Cost-Aware Routing**: Learns to select models and tools based on task complexity and performance thresholds
- **Learning Mechanisms**: Combines supervised fine-tuning for decomposition quality with in-context learning for iterative cost optimization

## Implementation

### Step 1: Define the Four-Tuple Abstraction

The orchestrator maintains a template for sub-agent instantiation, specifying how to construct each tuple from task decomposition outputs.

```python
def create_sub_agent_tuple(task_decomposition, available_models, available_tools):
    """
    Maps task decomposition to four-tuple specification.
    Returns: (instruction, context, tools, model)
    """
    instruction = f"Solve this subtask: {task_decomposition['subtask']}"
    context = retrieve_relevant_context(task_decomposition['subtask'])
    tools = select_tools_for_task(task_decomposition['subtask'], available_tools)
    model = select_model_by_complexity(task_decomposition['subtask'], available_models)
    return (instruction, context, tools, model)
```

### Step 2: Implement Orchestrator Actions

The orchestrator uses two primitive actions: delegating to sub-agents and finishing.

```python
def orchestrator_policy(problem_state, learned_delegate_policy):
    """
    Generates delegation decisions via learned policy.
    Outputs: either ("delegate", subtask_list) or ("finish", final_answer)
    """
    if should_finish(problem_state):
        return ("finish", synthesize_final_answer(problem_state))
    else:
        subtasks = learned_delegate_policy(problem_state)
        return ("delegate", subtasks)
```

### Step 3: Supervised Fine-Tuning for Decomposition

Train the orchestrator to generate high-quality task decompositions through behavior cloning.

```python
def sft_training(orchestrator_model, expert_trajectories):
    """
    Fine-tune on expert decomposition examples.
    expert_trajectories: list of (problem, expert_decomposition) pairs
    """
    optimizer = torch.optim.Adam(orchestrator_model.parameters(), lr=1e-4)
    for problem, expert_decomp in expert_trajectories:
        predicted_decomp = orchestrator_model(problem)
        loss = cross_entropy_loss(predicted_decomp, expert_decomp)
        loss.backward()
        optimizer.step()
    return orchestrator_model
```

### Step 4: In-Context Learning for Cost Optimization

Refine routing decisions iteratively based on observed costs and successes.

```python
def in_context_learning_iteration(orchestrator, task_history):
    """
    Adjust model selection based on recent task performance.
    task_history: list of (subtask, model, cost, success) tuples
    """
    performance_per_model = aggregate_performance(task_history)
    routing_instruction = build_routing_prompt(performance_per_model)
    orchestrator.system_prompt = routing_instruction
    return orchestrator
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|----------------|-------|
| Task Complexity | 3-8 subtasks per task | Avoid under-decomposition and over-decomposition overhead |
| Model Selection | Align with subtask complexity | Small models for simple tasks, large for reasoning-heavy |
| Context Curation | Task-relevant facts only | Reduces token costs while maintaining performance |
| Learning Strategy | SFT first, then in-context learning | Supervised stabilizes decomposition; in-context optimizes costs |
| Failure Mode | Incomplete decomposition | Monitor coverage of subgoals to catch missing steps |

**When to Use:**
- Multi-step problems requiring specialized execution modes
- Cost-sensitive agent deployments with per-subtask optimization
- Systems needing flexible agent creation without predefined roles

**When Not to Use:**
- Single-turn reasoning tasks (orchestration overhead not justified)
- Systems requiring strict latency bounds (spawning adds delay)
- Tasks with heavy inter-subtask dependencies (limits parallelization)

## Reference

Achieves 16.28% relative improvement on GAIA, Terminal-Bench 2.0, and SWE-Bench-Verified benchmarks while maintaining compatibility across diverse agent implementations.
