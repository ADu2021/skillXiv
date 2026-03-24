---
name: flash-searcher-dag-parallel-agents
title: "Flash-Searcher: DAG-Based Parallel Execution for LLM Web Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2509.25301
keywords: [agent-orchestration, parallel-execution, DAG, web-agents, efficiency]
description: "Reduce agent execution steps by 35% and latency by parallelizing sequential tool calls through task dependency graphs (DAGs). Use when deploying information-retrieval agents where tool execution ordering is flexible."
---

# Flash-Searcher: DAG-Based Parallel Execution for LLM Web Agents

Flash-Searcher replaces sequential agent reasoning with concurrent subtask execution via dependency-aware directed acyclic graphs (DAGs). This framework decomposes complex agent tasks into independent execution paths while maintaining logical coherence, reducing steps by 35% and improving overall latency.

## Core Architecture

- **DAG decomposition**: Analyzes task dependencies to identify independent subtasks
- **Dynamic workflow optimization**: Adjusts parallelization strategy based on runtime execution
- **Dependency tracking**: Ensures downstream tasks receive upstream results appropriately
- **Lightweight fine-tuning**: Adaptable to smaller models (8B parameters)

## Implementation Steps

Setup DAG-based agent execution framework:

```python
# Initialize DAG-based parallel agent executor
from flash_searcher import DAGAgentExecutor, DependencyAnalyzer

executor = DAGAgentExecutor(
    model=your_llm,
    max_parallel_tasks=4,
    task_timeout=30,
    execution_strategy="dynamic"
)

# Create dependency analyzer for task decomposition
analyzer = DependencyAnalyzer(
    llm=your_llm,
    tool_catalog=web_tools
)
```

Execute task decomposition and parallel execution:

```python
# Decompose task into DAG structure
task = "Find the top 3 restaurants in NYC with highest ratings, then check their hours"

dag = analyzer.decompose(
    task=task,
    tools=["search", "get_info", "check_hours"]
)

# DAG structure shows parallelizable stages:
# Stage 1: search(restaurant_query)  [parallel with start]
# Stage 2: get_info(restaurant_results)  [parallel with start]
# Stage 3: check_hours(hours_info)  [depends on Stage 2]

# Execute with parallelization
results = executor.execute(
    dag=dag,
    max_parallel=4,
    dynamic_optimization=True
)

# Return integrated results (67.7% success rate on BrowseComp)
print(f"Task completed in {results.execution_time:.2f}s ({35% fewer steps})")
```

## Practical Guidance

**When to use Flash-Searcher:**
- Information retrieval agent systems where tool dependencies are partially ordered
- Web search and data aggregation tasks with flexible execution ordering
- Scenarios where latency reduction is critical (customer-facing systems)
- Deployments on constrained hardware (fine-tuned 8B models achieve near-GPT4 performance)

**When NOT to use:**
- Code generation and execution (omitted from this framework due to sequentiality)
- Mathematical reasoning requiring step-by-step verification
- Tasks with strict sequential dependencies where parallelization offers no benefit
- Systems where tool execution order significantly affects result quality

**Hyperparameters:**
- **Max parallel tasks (4)**: Increase to 6-8 for retrieval-heavy workloads; decrease to 2 for complex dependencies
- **Task timeout (30s)**: Adjust based on slowest tool latency; shorter timeouts increase failure risk
- **Dynamic optimization**: Enable for unpredictable task patterns; disable for consistent workloads
- **Model size**: 8B models sufficient for BrowseComp (68% success); use 32B for complex decomposition

## Performance Metrics

- **Step reduction**: 35% fewer intermediate steps vs. sequential agents
- **Latency improvement**: 1.7x faster execution vs. sequential baselines
- **Success rate**: 67.7% on BrowseComp benchmark
- **Model transferability**: Lightweight fine-tuning on Qwen-2.5 (7B) achieves 68% on xbench

## Training Data

The paper provides 3,354 curated task decomposition examples enabling:
- Knowledge distillation to smaller models
- Domain-specific fine-tuning
- Transfer learning to new agent tasks

## Notable Limitation

The framework deliberately omits code execution tools due to sequentiality requirements. Mathematical reasoning performance could improve with appropriate computational tools but accepts latency tradeoff.

## Architecture Notes

DAG decomposition enables two complementary benefits:
1. **Parallel execution**: Independent tasks run concurrently
2. **Efficient memory management**: Minimize intermediate result retention
3. **Dynamic reconfiguration**: Adjust parallelization mid-execution based on task progress

## References

Extends prior work on parallel decoding and multi-agent orchestration systems.
