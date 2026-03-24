---
name: acon-context-compression-long-horizon
title: "ACON: Agent Context Optimization for Long-Horizon Tasks"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.00615
keywords: [context-compression, long-horizon-agents, efficiency, prompt-optimization]
description: "Compress agent interaction histories and environment observations through natural language guideline optimization, reducing token usage by 26-54% while preserving 95%+ accuracy. Use for cost/latency reduction in multi-step agent tasks."
---

# ACON: Agent Context Optimization for Long-Horizon Tasks

ACON addresses unbounded context growth in multi-step agents through compression guideline optimization in natural language space. Rather than parameter fine-tuning, the approach optimizes natural language instructions that guide what to compress, enabling closed-source API compatibility and rapid iteration.

## Core Architecture

- **Compression guidelines**: Natural language rules describing what to compress
- **Gradient-free optimization**: No parameter updates; pure prompt/instruction learning
- **Contrastive training**: Optimize guidelines using compressed vs. uncompressed agent performance
- **Flexible integration**: Works with closed-source models (GPT-4, Claude)
- **Composable**: Stack compression guidelines for layered reduction

## Implementation Steps

Setup ACON compression optimizer:

```python
# Initialize ACON for agent context compression
from acon import CompressionOptimizer, CompressionGuidelineManager

# Create compression guideline manager
guideline_manager = CompressionGuidelineManager(
    initial_guidelines=[
        "Remove intermediate steps that don't affect final decisions",
        "Abbreviate repetitive tool outputs",
        "Summarize multi-turn conversation threads"
    ],
    optimization_strategy="contrastive"
)

# Initialize optimizer
optimizer = CompressionOptimizer(
    agent_model="gpt-4",
    compression_model="gpt-4",  # can differ from agent
    guideline_manager=guideline_manager,
    max_iterations=10
)
```

Execute contrastive guideline optimization:

```python
# Optimization loop for compression guidelines
from acon import AgentExecution

for iteration in range(num_optimization_iterations):
    # Task set to optimize over
    test_tasks = load_benchmark_tasks()  # e.g., AppWorld, OfficeBench

    scores_by_guideline = {}

    for task in test_tasks:
        # Execute agent with full uncompressed context
        full_execution = AgentExecution(
            agent_model="gpt-4",
            task=task
        )
        full_result = full_execution.run()
        full_accuracy = evaluate(full_result, task.ground_truth)
        full_tokens = count_tokens(full_execution.trajectory)

        # Execute agent with current compression guidelines
        compressed_execution = AgentExecution(
            agent_model="gpt-4",
            task=task,
            compression_guidelines=guideline_manager.current_guidelines
        )

        # Compressor applies guidelines to interaction history
        compressed_trajectory = compressed_execution.compress_context(
            full_trajectory=full_execution.trajectory,
            guidelines=guideline_manager.current_guidelines,
            compression_ratio_target=0.5  # 50% of original
        )

        # Continue execution with compressed context
        compressed_result = compressed_execution.run(
            compressed_trajectory=compressed_trajectory
        )
        compressed_accuracy = evaluate(compressed_result, task.ground_truth)
        compressed_tokens = count_tokens(compressed_trajectory)

        # Evaluation metrics
        accuracy_retained = compressed_accuracy / full_accuracy
        token_reduction = 1 - (compressed_tokens / full_tokens)

        # Record score (reward for efficiency + penalty for accuracy loss)
        score = 0.6 * token_reduction - 0.4 * max(0, 1 - accuracy_retained)
        scores_by_guideline[task.id] = score

    # Optimize guidelines based on scores
    avg_score = np.mean(list(scores_by_guideline.values()))
    print(f"Iteration {iteration}: Avg score = {avg_score:.3f}")

    # Generate improved guidelines
    improved_guidelines = optimize_guidelines_with_lm(
        current_guidelines=guideline_manager.current_guidelines,
        evaluation_scores=scores_by_guideline,
        full_trajectories=[e.trajectory for e in test_executions],
        optimization_prompt=GUIDELINE_OPTIMIZATION_PROMPT
    )

    guideline_manager.update_guidelines(improved_guidelines)
```

## Practical Guidance

**When to use ACON:**
- Multi-step agent tasks with unbounded context growth (web agents, API orchestration)
- Cost-sensitive deployments where token reduction directly impacts budget
- Closed-source model usage (OpenAI, Anthropic APIs)
- Scenarios where rapid guideline iteration useful (developing new task types)

**When NOT to use:**
- Tasks where compression accuracy loss unacceptable
- Real-time systems where optimization overhead prohibitive
- Open-source models where parameter fine-tuning more efficient
- Domains where context preservation critical (legal, medical reasoning)

**Benchmark domains:**
- **AppWorld**: Desktop application automation
- **OfficeBench**: Multi-application office tasks
- **Multi-objective QA**: Complex question-answering requiring multi-step planning

**Hyperparameters:**
- **Compression ratio target (0.5)**: 50% token reduction typical. Test 0.3-0.7 range
- **Optimization iterations (10)**: More iterations refine guidelines; 5 for rapid prototyping
- **Accuracy retention threshold (0.95)**: Preserve 95%+ of full context accuracy
- **Token reduction weight (0.6)**: Balance efficiency vs. accuracy; adjust to 0.4-0.8
- **Guideline count (3-5)**: Start small; add more for complex compression needs

## Compression Strategy

Guidelines focus on:
- **Temporal compression**: Remove old/irrelevant interactions
- **Semantic compression**: Abbreviate verbose outputs while preserving meaning
- **Structural compression**: Remove intermediate decision paths, keep final results
- **Redundancy elimination**: Deduplicate repeated tool outputs

## Performance Results

**Token reduction:** 26-54% depending on domain
- AppWorld: 35-45% reduction
- OfficeBench: 26-35% reduction
- Multi-objective QA: 40-54% reduction

**Accuracy retention:** 95%+ maintained
- Minimal performance loss despite aggressive compression
- Larger models (32B) compress better than smaller (7B)

## Computational Cost

- **Guideline optimization**: 10 iterations ~$10-50 (depends on model/token usage)
- **Per-task compression**: <5% overhead vs. full context execution
- **Total savings**: Break-even after 5-10 deployed tasks

## Distillation Capability

Compressed guideline knowledge transfers to smaller models:
- **32B agent compressed**: 35-45% token reduction
- **Distill guidelines to 8B agent**: Achieves 70-80% of 32B performance at 80% lower cost

## Implementation Notes

**Gradient-free design enables:**
- Rapid iteration without retraining
- Closed-source model compatibility
- Easy rollback and A/B testing
- Version control and guideline inheritance

## Current Limitations

Approach requires running agents twice (full + compressed) for optimization loop, creating upfront compute overhead. Amortized over many deployments, but matters for one-shot tasks.

## References

Builds on prompt optimization, context compression, and agent efficiency literature.
