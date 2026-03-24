---
name: unreasonable-scaling-computer-use-agents
title: "Unreasonable Effectiveness of Scaling Computer Use Agents with Behavior Judgment"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.02250
keywords: [computer-use-agents, scaling, evaluation, trajectory-understanding]
description: "Scale computer-use agents from 30% to 72% success rate by generating parallel rollouts and selecting best trajectories through behavior narrative evaluation. Use when deploying desktop agents on complex, high-variance task scenarios."
---

# Unreasonable Effectiveness of Scaling Computer-Use Agents with Behavior Judgment

This work identifies a fundamental bottleneck in scaling computer-use agents: "wide scaling exposes a fundamental bottleneck: evaluation." The solution converts dense agent trajectories into compact behavior narratives capturing task-relevant changes, enabling intelligent best-of-N selection via comparative evaluation.

## Core Architecture

- **Parallel trajectory execution**: Generate multiple independent agent rollouts per task
- **Behavior narrative generation**: Compress trajectories into task-relevant summaries (10-20% original length)
- **Comparative evaluation**: Select best trajectory through structured pairwise comparison
- **Cross-platform validation**: Works on OSWorld (desktop), WindowsAgentArena (Windows), AndroidWorld (mobile)

## Implementation Steps

Setup parallel agent executor with behavior narrative evaluation:

```python
# Initialize parallel computer-use agent system
from bjudge import ParallelAgentExecutor, BehaviorNarrativeGenerator

executor = ParallelAgentExecutor(
    base_agent=your_computer_agent,
    num_rollouts=4,  # generate 4 parallel attempts per task
    executor="isolated_vms",  # each rollout in isolated environment
    snapshot_capability=True  # required for rollback
)

narrative_generator = BehaviorNarrativeGenerator(
    llm=your_evaluator_model,
    max_narrative_length=1000,  # tokens
    focus="task_relevant_changes"
)
```

Execute parallel agent execution with trajectory selection:

```python
# Task execution with parallel rollouts
task = "Download Q1 financial reports from company website and extract revenue"

# Stage 1: Generate multiple rollouts in parallel
trajectories = executor.execute_parallel(
    task=task,
    num_rollouts=4,
    max_steps_per_rollout=50,
    environment_snapshots=True
)

# Stage 2: Convert trajectories to behavior narratives
narratives = []
for trajectory in trajectories:
    narrative = narrative_generator.generate(
        trajectory=trajectory,
        task_description=task,
        include_action_sequence=True,
        include_state_changes=True,
        summarization_ratio=0.1  # compress to 10% of original
    )
    narratives.append(narrative)

# Stage 3: Comparative evaluation to select best trajectory
best_trajectory_idx = executor.select_best_trajectory(
    narratives=narratives,
    task=task,
    evaluation_criteria=[
        "task_completion",
        "action_efficiency",
        "error_recovery",
        "state_consistency"
    ]
)

# Stage 4: Return best result
best_result = trajectories[best_trajectory_idx]
print(f"Task success rate: {best_result.success}, Steps: {best_result.num_steps}")
```

## Practical Guidance

**When to use Behavior Judgment:**
- Desktop automation with complex, multi-step workflows
- Web scraping and data extraction tasks with high variance
- Software testing and interaction workflows
- Systems where user trust depends on interpretable agent decisions

**When NOT to use:**
- Real-time systems (parallel rollouts add latency overhead)
- Tasks where reproducibility is essential (multiple attempts may produce different results)
- Systems without snapshot/rollback capability (required for isolated rollouts)
- Single-step deterministic tasks (parallelization overhead not justified)

**Performance characteristics:**
- **OSWorld (desktop)**: 72.6% success rate (up from 30% baseline)
- **WindowsAgentArena**: 56.6% success rate
- **AndroidWorld**: 71.6% success rate
- **Human performance on OSWorld**: ~75% (agent approaching human level)

**Hyperparameters:**
- **Num rollouts (4)**: Sweet spot between quality and latency. Test 3-5; 2 insufficient, 6+ diminishing returns
- **Max steps (50)**: Adjust based on task complexity; increase to 80 for very complex workflows
- **Narrative length (1000 tokens)**: Must fit in evaluator context; reduce to 500 for larger models
- **Summarization ratio (0.1)**: 10% of original trajectory length; adjust to 0.05-0.15 based on task specificity

**Computational requirements:**
- **Parallel VM setup**: 4 isolated environments for baseline (4 simultaneous rollouts)
- **Total latency**: 4x single-agent time (sequential execution) or near-equal (parallel VMs)
- **Evaluation overhead**: 10-15% additional compute for trajectory selection

## Trajectory Compression Strategy

Behavior narratives preserve critical information while eliminating noise:
- **Keep**: State changes, tool results, error messages, decision points
- **Discard**: Intermediate steps, UI details, navigation noise
- **Compress**: Action sequences into high-level summaries (e.g., "searched for Q1 report PDF")

## Cross-Domain Results

The approach generalizes across distinct platforms:
- **OSWorld**: Complex desktop web interactions
- **WindowsAgentArena**: Windows-specific applications
- **AndroidWorld**: Mobile app workflows
- Performance scales reasonably across domains (56-72% success)

## Key Limitation

Approach requires isolated execution environments capable of snapshots. Standard cloud VMs support this; real user systems may not, limiting real-world deployment in some contexts.

## References

Builds on work in agent evaluation, trajectory-based learning, and multi-rollout selection strategies.
