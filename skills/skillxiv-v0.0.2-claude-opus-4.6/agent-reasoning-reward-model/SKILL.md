---
name: agent-reasoning-reward-model
title: "Exploring Reasoning Reward Model for Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.22154"
keywords: [agent-reinforcement-learning, reward-modeling, reasoning-trajectory, process-reward, agent-training]
description: "Build multi-faceted reward models for agent trajectories that provide structured feedback on intermediate reasoning quality. Implement explicit reasoning traces, focused critiques with refinement guidance, and overall process scores to train more effective agentic agents without relying solely on sparse outcome rewards."
---

## Problem

Agentic reinforcement learning systems typically rely on sparse outcome-based rewards that fail to differentiate intermediate reasoning quality, leading to suboptimal agent training. Agents need process-level feedback during their reasoning chains to improve beyond trial-and-error learning.

## Solution

Implement an Agent Reasoning Reward Model (Agent-RRM) that produces three types of structured feedback for agentic trajectories:

1. **Explicit Reasoning Trace**: Extract and validate the logical flow of agent reasoning steps
2. **Focused Critique**: Identify specific reasoning flaws and provide refinement guidance
3. **Overall Process Score**: Evaluate the quality and efficiency of the entire reasoning sequence

## When to Use

- Training agentic systems that perform complex reasoning (planning, tool use, multi-step problem-solving)
- When outcome-only rewards are too sparse to guide learning effectively
- When you need to improve agent trajectory quality beyond simple success/failure signals
- For agents performing web navigation, code generation, or research tasks

## When NOT to Use

- Simple one-step decision tasks (reward shaping may be overkill)
- Environments where outcome rewards are dense and informative
- Real-time agents where computing process rewards adds prohibitive latency

## Implementation

### Step 1: Design the Reasoning Trace Extractor

Implement a component that identifies and validates reasoning steps in agent trajectories.

```python
def extract_reasoning_trace(trajectory):
    """
    Extract logical steps from agent trajectory.
    trajectory: list of (observation, action, thought) tuples
    Returns: structured trace with step IDs and dependencies
    """
    trace_steps = []
    for i, (obs, action, thought) in enumerate(trajectory):
        step = {
            "id": i,
            "observation": obs,
            "thought": thought,
            "action": action,
            "is_valid": validate_step_logic(thought, action)
        }
        trace_steps.append(step)
    return {"steps": trace_steps, "total_length": len(trace_steps)}
```

### Step 2: Build the Critique Generator

Create a module that identifies reasoning flaws and suggests improvements.

```python
def generate_critique(trajectory, trace):
    """
    Identify reasoning flaws and provide refinement guidance.
    Returns: list of critiques with locations and suggestions
    """
    critiques = []

    for step in trace["steps"]:
        if not step["is_valid"]:
            critique = {
                "step_id": step["id"],
                "flaw_type": classify_reasoning_flaw(step),
                "description": describe_flaw(step),
                "suggestion": generate_refinement(step)
            }
            critiques.append(critique)

    return critiques
```

### Step 3: Implement Process Scoring

Score the overall reasoning process quality.

```python
def score_process(trajectory, trace, critiques):
    """
    Evaluate process performance using multiple signals:
    - Reasoning clarity and coherence
    - Tool use efficiency
    - Goal alignment
    """
    scores = {
        "clarity": measure_clarity(trace),
        "efficiency": measure_efficiency(trajectory),
        "alignment": measure_goal_alignment(trace),
        "flaw_count": len(critiques)
    }

    # Weighted combination: penalize flaws, reward efficiency
    overall_score = (
        0.4 * scores["clarity"] +
        0.3 * scores["efficiency"] +
        0.3 * scores["alignment"] -
        0.1 * min(len(critiques), 10)
    )

    return {"scores": scores, "overall": max(0, min(1, overall_score))}
```

### Step 4: Integration Strategies

Implement three training strategies using the reward signals:

**Reagent-C (Text-Augmented Refinement)**: Augment trajectories with textual critiques during training.

```python
def reagent_c_augment(trajectory, critiques):
    """Add critique text to trajectory for language-model-guided learning"""
    augmented = trajectory.copy()
    for critique in critiques:
        step_id = critique["step_id"]
        augmented[step_id]["refinement_hint"] = critique["suggestion"]
    return augmented
```

**Reagent-R (Reward-Augmented Guidance)**: Use process scores to shape reward signal.

```python
def reagent_r_reward(outcome_reward, process_score, weight=0.3):
    """Combine outcome and process rewards"""
    return (1 - weight) * outcome_reward + weight * process_score
```

**Reagent-U (Unified Integration)**: Combine all signals—trace, critique, and scores—in a joint training objective.

```python
def reagent_u_objective(trajectory, trace, critiques, process_score, outcome_reward):
    """Unified loss combining all reward signals"""
    trace_loss = 0.2 * evaluate_trace_quality(trace)
    critique_loss = 0.3 * len(critiques)
    score_loss = 0.3 * (1 - process_score)
    outcome_loss = 0.2 * (1 - outcome_reward)

    return trace_loss + critique_loss + score_loss + outcome_loss
```

## Key Insights

- **Process vs. Outcome**: Process-level rewards provide denser learning signals than sparse outcomes
- **Multiple Feedback Types**: Combining reasoning traces, critiques, and scores is more effective than single-signal rewards
- **Unified Integration Works Best**: Reagent-U (unified feedback) outperforms selective signal usage
- **Benchmark Results**: Achieves 43.7% on GAIA and 46.2% on WebWalkerQA with this approach

## References

- arXiv:2601.22154: Full paper with evaluation across 12 benchmarks
- Code and models released by authors for reproducibility
