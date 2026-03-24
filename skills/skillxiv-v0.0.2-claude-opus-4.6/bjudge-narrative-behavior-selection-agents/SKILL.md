---
name: bjudge-narrative-behavior-selection-agents
title: "The Unreasonable Effectiveness of Scaling Agents for Computer Use"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.02250"
keywords: [agent-scaling, behavior-selection, computer-use, rollout-aggregation, reasoning]
description: "Improve computer-use agent performance by running multiple rollouts and selecting the best trajectory using narrative-level reasoning. The Behavior Judge (BJudge) converts raw execution traces into behavior narratives, enabling intelligent trajectory selection that scales agent effectiveness beyond single-rollout limitations."
---

# BJudge: Intelligent Behavior Selection for Computer-Use Agents

Single-rollout agent execution is inherently unreliable for complex desktop automation. Small errors compound over long interaction sequences—a misread button, a wrong form field—cascading into task failure. The naive solution is to run more rollouts and pick the one that succeeds. But how do you reliably identify the best trajectory when ground truth is ambiguous (did the agent reach the goal?) and trajectory lengths vary?

BJudge solves this with a narrative-level comparison approach. Instead of analyzing raw action sequences, it converts them into natural-language behavior summaries, then uses reasoning to compare and select the superior trajectory. This enables robust agent scaling with significant performance gains.

## Core Concept

BJudge operates in three stages for each attempt:

1. **Trajectory capture**: Record all agent actions, screenshots, and state changes
2. **Narrative generation**: Summarize the trajectory as a natural-language behavior description
3. **Comparative selection**: Compare narrative summaries to identify the best trajectory

The key insight is that **narrative comparison is more robust than action-sequence matching**. Two very different action sequences can achieve the same goal differently (e.g., find a button via search vs. scroll), but narratives capture the essence: "Agent successfully navigated to settings."

## Architecture Overview

- **Executor**: Runs agent rollouts, capturing full execution traces
- **Narrator**: Converts traces to narrative summaries using vision-language models
- **Comparator**: Ranks narratives to identify best trajectory
- **Validator**: Verifies selected trajectory achieves task goal
- **Dispatcher**: Routes best trajectory for downstream use or error recovery

## Implementation Steps

First, set up trajectory capture infrastructure. Record all relevant signals during execution:

```python
from computer_use_agent import Trajectory, TrajectoryRecorder

class RolloutTrajectory:
    """
    Capture full execution trace for later analysis.
    """
    def __init__(self, task_description):
        self.task = task_description
        self.actions = []  # List of (action_type, action_param, screenshot_before)
        self.screenshots = []
        self.action_types = []
        self.timestamps = []

    def record_action(self, action_type, action_param, screenshot_before, timestamp):
        """Log a single agent action."""
        self.actions.append((action_type, action_param))
        self.screenshots.append(screenshot_before)
        self.action_types.append(action_type)
        self.timestamps.append(timestamp)

    def add_final_screenshot(self, screenshot_final):
        """Add final state after all actions."""
        self.final_screenshot = screenshot_final

    def get_summary_stats(self):
        """Quick stats about trajectory."""
        return {
            "num_actions": len(self.actions),
            "action_types": set(self.action_types),
            "duration_seconds": self.timestamps[-1] - self.timestamps[0]
        }
```

Next, generate narrative summaries of trajectories using a vision-language model:

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

class NarrativeGenerator:
    """
    Convert trajectory to natural-language behavior description.
    """
    def __init__(self, model_name="llava-1.5-13b-hf"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name)

    def generate_narrative(self, trajectory):
        """
        Summarize trajectory as behavior narrative.

        Args:
            trajectory: RolloutTrajectory object

        Returns:
            narrative: Natural-language summary of trajectory
        """
        # Sample key frames from trajectory
        key_frames = self._extract_key_frames(trajectory)

        # Build prompt describing the task and asking for summary
        prompt = f"""You are analyzing a desktop automation agent's execution.

Task: {trajectory.task}

The agent performed these actions in order: {trajectory.action_types}

Key frames of the execution:
"""

        for i, (frame, action) in enumerate(zip(key_frames, trajectory.actions)):
            action_type, action_param = action
            prompt += f"\n{i+1}. After action '{action_type} {action_param}'"

        prompt += """

Summarize what the agent accomplished in this trajectory:
- Did it successfully complete the task?
- What were the key steps?
- Were there any errors or detours?
- Rate the quality of the execution (excellent/good/fair/poor)

Provide a 2-3 sentence behavior summary."""

        # Generate narrative with vision context
        inputs = self.processor(prompt, key_frames, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        narrative = self.processor.decode(outputs[0], skip_special_tokens=True)

        return narrative

    def _extract_key_frames(self, trajectory):
        """Sample 3-5 important frames from trajectory."""
        if len(trajectory.screenshots) <= 5:
            return trajectory.screenshots

        # Sample first, middle, last, plus any outliers
        indices = [
            0,
            len(trajectory.screenshots) // 3,
            2 * len(trajectory.screenshots) // 3,
            len(trajectory.screenshots) - 1
        ]
        return [trajectory.screenshots[i] for i in sorted(set(indices))]
```

Now implement the comparator that selects the best trajectory:

```python
class BehaviorComparator:
    """
    Rank trajectories by comparing their narratives.
    """
    def __init__(self, model_name="gpt-4-vision"):
        self.model = model_name
        self.scorer = Ranker()

    def compare_narratives(self, narratives, task_description):
        """
        Rank narratives to find best trajectory.

        Args:
            narratives: List of behavior narratives
            task_description: Original task

        Returns:
            ranked_indices: Indices sorted by quality (best first)
            scores: Quality scores for each narrative
        """
        prompt = f"""You are evaluating agent execution narratives.

Task: {task_description}

Compare these behavior summaries and rank them by quality:
"""

        for i, narrative in enumerate(narratives):
            prompt += f"\n[Trajectory {i+1}]:\n{narrative}\n"

        prompt += """
Ranking criteria:
1. Did it accomplish the task goal?
2. Efficiency: minimum unnecessary actions?
3. Correctness: no hallucinations or false claims?
4. Robustness: would it work reliably?

Return a ranking from best to worst (e.g., "2, 1, 3") and brief reasons."""

        # Get ranking from model
        response = llm_call(self.model, prompt)
        ranking_text = response.split('\n')[0]  # First line usually has ranking

        # Parse ranking
        try:
            indices = [int(x.strip()) - 1 for x in ranking_text.split(',')]
        except:
            # Fallback: score all narratives
            indices = self.scorer.rank_by_likelihood(narratives, task_description)

        return indices
```

Finally, implement the multi-rollout execution loop with intelligent selection:

```python
def run_agent_with_behavior_selection(agent, task, num_rollouts=4):
    """
    Execute agent multiple times, selecting best via narrative comparison.

    Args:
        agent: Computer-use agent
        task: Task description
        num_rollouts: Number of attempts to try

    Returns:
        best_trajectory: Selected trajectory
        all_narratives: Summaries for analysis
    """
    executor = Executor(agent)
    narrator = NarrativeGenerator()
    comparator = BehaviorComparator()

    trajectories = []
    narratives = []

    print(f"Running {num_rollouts} rollouts...")

    # Execute multiple times
    for i in range(num_rollouts):
        print(f"Rollout {i+1}/{num_rollouts}")
        trajectory = executor.run(task)
        trajectories.append(trajectory)

        # Generate narrative for this trajectory
        narrative = narrator.generate_narrative(trajectory)
        narratives.append(narrative)
        print(f"  Actions: {trajectory.get_summary_stats()['num_actions']}")
        print(f"  Narrative: {narrative[:80]}...")

    # Compare and select best
    print("\nComparing trajectories...")
    ranked_indices = comparator.compare_narratives(narratives, task)

    best_idx = ranked_indices[0]
    best_trajectory = trajectories[best_idx]

    print(f"\nSelected trajectory {best_idx+1} as best")
    print(f"Narrative: {narratives[best_idx]}")

    return best_trajectory, narratives
```

## Practical Guidance

**When to use multi-rollout selection:**
- Long-horizon desktop automation (5+ steps)
- Tasks with environment variability (web pages differ by region/time)
- Critical tasks where single failure is unacceptable
- Unknown environments where agent reliability is unproven

**When NOT to use:**
- Real-time interactions (too slow for multiple rollouts)
- Simple 1-2 step tasks (overhead exceeds benefit)
- Deterministic environments with single optimal path
- Resource-constrained settings (mobile, embedded)

**Performance vs. cost tradeoff:**

| Num Rollouts | Time Overhead | OSWorld Success Gain | Best For |
|--------------|---------------|---------------------|----------|
| 1 | 1x | Baseline (≈60%) | Development |
| 2 | 2x | +4-6% | Production |
| 4 | 4x | +8-12% | High-stakes |
| 8 | 8x | +12-15% (diminishing) | Thorough verification |

**Narrative quality matters more than trajectory length:** A concise, accurate narrative beats verbose but vague summaries. Fine-tune your narrator on domain tasks if possible.

**Common pitfalls:**
- **Narratives too similar**: If all narratives say "agent tried to complete task," comparator can't distinguish. Ensure narrator describes *how* the agent approached the task, not just intent.
- **Ground truth ambiguity**: Some tasks have multiple valid solutions (e.g., "find the settings button"). Use task-specific validators or human review to break ties.
- **Narrator hallucination**: Vision models sometimes confess actions the agent never took. Validate narrative against actual action log before comparison.
- **Ignoring efficiency**: Selecting by success alone may pick a trajectory with 100 actions when 10 would do. Include efficiency metrics in comparison.

**Integration checklist:**
- [ ] Instrument agent to capture full execution traces (screenshots + action logs)
- [ ] Test narrator on 20 sample trajectories; validate narratives are accurate
- [ ] Try 2-rollout baseline to measure overhead and success gain on your tasks
- [ ] Implement task-specific validators to provide ground truth for comparison
- [ ] Set num_rollouts based on time budget (target <10s total including narrator)
- [ ] Monitor narrative length and diversity; address if all narratives are identical

Reference: https://arxiv.org/abs/2510.02250
