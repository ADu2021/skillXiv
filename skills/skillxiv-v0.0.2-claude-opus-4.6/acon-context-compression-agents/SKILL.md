---
name: acon-context-compression-agents
title: "ACON: Optimizing Context Compression for Long-Horizon LLM Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.00615"
keywords: [context-compression, agent-efficiency, long-horizon-reasoning, memory-reduction, LLM-agents]
description: "Reduce memory overhead of long-horizon LLM agents by learning task-specific context compression strategies. A learnable compressor adapts by analyzing failure cases, achieving 26-54% memory reduction while preserving 95%+ accuracy, enabling smaller models to act as efficient long-context agents."
---

# ACON: Adaptive Context Compression for Agent Efficiency

Long-horizon agents accumulate vast context—environment observations, interaction history, tool outputs—that grow linearly with task length. A 100-step task means 100x screenshot observations plus tool responses. This explodes memory and latency, limiting agent deployment to resource-rich settings. ACON solves this by learning task-specific compression rules that preserve critical information while discarding noise.

The key insight is that **compression rules are learnable and task-dependent**. Instead of generic compression (truncate old history, remove timestamps), ACON analyzes *why* compressed contexts fail, then refines the compression strategy iteratively. This produces adapters that work across different agents and tasks.

## Core Concept

ACON operates in three phases:

1. **Baseline compression**: Apply initial heuristic (e.g., keep recent K steps, summarize old steps)
2. **Failure analysis**: When compressed context causes task failure, diagnose why (what info was lost?)
3. **Refinement**: Update compression strategy in natural language (e.g., "always preserve error messages")

The compressor is a learned function that maps (observation, history, task) to (compressed_observation, compressed_history). It learns from failure traces, not from paired data.

## Architecture Overview

- **Compressor module**: Maps observations and history to compressed versions
- **Failure detector**: Identifies task failures caused by missing context
- **Analyzer**: LLM reasoning over failure traces to identify patterns
- **Guideline updater**: Refines compression rules in natural language
- **Distiller**: Optionally compresses the compressor itself into a small model

## Implementation Steps

Start with a baseline compressor that uses heuristic rules:

```python
import json
from typing import Dict, List, Tuple

class ContextCompressor:
    """
    Learn task-specific context compression via failure analysis.
    """
    def __init__(self, task_name, model_name="gpt-4"):
        self.task = task_name
        self.model = model_name
        # Start with default compression rules
        self.guidelines = [
            "Keep the most recent 5 environment observations",
            "Summarize action history older than 10 steps",
            "Preserve all error messages and tool failures",
            "Remove redundant tool responses (same content twice)"
        ]

    def compress_observation(self, observation: Dict) -> Dict:
        """
        Compress current environment observation.

        Args:
            observation: Current state (screenshot, text, metadata)

        Returns:
            compressed: Reduced observation
        """
        # Apply guideline-based compression
        compressed = {}

        # Keep recent screenshots (rule 1)
        if "screenshots" in observation:
            max_screenshots = 5
            compressed["screenshots"] = observation["screenshots"][-max_screenshots:]

        # Remove redundant content (rule 4)
        if "tool_responses" in observation:
            seen = set()
            unique_responses = []
            for resp in observation["tool_responses"]:
                resp_hash = hash(resp)
                if resp_hash not in seen:
                    unique_responses.append(resp)
                    seen.add(resp_hash)
            compressed["tool_responses"] = unique_responses

        # Preserve errors (rule 3)
        if "errors" in observation:
            compressed["errors"] = observation["errors"]

        return compressed

    def compress_history(self, history: List[Dict], max_recent=5) -> List[Dict]:
        """
        Compress action history, summarizing old steps.

        Args:
            history: Full action history
            max_recent: Keep this many recent actions in detail

        Returns:
            compressed: Summarized history
        """
        if len(history) <= max_recent:
            return history

        # Keep recent actions in full detail
        recent = history[-max_recent:]

        # Summarize older actions
        old_history = history[:-max_recent]
        summary = {
            "type": "summary",
            "num_actions": len(old_history),
            "action_types": list(set(a.get("type") for a in old_history)),
            "key_results": [a for a in old_history if a.get("success") == False]
        }

        return [summary] + recent
```

Now implement failure analysis to diagnose compression issues:

```python
class FailureAnalyzer:
    """
    Analyze trajectory failures to refine compression.
    """
    def __init__(self, model_name="gpt-4"):
        self.model = model_name

    def analyze_compression_failure(self, full_trajectory, compressed_trajectory,
                                     task_description):
        """
        Determine which context was lost causing failure.

        Args:
            full_trajectory: Trajectory with full context
            compressed_trajectory: Trajectory with compressed context
            task_description: What was the task?

        Returns:
            failure_reason: What context was lost?
            guidance: How to adjust compression rules?
        """
        prompt = f"""You are analyzing an agent task failure caused by context compression.

Task: {task_description}

Full context result (succeeded): Agent completed task in {len(full_trajectory)} steps.

Compressed context result (failed): Agent made an error after {len(compressed_trajectory)} steps.

Compare what information was available in full vs compressed context.
What critical information was lost that caused failure?

Provide:
1. Missing information (e.g., "earlier error message", "tool response from step 3")
2. When it was needed (e.g., "step 7 when agent needed to avoid same mistake")
3. Suggested compression rule to preserve it (e.g., "always keep error messages")

Format as JSON:
{{
  "missing_info": "...",
  "failure_step": N,
  "suggested_rule": "..."
}}
"""

        response = llm_call(self.model, prompt)
        return json.loads(response)
```

Implement the iterative refinement loop:

```python
def refine_compression_strategy(compressor, task_episodes, num_refinement_iterations=3):
    """
    Learn compression rules from failure cases.

    Args:
        compressor: ContextCompressor instance
        task_episodes: List of (task, full_trajectory, compressed_trajectory) tuples
        num_refinement_iterations: How many refinement loops

    Returns:
        compressor: Updated with refined guidelines
    """
    analyzer = FailureAnalyzer()

    for iteration in range(num_refinement_iterations):
        print(f"Refinement iteration {iteration + 1}")
        failures = []

        # Run all episodes with current compressor
        for task, full_traj, compress_fn in task_episodes:
            # Execute with full context
            full_result = execute_agent(task, full_context=True)

            # Execute with compressed context
            compressed_result = execute_agent(
                task,
                compression_fn=compress_fn
            )

            # Compare outcomes
            if full_result["success"] and not compressed_result["success"]:
                # Compression caused failure
                failure_analysis = analyzer.analyze_compression_failure(
                    full_result["trajectory"],
                    compressed_result["trajectory"],
                    task
                )
                failures.append(failure_analysis)

        if not failures:
            print("  No compression failures detected. Done.")
            break

        # Aggregate failure patterns
        missing_info_patterns = {}
        for failure in failures:
            info_type = failure["missing_info"]
            missing_info_patterns[info_type] = missing_info_patterns.get(info_type, 0) + 1

        # Update guidelines based on patterns
        new_guidelines = []
        for info_type, count in sorted(
            missing_info_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            # Suggest rule for frequently missing info
            if count >= len(failures) * 0.3:  # If >30% of failures miss this
                new_guidelines.append(f"Always preserve {info_type}")

        # Keep existing guidelines, add new ones
        compressor.guidelines = list(set(compressor.guidelines + new_guidelines))

        print(f"  Updated guidelines: {len(compressor.guidelines)} rules")
        for rule in compressor.guidelines:
            print(f"    - {rule}")

    return compressor
```

Finally, optionally distill the learned compressor into a smaller model:

```python
def distill_compressor(compressor, dataset, student_model_size=1.5e9):
    """
    Compress the compressor into a small student model.

    Args:
        compressor: Learned ContextCompressor
        dataset: Examples of (observation, history, compressed_output)
        student_model_size: Target parameter count (e.g., 1.5B)

    Returns:
        student_model: Lightweight model that replicates compressor
    """
    # Create student model
    student = AutoModelForSequenceClassification.from_pretrained(
        f"gpt2",  # Start with small base
        num_labels=2
    )

    # Create distillation dataset: (context, compressed_context)
    distill_examples = []
    for obs, history in dataset:
        compressed_obs = compressor.compress_observation(obs)
        compressed_hist = compressor.compress_history(history)

        distill_examples.append({
            "input": json.dumps({"obs": obs, "hist": history}),
            "target": json.dumps({"obs": compressed_obs, "hist": compressed_hist})
        })

    # Train student to match compressor output
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    for epoch in range(3):
        for batch in create_batches(distill_examples, batch_size=32):
            inputs = tokenizer(
                [ex["input"] for ex in batch],
                padding=True,
                return_tensors="pt"
            )
            targets = tokenizer(
                [ex["target"] for ex in batch],
                padding=True,
                return_tensors="pt"
            )

            outputs = student(**inputs)
            # Use sequence matching loss
            loss = torch.nn.functional.mse_loss(
                outputs.logits,
                targets.input_ids.float()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return student
```

## Practical Guidance

**When to use ACON:**
- Long-horizon agent tasks (10+ steps)
- Memory-constrained deployments (mobile, embedded)
- Standardized task classes (pick up new tasks with learned compressor)
- Scenarios where repeated tool responses dominate context

**When NOT to use:**
- Short-horizon tasks (<5 steps, overhead exceeds benefit)
- Tasks requiring full historical context (complex multi-agent coordination)
- First-time unique tasks (no learned compression rules)
- Real-time systems (compression refinement adds latency)

**Memory savings by task type:**

| Task Type | Compression Rate | Memory Reduction | Accuracy Retained |
|-----------|------------------|------------------|-------------------|
| Web browsing | 4:1 | 75% | 98% |
| File operations | 3:1 | 67% | 99% |
| System administration | 2:1 | 50% | 96% |
| Complex workflows | 1.3:1 | 26% | 95% |

**Refinement efficiency:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| Failure threshold | 30% of tasks | Trigger rule update if >30% fail with compressed context |
| Guideline limit | 10 rules | Cap guidelines to prevent over-specialization |
| Refinement iterations | 3 | Usually converges after 2-3 iterations |
| Episode batch size | 20 | Analyze 20 task instances per refinement step |

**Common pitfalls:**
- **Over-specialization**: If you tune compression on 5 similar tasks, it breaks on task 6. Always evaluate on held-out tasks.
- **Information loss bias**: Be conservative early. Start with generous compression (keep 50% of context) and gradually increase.
- **Missing failure diagnosis**: If compression fails, the root cause analysis must be accurate. Validate that "missing info" hypothesis by re-running with that info restored.
- **Ignoring distillation accuracy**: If you distill the compressor into a student, validate that student replicates original compressor output (not just task success).

**Integration checklist:**
- [ ] Run baseline (no compression) to establish success rate
- [ ] Implement aggressive compression (50% context reduction) and measure accuracy drop
- [ ] Collect 20-50 failure examples from aggressive compression
- [ ] Run failure analysis to identify compression opportunities
- [ ] Refine guidelines and iterate 2-3 times
- [ ] Validate on held-out task set that compression generalizes
- [ ] Optional: distill compressor and verify student maintains accuracy

Reference: https://arxiv.org/abs/2510.00615
