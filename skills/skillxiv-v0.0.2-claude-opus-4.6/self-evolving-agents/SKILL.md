---
name: self-evolving-agents
title: "Learning on the Job: Experience-Driven Self-Evolving Agents (MUSE)"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.08002
keywords: [agent-learning, self-evolution, hierarchical-memory, autonomous-reflection, continuous-improvement]
description: "Enable agents to learn continuously from execution experience through hierarchical memory and autonomous reflection. Trigger: improve agent performance on long-horizon tasks by accumulating and applying experience."
---

# MUSE: Multi-Turn Self-Evolving Agents

## Core Concept

Traditional LLM agents are frozen after training and cannot learn from deployment experience. MUSE enables agents to continuously improve on-the-job by accumulating execution experience through three mechanisms: a hierarchical memory module that guides planning, autonomous reflection after each subtask, and continuous integration of converted experience back into the memory. This allows agents to improve performance as they solve more tasks, mimicking how humans learn through experience.

The key insight: Agents' own execution trajectories contain rich learning signals; structured reflection converts raw experience into generalizable knowledge.

## Architecture Overview

- **Hierarchical Memory Module**: Multi-level experience storage (task patterns, subtask solutions, failure modes)
- **Autonomous Reflection**: Structured analysis of each subtask outcome
- **Experience Integration**: Feedback loop converting experience into memory updates
- **Zero-Shot Transfer**: Knowledge from one task generalizes to new tasks
- **State-Agnostic Learning**: Improves across diverse problem domains

## Implementation Steps

### 1. Design Hierarchical Memory Structure

Organize agent experience at multiple abstraction levels.

```python
class HierarchicalMemoryModule:
    """
    Multi-level memory for storing and retrieving experience.
    """
    def __init__(self):
        # Hierarchy levels
        self.task_patterns = {}  # High-level task decompositions
        self.subtask_solutions = {}  # Solved subtasks with approaches
        self.failure_modes = {}  # Common failure patterns and recovery
        self.decision_heuristics = {}  # Learned decision rules

    def retrieve_task_pattern(self, task_description):
        """
        Find similar tasks and their decompositions.

        Returns:
            Task pattern (list of subtasks) or None if no match
        """
        task_embedding = encode_task(task_description)

        best_match = None
        best_similarity = 0

        for stored_task, pattern in self.task_patterns.items():
            stored_embedding = encode_task(stored_task)
            similarity = cosine_similarity(task_embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern

        if best_similarity > 0.7:  # Threshold for relevance
            return best_match
        return None

    def retrieve_subtask_solution(self, subtask, context=None):
        """
        Find approach for similar subtask.

        Returns:
            Solution approach or None
        """
        subtask_embedding = encode_subtask(subtask)

        # If context available, weight by relevance
        candidates = []

        for stored_subtask, solution in self.subtask_solutions.items():
            stored_embedding = encode_subtask(stored_subtask)
            similarity = cosine_similarity(subtask_embedding, stored_embedding)

            if similarity > 0.6:
                candidates.append((solution, similarity))

        if candidates:
            # Return best match
            best = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
            return best[0]

        return None

    def retrieve_failure_recovery(self, failure_mode):
        """
        Find recovery strategies for known failure modes.
        """
        return self.failure_modes.get(failure_mode)

    def add_task_pattern(self, task, subtask_decomposition):
        """Add new task pattern to memory."""
        self.task_patterns[task] = subtask_decomposition

    def add_subtask_solution(self, subtask, approach):
        """Add successful subtask solution."""
        self.subtask_solutions[subtask] = approach

    def add_failure_mode(self, failure_type, recovery_strategy):
        """Learn from failures."""
        self.failure_modes[failure_type] = recovery_strategy
```

### 2. Implement Autonomous Reflection

After each subtask, structured reflection converts experience into learning.

```python
class AutonomousReflectionEngine:
    """
    Reflect on task execution and extract learnings.
    """
    def __init__(self, model):
        self.model = model

    def reflect_on_subtask(self, subtask, approach_taken, result, success):
        """
        Generate reflection on subtask execution.

        Args:
            subtask: What was attempted
            approach_taken: How it was attempted
            result: What happened
            success: Boolean outcome

        Returns:
            Reflection dictionary with learnings
        """
        if success:
            # Successful execution: generalize approach
            reflection_prompt = (
                f"Subtask: {subtask}\n"
                f"Approach used: {approach_taken}\n"
                f"Result: {result}\n\n"
                f"Why did this approach work? "
                f"What patterns made it successful? "
                f"When would this approach work elsewhere?"
            )
        else:
            # Failed execution: identify failure mode
            reflection_prompt = (
                f"Subtask: {subtask}\n"
                f"Approach attempted: {approach_taken}\n"
                f"Result: Failed - {result}\n\n"
                f"What went wrong? "
                f"Why did this approach fail? "
                f"How to recover or adapt?"
            )

        reflection_text = self.model.generate(
            reflection_prompt,
            max_tokens=200,
            temperature=0.5
        )

        # Parse reflection into structured form
        reflection = {
            "subtask": subtask,
            "approach": approach_taken,
            "outcome": result,
            "success": success,
            "reflection_text": reflection_text,
            "learned_heuristics": self.extract_heuristics(reflection_text),
            "failure_modes": self.extract_failures(reflection_text) if not success else []
        }

        return reflection

    def extract_heuristics(self, reflection_text):
        """Extract generalizable decision rules from reflection."""
        # Parse reflection to find patterns
        # "When X, do Y" patterns
        heuristics = []

        if "works" in reflection_text.lower():
            # Extract what works
            parts = reflection_text.split("works")
            for part in parts:
                if len(part) > 20:
                    heuristics.append(part[:100].strip())

        return heuristics

    def extract_failures(self, reflection_text):
        """Extract failure modes and recovery strategies."""
        failures = []

        failure_keywords = ["failed", "wrong", "problem", "issue", "broke"]
        recovery_keywords = ["instead", "should", "try", "alternative"]

        for keyword in failure_keywords:
            if keyword in reflection_text.lower():
                idx = reflection_text.lower().index(keyword)
                failure_context = reflection_text[max(0, idx-50):min(len(reflection_text), idx+100)]
                failures.append(failure_context)

        return failures
```

### 3. Implement Experience Integration Loop

Continuously update memory with new learnings.

```python
class ExperienceIntegrationLoop:
    """
    Feed reflected experience back into memory.
    """
    def __init__(self, memory_module, reflection_engine):
        self.memory = memory_module
        self.reflection = reflection_engine

    def integrate_experience(self, task_trajectory):
        """
        Process complete task execution into memory updates.

        Args:
            task_trajectory: Record of task with subtask executions

        Returns:
            Updated memory
        """
        task = task_trajectory["task"]
        subtasks = task_trajectory["subtasks"]

        # Extract task decomposition pattern
        task_pattern = [st["name"] for st in subtasks]
        self.memory.add_task_pattern(task, task_pattern)

        # Reflect on each subtask
        for subtask_record in subtasks:
            subtask = subtask_record["name"]
            approach = subtask_record["approach"]
            result = subtask_record["result"]
            success = subtask_record["success"]

            # Generate reflection
            reflection = self.reflection.reflect_on_subtask(
                subtask,
                approach,
                result,
                success
            )

            # Update memory with learnings
            if success:
                # Store successful approach
                self.memory.add_subtask_solution(
                    subtask,
                    {
                        "approach": approach,
                        "heuristics": reflection["learned_heuristics"],
                        "success_rate": 1.0  # Update with frequency
                    }
                )
            else:
                # Store failure mode and recovery
                for failure_mode in reflection["failure_modes"]:
                    self.memory.add_failure_mode(
                        failure_mode,
                        recovery_strategy="Try alternative approach"
                    )

        return self.memory
```

### 4. Implement Agent with Memory-Guided Planning

Use hierarchical memory to improve decision-making.

```python
class MemoryGuidedAgent:
    """
    LLM agent enhanced with hierarchical memory.
    """
    def __init__(self, model, memory_module):
        self.model = model
        self.memory = memory_module

    def plan_task(self, task_description):
        """
        Plan task execution, informed by past experience.

        Returns:
            List of planned subtasks
        """
        # Try to retrieve similar task pattern
        task_pattern = self.memory.retrieve_task_pattern(task_description)

        if task_pattern:
            # Use learned pattern as starting point
            planning_prompt = (
                f"Task: {task_description}\n\n"
                f"Based on similar tasks, a good decomposition is:\n"
                f"{'; '.join(task_pattern)}\n\n"
                f"Adapt this plan specifically for this task: "
            )
        else:
            # Generate fresh plan
            planning_prompt = (
                f"Task: {task_description}\n\n"
                f"Break this task into subtasks: "
            )

        plan = self.model.generate(
            planning_prompt,
            max_tokens=200
        )

        return parse_subtasks(plan)

    def execute_subtask(self, subtask, context):
        """
        Execute subtask, using memory for guidance.

        Args:
            subtask: Subtask to execute
            context: Current task context

        Returns:
            (result, success)
        """
        # Try to retrieve similar subtask solution
        suggested_approach = self.memory.retrieve_subtask_solution(
            subtask,
            context=context
        )

        if suggested_approach:
            # Use suggested approach as prompt
            execution_prompt = (
                f"Task context: {context}\n"
                f"Subtask: {subtask}\n"
                f"Suggested approach: {suggested_approach['approach']}\n\n"
                f"Execute this subtask: "
            )
        else:
            # Generate fresh approach
            execution_prompt = (
                f"Subtask: {subtask}\n"
                f"Context: {context}\n\n"
                f"Execute this subtask: "
            )

        result = self.model.generate(
            execution_prompt,
            max_tokens=300
        )

        # Evaluate success
        success = evaluate_subtask_result(subtask, result, context)

        return result, success
```

### 5. Full Self-Evolving Loop

Orchestrate agent execution with continuous learning.

```python
def self_evolving_agent_loop(
    model,
    task_list,
    num_iterations=1
):
    """
    Run agent with continuous self-evolution.
    """
    # Initialize modules
    memory = HierarchicalMemoryModule()
    reflection_engine = AutonomousReflectionEngine(model)
    integration_loop = ExperienceIntegrationLoop(memory, reflection_engine)
    agent = MemoryGuidedAgent(model, memory)

    all_results = []

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}")

        for task_id, task in enumerate(task_list):
            print(f"  Task {task_id}: {task[:50]}...")

            # Plan execution
            subtask_plan = agent.plan_task(task)

            # Execute each subtask
            task_trajectory = {
                "task": task,
                "subtasks": []
            }

            for subtask_idx, subtask in enumerate(subtask_plan):
                # Execute subtask
                result, success = agent.execute_subtask(subtask, task)

                task_trajectory["subtasks"].append({
                    "name": subtask,
                    "approach": f"Step {subtask_idx + 1}",
                    "result": result,
                    "success": success
                })

                if not success:
                    # Try recovery strategy if available
                    recovery = memory.retrieve_failure_recovery("generic_failure")
                    if recovery:
                        print(f"    Recovery attempt for {subtask[:30]}...")

            # Integrate experience into memory
            memory = integration_loop.integrate_experience(task_trajectory)

            all_results.append(task_trajectory)

    return agent, memory, all_results
```

### 6. Evaluation: Learning Over Time

Measure improvement as agent accumulates experience.

```python
def evaluate_self_evolution(agent, memory, benchmark_tasks):
    """
    Assess improvement in performance across tasks.
    """
    results_over_time = []

    initial_accuracy = 0
    final_accuracy = 0

    for task_idx, task in enumerate(benchmark_tasks):
        # Execute task
        plan = agent.plan_task(task)

        success_count = 0
        for subtask in plan:
            result, success = agent.execute_subtask(subtask, task)
            if success:
                success_count += 1

        task_success_rate = success_count / len(plan)
        results_over_time.append(task_success_rate)

        if task_idx == 0:
            initial_accuracy = task_success_rate
        if task_idx == len(benchmark_tasks) - 1:
            final_accuracy = task_success_rate

        print(f"Task {task_idx}: {task_success_rate * 100:.1f}% success")

    improvement = (final_accuracy - initial_accuracy) * 100
    print(f"\nLearning improvement: +{improvement:.1f} percentage points")
    print(f"Memory size: {len(memory.task_patterns)} patterns, "
          f"{len(memory.subtask_solutions)} solutions")

    return {
        "accuracy_over_time": results_over_time,
        "improvement": improvement
    }
```

## Practical Guidance

**Hyperparameters:**
- **Task pattern similarity threshold**: 0.7 (for retrieval)
- **Subtask similarity threshold**: 0.6
- **Reflection temperature**: 0.5 (structured output)
- **Memory update frequency**: After each subtask
- **Experience integration batch**: Per-task level

**When to Use:**
- Long-horizon multi-task agents (e.g., productivity tools)
- Scenarios where agent serves same user repeatedly
- Want continuous improvement without retraining
- Memory generalizes across diverse problem types

**When NOT To Use:**
- Single-task agents or one-off interactions
- Strict privacy requirements (memory stores task details)
- Real-time constraints (memory retrieval adds latency)
- Domain where generalization uncertain (isolated tasks)

## Reference

[Learning on the Job: Experience-Driven Self-Evolving Agents](https://arxiv.org/abs/2510.08002) — arXiv:2510.08002
