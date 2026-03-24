---
name: recode-unified-planning
title: "ReCode: Unify Plan and Action for Universal Granularity Control"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.23564"
keywords: [Planning, Agent, Hierarchical Control, Code Generation, Abstraction]
description: "Unifies planning and action by treating plans as abstract placeholder functions recursively decomposed to primitive actions. Enables agents to dynamically adjust abstraction levels per task without rigid hierarchies. Improves inference performance and training efficiency through automatic multi-level data generation."
---

# ReCode: Unified Recursive Action Planning

Conventional agents separate high-level planning from low-level execution, forcing rigid abstraction boundaries. ReCode dissolves this separation using unified code representation, enabling agents to fluidly adjust granularity from abstract goals to concrete actions.

By representing both plans and actions as code, the system recursively decomposes abstractions, generating implicit hierarchical training data.

## Core Concept

Key insight: **represent planning and action in the same code language**, treating plans as abstract functions that recursively decompose:
- High-level plan: abstract placeholder functions
- Recursive decomposition: progressively substitute functions with implementations
- Unified representation: both planning and action in code syntax
- Contextual granularity: model chooses abstraction level per problem

## Architecture Overview

- Code-based representation for plans and actions
- Recursive function decomposition tree
- Dynamic abstraction level selection during inference
- Implicit hierarchical training through decomposition process

## Implementation Steps

Design a code representation that allows both abstract plans and concrete implementations. Use function signatures to represent abstraction levels:

```python
class CodePlanAction:
    def __init__(self):
        self.code_template = """
def solve_task(task_desc):
    # Abstract step 1: decompose into subgoals
    subgoal_1 = abstract_function_1(task_desc)
    subgoal_2 = abstract_function_2(task_desc)

    # Execute subgoals with varying granularity
    result_1 = execute(subgoal_1)
    result_2 = execute(subgoal_2)

    return combine_results(result_1, result_2)
"""

    def generate_plan_action_code(self, task, granularity='medium'):
        """Generate code that unifies planning and action."""
        if granularity == 'high':
            # Abstract level: broad functions
            code = f"""
def solve(task):
    plan = generate_abstract_plan('{task}')
    return execute_plan(plan)
"""
        elif granularity == 'medium':
            # Decomposed level: intermediate functions
            code = f"""
def solve(task):
    steps = decompose_into_steps('{task}')
    results = []
    for step in steps:
        results.append(execute_step(step))
    return aggregate_results(results)
"""
        else:  # low/detailed
            # Concrete level: primitive operations
            code = f"""
def solve(task):
    {self._generate_primitives(task)}
    return final_result
"""
        return code

    def _generate_primitives(self, task):
        """Generate primitive-level implementations."""
        return "# Primitive operations: for loop, condition, etc."
```

Implement recursive decomposition that transforms abstract functions into concrete ones:

```python
class RecursiveDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def decompose_function(self, function_code, current_depth=0, max_depth=3):
        """Recursively decompose abstract functions to primitives."""
        if current_depth >= max_depth:
            # Reached maximum depth: return concrete implementation
            return self._generate_primitive(function_code)

        # Decompose: replace abstract function with sub-functions
        prompt = f"""
Given this abstract function:
{function_code}

Generate 2-3 more concrete sub-functions that implement it.
Format each as a Python function definition.
"""

        sub_functions = self.llm.generate(prompt)

        # Recursively decompose sub-functions
        decomposed = []
        for sub_func in sub_functions:
            decomposed.append(
                self.decompose_function(sub_func, current_depth + 1, max_depth)
            )

        return {
            'level': current_depth,
            'sub_functions': decomposed
        }

    def _generate_primitive(self, function_code):
        """Generate primitive-level operations."""
        prompt = f"""
Implement this function using only primitive operations:
{function_code}
"""
        return self.llm.generate(prompt)
```

Implement training that uses the decomposition tree to create hierarchical training data:

```python
def train_unified_planner(model, tasks, decomposer):
    """Train model on unified planning/action representation."""
    training_data = []

    for task in tasks:
        # Generate code representation
        code = decomposer.generate_plan_action_code(task, granularity='high')

        # Recursively decompose to create multiple training examples
        decomposition_tree = decomposer.decompose_function(code)

        # Extract training examples at each level
        for level, node in enumerate(decomposition_tree):
            training_data.append({
                'task': task,
                'level': level,
                'code': node,
                'label': 'valid_decomposition'
            })

    # Train model to predict next level given current level
    for epoch in range(num_epochs):
        for sample in training_data:
            code_pred = model.generate(sample['code'])
            loss = compute_decomposition_loss(code_pred, sample)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Max decomposition depth | 3-4 (prevents over-abstraction) |
| Granularity levels | High, Medium, Low (three tiers) |
| Sub-functions per decomposition | 2-3 (balance complexity) |
| Training examples per task | 1 high-level + 3-5 decomposed |

**When to use:**
- Long-horizon tasks with variable complexity
- Scenarios where abstraction level should adapt per input
- Multi-step reasoning tasks
- Agent training where hierarchical data helps

**When NOT to use:**
- Single-step or simple tasks (over-engineered)
- Fixed abstraction requirements
- Systems with clear task boundaries (simpler hierarchies work)

**Common pitfalls:**
- Decomposition depth too large (explosion of possibilities)
- Over-abstracting concrete operations (lost execution fidelity)
- Not validating decomposed code (generates invalid functions)
- Imbalanced training data across levels (bias toward high-level)

Reference: [ReCode on arXiv](https://arxiv.org/abs/2510.23564)
