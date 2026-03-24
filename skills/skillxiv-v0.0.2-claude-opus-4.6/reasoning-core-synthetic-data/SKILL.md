---
name: reasoning-core-synthetic-data
title: "Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre/Post-Training"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.02208"
keywords: [Synthetic Data, Symbolic Reasoning, Curriculum Learning, Data Generation, Verification]
description: "Reasoning Core procedurally generates verifiable symbolic reasoning datasets across formal domains (planning, logic, parsing), with external solvers and curriculum control."
---

# Technique: Procedural Symbolic Reasoning Dataset Generation

Training language models for reasoning requires large amounts of high-quality data. However, collecting real reasoning data is expensive and limited in scope. Reasoning Core addresses this by procedurally generating symbolic reasoning tasks across five core formal domains: PDDL planning, first-order logic, context-free grammars, causal reasoning, and equation solving.

The key innovation: each generated task includes (1) an external solver that produces verifiable answers, (2) explicit reasoning traces that can be used for supervised training, and (3) difficulty control for curriculum learning. This makes it possible to generate infinite high-quality reasoning training data.

## Core Concept

The core insight is that symbolic reasoning domains have well-defined semantics and external solvers. Rather than generating unverifiable reasoning data, generate problems and solutions using domain-specific solvers, then use them for LLM training.

**Key properties:**
- **Verifiable**: External solvers guarantee correctness
- **Scalable**: Generate unlimited data procedurally
- **Curriculum-enabled**: Difficulty control for progressive training
- **Trace-aware**: Solutions include reasoning steps, not just answers
- **Domain-diverse**: Five formal domains covering different reasoning types

## Architecture Overview

- **Domain-Specific Generators**: Procedural problem creation for each domain
- **External Solvers**: PDDL planner, SAT solver, CYK parser, Bayesian inference, linear algebra
- **Trace Extraction**: Capture solution reasoning paths
- **Difficulty Control**: Parametric adjustment of problem complexity
- **Reward Functions**: Verifiable correctness signals for RL

## Implementation Steps

Reasoning Core generates data by sampling problems and solving them. Here's how to implement it:

Implement a procedural problem generator for one domain (e.g., planning):

```python
import random
from typing import List, Dict, Any, Tuple

class PDDLDomainGenerator:
    """Generates PDDL planning problems with controllable difficulty."""

    def __init__(self, seed=42):
        random.seed(seed)
        self.difficulties = []

    def generate_problem(
        self,
        num_objects: int = 5,
        num_predicates: int = 10,
        plan_length: int = 5,
        difficulty: str = 'easy'
    ) -> Tuple[str, str]:
        """
        Generate a PDDL planning problem.
        Returns: (domain_str, problem_str) in PDDL format
        """
        # Generate objects (e.g., locations, blocks)
        objects = [f"obj{i}" for i in range(num_objects)]

        # Generate predicates based on difficulty
        if difficulty == 'easy':
            predicates = self._generate_simple_predicates(objects)
        elif difficulty == 'medium':
            predicates = self._generate_medium_predicates(objects)
        else:  # hard
            predicates = self._generate_complex_predicates(objects)

        # Generate goal state
        goal_state = self._generate_goal(objects, plan_length)

        # Format as PDDL
        domain = self._format_pddl_domain(predicates)
        problem = self._format_pddl_problem(objects, goal_state)

        return domain, problem

    def _generate_simple_predicates(self, objects: List[str]) -> List[Dict]:
        """Easy: single predicate relations."""
        predicates = []
        for i in range(0, len(objects) - 1, 2):
            predicates.append({
                'name': 'connected',
                'args': (objects[i], objects[i+1])
            })
        return predicates

    def _generate_medium_predicates(self, objects: List[str]) -> List[Dict]:
        """Medium: multiple predicates with chains."""
        predicates = []
        for i, obj in enumerate(objects):
            # Multi-step dependencies
            predicates.append({'name': 'at', 'args': (obj, 'start')})
            if i < len(objects) - 1:
                predicates.append({
                    'name': 'can_reach',
                    'args': (obj, objects[i+1])
                })
        return predicates

    def _generate_complex_predicates(self, objects: List[str]) -> List[Dict]:
        """Hard: nested dependencies and constraints."""
        predicates = []
        for i, obj in enumerate(objects):
            predicates.append({'name': 'at', 'args': (obj, 'start')})
            for j in range(i+1, min(i+3, len(objects))):
                predicates.append({
                    'name': 'constraint',
                    'args': (obj, objects[j], f'type_{random.randint(1,3)}')
                })
        return predicates

    def _generate_goal(self, objects: List[str], plan_length: int) -> List[Dict]:
        """Generate goal state."""
        goal = []
        for i in range(min(plan_length, len(objects))):
            goal.append({'name': 'at', 'args': (objects[i], f'goal_{i}')})
        return goal

    def _format_pddl_domain(self, predicates: List[Dict]) -> str:
        """Format as PDDL domain."""
        domain = """
(define (domain planning-domain)
  (:requirements :typing :action-costs)
  (:types location object)
  (:predicates
    (at ?obj - object ?loc - location)
    (connected ?l1 ?l2 - location)
    (can_reach ?o1 ?o2 - object)
  )
  (:action move
    :parameters (?obj - object ?from ?to - location)
    :precondition (and (at ?obj ?from) (connected ?from ?to))
    :effect (and (not (at ?obj ?from)) (at ?obj ?to))
  )
)
        """
        return domain.strip()

    def _format_pddl_problem(self, objects: List[str], goal: List[Dict]) -> str:
        """Format as PDDL problem."""
        objects_str = ' '.join(objects)
        goal_strs = [f"(at {g['args'][0]} {g['args'][1]})" for g in goal]
        goal_clause = ' '.join(goal_strs)

        problem = f"""
(define (problem planning-problem)
  (:domain planning-domain)
  (:objects {objects_str})
  (:init (at obj0 start))
  (:goal (and {goal_clause}))
)
        """
        return problem.strip()
```

Implement a wrapper that generates data with difficulty control:

```python
from dataclasses import dataclass

@dataclass
class ReasoningExample:
    """A reasoning task with solution and reasoning trace."""
    problem: str
    solution: str
    reasoning_trace: List[str]
    difficulty: str
    domain: str
    is_correct: bool

class ReasoningCoreGenerator:
    """Main interface for generating reasoning data."""

    def __init__(self):
        self.domains = {
            'planning': PDDLDomainGenerator(),
            # Add other domains: logic, grammar, causal, equations
        }
        self.generated_count = 0

    def generate_batch(
        self,
        domain: str,
        num_examples: int,
        difficulty: str = 'medium',
        seed: int = None,
    ) -> List[ReasoningExample]:
        """
        Generate a batch of reasoning examples.
        """
        if seed is not None:
            random.seed(seed)

        examples = []
        generator = self.domains.get(domain)
        if not generator:
            raise ValueError(f"Unknown domain: {domain}")

        for i in range(num_examples):
            # Generate problem
            problem_data = generator.generate_problem(difficulty=difficulty)

            # Solve using external solver
            solution, trace = self._solve_with_external_solver(
                domain, problem_data
            )

            # Create example
            example = ReasoningExample(
                problem=str(problem_data),
                solution=solution,
                reasoning_trace=trace,
                difficulty=difficulty,
                domain=domain,
                is_correct=len(trace) > 0  # Simplified check
            )
            examples.append(example)
            self.generated_count += 1

        return examples

    def _solve_with_external_solver(self, domain: str, problem) -> Tuple[str, List[str]]:
        """Call external solver and extract reasoning trace."""
        if domain == 'planning':
            # Call PDDL planner (e.g., FF, Metric-FF)
            solution, trace = self._solve_pddl(problem)
        elif domain == 'logic':
            # Call SAT/SMT solver
            solution, trace = self._solve_logic(problem)
        else:
            solution, trace = "", []

        return solution, trace

    def _solve_pddl(self, problem) -> Tuple[str, List[str]]:
        """Solve PDDL problem using external planner."""
        import subprocess
        # Example: use FF planner
        # domain_str, problem_str = problem
        # In practice, write to files and call: ff -o domain.pddl -f problem.pddl
        trace = ["move obj0 from start to goal_0"]  # Simplified
        solution = "move obj0 from start to goal_0"
        return solution, trace

    def _solve_logic(self, problem) -> Tuple[str, List[str]]:
        """Solve first-order logic problem."""
        # Placeholder for logic solving
        return "solution", []

    def create_training_dataset(
        self,
        domain: str,
        num_examples: int,
        split_ratios: Dict[str, float] = None,
    ) -> Dict[str, List[ReasoningExample]]:
        """
        Create full training dataset with curriculum progression.
        """
        if split_ratios is None:
            split_ratios = {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}

        dataset = {'train': [], 'val': [], 'test': []}

        # Generate examples with curriculum: easy -> medium -> hard
        for difficulty, ratio in split_ratios.items():
            examples = self.generate_batch(
                domain,
                int(num_examples * ratio),
                difficulty=difficulty
            )

            # Split examples
            n_train = int(len(examples) * 0.7)
            n_val = int(len(examples) * 0.15)

            dataset['train'].extend(examples[:n_train])
            dataset['val'].extend(examples[n_train:n_train+n_val])
            dataset['test'].extend(examples[n_train+n_val:])

        return dataset

    def to_supervised_training_format(
        self,
        example: ReasoningExample
    ) -> Dict[str, str]:
        """
        Convert reasoning example to supervised training format.
        """
        return {
            'input': f"Solve this {example.domain} problem:\n{example.problem}",
            'output': f"Step by step solution:\n" + "\n".join(example.reasoning_trace) +
                     f"\nFinal answer: {example.solution}",
            'metadata': {
                'domain': example.domain,
                'difficulty': example.difficulty,
                'is_correct': example.is_correct
            }
        }
```

## Practical Guidance

**When to Use:**
- Pre-training or post-training language models for reasoning
- When you need large-scale, diverse reasoning data
- For curriculum learning (start easy, progress to hard)
- When you want guaranteed correctness (external solver verification)

**When NOT to Use:**
- Open-ended generation tasks (not designed for these)
- Real-time data generation (generation happens offline)
- When symbolic reasoning doesn't apply to your domain

**Data Generation Strategy:**
- Start with 10K examples per domain, scale up based on model size
- Use curriculum: 30% easy, 50% medium, 20% hard
- Generate 5–10x more data than you need; filter by solver confidence
- Mix domains for diverse reasoning skill development

**Integration with Training:**
1. Generate offline dataset using Reasoning Core
2. Convert to supervised training format
3. Fine-tune model on mixed-domain data
4. Optionally use for RL reward signals

**Difficulty Control:**
- Easy: Single-step reasoning, simple preconditions
- Medium: Multi-step with dependencies
- Hard: Complex constraints, multiple interacting predicates

**Results:**
- Mixing Reasoning Core data into pre-training improves downstream reasoning
- Preserves or slightly improves general language modeling quality
- Particularly effective for mathematical and symbolic tasks

---

**Reference:** [Reasoning Core: A Scalable Procedural Data Generation Suite for Symbolic Pre/Post-Training](https://arxiv.org/abs/2603.02208)
