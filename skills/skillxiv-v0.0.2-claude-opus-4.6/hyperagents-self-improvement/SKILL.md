---
name: hyperagents-self-improvement
title: "Hyperagents: Open-Ended AI Systems via Recursive Self-Modification"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.19461"
keywords: [Self-Improvement, Meta-Learning, Program Synthesis, Open-Ended Learning]
description: "Enable AI systems to recursively improve themselves by making the meta-level modification procedure itself editable, achieving open-ended capability growth."
---

# Hyperagents: Recursive Self-Improvement Through Editable Meta-Procedures

Most self-improving AI systems hit a fundamental limit: they have a fixed meta-level improvement mechanism. A reinforcement learning agent can improve its policy, but its reward function is static. A program synthesis agent can generate code, but the synthesis procedure itself never changes.

Hyperagents solve this through a simple but profound insight: make everything—including the improvement mechanism itself—subject to modification. This creates a recursive structure where the system improves not just its task performance, but also how it searches for improvements. The result is open-ended capability growth: each iteration improves the system's ability to improve further.

## Core Concept

Hyperagents implement a Darwin-Gödel Machine (DGM) with full editability:

**Task Agent:** Solves the target problem, can be modified.

**Meta Agent:** Modifies itself and the task agent, can also be modified.

**Key Innovation:** The meta-modification procedure is itself editable. Rather than having humans design the self-improvement algorithm, the system evolves its own improvement strategies.

The system generates variants of itself (including different meta-level procedures), evaluates which variant performs best, and keeps the improvement. This creates a recursive loop where the system literally improves its own source code.

## Architecture Overview

- **Unified Program Representation**: Task and meta agents stored as editable programs
- **Self-Variant Generation**: Systematically mutate and recombine program code
- **Evaluation Framework**: Test each variant to measure performance improvement
- **Persistent Memory**: Store successful modifications for reuse across runs
- **Metacognitive Loop**: Meta-improvements compound across iterations
- **Domain-Agnostic**: Works across any domain with computable task and evaluation

## Implementation Steps

### Step 1: Represent Agents as Editable Programs

Store both task and meta logic as modifiable code.

```python
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Any
import hashlib
import copy

@dataclass
class AgentProgram:
    """
    Editable representation of an agent.
    Can be task logic or meta-improvement logic.
    """
    code: str  # Python code defining the agent
    metadata: Dict[str, Any]  # Task params, performance history
    version: int  # Iteration when created
    performance: float = 0.0  # Cached performance score
    creation_time: Optional[str] = None

    def compute_hash(self) -> str:
        """Unique identifier for this program."""
        return hashlib.sha256(self.code.encode()).hexdigest()[:8]

    def execute(self, task_input: Any, env_context: Dict = None) -> Any:
        """Run this agent program on a task."""
        # Create execution namespace
        namespace = {'task_input': task_input}
        if env_context:
            namespace.update(env_context)

        try:
            exec(self.code, namespace)
            return namespace.get('output', None)
        except Exception as e:
            return {'error': str(e), 'output': None}

    def clone(self, mutation: Optional[str] = None) -> 'AgentProgram':
        """Create a variant (optionally with code mutation)."""
        new_code = mutation if mutation else self.code
        new_program = AgentProgram(
            code=new_code,
            metadata=copy.deepcopy(self.metadata),
            version=self.version + 1,
            performance=0.0
        )
        return new_program

class HyperagentSystem:
    """
    System that recursively improves via self-modification.
    """

    def __init__(self, initial_task_agent: AgentProgram,
                 initial_meta_agent: AgentProgram):
        self.task_agent = initial_task_agent
        self.meta_agent = initial_meta_agent
        self.improvement_history = []  # Track successful modifications
        self.persistent_modifications = {}  # Reusable code patterns

    def create_initial_agents(self, task_description: str) - None:
        """Initialize task and meta agents for a domain."""

        # Task agent: solves the target task
        task_code = f"""
# Task Agent for: {task_description}
def task_agent(task_input):
    # Initial strategy: simple heuristic
    output = solve_task(task_input)
    return output

def solve_task(x):
    # Basic implementation (will be improved)
    return x
"""

        # Meta agent: improves the task agent
        meta_code = """
# Meta Agent: Improves task agent
def meta_agent(task_agent_code, evaluation_results):
    # Strategy 1: Add logging for debugging
    # Strategy 2: Try parameter variations
    # Strategy 3: Apply remembered successful modifications

    if 'high_variance' in evaluation_results:
        modified_code = add_parameter_sweep(task_agent_code)
    elif 'low_performance' in evaluation_results:
        modified_code = apply_heuristic_refinement(task_agent_code)
    else:
        modified_code = apply_random_local_search(task_agent_code)

    return modified_code

def add_parameter_sweep(code):
    # Insert parameter tuning
    return code.replace(
        'return x',
        'return x * 1.1'  # Simple modification
    )

def apply_heuristic_refinement(code):
    # Add domain-specific logic
    return code.replace(
        'return x',
        'return x if x > 0 else -x'
    )

def apply_random_local_search(code):
    # Small syntactic variations
    return code
"""

        self.task_agent = AgentProgram(
            code=task_code,
            metadata={'task': task_description},
            version=0
        )

        self.meta_agent = AgentProgram(
            code=meta_code,
            metadata={'role': 'meta_agent'},
            version=0
        )
```

### Step 2: Generate Self-Variants

Create mutations of task and meta agents.

```python
import random
import re

class VariantGenerator:
    """Generate program variants through mutation."""

    def __init__(self):
        self.mutation_operators = [
            self.add_parameter,
            self.refactor_logic,
            self.add_early_exit,
            self.add_memoization,
            self.modify_constants
        ]

    def generate_variants(self, program: AgentProgram,
                         num_variants: int = 5) -> List[AgentProgram]:
        """Create multiple variants of a program."""
        variants = []

        for _ in range(num_variants):
            mutation_fn = random.choice(self.mutation_operators)
            mutated_code = mutation_fn(program.code)
            variant = program.clone(mutation=mutated_code)
            variants.append(variant)

        return variants

    def add_parameter(self, code: str) -> str:
        """Introduce tunable parameters."""
        # Find return statements and add a parameter multiplier
        modified = re.sub(
            r'return ([^;\n]+)',
            r'return param * (\1)',
            code
        )
        # Initialize parameter
        modified = 'param = 1.0\n' + modified
        return modified

    def refactor_logic(self, code: str) -> str:
        """Simplify or restructure logic."""
        # Try to identify and refactor loops/conditionals
        if 'while' in code or 'for' in code:
            # Suggest vectorization
            modified = code.replace('for ', '# (refactored) for ')
            return modified
        return code

    def add_early_exit(self, code: str) -> str:
        """Add early termination conditions."""
        modified = code.replace(
            'def solve_task(',
            'def solve_task(\n    # Early exit on impossible inputs\n'
        )
        modified = re.sub(
            r'return ([^;\n]+)',
            r'if check_feasibility(): return \1\nelse: return None',
            modified
        )
        return modified

    def add_memoization(self, code: str) -> str:
        """Cache computation results."""
        modified = 'cache = {}\n' + code
        modified = modified.replace(
            'def solve_task(',
            'def solve_task(\n    if task_input in cache: return cache[task_input]\n'
        )
        modified = modified.replace(
            'return ',
            'cache[task_input] = result; return result\n    result = '
        )
        return modified

    def modify_constants(self, code: str) -> str:
        """Adjust numerical constants."""
        def adjust_number(match):
            num = float(match.group())
            # Random ±10% variation
            adjusted = num * (0.9 + 0.2 * random.random())
            return str(adjusted)

        modified = re.sub(r'\d+\.?\d*', adjust_number, code)
        return modified
```

### Step 3: Evaluate Variants

Test variants and identify improvements.

```python
class PerformanceEvaluator:
    """Measure agent performance on tasks."""

    def __init__(self, test_tasks: List[Dict], success_metric: Callable):
        self.test_tasks = test_tasks
        self.success_metric = success_metric

    def evaluate_agent(self, agent: AgentProgram,
                      num_evals: int = 10) -> Dict[str, float]:
        """
        Run agent on test tasks and measure performance.
        Returns: {accuracy, speed, stability, etc.}
        """

        results = {
            'success_rate': 0.0,
            'avg_latency': 0.0,
            'error_count': 0,
            'consistency': 0.0
        }

        successes = 0
        latencies = []
        outputs = []

        for task in self.test_tasks[:num_evals]:
            import time
            start = time.time()

            try:
                output = agent.execute(task)
                latency = time.time() - start

                if self.success_metric(output, task):
                    successes += 1

                latencies.append(latency)
                outputs.append(output)

            except Exception as e:
                results['error_count'] += 1

        # Aggregate metrics
        results['success_rate'] = successes / max(num_evals, 1)
        results['avg_latency'] = sum(latencies) / max(len(latencies), 1)

        # Consistency: do we get same output for same input?
        consistency_score = 1.0 if len(set(str(o) for o in outputs)) == 1 else 0.5
        results['consistency'] = consistency_score

        # Combined score
        results['combined_score'] = (
            0.6 * results['success_rate'] +
            0.2 * (1.0 / max(results['avg_latency'], 0.01)) +
            0.2 * results['consistency']
        )

        return results

    def select_best_variant(self, variants: List[AgentProgram]) -> AgentProgram:
        """Evaluate all variants and return the best."""
        best_variant = None
        best_score = -float('inf')

        for variant in variants:
            results = self.evaluate_agent(variant)
            score = results['combined_score']

            if score > best_score:
                best_score = score
                best_variant = variant

        if best_variant:
            best_variant.performance = best_score

        return best_variant
```

### Step 4: Recursive Meta-Improvement

Let the meta-agent improve itself.

```python
class RecursiveImprover:
    """
    Enables the meta-agent to improve itself.
    This is the key to open-ended growth.
    """

    def __init__(self, evaluator: PerformanceEvaluator):
        self.evaluator = evaluator
        self.meta_improvement_history = []
        self.variant_generator = VariantGenerator()

    def improve_task_agent(self, task_agent: AgentProgram,
                          meta_agent: AgentProgram) -> AgentProgram:
        """Use meta-agent to improve task-agent."""

        # Get meta-agent's suggested improvement
        improvement_suggestion = meta_agent.execute(
            task_input=task_agent.code,
            env_context={'evaluation_results': {}}
        )

        if improvement_suggestion and 'error' not in improvement_suggestion:
            improved_code = improvement_suggestion
        else:
            # Fallback: use variant generator
            improved_code = self.variant_generator.add_parameter(task_agent.code)

        improved_agent = task_agent.clone(mutation=improved_code)
        return improved_agent

    def improve_meta_agent(self, meta_agent: AgentProgram,
                          improvement_history: List[Dict]) -> AgentProgram:
        """
        CRITICALLY: Improve the meta-agent itself.
        This enables open-ended improvement.
        """

        # Generate variants of the meta-agent
        meta_variants = self.variant_generator.generate_variants(
            meta_agent, num_variants=3
        )

        # Evaluate each meta-variant by using it to improve a task agent
        # This is slower but demonstrates true meta-improvement

        best_meta_variant = None
        best_meta_score = -float('inf')

        for meta_variant in meta_variants:
            # Quick evaluation: does it produce sensible modifications?
            # (Full evaluation would be expensive)

            suggested_modification = meta_variant.execute(
                task_input={'sample': 'task_code'},
                env_context={}
            )

            # Score: preference for non-error modifications
            is_valid = suggested_modification and 'error' not in str(suggested_modification)
            meta_score = 1.0 if is_valid else 0.0

            if meta_score > best_meta_score:
                best_meta_score = meta_score
                best_meta_variant = meta_variant

        if best_meta_variant and best_meta_score > 0:
            return best_meta_variant

        return meta_agent

    def run_improvement_loop(self, task_agent: AgentProgram,
                            meta_agent: AgentProgram,
                            num_iterations: int = 10) -> Dict[str, AgentProgram]:
        """
        Run recursive improvement loop.
        Each iteration: improve task agent, then improve meta-agent.
        """

        current_task = task_agent
        current_meta = meta_agent

        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration} ===")

            # Phase 1: Improve task agent
            improved_task = self.improve_task_agent(current_task, current_meta)
            task_perf = self.evaluator.evaluate_agent(improved_task)
            print(f"Task agent score: {task_perf['combined_score']:.3f}")

            # Phase 2: Improve meta-agent (the key step!)
            improved_meta = self.improve_meta_agent(current_meta, self.meta_improvement_history)
            print(f"Meta-agent improved (v{improved_meta.version})")

            # Record improvement
            self.meta_improvement_history.append({
                'iteration': iteration,
                'task_score': task_perf['combined_score'],
                'task_agent_hash': improved_task.compute_hash(),
                'meta_agent_hash': improved_meta.compute_hash()
            })

            current_task = improved_task
            current_meta = improved_meta

        return {
            'final_task_agent': current_task,
            'final_meta_agent': current_meta,
            'improvement_history': self.meta_improvement_history
        }
```

## Practical Guidance

**Hyperparameters:**
- Number of variants per iteration: 3-7 (balance exploration vs. compute)
- Mutation types: use 3-5 different operators (diversity improves search)
- Evaluation budget per variant: 10-50 test cases (faster evals allow more iterations)
- Meta-improvement frequency: every 2-5 task iterations

**When to Use:**
- Long-running systems where continuous improvement is valuable
- Domains where the improvement strategy itself can vary
- Research environments exploring open-ended learning
- When you have compute budget for recursive evaluation

**When NOT to Use:**
- Real-time systems (recursive improvement adds latency)
- Safety-critical domains (uncontrolled self-modification is risky)
- Single-shot tasks (improvement overhead not justified)
- Systems requiring formal verification (self-modification hard to analyze)

**Pitfalls:**
- Runaway mutations: without checks, code can diverge into nonsense; validate structure
- Evaluation noise: small performance differences lead to random direction; use multiple runs
- Positive feedback loops: once an improvement strategy works, it dominates; diversify
- Unbounded code growth: programs can bloat; track code size and penalize if necessary

## Reference

Paper: [arxiv.org/abs/2603.19461](https://arxiv.org/abs/2603.19461)
