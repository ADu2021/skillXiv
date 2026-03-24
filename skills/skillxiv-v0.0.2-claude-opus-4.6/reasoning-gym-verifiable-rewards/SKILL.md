---
name: reasoning-gym-verifiable-rewards
title: "REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24760"
keywords: [Reinforcement Learning, Verifiable Rewards, Reasoning, Procedural Generation]
description: "Create infinite training environments for reasoning with automatic verification using procedural generation and domain-specific evaluators."
---

# Reasoning Gym: Train RL on Infinite Verifiable Reasoning Tasks

Standard RL for reasoning bottlenecks on fixed datasets: collect 10k problems, train until convergence, hit a ceiling. Reasoning Gym inverts this by providing procedurally generated reasoning environments with automatic correctness verification. Generate virtually infinite algebra problems, logic puzzles, geometry proofs, and games at tunable difficulty. Each problem includes a verifier that checks solutions automatically, providing reliable reward signals for RL. Train continuously on increasingly difficult problems without ever repeating an example.

This enables continuous curriculum learning where models improve across escalating complexity, and eliminates the data collection bottleneck that limits reasoning research.

## Core Concept

Procedural generation + automatic verification = infinite training signal. For each domain (algebra, logic, geometry, etc.), implement a generator that produces valid problem instances with adjustable parameters, and a verifier that evaluates solution correctness deterministically. This decouples problem quantity from manual annotation effort. As models improve, increase difficulty parameters; environments adapt to learner progress automatically.

## Architecture Overview

- **Procedural Generators**: Per-domain problem generators (algebra equations, logic formulas, geometry diagrams) parameterized by difficulty
- **Verifiers**: Deterministic correctness checkers for each domain (symbolic equation solvers, proof validators, game rule checkers)
- **Difficulty Controller**: Automatically adjusts problem complexity based on learner performance (curriculum learning)
- **Multi-Domain Coverage**: 100+ generators spanning math, logic, games, cognition (sorting, counting), and more
- **RL Integration**: Seamless integration with standard RL algorithms (PPO, GRPO) providing reward = correctness signal

## Implementation

This implementation demonstrates procedural generation with verification for key reasoning domains.

Implement algebra problem generation and verification:

```python
import random
import sympy as sp
from typing import Tuple, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Problem:
    task_id: str
    problem_text: str
    solution_correct: bool
    difficulty: float

class ReasoningEnvironment(ABC):
    """Base class for procedurally generated reasoning environments."""

    @abstractmethod
    def generate_problem(self, difficulty: float) -> Problem:
        """Generate a problem at given difficulty level."""
        pass

    @abstractmethod
    def verify_solution(self, problem: Problem, solution: str) -> bool:
        """Verify if solution is correct."""
        pass

class AlgebraEnvironment(ReasoningEnvironment):
    """Linear and quadratic equation solving."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_problem(self, difficulty: float = 0.5) -> Problem:
        """
        Generate algebra problems with difficulty 0-1.
        0: simple linear (x + 2 = 5)
        1: quadratic with fractions (2x^2 - 3x + 1 = 0)
        """
        if difficulty < 0.33:
            # Linear: ax + b = c
            a = random.randint(1, 5)
            b = random.randint(-10, 10)
            c = random.randint(-20, 20)
            problem_text = f"Solve: {a}x + {b} = {c}"
            solution_value = (c - b) / a

        elif difficulty < 0.66:
            # Linear with two variables or more complex
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            c = random.randint(-50, 50)
            problem_text = f"Solve: {a}x + {b}x = {c}"
            solution_value = c / (a + b)

        else:
            # Quadratic: ax^2 + bx + c = 0
            a = random.randint(1, 5)
            b = random.randint(-10, 10)
            c = random.randint(-10, 10)
            problem_text = f"Solve: {a}x^2 + {b}x + {c} = 0"
            # Use sympy to get exact solution
            x = sp.Symbol('x')
            eq = a*x**2 + b*x + c
            solutions = sp.solve(eq, x)
            solution_value = float(solutions[0]) if solutions else None

        return Problem(
            task_id=f"algebra_{difficulty:.2f}_{hash(problem_text)}",
            problem_text=problem_text,
            solution_correct=True,
            difficulty=difficulty
        )

    def verify_solution(self, problem: Problem, solution_str: str) -> bool:
        """
        Verify algebraic solution using sympy.
        Extract numeric answer from solution text and verify.
        """
        try:
            # Simple extraction: look for "x =" pattern
            if "=" not in solution_str:
                return False

            answer_part = solution_str.split("=")[-1].strip()
            proposed_answer = float(answer_part)

            # Extract original equation from problem
            problem_text = problem.problem_text
            if "Solve:" not in problem_text:
                return False

            eq_str = problem_text.split("Solve:")[-1].strip()
            x = sp.Symbol('x')

            # Parse and solve
            eq = sp.sympify(eq_str)
            solutions = sp.solve(eq, x)

            # Check if proposed answer matches any solution
            for sol in solutions:
                if abs(float(sol) - proposed_answer) < 1e-6:
                    return True
            return False

        except (ValueError, SyntaxError, sp.SympifyError):
            return False

class LogicEnvironment(ReasoningEnvironment):
    """Propositional and first-order logic problems."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.predicates = ["likes", "knows", "owns", "visited"]
        self.entities = ["Alice", "Bob", "Charlie", "Diana"]

    def generate_problem(self, difficulty: float = 0.5) -> Problem:
        """Generate logic reasoning problems."""
        if difficulty < 0.5:
            # Simple propositional logic
            propositions = ["P", "Q", "R"]
            selected = random.sample(propositions, 2)
            op = random.choice(["and", "or", "implies"])
            problem_text = f"Evaluate: {selected[0]} {op} {selected[1]}"

        else:
            # First-order logic with predicates
            pred = random.choice(self.predicates)
            e1, e2 = random.sample(self.entities, 2)
            problem_text = f"Is it true that {e1} {pred} {e2}?"

        return Problem(
            task_id=f"logic_{difficulty:.2f}_{hash(problem_text)}",
            problem_text=problem_text,
            solution_correct=True,
            difficulty=difficulty
        )

    def verify_solution(self, problem: Problem, solution_str: str) -> bool:
        """Verify logical solutions using symbolic evaluation."""
        try:
            # Extract True/False answer
            answer = solution_str.strip().upper()
            if "TRUE" in answer or answer == "T":
                return True  # Placeholder: in real system, verify against KB
            elif "FALSE" in answer or answer == "F":
                return False
            return False

        except Exception:
            return False

# Test environments
algebra_env = AlgebraEnvironment()
logic_env = LogicEnvironment()

# Generate and verify problems
for difficulty in [0.2, 0.5, 0.8]:
    alg_prob = algebra_env.generate_problem(difficulty)
    print(f"Algebra (difficulty {difficulty}): {alg_prob.problem_text}")

    logic_prob = logic_env.generate_problem(difficulty)
    print(f"Logic (difficulty {difficulty}): {logic_prob.problem_text}")
    print()
```

Build a curriculum learning controller that adapts difficulty:

```python
class CurriculumController:
    """Automatically adjust problem difficulty based on performance."""

    def __init__(self, initial_difficulty: float = 0.2,
                 performance_window: int = 100):
        self.current_difficulty = initial_difficulty
        self.performance_window = performance_window
        self.recent_performance = []
        self.episode_count = 0

    def update_performance(self, was_correct: bool):
        """Record episode result."""
        self.recent_performance.append(was_correct)
        if len(self.recent_performance) > self.performance_window:
            self.recent_performance.pop(0)
        self.episode_count += 1

    def get_current_difficulty(self) -> float:
        """Return current difficulty level (0-1)."""
        return min(1.0, self.current_difficulty)

    def adjust_difficulty(self):
        """Increase/decrease difficulty based on recent performance."""
        if len(self.recent_performance) < self.performance_window:
            return  # Not enough data yet

        recent_accuracy = sum(self.recent_performance) / len(self.recent_performance)

        # Increase difficulty if doing well, decrease if struggling
        if recent_accuracy > 0.8:
            # Model is succeeding: make problems harder
            self.current_difficulty = min(1.0, self.current_difficulty + 0.05)
        elif recent_accuracy < 0.4:
            # Model is struggling: make problems easier
            self.current_difficulty = max(0.0, self.current_difficulty - 0.05)

    def episode_done(self, was_correct: bool):
        """Call at end of each RL episode."""
        self.update_performance(was_correct)
        if self.episode_count % 50 == 0:
            self.adjust_difficulty()

# Usage in RL training loop
curriculum = CurriculumController(initial_difficulty=0.2)

for episode in range(1000):
    # Get problem at current difficulty
    difficulty = curriculum.get_current_difficulty()
    problem = algebra_env.generate_problem(difficulty)

    # RL agent solves problem (placeholder)
    agent_solution = "x = 3"  # In practice: LLM generates this

    # Verify solution
    is_correct = algebra_env.verify_solution(problem, agent_solution)

    # Update curriculum
    curriculum.episode_done(is_correct)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: Difficulty={difficulty:.2f}, "
              f"Recent accuracy={sum(curriculum.recent_performance[-50:])/50:.1%}")
```

Build a multi-domain reasoning gym:

```python
class ReasoningGym:
    """Multi-domain reasoning environment with curriculum learning."""

    def __init__(self):
        self.environments = {
            "algebra": AlgebraEnvironment(),
            "logic": LogicEnvironment(),
            # Add more domains: geometry, arithmetic, games, etc.
        }
        self.curriculums = {
            domain: CurriculumController()
            for domain in self.environments
        }
        self.current_domain = None

    def sample_task(self, domain: str = None) -> Tuple[Problem, str]:
        """Sample a reasoning task at curriculum-appropriate difficulty."""
        if domain is None:
            # Sample domain uniformly
            domain = random.choice(list(self.environments.keys()))

        self.current_domain = domain
        env = self.environments[domain]
        curriculum = self.curriculums[domain]

        difficulty = curriculum.get_current_difficulty()
        problem = env.generate_problem(difficulty)

        return problem, domain

    def evaluate_solution(self, solution: str) -> bool:
        """Verify solution for current task."""
        env = self.environments[self.current_domain]
        return env.verify_solution(problem, solution)

    def step(self, solution: str) -> Tuple[float, bool]:
        """
        Execute step: evaluate solution and update curriculum.
        Returns (reward, done).
        """
        is_correct = self.evaluate_solution(solution)
        reward = 1.0 if is_correct else 0.0

        curriculum = self.curriculums[self.current_domain]
        curriculum.episode_done(is_correct)

        return reward, True  # Each problem is one episode

# Create multi-domain gym
gym = ReasoningGym()

# Simulate RL training across domains
for step in range(500):
    problem, domain = gym.sample_task()

    # Placeholder: agent generates solution
    if random.random() < 0.6:
        agent_solution = "x = 2"  # Correct answer
    else:
        agent_solution = "x = 999"  # Wrong answer

    reward, done = gym.step(agent_solution)

    if (step + 1) % 100 == 0:
        print(f"Step {step+1}: Sampled from {gym.environments.keys()}")
        for domain, curr in gym.curriculums.items():
            print(f"  {domain}: difficulty={curr.get_current_difficulty():.2f}")
```

## Practical Guidance

| Aspect | Details |
|--------|---------|
| **Initial Difficulty** | Start at 0.1-0.3; too hard causes early failure, too easy wastes steps |
| **Performance Window** | 50-200 episodes; balance responsiveness to improvement vs. noise |
| **Difficulty Step Size** | 0.05 per adjustment; 0.02 for fine-grained, 0.10 for coarse |
| **Domain Selection** | Sample uniformly or weight by current performance (easier domains = boost signal) |
| **Verifier Reliability** | Test verifiers against ground truth before training; bugs here invalidate entire RL signal |

**When to Use:**
- RL on reasoning tasks where you want continuous improvement without data collection ceiling
- Research comparing RL algorithms: eliminate dataset variance by using same procedural env
- Curriculum learning: automatically escalate problem difficulty as model improves
- Multi-task learning: train single model across domains with automatic mixing
- Scaling RL training indefinitely without hitting dataset limits

**When NOT to Use:**
- Domain has no natural parametric difficulty (open-ended creative tasks)
- Verifier is expensive to run (slows down RL training significantly)
- Need to evaluate on real-world distribution: procedural env may not match target
- Domains with sparse ground truth (open-ended generation, dialogue)

**Common Pitfalls:**
- Verifier bugs: wrong reward signals corrupt entire training; test extensively first
- Difficulty oscillation: curriculum adjusts too aggressively; use larger performance windows
- Domain imbalance: some procedural generators produce easier/harder problems; normalize difficulty
- Overfitting to procedural structure: models learn to exploit generator quirks; randomize problem generation thoroughly

## Reference

REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards
https://arxiv.org/abs/2505.24760
