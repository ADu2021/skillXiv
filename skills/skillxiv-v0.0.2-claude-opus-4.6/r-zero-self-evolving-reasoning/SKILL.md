---
name: r-zero-self-evolving-reasoning
title: R-Zero - Self-Evolving Reasoning LLM from Zero Data
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.05004
keywords: [self-play, co-evolution, reasoning, data-generation, bootstrapping]
description: "Co-evolutionary framework where Challenger generates tasks and Solver solves them. Models evolve autonomously from scratch without human annotations. Improves math reasoning +6.49pts and general reasoning +7.54pts."
---

# R-Zero: Self-Evolving Reasoning LLM from Zero Data

## Core Concept

R-Zero eliminates dependency on human-curated task datasets through autonomous co-evolution. Two models—a Challenger that generates progressively harder tasks and a Solver that tackles them—interact in a feedback loop that generates training data and improves both models without any initial labeled dataset. This enables models to break through baseline capabilities through self-generated, curriculum-learned tasks.

## Architecture Overview

- **Challenger Model**: Generates tasks at the frontier of Solver capability
- **Solver Model**: Solves Challenger's tasks and learns from outcomes
- **Co-evolutionary Feedback**: Challenger rewarded for hard-but-solvable tasks; Solver for solving harder tasks
- **Curriculum Learning**: Task difficulty naturally increases as Solver improves
- **Data Generation**: Training data emerges from interaction, not annotation

## Implementation Steps

### Step 1: Initialize Challenger and Solver Models

Set up two models with appropriate initializations and objectives.

```python
import torch
from typing import Dict, List, Tuple

class ChallengerModel(torch.nn.Module):
    """
    Generates tasks at the frontier of solver capability.
    """

    def __init__(self, base_model, task_domain="math"):
        super().__init__()
        self.base_model = base_model
        self.task_domain = task_domain
        self.task_buffer = []
        self.difficulty_estimate = 0.5  # 0-1 scale

    def generate_task(self, difficulty_target: float) -> str:
        """
        Generate a task at target difficulty.

        Args:
            difficulty_target: Target difficulty 0-1 (0=trivial, 1=impossible)

        Returns:
            Task prompt
        """
        prompt = f"""
        Generate a {self.task_domain} problem.

        Difficulty target: {difficulty_target:.1f}/1.0
        - 0.0: Trivial, easily solvable
        - 0.5: Medium difficulty, requires some reasoning
        - 1.0: Very hard, near impossible

        Ensure the problem is clear and has a verifiable solution.
        Problem:
        """

        task = self.base_model.generate(prompt, max_length=300)

        return task

    def estimate_task_difficulty(self, task: str, solver_accuracy: float) -> float:
        """
        Estimate task difficulty based on solver performance.

        Args:
            task: Task description
            solver_accuracy: Proportion of attempts solver succeeds

        Returns:
            Estimated difficulty 0-1
        """
        # Difficulty inversely related to solver accuracy
        estimated_difficulty = 1.0 - solver_accuracy

        return estimated_difficulty

    def compute_reward(self, task: str, solver_solved: bool, solver_accuracy: float) -> float:
        """
        Compute reward for Challenger.

        Challenger rewarded for:
        - Generating tasks Solver cannot solve (high difficulty)
        - But not so hard Solver never solves them (curriculum)

        Args:
            task: Generated task
            solver_solved: Whether Solver succeeded
            solver_accuracy: Solver success rate on similar tasks

        Returns:
            Reward signal for Challenger
        """
        # Target solver success rate around 30-50% (challenging but solvable)
        target_success_rate = 0.4

        difficulty = 1.0 - solver_accuracy

        # Reward is highest when difficulty is at target
        reward = -(difficulty - target_success_rate) ** 2

        return reward
```

### Step 2: Implement Solver Model

Create a model that solves Challenger tasks and provides learning signal.

```python
class SolverModel(torch.nn.Module):
    """
    Solves Challenger's tasks and learns from outcomes.
    """

    def __init__(self, base_model, task_domain="math"):
        super().__init__()
        self.base_model = base_model
        self.task_domain = task_domain
        self.solution_buffer = []
        self.performance_history = []

    def solve_task(self, task: str, max_attempts: int = 1) -> Tuple[str, bool]:
        """
        Attempt to solve a task.

        Args:
            task: Task prompt
            max_attempts: Number of solution attempts

        Returns:
            (solution, is_correct)
        """
        prompt = f"""
        Solve this {self.task_domain} problem:

        {task}

        Show your work and provide final answer.
        Solution:
        """

        solution = self.base_model.generate(prompt, max_length=500)

        # Verify solution (domain-specific)
        is_correct = self.verify_solution(task, solution)

        return solution, is_correct

    def verify_solution(self, task: str, solution: str) -> bool:
        """
        Verify correctness of solution.

        Args:
            task: Original task
            solution: Proposed solution

        Returns:
            True if solution is correct
        """
        verification_prompt = f"""
        Task: {task}
        Proposed solution: {solution}

        Is this solution mathematically correct?
        Respond with only YES or NO.
        """

        response = self.base_model.generate(verification_prompt, max_length=10)

        return "YES" in response.upper()

    def compute_reward(self, task: str, solution: str, is_correct: bool) -> float:
        """
        Compute reward for Solver.

        Solver rewarded for solving increasingly hard tasks.

        Args:
            task: Task that was solved
            solution: Solution provided
            is_correct: Whether solution is correct

        Returns:
            Reward signal for Solver
        """
        if is_correct:
            # Reward based on how hard the task is
            task_difficulty = self.estimate_task_difficulty(task)
            reward = task_difficulty  # Harder tasks = higher reward
        else:
            reward = -0.5  # Penalty for incorrect solution

        return reward

    def estimate_task_difficulty(self, task: str) -> float:
        """Estimate difficulty of a task."""
        # Heuristic: complexity based on task description
        complexity_score = min(len(task.split()), 100) / 100.0
        return complexity_score

    def learn_from_tasks(self, tasks: List[str], batch_size: int = 4):
        """
        Learn from solved tasks.

        Args:
            tasks: List of tasks to train on
            batch_size: Training batch size
        """
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i+batch_size]

            for task in batch_tasks:
                solution, is_correct = self.solve_task(task)

                if is_correct:
                    # Learn from correct solutions
                    self.train_on_solution(task, solution)

    def train_on_solution(self, task: str, solution: str):
        """Fine-tune on task-solution pair."""
        # Standard supervised fine-tuning on correct solution
        prompt = f"Problem: {task}\nSolution:"
        self.base_model.finetune(prompt, solution)
```

### Step 3: Implement Co-evolutionary Training Loop

Create the main loop where Challenger and Solver interact.

```python
class CoevolutionaryFramework:
    """
    Framework orchestrating Challenger-Solver co-evolution.
    """

    def __init__(self, challenger: ChallengerModel, solver: SolverModel):
        self.challenger = challenger
        self.solver = solver
        self.generation = 0

        self.task_archive = []
        self.performance_log = []

    def run_generation(self, num_tasks: int = 20) -> Dict:
        """
        Run one generation of co-evolution.

        Args:
            num_tasks: Number of tasks to generate and solve

        Returns:
            Metrics for this generation
        """
        self.generation += 1

        tasks_generated = []
        tasks_solved_correctly = 0
        challenger_rewards = []
        solver_rewards = []

        # Dynamic difficulty: increase as solver improves
        base_difficulty = self._compute_difficulty_target()

        for task_idx in range(num_tasks):
            # 1. Challenger generates task
            task = self.challenger.generate_task(difficulty_target=base_difficulty)
            tasks_generated.append(task)

            # 2. Solver attempts to solve
            solution, is_correct = self.solver.solve_task(task)

            if is_correct:
                tasks_solved_correctly += 1

            # 3. Compute rewards
            solver_success_rate = tasks_solved_correctly / (task_idx + 1)

            # Challenger reward: task at right difficulty
            challenger_reward = self.challenger.compute_reward(
                task,
                is_correct,
                solver_success_rate
            )
            challenger_rewards.append(challenger_reward)

            # Solver reward: harder tasks give more reward
            solver_reward = self.solver.compute_reward(task, solution, is_correct)
            solver_rewards.append(solver_reward)

            # Store task for learning
            self.task_archive.append({
                "task": task,
                "solution": solution,
                "correct": is_correct,
                "generation": self.generation
            })

        # 4. Learn from generated tasks
        correct_tasks = [t["task"] for t in self.task_archive
                        if t["correct"] and t["generation"] == self.generation]

        self.solver.learn_from_tasks(correct_tasks)

        # 5. Update models via RL
        # (In practice, use policy gradient methods)
        self._update_models_from_rewards(challenger_rewards, solver_rewards)

        # Compute metrics
        metrics = {
            "generation": self.generation,
            "tasks_generated": len(tasks_generated),
            "tasks_solved": tasks_solved_correctly,
            "success_rate": tasks_solved_correctly / num_tasks,
            "avg_challenger_reward": sum(challenger_rewards) / len(challenger_rewards),
            "avg_solver_reward": sum(solver_rewards) / len(solver_rewards),
            "avg_difficulty": base_difficulty
        }

        self.performance_log.append(metrics)

        return metrics

    def _compute_difficulty_target(self) -> float:
        """
        Compute target difficulty for this generation.

        Difficulty increases as solver improves (curriculum learning).

        Returns:
            Difficulty target 0-1
        """
        if not self.performance_log:
            return 0.3  # Start easy

        # Increase difficulty if solver succeeds >50%
        recent_success = self.performance_log[-1]["success_rate"]

        if recent_success > 0.5:
            # Increase difficulty
            new_difficulty = min(1.0, self.performance_log[-1]["avg_difficulty"] * 1.1)
        elif recent_success < 0.2:
            # Decrease difficulty
            new_difficulty = max(0.1, self.performance_log[-1]["avg_difficulty"] / 1.1)
        else:
            new_difficulty = self.performance_log[-1]["avg_difficulty"]

        return new_difficulty

    def _update_models_from_rewards(self, challenger_rewards: List[float], solver_rewards: List[float]):
        """Update models using rewards (simplified)."""
        # In practice, use REINFORCE or other policy gradient methods
        # For now, just update solver on correct tasks
        pass

    def run_coevolution(self, num_generations: int = 10):
        """
        Run full co-evolutionary training.

        Args:
            num_generations: Number of generations to evolve
        """
        for gen in range(num_generations):
            metrics = self.run_generation(num_tasks=20)

            print(f"Generation {gen}: "
                  f"Success Rate={metrics['success_rate']:.1%}, "
                  f"Difficulty={metrics['avg_difficulty']:.2f}")

        return self.performance_log
```

### Step 4: Evaluate Final Performance

Benchmark the evolved model on standard benchmarks.

```python
def evaluate_r_zero_model(solver_model, benchmark_suites: Dict[str, List]) -> Dict:
    """
    Evaluate R-Zero trained model on benchmarks.

    Args:
        solver_model: Evolved solver model
        benchmark_suites: Different benchmark suites (math, reasoning, etc.)

    Returns:
        Performance on each benchmark
    """
    results = {}

    for benchmark_name, problems in benchmark_suites.items():
        correct = 0

        for problem in problems:
            solution, is_correct = solver_model.solve_task(problem["prompt"])
            if is_correct:
                correct += 1

        accuracy = correct / len(problems)
        results[benchmark_name] = accuracy

        print(f"{benchmark_name}: {accuracy:.1%}")

    return results
```

## Practical Guidance

### When to Use R-Zero

- **No human annotations available**: Bootstrapping models from scratch
- **Continuous learning scenarios**: Models that improve by self-play
- **Curriculum learning applications**: Progressive task difficulty needed
- **Research on self-improvement**: Understanding autonomous capability growth

### When NOT to Use R-Zero

- **Abundant labeled data**: Supervised learning likely more efficient
- **Fixed task distribution**: No need for curriculum generation
- **Real-time constraints**: Co-evolution is slow and compute-intensive
- **Safety-critical systems**: Hard to control generated task distribution

### Hyperparameter Recommendations

- **Target solver success rate**: 0.3-0.5 (challenging but solvable)
- **Difficulty increase factor**: 1.05-1.15 per generation
- **Task batch size**: 16-32 tasks per generation
- **Number of generations**: 10-20 for meaningful improvement
- **Tasks per generation**: 20-50 depends on task complexity

### Key Insights

The critical insight is that task generation and solving can bootstrap each other. By rewarding the Challenger for hard-but-solvable tasks, the framework naturally creates a curriculum. The Solver improves by learning from successfully solved tasks, which in turn motivates the Challenger to generate harder problems. This virtuous cycle enables capability growth without human annotation.

## Reference

**R-Zero: Self-Evolving Reasoning LLM from Zero Data** (arXiv:2508.05004)

Introduces co-evolutionary framework with Challenger and Solver models that generate training data and improve autonomously. Demonstrates +6.49pts math and +7.54pts general reasoning improvement without pre-existing labeled data.
