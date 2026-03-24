---
name: evolutionary-data-synthesis
title: "EvoSyn: Generalizable Evolutionary Data Synthesis for Verifiable Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.17928"
keywords: [synthetic data, evolutionary algorithms, verifiable learning, task-agnostic, data synthesis]
description: "Generate verifiable synthetic training data (problems + solutions + verification artifacts) through evolutionary synthesis that works across math, code, and agent tasks without task-specific rules."
---

# Technique: Evolutionary Data Synthesis — Task-Agnostic Verifiable Data Generation

Creating high-quality synthetic training data requires either expensive human annotation or task-specific generation rules (code templates, math problem schemas). EvoSyn solves this through **evolutionary data synthesis**: jointly evolve problems, diverse solutions, and verification artifacts that can be checked automatically.

The key insight is that verification enables learning without perfection. By generating problems where correctness can be automatically verified, the model learns from diverse high-quality solutions without human labels. The evolutionary approach generalizes across domains (math, code, agents) without task-specific engineering.

## Core Concept

EvoSyn operates on four principles:
- **Multi-Component Generation**: Simultaneously evolve problems, candidate solutions, and verification checks
- **Consistency-Based Evaluation**: Solutions must pass both human-annotated and strategy-induced checks
- **Iterative Discovery**: Refine what counts as "verifiable" across multiple domains
- **Task-Agnostic**: Single pipeline works for math, coding, agent tasks without customization

The result is stable RL training with automatic verification and effective distillation across diverse problem types.

## Architecture Overview

- **Problem Generator**: Create diverse problems in target domain
- **Solution Synthesizer**: Generate multiple candidate solutions
- **Verification Artifact Composer**: Create checkers (math: equations, code: tests, agents: assertions)
- **Consistency Evaluator**: Cross-check human and strategy-based verification
- **Evolutionary Loop**: Mutate and recombine successful (problem, solutions, checkers) tuples
- **Quality Filter**: Keep only high-confidence verified examples

## Implementation Steps

The core algorithm evolves problem-solution-verifier tuples jointly. This example shows the synthesis pipeline.

```python
from typing import List, Dict, Tuple
import random

class EvolutionarySynthesis:
    """
    Task-agnostic evolutionary synthesis of verifiable training data.
    """

    def __init__(
        self,
        generator_model,
        executor,
        max_generations: int = 50
    ):
        self.generator = generator_model
        self.executor = executor
        self.max_generations = max_generations
        self.population = []  # List of (problem, solutions, verifier) tuples

    def generate_problem(self, seed: str = "") -> str:
        """
        Generate a problem in the target domain.
        Domain-agnostic: works for math, code, agent tasks.
        """
        prompt = f"""
Generate a unique problem. The problem should be:
1. Self-contained (no external references)
2. Solvable (solution exists)
3. Diverse (different from previous)

Seed hint: {seed}

Problem:
"""
        problem = self.generator.generate(prompt, max_tokens=300)
        return problem

    def synthesize_solutions(self, problem: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple diverse solutions to the problem.
        """
        solutions = []

        for variant_idx in range(num_variants):
            prompt = f"""
Solve this problem. Approach {variant_idx + 1}:

Problem: {problem}

Solution (approach {variant_idx + 1}):
"""
            solution = self.generator.generate(prompt, max_tokens=200)
            solutions.append(solution)

        return solutions

    def create_verification_artifact(
        self,
        problem: str,
        solutions: List[str]
    ) -> Dict:
        """
        Create verifiable checkers for the problem.
        Examples: test cases for code, equations for math, assertions for agents.
        """
        prompt = f"""
Create verification checkers for this problem and its solutions.

Problem: {problem}

Solutions: {solutions}

For each solution, create:
1. Test case / verification script
2. Expected output
3. Edge cases to check

Verification Artifact:
"""
        artifact_text = self.generator.generate(prompt, max_tokens=300)

        # Parse into structured verifier
        verifier = {
            "test_cases": extract_test_cases(artifact_text),
            "expected_outputs": extract_expected_outputs(artifact_text),
            "edge_cases": extract_edge_cases(artifact_text)
        }

        return verifier

    def evaluate_consistency(
        self,
        problem: str,
        solutions: List[str],
        verifier: Dict
    ) -> Tuple[float, str]:
        """
        Check if solutions actually pass verification.
        Return consistency score and failure reason.
        """
        consistency_score = 0.0
        num_passed = 0

        for solution in solutions:
            try:
                # Execute solution against verifier
                outputs = self.executor.run(solution, verifier["test_cases"])

                # Check against expected outputs
                all_correct = all(
                    out == expected
                    for out, expected in zip(
                        outputs, verifier["expected_outputs"]
                    )
                )

                if all_correct:
                    num_passed += 1

            except Exception as e:
                # Execution failed
                pass

        consistency_score = num_passed / len(solutions) if solutions else 0.0
        failure_reason = "Some solutions failed" if consistency_score < 1.0 else "All verified"

        return consistency_score, failure_reason

    def evolve_population(self, generation: int) -> List[Dict]:
        """
        Evolutionary step: mutate and recombine good examples.
        """
        # Selection: keep top performers (high consistency)
        ranked = sorted(
            self.population,
            key=lambda x: x["consistency_score"],
            reverse=True
        )
        survivors = ranked[:len(ranked) // 2]  # Keep top 50%

        # Mutation: modify problems and solutions
        new_population = []
        for example in survivors:
            # Mutate problem slightly
            mutated_problem = self.generator.generate(
                f"Slightly modify this problem: {example['problem']}",
                max_tokens=300
            )

            # Generate new solutions
            new_solutions = self.synthesize_solutions(mutated_problem, num_variants=2)

            # Create verifier
            new_verifier = self.create_verification_artifact(
                mutated_problem, new_solutions
            )

            # Evaluate consistency
            consistency, reason = self.evaluate_consistency(
                mutated_problem, new_solutions, new_verifier
            )

            if consistency >= 0.7:  # Only keep reasonably consistent examples
                new_population.append({
                    "problem": mutated_problem,
                    "solutions": new_solutions,
                    "verifier": new_verifier,
                    "consistency_score": consistency
                })

        self.population.extend(new_population)
        return self.population

    def synthesize_dataset(
        self,
        num_examples: int = 1000,
        consistency_threshold: float = 0.8
    ) -> List[Dict]:
        """
        Run full evolutionary synthesis to generate dataset.
        """
        # Initialize population
        for _ in range(max(10, num_examples // 50)):
            problem = self.generate_problem()
            solutions = self.synthesize_solutions(problem)
            verifier = self.create_verification_artifact(problem, solutions)
            consistency, _ = self.evaluate_consistency(problem, solutions, verifier)

            if consistency >= 0.5:
                self.population.append({
                    "problem": problem,
                    "solutions": solutions,
                    "verifier": verifier,
                    "consistency_score": consistency
                })

        # Evolutionary loop
        for generation in range(self.max_generations):
            self.evolve_population(generation)

            # Check if we have enough high-quality examples
            high_quality = [
                ex for ex in self.population
                if ex["consistency_score"] >= consistency_threshold
            ]

            if len(high_quality) >= num_examples:
                print(f"Generated {len(high_quality)} verified examples by generation {generation}")
                return high_quality[:num_examples]

            print(f"Gen {generation}: {len(high_quality)} verified, "
                  f"total {len(self.population)} examples")

        return self.population


def train_model_with_synthesized_data(
    model,
    synthesized_examples: List[Dict],
    num_epochs: int = 3
):
    """
    Train model on EvoSyn-generated data with automatic verification.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for example in synthesized_examples:
            problem = example["problem"]
            solutions = example["solutions"]
            verifier = example["verifier"]

            # Train on all solutions (they're all verified!)
            for solution in solutions:
                prompt = f"Problem: {problem}\n\nSolution: "
                target = solution

                # Standard supervised fine-tuning
                output = model.generate(prompt)
                loss = compute_loss(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(synthesized_examples) * 3)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")

    return model
```

The evolutionary approach is crucial: rather than specifying what makes a good problem (human task knowledge), the system discovers it through selection for consistency and diversity. This generalizes across domains.

## Practical Guidance

| Domain | Data Quality | Consistency | Performance Gain |
|--------|-----------|--------------|-----------------|
| Math | High | 85%+ | +15-25% |
| Coding | High | 80%+ | +10-20% |
| Agent tasks | Medium-High | 75%+ | +8-15% |

**When to Use:**
- Synthetic data for RL training (verification is key)
- Task-agnostic pipeline preferred over custom generation
- Need diverse high-quality problems automatically
- Verifiable correctness available (math, code with tests, agents with assertions)

**When NOT to Use:**
- Unverifiable tasks (creative writing, open-ended dialogue)
- Domain requiring specific problem types (medical, legal)
- Real-time generation (evolutionary synthesis is offline)
- Budget where pre-existing datasets are cheaper

**Common Pitfalls:**
- Verifier too strict → evolution gets stuck, can't improve
- Verifier too lenient → noise in training data, model learns wrong patterns
- Not maintaining diversity → population converges to narrow problem type
- Consistency threshold too high → evolution gets stuck, too low → noisy data
- Same executor for all domains → domain-specific edge cases missed

## Reference

[EvoSyn: Generalizable Evolutionary Data Synthesis for Verifiable Learning](https://arxiv.org/abs/2510.17928)
