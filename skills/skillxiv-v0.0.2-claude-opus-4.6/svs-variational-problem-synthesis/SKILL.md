---
name: svs-variational-problem-synthesis
title: "Self-Play with Variational Problem Synthesis for Sustained Reasoning Diversity"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.14029
keywords: [reinforcement-learning, problem-synthesis, entropy-preservation, diversity, self-play]
description: "Generate problem variants from correct model solutions while preserving answer equivalence, enabling self-play training that maintains output diversity and prevents entropy collapse."
---

# Self-Play with Variational Problem Synthesis (SvS)

## Core Concept

SvS addresses entropy collapse in reasoning RL by synthesizing problem variants derived from correct model solutions. The key insight is that while Pass@1 improves with RL training, Pass@k degrades due to reduced output diversity. By generating new problems from correct solutions (maintaining identical answers), SvS sustains policy entropy during training. This approach achieves 18-22% improvements on AIME benchmarks while preserving the model's ability to generate diverse high-quality solutions.

## Architecture Overview

- **Correct Solution Extraction**: Identify and isolate correct reasoning steps
- **Invariant Answer Identification**: Determine what must remain constant across variants
- **Problem Variant Generation**: Create mathematically equivalent problems
- **Self-Play Training Loop**: Iteratively improve on synthetic problems
- **Entropy Preservation**: Monitor and maintain output diversity metrics

## Implementation Steps

### 1. Extract and Analyze Correct Solutions

Identify successful reasoning traces for variant generation:

```python
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Solution:
    """Representation of a model solution."""
    problem: str
    reasoning_steps: List[str]
    final_answer: str
    intermediate_values: Dict[str, any]
    confidence: float
    is_correct: bool

class SolutionAnalyzer:
    def __init__(self, verifier: "AnswerVerifier"):
        self.verifier = verifier

    def extract_correct_solutions(
        self,
        samples: List[Tuple[str, str, str]],  # (problem, response, answer)
        num_best: int = 5
    ) -> List[Solution]:
        """
        Extract solutions verified as correct.
        """
        correct_solutions = []

        for problem, response, predicted_answer in samples:
            # Verify correctness
            is_correct = self.verifier.verify(predicted_answer, problem)

            if is_correct:
                # Parse solution structure
                solution = Solution(
                    problem=problem,
                    reasoning_steps=self._parse_steps(response),
                    final_answer=predicted_answer,
                    intermediate_values=self._extract_values(response),
                    confidence=1.0 if is_correct else 0.0,
                    is_correct=is_correct
                )
                correct_solutions.append(solution)

        # Return top solutions by confidence/quality
        correct_solutions.sort(key=lambda s: s.confidence, reverse=True)
        return correct_solutions[:num_best]

    def _parse_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response."""
        # Split by logical boundaries (Step, Therefore, etc.)
        import re
        steps = re.split(r'Step|Therefore|Thus|So', response)
        return [s.strip() for s in steps if s.strip()]

    def _extract_values(self, response: str) -> Dict[str, any]:
        """Extract intermediate numerical values."""
        import re
        # Match patterns like "x = 5", "area = 20", etc.
        pattern = r'(\w+)\s*=\s*([\d\.\-]+)'
        matches = re.findall(pattern, response)
        return {k: float(v) for k, v in matches}
```

### 2. Implement Problem Variant Generation

Create mathematically equivalent problems from solutions:

```python
class ProblemVariantGenerator:
    """Generate problem variants maintaining answer equivalence."""

    def __init__(self, problem_template_library: Dict[str, List[str]]):
        self.templates = problem_template_library

    def generate_variants(
        self,
        solution: Solution,
        num_variants: int = 3,
        variant_type: str = "substitution"
    ) -> List[Tuple[str, str]]:
        """
        Generate problem variants from correct solution.

        Returns: [(new_problem, expected_answer), ...]
        """
        variants = []

        if variant_type == "substitution":
            # Change numbers but preserve answer
            for i in range(num_variants):
                new_problem = self._substitute_values(
                    solution.problem,
                    solution.intermediate_values,
                    variant_index=i
                )
                variants.append((new_problem, solution.final_answer))

        elif variant_type == "reformulation":
            # Rephrase problem while preserving structure
            for i in range(num_variants):
                new_problem = self._reformulate_problem(
                    solution.problem,
                    solution.reasoning_steps,
                    variant_index=i
                )
                variants.append((new_problem, solution.final_answer))

        elif variant_type == "context_swap":
            # Use different context but same mathematical structure
            for i in range(num_variants):
                new_problem = self._swap_context(
                    solution.problem,
                    solution.final_answer,
                    variant_index=i
                )
                variants.append((new_problem, solution.final_answer))

        return variants

    def _substitute_values(
        self,
        original_problem: str,
        known_values: Dict[str, float],
        variant_index: int = 0
    ) -> str:
        """
        Create variant by substituting different numbers.

        Key: keep mathematical relationships intact.
        """
        import re

        problem = original_problem
        seed = variant_index

        for var_name, original_val in known_values.items():
            # Generate new value preserving answer
            if isinstance(original_val, float) and original_val != 0:
                # Scale by small factor
                scale = 1.0 + (seed * 0.1) % 0.5
                new_val = original_val * scale

                # Replace in problem
                pattern = re.escape(str(int(original_val) if original_val == int(original_val) else original_val))
                problem = re.sub(pattern, str(int(new_val) if new_val == int(new_val) else new_val), problem)

        return problem

    def _reformulate_problem(
        self,
        problem: str,
        reasoning_steps: List[str],
        variant_index: int = 0
    ) -> str:
        """
        Reformulate problem preserving mathematical structure.

        Example: "Find x such that..." -> "Solve for x where..."
        """
        reformulations = {
            "find": "calculate",
            "what is": "determine",
            "compute": "evaluate",
            "how many": "count the number of",
            "if": "suppose",
        }

        problem_lower = problem.lower()
        for original, replacement in reformulations.items():
            if original in problem_lower:
                problem = problem.replace(original, replacement)
                break

        return problem

    def _swap_context(
        self,
        original_problem: str,
        answer: str,
        variant_index: int = 0
    ) -> str:
        """
        Create variant with different real-world context.
        """
        contexts = [
            "In a classroom with",
            "At a store with",
            "During a game with",
            "In a garden with",
            "For a project with",
        ]

        # Extract numerical content from original
        import re
        numbers = re.findall(r'\d+', original_problem)

        if variant_index < len(contexts):
            context = contexts[variant_index]
        else:
            context = contexts[0]

        # Reconstruct with new context
        new_problem = f"{context} {' and '.join(numbers)} items..."
        return new_problem
```

### 3. Implement Self-Play Training Loop

Train model on generated variants:

```python
class SelfPlayTrainer:
    """Trains model on self-generated problem variants."""

    def __init__(
        self,
        model: "LLM",
        variant_generator: ProblemVariantGenerator,
        solution_analyzer: SolutionAnalyzer,
        reward_model: "RewardModel"
    ):
        self.model = model
        self.variant_generator = variant_generator
        self.analyzer = solution_analyzer
        self.reward_model = reward_model
        self.training_history = []

    def self_play_iteration(
        self,
        base_problems: List[str],
        num_solutions_per_problem: int = 8,
        num_variants_per_solution: int = 2,
        num_training_steps: int = 100
    ) -> Dict[str, float]:
        """
        Execute single self-play iteration.
        """

        iteration_metrics = {}

        # Step 1: Generate solutions on base problems
        print("Generating solutions on base problems...")
        base_solutions = []
        for problem in base_problems:
            solutions = self._generate_n_solutions(
                problem,
                num_solutions_per_problem
            )
            base_solutions.extend(solutions)

        # Step 2: Extract correct solutions
        print("Extracting correct solutions...")
        correct_solutions = self.analyzer.extract_correct_solutions(
            [(s.problem, s.reasoning_steps, s.final_answer) for s in base_solutions],
            num_best=len(base_solutions) // 2
        )

        iteration_metrics["num_correct"] = len(correct_solutions)
        iteration_metrics["base_accuracy"] = len(correct_solutions) / len(base_solutions)

        # Step 3: Generate variants
        print("Generating problem variants...")
        variant_dataset = []
        for solution in correct_solutions:
            variants = self.variant_generator.generate_variants(
                solution,
                num_variants=num_variants_per_solution,
                variant_type="substitution"
            )
            variant_dataset.extend(variants)

        # Step 4: Train on variants
        print(f"Training on {len(variant_dataset)} variants...")
        variant_loss = self._train_on_variants(
            variant_dataset,
            num_steps=num_training_steps
        )

        iteration_metrics["variant_loss"] = variant_loss

        # Step 5: Measure entropy preservation
        print("Measuring output diversity...")
        base_entropy, variant_entropy = self._measure_entropy(
            base_problems,
            num_solutions_per_problem
        )

        iteration_metrics["base_entropy"] = base_entropy
        iteration_metrics["variant_entropy"] = variant_entropy
        iteration_metrics["entropy_preservation"] = variant_entropy / (base_entropy + 1e-8)

        self.training_history.append(iteration_metrics)
        return iteration_metrics

    def _generate_n_solutions(
        self,
        problem: str,
        n: int = 8
    ) -> List[Solution]:
        """Generate n diverse solutions for a problem."""
        solutions = []

        for i in range(n):
            # Generate with different random seeds for diversity
            response = self.model.generate(
                problem,
                temperature=0.7 + (i % 3) * 0.1,  # Vary temperature
                max_tokens=500
            )

            solution = Solution(
                problem=problem,
                reasoning_steps=self.analyzer._parse_steps(response),
                final_answer=self._extract_answer(response),
                intermediate_values=self.analyzer._extract_values(response),
                confidence=0.5,  # Will be verified later
                is_correct=False
            )
            solutions.append(solution)

        return solutions

    def _train_on_variants(
        self,
        variant_dataset: List[Tuple[str, str]],
        num_steps: int = 100,
        batch_size: int = 4
    ) -> float:
        """
        Train model on variant dataset using RL.
        """
        total_loss = 0.0

        for step in range(num_steps):
            # Sample batch
            batch_variants = variant_dataset[
                (step * batch_size) % len(variant_dataset):
                ((step + 1) * batch_size) % len(variant_dataset)
            ]

            # Collect responses
            batch_loss = 0.0
            for problem, expected_answer in batch_variants:
                response = self.model.generate(problem, max_tokens=500)
                predicted_answer = self._extract_answer(response)

                # Compute reward
                reward = self.reward_model.compute_reward(
                    problem,
                    response,
                    expected_answer
                )

                # Policy gradient loss
                loss = -reward * self.model.log_prob(response)
                batch_loss += loss.item()

            avg_batch_loss = batch_loss / len(batch_variants)
            total_loss += avg_batch_loss

            # Optimize
            self.model.optimizer.zero_grad()
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.model.optimizer.step()

        return total_loss / num_steps

    def _measure_entropy(
        self,
        problems: List[str],
        num_solutions: int = 8
    ) -> Tuple[float, float]:
        """
        Measure output diversity (Shannon entropy of solution tokens).
        """
        import torch
        from scipy.stats import entropy

        all_solutions = []
        for problem in problems:
            for _ in range(num_solutions):
                response = self.model.generate(problem, max_tokens=300)
                all_solutions.append(response)

        # Compute token entropy
        # Collect all unique tokens
        token_counts = {}
        for solution in all_solutions:
            tokens = solution.split()
            for token in set(tokens):  # Unique tokens per solution
                token_counts[token] = token_counts.get(token, 0) + 1

        token_probs = list(token_counts.values())
        token_probs = [p / sum(token_probs) for p in token_probs]

        avg_entropy = entropy(token_probs)
        return avg_entropy, avg_entropy  # Placeholder for variant_entropy

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response."""
        # Find numerical answer or final statement
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        return numbers[-1] if numbers else response.split()[-1]
```

### 4. Track and Monitor Entropy Collapse

Monitor diversity metrics:

```python
class EntropyMonitor:
    """Monitor output diversity and entropy metrics."""

    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self.entropy_history = []
        self.pass_k_history = []

    def compute_pass_k(
        self,
        problems: List[str],
        model: "LLM",
        k_values: List[int] = [1, 3, 5],
        num_generations: int = 32
    ) -> Dict[int, float]:
        """
        Compute Pass@k metric (k solutions needed for one correct).
        """
        pass_k_scores = {k: 0.0 for k in k_values}

        for problem in problems:
            # Generate multiple solutions
            solutions = []
            for _ in range(num_generations):
                response = model.generate(problem, max_tokens=500)
                solutions.append(response)

            # Check which have correct answers
            correct_mask = [self._verify_solution(s, problem) for s in solutions]

            # Compute Pass@k
            for k in k_values:
                if any(correct_mask[:k]):
                    pass_k_scores[k] += 1.0 / len(problems)

        return pass_k_scores

    def compute_output_entropy(
        self,
        problems: List[str],
        model: "LLM",
        num_generations: int = 8
    ) -> float:
        """
        Compute Shannon entropy of outputs.

        High entropy = diverse outputs
        Low entropy = repetitive outputs
        """
        from scipy.stats import entropy as scipy_entropy

        all_solutions = []
        for problem in problems:
            for _ in range(num_generations):
                response = model.generate(problem, max_tokens=300)
                all_solutions.append(response)

        # Tokenize and compute entropy over token vocabulary
        token_freq = {}
        for solution in all_solutions:
            for token in set(solution.split()):
                token_freq[token] = token_freq.get(token, 0) + 1

        probs = list(token_freq.values())
        probs = [p / sum(probs) for p in probs]

        return scipy_entropy(probs)

    def check_entropy_collapse(self, current_entropy: float) -> bool:
        """Detect if entropy is dropping (collapse pattern)."""
        if len(self.entropy_history) < 2:
            return False

        recent_entropy = self.entropy_history[-1]
        entropy_drop = recent_entropy - current_entropy

        # Collapse threshold: >10% entropy drop
        return entropy_drop > 0.1 * recent_entropy

    def _verify_solution(self, solution: str, problem: str) -> bool:
        """Verify if solution is correct."""
        # In practice: use answer verification
        pass
```

## Practical Guidance

### When to Use SvS

- Mathematical reasoning benchmarks (AIME, high school math)
- Tasks with clear correct/incorrect answers
- Scenarios where output diversity matters (Pass@k evaluation)
- Self-play learning scenarios
- Models where entropy collapse is observed

### When NOT to Use

- Creative generation (poetry, stories)
- Tasks without canonical correct answers
- Low-diversity tasks where Pass@1 is sole metric
- Real-time inference (variant generation adds latency)

### Key Hyperparameters

- **num_variants_per_solution**: 2-5 (more = better coverage)
- **variant_generation_type**: "substitution" recommended for math
- **entropy_threshold**: 10-20% drop before intervention
- **self_play_iterations**: 3-10 per model size
- **temperature_variation**: 0.7-1.0 range for diversity

### Performance Expectations

- AIME24 Improvement: +18.3%
- AIME25 Improvement: +22.8%
- Entropy Preservation: 80-95% of baseline diversity
- Pass@k Ceiling: Sustained rather than degrading
- Model Scales: Works across 3B-32B parameter models

## Reference

Researchers. (2024). Beyond Pass@1: Self-Play with Variational Problem Synthesis. arXiv preprint arXiv:2508.14029.
