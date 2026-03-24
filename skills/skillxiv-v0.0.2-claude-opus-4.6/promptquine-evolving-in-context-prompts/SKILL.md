---
name: promptquine-evolving-in-context-prompts
title: "Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.17930"
keywords: [PromptOptimization, EvolutionarySearch, InContextLearning, AutomatedPrompting]
description: "Automatically discovers optimal in-context learning prompts through evolutionary token pruning that removes redundant demonstrations to create effective 'gibberish' prompts. Matches state-of-the-art optimization with low-data regimes. Use for automated prompt discovery without manual tuning or human expertise."
---

# PromptQuine: Evolutionary Token Pruning for Optimal In-Context Learning

Conventional wisdom says that good in-context learning prompts must be well-formed, semantically coherent examples that clearly demonstrate the task. PromptQuine challenges this by discovering that removing tokens from coherent demonstrations to create seemingly incoherent sequences often improves performance. This counterintuitive finding suggests language models respond to minimal features rather than full linguistic structure. PromptQuine formalizes this as an evolutionary search problem: given a prompt with n tokens, discover which subset of m tokens maximizes task performance. The result is a framework that discovers effective prompts faster and cheaper than human optimization or prior automatic methods.

The insight is that natural language prompts contain redundant features. Strategic token removal eliminates this redundancy while preserving core task-relevant information that models actually use. This "Partial Context Hypothesis" explains why seemingly garbled prompts work—they contain just enough signal.

## Core Concept

PromptQuine treats prompt optimization as an evolutionary search where the population is binary token masks. Each mask represents which tokens to keep from an original demonstration. Genetic algorithms with regularized evolution search this high-dimensional space, using low-data task performance as the fitness function. The framework includes:

1. **Token Mask Representation**: Binary vector indicating which tokens are retained (1) or pruned (0)
2. **Bit-Flip Mutations**: Only allow 1→0 operations (removing tokens), preventing expansion
3. **Calibration-then-Selection**: Re-rank candidates to reduce overfitting to small validation sets
4. **Tournament Selection**: Probabilistic selection favoring high-fitness masks
5. **Regularized Evolution**: Balance exploration vs. exploitation to avoid premature convergence

The search landscape exhibits surprising properties: sparse pruned prompts often plateau at excellent performance, suggesting models extract sufficient information from minimal tokens.

## Architecture Overview

- **Genetic Algorithm Core**: Population of binary masks evolving through mutation and selection
- **Fitness Function**: Task performance on small validation set (low-data regime)
- **Mutation Strategy**: Token removal (bit-flip 1→0) only
- **Calibration Module**: Re-ranking to mitigate overfitting
- **Tournament Selection**: Fitness-biased probabilistic selection
- **Stopping Criteria**: Convergence when elite fitness plateaus

## Implementation

Evolutionary token pruning search for in-context learning prompts:

```python
import numpy as np
from typing import List, Tuple, Callable
import torch

class TokenMask:
    """Represents a binary mask over prompt tokens."""
    def __init__(self, prompt_tokens: List[str]):
        self.prompt_tokens = prompt_tokens
        self.num_tokens = len(prompt_tokens)
        # Start with all tokens (mask = all 1s)
        self.mask = np.ones(self.num_tokens, dtype=int)

    def apply_mask(self) -> str:
        """Apply mask to tokens, returning pruned prompt."""
        masked_tokens = [
            token for token, keep in zip(self.prompt_tokens, self.mask)
            if keep == 1
        ]
        return ' '.join(masked_tokens)

    def mutate(self) -> 'TokenMask':
        """
        Bit-flip mutation: randomly remove one token (1→0).
        Prevents prompt expansion.
        """
        new_mask = self.TokenMask(self.prompt_tokens)
        new_mask.mask = self.mask.copy()

        # Find indices where token is present (mask==1)
        present_indices = np.where(self.mask == 1)[0]

        if len(present_indices) > 1:
            # Remove one present token randomly
            idx_to_remove = np.random.choice(present_indices)
            new_mask.mask[idx_to_remove] = 0

        return new_mask

    def copy(self) -> 'TokenMask':
        """Create independent copy."""
        new_mask = TokenMask(self.prompt_tokens)
        new_mask.mask = self.mask.copy()
        return new_mask


class PromptQuineEvolutionarySearch:
    """
    Evolutionary algorithm for discovering optimal token pruning patterns.
    Searches 2^n space (n = token count) for high-performance masks.
    """
    def __init__(
        self,
        initial_prompt: str,
        eval_function: Callable[[str], float],
        population_size: int = 128,
        mutation_rate: float = 0.1,
        max_iterations: int = 200
    ):
        """
        Args:
            initial_prompt: Original well-formed in-context example
            eval_function: Takes pruned_prompt → accuracy score
            population_size: Number of masks in population
            mutation_rate: Fraction of population mutated per generation
            max_iterations: Maximum generations before stopping
        """
        self.initial_prompt = initial_prompt
        self.eval_function = eval_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations

        # Parse prompt into tokens
        self.prompt_tokens = initial_prompt.split()

        # Initialize population: all masks start as identity
        self.population = [
            TokenMask(self.prompt_tokens) for _ in range(population_size)
        ]

        # Track elite candidates (best masks seen)
        self.elite = []

    def evaluate_population(self) -> np.ndarray:
        """
        Evaluate fitness (task accuracy) for entire population.
        Returns array of fitness scores.
        """
        fitness = np.zeros(len(self.population))

        for idx, mask in enumerate(self.population):
            pruned_prompt = mask.apply_mask()
            # Evaluate on validation set
            fitness[idx] = self.eval_function(pruned_prompt)

        return fitness

    def calibrate_ranking(self, fitness: np.ndarray, num_evals: int = 1000):
        """
        Re-rank candidates to reduce overfitting.
        For top candidates, re-evaluate on larger held-out set.
        """
        # Sort by fitness
        sorted_indices = np.argsort(-fitness)[:10]  # Top 10

        # Re-evaluate top candidates with more samples
        recalibrated_fitness = fitness.copy()
        for idx in sorted_indices:
            # Run more evaluations
            mask = self.population[idx]
            pruned_prompt = mask.apply_mask()
            robust_fitness = self.eval_function(pruned_prompt)
            recalibrated_fitness[idx] = robust_fitness

        return recalibrated_fitness

    def tournament_selection(self, fitness: np.ndarray) -> int:
        """
        Tournament selection: select best from random subset.
        Probability proportional to fitness.
        """
        tournament_size = 4
        candidates = np.random.choice(
            len(self.population), size=tournament_size, replace=False
        )
        # Return index of best candidate in tournament
        best_in_tournament = candidates[np.argmax(fitness[candidates])]
        return best_in_tournament

    def evolve(self) -> Tuple[str, float]:
        """
        Run evolutionary search for optimal token pruning.

        Returns:
            best_prompt: Pruned prompt with highest validation accuracy
            best_fitness: Corresponding fitness score
        """
        best_fitness_overall = -1.0
        best_mask_overall = None
        plateau_counter = 0
        convergence_threshold = 10  # Stop if no improvement for 10 iterations

        for iteration in range(self.max_iterations):
            # Evaluate population
            fitness = self.evaluate_population()

            # Calibrate rankings to reduce overfitting
            fitness = self.calibrate_ranking(fitness)

            # Track elite
            current_best_idx = np.argmax(fitness)
            current_best_fitness = fitness[current_best_idx]

            if current_best_fitness > best_fitness_overall:
                best_fitness_overall = current_best_fitness
                best_mask_overall = self.population[current_best_idx].copy()
                plateau_counter = 0
            else:
                plateau_counter += 1

            # Stopping criterion: convergence
            if plateau_counter >= convergence_threshold:
                print(f"Converged at iteration {iteration}")
                break

            # Mutation and selection: create next generation
            next_population = []

            # Elitism: keep best individuals
            elite_size = max(1, int(self.population_size * 0.1))
            elite_indices = np.argsort(-fitness)[:elite_size]
            for elite_idx in elite_indices:
                next_population.append(self.population[elite_idx].copy())

            # Fill rest via tournament selection + mutation
            while len(next_population) < self.population_size:
                # Select parent via tournament
                parent_idx = self.tournament_selection(fitness)
                parent = self.population[parent_idx]

                # Mutate: remove tokens
                child = parent.mutate()
                next_population.append(child)

            self.population = next_population

            if iteration % 20 == 0:
                print(
                    f"Iteration {iteration}: Best fitness = {best_fitness_overall:.4f}, "
                    f"Pruned tokens = {np.sum(best_mask_overall.mask)}/{len(self.prompt_tokens)}"
                )

        # Return best discovered prompt
        best_prompt = best_mask_overall.apply_mask()
        return best_prompt, best_fitness_overall


def evaluate_prompt_on_task(pruned_prompt: str, test_samples: List[dict]) -> float:
    """
    Evaluate prompt performance on a specific task.
    Returns accuracy as fitness metric.

    Args:
        pruned_prompt: Token-pruned demonstration
        test_samples: List of {'input': ..., 'label': ...}

    Returns:
        accuracy: Fraction correct predictions
    """
    correct = 0
    total = len(test_samples)

    for sample in test_samples:
        # Build full prompt: pruned_demonstration + input
        full_prompt = pruned_prompt + "\n" + sample['input']

        # Get LLM prediction (simplified)
        prediction = get_llm_response(full_prompt)

        if prediction == sample['label']:
            correct += 1

    accuracy = correct / total
    return accuracy
```

Multi-task evaluation showing prompt transferability:

```python
def evolve_prompts_multimodal(
    tasks: List[dict],
    task_names: List[str],
    population_size: int = 128,
    max_iterations: int = 200
) -> dict:
    """
    Discover optimal prompts across diverse tasks.
    Tests transfer of pruning patterns.

    Args:
        tasks: List of task configs with 'initial_prompt' and 'eval_func'
        task_names: Names of tasks

    Returns:
        results: Dict mapping task_name → (best_prompt, fitness)
    """
    results = {}

    for task, task_name in zip(tasks, task_names):
        print(f"\nOptimizing prompt for {task_name}")

        searcher = PromptQuineEvolutionarySearch(
            initial_prompt=task['initial_prompt'],
            eval_function=task['eval_func'],
            population_size=population_size,
            max_iterations=max_iterations
        )

        best_prompt, best_fitness = searcher.evolve()

        results[task_name] = {
            'prompt': best_prompt,
            'fitness': best_fitness,
            'original_length': len(task['initial_prompt'].split()),
            'pruned_length': len(best_prompt.split()),
            'compression_ratio': len(best_prompt.split()) / len(task['initial_prompt'].split())
        }

        print(f"Task {task_name}:")
        print(f"  Best accuracy: {best_fitness:.4f}")
        print(f"  Pruned prompt length: {len(best_prompt.split())} tokens")
        print(f"  Compression: {results[task_name]['compression_ratio']:.2%}")

    return results
```

## Practical Guidance

| Aspect | Value | Notes |
|--------|-------|-------|
| Search Space | 2^n (n=token count) | Exponential but explorable with evolution |
| Population Size | 128 | Typical setting |
| Convergence | 50-200 iterations | Task-dependent |
| Token Compression | 40-70% | Effective prompts often highly pruned |
| Classification Accuracy | 77.5% avg (1-shot) | Matches or exceeds Promptbreeder |
| Search Time | Minutes | Much faster than human optimization |
| Improvement Over Baseline | +2-8% | Task-dependent |

**When to use:**
- Optimizing in-context learning prompts without manual effort
- Discovering effective prompts in low-data regimes (1-4 shot)
- Tasks where standard prompt engineering struggles
- Understanding what information models actually use from demonstrations
- Automating prompt discovery across many tasks
- Situations where compressed prompts are valuable (API costs, context length)

**When NOT to use:**
- High-data regimes where full fine-tuning is preferable
- Tasks requiring explicit structured reasoning (pruning may remove it)
- Scenarios needing interpretable prompts for humans
- Real-time applications where search latency matters
- If you have domain expertise to write better prompts manually
- Few-shot learning where example quality is critical

**Common pitfalls:**
- Overfitting to small validation set during search (calibration-then-selection essential)
- Population size too small, insufficient exploration
- Mutation rate too high causing random search behavior
- Not tracking elite separately (early convergence to local optima)
- Evaluating on same samples used for search (use proper holdout)
- Assuming pruning patterns transfer across very different tasks
- Over-pruning valid semantic structure while chasing compression

## Reference

"Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective", 2025. [arxiv.org/abs/2506.17930](https://arxiv.org/abs/2506.17930)
