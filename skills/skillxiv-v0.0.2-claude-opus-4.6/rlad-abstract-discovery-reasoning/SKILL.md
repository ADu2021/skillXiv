---
name: rlad-abstract-discovery-reasoning
title: "RLAD: Learning to Discover Abstractions via Reasoning RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.02263
keywords: [reasoning-abstraction, RLVR, strategy-discovery, curriculum-learning]
description: "Train reasoning models to discover diverse solution strategies through two-player RL that jointly optimizes abstraction generation and solution derivation. Use when exploring multiple reasoning approaches is preferable to single-strategy depth."
---

# RLAD: Learning to Discover Abstractions via Reasoning RL

RLAD introduces a two-player RL framework where abstraction generators and solution generators co-evolve. This addresses a fundamental inefficiency: RL often optimizes for solution depth rather than strategy diversity. By explicitly rewarding abstraction diversity, models explore broader solution landscapes.

## Core Architecture

- **Reasoning abstractions**: Concise natural language descriptions of procedural/factual knowledge
- **Two-player dynamics**: Generator vs. Verifier with competing objectives
- **Abstraction-guided generation**: Solutions generated conditioned on discovered abstractions
- **Modified reward system**: Prevents failure modes (e.g., gaming with trivial abstractions)
- **Multi-domain generalization**: Works across math, coding, and diverse reasoning domains

## Implementation Steps

Setup two-player RL framework for abstraction discovery:

```python
# Initialize RLAD trainer with abstraction and solution components
from rlad import AbstractionRL, SolutionRL, TwoPlayerOptimizer

# Create abstraction generator
abstraction_generator = AbstractionRL(
    model="gpt-4-mini",  # or stronger model for warm-starting
    max_abstraction_length=256,
    abstraction_style="knowledge_based"
)

# Create solution generator conditioned on abstractions
solution_generator = SolutionRL(
    model="your_reasoning_llm",
    conditioning="abstraction_aware",
    max_solution_steps=50
)

# Setup two-player optimizer
optimizer = TwoPlayerOptimizer(
    abstraction_generator=abstraction_generator,
    solution_generator=solution_generator,
    reward_mode="abstraction_diversity"
)
```

Execute two-player RL training:

```python
# Training loop optimizing both abstractors and solvers
for epoch in range(num_epochs):
    for batch in training_data:
        problem = batch["problem"]
        ground_truth = batch["solution"]

        # Player 1: Generate diverse abstractions
        abstractions = abstraction_generator.sample(
            problem=problem,
            num_samples=5,
            temperature=1.0,  # high diversity
            beam_size=None    # sample-based, not beam search
        )

        # Player 2: Solve conditioned on each abstraction
        solutions = []
        rewards = []

        for abstraction in abstractions:
            solution = solution_generator.generate(
                problem=problem,
                abstraction=abstraction,
                max_length=512,
                temperature=0.7
            )
            solutions.append(solution)

            # Compute solution reward
            is_correct = verify_solution(solution, ground_truth)
            reward = 1.0 if is_correct else 0.0
            rewards.append(reward)

        # Compute abstraction diversity reward
        abstraction_reward = compute_diversity_score(abstractions)

        # Modified reward to prevent gaming
        # Penalize abstract-only solutions that don't produce correct answers
        for i, abstraction in enumerate(abstractions):
            if rewards[i] == 0:
                abstraction_reward *= 0.5  # penalty for failed solutions

        # Update policies with advantage-based RLVR
        abstraction_loss = optimizer.compute_abstraction_loss(
            abstractions=abstractions,
            diversity_reward=abstraction_reward,
            solution_success=rewards
        )

        solution_loss = optimizer.compute_solution_loss(
            solutions=solutions,
            targets=ground_truth,
            abstractions=abstractions
        )

        # Joint optimization
        (abstraction_loss + solution_loss).backward()
        optimizer.step()
```

## Practical Guidance

**When to use RLAD:**
- Complex reasoning tasks with multiple valid solution approaches
- Scenarios where solution diversity improves robustness
- Testing and evaluation requiring different reasoning strategies
- Models that tend to converge prematurely to single strategies

**When NOT to use:**
- Simple deterministic problems with unique solutions
- Real-time systems where diverse strategies add latency overhead
- Domains where single-strategy depth beats breadth (e.g., chess-like games)
- Low-compute settings (two-player optimization more expensive than single-player)

**Hyperparameter considerations:**
- **Abstraction model strength**: Use strongest available for warm-starting; gpt-4-mini minimum
- **Num abstraction samples (5)**: Increase to 8-10 for broader exploration; decrease to 3 for speed
- **Temperature (1.0 for abstractions)**: Keep high; controls diversity
- **Temperature (0.7 for solutions)**: Moderate; balance exploration vs. exploitation
- **Diversity weight**: Adjust reward weighting for abstraction vs. solution success
- **Epoch count**: 3-5 epochs usually sufficient; 10 for thorough exploration

## Multi-Domain Performance

Consistent improvements across domains:
- **Mathematics (AIME, AMC)**: Better coverage of problem classes
- **Coding**: Multiple algorithmic approaches discovered
- **DeepScaleR**: Complex reasoning with diverse strategies
- **37 diverse benchmarks**: Generalizes across task types

## Key Insight

The critical realization: "allocating more compute to abstraction generation rather than solution generation" yields consistent gains. Test-time compute distribution matters as much as pure volume. Figure 5 shows:
- Dedicated abstraction-only budget: +3-7% improvement
- vs. equal budget split between abstraction and solution
- vs. all budget on solution generation

## Failure Mode Prevention

The modified reward system addresses three failure modes:
1. **Trivial abstractions**: Solutions that don't leverage abstractions still rewarded; prevent with solution verification
2. **Incorrect solutions**: Abstractions for failed solutions penalized by 50%
3. **Repetitive abstractions**: Diversity metric prevents degenerate solutions

## References

Extends prior work on abstraction learning, curriculum design, and multi-agent RL.
