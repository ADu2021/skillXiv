---
name: spice-corpus-self-play
title: "SPICE: Self-Play In Corpus Environments Improves Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.24684"
keywords: [Self-play, Reasoning, Corpus Grounding, RL, Curriculum Learning]
description: "Enables continuous self-improvement through corpus-grounded self-play. Challenger mines difficult examples from document corpus for Reasoner to solve. External corpus prevents task stagnation that plagues closed-loop self-play. Achieves 8.9% math, 9.8% general reasoning improvements."
---

# SPICE: Corpus-Grounded Self-Play for Reasoning

Self-play RL struggles with improvement plateaus because agents exhaust limited task spaces. SPICE grounds self-play in a real-world document corpus, enabling continuous challenge generation at the frontier of capability.

The corpus provides the rich, near-inexhaustible signal necessary for sustained reasoning improvement.

## Core Concept

Two complementary roles:
- **Challenger**: mines difficult examples from corpus to create tasks
- **Reasoner**: solves Challenger-generated tasks

The Challenger creates automatic curriculum using corpus, addressing the fundamental limitation of closed-loop self-play: limited task diversity.

## Architecture Overview

- Corpus of documents for task mining
- Challenger model: identifies difficult, solvable problems
- Reasoner model: attempts to solve mined problems
- Shared reward signal for co-training

## Implementation Steps

Implement Challenger that mines tasks from corpus:

```python
class ChallengerModel:
    def __init__(self, llm, corpus):
        self.llm = llm
        self.corpus = corpus

    def mine_task(self, reasoner_capability_level):
        """Generate task at boundary of reasoner capability."""
        # Sample documents from corpus
        candidates = self.corpus.sample(num_candidates=100)

        difficulty_scores = []
        for doc in candidates:
            # Extract problem from document
            problem = self.llm.extract_problem(doc)

            # Estimate difficulty relative to reasoner capability
            difficulty = self.estimate_difficulty(problem, reasoner_capability_level)

            difficulty_scores.append((problem, difficulty))

        # Select problems at frontier of capability
        # Not too easy (already solved), not impossible
        target_difficulty = reasoner_capability_level + 0.2
        frontier_problems = [
            p for p, d in difficulty_scores
            if abs(d - target_difficulty) < 0.1
        ]

        return frontier_problems[0] if frontier_problems else candidates[0]

    def estimate_difficulty(self, problem, current_capability):
        """Score problem difficulty."""
        # Use embedding similarity to past problems
        problem_emb = self.llm.embed(problem)

        # Find similar problems reasoner has solved
        similar_solved = []
        for solved_problem in self.reasoner_history:
            similarity = cosine_similarity(
                problem_emb,
                self.llm.embed(solved_problem)
            )
            similar_solved.append(similarity)

        # Difficulty = opposite of average similarity to solved problems
        avg_similarity = sum(similar_solved) / len(similar_solved) if similar_solved else 0
        return 1 - avg_similarity
```

Implement Reasoner training with corpus-generated tasks:

```python
class ReasonerModel:
    def __init__(self, llm):
        self.llm = llm
        self.solved_problems = []

    def solve(self, task):
        """Attempt to solve mined task."""
        solution = self.llm.generate(task, max_tokens=500)
        return solution

    def get_reward(self, task, solution):
        """Verify solution quality."""
        # Use external verifier if available
        # E.g., Python execution, ground truth, heuristics
        return verify_solution(task, solution)

    def train_step(self, task, solution, reward):
        """RL update from corpus-generated task."""
        # Policy gradient update
        log_prob = self.llm.compute_log_prob(task, solution)
        loss = -log_prob * reward

        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track solved problems for difficulty estimation
        if reward > threshold:
            self.solved_problems.append(task)
```

Implement the co-training loop:

```python
def train_spice(challenger, reasoner, corpus, num_rounds=100):
    """Self-play loop with corpus grounding."""
    for round_idx in range(num_rounds):
        # Challenger mines task at frontier
        task = challenger.mine_task(reasoner.capability_level)

        # Reasoner attempts task
        solution = reasoner.solve(task)

        # Evaluate solution
        reward = reasoner.get_reward(task, solution)

        # Train both models
        reasoner.train_step(task, solution, reward)

        # Update Challenger's understanding of reasoner capability
        challenger.update_reasoner_profile(task, reward)

        if round_idx % 100 == 0:
            # Evaluate on held-out benchmarks
            benchmark_score = evaluate_on_benchmark(reasoner)
            print(f"Round {round_idx}: Benchmark = {benchmark_score:.2%}")

        # Corpus prevents task exhaustion
        # Challenger can always mine new problems
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Corpus size | 100K+ documents |
| Difficulty tolerance | ±0.2 relative to capability |
| Solved problem memory | Track last 1000 problems |
| Reward threshold | Task-specific |

**When to use:**
- Reasoning task improvement without external data
- Scenarios with accessible document corpus
- Long-horizon training where diversity matters
- Scaling reasoning capabilities efficiently

**When NOT to use:**
- Tasks with no document corpus
- When explicit ground truth supervision exists (supervised better)
- Real-time systems (mining adds latency)

**Common pitfalls:**
- Corpus too small (task repetition, plateau)
- Difficulty estimation too loose (not at frontier)
- Not tracking solved problems (difficulty unchanged)
- Corpus containing solutions (shortcuts learning)

Reference: [SPICE on arXiv](https://arxiv.org/abs/2510.24684)
