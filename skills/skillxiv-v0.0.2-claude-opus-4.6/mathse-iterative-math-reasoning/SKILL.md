---
name: mathse-iterative-math-reasoning
title: "MathSE: Improving Multimodal Mathematical Reasoning via Self-Evolving Reflection"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.06805"
keywords: [Mathematical Reasoning, Self-Improvement, Iterative Refinement, Outcome Reward Models, Multimodal Learning]
description: "Improve multimodal mathematical reasoning through iterative reflection cycles where an outcome reward model provides feedback on reasoning quality, and correct solutions are incorporated back into training—enabling continuous model adaptation beyond static datasets."
---

# Iteratively Improve Mathematical Reasoning Through Self-Reflection

Traditional mathematical training relies on static datasets of teacher-generated solutions, which capture only fixed reasoning patterns. MathSE enables continuous improvement through iterative refinement: models generate solutions, an outcome reward model (ORM) evaluates reasoning quality, and successful reasoning paths are fed back into training. This creates a virtuous cycle where the model adapts to progressively more difficult problems.

By treating mathematical reasoning as self-evolving rather than static, models develop robust problem-solving that generalizes beyond training distribution.

## Core Concept

MathSE implements a closed-loop learning cycle:

1. **Inference** - Model generates solutions to problems (including harder ones beyond training)
2. **Reflection** - Outcome Reward Model rates solution quality and reasoning soundness
3. **Refinement** - Correct reasoning paths from this iteration feed back into training
4. **Iteration** - Process repeats, exposing model to progressively harder problems

This approach contrasts sharply with static datasets which distill only teacher reasoning patterns. Through self-evolution, the model continuously discovers new solution strategies and generalizes to novel problem structures.

## Architecture Overview

- **Base Model**: Multimodal model (vision + text) for mathematical reasoning
- **Outcome Reward Model (ORM)**: Evaluates correctness and reasoning quality
- **Solution Generator**: Creates candidate solutions for problems
- **Trajectory Replay Buffer**: Stores successful solution trajectories
- **Iterative Training Loop**: Fine-tunes model on newly discovered good solutions
- **Problem Curriculum**: Gradually increases difficulty based on model performance

## Implementation Steps

**Step 1: Outcome Reward Model (ORM)**

Train a model to evaluate solution quality and identify sound reasoning.

```python
import torch
import torch.nn as nn
from typing import Tuple

class OutcomeRewardModel(nn.Module):
    """
    Evaluates quality and correctness of mathematical solutions.
    """

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 1024):
        """
        Args:
            embedding_dim: Dimension of problem/solution embeddings
            hidden_dim: Hidden dimension for reward network
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Problem encoder (text)
        self.problem_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Solution encoder (text)
        self.solution_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Reward head: outputs score [0, 1]
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Reasoning quality head: evaluates logical soundness
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, problem_embedding: torch.Tensor,
               solution_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate solution for given problem.

        Args:
            problem_embedding: Encoded problem [batch, embedding_dim]
            solution_embedding: Encoded solution [batch, embedding_dim]

        Returns:
            correctness_score: Expected correctness [batch, 1]
            quality_score: Reasoning quality [batch, 1]
        """
        # Encode
        problem_feat = self.problem_encoder(problem_embedding)
        solution_feat = self.solution_encoder(solution_embedding)

        # Combine features
        combined = torch.cat([problem_feat, solution_feat], dim=1)

        # Predict scores
        correctness = self.reward_head(combined)
        quality = self.quality_head(combined)

        return correctness, quality

def train_orm(orm: OutcomeRewardModel, train_pairs, num_epochs: int = 10):
    """
    Train outcome reward model on labeled solution pairs.

    Args:
        orm: OutcomeRewardModel instance
        train_pairs: List of (problem, solution, label) tuples
                    where label is {correct: bool, quality: float}
        num_epochs: Training epochs
    """
    optimizer = torch.optim.Adam(orm.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        total_loss = 0

        for problem_emb, solution_emb, label in train_pairs:
            # Forward pass
            correctness, quality = orm(problem_emb.unsqueeze(0),
                                      solution_emb.unsqueeze(0))

            # Loss on correctness
            correct_target = torch.tensor([[1.0 if label['correct'] else 0.0]])
            loss = criterion(correctness, correct_target)

            # Loss on reasoning quality
            quality_target = torch.tensor([[label.get('quality', 0.5)]])
            loss += 0.5 * criterion(quality, quality_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"ORM Epoch {epoch}: Loss {total_loss / len(train_pairs):.4f}")
```

**Step 2: Solution Generation and Reflection**

Generate solutions and reflect on their correctness.

```python
class SolutionGenerator:
    """
    Generates mathematical solutions and evaluates them.
    """

    def __init__(self, model, orm: OutcomeRewardModel, max_tokens: int = 1024):
        """
        Args:
            model: Multimodal model for solution generation
            orm: Outcome Reward Model for evaluation
            max_tokens: Maximum tokens in generated solution
        """
        self.model = model
        self.orm = orm
        self.max_tokens = max_tokens

    def generate_solution(self, problem: str, num_attempts: int = 1) -> List[Dict]:
        """
        Generate multiple solution attempts for a problem.

        Args:
            problem: Problem description
            num_attempts: Number of solution attempts

        Returns:
            solutions: List of {solution_text, reasoning, score}
        """
        solutions = []

        for attempt in range(num_attempts):
            # Generate solution with sampling for diversity
            prompt = f"""Solve this mathematical problem step-by-step:

{problem}

Provide a detailed solution with all intermediate steps:"""

            solution_text = self.model.generate(
                prompt, max_tokens=self.max_tokens,
                temperature=0.7 if attempt > 0 else 0.5  # Deterministic first attempt
            )

            solutions.append({
                'solution_text': solution_text,
                'attempt': attempt
            })

        return solutions

    def evaluate_solutions(self, problem: str, solutions: List[Dict],
                         ground_truth: str = None) -> List[Dict]:
        """
        Evaluate solutions using ORM and ground truth comparison.

        Args:
            problem: Original problem
            solutions: Generated solutions
            ground_truth: Correct answer (if available)

        Returns:
            evaluated: Solutions with scores and labels
        """
        # Encode problem once
        problem_embedding = self.model.encode(problem)

        evaluated = []

        for solution in solutions:
            # Encode solution
            solution_embedding = self.model.encode(solution['solution_text'])

            # Get ORM scores
            with torch.no_grad():
                correctness, quality = self.orm(
                    problem_embedding.unsqueeze(0),
                    solution_embedding.unsqueeze(0)
                )

            correctness_score = correctness.item()
            quality_score = quality.item()

            # Verify against ground truth if available
            is_correct = False
            if ground_truth:
                is_correct = self._check_correctness(
                    solution['solution_text'], ground_truth
                )

            evaluated.append({
                **solution,
                'orm_correctness': correctness_score,
                'orm_quality': quality_score,
                'verified_correct': is_correct,
                'combined_score': 0.7 * correctness_score + 0.3 * quality_score
            })

        return evaluated

    def _check_correctness(self, solution_text: str, ground_truth: str) -> bool:
        """Check if solution matches ground truth."""
        # Extract final answer from solution
        lines = solution_text.split('\n')
        final_answer = lines[-1].strip() if lines else ''

        # Simple matching
        if final_answer == ground_truth:
            return True

        # Numeric matching (if both are numbers)
        try:
            sol_num = float(final_answer)
            truth_num = float(ground_truth)
            return abs(sol_num - truth_num) < 1e-6
        except:
            return False
```

**Step 3: Trajectory Replay Buffer**

Maintain collection of successful solutions for replay during training.

```python
from collections import deque
import json

class TrajectoryBuffer:
    """
    Stores successful solution trajectories for iterative training.
    """

    def __init__(self, capacity: int = 10000, quality_threshold: float = 0.7):
        """
        Args:
            capacity: Maximum stored trajectories
            quality_threshold: Minimum quality to store
        """
        self.buffer = deque(maxlen=capacity)
        self.quality_threshold = quality_threshold

    def add_trajectory(self, problem: str, solution: str, score: float,
                      is_correct: bool, difficulty: str = 'medium'):
        """
        Add successful trajectory to buffer.

        Args:
            problem: Problem description
            solution: Complete solution text
            score: ORM-computed quality score
            is_correct: Whether answer is verified correct
            difficulty: Problem difficulty estimate
        """
        if score >= self.quality_threshold or is_correct:
            trajectory = {
                'problem': problem,
                'solution': solution,
                'score': score,
                'correct': is_correct,
                'difficulty': difficulty,
                'timestamp': time.time()
            }
            self.buffer.append(trajectory)

    def sample_batch(self, batch_size: int) -> List[Dict]:
        """
        Sample trajectory batch, emphasizing recent and high-quality solutions.

        Args:
            batch_size: Batch size

        Returns:
            batch: Sampled trajectories
        """
        if len(self.buffer) == 0:
            return []

        # Weight by recency and quality
        trajectories = list(self.buffer)
        weights = []

        now = time.time()
        for traj in trajectories:
            # Recency factor: newer solutions weighted higher
            age = now - traj['timestamp']
            recency = 1.0 / (1.0 + age / 3600)  # Half-life: 1 hour

            # Quality factor
            quality = traj['score']
            if traj['correct']:
                quality = 1.0

            weight = recency * quality
            weights.append(weight)

        # Sample according to weights
        import random
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        batch = random.choices(trajectories, weights=normalized_weights,
                              k=min(batch_size, len(trajectories)))

        return batch

    def get_statistics(self) -> Dict:
        """Compute buffer statistics."""
        if len(self.buffer) == 0:
            return {'size': 0}

        trajectories = list(self.buffer)
        correct_count = sum(1 for t in trajectories if t['correct'])

        return {
            'size': len(trajectories),
            'correct_count': correct_count,
            'correct_ratio': correct_count / len(trajectories),
            'avg_score': sum(t['score'] for t in trajectories) / len(trajectories)
        }
```

**Step 4: Iterative Training Loop**

Implement the main self-evolving training cycle.

```python
def self_evolving_training(base_model, orm: OutcomeRewardModel,
                          problem_pool: List[str], num_iterations: int = 5):
    """
    Main training loop for mathematical reasoning self-evolution.

    Args:
        base_model: Multimodal model to train
        orm: Trained outcome reward model
        problem_pool: Pool of problems to solve
        num_iterations: Number of self-evolution iterations
    """
    generator = SolutionGenerator(base_model, orm)
    buffer = TrajectoryBuffer(capacity=5000)

    for iteration in range(num_iterations):
        print(f"\n=== Self-Evolution Iteration {iteration + 1} ===")

        # Sample problems (mix of training and harder problems)
        num_sample = int(len(problem_pool) * 0.7)
        sampled_problems = random.sample(problem_pool, num_sample)

        # Add harder problems beyond training distribution
        if iteration > 0:
            harder_problems = problem_pool[num_sample:]
            sampled_problems.extend(harder_problems[:len(sampled_problems) // 4])

        # Generate and evaluate solutions
        for problem in sampled_problems:
            # Generate solution attempts
            solutions = generator.generate_solution(problem, num_attempts=2)

            # Evaluate with ORM
            evaluated = generator.evaluate_solutions(problem, solutions)

            # Store best solutions
            best = max(evaluated, key=lambda x: x['combined_score'])
            buffer.add_trajectory(
                problem, best['solution_text'],
                best['combined_score'], best['verified_correct']
            )

        # Print iteration statistics
        stats = buffer.get_statistics()
        print(f"Buffer size: {stats['size']}")
        print(f"Correct ratio: {stats['correct_ratio']:.2%}")
        print(f"Avg score: {stats['avg_score']:.3f}")

        # Fine-tune on collected trajectories
        print("Fine-tuning on successful trajectories...")
        batch = buffer.sample_batch(batch_size=32)

        optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-5)

        for epoch in range(2):
            total_loss = 0

            for trajectory in batch:
                prompt = f"Problem: {trajectory['problem']}\n\nSolution:"
                logits = base_model.forward(prompt)

                # Supervise on ground truth solution
                target_ids = base_model.tokenize(trajectory['solution'])
                loss = compute_language_loss(logits, target_ids)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"  Epoch {epoch}: Loss {total_loss / len(batch):.4f}")

    return base_model
```

## Practical Guidance

**When to Use MathSE:**
- Mathematical problem-solving tasks requiring generalization
- Scenarios where oracle models are expensive (use self-generated trajectories)
- Tasks benefiting from iterative refinement over training distribution

**When NOT to Use:**
- Tasks with no clear correctness signal (needs ORM or ground truth)
- Real-time scenarios (iterative training is time-consuming)
- Domains where teacher-generated data is already comprehensive

**Hyperparameters and Configuration:**
- ORM quality threshold: 0.6-0.8 (balance quality with buffer diversity)
- Buffer capacity: 5000-10000 (larger for more diverse sampling)
- Iteration count: 3-5 (diminishing returns after few iterations)
- Sample temperature: 0.5 first attempt, 0.7+ for diversity

**Pitfalls to Avoid:**
1. **Incorrect labeling** - ORM may mislabel solutions; validate with ground truth periodically
2. **Distribution drift** - Self-generated solutions may diverge from true distribution; check diversity
3. **Overfitting to ORM** - Model optimizes for ORM score, not correctness; weight ground truth heavily
4. **Positive feedback loops** - Easy problems accumulate; ensure problem curriculum increases difficulty

---

Reference: https://arxiv.org/abs/2511.06805
