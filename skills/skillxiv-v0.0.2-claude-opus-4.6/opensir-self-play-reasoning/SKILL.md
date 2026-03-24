---
name: opensir-self-play-reasoning
title: "OpenSIR: Open-Ended Self-Improving Reasoner"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.00602"
keywords: [Self-Play, Autonomous Learning, Mathematical Reasoning, Curriculum Learning, Problem Generation]
description: "Enable open-ended mathematical learning through co-evolutionary teacher-student self-play where a single model alternates roles to generate progressively harder problems and solve them, creating a virtuous cycle without external data."
---

# Title: Bootstrap Mathematical Reasoning From Seed Problems Without Annotation

Traditional supervised learning requires massive annotated datasets. OpenSIR achieves autonomous improvement through **co-evolutionary self-play**: a single model acts as both problem-setter (teacher) and problem-solver (student), generating novel problems calibrated to current capability and solving them. The framework requires only a small seed of initial problems, then bootstraps to arbitrary complexity.

The key is maintaining optimal difficulty: problems easy enough to learn from but hard enough to improve capability.

## Core Concept

**Co-Evolutionary Teacher-Student Self-Play**:
- **Single Policy, Two Roles**: Model alternates between generating problems (teacher) and solving them (student)
- **Difficulty Calibration**: Teacher learns to generate problems at ~70% solve rate (optimal difficulty)
- **Diversity Rewards**: Encourage distinct mathematical concepts vs. repeating same patterns
- **No External Supervision**: Learn entirely from self-generated data
- **Emergent Curriculum**: Naturally progress from arithmetic to advanced mathematics

## Architecture Overview

- **Problem Generator (Teacher)**: Produces novel math problems conditioned on difficulty level and concept diversity
- **Problem Solver (Student)**: Generates step-by-step solutions via reasoning
- **Capability Estimator**: Tracks model performance across problem categories
- **Curriculum Controller**: Adjusts teacher difficulty based on student performance
- **Scoring System**: Rewards for both solution accuracy and problem novelty

## Implementation Steps

**1. Implement Teacher (Problem Generation)**

Design the teacher to generate diverse, well-formed mathematical problems.

```python
class MathTeacher(nn.Module):
    def __init__(self, model, difficulty_levels=[1, 2, 3, 4, 5]):
        self.model = model
        self.difficulty_levels = difficulty_levels
        self.concept_tracker = defaultdict(int)

    def generate_problem(self, concepts=None, difficulty=None):
        if difficulty is None:
            difficulty = np.random.choice(self.difficulty_levels)

        # Prompt teacher to generate problem
        prompt = f"""Generate a {difficulty}/5 difficulty math problem.
        Previous problems covered: {list(self.concept_tracker.keys())}
        Please avoid repeating the same concepts repeatedly.
        Problem:"""

        problem = self.model.generate(prompt, max_tokens=200)

        # Track concepts used
        if concepts:
            self.concept_tracker[concepts] += 1

        return problem, difficulty

    def sample_problems(self, num_problems=10):
        # Generate diverse problems across difficulties
        problems = []
        for _ in range(num_problems):
            # Vary difficulty
            difficulty = np.random.choice(self.difficulty_levels)
            problem, _ = self.generate_problem(difficulty=difficulty)
            problems.append(problem)
        return problems
```

**2. Implement Student (Problem Solver)**

Create the solver that learns from generated problems.

```python
class MathStudent(nn.Module):
    def __init__(self, model):
        self.model = model
        self.performance_tracker = defaultdict(lambda: {'solved': 0, 'attempts': 0})

    def solve_problem(self, problem, return_confidence=False):
        # Prompt model to solve problem with step-by-step reasoning
        prompt = f"""Solve this problem step-by-step:
        {problem}

        Solution:"""

        solution = self.model.generate(prompt, max_tokens=500)

        # Extract final answer
        answer = self.extract_answer(solution)

        if return_confidence:
            # Get confidence score from model
            confidence_prompt = f"How confident are you in this answer (0-1)? {answer}"
            confidence = float(self.model.generate(confidence_prompt, max_tokens=10))
            return solution, answer, confidence
        return solution, answer

    def extract_answer(self, solution):
        # Parse answer from solution text (problem-dependent)
        # Try multiple extraction strategies
        if 'answer is' in solution:
            return solution.split('answer is')[-1].strip()
        if '=' in solution:
            return solution.split('=')[-1].strip()
        return solution.split()[-1]

    def verify_answer(self, problem, answer):
        # Use ground truth verification or self-consistency
        # For now, simplified
        prompt = f"Is '{answer}' the correct answer to '{problem}'? Yes/No:"
        response = self.model.generate(prompt, max_tokens=10)
        return 'yes' in response.lower()

    def track_performance(self, category, solved):
        self.performance_tracker[category]['attempts'] += 1
        if solved:
            self.performance_tracker[category]['solved'] += 1
```

**3. Implement Co-Evolution Training Loop**

Train both teacher and student through self-play.

```python
def co-evolutionary_training(model, initial_problems, num_iterations=100):
    teacher = MathTeacher(model)
    student = MathStudent(model)

    # Initialize from seed problems
    problem_pool = initial_problems.copy()

    for iteration in range(num_iterations):
        # Teacher phase: generate new problems
        new_problems = teacher.sample_problems(num_problems=5)
        problem_pool.extend(new_problems)

        # Student phase: solve problems from pool
        solve_correct = 0
        solve_total = 0

        for problem in problem_pool[-50:]:  # Recent problems
            solution, answer, confidence = student.solve_problem(
                problem, return_confidence=True
            )
            verified = student.verify_answer(problem, answer)

            solve_total += 1
            if verified:
                solve_correct += 1

            # Track performance
            category = extract_category(problem)  # algebra, geometry, etc.
            student.track_performance(category, verified)

        # Compute solve rate
        solve_rate = solve_correct / solve_total if solve_total > 0 else 0

        # Curriculum adjustment
        # If solve_rate > 0.7, increase difficulty
        # If solve_rate < 0.5, decrease difficulty
        if solve_rate > 0.7:
            teacher.difficulty_levels = [d + 0.1 for d in teacher.difficulty_levels]
        elif solve_rate < 0.5:
            teacher.difficulty_levels = [max(1, d - 0.1) for d in teacher.difficulty_levels]

        print(f"Iteration {iteration}: Solve Rate {solve_rate:.2%}, Pool Size {len(problem_pool)}")

        # Periodic reinforcement signal
        if iteration % 10 == 0:
            # Train on successful problem-solution pairs
            successful_pairs = [
                (p, s) for p, s in problem_solution_pairs if verified[p]
            ]
            # Fine-tune on successful patterns
            update_model(model, successful_pairs)
```

**4. Implement Diversity and Difficulty Rewards**

Ensure teacher generates diverse problems while maintaining difficulty calibration.

```python
def compute_teacher_reward(new_problem, problem_pool, solve_rate, category_coverage):
    # Reward 1: Difficulty calibration
    # Want solve rate around 70%
    difficulty_reward = 1.0 - abs(solve_rate - 0.7)

    # Reward 2: Diversity
    # Penalize similar problems
    similarity_to_existing = max(
        cosine_similarity(embed(new_problem), embed(p)) for p in problem_pool[-20:]
    )
    diversity_reward = 1.0 - 0.5 * similarity_to_existing

    # Reward 3: Coverage
    # Encourage unexplored categories
    new_category = extract_category(new_problem)
    coverage_reward = 1.0 / (1.0 + category_coverage[new_category])

    total_reward = 0.6 * difficulty_reward + 0.3 * diversity_reward + 0.1 * coverage_reward
    return total_reward
```

## Practical Guidance

**When to Use**:
- Mathematical reasoning tasks
- Domains where problem generation is straightforward
- Building reasoning capabilities from minimal seed data
- Educational applications requiring progressive difficulty

**Hyperparameters**:
- target_solve_rate: 0.7 (70% correctness for optimal difficulty)
- difficulty_adjustment_step: 0.1 per iteration
- problem_pool_retention: Keep last 50-100 problems for training

**When NOT to Use**:
- Domains where problem generation requires human judgment (e.g., physics)
- Tasks with complex evaluation metrics
- Real-time applications (self-play generation is slow)

**Pitfalls**:
- **Mode collapse**: Teacher converges to generating same problem repeatedly; use diversity rewards
- **Easy problems dominating**: Student solves simple problems but skill doesn't transfer; enforce difficulty growth
- **Verification errors**: If answer verification is wrong, learning goes off-track; use multiple verification strategies

**Key Insight**: Co-evolution only works if teacher and student improve together. If one diverges, the loop breaks. Monitor their relative performance.

## Reference

arXiv: https://arxiv.org/abs/2511.00602
