---
name: frugal-reasoning-short-math-solutions
title: "Shorter but not Worse: Frugal Reasoning via Easy Samples in Math RLVR"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.01937"
keywords: [Efficient Reasoning, Length Control, Reinforcement Learning, Mathematical Problem Solving]
description: "Achieve emergent brevity in reasoning by retaining and up-weighting easy problems during RL training, implicitly regularizing solution length without explicit penalties while maintaining accuracy on hard problems."
---

# Title: Train Efficient Reasoners by Including Easy Problems During RL

Standard RLVR filters out easy problems to focus on challenging ones. This creates a problem: models learn that harder problems require longer solutions, so they generate unnecessarily verbose reasoning even on solvable tasks. Frugal reasoning reverses this: by retaining moderately easy problems and up-weighting them, the model learns compact solution patterns. The result is "emergent brevity"—solutions nearly twice as short without explicit length penalties, while maintaining baseline accuracy.

The mechanism is implicit length regularization through training distribution bias.

## Core Concept

**Implicit Length Regularization via Easy Sample Up-Weighting**:
- **Data Curriculum**: Include easy, medium, and hard problems (don't filter easy)
- **Weighting Strategy**: Up-weight easy/medium problems in RL loss
- **Emergent Brevity**: Model learns efficiency naturally without explicit length penalties
- **Two-Stage Training**: Stage 1 learns brevity, Stage 2 improves on harder problems
- **Token Budget**: Maintain fixed context limit (16K tokens typical)

## Architecture Overview

- **Problem Difficulty Estimation**: Automatic or heuristic classification
- **Curriculum Weighting**: Exponential down-weighting of hard problems
- **GRPO Optimization**: Group Relative Policy Optimization on mixed-difficulty data
- **Token Budget Constraint**: Enforce maximum sequence length
- **Evaluation**: Accuracy on hard problems + length efficiency

## Implementation Steps

**1. Classify Problem Difficulty**

Estimate which problems are easy vs. hard.

```python
class ProblemDifficultyClassifier:
    def __init__(self):
        # Simple heuristics for difficulty estimation
        self.difficulty_features = {
            'num_digits': lambda x: x.count(str(i)) for i in range(10),
            'operators': lambda x: sum(x.count(op) for op in ['+', '-', '*', '/', '%']),
            'parentheses': lambda x: x.count('('),
            'text_length': lambda x: len(x.split())
        }

    def estimate_difficulty(self, problem_text, solution_length=None):
        """Estimate if problem is easy or hard"""
        features = {
            'text_len': len(problem_text.split()),
            'num_operators': sum(problem_text.count(op) for op in ['+', '-', '*', '/', '%']),
            'has_fractions': '/' in problem_text,
            'has_equations': '=' in problem_text
        }

        # Simple scoring
        score = (
            features['text_len'] / 20 +
            features['num_operators'] / 3 +
            features['has_fractions'] * 2 +
            features['has_equations'] * 1
        )

        # Normalize to [0, 1]
        difficulty = min(1.0, score / 10)

        # Adjust by solution length if available
        if solution_length:
            # Long solutions suggest hard problems
            difficulty += min(0.5, solution_length / 1000)

        return difficulty

    def categorize_problems(self, problems):
        """Split into easy/medium/hard"""
        difficulties = [self.estimate_difficulty(p) for p in problems]

        easy_threshold = np.percentile(difficulties, 33)
        hard_threshold = np.percentile(difficulties, 67)

        categorized = {
            'easy': [p for p, d in zip(problems, difficulties) if d < easy_threshold],
            'medium': [p for p, d in zip(problems, difficulties) if easy_threshold <= d < hard_threshold],
            'hard': [p for p, d in zip(problems, difficulties) if d >= hard_threshold]
        }

        return categorized
```

**2. Implement Curriculum Weighting**

Up-weight easy/medium problems in RL loss.

```python
def compute_curriculum_weights(difficulty_scores, up_weighting=2.0):
    """Compute importance weights for curriculum"""
    # Invert: easy (low difficulty) get higher weight
    base_weights = 1.0 - np.array(difficulty_scores)

    # Up-weight easy problems
    curriculum_weights = np.where(
        base_weights > 0.5,  # Easy if difficulty < 0.5
        base_weights * up_weighting,  # Up-weight easy
        base_weights  # Normal weight for hard
    )

    # Normalize
    curriculum_weights /= curriculum_weights.mean()

    return curriculum_weights

def compute_grpo_loss_with_curriculum(
    model_outputs, reference_outputs, rewards,
    difficulty_scores, curriculum_upweight=2.0
):
    """GRPO loss weighted by curriculum"""
    # Compute curriculum weights
    weights = compute_curriculum_weights(difficulty_scores, curriculum_upweight)

    # Group by reward quartiles
    sorted_indices = np.argsort(rewards)
    group_size = len(rewards) // 4

    loss = 0
    for quartile in range(4):
        group_start = quartile * group_size
        group_end = (quartile + 1) * group_size
        group_indices = sorted_indices[group_start:group_end]

        group_rewards = torch.tensor([rewards[i] for i in group_indices])
        group_weights = torch.tensor([weights[i] for i in group_indices])

        # Higher quartiles are positive examples
        if quartile >= 2:
            # Positive group: maximize likelihood
            group_loss = -torch.log(torch.tensor([model_outputs[i] for i in group_indices]) + 1e-8)
        else:
            # Negative group: minimize likelihood
            group_loss = torch.log(torch.tensor([reference_outputs[i] for i in group_indices]) + 1e-8)

        # Weight by curriculum
        weighted_loss = (group_loss * group_weights).mean()
        loss += weighted_loss

    return loss / 4
```

**3. Implement Two-Stage Training**

Stage 1: Learn brevity via easy samples. Stage 2: Improve on hard problems.

```python
def two_stage_frugal_training(model, dataset, token_budget=16000):
    """Train with frugal curriculum"""
    classifier = ProblemDifficultyClassifier()

    # Categorize problems
    categorized = classifier.categorize_problems([d['problem'] for d in dataset])

    # Stage 1: Learn brevity with heavy easy-sample weighting
    print("Stage 1: Learning brevity via easy samples...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    stage1_data = categorized['easy'] + categorized['medium']  # Mix easy and medium
    stage1_weights = compute_curriculum_weights(
        [classifier.estimate_difficulty(p) for p in stage1_data],
        up_weighting=3.0  # Heavy up-weighting for brevity
    )

    for epoch in range(3):
        for problem, weight in zip(stage1_data, stage1_weights):
            # Generate solution with token budget
            solution = model.generate(problem, max_tokens=token_budget)

            # Reward: correctness + brevity
            correct = verify_solution(solution, problem)
            length_efficiency = 1.0 / (1.0 + len(solution.split()) / 100)
            reward = 0.8 * correct + 0.2 * length_efficiency

            # Loss weighted by curriculum
            loss = -torch.log(torch.tensor(reward)) * weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Stage 2: Improve on hard problems with learned brevity
    print("Stage 2: Improving on hard problems...")
    stage2_data = categorized['hard']
    stage2_weights = compute_curriculum_weights(
        [classifier.estimate_difficulty(p) for p in stage2_data],
        up_weighting=1.0  # Normal weighting for hard problems
    )

    for epoch in range(10):
        for problem, weight in zip(stage2_data, stage2_weights):
            # Generate with token budget (model has learned brevity)
            solution = model.generate(problem, max_tokens=token_budget)

            # Reward: emphasis on correctness
            correct = verify_solution(solution, problem)
            length_efficiency = 1.0 / (1.0 + len(solution.split()) / 100)
            reward = 0.95 * correct + 0.05 * length_efficiency

            loss = -torch.log(torch.tensor(reward)) * weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

**4. Evaluate Emergent Brevity**

Measure length and accuracy trade-off.

```python
def evaluate_frugal_reasoning(model, test_problems, ground_truth_answers):
    """Evaluate model on length and accuracy"""
    results = {
        'accuracies': [],
        'solution_lengths': [],
        'tokens_per_problem': []
    }

    for problem, gt_answer in zip(test_problems, ground_truth_answers):
        # Generate solution
        solution = model.generate(problem, max_tokens=16000)

        # Check correctness
        predicted_answer = extract_answer(solution)
        correct = predicted_answer == gt_answer

        results['accuracies'].append(correct)
        results['solution_lengths'].append(len(solution.split()))
        results['tokens_per_problem'].append(len(solution.split()) * 1.3)  # Token estimate

    # Compute metrics
    accuracy = np.mean(results['accuracies'])
    avg_length = np.mean(results['solution_lengths'])
    efficiency = np.mean(results['tokens_per_problem'])

    print(f"Accuracy: {accuracy:.1%}")
    print(f"Avg solution length: {avg_length:.0f} words")
    print(f"Token efficiency: {efficiency:.0f} tokens/problem")

    return results
```

## Practical Guidance

**When to Use**:
- Mathematical reasoning where solution length correlates with problem hardness
- Settings with token budgets or latency constraints
- Training reasoning models post-SFT

**Hyperparameters**:
- curriculum_upweight: 2-3 (how much to favor easy samples)
- token_budget: 16K typical for math, adjust for domain
- easy_problem_ratio: 0.4-0.6 of training set

**When NOT to Use**:
- Domains where complexity doesn't map to problem difficulty
- Models without clear solution length patterns
- Settings requiring verbose reasoning for auditing

**Pitfalls**:
- **Oversimplification**: Model learns to give short wrong answers; mitigate with correctness rewards
- **Difficulty estimation failure**: Heuristic difficulty estimates may be inaccurate; validate
- **Stage 1 undertraining**: If stage 1 doesn't converge to brevity, stage 2 won't benefit

**Key Insight**: The mechanism works because easy samples teach "fast solution patterns." Once the model learns to solve easy problems quickly, these patterns transfer to hard problems. No explicit length penalty needed.

## Reference

arXiv: https://arxiv.org/abs/2511.01937
