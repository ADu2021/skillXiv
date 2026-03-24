---
name: rlve-scaling-rl-with-adaptive-environments
title: "RLVE: Scaling Up RL for LMs with Adaptive Verifiable Environments"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.07317"
keywords: [Reinforcement Learning, Adaptive Difficulty, Curriculum Learning, Verifiable Environments, Language Models]
description: "Scale RL training for language models by automatically adapting problem difficulty to match model capabilities using verifiable reward environments—achieving 3.37% absolute improvement on reasoning benchmarks through environment diversity rather than compute scaling alone."
---

# Scale LLM Reasoning Through Adaptive Difficulty Environments

Training language models with reinforcement learning on fixed problem distributions fails to drive meaningful improvement—learning signals vanish as problems become either trivially easy or impossibly hard relative to the model's current capabilities. RLVE solves this through adaptive verifiable environments that algorithmically generate problems calibrated to the model's current performance level.

Rather than investing more compute in static training setups, RLVE multiplies training effectiveness by continuously adapting problem difficulty and diversifying the environment suite. A 1.5B parameter model trained on RLVE-Gym (400 engineered environments) achieves 3.37% absolute improvement, substantially outperforming 3x more compute on fixed problem sets (0.49% gain).

## Core Concept

RLVE reframes RL training as an adaptive curriculum problem. The system maintains a portfolio of verifiable environments—problem generators with explicit reward functions—and dynamically adjusts problem difficulty to match each model's learning trajectory. Rather than one difficult benchmark, use 400 environments calibrated across difficulty ranges, allowing the model to graduate from easy to hard naturally.

Verifiable environments enable cost-free reward signals: problems have explicit solutions (math, code, logic) that can be automatically checked without oracle models or human labeling.

## Architecture Overview

- **Environment Pool**: 400+ verifiable environments spanning reasoning domains (math, coding, logic, knowledge graphs)
- **Difficulty Calibration**: Monitor model accuracy and adjust problem generation parameters to maintain ~50% success rate
- **Verifiable Rewards**: Automatic checking against ground-truth solutions (no oracle needed)
- **Environment Diversity**: Mix of problem types, difficulty levels, and reasoning patterns
- **Adaptive Curriculum**: Route model to environments matching current capability level during training

## Implementation Steps

**Step 1: Design Verifiable Environment Generator**

Create parameterized problem generators where difficulty can be controlled and solutions automatically verified.

```python
class VerifiableEnvironment:
    """Base class for verifiable problem environments."""

    def __init__(self, difficulty_level=1.0):
        """
        Args:
            difficulty_level: Float in [0.5, 3.0], controlling problem complexity
        """
        self.difficulty = difficulty_level

    def generate_problem(self):
        """
        Generate a problem with ground-truth solution.

        Returns:
            problem_text: Problem description as string
            solution: Ground-truth solution (comparable to model output)
        """
        raise NotImplementedError

    def verify_solution(self, model_output, ground_truth):
        """
        Check if model output matches ground truth.

        Args:
            model_output: Model's response text
            ground_truth: Expected solution

        Returns:
            is_correct: Boolean correctness
        """
        raise NotImplementedError

class MathProofEnvironment(VerifiableEnvironment):
    """Generates mathematical proof problems of variable difficulty."""

    def generate_problem(self):
        """Generates problems like: 'Prove that sqrt(2) is irrational'."""
        num_concepts = int(2 + self.difficulty * 3)  # Difficulty scales problem depth

        problem = f"""Prove the following with {num_concepts} logical steps:
        [Problem statement parametrized by difficulty]"""

        solution = self._synthesize_proof(num_concepts)
        return problem, solution

    def verify_solution(self, model_output, solution):
        """Check proof structure: has required logical steps, no contradictions."""
        steps = model_output.split('\n')
        if len(steps) >= len(solution):
            return True  # Simplified: check step count
        return False

class CodeGenerationEnvironment(VerifiableEnvironment):
    """Generates coding problems with test cases."""

    def generate_problem(self):
        """Generate problems like: 'Write a function that sorts and filters'."""
        problem_class = random.choice(['sorting', 'search', 'graph', 'dp'])
        constraints = self._get_constraints_by_difficulty(problem_class)

        problem = f"Implement {problem_class} with constraints: {constraints}"
        test_cases = self._generate_test_cases(problem_class, constraints)

        return problem, test_cases

    def verify_solution(self, model_output, test_cases):
        """Execute model code against test cases."""
        try:
            exec(model_output)
            for input_data, expected in test_cases:
                if execute_code(model_output, input_data) != expected:
                    return False
            return True
        except:
            return False
```

**Step 2: Implement Adaptive Curriculum Controller**

Monitor model performance and route to environments at appropriate difficulty levels.

```python
def adaptive_curriculum_controller(model, environment_pool, target_success=0.5):
    """
    Adaptively select environments matching model capability.

    Args:
        model: Language model to train
        environment_pool: List of VerifiableEnvironment instances
        target_success: Target accuracy (keep ~50% to maintain learning signal)

    Yields:
        (problem, ground_truth, environment): Training samples
    """
    # Track per-environment performance
    env_success_rates = {env.name: 0.5 for env in environment_pool}
    env_sample_counts = {env.name: 0 for env in environment_pool}

    for training_step in range(num_training_steps):
        # Select environment based on recent performance
        selected_env = select_environment_by_performance(
            env_success_rates, env_sample_counts, temperature=1.0
        )

        # Adjust difficulty toward target success rate
        current_success = env_success_rates[selected_env.name]
        if current_success > target_success:
            selected_env.difficulty *= 1.1  # Increase difficulty
        elif current_success < target_success - 0.1:
            selected_env.difficulty *= 0.95  # Decrease difficulty

        # Clamp difficulty in valid range
        selected_env.difficulty = np.clip(selected_env.difficulty, 0.5, 3.0)

        # Generate problem and get model response
        problem, solution = selected_env.generate_problem()
        model_response = model.generate(problem, max_tokens=2048)

        # Verify correctness
        is_correct = selected_env.verify_solution(model_response, solution)

        # Update success rate (exponential moving average)
        alpha = 0.1
        old_rate = env_success_rates[selected_env.name]
        env_success_rates[selected_env.name] = (
            alpha * float(is_correct) + (1 - alpha) * old_rate
        )
        env_sample_counts[selected_env.name] += 1

        yield (problem, solution, selected_env), is_correct
```

**Step 3: Verifiable Reward Computation**

Define reward functions based on solution correctness (verifiable) rather than oracle models.

```python
def compute_verifiable_reward(model_response, environment, ground_truth):
    """
    Compute reward based on automatic solution verification.

    Args:
        model_response: Model's generated output
        environment: VerifiableEnvironment instance
        ground_truth: Expected solution

    Returns:
        reward: Float in [-1, 1]
    """
    # Primary signal: correctness
    is_correct = environment.verify_solution(model_response, ground_truth)

    if is_correct:
        # Bonus for efficiency/conciseness (e.g., shorter code)
        efficiency_bonus = compute_efficiency_metric(model_response)
        return 1.0 + 0.1 * efficiency_bonus
    else:
        # Penalty: how far from correctness?
        partial_credit = compute_partial_credit(model_response, ground_truth)
        return -1.0 + partial_credit

def compute_partial_credit(model_output, solution):
    """
    Award partial credit for partially correct solutions.
    Examples: right approach but wrong constant, correct structure with bugs.
    """
    # Check if output has correct logical structure
    if has_correct_structure(model_output, solution):
        return 0.5
    # Check if output attempts the right approach
    if uses_correct_approach(model_output, solution):
        return 0.2
    return 0.0
```

**Step 4: Train with Adaptive RL Loop**

Integrate adaptive curriculum with RL optimization (e.g., PPO or GRPO).

```python
def train_with_adaptive_rl(model, environment_pool, num_steps=100000):
    """
    Main training loop combining adaptive curriculum with on-policy RL.

    Args:
        model: LLM to optimize
        environment_pool: Verifiable environments
        num_steps: Total training iterations
    """
    from transformers import AutoModelForCausalLM
    import torch.optim as optim

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    curriculum = adaptive_curriculum_controller(model, environment_pool)

    for step in range(num_steps):
        # Get problem from adaptive curriculum
        (problem, solution, env), is_correct = next(curriculum)

        # Generate response with temperature for exploration
        with torch.no_grad():
            response = model.generate(
                problem, max_tokens=2048, temperature=0.8
            )

        # Compute verifiable reward
        reward = compute_verifiable_reward(response, env, solution)

        # On-policy RL update (simplified; use GRPO in practice)
        logprobs = model.forward(response).log_probs
        loss = -reward * logprobs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            # Evaluate on benchmark
            benchmark_score = evaluate_on_benchmarks(model)
            print(f"Step {step}: Reward {reward:.2f}, Benchmark {benchmark_score:.2f}")

    return model
```

## Practical Guidance

**When to Use RLVE:**
- Large-scale RL training for reasoning tasks (math, coding, logic)
- Scenarios where compute scaling alone provides diminishing returns
- Domains with automatic verification (not human evaluation)

**When NOT to Use:**
- Real-world tasks without ground-truth verification (subjective quality)
- Limited environment diversity available
- Preference learning scenarios requiring human feedback

**Hyperparameters and Configuration:**
- Target success rate: 0.5 (maintains learning signal; adjust ±0.1 based on domain)
- Difficulty bounds: [0.5, 3.0] (start conservative; expand if needed)
- Environment pool size: 100+ for diverse coverage; 400+ for production
- Curriculum temperature: 1.0 (uniform selection); increase to explore easier tasks more

**Pitfalls to Avoid:**
1. **Static environment pools** - Diversity drives improvement; single benchmark provides weak signal
2. **Ignoring partial credit** - Binary right/wrong loses information; award gradual credit for partial solutions
3. **Over-aggressive difficulty scaling** - Adapt smoothly (×1.05-1.1); jumping difficulty too fast destabilizes learning
4. **Verification gaps** - Ensure verifiers are fast and accurate; bugs in verification corrupt reward signal

---

Reference: https://arxiv.org/abs/2511.07317
