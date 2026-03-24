---
name: cure-coevolving-llm-testing
title: "Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.03136"
keywords: [code-generation, unit-testing, reinforcement-learning, mutual-supervision, software-engineering]
description: "Improve code and test generation through co-evolution where LLMs generate both solutions and tests, optimizing each based on mutual evaluation and discriminative testing performance."
---

# Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning

## Core Concept

CURE introduces "mutual supervision"—a mechanism where code generators and unit test generators co-evolve through RL without requiring ground-truth solutions. Rather than static code-test pairs, the framework generates candidate code, evaluates it against generated tests, and optimizes both components based on test discriminative power. The innovation is theoretically grounded: reward precision (probability generated tests rank correct code above incorrect code) converges to 1 as more tests are generated, enabling self-supervised learning.

This eliminates dependency on expensive labeled datasets and achieves 5.3% improvement in one-shot accuracy and 9.0% in Best-of-N performance while reducing inference by 64.8% on long-CoT variants.

## Architecture Overview

- **Joint Optimization**: Simultaneously trains code generators and test generators using shared reward signals
- **Test Discriminativeness**: Rewards unit tests based on their ability to distinguish correct from incorrect code solutions
- **Theoretical Foundation**: Derives individual-level rewards from first principles using reward precision convergence
- **Long-CoT Efficiency**: Response-length-guided reward transformations penalize verbosity while preserving solution quality
- **Self-Supervision**: Trained unit test generators transfer as reward models to other base models

## Implementation

1. **Reward Function Design**: Ground test rewards in discriminative ability

```python
# Pseudo-code for reward calculation
def compute_joint_rewards(candidate_codes, candidate_tests, ground_truth_solution):
    """
    Score code solutions and unit tests based on mutual evaluation.
    Tests rewarded by discriminative power; code by correctness and test agreement.
    """
    # Execute all code solutions against all tests
    execution_matrix = []  # [num_solutions, num_tests]

    for solution in candidate_codes:
        results = []
        for test in candidate_tests:
            try:
                passed = execute_test(test, solution)
                results.append(passed)
            except:
                results.append(False)
        execution_matrix.append(results)

    # Compute code rewards: correct solutions + those passing more tests
    code_rewards = []
    for i, solution in enumerate(candidate_codes):
        passes_ground_truth = solution_correct(solution, ground_truth_solution)
        num_passing_tests = sum(execution_matrix[i])

        code_reward = 1.0 if passes_ground_truth else 0.0
        code_reward += 0.1 * (num_passing_tests / len(candidate_tests))
        code_rewards.append(code_reward)

    # Compute test rewards: reward based on discriminative power
    # Higher reward if test separates correct from incorrect solutions
    test_rewards = []
    for j, test in enumerate(candidate_tests):
        test_results = [execution_matrix[i][j] for i in range(len(candidate_codes))]

        # Reward precision: probability test ranks correct above incorrect
        correct_indices = [i for i, code in enumerate(candidate_codes)
                          if solution_correct(code, ground_truth_solution)]
        incorrect_indices = [i for i, code in enumerate(candidate_codes)
                            if not solution_correct(code, ground_truth_solution)]

        discrimination_score = 0.0
        if correct_indices and incorrect_indices:
            correct_pass_rate = sum(test_results[i] for i in correct_indices) / len(correct_indices)
            incorrect_pass_rate = sum(test_results[i] for i in incorrect_indices) / len(incorrect_indices)
            discrimination_score = correct_pass_rate - incorrect_pass_rate

        test_rewards.append(max(0.0, discrimination_score))

    return code_rewards, test_rewards
```

2. **Long-CoT Response Length Penalty**: Prevent excessive verbosity in reasoning chains

```python
def apply_length_penalty(reward, response_tokens, target_length=None):
    """
    Penalize overly long responses while preserving correctness signal.
    Useful for chain-of-thought models.
    """
    if target_length is None:
        target_length = 512  # Typical reasoning length

    if response_tokens > target_length:
        # Exponential penalty for excessive length
        excess = response_tokens - target_length
        length_penalty = 1.0 - min(0.5, excess / (2 * target_length))
    else:
        # Slight bonus for conciseness
        length_penalty = 1.0 + 0.05 * (1 - response_tokens / target_length)

    adjusted_reward = reward * length_penalty
    return max(0.0, adjusted_reward)
```

3. **Training Loop**: GRPO-based optimization with dual rollouts

```python
def training_step(coder, tester, task_batch, optimizer_code, optimizer_test):
    """
    Single training iteration co-optimizing code and test generators.
    Generates multiple rollouts per task for stable estimates.
    """
    total_code_loss = 0.0
    total_test_loss = 0.0

    for task in task_batch:
        # Generate candidate solutions and tests
        code_rollouts = coder.sample(task, num_samples=16, temperature=1.0)
        test_rollouts = tester.sample(task, num_samples=16, temperature=1.0)

        # Compute mutual rewards
        code_rewards, test_rewards = compute_joint_rewards(
            code_rollouts, test_rollouts, task['solution']
        )

        # GRPO losses: relative policy optimization
        code_loss = grpo_loss(coder, code_rollouts, code_rewards)
        test_loss = grpo_loss(tester, test_rollouts, test_rewards)

        # Apply KL penalty for stability
        code_loss += 0.01 * kl_divergence(coder, reference_model)
        test_loss += 0.01 * kl_divergence(tester, reference_model)

        total_code_loss += code_loss
        total_test_loss += test_loss

    # Update both models
    optimizer_code.zero_grad()
    total_code_loss.backward()
    optimizer_code.step()

    optimizer_test.zero_grad()
    total_test_loss.backward()
    optimizer_test.step()

    return total_code_loss.item(), total_test_loss.item()
```

4. **Configuration**: Recommended hyperparameters for stable training
   - Batch size: 16 code rollouts, 16 test rollouts per task
   - Learning rate: 1×10⁻⁶ with KL coefficient 0.01
   - Training steps: 350 for standard models, 50 for long-CoT variants
   - Temperature: 1.0 for diverse rollouts

## Practical Guidance

**When to Apply:**
- Training code generation models without expensive labeled test suites
- Need to improve both code quality and test effectiveness simultaneously
- Working with long-CoT models where verbosity is problematic

**Setup Requirements:**
- Base LLM supporting code generation (Qwen2.5 7B+, Llama 3.1+)
- Code execution sandbox for safe test evaluation
- Multiple coding benchmarks for validation

**Expected Results:**
- One-shot accuracy improvement: 5.3% over base models
- Best-of-N performance: 9.0% improvement
- Inference efficiency: 64.8% reduction on 4B long-CoT variants
- Test quality: Discriminative power improved through RL signal

**Key Tuning Decisions:**
- Number of rollouts: Balance between sample efficiency and gradient stability (16 typically good)
- KL coefficient: Higher values (0.05) prevent distribution drift, lower (0.005) allow faster improvement
- Long-CoT penalty: Adjust target_length based on task complexity (512-2048 typical range)
- Reference model: Pre-trained code LLM provides better KL anchoring than random initialization

**Transferability:**
- Trained unit test generator transfers to other base models as reward model
- Test reward signal validates across model architectures
- Code patterns learned may be model-specific—re-train tester for new base models

## Reference

Implemented on Qwen2.5 (7B, 14B) and Qwen3-4B. Evaluated across LiveBench, MBPP, LiveCodeBench, CodeContests, and CodeForces. Training uses 16 A100 GPUs with GRPO algorithm and mutual supervision. Demonstrates label-free optimization without dependency on proprietary model outputs.
