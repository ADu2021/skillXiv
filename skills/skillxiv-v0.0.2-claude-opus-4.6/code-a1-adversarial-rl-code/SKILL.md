---
name: code-a1-adversarial-rl-code
title: "Code-A1: Adversarial Evolving of Code LLM and Test LLM via RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.15611"
keywords: [Code Generation, Reinforcement Learning, Adversarial Training, Test Generation, Self-Collusion Prevention]
description: "Train code and test generators through adversarial co-evolution where test LLM generates adversarial test cases to expose code defects. Prevent self-collusion by separating models and enabling white-box test generation."
---

# Code-A1: Adversarial Co-Evolution for Code and Test Generation

Standard reinforcement learning for code generation suffers from self-collusion: a single model learns to generate trivial tests to easily pass, or creates generic tests that miss implementation-specific bugs. Code-A1 solves this through architectural separation—a Code LLM generates implementations while a Test LLM generates adversarial tests designed to expose defects. This adversarial co-evolution enables the Code LLM to improve robustly against high-quality tests without gaming the reward signal, matching or exceeding performance on human-annotated test benchmarks.

The technique combines architectural separation with two stabilizing mechanisms: a mistake book preserving high-quality test cases, and composite rewards balancing test validity against difficulty.

## Core Concept

The adversarial co-evolution loop operates as:

1. **Code Generation** — Code LLM proposes solution implementations
2. **Adversarial Test Creation** — Test LLM inspects code to craft targeted tests exposing bugs
3. **Execution & Reward** — Evaluate code against tests; track both code and test quality
4. **Co-Evolution** — Update both models, Code LLM to pass tests, Test LLM to expose defects
5. **Stability Mechanisms** — Preserve high-quality tests and prevent reward gaming

The architectural separation ensures the Test LLM cannot trivially satisfy itself, forcing generation of genuinely challenging test cases.

## Architecture Overview

- **Separate LLM Instances** — Independent Code LLM and Test LLM models, each optimized for different objectives
- **White-Box Test Generation** — Test LLM accesses code implementation to craft targeted adversarial tests
- **Mistake Book** — Experience replay mechanism storing high-quality failing test cases
- **Composite Reward** — Combines test validity (code actually passes) with adversarial difficulty (test exposes bugs)
- **Trial Execution Environment** — Sandboxed runtime for safe code and test execution
- **Quality Filtering** — Reject trivial tests and syntactically invalid code before inclusion in training

## Implementation Steps

Start by setting up the adversarial reward structure that scores both code and test quality.

```python
import ast
import subprocess
from typing import Tuple

class AdversarialReward:
    """Compute rewards for code and test quality in adversarial setting."""

    def __init__(self, validity_weight=0.6, difficulty_weight=0.4):
        self.validity_weight = validity_weight
        self.difficulty_weight = difficulty_weight

    def score_code(self, code: str, tests: list) -> float:
        """Score code by tests passed / failed."""
        if not self._is_valid_code(code):
            return -1.0  # Invalid code

        passed = 0
        failed = 0
        for test in tests:
            result = self._run_test(code, test)
            if result['success']:
                passed += 1
            else:
                failed += 1

        # Code reward: higher for passing more tests
        return passed / max(passed + failed, 1)

    def score_test(self, test: str, code: str, reference_code: str = None) -> float:
        """Score test by validity and difficulty (finding bugs)."""
        if not self._is_valid_test(test, code):
            return -0.5  # Invalid test

        # Validity: does it run without error on correct code?
        validity_result = self._run_test(reference_code or code, test)
        validity_score = 1.0 if validity_result['success'] else 0.0

        # Difficulty: does it catch bugs in candidate code?
        candidate_result = self._run_test(code, test)
        difficulty_score = 1.0 if not candidate_result['success'] else 0.0

        # Composite: value tests that are valid and expose bugs
        return (self.validity_weight * validity_score +
                self.difficulty_weight * difficulty_score)

    def _is_valid_code(self, code: str) -> bool:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _is_valid_test(self, test: str, code: str) -> bool:
        """Check if test is syntactically valid and runs."""
        try:
            ast.parse(test)
            # Test should import or reference the code
            return 'assert' in test or 'self.assert' in test
        except SyntaxError:
            return False

    def _run_test(self, code: str, test: str, timeout=5) -> dict:
        """Execute test against code in sandbox."""
        full_script = f"{code}\n\n{test}"
        try:
            result = subprocess.run(['python', '-c', full_script],
                                  capture_output=True,
                                  timeout=timeout,
                                  text=True)
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'output': '', 'error': 'Timeout'}
```

Next, implement the Mistake Book—a memory buffer storing high-quality failing test cases to maintain training stability.

```python
from collections import deque
import heapq

class MistakeBook:
    """Store high-quality test cases that expose bugs."""

    def __init__(self, max_size=1000):
        self.tests = deque(maxlen=max_size)
        self.quality_scores = []

    def add_test(self, test: str, quality_score: float):
        """Add test if it has sufficient quality."""
        if quality_score > 0.3:  # Threshold for "good" test
            self.tests.append(test)
            self.quality_scores.append(quality_score)

    def sample_batch(self, batch_size: int) -> list:
        """Sample tests, biasing toward high-quality ones."""
        if not self.tests:
            return []

        # Probability proportional to quality
        total_quality = sum(self.quality_scores)
        if total_quality == 0:
            # Uniform sampling if all quality is zero
            sampled_indices = np.random.choice(len(self.tests), batch_size,
                                             replace=True)
        else:
            probabilities = [q / total_quality for q in self.quality_scores]
            sampled_indices = np.random.choice(len(self.tests), batch_size,
                                             p=probabilities, replace=True)

        return [self.tests[i] for i in sampled_indices]

    def get_top_k(self, k: int) -> list:
        """Return top-k quality tests for evaluation."""
        indexed = list(enumerate(zip(self.tests, self.quality_scores)))
        top_k = heapq.nlargest(k, indexed, key=lambda x: x[1][1])
        return [test for _, (test, _) in top_k]
```

Now implement the main adversarial training loop coordinating code and test generation.

```python
import torch
from torch.optim import AdamW

class AdversarialCodeTrainer:
    """Co-train Code LLM and Test LLM with adversarial objectives."""

    def __init__(self, code_llm, test_llm, reward_fn, max_code_len=1024):
        self.code_llm = code_llm
        self.test_llm = test_llm
        self.reward_fn = reward_fn
        self.mistake_book = MistakeBook(max_size=2000)
        self.max_code_len = max_code_len

    def step(self, problem_specs: list, num_samples=4):
        """One training step with adversarial loop."""
        results = {
            'code_rewards': [],
            'test_rewards': [],
            'code_loss': 0,
            'test_loss': 0
        }

        for spec in problem_specs:
            # Phase 1: Code LLM generates implementations
            code_samples = self.code_llm.generate_batch(
                spec['prompt'],
                num_samples=num_samples,
                max_length=self.max_code_len
            )

            # Filter valid code
            valid_code = [c for c in code_samples
                         if self.reward_fn._is_valid_code(c)]

            if not valid_code:
                continue

            # Phase 2: Test LLM generates tests with white-box access
            # Concatenate code as context for Test LLM
            test_context = f"Code under test:\n{valid_code[0]}\n\nGenerate adversarial tests:"
            test_samples = self.test_llm.generate_batch(
                test_context,
                num_samples=num_samples,
                max_length=512
            )

            # Filter valid tests
            valid_tests = [t for t in test_samples
                          if self.reward_fn._is_valid_test(t, valid_code[0])]

            # Phase 3: Score implementations
            code_rewards = []
            for code in valid_code:
                # Score against all valid tests
                code_reward = self.reward_fn.score_code(code, valid_tests)
                code_rewards.append(code_reward)
                results['code_rewards'].append(code_reward)

            # Phase 4: Score tests
            test_rewards = []
            for test in valid_tests:
                # Test quality on best code sample
                best_code = valid_code[np.argmax(code_rewards)]
                test_reward = self.reward_fn.score_test(test, best_code)
                test_rewards.append(test_reward)
                results['test_rewards'].append(test_reward)

                # Add high-quality tests to mistake book
                if test_reward > 0.5:
                    self.mistake_book.add_test(test, test_reward)

            # Phase 5: Update Code LLM with policy gradient
            code_rewards_tensor = torch.tensor(code_rewards, dtype=torch.float32)
            code_loss = self._compute_policy_loss(code_samples, code_rewards_tensor)
            self._update_model(self.code_llm, code_loss)
            results['code_loss'] += code_loss.item()

            # Phase 6: Update Test LLM with policy gradient
            test_rewards_tensor = torch.tensor(test_rewards, dtype=torch.float32)
            test_loss = self._compute_policy_loss(test_samples, test_rewards_tensor)
            self._update_model(self.test_llm, test_loss)
            results['test_loss'] += test_loss.item()

        return results

    def _compute_policy_loss(self, samples: list, rewards: torch.Tensor):
        """Policy gradient loss: maximize reward * log_prob."""
        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Get log probabilities of samples
        logprobs = self.code_llm.compute_logprob_batch(samples)

        # Policy gradient objective
        loss = -(rewards * logprobs).mean()
        return loss

    def _update_model(self, model, loss):
        """Gradient update step."""
        optimizer = AdamW(model.parameters(), lr=1e-5)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    def train(self, problem_specs, num_steps=100):
        """Run full adversarial training."""
        for step in range(num_steps):
            results = self.step(problem_specs)

            if (step + 1) % 10 == 0:
                avg_code_reward = np.mean(results['code_rewards'])
                avg_test_reward = np.mean(results['test_rewards'])
                print(f"Step {step+1}: Code Reward={avg_code_reward:.3f}, "
                      f"Test Reward={avg_test_reward:.3f}")
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Validity weight typically 0.6, difficulty weight 0.4; adjust based on how strict test quality requirements are
- Mistake book size of 1000-2000 balances diversity with computational cost
- Use separate model instances for code and test LLMs; same architecture works but keep parameters independent
- Effective for programming problems with clear test cases (competitive programming, algorithmic problems, code completion)

**When NOT to use:**
- For problems without clear oracles (creative coding, open-ended design tasks)
- When high-quality human-written tests are already available; use supervised learning instead
- For code generation in domains requiring safety verification (e.g., cryptography, safety-critical systems)

**Common Pitfalls:**
- Test generation becoming too difficult; use curriculum learning starting with simpler problems
- Code and test LLMs converging to trivial solutions; regularly refresh the mistake book with historical high-quality tests
- Insufficient diversity in generated tests; apply dropout/temperature sampling during generation
- Execution timeout issues from infinite loops; set strict time limits (typically 1-5 seconds per execution)
- Test LLM seeing answers in code context leading to "teaching to the test"; use code anonymization if possible

## Reference

Paper: [Code-A1: Adversarial Evolving of Code LLM and Test LLM via RL](https://arxiv.org/abs/2603.15611)
