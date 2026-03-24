---
name: tool-verification-reasoning
title: "Tool Verification for Test-Time Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.02203"
keywords: [Test-Time RL, Verification, Self-Improvement, Reasoning, Consensus]
description: "Tool Verification stabilizes self-improving reasoning models by using external tool execution as ground-truth evidence to prevent spurious consensus from becoming reinforced training signals."
---

# Technique: Verification-Aware Voting for Self-Improving Reasoning

Self-improving language models trained with test-time RL face a critical failure mode: spurious high-frequency consensus. When many sampling trajectories produce the same incorrect answer (by coincidence, not correctness), the model receives false positive reward signals and reinforces these errors. This is particularly dangerous for mathematical and coding tasks where verification is possible but frequently ignored.

Tool Verification addresses this by leveraging external tools (code executors, math solvers) as ground-truth evidence. During reward estimation, the system upweights rollouts that pass tool verification, creating verification-aware voting that prevents spurious consensus from corrupting the learning signal.

## Core Concept

The core insight: external tools provide objective correctness signals that are far more reliable than consensus voting. When training self-improving models:

1. **Collect diverse rollouts**: Generate multiple solution attempts
2. **Verify with tools**: Execute code, run math solver, etc.
3. **Verification-aware voting**: Count only verified-correct rollouts
4. **Supervised data synthesis**: Use verified outputs as training targets
5. **Update model**: Train on verified data, not spurious consensus

This prevents mode collapse to incorrect solutions and stabilizes self-evolution.

## Architecture Overview

- **Tool Execution**: External verifiers (code runners, equation solvers, theorem provers)
- **Verification Labels**: Binary correctness signals from tools
- **Weighted Voting**: Upweight verified trajectories in reward computation
- **Test-Time RL**: Standard RL loop with verification filtering
- **Online Data Synthesis**: Create training data from verified rollouts

## Implementation Steps

Implement tool verification in a self-improving RL loop. Here's how:

Define verification functions for different problem domains:

```python
import subprocess
import re
from typing import Dict, Any, Tuple

class ToolVerifier:
    """Verifies solution correctness using external tools."""

    def verify_python_code(
        self,
        code: str,
        test_input: str = None,
        expected_output: str = None,
    ) -> Dict[str, Any]:
        """
        Execute Python code and verify correctness.
        """
        try:
            # Execute code in sandbox
            result = subprocess.run(
                ['python', '-c', code],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=5,
            )

            success = result.returncode == 0
            output = result.stdout.strip()

            # Check against expected output if provided
            verified = True
            if expected_output is not None:
                verified = output == expected_output.strip()

            return {
                'verified': verified,
                'success': success,
                'output': output,
                'error': result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {'verified': False, 'success': False, 'error': 'Timeout'}
        except Exception as e:
            return {'verified': False, 'success': False, 'error': str(e)}

    def verify_math_solution(
        self,
        equation: str,
        solution: str,
    ) -> Dict[str, Any]:
        """
        Verify mathematical solution using symbolic math.
        """
        try:
            from sympy import sympify, simplify, solve, symbols

            # Parse equation and solution
            eq = sympify(equation)
            sol = sympify(solution)

            # Verify by substitution
            # Extract variable from equation
            vars_in_eq = list(eq.free_symbols)
            if not vars_in_eq:
                return {'verified': False, 'error': 'No variables in equation'}

            var = vars_in_eq[0]

            # Check if solution satisfies equation
            result = eq.subs(var, sol)
            verified = simplify(result) == 0

            return {
                'verified': verified,
                'solution': str(sol),
                'verification_result': str(result),
            }
        except Exception as e:
            return {'verified': False, 'error': str(e)}

    def verify_multiple_choice(
        self,
        answer: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        """
        Verify multiple choice answer.
        """
        # Exact match or case-insensitive
        verified = answer.strip().lower() == ground_truth.strip().lower()

        return {
            'verified': verified,
            'answer': answer,
            'ground_truth': ground_truth,
        }
```

Implement verification-aware voting in the RL loop:

```python
import torch
import numpy as np
from typing import List, Tuple

class VerificationAwareVoting:
    """
    Compute rewards using verified-correct rollouts.
    """

    def __init__(self, verifier: ToolVerifier, alpha=0.8):
        self.verifier = verifier
        self.alpha = alpha  # Weight for verified vs. unverified

    def compute_verification_scores(
        self,
        rollouts: List[str],
        verification_type: str = 'code',
        test_input: str = None,
        expected_output: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Verify each rollout and compute weighted scores.

        Returns:
          (scores, verification_labels):
            scores shape [num_rollouts], verification_labels shape [num_rollouts]
        """
        verification_results = []
        verification_labels = []

        for rollout in rollouts:
            if verification_type == 'code':
                result = self.verifier.verify_python_code(
                    rollout,
                    test_input=test_input,
                    expected_output=expected_output,
                )
            elif verification_type == 'math':
                result = self.verifier.verify_math_solution(
                    test_input,  # equation
                    rollout,      # solution
                )
            else:
                result = {'verified': False}

            verification_results.append(result)
            verification_labels.append(1 if result['verified'] else 0)

        verification_labels = np.array(verification_labels, dtype=np.float32)

        # Compute scores: verified=1, unverified weighted lower
        scores = verification_labels * 1.0 + (1 - verification_labels) * (1 - self.alpha)

        return scores, verification_labels

    def compute_verification_weighted_reward(
        self,
        rollouts: List[str],
        logprobs: np.ndarray,  # [num_rollouts]
        verification_type: str = 'code',
        test_input: str = None,
        expected_output: str = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute rewards weighted by verification status.
        """
        scores, labels = self.compute_verification_scores(
            rollouts,
            verification_type=verification_type,
            test_input=test_input,
            expected_output=expected_output,
        )

        # Rewards: verified solutions get +1, unverified get downweighted
        rewards = scores - 1.0  # -1 to 0 range

        # Statistics for monitoring
        stats = {
            'num_verified': int(np.sum(labels)),
            'num_total': len(rollouts),
            'verification_rate': float(np.sum(labels)) / len(rollouts),
            'avg_verified_logprob': float(logprobs[labels > 0].mean()) if np.any(labels) else 0.0,
            'avg_unverified_logprob': float(logprobs[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        }

        return rewards, stats
```

Integrate verification into standard RL training:

```python
class VerificationAwareRLTrainer:
    """
    Test-time RL with verification-aware reward computation.
    """

    def __init__(self, model, verifier, learning_rate=1e-5):
        self.model = model
        self.verifier = ToolVerifier()
        self.verifier_voter = VerificationAwareVoting(verifier)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def generate_rollouts(
        self,
        prompt: str,
        num_rollouts: int = 8,
        max_length: int = 512,
    ) -> Tuple[List[str], np.ndarray]:
        """Generate multiple solution attempts."""
        rollouts = []
        logprobs = []

        for _ in range(num_rollouts):
            output = self.model.generate(
                prompt,
                max_length=max_length,
                temperature=0.7,
                output_scores=True,
                return_dict_in_generate=True,
            )

            rollouts.append(output.sequences)
            logprobs.append(output.scores)

        return rollouts, np.array(logprobs)

    def training_step(
        self,
        prompt: str,
        expected_output: str,
        verification_type: str = 'code',
    ) -> Dict[str, float]:
        """
        Single RL training step with verification.
        """
        # Generate rollouts
        rollouts, logprobs = self.generate_rollouts(prompt, num_rollouts=8)

        # Compute verification-aware rewards
        rewards, stats = self.verifier_voter.compute_verification_weighted_reward(
            rollouts,
            logprobs,
            verification_type=verification_type,
            expected_output=expected_output,
        )

        # Compute policy gradient
        # In practice, use REINFORCE or PPO loss
        policy_loss = -(torch.tensor(rewards) * torch.tensor(logprobs)).mean()

        # Backward pass
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': policy_loss.item(),
            **stats
        }

    def train_batch(self, examples: List[Dict], num_epochs: int = 3):
        """Train on batch of examples."""
        for epoch in range(num_epochs):
            total_stats = {}

            for example in examples:
                stats = self.training_step(
                    prompt=example['prompt'],
                    expected_output=example['expected_output'],
                    verification_type=example.get('type', 'code'),
                )

                # Accumulate stats
                for key, val in stats.items():
                    if key not in total_stats:
                        total_stats[key] = 0
                    total_stats[key] += val

            # Average stats
            n = len(examples)
            avg_stats = {k: v/n for k, v in total_stats.items()}
            print(f"Epoch {epoch+1}: {avg_stats}")
```

## Practical Guidance

**When to Use:**
- Self-improving reasoning models (math, code, logic)
- When external verification tools are available
- Training models to reach increasingly difficult problems
- Preventing spurious consensus from corrupting training

**When NOT to Use:**
- Generation tasks without clear verification (open-ended text)
- When verification is expensive or unavailable
- Real-time inference (verification happens offline, not needed at test time)

**Verification Strategy:**
- Code: Use subprocess execution with test cases
- Math: Use symbolic math libraries (SymPy) for equation verification
- Logic: Use SAT/SMT solvers for logical correctness
- Multiple choice: Exact or pattern-based matching

**Implementation Notes:**
- Verification can be slow; batch verification operations
- Sandbox code execution carefully (resource limits, timeout)
- Handle verification failures gracefully (don't crash training)
- Track verification statistics for monitoring

**Hyperparameters:**
- `alpha`: Weight for unverified vs. verified (0.1–0.5). Lower = stronger penalty for unverified
- `num_rollouts`: 4–16 typical (more rollouts = better consensus estimates)
- Verification timeout: 5–30 seconds depending on task

**Results:**
- Prevents mode collapse to incorrect solutions
- Larger gains on harder problems (AMC, AIME)
- Stable self-improvement without divergence
- Generalizes across model architectures

---

**Reference:** [Tool Verification for Test-Time Reinforcement Learning](https://arxiv.org/abs/2603.02203)
