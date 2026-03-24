---
name: spark-process-aware-rewards
title: "SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.03244
keywords: [process rewards, reference-free learning, mathematical reasoning, verification, reward models]
description: "Train process reward models without ground-truth references using synthetic verification data from generators and verifiers. SPARK achieves 67.5 F1 on ProcessBench—ideal when step-level annotations are expensive but verification is available."
---

## Overview

SPARK eliminates the need for ground-truth step-level annotations by training process reward models (PRMs) using synthetic verification data. A generator produces diverse solutions while a verifier evaluates them, providing supervision that outperforms ground-truth-based training when used to train generative PRMs.

## When to Use

- Training process reward models without expensive step annotations
- Mathematical reasoning and problem-solving tasks
- Scenarios where verification is cheaper than annotation
- Applications needing to identify erroneous reasoning steps
- RL training requiring step-level reward signals

## When NOT to Use

- Tasks with abundant ground-truth step-level annotations
- Domains where verification is as expensive as annotation
- Simple tasks not benefiting from step-level rewards
- Outcome-only verification approaches are sufficient

## Core Technique

Three-stage framework combining generation, verification, and PRM training:

```python
# Reference-Free Process Reward Model Training
class SPARKFramework:
    def __init__(self, generator, verifier):
        self.generator = generator
        self.verifier = verifier
        self.prm = None

    def stage1_generate_and_verify(self, problems, num_diverse_solutions=10):
        """
        Stage 1: Generate diverse solutions and verify them.
        Creates synthetic supervision without ground truth.
        """
        synthetic_data = []

        for problem in problems:
            # Generate diverse solution trajectories
            solutions = []
            for _ in range(num_diverse_solutions):
                solution = self.generator.generate(
                    problem,
                    temperature=0.7  # Diversity through temperature
                )
                solutions.append(solution)

            # Parallel verification: quick pass/fail
            for solution in solutions:
                parallel_result = self.verifier.verify_parallel(
                    problem, solution
                )

                # Sequential verification: detailed step analysis
                sequential_result = self.verifier.verify_sequential(
                    problem, solution
                )

                # Combine verification signals
                verification_data = {
                    'problem': problem,
                    'solution': solution,
                    'steps': solution.steps,
                    'parallel_verdict': parallel_result,
                    'sequential_verdict': sequential_result,
                    'step_judgments': sequential_result.step_judgments
                }

                synthetic_data.append(verification_data)

        return synthetic_data

    def stage2_train_process_reward_model(self, synthetic_data):
        """
        Stage 2: Train generative PRM on synthetic verification data.
        Outputs are reward predictions at each step.
        """
        # Initialize PRM
        self.prm = GenerativePRM()

        for batch in self.create_batches(synthetic_data):
            problems = [d['problem'] for d in batch]
            solutions = [d['solution'] for d in batch]
            step_judgments = [d['step_judgments'] for d in batch]

            # Forward pass: predict step-level rewards
            step_predictions = self.prm.predict_step_rewards(
                problems,
                solutions
            )

            # Loss based on synthetic supervision
            loss = self.compute_pairwise_loss(
                step_predictions,
                step_judgments
            )

            # Update PRM
            self.prm.train_step(loss)

        return self.prm

    def stage3_apply_as_reward_signal(self, problems_to_solve):
        """
        Stage 3: Use trained PRM as reward signal during RL.
        Includes format constraints to prevent gaming.
        """
        rl_agent = RLAgent()

        for problem in problems_to_solve:
            # Generate with RL
            trajectory = rl_agent.generate_with_rl(problem)

            # Compute step-level rewards using PRM
            step_rewards = []
            for i, step in enumerate(trajectory.steps):
                # PRM predicts correctness probability for step
                reward = self.prm.predict_reward(
                    problem,
                    trajectory.steps[:i+1]
                )

                # Add chain-of-thought verification
                # Prevents output gaming by checking reasoning
                verified = self.verify_step_reasoning(
                    problem,
                    trajectory.steps[:i+1]
                )

                # Format constraints ensure valid structure
                if not self.check_format_constraints(step):
                    reward = 0.0  # Penalize malformed steps

                step_rewards.append(reward)

            # Use step rewards to train RL agent
            rl_agent.update(trajectory, step_rewards)

        return rl_agent

    def compute_pairwise_loss(self, predictions, labels):
        """
        Loss comparing predicted vs verified correctness.
        Handles multiple verification signals per step.
        """
        loss = 0.0
        for pred, label in zip(predictions, labels):
            # Log-likelihood of correct classification
            loss += torch.nn.functional.binary_cross_entropy(
                pred, label.float()
            )
        return loss / len(predictions)

    def verify_step_reasoning(self, problem, steps_so_far):
        """
        Secondary verification ensuring steps are logically sound.
        Prevents reward hacking through malformed proofs.
        """
        # Apply symbolic reasoning checks
        logical_valid = self.check_logical_soundness(steps_so_far)

        # Check mathematical consistency
        mathematically_sound = self.check_mathematical_validity(
            problem, steps_so_far
        )

        return logical_valid and mathematically_sound
```

Key insight: synthetic verification data outperforms ground-truth when used to train generative PRMs, eliminating expensive annotation requirements.

## Key Results

- 67.5 F1 on ProcessBench (exceeds ground-truth: 66.4, GPT-4o: 61.9)
- 47.4% average accuracy on six mathematical reasoning benchmarks
- Reference-free training eliminates annotation bottleneck
- Works effectively with Qwen2.5-Math-7B baseline

## Implementation Notes

- Generator and verifier create diverse training data
- Parallel and sequential verification provide complementary signals
- Generative PRM outputs step-level rewards
- Format constraints in RL prevent output gaming
- Chain-of-thought verification ensures reasoning validity

## References

- Original paper: https://arxiv.org/abs/2512.03244
- Focus: Reference-free process reward model training
- Domain: Mathematical reasoning, reinforcement learning
