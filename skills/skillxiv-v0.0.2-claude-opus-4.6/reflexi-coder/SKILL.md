---
name: reflexi-coder
title: "ReflexiCoder: Teaching LLMs to Self-Reflect on Generated Code via RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.05863"
keywords: [Code Generation, Self-Refinement, Reinforcement Learning, Iterative Improvement, GRPO]
description: "Trains LLMs to autonomously debug and improve code through structured RL-optimized reflection cycles. Internalizes debugging process into model weights rather than relying on external oracles or expensive iterative prompting."
---

# ReflexiCoder: Internalizing Code Refinement Through Reflection-Augmented RL

LLMs generate code but struggle with complex algorithmic tasks requiring iteration and refinement. Existing approaches rely on external feedback (compilers, test suites) or expensive prompt-response cycles, preventing models from developing intrinsic self-correction capabilities. ReflexiCoder teaches models to autonomously debug through reinforcement learning on structured reflection trajectories: initial attempt → identify bugs → self-correct → repeat.

## Core Concept

Standard approach: Generate code once, hope it works

ReflexiCoder: Generate code → reflect on bugs → self-correct → reflect on optimizations → repeat (structured trajectory)

The key insight: By training with RL to optimize entire reflection trajectories (not just code quality), the model learns when to reflect, how to identify specific bugs, and how to apply fixes. This internalizes debugging into the model, enabling autonomous refinement at inference time without external feedback.

The structured format (reasoning → answer → reflection → answer pairs) gates the model into thinking about multiple attempts, creating natural opportunities for improvement.

## Architecture Overview

- **Structured Output Format**: Mandatory format with reasoning, initial answer, and reflection-answer pairs
- **Multi-Component Reward Design**: Format compliance, cycle regulation, trajectory quality, efficiency bonus
- **Format Compliance Gate**: Binary check ensuring structural validity
- **Reflection-Aware GRPO**: Group Relative Policy Optimization accounting for variable-length trajectories
- **Exponential Time-Weighting**: Prefer improvements in later cycles
- **Efficiency Encouragement**: Bonus for achieving quality improvement with minimal iterations

## Implementation Steps

Implement reflection-augmented training with RL to optimize structured debugging trajectories.

**Structured Output Format Definition**

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ReflectionTrajectory:
    """Structured trajectory with mandatory reflection cycles."""
    reasoning: str  # Initial problem analysis
    initial_answer: str  # First code attempt
    reflection_cycles: List[dict]  # List of {reflection, answer} pairs

    def to_output_string(self) -> str:
        """Format trajectory for model generation."""
        output = f"<reasoning>\n{self.reasoning}\n</reasoning>\n"
        output += f"<initial_answer>\n{self.initial_answer}\n</initial_answer>\n"

        for i, cycle in enumerate(self.reflection_cycles):
            output += f"<reflection_{i+1}>\n{cycle['reflection']}\n</reflection_{i+1}>\n"
            output += f"<answer_{i+1}>\n{cycle['answer']}\n</answer_{i+1}>\n"

        return output

    @classmethod
    def from_model_output(cls, model_text: str) -> Optional['ReflectionTrajectory']:
        """Parse trajectory from model output."""
        try:
            import re

            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', model_text, re.DOTALL)
            initial_match = re.search(r'<initial_answer>(.*?)</initial_answer>', model_text, re.DOTALL)

            if not reasoning_match or not initial_match:
                return None

            reasoning = reasoning_match.group(1).strip()
            initial_answer = initial_match.group(1).strip()

            cycles = []
            for i in range(1, 10):  # Max 10 reflection cycles
                refl_match = re.search(
                    rf'<reflection_{i}>(.*?)</reflection_{i}>', model_text, re.DOTALL
                )
                ans_match = re.search(
                    rf'<answer_{i}>(.*?)</answer_{i}>', model_text, re.DOTALL
                )

                if refl_match and ans_match:
                    cycles.append({
                        'reflection': refl_match.group(1).strip(),
                        'answer': ans_match.group(1).strip()
                    })
                else:
                    break

            return cls(reasoning, initial_answer, cycles)
        except:
            return None

    def format_compliance_check(self) -> bool:
        """Verify format is valid."""
        if not self.reasoning or not self.initial_answer:
            return False
        # Every reflection must have corresponding answer
        return all('reflection' in c and 'answer' in c for c in self.reflection_cycles)
```

**Reward Computation**

```python
import torch
from typing import Tuple

class ReflectionRewardCalculator:
    """Compute multi-component rewards for reflection trajectories."""

    def __init__(self, test_executor, test_cases):
        self.executor = test_executor  # Function to test code
        self.test_cases = test_cases

    def compute_trajectory_reward(
        self,
        trajectory: ReflectionTrajectory,
        ground_truth: str
    ) -> Tuple[float, dict]:
        """
        Compute total reward for a reflection trajectory.

        Args:
            trajectory: ReflectionTrajectory instance
            ground_truth: expected output

        Returns:
            total_reward: scalar reward
            components: dict with individual reward components
        """
        components = {}

        # 1. Format Compliance (binary gate)
        format_valid = trajectory.format_compliance_check()
        components['format_compliance'] = 1.0 if format_valid else 0.0

        if not format_valid:
            # Short-circuit: invalid format = 0 reward
            return 0.0, components

        # 2. Trajectory Quality: measure improvement across cycles
        initial_code = trajectory.initial_answer
        all_answers = [initial_code] + [c['answer'] for c in trajectory.reflection_cycles]

        correctness_scores = []
        for code in all_answers:
            try:
                result = self.executor(code, self.test_cases)
                score = self._compute_correctness(result, ground_truth)
            except:
                score = 0.0
            correctness_scores.append(score)

        # Trajectory quality: max correctness achieved
        max_score = max(correctness_scores) if correctness_scores else 0.0
        components['max_correctness'] = max_score

        # Improvement: did we do better than initial attempt?
        improvement = max_score - correctness_scores[0]
        components['improvement'] = max(0.0, improvement)

        # 3. Cycle Regulation: penalize excessive iterations (with exponential decay)
        num_cycles = len(trajectory.reflection_cycles)
        cycle_penalty = self._compute_cycle_penalty(num_cycles)
        components['cycle_regulation'] = cycle_penalty

        # 4. Efficiency Bonus: reward achieving high score with fewer cycles
        if max_score > 0.8:  # Only reward if generally working
            efficiency_bonus = 1.0 / (1.0 + num_cycles * 0.3)
            components['efficiency_bonus'] = efficiency_bonus
        else:
            components['efficiency_bonus'] = 0.0

        # 5. Temporal Weighting: prefer improvements in later cycles
        temporal_weights = self._compute_temporal_weights(correctness_scores)
        temporal_contribution = sum(
            temporal_weights[i] * (correctness_scores[i] - correctness_scores[i-1])
            for i in range(1, len(correctness_scores))
        )
        components['temporal_improvement'] = temporal_contribution

        # Total reward (weighted sum)
        total_reward = (
            1.0 * components['format_compliance'] * (
                0.5 * components['max_correctness'] +
                0.3 * components['improvement'] +
                0.1 * components['temporal_improvement'] +
                0.1 * components['efficiency_bonus']
            ) + 0.1 * components['cycle_regulation']
        )

        return total_reward, components

    def _compute_correctness(self, result, ground_truth) -> float:
        """Compute correctness score (0-1) for code output."""
        if result == ground_truth:
            return 1.0
        # Partial credit for partially correct
        return 0.0

    def _compute_cycle_penalty(self, num_cycles: int) -> float:
        """Penalty for using too many cycles."""
        # Polynomial decay: each cycle adds cost
        if num_cycles <= 1:
            return 1.0
        elif num_cycles == 2:
            return 0.8
        elif num_cycles == 3:
            return 0.6
        else:
            return max(0.2, 0.6 - num_cycles * 0.1)  # Exponential decay

    def _compute_temporal_weights(self, scores: list) -> list:
        """Compute exponential weights favoring later cycles."""
        num_scores = len(scores)
        weights = []
        for i in range(num_scores):
            # Weight increases with position (favor later improvements)
            weight = 2.0 ** (i / max(1, num_scores - 1))
            weights.append(weight / sum([2.0 ** (j / max(1, num_scores - 1)) for j in range(num_scores)]))
        return weights
```

**Reflection-Aware GRPO Training**

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class ReflectionAwareGRPO:
    """Group Relative Policy Optimization adapted for reflection trajectories."""

    def __init__(self, model: nn.Module, optimizer, reward_calculator, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.reward_calculator = reward_calculator
        self.device = device

    def training_step(self, batch_data: list) -> float:
        """
        Single GRPO training step on reflection trajectories.

        Args:
            batch_data: list of {prompt, generated_text, ground_truth} dicts

        Returns:
            loss: scalar loss value
        """
        batch_size = len(batch_data)
        all_rewards = []
        all_log_probs = []
        all_lengths = []

        for sample in batch_data:
            # Parse trajectory from model output
            trajectory = ReflectionTrajectory.from_model_output(sample['generated_text'])

            if trajectory is None:
                # Invalid format: zero reward
                all_rewards.append(0.0)
                all_log_probs.append(torch.tensor(0.0, device=self.device))
                continue

            # Compute reward
            reward, components = self.reward_calculator.compute_trajectory_reward(
                trajectory, sample['ground_truth']
            )
            all_rewards.append(reward)

            # Get log probability of generated text
            input_ids = self.model.tokenizer.encode(sample['prompt'], return_tensors='pt').to(self.device)
            output_ids = self.model.tokenizer.encode(sample['generated_text'], return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(torch.cat([input_ids, output_ids], dim=1))
                logits = outputs.logits

            # Compute log probability
            log_prob = self._compute_log_probability(logits, output_ids)
            all_log_probs.append(log_prob)
            all_lengths.append(len(output_ids[0]))

        # Convert to tensors
        all_rewards = torch.tensor(all_rewards, device=self.device)
        all_log_probs = torch.stack(all_log_probs)

        # Group-relative advantages (normalize within batch)
        advantages = all_rewards - all_rewards.mean()
        advantages = advantages / (all_rewards.std() + 1e-8)

        # Length-normalized advantages (prefer efficient solutions)
        lengths_tensor = torch.tensor(all_lengths, device=self.device, dtype=torch.float)
        length_penalty = lengths_tensor / lengths_tensor.max()
        advantages = advantages - 0.1 * length_penalty

        # Policy gradient: maximize advantage-weighted log probability
        loss = -(all_log_probs * advantages.detach()).mean()

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def _compute_log_probability(self, logits, output_ids) -> torch.Tensor:
        """Compute log probability of generated sequence."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Sum log probs for entire sequence
        return log_probs.gather(-1, output_ids.unsqueeze(-1)).squeeze(-1).sum()


def training_loop(
    model_name: str,
    train_data: list,
    num_epochs: int = 3,
    batch_size: int = 8
) -> nn.Module:
    """
    Full training loop for ReflexiCoder.

    Args:
        model_name: HuggingFace model identifier
        train_data: list of {prompt, ground_truth} dicts
        num_epochs: training epochs
        batch_size: batch size

    Returns:
        trained_model: ReflexiCoder model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Initialize reward calculator (simplified)
    def dummy_executor(code, test_cases):
        try:
            exec(code)
            return "success"
        except:
            return "error"

    reward_calc = ReflectionRewardCalculator(dummy_executor, [])

    grpo = ReflectionAwareGRPO(model, optimizer, reward_calc, device)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_start in range(0, len(train_data), batch_size):
            batch = train_data[batch_start:batch_start + batch_size]

            # Generate trajectories
            batch_with_generations = []
            for sample in batch:
                # Generate with model
                input_ids = model.tokenizer.encode(sample['prompt'], return_tensors='pt').to(device)
                output_ids = model.generate(input_ids, max_length=1000, temperature=0.7)
                generated_text = model.tokenizer.decode(output_ids[0])

                batch_with_generations.append({
                    'prompt': sample['prompt'],
                    'generated_text': generated_text,
                    'ground_truth': sample.get('ground_truth', '')
                })

            # Training step
            loss = grpo.training_step(batch_with_generations)
            epoch_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

    return model
```

## Practical Guidance

**Hyperparameters**:
- Max reflection cycles: 3-5 (more becomes less helpful)
- Cycle penalty decay rate: 0.1 per cycle
- Format compliance weight: 1.0 (binary gate)
- Temporal weight exponent: 2.0
- Efficiency bonus threshold: 0.8

**When to Apply**:
- Complex algorithmic code generation
- Tasks with clear correctness criteria (test cases available)
- Scenarios where iterative refinement helps
- Training on specific coding domains (competitive programming, DSA)

**When NOT to Apply**:
- Simple one-liner code generation
- Tasks without executable evaluation
- Real-time applications where inference time matters
- Domains where external test suites already solve the problem

**Key Pitfalls**:
- Format validation too strict—limits natural variation
- Reward signal too lenient—model learns trivial cycles
- Test executor fails on valid code—false negatives
- Not normalizing rewards by trajectory length—biases toward shorter solutions

**Integration Notes**: Works with any causal LLM; requires test case execution capability; reflection format is rigid but can be customized; temporal weighting encourages fixing bugs in later cycles.

**Evidence**: Achieves 15-25% improvement over single-attempt baselines on competitive programming; reduces need for external feedback; enables models to autonomously debug without oracles; internalizes reasoning process into weights.

Reference: https://arxiv.org/abs/2603.05863
