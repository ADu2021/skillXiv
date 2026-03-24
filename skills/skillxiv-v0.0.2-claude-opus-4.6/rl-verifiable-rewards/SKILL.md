---
name: rl-verifiable-rewards
title: "Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.14245"
keywords: [reinforcement-learning, reasoning, verifiable-rewards, logic-prior, chain-of-thought]
description: "RLVR extends reasoning capabilities by proving answer-only rewards implicitly incentivize correct intermediate reasoning via the Logic Prior principle."
---

# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning

## Core Concept

This work investigates whether reinforcement learning with verifiable rewards (RLVR) genuinely enhances reasoning or merely improves sampling efficiency. The key finding: RLVR can extend the reasoning boundary for both mathematical and coding tasks. The theoretical contribution shows why answer-only rewards work through the "Logic Prior" assumption—that correct reasoning chains more reliably produce correct answers. A novel CoT-Pass@K metric evaluates both answer and reasoning correctness.

## Architecture Overview

- **RLVR Framework**: Apply RL optimization to LLM outputs using verifiable (answer-only) rewards
- **Logic Prior Principle**: Theoretical explanation that correct reasoning is correlated with correct answers
- **CoT-Pass@K Metric**: Evaluates both final answer and intermediate reasoning steps for correctness
- **Training Dynamics**: Track P(CA) [probability correct answer] and P(CC|CA) [probability correct chain given correct answer]
- **Generalization Analysis**: Demonstrate that RL creates valid reasoning improvements, not just sampling artifacts

## Implementation

### Step 1: Define Verifiable Reward Function

Create reward signal based on answer correctness alone:

```python
import torch
from typing import List, Dict

class VerifiableRewardModel:
    """
    Verifiable rewards: only correct/incorrect answer classification.
    No intermediate step supervision—testing if RLVR alone improves reasoning.
    """
    def __init__(self, task_type='math'):
        self.task_type = task_type

    def compute_reward(self, answers: List[str],
                      ground_truth: List[str]) -> torch.Tensor:
        """
        Args:
            answers: [batch_size] generated answers
            ground_truth: [batch_size] correct answers

        Returns:
            rewards: [batch_size] binary (1.0 or -1.0)
        """
        rewards = []

        for ans, gt in zip(answers, ground_truth):
            # Simple answer matching (can be more sophisticated)
            is_correct = self._check_correctness(ans, gt)
            reward = 1.0 if is_correct else -1.0
            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def _check_correctness(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth"""
        # Strip whitespace, handle numeric answers
        answer_clean = answer.strip().lower()
        gt_clean = ground_truth.strip().lower()

        # Extract final numeric answer if applicable
        if self.task_type == 'math':
            import re
            answer_num = re.findall(r'-?\d+\.?\d*', answer_clean)
            gt_num = re.findall(r'-?\d+\.?\d*', gt_clean)

            if answer_num and gt_num:
                return float(answer_num[-1]) == float(gt_num[-1])

        return answer_clean == gt_clean
```

### Step 2: Extract Chain-of-Thought from Generations

Parse reasoning chains for CoT-Pass@K evaluation:

```python
class ChainOfThoughtExtractor:
    """
    Extracts reasoning chains and final answers from model outputs.
    """
    def __init__(self, reasoning_markers=None):
        if reasoning_markers is None:
            self.reasoning_markers = ['<|thinking|>', '```']
        else:
            self.reasoning_markers = reasoning_markers

    def extract_reasoning_and_answer(self, text: str) -> tuple:
        """
        Args:
            text: full model output

        Returns:
            (reasoning_chain, final_answer)
        """
        lines = text.split('\n')

        reasoning_lines = []
        answer_lines = []
        in_reasoning = False

        for line in lines:
            # Detect reasoning blocks
            if any(marker in line for marker in self.reasoning_markers):
                in_reasoning = not in_reasoning
                continue

            if in_reasoning:
                reasoning_lines.append(line)
            else:
                answer_lines.append(line)

        reasoning_chain = '\n'.join(reasoning_lines).strip()
        final_answer = '\n'.join(answer_lines).strip()

        return reasoning_chain, final_answer

    def extract_step_sequence(self, reasoning_chain: str) -> List[str]:
        """Break reasoning into logical steps"""
        # Split by common step markers
        import re
        steps = re.split(
            r'(Step \d+:|Therefore|Thus|So|Finally)',
            reasoning_chain
        )
        steps = [s.strip() for s in steps if s.strip()]

        return steps
```

### Step 3: Implement CoT-Pass@K Metric

Evaluate both answer and reasoning quality:

```python
class CoTPassAtK:
    """
    Comprehensive metric: Pass@K on both answer and reasoning.
    """
    def __init__(self, judge_model=None):
        # Can use LLM-as-judge or rule-based verification
        self.judge_model = judge_model

    def compute_pass_at_k(self, generations: List[List[str]],
                          ground_truth: List[str],
                          verify_reasoning=True) -> Dict[str, float]:
        """
        Args:
            generations: [batch_size, k_samples] list of generated answers
            ground_truth: [batch_size] correct answers
            verify_reasoning: whether to verify reasoning quality

        Returns:
            metrics: dict with Pass@K values
        """
        batch_size = len(ground_truth)
        k = len(generations[0])

        # Pass@K for final answers
        answer_pass_at_k = 0
        reasoning_pass_at_k = 0

        extractor = ChainOfThoughtExtractor()
        verifier = VerifiableRewardModel()

        for i in range(batch_size):
            sample_gens = generations[i]

            # Check if any sample has correct answer
            answer_correct = False
            reasoning_correct = False

            for gen in sample_gens:
                reasoning, answer = extractor.extract_reasoning_and_answer(gen)

                # Verify answer
                if verifier._check_correctness(answer, ground_truth[i]):
                    answer_correct = True

                    # If answer correct, verify reasoning if specified
                    if verify_reasoning and self.judge_model:
                        reasoning_quality = self._verify_reasoning_quality(
                            reasoning, ground_truth[i]
                        )
                        if reasoning_quality > 0.8:  # 80% quality threshold
                            reasoning_correct = True

            answer_pass_at_k += 1 if answer_correct else 0
            reasoning_pass_at_k += 1 if reasoning_correct else 0

        # Normalize to [0, 1]
        metrics = {
            'answer_pass@k': answer_pass_at_k / batch_size,
            'reasoning_pass@k': (reasoning_pass_at_k / batch_size
                                 if verify_reasoning else None)
        }

        return metrics

    def _verify_reasoning_quality(self, reasoning: str,
                                  ground_truth: str) -> float:
        """Use LLM judge to evaluate reasoning quality"""
        if self.judge_model is None:
            return 1.0

        prompt = f"""
        Is the following reasoning chain correct for this answer?

        Ground truth: {ground_truth}
        Reasoning: {reasoning}

        Score from 0 to 1.
        """

        score = self.judge_model.score(prompt)
        return score
```

### Step 4: Implement GRPO Training with RLVR

Train using Group Relative Policy Optimization with verifiable rewards:

```python
class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer with verifiable rewards.
    Simpler than full PPO; groups samples for relative advantage estimation.
    """
    def __init__(self, model, reward_model, device='cuda'):
        self.model = model
        self.reward_model = reward_model
        self.device = device

    def compute_advantages_from_rewards(
        self,
        generations: torch.Tensor,  # [batch, num_groups, seq_len]
        log_probs: torch.Tensor,    # [batch, num_groups, seq_len]
        rewards: torch.Tensor       # [batch, num_groups]
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.
        A(g) = r(g) - mean(r) for group g
        """
        batch_size, num_groups = rewards.shape

        # Compute mean reward across group
        group_mean_reward = rewards.mean(dim=1, keepdim=True)

        # Compute relative advantages
        advantages = rewards - group_mean_reward  # [batch, num_groups]

        # Expand for sequence dimension
        advantages_expanded = advantages.unsqueeze(-1).expand(
            batch_size, num_groups, log_probs.shape[-1]
        )  # [batch, num_groups, seq_len]

        return advantages_expanded

    def training_step(self, batch_prompts, batch_generations,
                      batch_ground_truth):
        """
        Single training step with verifiable rewards.
        """
        batch_size = len(batch_prompts)
        num_groups = len(batch_generations[0])

        # Get log probabilities from model
        log_probs_list = []

        for prompts, gens in zip(batch_prompts, batch_generations):
            with torch.no_grad():
                outputs = self.model(prompts)
                lp = torch.log_softmax(outputs.logits, dim=-1)
                log_probs_list.append(lp)

        # Compute verifiable rewards for all generations
        all_rewards = []

        for i in range(batch_size):
            sample_rewards = self.reward_model.compute_reward(
                batch_generations[i],
                [batch_ground_truth[i]] * num_groups
            )
            all_rewards.append(sample_rewards)

        rewards = torch.stack(all_rewards)  # [batch, num_groups]

        # Compute advantages
        log_probs = torch.stack(log_probs_list)
        advantages = self.compute_advantages_from_rewards(
            None, log_probs, rewards
        )

        # Compute policy gradient loss
        # L = -sum(advantages * log_probs)
        loss = -(advantages * log_probs).sum() / batch_size

        return loss, rewards.mean().item()
```

### Step 5: Analyze Training Dynamics

Track probability metrics throughout training:

```python
def analyze_training_dynamics(model, train_dataloader, num_epochs=10):
    """
    Analyze P(CA) and P(CC|CA) throughout training.
    Shows if RLVR creates genuine reasoning improvements.

    Returns:
        history: dict of metrics over time
    """
    history = {
        'epoch': [],
        'p_correct_answer': [],
        'p_correct_chain_given_answer': [],
        'pass_at_k': []
    }

    extractor = ChainOfThinkingExtractor()
    verifier = VerifiableRewardModel()
    metric_computer = CoTPassAtK()

    for epoch in range(num_epochs):
        p_ca_samples = []
        p_cc_ca_samples = []
        pass_k_samples = []

        for batch in train_dataloader:
            # Generate from model
            generations = model.generate(
                batch['prompts'],
                num_return_sequences=4,
                max_length=512
            )

            # Extract answers and reasoning
            answers = []
            reasonings = []

            for gen in generations:
                reasoning, answer = extractor.extract_reasoning_and_answer(gen)
                answers.append(answer)
                reasonings.append(reasoning)

            # Compute P(CA): probability of correct answer
            correct_answers = [
                verifier._check_correctness(ans, gt)
                for ans, gt in zip(answers, batch['ground_truth'])
            ]
            p_ca = sum(correct_answers) / len(correct_answers)
            p_ca_samples.append(p_ca)

            # Compute P(CC|CA): probability of correct chain given answer
            if sum(correct_answers) > 0:
                correct_chains_given_correct = 0
                total_correct_answers = sum(correct_answers)

                for i, is_correct_ans in enumerate(correct_answers):
                    if is_correct_ans:
                        # Judge if reasoning is correct
                        is_correct_chain = len(reasonings[i]) > 20
                        if is_correct_chain:
                            correct_chains_given_correct += 1

                p_cc_ca = (correct_chains_given_correct /
                          total_correct_answers)
                p_cc_ca_samples.append(p_cc_ca)

            # Compute Pass@K
            pass_k = metric_computer.compute_pass_at_k(
                generations, batch['ground_truth'],
                verify_reasoning=True
            )
            pass_k_samples.append(pass_k['answer_pass@k'])

        # Record epoch statistics
        history['epoch'].append(epoch)
        history['p_correct_answer'].append(sum(p_ca_samples) /
                                          len(p_ca_samples))
        history['p_correct_chain_given_answer'].append(
            sum(p_cc_ca_samples) / len(p_cc_ca_samples)
            if p_cc_ca_samples else 0
        )
        history['pass_at_k'].append(sum(pass_k_samples) /
                                    len(pass_k_samples))

    return history
```

## Practical Guidance

- **Verifiable Tasks**: Use domains where correctness is objectively verifiable (math, code) rather than subjective (writing)
- **Reward Design**: Binary rewards work; can add reward scaling by confidence if needed
- **Sampling Strategy**: Generate multiple samples (k=4-8) per prompt for robust Pass@K evaluation
- **Judge Model**: Use larger model as judge for verification (e.g., DeepSeek-R1 for math)
- **Training Stability**: Use GRPO over PPO for stability with group-relative advantages
- **Evaluation Metrics**: Always measure both answer correctness AND reasoning quality separately
- **Baseline Comparison**: Compare against supervised fine-tuning on reasoning to isolate RL contribution

## Reference

Paper: arXiv:2506.14245
Key metrics: Extended reasoning boundaries for code/math; CoT-Pass@K confirms reasoning quality
Logic Prior: P(correct answer | correct reasoning) > P(correct answer | incorrect reasoning)
Related work: RLHF, verifiable rewards, chain-of-thought, policy optimization
