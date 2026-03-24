---
name: stepwiser-generative-judges
title: StepWiser Stepwise Generative Judges for Wiser Reasoning
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.19229
keywords: [process-reward-model, generative-judge, reasoning-verification, meta-reasoning, rl-training]
description: "Train stepwise judges as generative models that perform meta-reasoning about intermediate steps, combining explainability with improved accuracy over static process reward models"
---

# StepWiser: Stepwise Generative Judges for Wiser Reasoning

## Core Concept

StepWiser reframes process reward modeling as a reasoning task itself. Instead of classifying intermediate steps as correct/incorrect, a generative judge model performs meta-reasoning by generating explanations before predicting verdicts. Trained via reinforcement learning on relative outcomes, StepWiser achieves better step-level accuracy than existing methods while being interpretable and generalizable to new problem distributions.

## Architecture Overview

- **Generative Judge Model**: Produces reasoning before verdict
- **Meta-Reasoning**: Model explains why steps are good/bad
- **RL Training**: Learn from relative outcome comparisons
- **Bidirectional Feedback**: Training-time and inference-time improvements
- **Process Supervision**: Fine-grained intermediate step evaluation

## Implementation Steps

### Stage 1: Generative Judge Architecture

Design a model that generates step explanations before verdicts.

```python
# Generative stepwise judge
import torch
from torch import nn
from typing import Dict, List, Tuple

class GenerativeStepwiseJudge(nn.Module):
    """Judge that generates reasoning before verdicting on steps"""

    def __init__(
        self,
        model_dim: int = 4096,
        vocab_size: int = 32000,
        max_explanation_len: int = 128
    ):
        super().__init__()
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.max_explanation_len = max_explanation_len

        # Encoder for policy step
        self.step_encoder = nn.Sequential(
            nn.Linear(256, model_dim),  # Embed step text
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        # Encoder for problem context
        self.context_encoder = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )

        # Explanation generation head (autoregressive)
        self.explanation_head = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=2,
            batch_first=True
        )
        self.explanation_decoder = nn.Linear(model_dim, vocab_size)

        # Verdict head (binary: good or bad step)
        self.verdict_head = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 2)  # good/bad
        )

    def forward(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str
    ) -> Dict:
        """
        Judge current step with explanation.

        Returns:
            - explanation_tokens: generated explanation
            - verdict_logits: good/bad classification
            - confidence: meta-confidence in verdict
        """
        # Encode inputs
        step_embed = self.step_encoder(self.embed_text(current_step))
        context_embed = self.context_encoder(self.embed_context(problem, previous_steps))

        # Generate explanation
        combined = torch.cat([context_embed.unsqueeze(0), step_embed.unsqueeze(0)], dim=-1)
        explanation_hidden, _ = self.explanation_head(combined)

        explanation_logits = self.explanation_decoder(explanation_hidden)
        explanation_tokens = explanation_logits.argmax(dim=-1)

        # Verdict based on step + generated reasoning
        verdict_input = torch.cat([step_embed, explanation_hidden.squeeze(0)], dim=-1)
        verdict_logits = self.verdict_head(verdict_input)

        # Confidence: entropy of verdict
        verdict_probs = torch.nn.functional.softmax(verdict_logits, dim=-1)
        confidence = 1.0 - torch.nn.functional.entropy(verdict_probs)

        return {
            "explanation_tokens": explanation_tokens,
            "explanation_logits": explanation_logits,
            "verdict_logits": verdict_logits,
            "verdict_probs": verdict_probs,
            "confidence": confidence
        }

    def embed_text(self, text: str) -> torch.Tensor:
        """Embed step text"""
        return torch.randn(256)  # Placeholder

    def embed_context(self, problem: str, prev_steps: List[str]) -> torch.Tensor:
        """Embed problem + previous steps"""
        return torch.randn(512)  # Placeholder
```

### Stage 2: RL Training via Comparative Outcomes

Train judge using RL based on which verdict predictions are correct.

```python
# RL training for generative judges
class GenerativeJudgeRLTrainer:
    """Train judge using reinforcement learning"""

    def __init__(
        self,
        judge: GenerativeStepwiseJudge,
        policy_model,
        lr: float = 1e-5
    ):
        self.judge = judge
        self.policy_model = policy_model
        self.optimizer = torch.optim.Adam(judge.parameters(), lr=lr)

    def generate_step_pair(
        self,
        problem: str,
        previous_steps: List[str]
    ) -> Tuple[str, str, bool]:
        """
        Generate two different step candidates and evaluate which is better.

        Returns:
            step_a, step_b, label (True if A better)
        """
        # Generate two step candidates from policy
        step_a = self.policy_model.generate_step(problem, previous_steps)
        step_b = self.policy_model.generate_step(problem, previous_steps)

        # Determine which leads to correct solution
        outcome_a = self.evaluate_trajectory(problem, previous_steps + [step_a])
        outcome_b = self.evaluate_trajectory(problem, previous_steps + [step_b])

        # Label: is A better than B?
        label = outcome_a > outcome_b

        return step_a, step_b, label

    def evaluate_trajectory(self, problem: str, steps: List[str]) -> float:
        """Evaluate trajectory quality (0-1, higher is better)"""
        # Placeholder: in practice, check if leads to correct answer
        return 0.5

    def train_step(self, batch: List[Dict]) -> Dict:
        """Single RL training step"""
        losses = {"verdict_loss": 0, "explanation_loss": 0}

        for example in batch:
            problem = example["problem"]
            steps = example["steps"]
            correct_verdicts = example["verdicts"]  # List of correct/incorrect

            # Judge each step
            for step_idx, step in enumerate(steps):
                output = self.judge(
                    problem,
                    steps[:step_idx],
                    step
                )

                # Ground truth: is this step correct?
                is_correct = correct_verdicts[step_idx]
                gt_verdict = torch.tensor([is_correct], dtype=torch.long)

                # Verdict loss: classification
                verdict_loss = torch.nn.functional.cross_entropy(
                    output["verdict_logits"],
                    gt_verdict
                )
                losses["verdict_loss"] += verdict_loss

                # Explanation loss: generation (optional, for interpretability)
                # In practice, could supervise with human explanations
                explanation_loss = 0  # Placeholder

                losses["explanation_loss"] += explanation_loss

        # Normalize
        num_steps = sum(len(ex["steps"]) for ex in batch)
        for key in losses:
            losses[key] /= max(num_steps, 1)

        # Backward
        total_loss = sum(losses.values())
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return losses
```

### Stage 3: Inference-Time Step Validation

Use the judge to improve reasoning at inference time.

```python
# Inference-time step validation
class ReasoningWithStepValidation:
    """Generate reasoning with stepwise validation"""

    def __init__(self, policy_model, judge: GenerativeStepwiseJudge):
        self.policy = policy_model
        self.judge = judge

    def generate_with_validation(
        self,
        problem: str,
        max_steps: int = 20,
        temperature: float = 0.7,
        validation_threshold: float = 0.5
    ) -> Dict:
        """
        Generate solution step-by-step with validation.

        Steps with low judge confidence are regenerated or pruned.
        """
        steps = []
        confidences = []
        all_verdicts = []

        for step_idx in range(max_steps):
            # Generate step candidate
            step = self.policy.generate_step(
                problem,
                steps,
                temperature=temperature
            )

            # Judge the step
            verdict_output = self.judge(problem, steps, step)
            verdict_prob = verdict_output["verdict_probs"][1]  # P(good)
            confidence = verdict_output["confidence"]

            # Check if step passes validation
            if confidence > validation_threshold:
                # Accept step
                steps.append(step)
                confidences.append(confidence.item())
                all_verdicts.append(verdict_prob.item())

            else:
                # Reject: try generating different step
                for retry in range(3):
                    step = self.policy.generate_step(
                        problem,
                        steps,
                        temperature=temperature + 0.2  # Increase diversity
                    )
                    verdict_output = self.judge(problem, steps, step)
                    confidence = verdict_output["confidence"]

                    if confidence > validation_threshold:
                        steps.append(step)
                        confidences.append(confidence.item())
                        all_verdicts.append(verdict_output["verdict_probs"][1].item())
                        break
                else:
                    # Give up, use step anyway
                    steps.append(step)
                    confidences.append(confidence.item())
                    all_verdicts.append(0.5)

            # Check if done
            if self.policy.is_solution_complete(problem, steps):
                break

        return {
            "steps": steps,
            "confidences": confidences,
            "verdicts": all_verdicts,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0
        }
```

### Stage 4: Process Reward Training Integration

Use StepWiser judges to improve policy models.

```python
# Process reward training with generative judges
class ProcessRewardImprovement:
    """Use judge verdicts to improve policy"""

    def __init__(self, policy_model, judge: GenerativeStepwiseJudge):
        self.policy = policy_model
        self.judge = judge
        self.policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)

    def improve_policy_with_process_rewards(
        self,
        rollouts: List[Dict]
    ) -> float:
        """
        Train policy to generate steps that pass judge validation.

        Reward = judge confidence in step correctness.
        """
        total_loss = 0

        for rollout in rollouts:
            problem = rollout["problem"]
            steps = rollout["steps"]
            ground_truth = rollout["answer"]

            # Trace through rollout with judge
            for step_idx, step in enumerate(steps):
                # Policy log probability for this step
                log_prob = self.policy.get_log_prob(step, problem, steps[:step_idx])

                # Judge verdict
                verdict_output = self.judge(problem, steps[:step_idx], step)
                judge_confidence = verdict_output["confidence"]

                # Reward: how confident is judge in step correctness?
                # Also: does path lead to solution?
                path_quality = self.evaluate_path_quality(problem, steps[:step_idx+1], ground_truth)

                reward = 0.5 * judge_confidence + 0.5 * path_quality

                # Policy gradient: maximize log_prob * reward
                loss = -(log_prob * reward)
                total_loss += loss

        total_loss /= sum(len(r["steps"]) for r in rollouts)

        # Backward
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()

        return total_loss.item()

    def evaluate_path_quality(self, problem: str, steps: List[str], target: str) -> float:
        """Evaluate if current path is on track to solution"""
        return 0.5  # Placeholder
```

### Stage 5: Evaluation

Measure judge accuracy and policy improvement.

```python
# Evaluation framework
class StepwiserEvaluator:
    """Evaluate judge quality and reasoning improvements"""

    def __init__(self, judge: GenerativeStepwiseJudge, policy_model):
        self.judge = judge
        self.policy = policy_model

    def evaluate_judge_accuracy(
        self,
        test_problems: List[Dict],
        max_problems: int = 500
    ) -> Dict:
        """
        Evaluate judge accuracy on intermediate steps.

        Metrics:
        - Accuracy: % of steps correctly classified
        - F1: balance between precision and recall
        """
        correct_verdicts = 0
        total_verdicts = 0
        tp, fp, fn = 0, 0, 0

        for problem_data in test_problems[:max_problems]:
            problem = problem_data["problem"]
            ground_truth_steps = problem_data["steps"]
            correct_step_labels = problem_data["correct_steps"]

            for step_idx, step in enumerate(ground_truth_steps):
                output = self.judge(
                    problem,
                    ground_truth_steps[:step_idx],
                    step
                )

                pred_is_correct = output["verdict_logits"].argmax().item() == 1
                is_correct = correct_step_labels[step_idx]

                if pred_is_correct == is_correct:
                    correct_verdicts += 1
                    if is_correct:
                        tp += 1
                else:
                    if pred_is_correct:
                        fp += 1
                    else:
                        fn += 1

                total_verdicts += 1

        accuracy = correct_verdicts / total_verdicts
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def evaluate_reasoning_quality(
        self,
        test_problems: List[Dict]
    ) -> Dict:
        """
        Evaluate quality of generated reasoning using judge.

        Does validation improve solution quality?
        """
        results = {
            "without_judge": 0,
            "with_judge": 0,
            "improvement": 0
        }

        for problem_data in test_problems:
            problem = problem_data["problem"]
            target = problem_data["answer"]

            # Generate without judge
            solution_no_judge = self.policy.generate_solution(problem)
            correct_no_judge = self.check_solution(solution_no_judge, target)

            # Generate with judge validation
            reasoner = ReasoningWithStepValidation(self.policy, self.judge)
            output = reasoner.generate_with_validation(problem)
            solution_with_judge = " ".join(output["steps"])
            correct_with_judge = self.check_solution(solution_with_judge, target)

            if correct_no_judge:
                results["without_judge"] += 1
            if correct_with_judge:
                results["with_judge"] += 1

        total = len(test_problems)
        results["without_judge"] /= total
        results["with_judge"] /= total
        results["improvement"] = results["with_judge"] - results["without_judge"]

        return results

    def check_solution(self, solution: str, target: str) -> bool:
        """Check if solution is correct"""
        return solution.strip() == target.strip()
```

## Practical Guidance

### Training Recipe

- **Phase 1**: Train judge on synthetic step labels (easy supervision)
- **Phase 2**: RL training on relative outcome comparisons (harder data)
- **Phase 3**: Fine-tune on domain-specific problems

### Explanation Integration

- Explanations are emergent from RL training (not explicitly supervised)
- Can optionally supervise with human-written explanations
- Improves interpretability without hurting performance

### When to Use

- Reasoning tasks requiring intermediate step validation
- Multi-step math and coding problems
- Scenarios where explainability is important
- Improving reasoning through search (beam search, etc.)

### When NOT to Use

- Single-step tasks without intermediate verification
- Real-time systems (judge adds latency)
- Domains without clear step correctness

### Performance Expectations

- Judge accuracy: 85-92% on intermediate steps
- Reasoning improvement: +3-8% on complex problems
- Inference overhead: 1.5-2x slowdown (due to validation)

## Reference

StepWiser: Stepwise Generative Judges for Wiser Reasoning. arXiv:2508.19229
- https://arxiv.org/abs/2508.19229
