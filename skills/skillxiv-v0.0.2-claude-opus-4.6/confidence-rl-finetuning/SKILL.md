---
name: confidence-rl-finetuning
title: "Confidence Is All You Need: Few-Shot RL Fine-Tuning of Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.06395"
keywords: [reinforcement learning, self-supervision, confidence reward, few-shot]
description: "Improve language model reasoning using only model confidence as reward signals, eliminating need for labels or preference models while achieving substantial gains with minimal data."
---

# Confidence Is All You Need: Few-Shot RL Fine-Tuning

## Core Concept

Rather than requiring external reward models, preference annotations, or label engineering, RLSC uses the model's own confidence as training signal. The insight: majority voting implicitly performs mode sharpening—concentrating probability mass on the most likely response. By directly optimizing confidence, you improve agreement and reasoning quality without annotations.

## Architecture Overview

- **Self-confidence objective**: Maximize probability that independent samples agree
- **Token-level probabilities**: Track confidence across generation sequence
- **Minimal data requirements**: Only 16 samples per question, 10-20 training steps
- **No labels needed**: Entirely self-supervised approach
- **Strong gains**: +13.4% AIME, +21.2% MATH, +21.7% Minerva Math

## Implementation

### Step 1: Implement Confidence Scoring

Compute confidence from token-level probabilities:

```python
class ConfidenceScorer:
    def __init__(self):
        self.epsilon = 1e-10

    def compute_token_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Get confidence (max probability) for each token."""

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Confidence is max probability
        confidence, _ = torch.max(probs, dim=-1)

        return confidence

    def compute_sequence_confidence(self,
                                   logits: torch.Tensor,
                                   input_length: int,
                                   mask: torch.Tensor = None
                                   ) -> torch.Tensor:
        """Get confidence for entire generation sequence."""

        # Get token-level confidence
        token_conf = self.compute_token_confidence(logits)

        # Remove input tokens
        generation_conf = token_conf[input_length:]

        # Mask padding if provided
        if mask is not None:
            generation_mask = mask[input_length:]
            generation_conf = generation_conf * generation_mask

        # Sequence confidence: product of token confidence
        # (Use log for numerical stability)
        log_conf = torch.log(generation_conf + self.epsilon)
        log_conf = log_conf.sum(dim=0)
        sequence_conf = torch.exp(log_conf)

        return sequence_conf

    def compute_agreement_confidence(self,
                                     logits_list: list,
                                     input_length: int
                                     ) -> torch.Tensor:
        """Confidence based on agreement between samples."""

        # Get token probabilities for each sample
        all_probs = []
        for logits in logits_list:
            probs = torch.softmax(logits[input_length:], dim=-1)
            all_probs.append(probs)

        # Compute probability of agreement
        # P(sample_1 = sample_2) = sum_y p_1(y) * p_2(y)
        avg_probs = torch.stack(all_probs).mean(dim=0)
        agreement = (avg_probs ** 2).sum(dim=-1)

        return agreement.mean()
```

### Step 2: Create Self-Confidence Reward Signal

Design reward from confidence without external labels:

```python
class ConfidenceReward:
    def __init__(self, alpha: float = 0.1,
                 validity_checker = None):
        self.alpha = alpha
        self.validity_checker = validity_checker
        self.scorer = ConfidenceScorer()

    def compute_reward(self, completions: list,
                      logits_list: list,
                      input_length: int,
                      question: str = None
                      ) -> dict:
        """Compute reward based on confidence.

        R(y) = p(y|x) + α * validity_bonus(y)
        """

        rewards = []

        for completion, logits in zip(completions, logits_list):
            # Base reward: sequence confidence
            seq_conf = self.scorer.compute_sequence_confidence(
                logits,
                input_length
            )
            reward = seq_conf.item()

            # Optional validity bonus (e.g., format checking)
            if self.validity_checker is not None:
                is_valid = self.validity_checker(completion)
                reward += self.alpha * (1.0 if is_valid else -1.0)

            rewards.append(reward)

        return {
            "rewards": torch.tensor(rewards),
            "mean_confidence": torch.tensor(rewards).mean(),
            "max_confidence": torch.tensor(rewards).max()
        }

    def compute_majority_vote_signal(self,
                                     completions: list,
                                     logits_list: list,
                                     input_length: int
                                     ) -> torch.Tensor:
        """Reward based on agreement with majority."""

        # Find majority answer
        majority = max(set(completions),
                      key=completions.count)

        rewards = []
        for completion, logits in zip(completions, logits_list):
            if completion == majority:
                # Reward matching majority
                seq_conf = self.scorer.compute_sequence_confidence(
                    logits,
                    input_length
                )
                rewards.append(seq_conf.item())
            else:
                rewards.append(0.0)

        return torch.tensor(rewards)
```

### Step 3: Implement RLSC Training Loop

Train with confidence rewards and PPO:

```python
class RLSCTrainer:
    def __init__(self, model,
                 num_samples: int = 16,
                 batch_size: int = 4):
        self.model = model
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=5e-6
        )
        self.reward_fn = ConfidenceReward(alpha=0.1)

    def generate_completions(self, question: str,
                            num_samples: int = 16
                            ) -> tuple:
        """Generate multiple completions."""

        input_ids = self.model.tokenize(question)
        input_length = input_ids.shape[-1]

        completions = []
        logits_list = []

        for _ in range(num_samples):
            output = self.model.generate(
                input_ids,
                max_new_tokens=256,
                output_hidden_states=False,
                return_dict_in_generate=True,
                output_scores=True
            )

            # Detokenize
            completion = self.model.decode(output.sequences[0])
            completions.append(completion)

            # Collect logits
            logits = torch.stack(output.scores)
            logits_list.append(logits)

        return completions, input_ids, input_length, logits_list

    def compute_policy_gradient(self,
                               logits_list: list,
                               rewards: torch.Tensor,
                               input_length: int
                               ) -> torch.Tensor:
        """Compute policy gradient from rewards."""

        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        total_loss = 0.0

        for logits, reward in zip(logits_list, rewards):
            # Compute log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Get token-level log probs
            generation_log_probs = log_probs[input_length:]

            # Policy gradient: maximize log p * reward
            loss = -(generation_log_probs.mean() * reward)
            total_loss += loss

        return total_loss / len(logits_list)

    def train_step(self, questions: list) -> dict:
        """Single training step on batch of questions."""

        total_loss = 0.0
        total_reward = 0.0
        total_samples = 0

        for question in questions:
            # Generate completions
            completions, input_ids, input_length, logits_list = (
                self.generate_completions(
                    question,
                    num_samples=self.num_samples
                )
            )

            # Compute rewards
            reward_dict = self.reward_fn.compute_reward(
                completions,
                logits_list,
                input_length,
                question
            )
            rewards = reward_dict["rewards"]

            # Compute policy gradient
            loss = self.compute_policy_gradient(
                logits_list,
                rewards,
                input_length
            )

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_reward += rewards.mean().item()
            total_samples += 1

        return {
            "avg_loss": total_loss / total_samples,
            "avg_reward": total_reward / total_samples,
            "num_questions": total_samples
        }
```

### Step 4: Evaluate Improvements

Assess reasoning gains without labels:

```python
def evaluate_reasoning(model,
                      benchmark_questions: list,
                      reference_answers: list,
                      metric: str = "exact_match"
                      ) -> dict:
    """Evaluate on reasoning benchmarks."""

    correct = 0
    avg_confidence = 0.0
    answer_lengths = []

    for question, ref_answer in zip(benchmark_questions,
                                    reference_answers):
        # Generate completion
        completion = model.generate(
            model.tokenize(question),
            max_new_tokens=512
        )

        # Extract answer (assuming last line or similar)
        answer = extract_answer(completion)

        # Check correctness
        is_correct = evaluate_answer(answer, ref_answer, metric)
        if is_correct:
            correct += 1

        avg_confidence += extract_confidence(completion)
        answer_lengths.append(len(completion.split()))

    return {
        "accuracy": correct / len(benchmark_questions),
        "avg_confidence": avg_confidence / len(benchmark_questions),
        "avg_answer_length": sum(answer_lengths) / len(
            answer_lengths
        ),
        "num_eval": len(benchmark_questions)
    }
```

## Practical Guidance

**Sample Efficiency**: 16 samples per question is sufficient due to confidence being a strong signal. More samples improve stability but require more compute.

**Training Duration**: 10-20 training steps on 16-sample batches is typical. Longer training may overfit to confidence metric.

**Confidence as Prior**: The confidence metric works because high-probability outputs tend to be correct. The method implicitly assumes model calibration.

**Majority Voting Equivalence**: Using agreement-based rewards is similar to majority voting but enables gradient-based optimization via confidence signals.

**Emergent Conciseness**: Models often produce shorter, more confident answers after RLSC training—an emergent behavior that improves interpretability.

**When to Apply**: Use RLSC for mathematical reasoning tasks where you want to improve performance without labels, or when external reward models are unavailable.

## Reference

RLSC leverages a simple insight: confidence is sufficient training signal without external rewards. By maximizing F(p_θ) = E_y~p_θ[p_θ(y|x)], the method sharpens probability distributions toward high-confidence modes. Substantial gains (+13-21%) demonstrate confidence is a reliable proxy for reasoning quality in few-shot settings.
