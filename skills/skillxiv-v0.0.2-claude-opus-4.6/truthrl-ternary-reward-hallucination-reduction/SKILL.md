---
name: truthrl-ternary-reward-hallucination-reduction
title: "TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.25760"
keywords: [hallucination-reduction, RL-training, reward-design, truthfulness, LLM-safety]
description: "Reduce LLM hallucinations by training with a ternary reward signal that distinguishes correct answers, hallucinations, and abstentions. This technique incentivizes truthfulness over accuracy-only metrics, enabling safer, more calibrated language models through GRPO-based optimization."
---

# TruthRL: Ternary Reward System for Truthful LLMs

The tension in LLM training is fundamental: optimizing purely for accuracy amplifies hallucinations because models learn to generate plausible-sounding text even when uncertain. Conversely, overly conservative training sacrifices correct answers. TruthRL bridges this gap by explicitly modeling uncertainty as a training signal.

Traditional RL reward functions for LLMs treat outcomes as binary: correct or incorrect. This forces models to choose between two equally bad outcomes—confidently hallucinating or refusing to answer useful questions. The core insight is that **uncertainty awareness is a learnable behavior**, not an inherent limitation.

## Core Concept

TruthRL uses a **ternary reward structure** implemented with Group Relative Policy Optimization (GRPO):

- **Correct answers**: +1 reward (model provides accurate response)
- **Hallucinations**: -1 reward (model provides false information)
- **Abstentions**: +0.5 reward (model declines to answer when uncertain)

The key innovation is treating abstention as a positive signal. When the model recognizes the boundaries of its knowledge and says "I don't know," it receives reward—not punishment. This teaches the model to distinguish between what it reliably knows and what it should decline to claim.

## Architecture Overview

- **Training loop**: Standard GRPO pipeline with modified reward calculation
- **Reward function**: Ternary classification of model outputs
- **Optimization objective**: Policy gradients weighted by ternary rewards
- **Evaluation metric**: Truthfulness (correct answers + abstentions) vs. hallucination rate

## Implementation Steps

The ternary reward system requires a verifier that can classify each model output. Here's how to set up the reward signal:

```python
def compute_ternary_reward(output, ground_truth, verifier_model):
    """
    Classify output as correct, hallucination, or abstention.

    Args:
        output: Model-generated text
        ground_truth: Reference answer or fact
        verifier_model: Fine-tuned verifier for this domain

    Returns:
        reward: -1, 0.5, or 1.0
    """
    # Check if model abstained (standard abstention phrases)
    abstention_phrases = ["I don't know", "I'm not sure", "I cannot"]
    if any(phrase in output for phrase in abstention_phrases):
        return 0.5

    # Verify if output matches ground truth
    is_correct = verifier_model.verify(output, ground_truth)

    if is_correct:
        return 1.0  # Correct answer
    else:
        return -1.0  # Hallucination
```

The verifier can be a fine-tuned classifier, semantic matcher, or domain-specific checker. For factual domains (QA, knowledge), use exact-match or semantic similarity. For reasoning tasks, verify intermediate steps.

Next, integrate this into your GRPO training loop by replacing the standard binary reward:

```python
def grpo_training_step(prompts, model, verifier, group_size=4):
    """
    Single GRPO update with ternary rewards.

    Args:
        prompts: List of input prompts
        model: Policy model to optimize
        verifier: Ternary reward classifier
        group_size: Number of rollouts per prompt

    Returns:
        loss: Policy gradient loss
    """
    all_rewards = []
    all_logprobs = []

    # Generate multiple responses per prompt
    for prompt in prompts:
        group_responses = [model.generate(prompt) for _ in range(group_size)]
        group_rewards = [
            compute_ternary_reward(resp, prompt, verifier)
            for resp in group_responses
        ]

        # Compute advantages relative to group mean
        advantage = group_rewards - np.mean(group_rewards)

        # Get log probabilities under current policy
        logprobs = model.get_logprob(group_responses, prompt)

        all_rewards.extend(advantage)
        all_logprobs.extend(logprobs)

    # Policy gradient: maximize log(pi) * advantage
    loss = -torch.mean(torch.stack(all_logprobs) * torch.tensor(all_rewards))
    return loss
```

## Practical Guidance

**When to use TruthRL:**
- Open-ended QA where hallucinations are costly (medical, legal, financial advice)
- Knowledge-intensive tasks where confidence calibration matters
- Long-horizon agent reasoning where errors compound
- Safety-critical applications requiring transparency

**When NOT to use:**
- Creative writing or brainstorming (abstention punishes exploration)
- Tasks with ambiguous ground truth
- Low-resource domains where verifiers are unreliable
- Single-turn completion where efficiency trumps certainty

**Hyperparameter considerations:**

| Parameter | Recommended | Notes |
|-----------|------------|-------|
| Abstention reward | 0.5 | Balance between correctness and safety |
| Group size | 4-8 | Larger groups reduce variance but increase cost |
| Learning rate | 1e-6 to 5e-6 | Standard for GRPO fine-tuning |
| Verifier confidence threshold | 0.8+ | Avoid weak verifiers creating conflicting signals |
| Training epochs | 3-5 | Diminishing returns beyond 5 for most domains |

**Common pitfalls:**
- **Weak verifiers**: If your verifier makes mistakes, the reward signal becomes noisy. Validate verifier accuracy on held-out test set first (target >95% agreement).
- **Over-rewarding abstention**: If reward=0.5 is too high relative to ground_truth=1.0, the model learns to abstain excessively. Tune the abstention reward down if abstention rate exceeds 40%.
- **Domain mismatch**: Verifiers trained on one domain (Wikipedia QA) often fail on another (medical facts). Create domain-specific verifiers.
- **Ignoring verifier calibration**: Use confidence scores from the verifier; don't just threshold outputs. A soft ternary reward based on confidence prevents premature convergence.

**Integration checklist:**
- [ ] Develop or source a verifier with >95% accuracy on your domain
- [ ] Validate that your verifier's output aligns with human preference (spot-check 50 examples)
- [ ] Pilot on 1K examples to tune abstention reward before full-scale training
- [ ] Monitor hallucination rate and abstention rate separately during training
- [ ] Evaluate on a held-out test set using human raters (not just verifier accuracy)

Reference: https://arxiv.org/abs/2509.25760
