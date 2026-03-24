---
name: critique-rl-training
title: "Critique-RL: Training Language Models for Critiquing through Two-Stage RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.24320"
keywords: [RL, Feedback Learning, Critic Models, Two-stage Training, Reasoning]
description: "Trains language models to provide quality feedback through two-stage RL. Stage 1 optimizes discriminability (distinguishing good vs bad responses). Stage 2 adds helpfulness rewards (improving actor after feedback). Achieves 9.02% improvement without requiring stronger supervisors for training data."
---

# Critique-RL: Learning to Provide Effective Feedback

Standard RL for critic models focuses on generating feedback, but doesn't ensure feedback quality. Critique-RL uses two-stage training to develop both discriminative ability (telling good from bad) and helpfulness (guiding improvement).

The approach enables cheaper critic training without external supervisors.

## Core Concept

Two-stage reinforcement learning:
- **Stage 1**: Optimize discriminability via direct reward signals
- **Stage 2**: Optimize helpfulness via actor improvement signals
- Stage 1 prevents feedback collapse into neutral comments
- Stage 2 ensures feedback actually helps (no false positives)

## Architecture Overview

- Actor-Critic pair: actor generates responses, critic evaluates
- Rule-based reward for Stage 1 (quality assessment)
- Actor improvement reward for Stage 2 (helpfulness)
- Regularization to preserve Stage 1 discriminability

## Implementation Steps

Implement the two-stage training pipeline with reward signals:

```python
class TwoStageCriticRL:
    def __init__(self, critic_model, actor_model):
        self.critic = critic_model
        self.actor = actor_model
        self.critic_optimizer = torch.optim.AdamW(critic_model.parameters())
        self.stage = 'discriminability'

    def stage1_discriminability(self, good_responses, bad_responses):
        """Stage 1: Train to distinguish quality levels."""
        # Direct reward: critic scores good higher than bad
        for good, bad in zip(good_responses, bad_responses):
            good_score = self.critic(good)['quality']
            bad_score = self.critic(bad)['quality']

            # Margin loss: good > bad by margin
            loss = -torch.log(torch.sigmoid(good_score - bad_score))

            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

    def stage2_helpfulness(self, actor_feedback_pairs, num_steps=5):
        """Stage 2: Ensure feedback helps actor improve."""
        for prompt, initial_response in actor_feedback_pairs:
            # Get critic feedback on initial response
            feedback = self.critic(initial_response)['text']

            # Use feedback to improve actor response
            improved = self._improve_via_feedback(
                initial_response, feedback, num_steps
            )

            # Reward based on actual improvement
            initial_score = self._evaluate_response(initial_response)
            improved_score = self._evaluate_response(improved)
            improvement = improved_score - initial_score

            # RL loss: maximize improvement
            if improvement > 0:
                feedback_score = self.critic(initial_response)['helpfulness']
                rl_loss = -feedback_score * improvement

                self.critic_optimizer.zero_grad()
                rl_loss.backward()
                self.critic_optimizer.step()

            # Regularization: preserve discriminability
            self._discriminability_regularization()

    def _improve_via_feedback(self, response, feedback, num_steps):
        """Use critic feedback to refine actor response."""
        current = response
        for _ in range(num_steps):
            improved = self.actor(current + "\n[FEEDBACK] " + feedback)
            current = improved
        return current

    def _evaluate_response(self, response):
        """Score response quality (external metric or proxy)."""
        return self.critic(response)['quality']

    def _discriminability_regularization(self):
        """Preserve Stage 1 discriminability during Stage 2."""
        # Periodically validate discriminability hasn't degraded
        validation_samples = []  # Hold-out set
        for good, bad in validation_samples:
            good_score = self.critic(good)['quality']
            bad_score = self.critic(bad)['quality']

            # If good < bad, add penalty
            if good_score < bad_score:
                penalty = torch.nn.functional.relu(bad_score - good_score)
                self.critic_optimizer.zero_grad()
                penalty.backward()
                self.critic_optimizer.step()
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Stage 1 iterations | 1000-2000 |
| Stage 2 iterations | 500-1000 |
| Margin threshold | 0.5 |
| Discriminability regularization weight | 0.1-0.2 |

**When to use:**
- Training critic models from scratch
- Improving feedback quality in RLHF pipelines
- Scenarios without expensive external judges

**When NOT to use:**
- When high-quality feedback is available (supervised learning better)
- Real-time systems (two-stage training adds overhead)

**Common pitfalls:**
- Stage 1 overfit to training quality signal
- Stage 2 not regularizing Stage 1 (discriminability collapse)
- Insufficient Stage 1 data (weak discriminator)

Reference: [Critique-RL on arXiv](https://arxiv.org/abs/2510.24320)
