---
name: exgrpo-learning-from-experience
title: "ExGRPO: Learning to Reason from Experience"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.02245"
keywords: [experience-replay, RL-reasoning, GRPO, sample-efficiency, reasoning-training]
description: "Improve LLM reasoning efficiency by systematically reusing past rollouts through experience replay. ExGRPO organizes training data by success and diversity, applying a mixed-policy objective that prioritizes high-quality examples while maintaining exploration, achieving 3.5-7.6 point gains over on-policy methods."
---

# ExGRPO: Experiential Group Relative Policy Optimization

Standard reinforcement learning for LLM reasoning discards data after a single gradient step. This is wasteful because good rollouts—ones that succeeded or showed diverse reasoning—are valuable for multiple training iterations. The challenge is that off-policy learning introduces distribution shift: if you trained on old data, current predictions diverge from that distribution, causing unstable gradients.

ExGRPO addresses this by intelligently organizing and reusing past rollouts. Rather than treating all experiences equally, it prioritizes successful completions and diverse outputs, then applies a mixed-policy objective that balances exploitation of good examples with exploration of new reasoning paths.

## Core Concept

ExGRPO organizes experience replay around two signals:

1. **Success signal**: Did the rollout solve the problem?
2. **Diversity signal**: How different is this rollout's reasoning from others?

High-quality examples (successful and diverse) are retained and replayed multiple times across training. A mixed-policy objective prevents overexploitation: gradient updates are weighted by both the policy's likelihood (exploitation) and a uniform baseline (exploration), creating a natural curriculum where early training explores broadly, then focuses on promising directions.

## Architecture Overview

- **Rollout collector**: Generate reasoning trajectories via forward sampling
- **Experience scorer**: Rate rollouts on success and diversity
- **Replay buffer**: Organize high-quality examples by tier (top 25%, 50%, etc.)
- **Mixed-policy optimizer**: Balance on-policy and off-policy updates
- **Curriculum scheduler**: Gradually shift focus from exploration to exploitation

## Implementation Steps

Start by implementing the experience scoring system:

```python
import torch
import numpy as np
from collections import defaultdict

class ExperienceScorer:
    """
    Score rollouts on success and diversity.
    """
    def __init__(self, verifier_model, embedding_model):
        self.verifier = verifier_model  # Checks if solution is correct
        self.embedder = embedding_model  # Encodes reasoning traces

    def score_rollout(self, rollout, problem, previous_rollouts):
        """
        Compute success and diversity scores for a rollout.

        Args:
            rollout: Generated reasoning trace and solution
            problem: Problem statement
            previous_rollouts: Other rollouts for this problem (for diversity)

        Returns:
            quality_score: Combined metric (0-1)
            success: Binary correctness (0/1)
            diversity: How different from previous attempts (0-1)
        """
        # Success: does the solution verify as correct?
        success = float(self.verifier.check(rollout["solution"], problem))

        # Diversity: embed reasoning, compare to previous
        rollout_embedding = self.embedder.encode(rollout["reasoning"])

        if previous_rollouts:
            previous_embeddings = [
                self.embedder.encode(r["reasoning"])
                for r in previous_rollouts
            ]

            # Diversity = average distance to previous rollouts
            distances = [
                torch.nn.functional.cosine_similarity(
                    rollout_embedding.unsqueeze(0),
                    pe.unsqueeze(0)
                ).item()
                for pe in previous_embeddings
            ]

            # Invert cosine similarity (1 - similarity = distance)
            diversity = 1.0 - np.mean(distances)
        else:
            diversity = 1.0  # First rollout is maximally diverse

        # Combine: success weighted higher than diversity
        quality_score = 0.7 * success + 0.3 * diversity

        return {
            "quality": quality_score,
            "success": success,
            "diversity": diversity
        }

    def score_batch(self, rollouts, problem, previous_all):
        """
        Score multiple rollouts for a problem.

        Args:
            rollouts: List of rollouts
            problem: Problem statement
            previous_all: All previous rollouts for this problem

        Returns:
            scores: List of scoring dictionaries
        """
        scores = []
        for rollout in rollouts:
            score = self.score_rollout(rollout, problem, previous_all)
            scores.append(score)

        return scores
```

Now implement the experience replay buffer organized by quality tiers:

```python
class ExperienceBuffer:
    """
    Organize high-quality experiences for efficient replay.
    """
    def __init__(self, max_size=100000, num_tiers=4):
        self.max_size = max_size
        self.num_tiers = num_tiers
        self.buffer = defaultdict(list)  # tier -> [experiences]
        self.size = 0

    def add_rollouts(self, problem_id, rollouts, scores):
        """
        Add scored rollouts to buffer, organizing by quality.

        Args:
            problem_id: Which problem these rollouts are for
            rollouts: List of reasoning traces
            scores: Quality scores from ExperienceScorer
        """
        # Sort by quality score
        ranked = sorted(
            zip(rollouts, scores),
            key=lambda x: x[1]["quality"],
            reverse=True
        )

        # Assign to tiers: top 25% -> tier 0, next 25% -> tier 1, etc.
        tier_size = len(ranked) // self.num_tiers
        for tier_idx in range(self.num_tiers):
            start = tier_idx * tier_size
            end = (tier_idx + 1) * tier_size if tier_idx < self.num_tiers - 1 else len(ranked)

            for rollout, score in ranked[start:end]:
                self.buffer[tier_idx].append({
                    "problem_id": problem_id,
                    "rollout": rollout,
                    "score": score
                })

                self.size += 1

                # Evict old examples if buffer full
                if self.size > self.max_size:
                    self._evict_oldest(tier_idx)

    def sample_batch(self, batch_size, tier_distribution=None):
        """
        Sample a batch prioritizing high-quality tiers.

        Args:
            batch_size: How many examples to sample
            tier_distribution: Probability of sampling from each tier
                               (e.g., [0.5, 0.3, 0.15, 0.05])

        Returns:
            batch: List of (rollout, score) tuples
        """
        if tier_distribution is None:
            # Default: prefer top tiers exponentially
            tier_distribution = [0.5, 0.3, 0.15, 0.05]

        batch = []
        for tier_idx, prob in enumerate(tier_distribution):
            tier_batch_size = int(batch_size * prob)

            if tier_idx in self.buffer and len(self.buffer[tier_idx]) > 0:
                samples = np.random.choice(
                    len(self.buffer[tier_idx]),
                    size=min(tier_batch_size, len(self.buffer[tier_idx])),
                    replace=True
                )

                for idx in samples:
                    batch.append(self.buffer[tier_idx][idx])

        return batch

    def _evict_oldest(self, tier):
        """Remove oldest example from tier."""
        if self.buffer[tier]:
            self.buffer[tier].pop(0)
            self.size -= 1
```

Now implement the mixed-policy training objective:

```python
def mixed_policy_grpo_update(
    model,
    batch,
    uniform_baseline_logprobs,
    group_size=4,
    on_policy_weight=0.5
):
    """
    GRPO update with mixed on-policy and off-policy terms.

    Args:
        model: Policy to optimize
        batch: Sampled experiences (mostly old data)
        uniform_baseline_logprobs: Log prob under uniform policy
        group_size: GRPO group size for advantage normalization
        on_policy_weight: Balance between on-policy and off-policy

    Returns:
        loss: Policy gradient loss
    """
    all_logprobs = []
    all_advantages = []
    all_rewards = []

    # Process batch as groups for advantage normalization
    for i in range(0, len(batch), group_size):
        group = batch[i : i + group_size]

        # Get log probabilities under current policy
        logprobs = []
        for experience in group:
            rollout = experience["rollout"]
            logprob = model.get_logprob(rollout["reasoning"], rollout["problem"])
            logprobs.append(logprob)

        logprobs = torch.tensor(logprobs)

        # Extract rewards from scores
        rewards = torch.tensor([
            exp["score"]["quality"]
            for exp in group
        ])

        # Compute advantages relative to group mean
        advantages = rewards - rewards.mean()

        all_logprobs.append(logprobs)
        all_advantages.append(advantages)
        all_rewards.append(rewards)

    all_logprobs = torch.cat(all_logprobs)
    all_advantages = torch.cat(all_advantages)
    all_rewards = torch.cat(all_rewards)

    # Mixed objective: blend on-policy and off-policy
    # On-policy term: standard policy gradient
    on_policy_loss = -(all_logprobs * all_advantages.detach()).mean()

    # Off-policy term: importance-weighted (using uniform baseline)
    importance_weights = torch.exp(all_logprobs - uniform_baseline_logprobs)
    importance_weights = torch.clamp(importance_weights, max=5.0)

    off_policy_loss = -(importance_weights * all_advantages.detach()).mean()

    # Combined loss with learnable weighting
    loss = on_policy_weight * on_policy_loss + (1 - on_policy_weight) * off_policy_loss

    return loss
```

Finally, implement the full training loop with experience replay:

```python
def train_with_experience_replay(
    model,
    problems,
    num_epochs=5,
    batch_size=32,
    num_rollouts_per_problem=4
):
    """
    Train reasoning model using ExGRPO with experience replay.

    Args:
        model: LLM to train
        problems: List of reasoning problems
        num_epochs: Training epochs
        batch_size: Batch size for optimization
        num_rollouts_per_problem: Rollouts per problem (for diversity)

    Returns:
        model: Trained model
    """
    scorer = ExperienceScorer(verifier_model, embedding_model)
    buffer = ExperienceBuffer(max_size=100000)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Collect fresh rollouts
        for problem in problems:
            rollouts = []
            for _ in range(num_rollouts_per_problem):
                # Generate rollout from current policy
                rollout = model.generate_reasoning(problem)
                rollouts.append(rollout)

            # Score rollouts
            scores = scorer.score_batch(
                rollouts,
                problem,
                buffer.buffer.get(problem["id"], [])
            )

            # Add to experience buffer
            buffer.add_rollouts(problem["id"], rollouts, scores)

        # Train on mixed batch (mostly replay, some fresh)
        total_loss = 0
        num_batches = len(problems) * num_rollouts_per_problem // batch_size

        for batch_idx in range(num_batches):
            # Sample batch prioritizing high-quality tiers
            batch = buffer.sample_batch(batch_size)

            if len(batch) == 0:
                continue

            # Compute uniform baseline for off-policy correction
            uniform_logprobs = torch.tensor([
                -np.log(model.vocab_size)
                for _ in batch
            ])

            # Update with mixed objective
            loss = mixed_policy_grpo_update(
                model,
                batch,
                uniform_logprobs,
                on_policy_weight=max(0.3, 0.7 - epoch * 0.1)  # Curriculum
            )

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Loss: {avg_loss:.4f}")

    return model
```

## Practical Guidance

**When to use ExGRPO:**
- Reasoning tasks (math, logic, code) where high-quality examples are rare
- Sample-efficient training (limited problems available)
- Iterative model refinement (continuous improvement from experience)
- Mixed-size model training (works well from 1.5B to 70B)

**When NOT to use:**
- One-shot generation tasks (experience replay adds overhead)
- High-quality fully-labeled datasets (standard supervised fine-tuning is simpler)
- Extreme distribution shift scenarios (off-policy corrections may not hold)
- Real-time learning (buffer management adds complexity)

**Empirical improvements over on-policy GRPO:**

| Model Size | GRPO Baseline | ExGRPO Gain | Cumulative Quality |
|-----------|---|---|---|
| 1.5B | 28% | +3.5 pts | 31.5% |
| 7B | 48% | +5.2 pts | 53.2% |
| 8B | 52% | +7.6 pts | 59.6% |

**Key hyperparameters:**

| Parameter | Default | Tuning Impact |
|-----------|---------|--------------|
| num_tiers | 4 | More tiers = finer quality control (but more complex) |
| tier_distribution | [0.5, 0.3, 0.15, 0.05] | Shift weights to exploit vs explore |
| on_policy_weight | Start 0.7, decay to 0.3 | Higher = stay on-policy longer |
| max_buffer_size | 100K | Increase if storage available |
| num_rollouts_per_problem | 4-8 | More rollouts = better diversity scoring |

**Common pitfalls:**
- **Stale data dominance**: If you keep replaying old rollouts too long, the policy diverges. Apply importance weighting and cap ratio to 5.0.
- **Tier distribution too aggressive**: If you over-exploit tier 0, diversity collapses. Ensure tier_distribution maintains exploration (keep tail distributions >5%).
- **Weak verifier**: If success scoring is inaccurate, quality tier organization fails. Validate verifier on 100 examples before full training.
- **Buffer pollution**: Don't replay failing rollouts. Score everything; only buffer experiences with quality >0.3.

**Integration checklist:**
- [ ] Implement and validate experience scorer on 50 sample rollouts
- [ ] Start with small buffer (1000 examples) to validate tier organization
- [ ] Measure success rate and diversity metrics separately
- [ ] Compare ExGRPO to GRPO baseline on same problems
- [ ] Monitor buffer composition: ensure all tiers remain populated
- [ ] Evaluate on held-out problems to confirm generalization

Reference: https://arxiv.org/abs/2510.02245
