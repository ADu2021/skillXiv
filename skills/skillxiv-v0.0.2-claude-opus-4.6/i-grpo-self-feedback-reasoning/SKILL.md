---
name: i-grpo-self-feedback-reasoning
title: "iGRPO: Self-Feedback-Driven LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.09000"
keywords: [Iterative Reasoning, Self-Feedback, Two-Stage Training, Policy Gradient, Self-Conditioning]
description: "Improve LLM reasoning through iterative refinement where the model refines its best previous attempts. Two-stage training: exploratory draft generation, then conditioned refinement using GRPO. Dynamic conditioning signals evolve with policy, enabling state-of-the-art math reasoning on AIME (85.62%)."
---

# iGRPO: Iterative Self-Feedback-Driven Reasoning

Standard RL reasoning processes each generation independently. iGRPO leverages human problem-solving patterns: draft a solution, review it, then refine. The method operates in two stages: Stage 1 generates k candidates and selects the best via reward function; Stage 2 appends this draft to the original prompt and generates refined solutions. As the policy improves, Stage 1 drafts become higher-quality, strengthening the conditioning signal for Stage 2.

This creates a bootstrapping effect where improvements feed forward through dynamic self-conditioning, enabling the model to learn iterative refinement patterns similar to human mathematical reasoning.

## Core Concept

Standard single-pass reasoning: prompt → draft → done. No opportunity for refinement.

iGRPO two-stage approach:
- **Stage 1 (Exploration)**: Sample k drafts, select best via reward
- **Stage 2 (Refinement)**: Append best draft to prompt, generate refined solution via GRPO
- **Dynamic Conditioning**: As policy improves, Stage 1 drafts improve, strengthening Stage 2 signals

Key insight: conditioning on evolved model outputs (not static examples) enables the model to learn task-specific refinement patterns.

## Architecture Overview

- **Stage 1 - Exploratory Draft**: Generate k candidates, select best using reward function
- **Stage 2 - Conditioned Refinement**: Append best draft to original prompt, apply GRPO training
- **Reward Function**: Verifiable signals (correctness, intermediate steps) guide Stage 1 selection
- **Dynamic Bootstrapping**: Stage 1 quality improves → Stage 2 conditioning improves → overall performance improves
- **Balanced Training**: Both stages optimized jointly; neither dominates

## Implementation

Implement two-stage generation:

```python
import torch
import torch.nn.functional as F

class iTwoStageReasoner:
    """Two-stage reasoning with iterative self-feedback."""

    def __init__(self, policy_model, reward_model, temperature=0.7):
        """
        Args:
            policy_model: Language model for generating solutions
            reward_model: Model to score solution quality
            temperature: Sampling temperature for diversity
        """
        self.policy = policy_model
        self.reward = reward_model
        self.temperature = temperature

    def stage1_exploratory_generation(self, prompt, num_drafts=4):
        """
        Stage 1: Generate k candidate drafts and select best.
        Args:
            prompt: Original problem statement
            num_drafts: Number of draft candidates to generate
        Returns:
            best_draft: Highest-scoring draft
            draft_scores: Scores for all drafts
        """
        drafts = []
        scores = []

        for _ in range(num_drafts):
            # Generate draft with sampling
            draft = self.policy.generate(
                prompt,
                max_tokens=300,
                temperature=self.temperature,
                do_sample=True
            )
            drafts.append(draft)

            # Score draft
            score = self.reward.score(prompt, draft)
            scores.append(score)

        # Select best
        best_idx = torch.tensor(scores).argmax()
        best_draft = drafts[best_idx]
        draft_scores = torch.tensor(scores)

        return best_draft, draft_scores

    def stage2_conditioned_refinement(self, prompt, best_draft, labels=None):
        """
        Stage 2: Refine best draft via conditioning and GRPO.
        Args:
            prompt: Original problem
            best_draft: Best draft from Stage 1
            labels: Ground-truth labels for reward computation
        Returns:
            refined_output: Refined solution
            loss: Training loss for optimization
        """
        # Create conditioned prompt
        conditioned_prompt = f"{prompt}\n\nPrevious attempt:\n{best_draft}\n\nRefined solution:"

        # Generate refined output
        refined = self.policy.generate(
            conditioned_prompt,
            max_tokens=300,
            temperature=0.5  # Lower temperature for refinement
        )

        # Compute reward for refined output
        reward = self.reward.score(prompt, refined, labels)

        # GRPO loss: maximize reward
        # (Simplified; full GRPO includes group-relative advantage)
        loss = -reward  # Maximize reward

        return refined, loss
```

Integrate full training loop:

```python
def igrpo_training_step(policy, reward_model, batch, optimizer, stage1_weight=0.5):
    """
    Single iGRPO training step: both stages.
    Args:
        policy: Language model
        reward_model: Reward scoring model
        batch: (prompts, labels) from training data
        optimizer: PyTorch optimizer
        stage1_weight: Balance between stage losses
    """
    prompts, labels = batch
    reasoner = iTwoStageReasoner(policy, reward_model)

    total_loss = 0.0

    for prompt, label in zip(prompts, labels):
        # Stage 1: Exploratory generation
        best_draft, draft_scores = reasoner.stage1_exploratory_generation(prompt, num_drafts=4)

        # Loss for Stage 1: maximize score of best draft
        # (In practice, optimize all drafts with advantage weighting)
        stage1_loss = -draft_scores.max()

        # Stage 2: Conditioned refinement
        refined, stage2_loss = reasoner.stage2_conditioned_refinement(prompt, best_draft, label)

        # Combined loss
        combined_loss = stage1_weight * stage1_loss + (1 - stage1_weight) * stage2_loss
        total_loss += combined_loss

    # Backward pass
    optimizer.zero_grad()
    (total_loss / len(prompts)).backward()
    optimizer.step()

    return total_loss.item() / len(prompts)

def train_igrpo(policy, reward_model, train_dataloader, num_epochs=10):
    """Full iGRPO training."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            loss = igrpo_training_step(policy, reward_model, batch, optimizer)
            total_loss += loss
            num_batches += 1

            if num_batches % 50 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch}, Batch {num_batches}: Loss = {avg_loss:.4f}")

        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed: Avg Loss = {avg_epoch_loss:.4f}")

    return policy
```

Implement inference with iterative refinement:

```python
def igrpo_inference(policy, reward_model, prompt, num_iterations=2, drafts_per_iteration=4):
    """
    Inference-time iterative refinement.
    Args:
        policy: Language model
        reward_model: Reward model
        prompt: Problem statement
        num_iterations: Number of refinement rounds
        drafts_per_iteration: Drafts per round
    Returns:
        final_answer: Best final answer
        reasoning_trace: Full reasoning history
    """
    reasoner = iTwoStageReasoner(policy, reward_model, temperature=0.7)
    reasoning_trace = []
    current_prompt = prompt

    for iteration in range(num_iterations):
        # Generate drafts
        best_draft, scores = reasoner.stage1_exploratory_generation(
            current_prompt,
            num_drafts=drafts_per_iteration
        )

        reasoning_trace.append({
            'iteration': iteration,
            'draft': best_draft,
            'score': scores.max().item()
        })

        if iteration < num_iterations - 1:
            # Prepare for next iteration: append draft and refine
            current_prompt = f"{current_prompt}\n\nIteration {iteration + 1} draft:\n{best_draft}"

    final_answer = best_draft
    return final_answer, reasoning_trace
```

## Practical Guidance

| Parameter | Recommendation | Notes |
|-----------|-----------------|-------|
| Stage 1 drafts | 4-8 | More drafts improve diversity; compute cost linear. |
| Stage 2 temperature | 0.3-0.5 | Lower than Stage 1; focus on refinement. |
| Stage 1 weight | 0.4-0.6 | Balance exploration (Stage 1) with refinement (Stage 2). |
| Iterations | 1-3 at inference | More iterations improve quality; diminishing returns after 3. |
| Reward model | Domain-specific | Use task-specific metrics (correctness, intermediate steps). |

**When to Use**
- Mathematical reasoning where refinement patterns are learnable
- Complex coding problems benefiting from draft-refine cycles
- Tasks with clear correctness signal for reward model
- Want to enable iterative improvement without changing architecture

**When NOT to Use**
- Simple tasks where single-pass sufficient
- No access to good reward model
- Inference latency critical (multiple rounds slower)

**Common Pitfalls**
- Stage 1 rewards too loose (all drafts equally good, no learning signal)
- Not conditioning Stage 2 properly (should clearly show previous draft)
- Stage weights unbalanced (one stage dominates, other stagnates)
- Reward model unreliable; cascading errors through iterations

## Reference

See https://arxiv.org/abs/2602.09000 for full training details, reward model design, and empirical results on AIME (85.62%), MATH500, and AMC benchmarks.
