---
name: rlad-knowledge-distillation
title: "Reinforcement-aware Knowledge Distillation for LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.22495"
keywords: [Knowledge Distillation, Reinforcement Learning, LLM Efficiency, Student-Teacher, GRPO]
description: "RLAD performs selective distillation during RL, using trust-region ratio objectives to balance exploration, exploitation, and teacher guidance for efficient reasoning model training."
---

# Technique: Reinforcement-Aware Distillation (RLAD)

Training smaller language models for reasoning typically fails because either (1) pure imitation of teacher solutions exceeds student capacity, or (2) RL training diverges from the teacher's knowledge, losing the guidance signal. The fundamental problem: standard knowledge distillation uses fixed KL regularization that conflicts with RL's policy optimization objectives.

RLAD solves this by performing selective distillation only when it improves the RL policy, and using trust-region ratio objectives (PPO/GRPO-style) instead of KL divergence. This allows students to explore and discover novel solutions while staying grounded in teacher knowledge, without training instability.

## Core Concept

The core insight: during RL, guide the student toward the teacher only when doing so improves the policy (increases expected reward). This requires a principled way to combine three competing objectives:

1. **Exploration**: Discover novel reasoning paths
2. **Exploitation**: Maximize rewards from discovered solutions
3. **Imitation**: Stay close to teacher's distribution

RLAD uses Trust Region Ratio Distillation (TRRD): replace the standard KL regularizer with a PPO-style likelihood-ratio objective. This naturally balances imitation and policy improvement by keeping trust regions—staying close to the teacher only when beneficial.

## Architecture Overview

- **Teacher Model**: Large frozen model (e.g., Qwen 32B or Gemini)
- **Student Model**: Smaller trainable model (e.g., Qwen 1.5B)
- **Reward Function**: Verifiable signal (correctness, code execution, etc.)
- **GRPO/PPO Loop**: Update student with trust-region policy gradients
- **Selective Distillation**: Only apply imitation loss when reward improves

## Implementation Steps

RLAD integrates into standard GRPO/PPO training loops. Here's how to implement it:

First, compute the trust region ratio for each student output—this tells us how far the student diverges from the teacher:

```python
import torch
import torch.nn.functional as F

def compute_trust_region_ratio(
    student_log_probs,      # [batch_size]
    teacher_log_probs,      # [batch_size]
):
    """
    Compute likelihood ratio: p_student(y) / p_teacher(y)
    Used to determine when to apply distillation.
    """
    log_ratio = student_log_probs - teacher_log_probs
    ratio = torch.exp(log_ratio)
    return ratio

def trust_region_ratio_distillation_loss(
    student_log_probs,
    teacher_log_probs,
    advantages,             # From RL algorithm (GRPO/PPO)
    beta_d=0.5,            # Distillation coefficient
    epsilon=0.2,           # Trust region bound
):
    """
    Trust Region Ratio Distillation Loss (TRRD).
    Combines policy gradient with selective teacher guidance.
    """
    ratio = compute_trust_region_ratio(student_log_probs, teacher_log_probs)

    # Policy gradient objective (standard PPO)
    policy_loss = -advantages * torch.log(ratio + 1e-8)

    # Selective distillation: only apply when student ratio is within trust region
    # If ratio > (1+epsilon), student over-confident; bring back to teacher
    # If ratio < (1-epsilon), student under-confident; allow exploration
    in_trust_region = (ratio >= (1 - epsilon)) & (ratio <= (1 + epsilon))

    # Distillation loss: KL between student and teacher, scaled by trust region
    kl_loss = (student_log_probs - teacher_log_probs).abs()
    distillation_loss = beta_d * in_trust_region.float() * kl_loss

    # Combined loss: policy gradient + selective distillation
    total_loss = policy_loss + distillation_loss
    return total_loss.mean()
```

Integrate RLAD into your GRPO training loop:

```python
class RLADTrainer:
    def __init__(
        self,
        student_model,
        teacher_model,
        reward_fn,
        learning_rate=1e-5,
        beta_d=0.5,
        epsilon=0.2,
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.reward_fn = reward_fn
        self.optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=learning_rate
        )
        self.beta_d = beta_d
        self.epsilon = epsilon

    def generate_with_teacher(self, prompts, max_new_tokens=128):
        """Generate responses from both student and teacher."""
        # Student generation
        student_outputs = self.student.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Teacher generation (for reference; optional)
        with torch.no_grad():
            teacher_outputs = self.teacher.generate(
                prompts,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        return student_outputs, teacher_outputs

    def compute_advantages(self, responses, rewards):
        """
        Compute advantage estimates from rewards.
        In practice, use more sophisticated advantage estimation (GAE, etc.).
        """
        advantages = rewards - rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        return advantages

    def training_step(self, prompts, max_new_tokens=128):
        """Single RLAD training step."""
        # Generate responses
        student_outputs, teacher_outputs = self.generate_with_teacher(
            prompts, max_new_tokens
        )

        student_responses = student_outputs.sequences
        teacher_responses = teacher_outputs.sequences

        # Compute rewards
        rewards = torch.tensor([
            self.reward_fn(response) for response in student_responses
        ], device=self.student.device, dtype=torch.float32)

        # Compute advantages
        advantages = self.compute_advantages(student_responses, rewards)

        # Compute log-probabilities for student and teacher
        with torch.no_grad():
            student_logits = self.student(student_responses).logits
            teacher_logits = self.teacher(teacher_responses).logits

        # For simplicity, average log-probs over sequence
        # In practice, compute per-token and properly align
        student_log_probs = F.log_softmax(student_logits[:, -1, :], dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits[:, -1, :], dim=-1)

        # Extract log-probs for generated tokens
        # (Simplified; real implementation tracks per-token scores)
        avg_student_log_prob = student_log_probs.mean()
        avg_teacher_log_prob = teacher_log_probs.mean()

        # Compute RLAD loss
        loss = trust_region_ratio_distillation_loss(
            avg_student_log_prob.unsqueeze(0),
            avg_teacher_log_prob.unsqueeze(0),
            advantages[:1],
            beta_d=self.beta_d,
            epsilon=self.epsilon,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'reward_mean': rewards.mean().item(),
            'reward_max': rewards.max().item(),
        }

    def train_epoch(self, dataloader, num_epochs=3):
        """Train for multiple epochs."""
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_reward = 0.0
            num_batches = 0

            for batch in dataloader:
                stats = self.training_step(batch['prompts'])
                total_loss += stats['loss']
                total_reward += stats['reward_mean']
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_reward = total_reward / num_batches
            print(
                f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                f"Reward={avg_reward:.4f}"
            )
```

## Practical Guidance

**When to Use:**
- Training student models for reasoning (math, code, logic)
- When you have a larger teacher model available
- When you want to preserve teacher knowledge while exploring new solutions
- For efficient deployment (1.5B/3B student models)

**When NOT to Use:**
- Simple classification or generation tasks (overkill)
- When student and teacher have similar capacities
- Real-time inference (training is offline, but requires teacher for initial guidance)

**Hyperparameters:**
- `beta_d`: Distillation weight (0.1–1.0). Higher = stronger teacher guidance
- `epsilon`: Trust region bound (0.1–0.3). Controls exploration freedom
- `learning_rate`: Typically lower for RL (1e-6 to 1e-5)

**Implementation Details:**
- Use same tokenizer for student and teacher
- Keep teacher frozen; only update student
- Per-token distillation is more accurate than sequence-level
- Combine with standard GRPO/PPO techniques

**Performance:**
- Student typically reaches 90–95% of teacher performance
- Outperforms pure GRPO and offline distillation baselines
- Stable training without reward collapse
- Consistent improvements across benchmarks (AIME, MATH, code tasks)

---

**Reference:** [Reinforcement-aware Knowledge Distillation for LLM Reasoning](https://arxiv.org/abs/2602.22495)
