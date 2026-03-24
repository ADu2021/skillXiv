---
name: chord-on-policy-off-policy-harmonization
title: "CHORD: Harmonizing SFT and RL via Dynamic Weighting"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.11408
keywords: [reinforcement-learning, supervised-fine-tuning, dynamic-weighting, policy-learning, llm-training]
description: "Harmonize supervised fine-tuning and reinforcement learning through dynamic weighting, balancing expert imitation and on-policy exploration to prevent response pattern disruption."
---

# CHORD: Harmonizing SFT and RL via Dynamic Weighting

## Core Concept

CHORD unifies Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) through a dual-control mechanism that maintains learned response patterns while enabling exploratory on-policy learning. Rather than treating SFT as a separate preliminary stage, CHORD integrates it as a dynamically weighted auxiliary objective within RL. A global coefficient guides the overall transition from imitation to exploration, while token-wise weighting enables fine-grained learning from expert data while suppressing off-policy interference.

## Architecture Overview

- **Global Weighting Coefficient**: Interpolates between SFT (expert imitation) and RL (on-policy exploration)
- **Token-Wise Weighting Function**: Fine-grained control at token level, reducing off-policy interference
- **Unified On/Off-Policy Framework**: Treats SFT as off-policy learning and RL as on-policy within same formulation
- **Expert Data Preservation**: Maintains quality of established response patterns during RL training
- **Adaptive Transition**: Smooth interpolation preventing catastrophic forgetting

## Implementation Steps

### 1. Define the Unified On/Off-Policy Formulation

Establish the mathematical framework combining SFT and RL:

```python
from typing import Tuple
import torch
import torch.nn.functional as F

def unified_policy_objective(
    model_logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    expert_logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    rl_rewards: torch.Tensor,  # (batch_size, seq_len)
    global_alpha: float = 0.5,  # [0=pure RL, 1=pure SFT]
    token_weights: torch.Tensor = None  # (batch_size, seq_len)
) -> Tuple[torch.Tensor, dict]:
    """
    Compute unified CHORD objective combining SFT and RL.

    Loss = alpha * SFT_loss + (1 - alpha) * RL_loss
    Token weights modulate per-token contribution.
    """
    batch_size, seq_len, vocab_size = model_logits.shape

    # Compute SFT loss (off-policy imitation)
    sft_loss = F.cross_entropy(
        model_logits.view(-1, vocab_size),
        expert_logits.argmax(-1).view(-1),
        reduction='none'
    ).view(batch_size, seq_len)

    # Compute RL loss (on-policy reward maximization)
    # Using policy gradient with reward baseline
    model_probs = F.softmax(model_logits, dim=-1)
    expert_probs = F.softmax(expert_logits, dim=-1)

    # KL divergence for policy regularization
    kl_loss = torch.sum(
        expert_probs * (torch.log(expert_probs) - torch.log(model_probs)),
        dim=-1
    )

    # Policy gradient: log prob * advantage
    rl_loss = -rl_rewards + 0.1 * kl_loss  # KL penalty to prevent drift

    # Apply token-wise weighting
    if token_weights is None:
        token_weights = torch.ones_like(sft_loss)

    weighted_sft = (sft_loss * token_weights).mean()
    weighted_rl = (rl_loss * token_weights).mean()

    # Combine with global coefficient
    total_loss = global_alpha * weighted_sft + (1.0 - global_alpha) * weighted_rl

    metrics = {
        "sft_loss": weighted_sft.item(),
        "rl_loss": weighted_rl.item(),
        "total_loss": total_loss.item(),
        "alpha": global_alpha
    }

    return total_loss, metrics
```

### 2. Implement Global Coefficient Scheduling

Create curriculum that transitions from imitation to exploration:

```python
class GlobalCoefficientScheduler:
    def __init__(
        self,
        initial_alpha: float = 1.0,  # Start with pure SFT
        final_alpha: float = 0.2,    # End with mostly RL
        total_steps: int = 10000,
        schedule_type: str = "linear"
    ):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_step = 0

    def get_alpha(self, step: int = None) -> float:
        """Compute global weighting coefficient for current step."""
        if step is None:
            step = self.current_step

        progress = min(step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            alpha = self.initial_alpha - (self.initial_alpha - self.final_alpha) * progress

        elif self.schedule_type == "cosine":
            # Cosine annealing for smooth transition
            import math
            alpha = self.final_alpha + 0.5 * (self.initial_alpha - self.final_alpha) * \
                   (1 + math.cos(math.pi * progress))

        elif self.schedule_type == "exponential":
            # Exponential decay
            import math
            alpha = self.final_alpha + (self.initial_alpha - self.final_alpha) * \
                   math.exp(-5 * progress)

        return alpha

    def step(self) -> float:
        """Advance schedule and return current alpha."""
        alpha = self.get_alpha(self.current_step)
        self.current_step += 1
        return alpha
```

### 3. Implement Token-Wise Weighting Function

Compute fine-grained weights that suppress off-policy interference:

```python
def compute_token_weights(
    model_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    rl_rewards: torch.Tensor,
    weighting_strategy: str = "adaptive"
) -> torch.Tensor:
    """
    Compute token-level weights balancing expert data learning and exploration.

    Strategies:
    - adaptive: Weight based on expert confidence and reward signal alignment
    - entropy: Weight by model confidence (high confidence = high weight for RL)
    - margin: Weight by prediction margin between expert and model
    """
    batch_size, seq_len, vocab_size = model_logits.shape

    if weighting_strategy == "adaptive":
        # Expert confidence: how concentrated is expert distribution?
        expert_probs = F.softmax(expert_logits, dim=-1)
        expert_entropy = -torch.sum(
            expert_probs * torch.log(expert_probs + 1e-10),
            dim=-1
        )
        expert_confidence = 1.0 - (expert_entropy / torch.log(torch.tensor(vocab_size)))

        # Reward alignment: is RL reward signal strong?
        reward_magnitude = torch.abs(rl_rewards)
        normalized_reward = (reward_magnitude - reward_magnitude.min()) / \
                           (reward_magnitude.max() - reward_magnitude.min() + 1e-10)

        # Adaptive weight: high expert confidence -> favor SFT (weight ~1)
        #                 low expert confidence -> favor RL (weight ~0)
        weights = expert_confidence * (1.0 - 0.5 * normalized_reward)

    elif weighting_strategy == "entropy":
        # High model confidence -> rely on RL signal (low weight for SFT)
        model_probs = F.softmax(model_logits, dim=-1)
        model_entropy = -torch.sum(
            model_probs * torch.log(model_probs + 1e-10),
            dim=-1
        )
        model_confidence = 1.0 - (model_entropy / torch.log(torch.tensor(vocab_size)))
        weights = 1.0 - model_confidence  # Inverted for SFT weight

    elif weighting_strategy == "margin":
        # Margin-based weighting
        model_probs = F.softmax(model_logits, dim=-1)
        expert_probs = F.softmax(expert_logits, dim=-1)

        model_top_prob = model_probs.max(dim=-1)[0]
        expert_top_prob = expert_probs.max(dim=-1)[0]

        # Large margin -> model confident -> favor RL
        margin = expert_top_prob - model_top_prob
        weights = torch.clamp(margin, 0, 1)

    else:
        weights = torch.ones(batch_size, seq_len)

    # Normalize to [0, 1]
    weights = torch.clamp(weights, 0, 1)
    return weights
```

### 4. Implement CHORD Training Loop

Integrate components into unified training procedure:

```python
class CHORDTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        initial_alpha: float = 1.0,
        final_alpha: float = 0.2,
        total_steps: int = 10000,
        weighting_strategy: str = "adaptive"
    ):
        self.model = model
        self.optimizer = optimizer
        self.alpha_scheduler = GlobalCoefficientScheduler(
            initial_alpha, final_alpha, total_steps
        )
        self.weighting_strategy = weighting_strategy
        self.current_step = 0

    def train_step(
        self,
        batch_inputs: torch.Tensor,  # (batch, seq_len)
        expert_outputs: torch.Tensor,  # Expert demonstrations
        rl_rewards: torch.Tensor,  # Reward signals from environment
    ) -> dict:
        """Execute single CHORD training step."""

        # Forward pass
        model_logits = self.model(batch_inputs)

        # Expert logits (from reference model or dataset)
        expert_logits = self.get_expert_logits(expert_outputs)

        # Get current alpha from scheduler
        alpha = self.alpha_scheduler.step()

        # Compute token-wise weights
        token_weights = compute_token_weights(
            model_logits,
            expert_logits,
            rl_rewards,
            self.weighting_strategy
        )

        # Compute unified loss
        loss, metrics = unified_policy_objective(
            model_logits,
            expert_logits,
            rl_rewards,
            global_alpha=alpha,
            token_weights=token_weights
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        metrics["step"] = self.current_step
        metrics["avg_weight"] = token_weights.mean().item()

        self.current_step += 1
        return metrics

    def get_expert_logits(self, expert_outputs):
        """Get logits from expert data."""
        # In practice: encode expert outputs through reference model
        # or use cached teacher logits
        pass
```

### 5. Validate Training Dynamics

Monitor prevention of catastrophic forgetting:

```python
def evaluate_chord_training(
    model: torch.nn.Module,
    sft_validation_set,
    rl_validation_set,
    alpha: float
) -> dict:
    """
    Evaluate CHORD training:
    - SFT performance shouldn't degrade
    - RL performance should improve
    """
    model.eval()

    with torch.no_grad():
        # SFT metrics: does model still follow expert demonstrations?
        sft_accuracy = evaluate_sft_accuracy(model, sft_validation_set)

        # RL metrics: does model improve on reward signal?
        rl_reward = evaluate_rl_reward(model, rl_validation_set)

    return {
        "sft_accuracy": sft_accuracy,
        "rl_reward": rl_reward,
        "alpha": alpha,
        "balance_score": alpha * sft_accuracy + (1 - alpha) * rl_reward
    }
```

## Practical Guidance

### When to Use CHORD

- Training LLMs where existing response patterns are valuable
- Combining expert demonstrations with reward signals
- Preventing catastrophic forgetting during RL training
- Gradual exploration without disrupting learned behaviors
- Hybrid supervised+reinforcement training scenarios

### When NOT to Use

- Pure supervised learning without RL signals
- Pure RL where forgetting established patterns is acceptable
- Scenarios requiring immediate full on-policy training
- Tasks where expert data quality is inconsistent

### Key Hyperparameters

- **initial_alpha**: 0.8-1.0 (start with strong SFT)
- **final_alpha**: 0.1-0.3 (end with strong RL)
- **Transition Schedule**: Linear or cosine annealing
- **Token Weighting Strategy**: "adaptive" recommended
- **Training Duration**: Typically 1-2 epochs with CHORD

### Performance Expectations

- SFT Preservation: Maintains 95%+ of original expert performance
- RL Improvement: 5-15% gains on task rewards
- Training Stability: Reduced variance vs. switching SFT→RL
- Convergence Speed: Similar or slightly faster than separate stages

## Reference

Researchers. (2024). On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL. arXiv preprint arXiv:2508.11408.
