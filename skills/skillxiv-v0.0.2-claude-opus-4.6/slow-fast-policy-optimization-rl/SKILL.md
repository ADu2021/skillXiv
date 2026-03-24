---
name: slow-fast-policy-optimization-rl
title: "Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.04072"
keywords: [policy optimization, RL stability, gradient control, off-policy divergence, reasoning models]
description: "Stabilize RL for LLM reasoning via three-phase decomposition: fast inner trajectory optimization, repositioning to manage off-policy drift, slow correction for stable updates. Achieve up to 2.80-point math reasoning gains over GRPO while reducing rollouts 4.93x and wall-clock time 4.19x via improved stability without changing reward structure."
---

# Slow-Fast Policy Optimization: Reposition-Before-Update

## Core Concept

On-policy RL for LLMs suffers gradient instability from noisy early-training rollouts. Slow-Fast Policy Optimization (SFPO) decomposes each training step into three phases: fast optimization on the current batch, repositioning to constrain off-policy divergence, and slow correction for stable updates. This decomposition is plug-compatible with existing pipelines while dramatically improving stability and efficiency.

## Architecture Overview

- **Three-Phase Framework**: (1) Fast trajectory optimization on same batch; (2) Reposition to manage divergence; (3) Slow correction phase
- **Off-Policy Divergence Control**: Track KL divergence during each phase to prevent catastrophic divergence
- **Plug-Compatible Design**: Works with existing policy gradient implementations (GRPO, PPO) without modification
- **Efficiency Gains**: 4.93× rollout reduction, 4.19× wall-clock speedup while improving performance

## Implementation Steps

### 1. Three-Phase Decomposition Framework

Structure each training iteration into coordinated phases.

```python
class SlowFastPolicyOptimizer:
    def __init__(self, policy_model, learning_rate=1e-6, max_kl=0.1):
        self.policy = policy_model
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate)
        self.max_kl = max_kl  # KL divergence threshold

    def train_step_sfpo(self, batch, initial_policy_state=None):
        """
        Single SFPO training step with three phases.

        Args:
            batch: (observations, actions, rewards, values) from single rollout
            initial_policy_state: Reference policy for KL computation
        """

        # Save initial policy state for divergence tracking
        if initial_policy_state is None:
            initial_policy_state = {
                name: param.detach().clone()
                for name, param in self.policy.named_parameters()
            }

        # Phase 1: Fast optimization on current batch
        print("Phase 1: Fast inner trajectory optimization")
        fast_loss = self._phase1_fast_optimization(batch)

        # Phase 2: Repositioning to manage off-policy drift
        print("Phase 2: Repositioning mechanism")
        repositioned_state = self._phase2_repositioning(initial_policy_state)

        # Phase 3: Slow correction phase
        print("Phase 3: Slow correction for stability")
        final_loss = self._phase3_slow_correction(batch, repositioned_state)

        return {
            'phase1_loss': fast_loss,
            'phase3_loss': final_loss,
            'total_divergence': self._compute_kl_divergence(initial_policy_state)
        }

    def _phase1_fast_optimization(self, batch):
        """
        Fast inner optimization: take multiple gradient steps on same batch.
        """

        obs, actions, rewards, values = batch

        # Compute initial advantages
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple inner updates (e.g., 5 steps)
        inner_losses = []

        for _ in range(5):  # Inner loop iterations
            # Forward pass
            action_logits = self.policy(obs)

            # Compute policy loss (GRPO/PPO style)
            log_probs = torch.log_softmax(action_logits, dim=-1)
            selected_log_probs = log_probs[range(len(actions)), actions]

            # Clipped objective
            ratio = torch.exp(selected_log_probs - selected_log_probs.detach())
            clipped_ratio = torch.clamp(ratio, 0.9, 1.1)

            loss = -torch.min(ratio, clipped_ratio) * advantages

            # Gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            inner_losses.append(loss.mean().item())

        return sum(inner_losses) / len(inner_losses)

    def _phase2_repositioning(self, initial_state):
        """
        Repositioning: Adjust policy to constrain KL divergence from initial state.
        Prevents divergence from accumulating across training steps.
        """

        # Compute current KL divergence
        current_kl = self._compute_kl_divergence(initial_state)

        if current_kl > self.max_kl:
            print(f"KL divergence {current_kl:.4f} exceeds threshold {self.max_kl:.4f}")
            print("Applying repositioning correction...")

            # Reposition: interpolate between current and initial state
            # α ∝ (KL - threshold) / KL
            excess_divergence = max(0, current_kl - self.max_kl)
            alpha = excess_divergence / (current_kl + 1e-8)

            # Linear interpolation toward initial state
            for (name, param), (init_name, init_param) in zip(
                self.policy.named_parameters(),
                initial_state.items()
            ):
                if name == init_name:
                    param.data = (1 - alpha) * param.data + alpha * init_param

        return {
            'kl_divergence': current_kl,
            'repositioned': current_kl > self.max_kl
        }

    def _phase3_slow_correction(self, batch, repositioned_state):
        """
        Slow correction: Final update step with reduced learning rate.
        Ensures stability after repositioning.
        """

        obs, actions, rewards, values = batch

        # Advantages
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Single slow update step
        action_logits = self.policy(obs)
        log_probs = torch.log_softmax(action_logits, dim=-1)
        selected_log_probs = log_probs[range(len(actions)), actions]

        # Clipped objective with reduced learning rate
        ratio = torch.exp(selected_log_probs - selected_log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 0.95, 1.05)  # Tighter clipping than Phase 1

        loss = -torch.min(ratio, clipped_ratio) * advantages

        # Very small update
        self.optimizer.param_groups[0]['lr'] = 1e-7  # Temporarily reduce LR
        self.optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)  # Tight clipping
        self.optimizer.step()
        self.optimizer.param_groups[0]['lr'] = 1e-6  # Restore LR

        return loss.mean().item()

    def _compute_kl_divergence(self, reference_state):
        """
        Compute KL divergence from reference policy parameters.
        Simplified: L2 distance in parameter space (proxy for KL).
        """

        total_kl = 0

        for (name, param), (ref_name, ref_param) in zip(
            self.policy.named_parameters(),
            reference_state.items()
        ):
            if name == ref_name:
                kl = torch.norm(param.data - ref_param) / (torch.norm(ref_param) + 1e-8)
                total_kl += kl.item()

        return total_kl / len(reference_state)
```

### 2. Integration with GRPO Pipeline

SFPO is plug-compatible; use as drop-in replacement for standard gradient steps.

```python
def grpo_training_with_sfpo(
    policy, base_model, train_loader, num_epochs=5, group_size=8
):
    """
    Standard GRPO training loop but with SFPO decomposition.
    """

    optimizer = SlowFastPolicyOptimizer(policy, learning_rate=1e-6)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Standard GRPO: generate rollouts
            obs, actions, rewards = batch

            # Compute value estimates
            with torch.no_grad():
                values = base_model.estimate_value(obs)

            # Group rewards (GRPO)
            group_rewards = []
            for i in range(0, len(rewards), group_size):
                group = rewards[i:i+group_size]
                group_mean = group.mean()
                group_std = group.std() + 1e-8
                normalized = (group - group_mean) / group_std
                group_rewards.extend(normalized)

            group_rewards = torch.tensor(group_rewards)

            # SFPO training step (replaces standard gradient update)
            batch_data = (obs, actions, group_rewards, values)
            sfpo_result = optimizer.train_step_sfpo(batch_data)

            total_loss += sfpo_result['phase3_loss']
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch}, Batch {batch_idx+1}: Loss={avg_loss:.4f}")

    return policy
```

### 3. Hyperparameter Configuration

SFPO requires minimal tuning; primary hyperparameter is KL threshold.

```python
sfpo_config = {
    'learning_rate': 1e-6,              # Standard LR
    'inner_loop_steps': 5,              # Phase 1 iterations
    'max_kl_divergence': 0.1,           # Repositioning threshold
    'phase3_learning_rate': 1e-7,       # Slow correction LR
    'phase1_clipping': (0.9, 1.1),      # Standard PPO clipping
    'phase3_clipping': (0.95, 1.05),    # Tighter Phase 3 clipping
    'grad_clip_norm': 1.0               # Gradient clipping
}

# Benchmark: Math reasoning (AIME subset)
# Configuration for 1.5B Qwen model
sfpo_math_config = {
    'inner_loop_steps': 3,
    'max_kl': 0.05,                    # Conservative for safety
    'batch_size': 32,
    'num_epochs': 3
}
```

## Performance Results

Evaluation on mathematical reasoning benchmarks:

```python
results = {
    'grpo_baseline': {
        'accuracy': 'Base',
        'rollouts_per_iter': 32,
        'wall_clock': '100% (baseline)'
    },
    'sfpo': {
        'accuracy_improvement': '+2.8 points (on AIME subset)',
        'rollouts_per_iter': 32,  # Same
        'actual_rollouts_used': '6.5 (4.93x reduction)',
        'wall_clock': '23.8% of baseline (4.19x speedup)',
        'mechanism': 'Better reuse of low-quality initial rollouts via repositioning'
    }
}
```

## Practical Guidance

**Phase Balancing**: Adjust inner loop steps (Phase 1) based on rollout quality. More steps needed if rollouts are noisy.

**KL Threshold**: 0.05-0.1 works well for most domains. Start conservative; increase if divergence becomes a bottleneck.

**Clipping Adjustment**: Phase 3 should use tighter clipping (0.95-1.05) to prevent destabilization after repositioning.

**Compatibility**: SFPO works with any policy gradient objective (GRPO, PPO, REINFORCE). No changes to reward structure needed.

## When to Use / When NOT to Use

**Use When**:
- Training reasoning models where rollout quality is variable
- Early training stages with high gradient noise
- Compute budget is constrained (4x speedup is significant)
- Stability is critical (robotics, safety-sensitive tasks)

**NOT For**:
- Tasks with uniformly high-quality rollouts
- Scenarios where convergence speed matters more than stability
- Very small batch sizes (<8) where repositioning has minimal effect

## Reference

This skill synthesizes findings from "Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning" (arXiv:2510.04072, ICLR 2026). Three-phase decomposition achieves stability without objective modification.
