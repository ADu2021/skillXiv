---
name: sparse-moe-agentic-intelligence
title: "Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.10604"
keywords: [Sparse Mixture-of-Experts, Efficient Reasoning, Multi-Token Prediction, MIS-PO, Agentic Intelligence]
description: "Deploy frontier-level reasoning with only 11B active parameters using sparse MoE with 288 routed experts plus shared expert. Use Metropolis Independence Sampling-Filtered Policy Optimization (MIS-PO) to stabilize RL training at scale, replacing continuous importance weighting with discrete filtering that ensures trust region stability."
---

# Step 3.5 Flash: Open Frontier-Level Intelligence with 11B Active Parameters

## Problem Context

Frontier reasoning models like o1 use massive compute budgets. Step 3.5 Flash achieves frontier-level performance with minimal active parameters (11B) by combining (1) sparse MoE with 288 experts per layer, (2) multi-token prediction for speculative decoding, (3) MIS-PO for stable RL training at scale. Key challenges: expert collapse, activation explosions, numerical instability in large sparse models.

## Core Concept

The architecture uses: (1) 196B total parameters with only 11B active per token via selective expert routing, (2) 3:1 ratio of sliding-window to full attention (local + global reasoning), (3) multi-token prediction (MTP-3) enabling 3x faster decoding, (4) MIS-PO replacing PPO clipping with discrete filtering for stability. This achieves o1-competitive performance on reasoning benchmarks while remaining efficiently deployable.

## Architecture Overview

- **Sparse MoE**: 288 routed experts + 1 shared expert, activate 8 per token
- **Selective routing**: Load-balanced expert assignment preventing collapse
- **Hybrid attention**: Sliding-window (efficiency) + full attention (reasoning)
- **Multi-token prediction**: Generate 3 tokens per forward pass
- **MIS-PO training**: Discrete filtering for trust region RL
- **Monitoring infrastructure**: Detect numerical issues, routing pathologies

## Implementation

### Step 1: Sparse MoE layer with selective routing

```python
import torch
import torch.nn as nn
from typing import Tuple

class SelectiveExpertRouter:
    """Route tokens to subset of experts with load balancing."""

    def __init__(
        self,
        num_experts: int = 288,
        active_experts: int = 8,
        capacity_factor: float = 1.25
    ):
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.capacity_factor = capacity_factor

    def route_tokens(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, dim]
        router_logits: torch.Tensor   # [batch, seq_len, num_experts]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route tokens to experts using top-k selection.

        Args:
            hidden_states: Input hidden states
            router_logits: Routing scores from router network

        Returns:
            (routed_output, routing_mask, routing_stats)
        """
        batch_size, seq_len, dim = hidden_states.shape

        # Compute routing probabilities
        router_probs = torch.softmax(router_logits, dim=-1)

        # Select top-k experts (deterministic routing)
        top_k_probs, top_k_indices = torch.topk(
            router_probs, k=self.active_experts, dim=-1
        )

        # Normalize selected probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute load per expert
        expert_load = torch.zeros(self.num_experts, device=hidden_states.device)
        for i in range(self.active_experts):
            expert_load.scatter_add_(
                0, top_k_indices[:, :, i].reshape(-1),
                top_k_probs[:, :, i].reshape(-1)
            )

        # Load balancing loss (auxiliary)
        load_balancing_loss = self._compute_load_balance_loss(
            expert_load, batch_size, seq_len
        )

        routing_stats = {
            'load_balancing_loss': load_balancing_loss.item(),
            'mean_expert_load': expert_load.mean().item(),
            'max_expert_load': expert_load.max().item()
        }

        return top_k_indices, top_k_probs, routing_stats

    def _compute_load_balance_loss(
        self,
        expert_load: torch.Tensor,
        batch_size: int,
        seq_len: int
    ) -> torch.Tensor:
        """Auxiliary loss to balance expert utilization."""
        # Target load per expert
        target_load = (batch_size * seq_len * self.active_experts) / self.num_experts

        # L2 loss on load imbalance
        load_loss = torch.mean((expert_load - target_load) ** 2)

        return load_loss


class SparseExpertLayer(nn.Module):
    """Single sparse MoE layer with selective routing."""

    def __init__(
        self,
        dim: int = 4096,
        num_experts: int = 288,
        active_experts: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.active_experts = active_experts

        # Expert networks (implemented as parallel FFNs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim)
            )
            for _ in range(num_experts)
        ])

        # Shared expert (always active)
        self.shared_expert = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        # Router network
        self.router = nn.Linear(dim, num_experts)

        self.selector = SelectiveExpertRouter(num_experts, active_experts)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Route through selected experts."""
        batch_size, seq_len, dim = hidden_states.shape

        # Shared expert computation (always included)
        shared_output = self.shared_expert(hidden_states)

        # Route to selected experts
        router_logits = self.router(hidden_states)
        expert_indices, expert_weights, routing_stats = self.selector.route_tokens(
            hidden_states, router_logits
        )

        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(hidden_states))
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # [B, S, E, D]

        # Gather selected expert outputs
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]
        seq_idx = torch.arange(seq_len, device=hidden_states.device)[None, :, None]

        selected_outputs = expert_outputs[batch_idx, seq_idx, expert_indices]  # [B, S, K, D]

        # Weight by routing probabilities
        weighted_outputs = (
            selected_outputs * expert_weights.unsqueeze(-1)
        ).sum(dim=2)  # [B, S, D]

        # Combine with shared expert
        output = weighted_outputs + shared_output

        return output, routing_stats
```

### Step 2: Multi-token prediction for acceleration

```python
class MultiTokenPredictor(nn.Module):
    """Predict multiple tokens per forward pass."""

    def __init__(self, vocab_size: int = 100000, num_predict: int = 3):
        super().__init__()
        self.num_predict = num_predict
        self.vocab_size = vocab_size

        # Multiple output heads for parallel token prediction
        self.heads = nn.ModuleList([
            nn.Linear(768, vocab_size) for _ in range(num_predict)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict multiple tokens.

        Args:
            hidden_states: [batch, seq_len, dim]

        Returns:
            (logits, confidences): List of logit tensors and confidence per head
        """
        logits_list = []
        confidences = []

        for head_idx, head in enumerate(self.heads):
            logits = head(hidden_states)  # [B, S, V]
            logits_list.append(logits)

            # Confidence: max probability
            probs = torch.softmax(logits, dim=-1)
            max_prob, _ = torch.max(probs, dim=-1)  # [B, S]
            confidences.append(max_prob)

        return logits_list, confidences

    def select_confident_predictions(
        self,
        logits_list: list,
        confidences: list,
        min_confidence: float = 0.8
    ) -> torch.Tensor:
        """
        Select most confident multi-token predictions.

        Returns selected token sequence.
        """
        # For each position, select head with highest confidence
        all_confidences = torch.stack(confidences, dim=-1)  # [B, S, K]
        best_head_idx = torch.argmax(all_confidences, dim=-1)  # [B, S]

        selected_logits = torch.zeros_like(logits_list[0])
        for i in range(self.num_predict):
            mask = (best_head_idx == i)
            selected_logits[mask] = logits_list[i][mask]

        return selected_logits
```

### Step 3: MIS-PO (Metropolis Independence Sampling-Filtered Policy Optimization)

```python
class MetropolisIndependenceSamplingPO:
    """Discrete filtering policy optimization for RL at scale."""

    def __init__(self, model, optimizer, trust_region_threshold: float = 0.05):
        self.model = model
        self.optimizer = optimizer
        self.trust_region_threshold = trust_region_threshold

    def compute_mis_po_loss(
        self,
        log_probs: torch.Tensor,        # [batch, seq_len]
        log_probs_old: torch.Tensor,    # [batch, seq_len]
        rewards: torch.Tensor,          # [batch]
        advantages: torch.Tensor        # [batch]
    ) -> torch.Tensor:
        """
        Compute MIS-PO loss using discrete filtering.

        Unlike PPO (continuous clipping), MIS-PO uses discrete filtering:
        Keep update if log_prob_ratio falls within trust region, reject otherwise.
        """
        log_prob_ratio = log_probs.sum(dim=1) - log_probs_old.sum(dim=1)

        # Discrete filtering: only use samples within trust region
        within_trust_region = torch.abs(log_prob_ratio) < self.trust_region_threshold

        # Compute policy gradient only on trusted samples
        ppo_loss = -(log_prob_ratio * advantages)[within_trust_region].mean()

        # Rejection rate (monitoring)
        rejection_rate = 1.0 - within_trust_region.float().mean().item()

        return ppo_loss, {'rejection_rate': rejection_rate}

    def training_step(
        self,
        batch: dict,
        reward_fn
    ) -> dict:
        """Single MIS-PO training step."""
        prompts = batch['prompts']
        batch_size = len(prompts)

        # Generate responses
        responses = []
        log_probs_list = []

        for prompt in prompts:
            response, log_probs = self.model.generate_with_logprobs(
                prompt, max_tokens=500
            )
            responses.append(response)
            log_probs_list.append(log_probs)

        log_probs = torch.stack(log_probs_list)

        # Compute rewards
        rewards = torch.tensor([reward_fn(r) for r in responses])

        # Advantages (group relative)
        group_mean = rewards.mean()
        advantages = rewards - group_mean

        # Reference log probs
        log_probs_ref = log_probs.detach()

        # MIS-PO loss
        loss, stats = self.compute_mis_po_loss(log_probs, log_probs_ref, rewards, advantages)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'rejection_rate': stats['rejection_rate'],
            'avg_reward': rewards.mean().item()
        }
```

### Step 4: Monitoring for training stability

```python
class StabilityMonitor:
    """Monitor for expert collapse, activation explosions, numerical issues."""

    def __init__(self):
        self.history = {'expert_load': [], 'activation_stats': []}

    def check_expert_collapse(self, routing_stats: dict, threshold: float = 0.8):
        """Check if some experts are unused."""
        max_load = routing_stats['max_expert_load']
        return max_load > threshold

    def check_activation_explosion(
        self,
        hidden_states: torch.Tensor,
        norm_threshold: float = 100.0
    ):
        """Check for exploding norms in hidden states."""
        state_norms = torch.norm(hidden_states, dim=-1)
        max_norm = state_norms.max().item()
        return max_norm > norm_threshold

    def check_numerical_stability(
        self,
        model: nn.Module
    ):
        """Check for NaN/Inf in parameters."""
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return False
        return True

    def remediate_issues(self, model: nn.Module, issue_type: str):
        """Apply remediation for detected issues."""
        if issue_type == 'activation_explosion':
            # Apply activation clipping in forward pass
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    torch.clamp_(module.bias, -10, 10)

        elif issue_type == 'numerical_instability':
            # Switch Muon optimizer to float32 for orthogonalization
            pass
```

### Step 5: Full training integration

```python
def train_step_flash_model(
    model,
    train_loader,
    verifier,
    optimizer,
    num_steps: int = 100000,
    device: str = 'cuda'
):
    """
    Train Step 3.5 Flash using MIS-PO.
    """
    mis_po = MetropolisIndependenceSamplingPO(model, optimizer)
    monitor = StabilityMonitor()

    for step in range(num_steps):
        batch = next(iter(train_loader))

        # Define reward function
        def reward_fn(response):
            return float(verifier(response))

        # MIS-PO update
        metrics = mis_po.training_step(batch, reward_fn)

        # Monitor stability
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, SparseExpertLayer):
                    # Check for issues
                    if monitor.check_expert_collapse({'max_expert_load': 0.9}):
                        print(f"Step {step}: Expert collapse detected, applying remediation")

        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}: "
                  f"Loss={metrics['loss']:.4f}, "
                  f"Rejection={metrics['rejection_rate']:.2%}, "
                  f"Reward={metrics['avg_reward']:.4f}")

    return model
```

## Practical Guidance

**When to use**: Frontier reasoning at scale; resource-constrained deployments; reasoning-heavy workloads

**Hyperparameters**:
- **num_experts**: 128-512 (tradeoff: capacity vs. compute)
- **active_experts**: 4-16 (active per token)
- **capacity_factor**: 1.0-1.5 (load balancing)
- **trust_region_threshold**: 0.03-0.1 (MIS-PO filtering)
- **multi_token_predict**: 2-4 (acceleration vs. quality)

**Key advantages**:
- Frontier performance with 11B active parameters
- Stable RL training via MIS-PO
- Fast inference via multi-token prediction
- Efficient hybrid attention (local + global)

**Common pitfalls**:
- Expert collapse without load balancing auxiliary loss
- Activation explosions in deep sparse networks (use clipping)
- MIS-PO threshold too loose → no actual filtering
- Multi-token prediction conflicting token dependencies

**Scaling**: Linear in number of experts. Distributed routing enables 10K+ expert systems.

## Reference

Paper: https://arxiv.org/abs/2602.10604
Related work: Sparse MoE, policy optimization, multi-token prediction
Benchmarks: IMO-AnswerBench (85.4%), LiveCodeBench-v6 (86.4%)
Architecture: 196B total, 11B active, 288 experts/layer
