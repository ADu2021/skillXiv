---
name: expert-threshold-routing
title: "Expert Threshold Routing for Autoregressive Language Modeling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.11535"
keywords: [Mixture-of-Experts, Routing, Language Models, Causal Inference, Efficiency]
description: "Improve MoE language model efficiency with causal threshold-based routing that eliminates auxiliary losses and enables dynamic per-token computation."
---

# Expert Threshold Routing for Autoregressive Language Modeling

Mixture-of-Experts (MoE) models promise efficient scaling by routing tokens to specialized experts, but standard token-choice MoE suffers from two critical limitations. First, it requires auxiliary loss functions to maintain load balance, adding complexity and tuning burden. Second, it is inherently batch-dependent, making it unsuitable for streaming autoregressive inference where tokens arrive sequentially.

Expert Threshold Routing (ET) solves both problems through a simple yet elegant mechanism: each expert maintains an exponential moving average (EMA) threshold, and tokens route to experts independently based on whether their score exceeds that threshold. This causal design eliminates batch dependencies and auxiliary losses while achieving natural load balancing through threshold dynamics.

## Core Concept

Expert Threshold Routing replaces token-level routing decisions with threshold-based gating:

**Independent Routing:** Each token independently decides whether to route to each expert by comparing its score against the expert's threshold. No global allocation constraints or auxiliary losses required.

**Dynamic Thresholds:** Expert thresholds are estimated as exponential moving averages of token score distributions, adapting to changing data characteristics.

**Causal Design:** The mechanism operates on a per-token basis, with thresholds updated incrementally. This enables streaming inference where tokens are processed one at a time without requiring future context.

**Balanced Load:** Thresholds automatically regulate expert utilization—when an expert is overutilized, its threshold rises; when underutilized, it falls.

## Architecture Overview

- **Token Scores**: Computed for each token-expert pair (e.g., via dot product with expert prototypes)
- **Expert Thresholds**: EMA-tracked per-expert thresholds initialized from global token distribution
- **Routing Decision**: Token routes to expert if score > threshold (binary decision per expert)
- **Load Balancing**: Thresholds adjust based on expert utilization, creating natural equilibrium
- **Causal Updates**: Threshold computation depends only on past and current data, not future

## Implementation Steps

### Step 1: Initialize Expert Thresholds

Compute initial thresholds from a small calibration dataset.

```python
import torch
import torch.nn as nn
from collections import defaultdict

class ExpertThresholdRouter(nn.Module):
    """
    Causal routing mechanism with expert thresholds.
    No auxiliary losses or batch-level constraints required.
    """

    def __init__(self, num_experts=16, num_tokens=4096, threshold_ema_decay=0.999):
        super().__init__()
        self.num_experts = num_experts
        self.threshold_ema_decay = threshold_ema_decay

        # Initialize thresholds as learnable parameters
        # Will be updated via EMA during training
        self.register_buffer(
            'expert_thresholds',
            torch.zeros(num_experts)
        )
        self.register_buffer(
            'threshold_momentum',
            torch.ones(num_experts) * 0.5  # Running average
        )

    def initialize_thresholds(self, calibration_scores):
        """
        Compute initial thresholds from calibration data.
        calibration_scores: (num_tokens, num_experts) tensor
        """
        # Compute percentiles for each expert
        for expert_idx in range(self.num_experts):
            expert_scores = calibration_scores[:, expert_idx]
            # Initialize at median score (50th percentile)
            initial_threshold = torch.quantile(expert_scores, 0.5)
            self.expert_thresholds[expert_idx] = initial_threshold

        print(f"Initialized thresholds: {self.expert_thresholds}")
```

### Step 2: Compute Token Scores

Generate routing scores for each token to each expert.

```python
    def compute_token_expert_scores(self, token_hidden_states, expert_prototypes):
        """
        Compute affinity score between each token and expert.
        token_hidden_states: (batch_size, seq_len, hidden_dim)
        expert_prototypes: (num_experts, hidden_dim)
        Returns: (batch_size, seq_len, num_experts)
        """
        batch_size, seq_len, hidden_dim = token_hidden_states.shape

        # Normalize for stability
        token_hidden_normalized = torch.nn.functional.normalize(
            token_hidden_states.reshape(-1, hidden_dim), dim=1
        )
        expert_normalized = torch.nn.functional.normalize(expert_prototypes, dim=1)

        # Compute cosine similarity scores
        scores = torch.matmul(token_hidden_normalized, expert_normalized.t())
        scores = scores.reshape(batch_size, seq_len, self.num_experts)

        return scores
```

### Step 3: Implement Causal Routing Decision

Route tokens based on threshold comparisons.

```python
    def route_tokens_causal(self, token_scores):
        """
        Route tokens to experts based on threshold comparisons.
        This is the core causal mechanism—fully independent per token.

        token_scores: (batch_size * seq_len, num_experts)
        Returns: routing_matrix (batch_size * seq_len, num_experts) binary
        """
        # Compare each score to expert threshold
        routing_matrix = (token_scores > self.expert_thresholds.unsqueeze(0)).float()

        return routing_matrix

    def get_routing_info(self, routing_matrix):
        """
        Extract useful routing statistics without auxiliary loss computation.
        routing_matrix: (num_tokens, num_experts) binary
        """
        num_tokens = routing_matrix.shape[0]

        # Compute expert loads (how many tokens routed to each expert)
        expert_loads = routing_matrix.sum(dim=0)

        # Average tokens per expert
        avg_load = expert_loads.mean()

        # Load balance metric (lower is better)
        load_variance = (expert_loads - avg_load).pow(2).mean()

        return {
            'expert_loads': expert_loads,
            'avg_load': avg_load,
            'load_variance': load_variance
        }
```

### Step 4: Update Thresholds with EMA

Dynamically adjust thresholds based on observed token scores.

```python
    def update_thresholds_ema(self, token_scores):
        """
        Update expert thresholds using exponential moving average.
        This is the core balancing mechanism—causal and requires no auxiliary loss.

        token_scores: (num_tokens, num_experts)
        """
        # Compute current empirical distribution for each expert
        current_means = token_scores.mean(dim=0)  # (num_experts,)
        current_stds = token_scores.std(dim=0)    # (num_experts,)

        # Update thresholds via EMA
        # If expert has high average score, raise threshold to reduce load
        # If expert has low average score, lower threshold to increase load
        new_thresholds = current_means - current_stds  # Approximately median

        # Apply EMA: threshold_t = decay * threshold_{t-1} + (1-decay) * new_threshold
        self.expert_thresholds = (
            self.threshold_ema_decay * self.expert_thresholds +
            (1 - self.threshold_ema_decay) * new_thresholds
        )

        return self.expert_thresholds
```

### Step 5: Forward Pass with Causal Routing

Integrate routing into the MoE forward pass.

```python
class MoELayerWithThresholdRouting(nn.Module):
    """
    MoE layer using Expert Threshold Routing.
    Enables causal, streaming inference without auxiliary losses.
    """

    def __init__(self, hidden_dim, num_experts=16, expert_dim=4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, hidden_dim)
            )
            for _ in range(num_experts)
        ])

        # Expert prototypes for routing (learnable)
        self.expert_prototypes = nn.Parameter(
            torch.randn(num_experts, hidden_dim)
        )

        # Router with thresholds
        self.router = ExpertThresholdRouter(num_experts=num_experts)

    def forward(self, token_hidden_states):
        """
        Forward pass with causal threshold routing.
        token_hidden_states: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = token_hidden_states.shape

        # Flatten for routing
        flat_hidden = token_hidden_states.reshape(-1, hidden_dim)

        # Compute token-expert scores
        scores = self.router.compute_token_expert_scores(
            token_hidden_states, self.expert_prototypes
        )
        flat_scores = scores.reshape(-1, self.num_experts)

        # Route tokens to experts (causal)
        routing_matrix = self.router.route_tokens_causal(flat_scores)

        # Update thresholds based on observed scores (EMA)
        self.router.update_thresholds_ema(flat_scores)

        # Apply experts
        expert_outputs = torch.stack([
            self.experts[i](flat_hidden) for i in range(self.num_experts)
        ], dim=1)  # (num_tokens, num_experts, hidden_dim)

        # Combine expert outputs via routing
        output = torch.bmm(
            routing_matrix.unsqueeze(1),  # (num_tokens, 1, num_experts)
            expert_outputs  # (num_tokens, num_experts, hidden_dim)
        ).squeeze(1)  # (num_tokens, hidden_dim)

        # Reshape back
        output = output.reshape(batch_size, seq_len, hidden_dim)

        # Get routing statistics
        stats = self.router.get_routing_info(routing_matrix)

        return output, stats
```

### Step 6: Training Integration

Incorporate into full language model training without auxiliary losses.

```python
def train_moe_language_model_with_threshold_routing(
    model, train_loader, num_epochs=3, learning_rate=1e-4
):
    """
    Train MoE language model using Expert Threshold Routing.
    No auxiliary loss needed—thresholds balance automatically.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        total_tokens = 0

        for batch_idx, (input_ids, attention_mask) in enumerate(train_loader):
            # Forward pass
            logits = model(input_ids, attention_mask=attention_mask)

            # Language modeling loss only (no auxiliary MoE loss!)
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, model.vocab_size),
                input_ids.reshape(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.shape[0]
            total_tokens += (input_ids != 0).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")

        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        print(f"Epoch {epoch}: Perplexity = {perplexity:.2f}")
```

## Practical Guidance

**Hyperparameters:**
- Number of experts: 8-32 (scale with model size; typically 1 expert per ~500M params)
- Threshold EMA decay: 0.99-0.999 (higher = slower threshold adaptation, more stability)
- Expert capacity factor: 1.0-2.0 (ratio of max tokens per expert to average)
- Initialize thresholds at 50th percentile (median) for balanced startup

**When to Use:**
- Autoregressive language model inference (streaming, causal constraint)
- When you need to eliminate auxiliary losses and their hyperparameter tuning
- Scenarios requiring dynamic per-token computation (some tokens need more experts)
- Large-scale language models where auxiliary loss tuning is a bottleneck

**When NOT to Use:**
- Batch inference where load balancing is known and fixed (standard MoE is simpler)
- Environments requiring explicit load constraints (e.g., guaranteed CPU/GPU utilization)
- Very small models where MoE overhead isn't justified
- Tasks where all tokens should use the same set of experts (dense routing better)

**Pitfalls:**
- Threshold initialization critical: poor initialization can cause extreme load imbalance; always calibrate on representative data
- EMA decay too high causes sluggish adaptation; too low causes oscillation
- Monitor load balance in early training; if thresholds diverge, manually reset
- Expert imbalance: if one expert is consistently underutilized, it won't receive gradient updates; consider expert dropout or diversity loss

## Reference

Paper: [arxiv.org/abs/2603.11535](https://arxiv.org/abs/2603.11535)
