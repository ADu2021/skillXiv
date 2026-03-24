---
name: think-router-hybrid-reasoning
title: "ThinkRouter: Efficient Reasoning via Routing between Latent and Discrete"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.11683"
keywords: [Reasoning, Latent Space, Discrete Tokens, Confidence-Based Routing, Efficiency]
description: "Route reasoning between discrete token space (when uncertain) and latent soft embeddings (when confident). Use maximum next-token probability as a routing threshold to dynamically select the reasoning space, improving accuracy under latent reasoning while reducing computational cost through selective discrete sampling."
---

# ThinkRouter: Efficient Reasoning via Routing between Latent and Discrete

## Problem Context

Extended reasoning models can operate in two spaces: discrete token space (one token sampled at a time) or latent space (soft embeddings aggregated from distributions). Latent reasoning is more efficient but accumulates noise from low-confidence steps, leading to spurious high confidence in wrong answers. Discrete reasoning commits to single tokens, avoiding aggregation noise but is slower. ThinkRouter solves this by routing between spaces based on model confidence.

## Core Concept

At each reasoning step, ThinkRouter measures the maximum next-token probability. If probability is below a threshold τ, reasoning operates in discrete token space (commitment under uncertainty). If probability meets or exceeds τ, reasoning uses latent soft embeddings (efficient exploration when confident).

This hybrid approach leverages discrete reasoning's robustness when uncertain and latent reasoning's efficiency when confident.

## Architecture Overview

- **Confidence detector**: Compute maximum next-token probability at each step
- **Routing threshold**: Learned or grid-searched threshold τ per model-dataset pair
- **Discrete branch**: Sample single token when max_prob < τ
- **Latent branch**: Aggregate soft embeddings when max_prob ≥ τ
- **Combined reasoning**: Seamlessly switch spaces within single reasoning trajectory

## Implementation

### Step 1: Compute confidence scores and routing decision

Measure model confidence and determine reasoning space.

```python
import torch
import torch.nn.functional as F
from typing import Tuple

class ConfidenceRouter:
    """Route reasoning between latent and discrete spaces based on confidence."""

    def __init__(self, routing_threshold: float = 0.8):
        """
        Args:
            routing_threshold: Probability threshold for routing decision.
                             P(next_token_max) < threshold -> discrete
                             P(next_token_max) >= threshold -> latent
        """
        self.routing_threshold = routing_threshold

    def compute_routing_decision(
        self,
        logits: torch.Tensor,  # [batch, vocab_size]
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing decision based on maximum token probability.

        Args:
            logits: Raw model logits
            temperature: Sampling temperature

        Returns:
            max_prob: Maximum next-token probability [batch]
            routing_mask: Boolean mask for routing (True=latent, False=discrete)
            routing_decision: String labels for each sample
        """
        # Compute probabilities
        probs = F.softmax(logits / temperature, dim=-1)
        max_prob, max_token_idx = torch.max(probs, dim=-1)

        # Routing decision
        route_to_latent = max_prob >= self.routing_threshold

        return max_prob, route_to_latent, max_token_idx

    def visualize_routing_statistics(
        self,
        max_probs: list,
        route_decisions: list
    ) -> dict:
        """Track routing statistics for analysis."""
        max_probs_tensor = torch.tensor(max_probs)
        route_to_latent = sum(route_decisions)
        route_to_discrete = len(route_decisions) - route_to_latent

        return {
            'mean_max_prob': max_probs_tensor.mean().item(),
            'std_max_prob': max_probs_tensor.std().item(),
            'latent_ratio': route_to_latent / len(route_decisions),
            'discrete_ratio': route_to_discrete / len(route_decisions)
        }
```

### Step 2: Discrete branch - token sampling with commitment

Implement discrete reasoning with single token selection.

```python
class DiscreteReasoningBranch:
    """Discrete reasoning: sample and commit to single token."""

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature

    def sample_discrete_token(
        self,
        logits: torch.Tensor,  # [batch, vocab_size]
        top_k: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample single token with commitment.

        Args:
            logits: Raw model logits
            top_k: Only sample from top-k tokens

        Returns:
            sampled_token_ids: Selected token indices
            log_probs: Log probability of selected tokens
        """
        # Top-k filtering
        probs = F.softmax(logits / self.temperature, dim=-1)

        # Get top-k probabilities
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)

        # Renormalize
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Sample
        batch_size = logits.shape[0]
        sampled_positions = torch.multinomial(topk_probs, num_samples=1)
        sampled_tokens = topk_indices.gather(-1, sampled_positions).squeeze(-1)

        # Compute log probability of sampled tokens
        log_probs_full = F.log_softmax(logits / self.temperature, dim=-1)
        log_probs = log_probs_full.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

        return sampled_tokens, log_probs
```

### Step 3: Latent branch - soft embedding aggregation

Implement latent reasoning with probability-weighted embeddings.

```python
class LatentReasoningBranch:
    """Latent reasoning: aggregate soft embeddings from distribution."""

    def __init__(self, embedding_dim: int = 768, temperature: float = 1.0):
        self.embedding_dim = embedding_dim
        self.temperature = temperature

    def aggregate_soft_embeddings(
        self,
        logits: torch.Tensor,           # [batch, vocab_size]
        embedding_matrix: torch.Tensor, # [vocab_size, embedding_dim]
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Create soft embedding by aggregating top-k token embeddings.

        Args:
            logits: Model logits over vocabulary
            embedding_matrix: Token embedding matrix
            top_k: Number of top tokens to aggregate

        Returns:
            soft_embeddings: Aggregated embeddings [batch, embedding_dim]
        """
        # Compute probabilities
        probs = F.softmax(logits / self.temperature, dim=-1)

        # Get top-k
        topk_probs, topk_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)

        # Gather embeddings for top tokens
        # topk_indices: [batch, top_k]
        batch_size = topk_indices.shape[0]

        topk_embeddings = embedding_matrix[topk_indices]  # [batch, top_k, embedding_dim]

        # Weight by probabilities
        weights = topk_probs.unsqueeze(-1)  # [batch, top_k, 1]
        weighted_embeddings = topk_embeddings * weights

        # Aggregate
        soft_embeddings = weighted_embeddings.sum(dim=1)  # [batch, embedding_dim]

        # Normalize
        soft_embeddings = F.normalize(soft_embeddings, p=2, dim=-1)

        return soft_embeddings
```

### Step 4: Hybrid reasoning step

Combine both branches in single reasoning step.

```python
class HybridReasoningStep:
    """Single reasoning step with routing."""

    def __init__(
        self,
        model,
        embedding_matrix: torch.Tensor,
        routing_threshold: float = 0.8,
        discrete_temperature: float = 0.7,
        latent_temperature: float = 1.0
    ):
        self.model = model
        self.embedding_matrix = embedding_matrix

        self.router = ConfidenceRouter(routing_threshold=routing_threshold)
        self.discrete_branch = DiscreteReasoningBranch(temperature=discrete_temperature)
        self.latent_branch = LatentReasoningBranch(
            embedding_dim=embedding_matrix.shape[-1],
            temperature=latent_temperature
        )

    def reasoning_step(
        self,
        current_embeddings: torch.Tensor,  # [batch, seq_len, embedding_dim]
        context_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Execute one hybrid reasoning step.

        Args:
            current_embeddings: Current state embeddings
            context_mask: Attention mask if needed

        Returns:
            next_state: New embeddings after reasoning step
            routing_stats: Statistics on routing decisions
        """
        # Forward pass to get logits
        logits = self.model.get_logits(current_embeddings, context_mask)

        # Routing decision
        max_probs, route_to_latent, max_token_idx = self.router.compute_routing_decision(logits)

        batch_size = logits.shape[0]
        next_embeddings = torch.zeros_like(current_embeddings[:, -1:, :])  # [batch, 1, dim]

        # Discrete branch
        discrete_mask = ~route_to_latent
        if discrete_mask.any():
            discrete_tokens, discrete_log_probs = self.discrete_branch.sample_discrete_token(
                logits[discrete_mask]
            )
            discrete_embeddings = self.embedding_matrix[discrete_tokens]
            next_embeddings[discrete_mask] = discrete_embeddings.unsqueeze(1)

        # Latent branch
        if route_to_latent.any():
            latent_embeddings = self.latent_branch.aggregate_soft_embeddings(
                logits[route_to_latent],
                self.embedding_matrix
            )
            next_embeddings[route_to_latent] = latent_embeddings.unsqueeze(1)

        # Concatenate with previous states
        next_state = torch.cat([current_embeddings, next_embeddings], dim=1)

        routing_stats = self.router.visualize_routing_statistics(
            max_probs.tolist(),
            route_to_latent.tolist()
        )

        return next_state, routing_stats
```

### Step 5: Full reasoning trajectory with ThinkRouter

Generate complete reasoning sequences with routing.

```python
def generate_with_thinkrouter(
    model,
    embedding_matrix: torch.Tensor,
    prompt_embedding: torch.Tensor,  # [batch, 1, embedding_dim]
    max_steps: int = 100,
    routing_threshold: float = 0.8,
    device: str = 'cuda'
) -> dict:
    """
    Generate reasoning trajectory with dynamic routing.

    Args:
        model: Language model with get_logits method
        embedding_matrix: Token embedding matrix
        prompt_embedding: Initial prompt embeddings
        max_steps: Maximum reasoning steps
        routing_threshold: Confidence threshold for routing
        device: Training device

    Returns:
        trajectory: Generated tokens and embeddings
        routing_stats: Aggregated routing statistics
    """
    hybrid_reasoner = HybridReasoningStep(
        model, embedding_matrix,
        routing_threshold=routing_threshold
    )

    current_state = prompt_embedding.to(device)
    generated_tokens = []
    routing_decisions = []
    routing_stats_list = []

    for step in range(max_steps):
        # Hybrid reasoning step
        next_state, step_stats = hybrid_reasoner.reasoning_step(current_state)
        current_state = next_state

        routing_stats_list.append(step_stats)

        # Get last token for next iteration
        last_logits = model.get_logits(current_state[:, -1:, :])
        max_prob, route_decision, token_idx = hybrid_reasoner.router.compute_routing_decision(
            last_logits
        )

        generated_tokens.append(token_idx)
        routing_decisions.append(route_decision)

    # Aggregate statistics
    avg_stats = {
        'mean_latent_ratio': sum(s['latent_ratio'] for s in routing_stats_list) / len(routing_stats_list),
        'mean_discrete_ratio': sum(s['discrete_ratio'] for s in routing_stats_list) / len(routing_stats_list),
        'mean_confidence': sum(s['mean_max_prob'] for s in routing_stats_list) / len(routing_stats_list)
    }

    return {
        'generated_tokens': generated_tokens,
        'routing_decisions': routing_decisions,
        'final_embedding': current_state,
        'stats': avg_stats
    }
```

### Step 6: Tune routing threshold

Grid search for optimal threshold.

```python
def tune_routing_threshold(
    model,
    embedding_matrix: torch.Tensor,
    validation_tasks: list,
    verifier,
    threshold_candidates: list = None,
    device: str = 'cuda'
) -> float:
    """
    Find optimal routing threshold via grid search.

    Args:
        threshold_candidates: Thresholds to evaluate (default: 0.5-0.95)

    Returns:
        best_threshold: Optimal threshold for this model-task pair
    """
    if threshold_candidates is None:
        threshold_candidates = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    best_score = -float('inf')
    best_threshold = 0.8

    for threshold in threshold_candidates:
        scores = []

        for task in validation_tasks[:10]:  # Validate on subset
            result = generate_with_thinkrouter(
                model, embedding_matrix, task['prompt'],
                routing_threshold=threshold, device=device
            )

            score = verifier(result['generated_tokens'], task)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        print(f"Threshold {threshold}: Score {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_threshold = threshold

    return best_threshold
```

## Practical Guidance

**When to use**: Reasoning tasks where model confidence correlates with reasoning quality (math, code, multi-step planning)

**Hyperparameters**:
- **routing_threshold**: 0.7-0.9 (grid-search on validation set; ~0.8 typical)
- **discrete_temperature**: 0.5-1.0 (commitment preference)
- **latent_temperature**: 0.8-1.2 (exploration preference)
- **top_k**: 30-100 (vocabulary size for latent aggregation)

**Key advantages**:
- Robust reasoning by avoiding low-confidence latent aggregation
- Efficient with selective discrete sampling
- Seamless space switching within trajectory
- Improved accuracy under soft-embedding constraints

**Common pitfalls**:
- Threshold not tuned per model-task pair → suboptimal routing
- Not analyzing routing statistics → missing efficiency gains
- Latent noise not addressed → still incorrect high-confidence predictions
- Threshold too high → defaults to all-discrete (loses latent efficiency)

**Scaling**: Negligible overhead. Routing is O(batch_size · vocab_size).

## Reference

Paper: https://arxiv.org/abs/2602.11683
Related work: Confidence-based decisions, latent reasoning, sampling strategies
Benchmarks: MATH, reasoning tasks requiring extended thinking
