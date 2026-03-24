---
name: recurrent-depth-vla
title: "Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of VLA Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.07845"
keywords: [Vision-Language-Action Models, Latent Reasoning, Adaptive Computation, Robotics, Weight-Tied Recurrence]
description: "Enable test-time compute scaling in vision-language-action models via weight-tied recurrent inference within latent space, with adaptive stopping based on action divergence."
---

# Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of VLA Models

## Problem Context

Current Vision-Language-Action (VLA) models suffer from fixed computational depth and inefficient reasoning. Token-based Chain-of-Thought methods scale memory linearly and are poorly suited for continuous action spaces required by robotic control. Yet task complexity varies significantly—easy navigation requires minimal reasoning while grasping requires extensive refinement.

## Core Concept

**Recurrent-Depth VLA (RD-VLA)** performs iterative reasoning entirely within latent representation space using a weight-tied recurrent transformer. This enables variable compute allocation based on task complexity without the memory overhead of autoregressive decoding, particularly suited for continuous action control.

## Architecture Overview

- **Prelude**: Processes learned queries via cross-attention to VLM features, creating grounded latent foundation
- **Recurrent Core**: Weight-tied transformer block iteratively refines latent "scratchpad" from noise initialization
- **Coda**: Final decoding stage projecting refined representations to action outputs
- **Adaptive Stopping**: Terminates when action divergence falls below convergence threshold

## Implementation

**Phase 1: Architecture Components**

```python
class RecurrentDepthVLA(nn.Module):
    """VLA with latent reasoning via weight-tied recurrence"""

    def __init__(self, vlm_dim=1024, latent_dim=512, action_dim=7):
        super().__init__()

        # Prelude: ground reasoning in VLM features
        self.prelude = nn.Sequential(
            nn.Linear(vlm_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

        # Learned queries for cross-attention
        self.learned_queries = nn.Parameter(
            torch.randn(8, latent_dim)  # 8 query vectors
        )

        # Cross-attention to VLM features
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads=4
        )

        # Recurrent core (weight-tied)
        self.recurrent_block = TransformerBlock(latent_dim)

        # Coda: decode latent state to actions
        self.coda = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, vlm_features, image, max_iterations=12):
        """
        vlm_features: (batch, seq_len, vlm_dim) - visual embeddings
        image: (batch, 3, H, W) - raw image
        max_iterations: maximum recurrence depth
        """

        batch_size = vlm_features.shape[0]

        # Prelude: ground in VLM space
        grounded = self.prelude(vlm_features.mean(dim=1))

        # Cross-attend to VLM features with learned queries
        queries = self.learned_queries.unsqueeze(0).expand(batch_size, -1, -1)
        latent_state, _ = self.cross_attn(
            queries, vlm_features, vlm_features
        )
        latent_state = latent_state.mean(dim=1)  # (batch, latent_dim)

        # Recurrent refinement
        actions_history = []
        prev_action = None

        for iteration in range(max_iterations):
            # Iterate weight-tied transformer
            latent_state = self.recurrent_block(latent_state)

            # Decode to action
            action = self.coda(latent_state)
            actions_history.append(action)

            # Check for convergence
            if prev_action is not None:
                divergence = kl_divergence(action, prev_action)

                if divergence < 0.01:  # Convergence threshold
                    break

            prev_action = action

        # Return final action
        return action, len(actions_history)

class TransformerBlock(nn.Module):
    """Single transformer block for recurrent application"""

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        """Apply transformer block (x is (batch, dim))"""
        # Self-attention (expand to sequence of 1)
        x_seq = x.unsqueeze(1)  # (batch, 1, dim)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x = x + attn_out.squeeze(1)

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x
```

**Phase 2: Training with Variable Recurrence Depth**

```python
def training_with_variable_depth(model, loader, max_depth=12):
    """
    Train model via truncated backprop through time (TBPTT).
    Randomly sample recurrence depth to force stable refinement.
    """

    for epoch in range(num_epochs):
        for batch in loader:
            vlm_features = batch['vlm_features']
            images = batch['images']
            actions_gt = batch['actions']

            # Sample random recurrence depth
            sample_depth = random.randint(1, max_depth)

            # Forward pass with sampled depth
            action_pred, actual_depth = model(
                vlm_features, images, max_iterations=sample_depth
            )

            # Loss on predicted action
            loss = mse_loss(action_pred, actions_gt)

            # Backward through sampled depth (TBPTT)
            loss.backward()
            optimizer.step()

            print(f"Actual depth: {actual_depth}, Loss: {loss.item():.4f}")
```

**Phase 3: Inference with Adaptive Stopping**

```python
def kl_divergence_action(action_a, action_b):
    """Approximate KL divergence between consecutive actions"""
    return torch.mean((action_a - action_b) ** 2)

def inference_adaptive(model, vlm_features, image, max_iterations=12,
                       divergence_threshold=0.01):
    """
    Inference with adaptive stopping based on action convergence.
    """

    action, num_iterations = model(
        vlm_features, image, max_iterations=max_iterations
    )

    return action, num_iterations

def benchmark_adaptive_vs_fixed(model, test_loader):
    """Compare adaptive vs. fixed-depth inference"""

    total_fixed_steps = 0
    total_adaptive_steps = 0
    accuracy_fixed = 0
    accuracy_adaptive = 0

    for batch in test_loader:
        vlm_features = batch['vlm_features']
        images = batch['images']
        actions_gt = batch['actions']

        # Fixed-depth inference (always 12 iterations)
        action_fixed, _ = model(vlm_features, images, max_iterations=12)
        total_fixed_steps += 12 * batch.shape[0]
        accuracy_fixed += compute_accuracy(action_fixed, actions_gt)

        # Adaptive inference
        action_adaptive, actual_depth = inference_adaptive(
            model, vlm_features, images, max_iterations=12
        )
        total_adaptive_steps += actual_depth * batch.shape[0]
        accuracy_adaptive += compute_accuracy(action_adaptive, actions_gt)

    print(f"Fixed-depth: {total_fixed_steps} total steps, "
          f"Accuracy: {accuracy_fixed / len(test_loader):.3f}")
    print(f"Adaptive: {total_adaptive_steps} total steps, "
          f"Accuracy: {accuracy_adaptive / len(test_loader):.3f}")
    print(f"Compute savings: "
          f"{100 * (1 - total_adaptive_steps / total_fixed_steps):.1f}%")
```

## Practical Guidance

**When to use**: Deploy for robotic control and continuous action tasks where compute budgets vary (easy vs. hard scenes). Less effective for discrete decision-making or classification.

**Latent dimension**: Use 512–1024 dims depending on action complexity. Larger dims provide more refinement capacity; smaller dims are more efficient.

**Convergence threshold**: Start with 0.01 KL divergence. Higher thresholds (0.02+) terminate earlier but risk suboptimal actions; lower thresholds (0.005) refine more but increase compute.

**Recurrent block design**: Weight-tied transformer works well; alternative is to use LSTM-style recurrence. Transformer provides better gradient flow.

**Training stability**: Truncated backprop with random depth (5–12) prevents overfitting to fixed depth and enables stable gradient flow through many iterations.

## Reference

RD-VLA achieves 93.0% success with fixed 12 iterations and 92.5% with adaptive computation on LIBERO benchmark, using only 0.5B parameters while outperforming larger baselines. The key insight is that latent reasoning (refining representations) is more compute-efficient than explicit token generation for continuous action spaces. Adaptive stopping provides 15–30% compute savings without significant accuracy loss.
