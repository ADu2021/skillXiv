---
name: mu-parametrization-moe
title: "μ-Parametrization for Mixture of Experts"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.09752
keywords: [mixture-of-experts, hyperparameter-transfer, mu-transfer, scaling, learning-rate]
description: "Apply μ-parametrization to Mixture-of-Experts architectures to enable reliable hyperparameter transfer across model sizes, eliminating costly retuning when scaling to trillion-parameter systems."
---

# μ-Parametrization for Mixture of Experts

## Core Concept

μ-Parametrization is a principled approach to hyperparameter tuning that enables hyperparameters (especially learning rates) to transfer reliably across different model sizes and widths. This work extends μ Transfer—previously developed for dense language models—to Mixture-of-Experts (MoE) architectures, which now dominate large-scale language models.

The key innovation is providing theoretical guarantees that feature learning remains consistent across different MoE widths, allowing practitioners to tune hyperparameters on smaller models and apply them directly to trillion-parameter systems without expensive retuning.

## Architecture Overview

- **Width-Invariant Learning**: Ensures that the optimal learning rate for a small model remains optimal (or requires minimal adjustment) for larger models
- **Scaling Principles**: Accounts for MoE-specific scaling factors (number of experts, routing dynamics, expert capacity)
- **Feature Learning Consistency**: Theoretical framework guarantees that learned representations remain stable across model scales
- **Efficient Hyperparameter Space**: Defines a reparametrized learning rate schedule that transfers across scales
- **Gradient Signal Preservation**: Maintains gradient magnitudes and step sizes despite changes in model width

## Implementation Steps

### 1. Understand μ-Parametrization Basics for Dense Models

The foundational concept: instead of using a fixed learning rate, scale it by model width dimensions. For a dense model with width d_model, the μ-parametrized learning rate is:

```
learning_rate = base_lr / sqrt(d_model)
```

This ensures that gradient updates have consistent effects across different model widths, preventing explosive gradients in wider models or vanishing gradients in narrower ones.

```python
# Basic μ-parametrization for dense models
def compute_mu_learning_rate(base_lr, d_model):
    """
    Compute width-invariant learning rate
    base_lr: target learning rate (tuned on small model)
    d_model: hidden dimension width
    """
    mu_lr = base_lr / (d_model ** 0.5)
    return mu_lr

# Example: a model with d_model=768 and target base_lr=1e-3
small_d_model = 768
large_d_model = 2048

small_lr = compute_mu_learning_rate(1e-3, small_d_model)  # ≈ 1e-3 / 27.7
large_lr = compute_mu_learning_rate(1e-3, large_d_model)  # ≈ 1e-3 / 45.3
# The *same* base_lr produces appropriate step sizes for both models
```

### 2. Extend to Mixture-of-Experts Dimensions

In MoE architectures, multiple dimensions affect gradient flow: expert width, number of experts, and routing distributions. Extend μ-parametrization to account for MoE-specific factors.

```python
def compute_moe_mu_learning_rate(base_lr, d_model, num_experts, expert_width,
                                  routing_load_balance=1.0):
    """
    μ-parametrization for MoE models.

    Args:
        base_lr: target learning rate (tuned on small MoE baseline)
        d_model: hidden dimension width
        num_experts: number of experts in the layer
        expert_width: width of each expert's FFN layer
        routing_load_balance: load balancing factor (1.0 = perfect balance)

    Returns:
        scaled learning rate suitable for this MoE configuration
    """
    # Base width scaling (from dense models)
    width_scale = d_model ** 0.5

    # MoE-specific scaling: account for expert parallelism
    # More experts with smaller width ≈ denser model with larger width
    expert_capacity = num_experts * expert_width
    moe_scale = expert_capacity ** 0.25  # Fourth-root to avoid over-scaling

    # Load balancing affects effective batch size
    load_balance_scale = routing_load_balance ** 0.5

    # Combined scaling factor
    total_scale = width_scale * moe_scale * load_balance_scale
    mu_lr = base_lr / total_scale

    return mu_lr
```

### 3. Calibrate Base Learning Rate on Small Model

Use a small MoE model (e.g., 7B) to find the optimal base_lr through standard hyperparameter tuning or grid search.

```python
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM

# Small baseline model for tuning
small_model = AutoModelForCausalLM.from_pretrained("moe-7b")

# Standard hyperparameter search
base_lr_candidates = [5e-4, 1e-3, 2e-3, 5e-3]
results = {}

for candidate_lr in base_lr_candidates:
    # Compute μ-parametrized learning rate for this model
    mu_lr = compute_moe_mu_learning_rate(
        candidate_lr,
        d_model=small_model.config.hidden_size,
        num_experts=small_model.config.num_local_experts,
        expert_width=small_model.config.intermediate_size
    )

    optimizer = AdamW(small_model.parameters(), lr=mu_lr)

    # Train on validation set, measure perplexity
    val_loss = train_and_evaluate(small_model, optimizer, val_dataset)
    results[candidate_lr] = val_loss

# Pick best base_lr
best_base_lr = min(results, key=results.get)
print(f"Optimal base_lr for small model: {best_base_lr}")
```

### 4. Transfer to Large Models Without Retuning

Apply the optimal base_lr to larger models by computing their MoE μ-learning rates. No expensive retuning needed.

```python
# Large model: 70B parameters
large_model = AutoModelForCausalLM.from_pretrained("moe-70b")

# Apply μ-parametrization with the SAME base_lr from small model
mu_lr_large = compute_moe_mu_learning_rate(
    base_lr=best_base_lr,  # ← Same base_lr from small model!
    d_model=large_model.config.hidden_size,
    num_experts=large_model.config.num_local_experts,
    expert_width=large_model.config.intermediate_size
)

optimizer_large = AdamW(large_model.parameters(), lr=mu_lr_large)

# Train with confidence that learning rate is appropriate for this scale
for epoch in range(num_epochs):
    train_step(large_model, optimizer_large, train_dataset)
    val_loss = evaluate(large_model, val_dataset)
    print(f"Epoch {epoch}, val_loss: {val_loss:.4f}")
```

### 5. Handle Routing Imbalance Correction

Account for realistic routing scenarios where load balancing is imperfect. Measure actual routing load balance and adjust.

```python
def measure_load_balance(model, batch):
    """Measure actual load balance ratio from routing decisions"""
    with torch.no_grad():
        # Forward pass to collect routing statistics
        outputs = model(batch, output_router_logits=True)

        total_tokens_per_expert = torch.zeros(model.config.num_local_experts)
        for layer_router in model.layer_routing_stats:
            # Count tokens routed to each expert
            router_logits = layer_router.logits
            expert_assignments = torch.argmax(router_logits, dim=-1)
            for expert_id in range(model.config.num_local_experts):
                count = (expert_assignments == expert_id).sum()
                total_tokens_per_expert[expert_id] += count

    # Compute load balance ratio (1.0 = perfect)
    max_tokens = total_tokens_per_expert.max()
    min_tokens = total_tokens_per_expert.min()
    load_balance = min_tokens / (max_tokens + 1e-8)
    return load_balance

# During training, periodically measure and log load balance
load_balance = measure_load_balance(large_model, batch)

# If imbalanced, can optionally adjust learning rate (conservative)
if load_balance < 0.7:
    adjusted_mu_lr = mu_lr_large * (load_balance ** 0.25)
    for param_group in optimizer_large.param_groups:
        param_group['lr'] = adjusted_mu_lr
```

### 6. Validation Across Scales

Train both small and large models with the same base_lr using μ-parametrization. Verify that convergence behavior and final performance are comparable.

```python
# Comparison training loop
results = {}

for model_size, model in [('small', small_model), ('large', large_model)]:
    mu_lr = compute_moe_mu_learning_rate(
        best_base_lr,
        d_model=model.config.hidden_size,
        num_experts=model.config.num_local_experts,
        expert_width=model.config.intermediate_size
    )

    optimizer = AdamW(model.parameters(), lr=mu_lr)
    losses = []

    for step, batch in enumerate(train_loader):
        loss = train_step(model, optimizer, batch)
        losses.append(loss)

        if step % 1000 == 0:
            val_loss = evaluate(model, val_dataset)
            print(f"{model_size} - Step {step}, train_loss: {loss:.4f}, val_loss: {val_loss:.4f}")

    results[model_size] = losses

# Verify: both models should show similar convergence patterns
import matplotlib.pyplot as plt
plt.plot(results['small'], label='Small MoE')
plt.plot(results['large'], label='Large MoE')
plt.legend()
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('μ-Parametrization: Consistent Convergence Across Scales')
plt.show()
```

## Practical Guidance

### Hyperparameters & Configuration

- **Base Learning Rate**: Typically in range 5e-4 to 5e-3; tune on small model with grid search or learning rate range test
- **Width Exponent**: Use 0.5 for main width scaling (square root rule); proven robust across architectures
- **Expert Capacity Factor**: Use 0.25-0.5 exponent; 0.25 recommended for standard MoE (balances expert and dense dimensions)
- **Load Balance Adjustment**: Optional; only apply if routing is severely imbalanced (< 0.5 ratio)
- **Gradient Clipping**: Pair with gradient norm clipping (e.g., max_norm=1.0) for stability

### When to Use μ-Parametrization for MoE

- You're training MoE models across multiple scales (7B to 70B to 700B)
- Hyperparameter tuning budget is limited (each tuning iteration is expensive)
- You want to ensure consistent training dynamics across different model sizes
- You're developing trillion-parameter systems where retuning is prohibitively expensive
- You have a well-tuned small baseline model to transfer from

### When NOT to Use μ-Parametrization

- You only train a single model size (no scaling needed)
- Your models use non-standard expert architectures (sparse mixture, hierarchical routing)
- You have unlimited compute budget for full retuning at each scale
- Routing patterns differ significantly between small and large models
- You're using custom optimizer-specific techniques that break scale invariance

### Common Pitfalls

1. **Ignoring MoE-Specific Dimensions**: Applying only dense-model μ-transfer to MoE produces suboptimal learning rates. Account for expert count and capacity.
2. **Tuning With Imbalanced Routing**: If small model has imbalanced routing that improves with size, the transfer assumes this won't change. Validate on large model.
3. **No Validation Across Scales**: Never assume transfer works without training both small and large models. Verify comparable convergence curves.
4. **Over-Scaling for Large Models**: If large model has different architecture (more experts, different widths), recompute carefully. Don't assume linear scaling.
5. **Forgetting Gradient Clipping**: μ-parametrization affects gradient magnitudes; pair with appropriate gradient norm clipping.

## Reference

μ-Parametrization for MoE (2508.09752): https://arxiv.org/abs/2508.09752

Extends μ Transfer to Mixture-of-Experts, enabling optimal learning rates to reliably transfer across model sizes and eliminating expensive hyperparameter retuning for trillion-scale MoE systems.
