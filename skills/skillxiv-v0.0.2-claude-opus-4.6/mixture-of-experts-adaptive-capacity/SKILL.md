---
name: mixture-of-experts-adaptive-capacity
title: "DynaMoE: Dynamic Token-Level Expert Activation with Layer-Wise Adaptive Capacity for MoE"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.01697"
keywords: [Mixture of Experts, Adaptive Routing, Token-Level Dynamics, Capacity Planning, Sparse Models]
description: "Optimize Mixture-of-Experts efficiency by decoupling token-level expert activation from layer architecture. Use dynamic threshold routing where expert count per token varies by input complexity, and apply layer-wise capacity scheduling to match representational diversity patterns."
---

# DynaMoE: Adaptive Expert Capacity Allocation in Mixture-of-Experts

Standard Mixture-of-Experts (MoE) architectures use fixed Top-K routing, where exactly K experts activate per token across all layers. This constraint wastes capacity on simple inputs and under-scales complex ones. DynaMoE relaxes both constraints: enabling dynamic per-token expert counts and layer-dependent expert capacity scheduling to match the representational structure of deep networks.

The core insight is that different layers have different optimal expert counts based on input diversity. Early layers process heterogeneous raw inputs requiring diverse expert specialization; deeper layers exhibit representational convergence where fewer experts suffice. By jointly optimizing token routing flexibility and layer capacity distribution, DynaMoE achieves superior efficiency-performance trade-offs.

## Core Concept

DynaMoE makes two related innovations:

1. **Token-Level Threshold Routing**: Replace fixed Top-K with a percentile threshold where expert count varies per token based on input-specific routing logits. Simple inputs use 1-2 experts; complex inputs use more.

2. **Layer-Wise Capacity Scheduling**: Distribute total expert budget across layers using scheduling strategies (descending, ascending, pyramid, wave) to match how representational complexity evolves through the network.

This two-level decomposition aligns model capacity with actual computational needs rather than fixed architectural assumptions.

## Architecture Overview

- **Input**: Token sequences with per-token routing scores from gating network
- **Dynamic Threshold Extraction**: For each token, compute expert activation threshold based on percentile of routing logits
- **Expert Selection**: Activate all experts exceeding the threshold (variable count per token)
- **Capacity Budget Distribution**: Assign expert counts to layers via scheduling strategy
- **Output**: Efficient sparse forward/backward passes with adaptive activation patterns

## Implementation Steps

**Step 1: Compute per-token routing logits and dynamic thresholds**

Standard MoE gating networks output logits for each expert. Instead of selecting top-K, use percentile-based thresholding to determine active experts adaptively.

```python
# Forward pass through MoE gating network
batch_size, seq_len, num_experts = tokens.shape[0], tokens.shape[1], 128
logits = gating_network(tokens)  # Shape: [batch, seq, num_experts]

# Dynamic threshold routing: for each token, activate experts above percentile p
percentile = 50  # Can vary by task; higher = fewer active experts
thresholds = np.percentile(logits, percentile, axis=-1, keepdims=True)

# Binary mask for expert activation
active_mask = (logits >= thresholds).astype(np.float32)
num_active = active_mask.sum(axis=-1)  # Varies per token
```

**Step 2: Implement capacity scheduling strategy**

Distribute expert budget across layers based on representational diversity patterns. Choose scheduling strategy aligned with task structure.

```python
def layer_capacity_schedule(layer_idx, num_layers, total_experts, strategy='descending'):
    """Allocate expert count per layer."""
    if strategy == 'descending':
        # More experts early (diverse raw input), fewer deep (converged representations)
        # Quadratic decay for smooth transition
        ratio = (num_layers - layer_idx) / num_layers
        return int(total_experts * (ratio ** 2))

    elif strategy == 'ascending':
        # Fewer experts early, more deep
        ratio = layer_idx / num_layers
        return int(total_experts * (ratio ** 2))

    elif strategy == 'pyramid':
        # Peak capacity in middle layers
        mid = num_layers / 2
        dist_to_mid = abs(layer_idx - mid) / mid
        ratio = 1.0 - dist_to_mid
        return int(total_experts * (ratio ** 2))

    else:  # uniform
        return total_experts

# Allocate capacity per layer for 24-layer network with 256 total experts
expert_counts = [
    layer_capacity_schedule(i, num_layers=24, total_experts=256, strategy='descending')
    for i in range(24)
]
```

**Step 3: Select experts and route tokens within capacity constraints**

Route tokens to selected experts, handling overflow when activated experts exceed layer capacity.

```python
def route_with_overflow_handling(logits, active_mask, layer_expert_capacity):
    """Route tokens respecting dynamic activation and layer capacity."""
    batch_size, seq_len, num_experts = logits.shape

    # Get indices of active experts per token
    active_expert_indices = np.where(active_mask)

    # Apply load-balancing: if too many experts active, keep top-scoring ones
    expert_scores = logits * active_mask - 1e9 * (1 - active_mask)

    # For each token, select up to layer_expert_capacity experts
    selected_experts = []
    for token_idx in range(batch_size * seq_len):
        scores = expert_scores.reshape(batch_size * seq_len, -1)[token_idx]
        active = np.where(active_mask.reshape(batch_size * seq_len, -1)[token_idx])[0]

        # Prioritize by score within active set
        active_scores = [(idx, scores[idx]) for idx in active]
        active_scores.sort(key=lambda x: x[1], reverse=True)

        selected = [idx for idx, _ in active_scores[:layer_expert_capacity]]
        selected_experts.append(selected)

    return selected_experts
```

**Step 4: Execute sparse forward pass with routing weights**

Compute token-to-expert assignments and weighted combination of expert outputs.

```python
# Compute routing weights via softmax over active expert logits
expert_logits_masked = logits * active_mask - 1e9 * (1 - active_mask)
routing_weights = softmax(expert_logits_masked, axis=-1)

# Expert forward pass: dispatch tokens to selected experts
expert_outputs = []
for expert_idx in range(num_experts):
    # Collect tokens assigned to this expert
    token_mask = (active_mask[:, :, expert_idx] > 0).astype(np.float32)
    expert_input = tokens * token_mask.reshape(batch_size, seq_len, 1)

    # Expert forward pass (can be parallelized across experts)
    expert_output = expert_networks[expert_idx](expert_input)
    expert_outputs.append(expert_output)

# Weighted combination of expert outputs
combined_output = np.zeros_like(tokens)
for expert_idx in range(num_experts):
    weight = routing_weights[:, :, expert_idx].reshape(batch_size, seq_len, 1)
    combined_output += weight * expert_outputs[expert_idx]
```

**Step 5: Backward pass with gradient accumulation across routing decisions**

Support gradient flow through both gating network and expert networks to enable end-to-end training.

```python
# Gradient accumulation through routing and expert selection
# (Simplified; real implementation uses automatic differentiation)

# Loss backprop to expert outputs
dL_dexpert_output = loss.backward()

# Gradient through weighted combination
for expert_idx in range(num_experts):
    dL_drouting_weight = (dL_dexpert_output * expert_outputs[expert_idx]).sum()
    dL_dlogits[:, :, expert_idx] = dL_drouting_weight * (routing_weights - routing_weights**2)

# Load balancing auxiliary loss to prevent expert collapse
auxiliary_loss = load_balance_loss(routing_weights, active_mask)
total_loss = task_loss + 0.01 * auxiliary_loss
```

## Practical Guidance

**Hyperparameter Selection:**
- **Percentile for threshold**: 30-70 (higher = fewer active experts). Start at 50; adjust based on latency budgets.
- **Capacity scheduling strategy**: For image tasks use descending (early diversity dominates); for language tasks try ascending or uniform.
- **Layer expert counts**: Range from 64-256 depending on model scale; total budget ~4-8% of dense equivalent.

**When to Use:**
- Large-scale models (7B+ parameters) where efficiency gains compound
- Tasks with variable computational demands across samples (e.g., document length varies)
- Scenarios prioritizing inference latency and memory over peak training performance

**When NOT to Use:**
- Small models (<1B) where MoE overhead dominates benefits
- Tasks requiring uniform compute per sample (real-time control systems)
- Environments with strict memory constraints per device

**Common Pitfalls:**
- **Imbalanced expert utilization**: Some experts become idle. Add explicit load-balancing losses.
- **Capacity mismatches**: Layer capacity too low causes token overflow; too high wastes resources. Validate with profiling.
- **Routing instability during training**: Thresholds can jitter. Use exponential moving averages of logits for stability.
- **Neglecting backward pass complexity**: Gradient computation through dynamic routing can be expensive; use gradient checkpointing selectively.

## Reference

arXiv: https://arxiv.org/abs/2603.01697
