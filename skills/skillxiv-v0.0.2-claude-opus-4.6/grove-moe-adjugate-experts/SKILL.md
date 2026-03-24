---
name: grove-moe-adjugate-experts
title: Grove MoE - Efficient Mixture of Experts with Adjugate Experts
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.07785
keywords: [mixture-of-experts, heterogeneous-experts, adaptive-activation, model-efficiency]
description: "Enables efficient MoE architectures through heterogeneous expert sizing and dynamic activation mechanisms that adjust parameter count based on input complexity."
---

## Grove MoE: Efficient Mixture of Experts with Adjugate Experts

### Core Concept

Grove MoE introduces heterogeneous expert sizing to Mixture of Experts architectures, inspired by big.LITTLE CPU designs. Instead of maintaining uniform-sized experts, Grove uses experts of varying sizes with dynamic activation that selects different parameter counts based on input complexity, enabling efficient computation for different token types.

### Architecture Overview

- **Heterogeneous Expert Pool**: Mix of small and large experts with varying parameter counts
- **Dynamic Activation Mechanism**: Routes tokens to appropriate expert sizes based on complexity signals
- **Adaptive Parameter Budget**: Activates 3-4B parameters selectively while maintaining 33B total capacity
- **Upcycling Strategy**: Transforms existing base models during continued training rather than retraining from scratch

### Implementation Steps

**Step 1: Design Heterogeneous Expert Architecture**

Create the expert layer with varying sizes:

```python
# Pseudocode for heterogeneous experts
class HeterogeneousExpertLayer(nn.Module):
    def __init__(self, hidden_dim, num_small_experts=16, num_large_experts=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Small experts (4x smaller parameters)
        self.small_experts = nn.ModuleList([
            Expert(hidden_dim, hidden_dim * 2, hidden_dim)
            for _ in range(num_small_experts)
        ])

        # Large experts (standard size)
        self.large_experts = nn.ModuleList([
            Expert(hidden_dim, hidden_dim * 4, hidden_dim)
            for _ in range(num_large_experts)
        ])

        # Token complexity predictor
        self.complexity_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Route tokens to experts based on predicted complexity.
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Predict token complexity
        complexity_scores = self.complexity_head(x)
        normalized_complexity = torch.sigmoid(complexity_scores)

        # Route: low complexity to small experts, high to large
        outputs = []
        for i in range(batch_size):
            token_out = []
            for j in range(seq_len):
                if normalized_complexity[i, j] < 0.5:
                    # Route to small expert
                    expert_idx = j % len(self.small_experts)
                    out = self.small_experts[expert_idx](x[i:i+1, j:j+1])
                else:
                    # Route to large expert
                    expert_idx = j % len(self.large_experts)
                    out = self.large_experts[expert_idx](x[i:i+1, j:j+1])
                token_out.append(out)
            outputs.append(torch.cat(token_out, dim=1))

        return torch.stack(outputs, dim=0)

class Expert(nn.Module):
    def __init__(self, hidden_dim, feedforward_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
```

**Step 2: Implement Adaptive Routing**

Design the routing mechanism that selects expert pool based on complexity:

```python
# Pseudocode for adaptive routing
class AdaptiveRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts_small, num_experts_large):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Routing networks for each expert pool
        self.small_router = nn.Linear(hidden_dim, num_experts_small)
        self.large_router = nn.Linear(hidden_dim, num_experts_large)

        # Complexity gate
        self.complexity_gate = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary: small or large pool
        )

    def forward(self, x):
        """
        Route to appropriate expert pool and select expert within pool.
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Compute gate logits (small vs large)
        gate_logits = self.complexity_gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Compute expert selection within each pool
        small_logits = self.small_router(x)
        large_logits = self.large_router(x)

        # Soft routing with gating
        small_probs = F.softmax(small_logits, dim=-1)
        large_probs = F.softmax(large_logits, dim=-1)

        return {
            'gate_probs': gate_probs,
            'small_expert_probs': small_probs,
            'large_expert_probs': large_probs
        }
```

**Step 3: Apply Upcycling Strategy**

Transform existing models during continued training:

```python
# Pseudocode for upcycling
def upcycle_model(base_model, target_moe_config, training_data, num_epochs=5):
    """
    Convert base model to Grove MoE during continued training.
    """
    # Initialize Grove MoE layers replacing feedforward layers
    grove_model = copy.deepcopy(base_model)

    for i, layer in enumerate(grove_model.layers):
        # Replace feedforward with heterogeneous expert layer
        layer.mlp = HeterogeneousExpertLayer(
            hidden_dim=layer.hidden_dim,
            num_small_experts=target_moe_config['num_small'],
            num_large_experts=target_moe_config['num_large']
        )

        # Copy weights from original FFN to initialize new experts
        initialize_from_original(layer.mlp, base_model.layers[i].mlp)

    # Train with upcycled architecture
    optimizer = AdamW(grove_model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in training_data:
            outputs = grove_model(batch['input_ids'])
            loss = compute_loss(outputs, batch['labels'])

            loss.backward()
            optimizer.step()

    return grove_model
```

**Step 4: Monitor and Optimize Activation Patterns**

Track actual parameter activation during inference:

```python
# Pseudocode for activation monitoring
class ActivationMonitor:
    def __init__(self):
        self.activation_counts = {}

    def track_activation(self, model, batch):
        """
        Monitor which experts activate for which tokens.
        """
        total_params = sum(p.numel() for p in model.parameters())
        active_params = 0

        # Forward pass with tracking
        with torch.no_grad():
            for token in batch:
                routing_info = model.router(token)

                # Count activated expert parameters
                for expert_idx, prob in enumerate(routing_info['small_expert_probs']):
                    if prob > 0.1:
                        active_params += estimate_expert_params('small', expert_idx)

        activation_ratio = active_params / total_params
        return activation_ratio
```

### Practical Guidance

**Hyperparameters and Configuration**:
- Small expert feedforward dimension: 2x hidden dim
- Large expert feedforward dimension: 4x hidden dim
- Expert ratio: ~16 small, 4 large experts per layer
- Routing temperature: 1.0-2.0 for sharper selection
- Upcycling learning rate: 1e-4 (lower than standard training)

**When to Use Grove MoE**:
- Scenarios requiring variable computational budgets for different inputs
- Inference systems with mixed token complexity (some simple, some complex)
- Models where heterogeneous scaling improves efficiency-performance tradeoff
- Continued training scenarios where model modification is acceptable

**When NOT to Use**:
- Strict latency constraints (routing overhead impacts latency)
- Workloads with uniform token complexity
- Systems requiring deterministic expert usage patterns
- When exact reproducibility across runs is critical

**Implementation Notes**:
- Complexity prediction must be fast (simple linear layer) to avoid routing overhead
- Initialize small experts from subsets of original FFN weights for faster convergence
- Monitor activation distribution to ensure both pools are utilized
- Consider load balancing losses if expert utilization becomes skewed

### Reference

Paper: Grove MoE: Efficient MoE LLMs with Adjugate Experts
ArXiv: 2508.07785
Performance: 3.14-3.28B activated parameters from 33B total, matching larger model performance
