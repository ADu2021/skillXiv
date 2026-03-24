---
name: dr-llm-dynamic-routing
title: "Dr.LLM: Dynamic Layer Routing in LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.12773"
keywords: [dynamic-routing, layer-skipping, inference-efficiency, transformer, mcts-training]
description: "Use per-layer routers trained with Monte Carlo Tree Search to dynamically skip, execute, or repeat transformer layers for each token. Saves ~5 layers per example while maintaining accuracy on diverse benchmarks."
---

# Dr.LLM: Dynamic Layer Routing for Inference Efficiency

Transformers process every token through every layer, wasting computation on simple tokens that don't need deep processing. Dr.LLM trains lightweight per-layer routers that independently decide whether to skip, execute, or repeat each layer for each input token.

Core insight: different tokens require different computational depths. Hard reasoning tokens may benefit from layer repeating; simple copy tokens can skip layers entirely. By training routers with MCTS to find optimal configurations, you preserve accuracy while cutting computational cost by 20-30%.

## Core Concept

**Per-Layer Routers**: Each transformer layer has a lightweight router that decides per-token whether to skip (reuse cache), execute (normal computation), or repeat (process twice) that layer.

**MCTS-Based Training**: Rather than heuristic routing, use Monte Carlo Tree Search to derive high-quality layer configurations that preserve accuracy under compute budget constraints.

**Retrofittable Design**: Works with existing pretrained models without modifying base weights, making it practical for any transformer architecture.

## Architecture Overview

- **Layer Router**: Lightweight binary/ternary classifier per layer
- **Routing Options**: skip, execute, repeat for each layer-token pair
- **MCTS Trainer**: Searches for optimal configurations during training
- **Cache Manager**: Handles skipped layer outputs efficiently

## Implementation Steps

**Stage 1: Implement Per-Layer Routers**

Add lightweight routing heads to each layer:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class LayerRouter(nn.Module):
    def __init__(self, hidden_dim, num_routes=3):
        """
        Lightweight router for a single transformer layer.
        Routes: 0=skip, 1=execute, 2=repeat
        """
        super().__init__()

        self.router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_routes)
        )

        self.num_routes = num_routes

    def forward(self, hidden_states):
        """
        Per-token routing decisions.
        Returns logits for skip/execute/repeat.
        """
        logits = self.router(hidden_states)  # [batch, seq_len, num_routes]
        return logits

class DynamicLayerLLM(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add routers to each layer
        self.routers = nn.ModuleList([
            LayerRouter(self.base_model.config.hidden_size)
            for _ in range(len(self.base_model.transformer.h))
        ])

    def forward(self, input_ids, routing_decisions=None):
        """
        Forward pass with dynamic layer routing.
        If routing_decisions not provided, use greedy routing.
        """

        hidden_states = self.base_model.get_input_embeddings()(input_ids)
        layer_outputs = []

        for layer_idx, layer in enumerate(self.base_model.transformer.h):
            # Router decision for this layer
            router_logits = self.routers[layer_idx](hidden_states)

            if routing_decisions is not None:
                decisions = routing_decisions[layer_idx]
            else:
                # Greedy: choose max logit
                decisions = router_logits.argmax(dim=-1)  # [batch, seq_len]

            # Apply routing
            batch_size, seq_len = hidden_states.shape[:2]
            new_hidden = hidden_states.clone()

            for route_type in [0, 1, 2]:  # skip, execute, repeat
                mask = (decisions == route_type)

                if not mask.any():
                    continue

                if route_type == 0:  # skip
                    # Keep previous layer output
                    pass

                elif route_type == 1:  # execute
                    # Normal layer processing
                    layer_out = layer(
                        hidden_states[mask],
                        attention_mask=None
                    )
                    new_hidden[mask] = layer_out[0]

                elif route_type == 2:  # repeat
                    # Process layer twice
                    layer_out = layer(
                        hidden_states[mask],
                        attention_mask=None
                    )
                    layer_out = layer(
                        layer_out[0],
                        attention_mask=None
                    )
                    new_hidden[mask] = layer_out[0]

            hidden_states = new_hidden
            layer_outputs.append((hidden_states, router_logits))

        # Language modeling head
        logits = self.base_model.lm_head(hidden_states)

        return logits, layer_outputs
```

**Stage 2: Monte Carlo Tree Search for Training**

Use MCTS to find optimal routing configurations:

```python
class RoutingMCTS:
    def __init__(self, model, num_simulations=100):
        self.model = model
        self.num_simulations = num_simulations

    def search(self, input_ids, target_budget=0.7):
        """
        Search for routing configuration that maximizes accuracy
        while keeping compute below budget.

        Args:
            input_ids: token sequence
            target_budget: fraction of original compute to use

        Returns:
            optimal_routing: [num_layers, batch, seq_len]
        """

        best_config = None
        best_score = -float('inf')

        for sim in range(self.num_simulations):
            # Sample routing configuration
            routing_config = self.sample_routing_config(input_ids)

            # Evaluate configuration
            logits, _ = self.model(input_ids, routing_decisions=routing_config)
            accuracy = self.compute_accuracy(logits)

            # Compute efficiency
            compute_cost = self.estimate_compute(routing_config)
            efficiency = compute_cost / target_budget

            # Score: maximize accuracy under budget
            score = accuracy - 0.5 * max(0, efficiency - 1.0)

            if score > best_score:
                best_score = score
                best_config = routing_config

        return best_config

    def sample_routing_config(self, input_ids):
        """
        Sample random routing configuration.
        """

        batch_size, seq_len = input_ids.shape
        num_layers = len(self.model.routers)

        config = []
        for layer_idx in range(num_layers):
            # Probability of each routing decision
            # Bias toward execution (route 1)
            probs = [0.2, 0.6, 0.2]  # skip, execute, repeat

            routes = torch.multinomial(
                torch.tensor(probs),
                batch_size * seq_len,
                replacement=True
            )

            layer_routes = routes.view(batch_size, seq_len)
            config.append(layer_routes)

        return config

    def estimate_compute(self, routing_config):
        """
        Estimate relative compute cost of routing configuration.
        """

        total_ops = 0.0

        for layer_idx, layer_routes in enumerate(routing_config):
            # skip = 0 ops, execute = 1x ops, repeat = 2x ops
            route_types = layer_routes.float()
            ops_per_token = route_types.clamp(min=0.1)  # Avoid 0 ops
            total_ops += ops_per_token.mean()

        # Normalize to full execution (all execute = num_layers)
        return total_ops / len(routing_config)
```

**Stage 3: Joint Training with Router Loss**

Train routers to find good configurations:

```python
def train_dynamic_routing(
    model,
    train_dataloader,
    num_epochs=5,
    target_budget=0.75
):
    """
    Train routers with MCTS-guided supervision.
    """

    optimizer = torch.optim.AdamW(
        [p for p in model.routers.parameters()],
        lr=1e-3
    )

    mcts = RoutingMCTS(model, num_simulations=50)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].cuda()
            target_ids = batch['input_ids'].cuda()  # Next token prediction

            # Find optimal routing via MCTS
            optimal_routing = mcts.search(
                input_ids,
                target_budget=target_budget
            )

            # Forward pass with optimal routing
            logits, layer_outputs = model(
                input_ids,
                routing_decisions=optimal_routing
            )

            # Language modeling loss
            lm_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model.base_model.config.vocab_size),
                target_ids.view(-1)
            )

            # Router loss: encourage predicted routing to match optimal
            router_loss = 0.0
            for layer_idx, (hidden, router_logits) in enumerate(
                layer_outputs
            ):
                optimal_routes = optimal_routing[layer_idx]

                # Cross-entropy between predicted and optimal
                layer_router_loss = torch.nn.functional.cross_entropy(
                    router_logits.view(-1, 3),
                    optimal_routes.view(-1)
                )

                router_loss = router_loss + layer_router_loss

            # Combined loss
            total_loss = lm_loss + 0.1 * router_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch}, Step {batch_idx}, "
                    f"Loss: {total_loss:.4f}"
                )
```

## Practical Guidance

**When to Use Dynamic Layer Routing:**
- Inference where compute efficiency matters (mobile, edge devices)
- Models already fine-tuned (retrofittable, no base weight modification)
- Workloads with variable complexity (mix of simple and hard tokens)

**When NOT to Use:**
- Applications requiring strict latency bounds (routing adds decision overhead)
- Tasks where all tokens need full depth (e.g., complex reasoning throughout)
- Models that don't have pretrained weights to retrofit

**Routing Strategy:**

| Route Type | Use Case | Compute |
|-----------|----------|---------|
| Skip | Simple tokens (spaces, punctuation) | 0.1x |
| Execute | Standard tokens | 1.0x |
| Repeat | Hard tokens (reasoning, ambiguous) | 2.0x |

**Hyperparameters:**

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| MCTS Simulations | 50-200 | Higher = better routing, slower training |
| Target Budget | 0.7-0.85 | 0.7 = aggressive skipping, 0.85 = conservative |
| Router Learning Rate | 1e-3 | Higher = faster convergence, less stability |
| Base LM Loss Weight | 1.0 | Keep high to preserve accuracy |

**Common Pitfalls:**
- Target budget too aggressive (significant accuracy loss)
- Routers undertrained (poor routing decisions)
- Not validating routing consistency across similar tokens
- Ignoring attention mask in router forward pass

## Reference

Based on the research at: https://arxiv.org/abs/2510.12773
