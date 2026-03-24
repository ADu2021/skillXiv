---
name: adaptive-agent-foundation-model
title: "A²FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.12838"
keywords: [agent foundation model, hybrid reasoning, tool use, task routing, cost efficiency]
description: "Route queries to specialized reasoning modes (internal reasoning, tool calling, or instant answers) using task-aware routing and Adaptive Policy Optimization to reduce inference costs by 45% while maintaining accuracy."
---

# Technique: Adaptive Agent Foundation Model — Efficient Hybrid Reasoning Routing

Current agent systems face a fundamental efficiency problem: reasoning-centric LLMs excel at internal chain-of-thought but cannot invoke external tools, while agentic LLMs can call tools but often lack deep reasoning. Both architectures tend to over-apply their primary capability—reasoning models overthink simple queries, agentic models make unnecessary tool calls. A²FM solves this by dynamically routing to the right mode for each query.

Rather than forcing all queries through the same pipeline, A²FM identifies simple queries that need instant answers, moderate queries requiring reasoning, and complex queries demanding tool interaction. This three-mode approach prevents wasted computation while maintaining performance across diverse benchmarks.

## Core Concept

A²FM operates on a **route-then-align** principle:
- **Task-aware routing**: Classify incoming queries into three modes
- **Mode-specific trajectories**: Maintain specialized reasoning paths for each mode
- **Shared foundation backbone**: All modes align under a single 32B model
- **Adaptive Policy Optimization**: Cost-regularized rewards enforce efficient mode selection

The innovation prevents over-specification: answering "What is the capital of France?" through multi-step tool calls is wasteful, as is reasoning deeply about straightforward requests.

## Architecture Overview

- **Input Router**: Query embedding → classify as instant/reasoning/tool mode
- **Instant Mode Handler**: Direct answer generation for factual questions (no reasoning overhead)
- **Reasoning Mode**: Deep chain-of-thought without external tool calls
- **Tool Mode**: Sequential tool invocation with verification loops
- **Unified Backbone**: Shared 32B foundation LLM with mode-specific adapter layers
- **APO Training**: Reward shaping that favors cost-efficient paths

## Implementation Steps

The routing decision happens once per query. This example shows how to implement query classification and mode-specific forward passes.

```python
import torch
import torch.nn as nn

class AdaptiveRouterA2FM(nn.Module):
    """Route queries to instant/reasoning/tool modes based on complexity."""

    def __init__(self, embedding_dim=768, hidden_dim=512):
        super().__init__()
        self.query_encoder = nn.Linear(embedding_dim, hidden_dim)
        self.router = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3 modes: instant, reasoning, tool
        )
        self.mode_names = ["instant", "reasoning", "tool"]

    def forward(self, query_embedding):
        """
        Args:
            query_embedding: shape (batch, embedding_dim)

        Returns:
            mode_logits: shape (batch, 3) - scores for each mode
            selected_mode: shape (batch,) - argmax mode indices
        """
        encoded = self.query_encoder(query_embedding)
        mode_logits = self.router(encoded)
        selected_mode = torch.argmax(mode_logits, dim=1)
        return mode_logits, selected_mode


def adaptive_forward_pass(
    query,
    router_model,
    instant_handler,
    reasoning_model,
    tool_model
):
    """
    Encode query, route to appropriate handler, return answer.
    """
    # Encode query
    query_emb = encode_query(query)  # placeholder

    # Route to mode
    mode_logits, selected_mode = router_model(query_emb)
    mode_idx = selected_mode.item()

    # Execute mode-specific handler
    if mode_idx == 0:  # Instant mode
        answer = instant_handler.answer(query)
        cost = 1  # Cheap

    elif mode_idx == 1:  # Reasoning mode
        answer = reasoning_model.chain_of_thought(query, max_steps=10)
        cost = 5  # Medium

    else:  # Tool mode
        answer = tool_model.invoke_tools_with_reasoning(
            query, available_tools=TOOLS, max_calls=3
        )
        cost = 15  # Expensive

    return answer, cost, mode_idx
```

Cost-regularized rewards: During RL training, reward = accuracy - 0.1 * cost. Higher lambda penalizes wasteful routing. Tune on development set to balance accuracy vs. efficiency.

```python
def adaptive_policy_optimization_reward(
    answer_correct,
    mode_cost,
    lambda_cost=0.1
):
    """
    Reward function for APO training.
    Penalizes both incorrect answers AND expensive routing.
    """
    accuracy_reward = float(answer_correct)
    cost_penalty = -lambda_cost * mode_cost
    total_reward = accuracy_reward + cost_penalty
    return total_reward
```

## Practical Guidance

| Query Type | Best Mode | Example |
|-----------|-----------|---------|
| Factual (capital, definition) | Instant | "What is Paris?" |
| Reasoning (math, analysis) | Reasoning | "Explain why democracies need checks and balances" |
| Tool-dependent (weather, current info) | Tool | "What events are trending today?" |

**When to Use:**
- Serving diverse query types with variable complexity
- Cost-constrained inference (fewer tokens per query matters)
- Mix of factual, reasoning, and tool-dependent tasks
- You have labeled training data for query complexity classification

**When NOT to Use:**
- All queries require deep reasoning (routing overhead not justified)
- Tool availability is sparse or unreliable
- No budget for routing classification overhead

**Common Pitfalls:**
- Router biases toward most common mode (oversample hard examples during training)
- Cost weights not calibrated to actual inference costs
- Instant handler lacks sufficient context—adds errors despite apparent cheapness
- Over-penalizing tool mode when tool calls are actually necessary

## Reference

[A²FM: An Adaptive Agent Foundation Model for Tool-Aware Hybrid Reasoning](https://arxiv.org/abs/2510.12838)
