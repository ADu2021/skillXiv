---
name: deep-agent-reasoning
title: "DeepAgent: A General Reasoning Agent with Scalable Toolsets"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.21618"
keywords: [Agent, Reasoning, Tool Learning, RL, Memory Management]
description: "Enables autonomous reasoning agents to discover and invoke tools efficiently through end-to-end training. Uses autonomous memory folding to compress interaction history and ToolPO to learn general-purpose tool use, applicable across diverse benchmarks from QA to web automation."
---

# DeepAgent: Unified Autonomous Reasoning with Tool Learning

Existing reasoning agents struggle with two key limitations: they accumulate errors across long-horizon tasks through verbose interaction histories, and they require task-specific tool interfaces rather than learning generalizable tool use patterns.

DeepAgent solves this by integrating autonomous thinking, tool discovery, and action execution into a single end-to-end reasoning process. The system combines memory compression with learned tool invocation, enabling agents to handle complex multi-step tasks efficiently.

## Core Concept

DeepAgent operates through three integrated mechanisms:

- **Autonomous Memory Folding**: Compresses past interactions into structured episodic, working, and tool memories, reducing error propagation
- **ToolPO (Tool Policy Optimization)**: An end-to-end RL strategy using simulated APIs and fine-grained tool-call advantage attribution
- **Tool Retrieval**: Handles both labeled-tool and open-set discovery scenarios

## Architecture Overview

- Memory compression captures essential interaction patterns without verbose history
- Tool-call advantage attribution isolates credit signals to tool invocation tokens
- Memory types (episodic, working, tool) serve different reasoning stages
- End-to-end training enables discovery of effective tool combinations

## Implementation Steps

The memory folding mechanism selectively summarizes interactions at each step. Rather than maintaining full conversation history, compress past state and actions into dense representations:

```python
class MemoryFolder:
    def fold_interaction(self, history, current_state):
        # Compress episodic memory: factual outcomes from past steps
        episodic = self.compress_facts(history)
        # Working memory: intermediate reasoning state
        working = self.compress_reasoning(current_state)
        # Tool memory: effective tool patterns
        tools = self.extract_tool_patterns(history)
        return {episodic, working, tools}

    def compress_facts(self, history):
        # Extract key outcomes and state changes
        return [fact for fact in history if is_critical(fact)]

    def extract_tool_patterns(self, history):
        # Track which tools succeeded in which contexts
        return {(context, goal): tool for context, goal, tool in history}
```

ToolPO applies advantage attribution at the token level for tool calls. Rather than assigning credit to entire generation steps, focus reward signals on the tokens that invoke tools:

```python
class ToolPO:
    def compute_advantage(self, trajectory, reward):
        # Identify tool-call tokens in the generation
        tool_tokens = [idx for idx, token in enumerate(trajectory)
                      if is_tool_invocation(token)]

        # Assign advantage only to tool-invocation tokens
        advantage = {}
        for idx in tool_tokens:
            # Fine-grained credit based on outcome
            advantage[idx] = compute_token_advantage(trajectory, idx, reward)

        return advantage
```

## Practical Guidance

| Aspect | Recommendation |
|--------|-----------------|
| Memory compression ratio | 4:1 to 8:1 (reduce interaction sequences by 75-87%) |
| Tool-call token weighting | 2-5x higher than other tokens during RL training |
| Episodic memory retention | Keep last N=10 critical facts per domain |
| Simulated API complexity | Match target environment sophistication |

**When to use DeepAgent:**
- Multi-step reasoning tasks requiring tool invocation
- Long-horizon problems where error accumulation matters
- Scenarios with large, diverse tool libraries to explore

**When NOT to use:**
- Single-step tasks without tool requirements
- Domains with strictly defined tool interfaces (use API-specific agents)
- Real-time systems where memory compression adds latency

**Common pitfalls:**
- Over-compressing memory and losing critical context
- Under-weighting tool-specific advantage signals
- Insufficient diversity in simulated API trajectories during training

Reference: [DeepAgent on arXiv](https://arxiv.org/abs/2510.21618)
