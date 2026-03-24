---
name: tcandon-multi-agent-router
title: "TCAndon-Router: Adaptive Reasoning Router for Multi-Agent Collaboration"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.04544"
keywords: [multi-agent, routing, task-assignment, dynamic-adaptation, agent-collaboration]
description: "Route queries to multiple specialized agents dynamically using reasoning-aware routing that generates natural-language justification before predicting candidate agents. Enables seamless addition of new agents without system redesign. Routes aggregate responses from multiple specialists into coherent final answers, supporting enterprise-scale multi-agent systems with overlapping capabilities."
---

## Problem

Multi-agent systems face critical routing challenges:

1. **Static Routing Bottleneck**: Traditional single-label routing (query → one agent) can't leverage multiple specialists with overlapping skills
2. **New Agent Integration**: Adding agents requires retraining routers and redesigning routing logic
3. **Capability Overlap Conflicts**: When multiple agents could handle a query, picking one wastes alternative perspectives
4. **Ambiguous Intent**: Many queries don't cleanly map to a single agent; forcing 1:1 assignment loses information

Current routers treat query-to-agent mapping as a classification problem, but real-world queries often benefit from multiple specialized perspectives.

## Solution

**TCAndon-Router (TCAR)** introduces **Multi-Candidate Reasoning-Aware Routing**:

1. **Reasoning-First Routing**: Generate natural-language reasoning chain explaining *why* an agent is appropriate before assigning queries
2. **Multi-Candidate Assignment**: Predict a *set* of candidate agents rather than single agent
3. **Lazy Agent Integration**: New agents register themselves; router adapts without retraining
4. **Response Refinement**: Aggregate responses from multiple agents with a dedicated Refining Agent producing coherent final answer

## When to Use

- **Enterprise Multi-Agent Systems**: Routing queries to teams of specialized agents (customer service, technical support)
- **Overlapping Expertise**: Domains where multiple agents have relevant but complementary knowledge
- **Scalable Agent Networks**: Systems that grow from 5 to 100+ agents over time
- **High-Confidence Requirements**: Critical decisions benefiting from multiple agent perspectives
- **Exploratory Agents**: Research systems where diverse viewpoints improve answer quality

## When NOT to Use

- For single-agent systems (router adds unnecessary overhead)
- In latency-critical applications (multi-agent routing adds response time)
- When computational budget for running multiple agents is unavailable
- For tasks with clear single-agent ownership (no overlap)

## Core Concepts

The framework operates on the principle that **reasoning improves routing**:

1. **Interpretable Decisions**: Before assigning agents, explain why in natural language
2. **Ensemble Decisions**: Use multiple perspectives to strengthen final answers
3. **Adaptive Architecture**: New agents self-integrate without retraining core router
4. **Conflict Resolution**: Disagreements between agents are opportunities for refinement

## Key Implementation Pattern

TCAR routing and aggregation pipeline:

```python
# Conceptual: reasoning-aware multi-agent routing
class TCAndonRouter:
    def route_and_aggregate(self, query):
        # Step 1: Generate routing reasoning
        reasoning = self.generate_reasoning(query)
        # "This query asks about technical implementations,
        #  suggesting DevOps and Backend specialists"

        # Step 2: Predict candidate agents
        candidates = self.predict_agents(query, reasoning)
        # candidates: [DevOpsAgent, BackendAgent, ArchitectureAgent]

        # Step 3: Run candidates in parallel
        responses = [agent.process(query) for agent in candidates]

        # Step 4: Refine into coherent answer
        final_answer = self.refining_agent.aggregate(
            query, reasoning, responses
        )
        return final_answer
```

Key mechanisms:
- Reasoning generation: explain routing decision in natural language
- Multi-candidate prediction: predict agent set, not single agent
- Parallel execution: run all candidates concurrently
- Refinement: dedicated agent merges candidate outputs

## Expected Outcomes

- **Improved Accuracy**: Multiple perspectives catch errors single agents miss
- **Scalability**: Add 10 new agents without retraining router
- **Transparency**: Reasoning traces show why agents were selected
- **Robustness**: Graceful handling of overlapping agent capabilities
- **Coverage**: Reduced ambiguity routing failures

## Limitations and Considerations

- Multi-agent execution adds computational cost vs. single-agent routing
- Response refinement quality depends on Refining Agent capability
- Scaling to 100+ agents requires efficient agent registry and concurrent execution
- Agent response disagreement can confuse refinement step

## Integration Pattern

For an enterprise support system:

1. **Query Arrives**: "How do I configure SSL certificates for my service?"
2. **Router Reasons**: "This spans DevOps (configuration), Security (SSL), and Architecture (service design)"
3. **Select Agents**: DevOpsAgent, SecurityAgent, ArchitectureAgent
4. **Parallel Execution**: All three process query independently
5. **Refine Response**: Merge recommendations into coherent implementation guide

This ensures customers get comprehensive answers leveraging all relevant expertise.

## Dynamic Agent Registration

New agents register via:

```python
router.register_agent(
    name="DatabaseOptimizationAgent",
    description="Optimizes database queries and indexing",
    capabilities=["performance", "indexing", "sql"]
)
```

Router adapts routing heuristics without retraining.

## Related Work Context

TCAR advances beyond static agent selection toward dynamic, reasoned multi-agent routing. Rather than treating agent assignment as a classification task, it recognizes routing as a reasoning problem benefiting from multiple specialist perspectives.
