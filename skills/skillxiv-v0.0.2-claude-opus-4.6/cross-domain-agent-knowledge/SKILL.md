---
name: cross-domain-agent-knowledge
title: "Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06229"
keywords: [Agent Memory, Knowledge Transfer, Cross-Domain Learning, Agent Frameworks, Experience Sharing]
description: "Create a universal memory infrastructure enabling agents across different frameworks to share experience trajectories without retraining. Improve agent performance by retrieving workflows from related domains and applying diagnostic fixes."
---

# Agent KB: Universal Knowledge Infrastructure for Heterogeneous Agents

Agent frameworks like smolagents and OpenHands operate in isolation, causing each agent to rediscover solutions and repeat mistakes independently. Agent KB establishes a shared knowledge repository that allows agents across heterogeneous frameworks to benefit from experience accumulated in other domains. By implementing hybrid retrieval (for planning seeds and feedback fixes) with a safeguard mechanism that prevents harmful knowledge transfer, the system achieves substantial improvements without requiring framework modifications or model retraining.

The core problem is knowledge silos: agents accumulate valuable trajectories but cannot share them across framework boundaries, creating duplicated effort and preventing collective intelligence from emerging.

## Core Concept

Agent KB operates on three key principles:

1. **Universal memory interface**: Trajectories are compiled into a standardized knowledge base accessible via lightweight APIs, agnostic to underlying agent framework
2. **Hybrid retrieval strategy**: Planning stage retrieves seed workflows from other domains; feedback stage applies diagnostic corrections based on failures
3. **Disagreement gating**: Before applying retrieved knowledge, verify it aligns with the current task and agent's reasoning to prevent negative transfer

This architecture enables seamless knowledge sharing without requiring agents to be aware of framework differences.

## Architecture Overview

- **Knowledge aggregation layer**: Collects trajectories from multiple frameworks and standardizes format
- **Planning retriever**: Seed-retrieves high-level workflows relevant to current problem
- **Feedback fixer**: Extracts diagnostic patterns from failed trajectories to suggest corrections
- **Disagreement gate**: Validator that ensures retrieved knowledge enhances rather than disrupts reasoning
- **Lightweight APIs**: Framework-agnostic interfaces for knowledge access
- **Multi-framework support**: Compatible with smolagents, OpenHands, LangChain, and custom agents

## Implementation

Set up the knowledge base infrastructure to aggregate trajectories from multiple sources:

```python
from agent_kb.knowledge_base import KnowledgeBase
from agent_kb.standardizers import TrajectoryStandardizer

# Initialize universal knowledge repository
kb = KnowledgeBase(storage="vector_db")

standardizer = TrajectoryStandardizer()

# Ingest trajectories from different agent frameworks
trajectories_smolagents = load_smolagents_trajectories("logs/smolagents/")
trajectories_openhands = load_openhands_trajectories("logs/openhands/")

for trajectory in trajectories_smolagents + trajectories_openhands:
    # Standardize to common format
    standardized = standardizer.standardize(
        trajectory=trajectory,
        framework="auto-detect"  # Auto-detects format
    )

    # Index for efficient retrieval
    kb.add_trajectory(
        trajectory=standardized,
        metadata={
            "task": trajectory.get("task_name"),
            "success": trajectory.get("success"),
            "domain": trajectory.get("domain"),
            "framework": trajectory.get("framework")
        }
    )

print(f"KB indexed {len(kb)} trajectories across {len(set(kb.get_domains()))} domains")
```

Implement planning-stage retrieval that seeds new agents with relevant workflows:

```python
from agent_kb.retrievers import PlanningRetriever

retriever = PlanningRetriever(kb=kb)

def solve_with_planning_seeds(task_description, agent):
    """Augment agent planning with retrieved seed workflows."""

    # Retrieve high-level workflows from successful trajectories
    seed_workflows = retriever.retrieve_planning_seeds(
        query=task_description,
        k=3,  # Top 3 relevant workflows
        filter_success=True  # Only successful trajectories
    )

    # Extract workflow skeleton (task sequence, not details)
    workflows = [
        {
            "task_sequence": extract_task_steps(traj),
            "estimated_success_rate": traj["metadata"]["success_rate"],
            "domain": traj["metadata"]["domain"]
        }
        for traj in seed_workflows
    ]

    # Provide as planning context to agent
    agent.set_planning_seeds(workflows)

    # Agent uses seeds to guide initial action sequence
    result = agent.solve(task_description)

    return result
```

Implement feedback-stage retrieval that suggests fixes after agent failures:

```python
from agent_kb.retrievers import FeedbackRetriever
from agent_kb.diagnostics import DiagnosticExtractor

feedback_retriever = FeedbackRetriever(kb=kb)
extractor = DiagnosticExtractor()

def recover_from_failure(failed_trajectory, agent, kb):
    """Use KB diagnostics to suggest recovery actions."""

    # Identify failure pattern in current trajectory
    failure_pattern = extractor.extract_failure_pattern(
        trajectory=failed_trajectory,
        agent_state=agent.current_state
    )

    # Find similar failures in KB and their resolutions
    diagnostic_fixes = feedback_retriever.retrieve_fixes(
        failure_pattern=failure_pattern,
        k=5  # Top 5 similar failure modes
    )

    # Rank fixes by applicability to current context
    applicable_fixes = [
        fix for fix in diagnostic_fixes
        if is_applicable(fix, agent.current_state)
    ]

    if not applicable_fixes:
        return None  # No relevant fixes found

    # Format as suggestions to agent
    recovery_suggestions = [
        {
            "action": fix["recovery_action"],
            "reasoning": fix["why_worked"],
            "confidence": fix["applicability_score"]
        }
        for fix in applicable_fixes[:3]
    ]

    return recovery_suggestions
```

Implement the disagreement gate to prevent negative transfer:

```python
from agent_kb.safety import DisagreementGate

gate = DisagreementGate()

def apply_retrieved_knowledge(retrieved_knowledge, agent_state, current_task):
    """Apply retrieved knowledge only if it aligns with current reasoning."""

    # Check for disagreement between retrieved knowledge and agent state
    disagreement_score = gate.evaluate(
        retrieved_knowledge=retrieved_knowledge,
        agent_reasoning=agent_state.get("reasoning"),
        task=current_task
    )

    # If disagreement is high, suppress the retrieved knowledge
    if disagreement_score > gate.threshold:
        print(f"Suppressing retrieved knowledge (disagreement: {disagreement_score:.2f})")
        return None  # Don't apply conflicting knowledge

    # Otherwise, integrate into agent planning
    enhanced_plan = integrate_knowledge(
        agent_plan=agent_state.get("current_plan"),
        retrieved_knowledge=retrieved_knowledge
    )

    return enhanced_plan
```

## Practical Guidance

### When to Use Agent KB

Use Agent KB when:
- Operating multiple agents across different frameworks
- Solving tasks similar to previously solved problems
- Feedback and failure recovery are valued
- Domains have overlapping problem structures
- You want to measure and reduce redundant work across agents

### When NOT to Use

Avoid Agent KB for:
- Completely novel domains with no prior trajectories
- Highly specialized agents requiring domain-specific knowledge
- Tasks where failure modes are unique and unsafe to transfer
- Real-time systems where retrieval latency is critical
- Domains where negative transfer is more costly than no transfer

### Trajectory Storage Format

Standard format for all trajectories (framework-agnostic):

```python
{
    "task": "Book a flight from NYC to LA",
    "domain": "travel",
    "framework": "smolagents",
    "success": True,
    "steps": [
        {
            "action": "search_flights",
            "input": {"origin": "NYC", "destination": "LA", "date": "2024-03-25"},
            "output": "Found 12 flights, cheapest $89",
            "tool": "FlightSearch"
        },
        {
            "action": "filter_results",
            "input": {"max_price": 150, "preferred_airline": "Delta"},
            "output": "3 flights match criteria"
        }
    ],
    "metadata": {
        "total_steps": 2,
        "tools_used": ["FlightSearch", "Filter"],
        "success_rate": 0.87,
        "timestamp": "2024-03-25T10:00:00Z"
    }
}
```

### Performance Improvements

The paper demonstrates substantial gains by framework:

| Framework | Baseline | With KB | Improvement |
|-----------|----------|---------|-------------|
| smolagents | 55.2% | 73.9% | +18.7 pp |
| OpenHands | 24.3% | 28.3% | +4.0 pp |
| Custom agent | Variable | +12-15 pp | Domain dependent |

Larger improvements occur when source and target domains are closely related.

### Retrieval Strategy Selection

| Stage | Strategy | When to Use |
|-------|----------|------------|
| Planning | Seed workflows | Early in task, need high-level guidance |
| Feedback | Diagnostic fixes | After agent failure, need recovery suggestions |
| Adaptation | Full trajectory | When source task is very similar to target |

### Common Pitfalls

1. **Over-trusting retrieved knowledge**: High disagreement scores indicate conflicting reasoning. Don't override agent skepticism.
2. **Cross-domain contamination**: Tasks in different domains have different constraints. Always validate applicability.
3. **Forgetting KB maintenance**: Trajectories accumulate noise over time. Periodically filter low-quality or obsolete trajectories.
4. **Ignoring retrieval latency**: Vector DB lookups add overhead. Cache frequently used trajectories.
5. **Missing knowledge standardization**: Trajectories from different frameworks may have different semantics. Standardize carefully.

### Quality Assurance

- [ ] Trajectories are standardized to consistent format
- [ ] Metadata includes task domain and success/failure labels
- [ ] Disagreement gate threshold is calibrated on validation set
- [ ] Retrieved knowledge is validated before agent use
- [ ] KB is periodically pruned of low-quality or contradictory trajectories
- [ ] Cross-domain transfers are logged for analysis

### Scaling Considerations

For KB scaling beyond 100k trajectories:
- Use vector embeddings for efficient semantic retrieval
- Implement hierarchical clustering by domain
- Add caching layer for frequently accessed trajectories
- Monitor retrieval latency and batch similar queries
- Prune trajectories older than 6 months (domain-dependent)

## Reference

"Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving" - [arXiv:2507.06229](https://arxiv.org/abs/2507.06229)
