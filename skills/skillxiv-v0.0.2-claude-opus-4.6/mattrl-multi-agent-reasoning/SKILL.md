---
name: mattrl-multi-agent-reasoning
title: "Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.09667"
keywords: [multi-agent, reasoning, test-time-RL, collaborative-deliberation, experience-retrieval]
description: "Enables LLM-based agent teams to improve reasoning accuracy at inference time through collaborative deliberation and structured experience retrieval, achieving 3-8% accuracy gains without expensive multi-agent training."
---

## Overview

Implement a multi-agent reasoning framework where specialist LLM agents collaborate to solve complex reasoning tasks (mathematical, medical, educational problems) without requiring resource-intensive training. The system uses test-time experience retrieval and consensus mechanisms to boost accuracy.

## When to Use

- When you have complex reasoning problems requiring multiple perspectives or domain expertise
- When you need to improve accuracy without retraining models
- For domains where specialist agents can provide meaningful cross-checking (medical, scientific, mathematical reasoning)
- When you want distribution-shift robustness across problem variations

## When NOT to Use

- For real-time applications requiring <100ms latency (multi-turn deliberation adds latency)
- For simple classification or single-turn tasks where multi-agent discussion adds overhead
- When agent diversity is limited (single-perspective problems)

## Key Technical Components

### Multi-Expert Agent Assembly

Create a team of specialist agents, each optimized for different reasoning approaches or problem domains.

```python
# Multi-expert team configuration
agents = [
    {"name": "math_specialist", "prompt_template": "solve this math problem step-by-step"},
    {"name": "logic_analyzer", "prompt_template": "analyze the logical structure"},
    {"name": "verification_agent", "prompt_template": "verify the solution correctness"},
]
```

### Test-Time Experience Retrieval

At inference, retrieve relevant solved problems from a memory pool and inject them into agent deliberation.

```python
# Experience pool construction at test-time
def retrieve_test_time_experiences(problem, pool, k=3):
    """Retrieve k most similar problems from the pool"""
    similarities = [compute_similarity(problem, exp) for exp in pool]
    return sorted(pool, key=lambda x: similarities[pool.index(x)], reverse=True)[:k]
```

### Turn-Level Credit Assignment

Assign rewards to individual interaction turns based on solution correctness and contribution quality.

```python
# Turn-level reward assignment
def compute_turn_rewards(dialogue_turns, final_solution):
    """Assign credit to each turn contributing to final solution"""
    turn_rewards = []
    for turn in dialogue_turns:
        # Credit based on contribution to reasoning chain
        credit = contribution_score(turn, final_solution)
        turn_rewards.append(credit)
    return turn_rewards
```

### Consensus Mechanism

Enable agents to converge on a final answer through agreement-based selection.

```python
# Consensus-based answer selection
def consensus_decision(agent_outputs):
    """Select answer with most agent agreement"""
    answer_votes = {}
    for output in agent_outputs:
        ans = extract_answer(output)
        answer_votes[ans] = answer_votes.get(ans, 0) + 1
    return max(answer_votes, key=answer_votes.get)
```

## Performance Characteristics

- Accuracy improvement: +3.67% over multi-agent baselines, +8.67% over single-agent
- Stable performance across distribution shifts
- Does not require updating model weights
- Inference cost scales with number of agents and dialogue turns

## References

- Multi-agent RL reduces to test-time experience injection
- Turn-level credit assignment enables focused agent learning
- Consensus mechanisms provide stability without explicit training
