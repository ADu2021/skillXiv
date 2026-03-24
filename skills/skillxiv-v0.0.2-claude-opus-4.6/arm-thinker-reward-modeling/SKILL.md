---
name: arm-thinker-reward-modeling
title: "ARM-Thinker: Agentic Reward Models with Tool-Grounded Multimodal Verification"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.05111
keywords: [reward-modeling, multimodal-agents, tool-use, verification, evidence-grounding]
description: "Agentic reward model framework enabling active tool invocation (cropping, retrieval, validation) to ground judgments in verifiable evidence, using multi-stage GRPO with adaptive reward shaping for systematic evidence-based evaluation."
---

## Summary

ARM-Thinker introduces an agentic reward modeling framework that enables multimodal reward models to actively invoke external tools—image cropping, document retrieval, instruction validators—to ground judgments in verifiable evidence. Rather than passively scoring responses, the framework follows a "think-act-observe" loop with multi-stage reinforcement learning for adaptive tool-calling optimization.

## Core Technique

**Think-Act-Observe Loop:** Reward models actively reason about what evidence is needed:
1. **Think:** Plan reasoning steps and identify required verification
2. **Act:** Call tools (crop image region, retrieve relevant document section)
3. **Observe:** Process tool outputs and base scores on verified facts

**Multi-Stage GRPO:** Two-phase reinforcement learning:
- **Phase 1:** Optimize tool-calling decisions via GRPO
- **Phase 2:** Refine final accuracy scoring on improved trajectories

**Adaptive Reward Shaping:** Progressively weight tool-calling versus accuracy optimization:
```
reward = λ_tool * tool_call_quality + λ_acc * accuracy_bonus
```

## Implementation

**Tool invocation architecture:**
```python
class ToolCallAgent:
    def __init__(self):
        self.planner = llm  # Plans tool calls
        self.tools = {
            'crop_image': crop_image_tool,
            'retrieve_docs': doc_retrieval_tool,
            'validate_instruction': instruction_validator
        }

    def think_act_observe(self, response, evidence):
        # Plan what to verify
        plan = self.planner(f"What should I verify about {response}?")

        # Execute tool calls
        observations = []
        for tool_call in parse_tool_calls(plan):
            tool_name, args = tool_call
            result = self.tools[tool_name](**args)
            observations.append((tool_name, result))

        return observations
```

**Reward computation with evidence:**
```python
def compute_evidence_reward(response, observations, ground_truth):
    # Score based on verified facts
    accuracy = 0.0

    for tool_name, observation in observations:
        if tool_name == 'crop_image':
            # Check if crop correctly identifies object
            matches_gt = compare_visual_features(observation, ground_truth_region)
            accuracy += 0.3 * matches_gt

        elif tool_name == 'retrieve_docs':
            # Check if retrieval is relevant
            relevance = compute_relevance(observation, response)
            accuracy += 0.4 * relevance

        elif tool_name == 'validate_instruction':
            # Check if instruction follows constraints
            valid = instruction_is_valid(observation)
            accuracy += 0.3 * valid

    return accuracy
```

**Multi-stage GRPO:**
```python
# Stage 1: Optimize tool selection
for step in range(num_steps_phase1):
    trajectory = agent.rollout(problem)
    tool_quality = evaluate_tool_calls(trajectory)
    agent.update(trajectory, reward=tool_quality)

# Stage 2: Refine accuracy
for step in range(num_steps_phase2):
    trajectory = agent.rollout(problem)
    accuracy = evaluate_response_accuracy(trajectory)
    agent.update(trajectory, reward=accuracy)
```

## When to Use

- Multimodal evaluation requiring fine-grained verification
- Scenarios where reward models need to justify decisions with evidence
- Applications combining visual, textual, and document analysis
- Tasks requiring systematic grounding in verifiable facts

## When NOT to Use

- Simple single-modality scoring without verification needs
- Real-time evaluation where tool latency is prohibitive
- Scenarios without access to multiple verification tools
- Applications where passive reward signals suffice

## Key References

- Reward modeling and preference learning
- Tool-use and agent planning
- Multimodal evidence grounding
- Multi-stage reinforcement learning
