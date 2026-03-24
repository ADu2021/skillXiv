---
name: lsrif-logic-structured-rl
title: "LSRIF: Logic-Structured Reinforcement Learning for Instruction Following"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.06431"
keywords: [instruction-following, logic-constraints, structured-rewards, RL-training, semantic-understanding]
description: "Improves instruction-following by using differentiated reward mechanisms that recognize logical dependencies (sequential, conditional, parallel) in instructions, enabling better model reasoning about task structure."
---

## Overview

Train agents to follow complex instructions that contain logical structures (sequential steps, conditional branches, parallel tasks). Rather than treating all instruction steps equally, apply structure-aware reward mechanisms that recognize when instructions have dependencies, enabling agents to learn the underlying logic.

## When to Use

- For instruction-following tasks with complex logical structures
- When instructions have sequential, conditional, or parallel dependencies
- For improving out-of-domain generalization in instruction understanding
- When you need agents to reason about task structure, not just surface form

## When NOT to Use

- For simple, single-action instructions without dependencies
- For domains where instruction structure is always flat/linear
- When you don't have labeled logical structure annotations
- For real-time applications where training time is critical

## Key Technical Components

### Logical Structure Annotation

Classify each instruction's logical type to enable targeted reward design.

```python
# Instruction structure classification
class InstructionStructure:
    SEQUENTIAL = "sequential"    # Steps must occur in order
    PARALLEL = "parallel"        # Steps independent, order irrelevant
    CONDITIONAL = "conditional"  # Branch based on conditions

    @staticmethod
    def analyze(instruction):
        """Determine logical structure of instruction"""
        if has_order_dependencies(instruction):
            return InstructionStructure.SEQUENTIAL
        elif has_condition_branches(instruction):
            return InstructionStructure.CONDITIONAL
        elif all_steps_independent(instruction):
            return InstructionStructure.PARALLEL

def detect_structure(text):
    """Identify logical operators and structure"""
    sequential_markers = ["then", "next", "after", "step"]
    parallel_markers = ["and", "also", "meanwhile", "concurrently"]
    conditional_markers = ["if", "unless", "when", "otherwise"]

    if any(m in text for m in sequential_markers):
        return "sequential"
    elif any(m in text for m in parallel_markers):
        return "parallel"
    elif any(m in text for m in conditional_markers):
        return "conditional"
```

### Structure-Aware Reward Assignment

Implement differentiated reward mechanisms for each structure type.

```python
# Reward computation for different structures
class StructuredReward:
    def compute(self, instruction_structure, action_sequence, ground_truth):
        """Compute rewards based on logical structure"""
        if instruction_structure == "sequential":
            return self.sequential_reward(action_sequence, ground_truth)
        elif instruction_structure == "parallel":
            return self.parallel_reward(action_sequence, ground_truth)
        elif instruction_structure == "conditional":
            return self.conditional_reward(action_sequence, ground_truth)

    def sequential_reward(self, actions, ground_truth):
        """Penalize early failures in sequence"""
        reward = 0.0
        for i, (action, correct_action) in enumerate(zip(actions, ground_truth)):
            if action == correct_action:
                reward += 1.0
            else:
                # Failure at step i affects all subsequent steps
                penalty = (len(ground_truth) - i) * 0.1
                reward -= penalty
                break  # Stop accumulating after first error
        return reward / len(ground_truth)

    def parallel_reward(self, actions, ground_truth):
        """Average reward across independent tasks"""
        correct_count = sum(
            1 for a, g in zip(actions, ground_truth) if a == g
        )
        return correct_count / len(ground_truth)

    def conditional_reward(self, actions, ground_truth):
        """Reward based on correct branch selection"""
        reward = 0.0
        for i, (action, correct_action) in enumerate(zip(actions, ground_truth)):
            if i == 0:  # First step: condition evaluation
                if action == correct_action:
                    reward += 1.0  # Correct branch selection
            else:
                # Subsequent steps in chosen branch
                if action == correct_action:
                    reward += 0.5
        return reward / max(1.0, len(ground_truth) * 0.75)
```

### LSRInstruct Dataset Construction

Build training data with explicit logical structure annotations.

```python
# Structured instruction dataset
class StructuredInstruction:
    def __init__(self, text, structure_type, steps, constraints=None):
        self.text = text
        self.structure_type = structure_type
        self.steps = steps
        self.constraints = constraints or {}

# Example: Sequential instruction
sequential_ex = StructuredInstruction(
    text="First gather ingredients, then mix them, finally bake.",
    structure_type="sequential",
    steps=[
        {"action": "gather", "object": "ingredients"},
        {"action": "mix", "object": "ingredients"},
        {"action": "bake"}
    ],
    constraints={"order_required": True}
)

# Example: Conditional instruction
conditional_ex = StructuredInstruction(
    text="If temperature is high, cool the mixture. Otherwise, proceed directly.",
    structure_type="conditional",
    steps=[],
    constraints={
        "condition": "temperature > threshold",
        "branches": {
            "true": [{"action": "cool"}],
            "false": [{"action": "proceed"}]
        }
    }
)
```

### Constraint Token Attention Analysis

Monitor which tokens the model attends to for logical operators.

```python
# Attention pattern analysis
def analyze_constraint_attention(model_outputs, instruction_text):
    """Check if model focuses on logical constraint tokens"""
    constraint_tokens = extract_logical_tokens(instruction_text)
    attention_weights = model_outputs["attention"]

    focus_on_constraints = 0
    for token_idx in constraint_tokens:
        if attention_weights[token_idx] > THRESHOLD:
            focus_on_constraints += 1

    return {
        "constraint_focus": focus_on_constraints / len(constraint_tokens),
        "improvement_signal": focus_on_constraints > len(constraint_tokens) * 0.5
    }
```

### Training Loop Integration

Incorporate structured rewards into standard RL training.

```python
# Training with structure-aware rewards
def train_with_structured_rewards(model, dataset, num_epochs=10):
    """Train agent using logic-aware reward signals"""
    for epoch in range(num_epochs):
        for instruction in dataset:
            # Detect structure
            structure = InstructionStructure.analyze(instruction.text)

            # Generate action
            actions = model.generate_actions(instruction.text)

            # Compute structure-aware reward
            reward = StructuredReward().compute(
                structure,
                actions,
                instruction.steps
            )

            # RL update
            loss = compute_policy_loss(actions, reward)
            model.backward(loss)
            model.update()
```

## Performance Characteristics

- In-domain improvement: +2-4% on structured instruction-following
- Out-of-domain generalization: Stronger generalization to new structures
- Model attention: Sharpens focus on logical constraint tokens
- Training efficiency: Learns structure faster with fewer examples

## Recommendations

- Annotate your instruction dataset with structure types upfront
- Use curriculum learning: start with sequential, add parallel, then conditional
- Monitor attention patterns on logical operators as a training signal
- Combine with Chain-of-Thought prompting for better structured reasoning

## References

- Logical structure in instructions creates non-uniform reward landscapes
- Structure-aware rewards align gradient signals with task semantics
- Explicit constraint learning improves both in-domain and out-of-domain performance
