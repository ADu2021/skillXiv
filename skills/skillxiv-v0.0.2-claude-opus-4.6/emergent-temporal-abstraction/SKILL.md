---
name: emergent-temporal-abstraction
title: "Emergent Temporal Abstractions in AR Models for Hierarchical RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.20605
keywords: [reinforcement-learning, hierarchy, autoregressive, temporal-abstraction, internal-rl]
description: "Discover hierarchical temporal abstractions within autoregressive models via internal RL, enabling efficient exploration of sparse-reward tasks. Metacontroller learns abstract action sequences modifying residual streams, switching gates enable quasi-binary patterns, and abstract-space RL achieves many orders-of-magnitude speedup over token-level learning."
---

## Overview

This technique enables autoregressive models to learn hierarchical behaviors through discovery of temporal abstractions, dramatically accelerating learning on sparse-reward tasks.

## Core Technique

**Internal RL with Discovered Abstractions:**

```python
class HierarchicalARModel:
    def __init__(self):
        self.base_ar_model = PretrainedAutoregressive()
        self.metacontroller = MetacontrollerPolicy()
        self.abstract_controllers = nn.ModuleList()

    def forward_hierarchical(self, state):
        # Metacontroller generates abstract action sequence
        abstract_actions = self.metacontroller.sample_actions(state)

        # Each abstract action is a sequence of residual stream modifications
        output = self.base_ar_model.initial_forward(state)

        for t, abstract_action in enumerate(abstract_actions):
            # Apply abstract action via residual stream modification
            controller_output = self.abstract_controllers[abstract_action](output)
            output = output + controller_output  # Residual addition
            
            # Check switching condition
            if self.should_switch(output, t):
                break  # Move to next abstract action

        return output
```

**Switching Gates and Temporal Patterns:**

```python
def switching_gate_mechanism(features, temperature=1.0):
    """
    Binary switching via gating, creating sparse temporal patterns.
    """
    gate_logits = nn.Linear(hidden_dim, 1)(features)
    gate_prob = sigmoid(gate_logits / temperature)
    
    # Gumbel-softmax for differentiable sampling
    gate_sample = gumbel_softmax(gate_prob)
    
    return gate_sample
```

**RL in Abstract Space:**

```python
def abstract_space_rl(model, env):
    for episode in range(num_episodes):
        state = env.reset()
        abstract_actions = model.metacontroller.sample_actions(state)
        
        # Accumulate token-level transitions
        tokens = []
        for abstract_action in abstract_actions:
            token_sequence = model.forward_with_controller(abstract_action, state)
            tokens.extend(token_sequence)
            state = env.step(tokens)
        
        # RL update on abstract actions, not tokens
        reward = env.get_reward()
        log_prob = model.metacontroller.log_prob(abstract_actions)
        loss = -reward * log_prob
        loss.backward()
```

## When to Use

Use when: Sparse-reward tasks, token-level RL too slow, hierarchical structure evident.

## References

- Metacontroller for abstract action selection
- Residual stream modification via controllers
- Switching gates for temporal abstraction
- RL in abstract action space
