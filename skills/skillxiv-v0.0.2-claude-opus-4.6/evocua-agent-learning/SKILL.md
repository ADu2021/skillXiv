---
name: evocua-agent-learning
title: "EvoCUA: Evolving Computer Use Agents via Learning from Scalable Synthetic Experience"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.15876"
keywords: [agent-learning, synthetic-experience, computer-use, evolution, reinforcement-learning]
description: "Train autonomous agents to use computers by generating synthetic task experiences and iterating on them, achieving 56.7% success on OSWorld benchmarks through scalable experience-driven optimization. Use when you need agents that autonomously learn complex computer interaction patterns without manual task curation."
---

# EvoCUA: Evolving Computer Use Agents

This skill enables training computer-use agents that autonomously generate diverse synthetic tasks and learn through iterative cycles, significantly improving performance on real-world benchmarks.

## When to Use
- Building agents that interact with computer interfaces autonomously
- Creating systems that improve through self-generated experience loops
- Optimizing agent performance on complex desktop/web interaction tasks
- Implementing scalable agent training without extensive manual annotation

## When NOT to Use
- Tasks requiring immediate inference (training takes multiple iterations)
- Domains where synthetic experience won't transfer to real tasks
- Simple rule-based automation (overkill for simple workflows)
- Systems where you already have extensive human-annotated training data

## Key Concept
EvoCUA evolves agents through a feedback loop: generate synthetic tasks → train agent → evaluate performance → refine task generation. This removes the bottleneck of manual task creation.

The approach combines two stages:
1. **Task Evolution**: Autonomously generate diverse synthetic computer use tasks
2. **Agent Learning**: Iteratively train the agent on evolved tasks, improving real-world performance

## Implementation Pattern

Generate diverse synthetic tasks and create a feedback loop for agent training:

```python
# Pseudocode for EvoCUA training loop
class EvoCUA:
    def __init__(self, agent, task_generator):
        self.agent = agent
        self.task_generator = task_generator

    def train_iteration(self, num_synthetic_tasks=100):
        # Generate diverse synthetic tasks
        tasks = self.task_generator.create_tasks(num_synthetic_tasks)

        # Train agent on synthetic tasks
        for task in tasks:
            trajectory = self.agent.execute(task)
            self.agent.learn_from(trajectory)

        # Evaluate on real benchmarks (OSWorld)
        real_score = self.evaluate_on_real_tasks()
        return real_score
```

Track agent improvement across iterations. Real performance on OSWorld improved from prior approaches to 56.7% success rate through multiple rounds of synthetic task generation and refinement.

## Key Results
- Achieves 56.7% success on OSWorld benchmarks
- Outperforms previous open-source and closed-weight models
- Scales through synthetic experience generation rather than manual annotation
- Enables autonomous continuous improvement cycles

## Research Context
This paper addresses the challenge that manual task curation doesn't scale for training generally-capable computer-use agents. By automating task generation, agents can improve through iterative learning loops similar to human learning through varied experiences.
