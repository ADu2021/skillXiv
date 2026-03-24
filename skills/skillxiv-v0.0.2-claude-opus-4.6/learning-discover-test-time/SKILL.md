---
name: learning-discover-test-time
title: "Learning to Discover at Test Time"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.16175"
keywords: [test-time-learning, reinforcement-learning, problem-solving, reasoning, discovery]
description: "Apply reinforcement learning at test time to enable language models to continue adapting on specific problems, achieving state-of-the-art results across mathematics, GPU optimization, algorithms, and biology. Use when you need models to discover domain-specific problem-solving strategies without retraining."
---

# Learning to Discover at Test Time

This skill enables language models to apply reinforcement learning during testing on individual problems, allowing models to discover domain-specific reasoning strategies and solve novel problems more effectively.

## When to Use
- Mathematical problem solving where discovery of approaches helps
- Algorithm optimization tasks requiring exploration of solution space
- Biology/science problems needing iterative hypothesis testing
- GPU kernel optimization or other technical problem-solving
- Any domain where test-time adaptation improves performance

## When NOT to Use
- Simple inference tasks where latency is critical
- Domains where best solution is obvious without exploration
- Tasks requiring immediate response (test-time RL takes multiple steps)
- Scenarios with strict budget on computation per problem

## Key Concept
TTT-Discover (Test-Time RL for Discovery) allows models to continue learning during inference on individual test problems. Instead of immediately generating an answer, the model:

1. **Explores**: Generate candidate solutions or approaches
2. **Evaluates**: Check validity/correctness using domain-specific feedback
3. **Refines**: Use successful trajectories to guide next attempts
4. **Adapts**: Learn domain-specific strategies for that problem

This is like "continuing to train" on each test example, finding problem-specific solutions.

## Implementation Pattern

Apply reinforcement learning to problem-solving trajectories at test time:

```python
# Pseudocode for test-time discovery learning
class TestTimeDiscovery:
    def __init__(self, base_llm, reward_function):
        self.llm = base_llm
        self.reward_fn = reward_function  # Problem-specific validation

    def solve_with_discovery(self, problem, max_attempts=5):
        best_trajectory = None
        best_reward = float('-inf')

        for attempt in range(max_attempts):
            # Generate solution trajectory
            trajectory = self.llm.generate_trajectory(problem)

            # Evaluate with domain-specific feedback
            reward = self.reward_fn(trajectory, problem)

            if reward > best_reward:
                best_reward = reward
                best_trajectory = trajectory

                # Learn from this successful trajectory
                self.update_llm_with_trajectory(trajectory, reward)

        return best_trajectory

    def update_llm_with_trajectory(self, trajectory, reward):
        # In-context learning: condition next generation on
        # successful trajectories for similar problems
        self.llm.add_to_context(trajectory, reward)
```

The key: use problem-specific validation as feedback signals to guide exploration.

## Key Results
- State-of-the-art on mathematics (reaching advanced reasoning)
- Superior performance on GPU optimization problems
- Improved algorithm design and discovery
- Bio-domain problem solving advances
- Works without any fine-tuning, purely test-time RL

## Research Context
This paper demonstrates that language models have dormant problem-solving capabilities that emerge when given feedback and the freedom to explore during inference. The insight: "continuing to train" on each test problem, even for a few steps, unlocks significantly better reasoning.
