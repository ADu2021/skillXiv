---
name: tarot-test-driven-curriculum-rl
title: "TAROT: Test-driven Curriculum RL for Code Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.15449"
keywords: [curriculum learning, reinforcement learning, code generation, test-driven development, model capability adaptation]
description: "Design capability-adaptive curricula for code generation by constructing per-problem test suites (basic, intermediate, complex, edge cases) and dynamically weighting training signals based on model capability rather than problem difficulty alone. Optimal curricula vary by model capacity: weaker models benefit from easy-to-hard progression while stronger models learn faster with complex-first strategies, enabling 1 dataset to serve multiple model scales efficiently."
---

# TAROT: Capability-Adaptive Test-Driven Curriculum for Code RL

Training reinforcement learning policies for code generation faces a fundamental trade-off: complex problems provide rich learning signal but can overwhelm models during early training, while simpler problems are easily solved but offer limited learning benefit. Traditional curriculum approaches sort problems by difficulty and train sequentially, assuming optimal progression is universal. This ignores a critical insight: optimal curriculum structure depends on model capability, not just problem count or parameter density.

The challenge is discovering which difficulty progression maximizes stable competency acquisition for a specific model at a specific stage of training. Applying the same curriculum to models of different scales or training phases often leads to either early saturation or optimization instability.

## Core Concept

TAROT implements capability-adaptive curriculum learning by decomposing each coding problem into an internal test hierarchy rather than treating each problem atomically. For each problem, the framework constructs a four-tier test suite (basic, intermediate, complex, edge-case) that mirrors test-driven development practices. During training, a capability-adaptive allocator determines what fraction of training compute each tier receives, and a reward weighting mechanism scales contributions based on the model's current capability level.

The key innovation is decoupling curriculum progression from raw problem difficulty by using two learnable components: a curriculum allocator (proportion of training focus per tier) and tier-specific reward weights (priority per tier based on observed capability).

## Architecture Overview

- **Test Suite Construction**: For each code problem, generate tests at 4 difficulty levels (basic validation through edge cases)
- **Curriculum Allocator**: Specifies target proportion of training trajectories per tier (e.g., 30% basic, 40% intermediate, 20% complex, 10% edge)
- **Capability Scorer**: Monitor model success rate on each tier to infer current capability level
- **Reward Weighting**: Dynamically scale per-tier rewards such that capability-appropriate successes contribute more to the training objective
- **Progression Controller**: Shift allocator weights over training epochs based on capability trajectory

## Implementation

Construct the test suite for a single problem by generating tests at each difficulty level, then implement the capability-adaptive weighting during reward computation:

```python
def construct_test_suite(problem_description, num_tests_per_tier=5):
    """
    Generate 4-tier test suite for a problem using code generation.
    Returns {tier_name: [test_cases]}
    """
    tiers = {
        'basic': 'Tests verifying basic functionality, no edge cases',
        'intermediate': 'Tests covering common use cases and input variations',
        'complex': 'Tests with complex logic, large inputs, or corner cases',
        'edge': 'Tests for boundary conditions, empty inputs, extreme values'
    }

    test_suite = {}
    for tier_name, tier_description in tiers.items():
        # LLM prompt: generate num_tests_per_tier tests of type tier_description
        test_suite[tier_name] = generate_tests(
            problem_description, tier_description, num_tests_per_tier
        )
    return test_suite
```

Compute per-tier success rates and reward weights dynamically during training:

```python
def compute_capability_weights(tier_results, alpha=0.1):
    """
    Scale rewards based on model capability per tier.
    tier_results: {tier: success_rate}
    Returns normalized weight vector
    """
    success_rates = torch.tensor([
        tier_results.get('basic', 0.0),
        tier_results.get('intermediate', 0.0),
        tier_results.get('complex', 0.0),
        tier_results.get('edge', 0.0)
    ])

    # Lower success = higher weight (focus on harder material)
    # Avoid dividing by zero; use epsilon
    weights = 1.0 / (success_rates + alpha)
    weights = weights / weights.sum()  # normalize

    return weights

def apply_curriculum_reward(trajectory_tier, base_reward, weights):
    """
    Scale reward by tier-specific capability weight.
    trajectory_tier: which tier this trajectory tested
    base_reward: scalar reward before weighting
    weights: weight vector from compute_capability_weights
    """
    tier_idx = {'basic': 0, 'intermediate': 1, 'complex': 2, 'edge': 3}[trajectory_tier]
    scaled_reward = base_reward * weights[tier_idx]
    return scaled_reward
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Test count per tier | 5 | Use 3-10; balance coverage with generation cost |
| Curriculum epoch | 100 steps | Evaluate capability and update allocator weights every 100 training steps |
| Alpha (capability smoothing) | 0.1 | Larger alpha → smoother weights; smaller → sharper tier focus |
| Basic-to-complex ratio | 30:10 | Weak models: 40:5; strong models: 10:30 |

**When to use**: Apply when training LLMs for code generation with variable model scales or when training epochs are long enough (>50k steps) to benefit from adaptive curriculum structure.

**When not to use**: Skip for single-problem fine-tuning or when computational budget is extremely limited (test generation overhead is non-negligible).

**Common pitfalls**:
- Generating tests with inconsistent quality across tiers; use the same generation pipeline with only difficulty prompts varying
- Forgetting to normalize weights; unscaled weights can cause extreme reward magnification
- Setting tier distribution statically; monitor success rates and adjust allocator weights actively

## Reference

TAROT achieves stable optimization and more efficient competency acquisition by allowing models to focus on capability-appropriate learning signals. Empirically, weak models (0.6B parameters) learn best with easy-to-hard progression while strong models (7B+) accelerate training with complex-first strategies, all on the same dataset with only weighting adjusted per model.
