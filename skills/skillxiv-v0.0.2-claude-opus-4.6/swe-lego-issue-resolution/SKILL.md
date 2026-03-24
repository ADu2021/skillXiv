---
name: swe-lego-issue-resolution
title: "SWE-Lego: Pushing the Limits of Supervised Fine-tuning for Software Issue Resolving"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.01426"
keywords: [Software Engineering, Fine-tuning, Issue Resolution, Test-time Scaling, LLM Agents]
description: "Achieve state-of-the-art software issue resolution through lightweight supervised fine-tuning with structured datasets and curriculum learning, plus test-time scaling—reaching 52.6% on SWE-Bench Verified and 58.8% with test-time strategies, outperforming complex multi-stage training."
---

## Overview

SWE-Lego demonstrates that sophisticated software engineering agent capabilities can be achieved through a carefully designed lightweight approach: supervised fine-tuning (SFT) only, without requiring complex multi-stage training pipelines (mid-training, SFT, RL combinations).

**Core Insight:** State-of-the-art software issue resolution doesn't require elaborate training paradigms. Instead, three key components—high-quality data, refined SFT procedure, and test-time scaling—deliver competitive performance at fraction of typical complexity.

## Three Building Blocks

### Block 1: SWE-Lego Dataset (32k + 18k trajectories)

A carefully curated dataset combining real and synthetic data for both quality and quantity:

**Composition:**
- **32,000 task instances** - High-quality, diverse software engineering tasks
- **18,000 validated trajectories** - Correct solution paths for training
- **Real data** - Authentic GitHub issues and PRs
- **Synthetic data** - Complementary examples filling capability gaps

**Quality Control:**
- Trajectory validation ensuring correctness
- Diversity across programming languages and issue types
- Difficulty stratification for curriculum learning
- Removal of low-quality or ambiguous examples

**Key Advantage:** Balanced coverage of both common patterns and edge cases through synthetic-real combination.

### Block 2: Refined SFT Procedure

Enhanced training methodology improving action quality and overall performance:

**Error Masking:**
Filter trajectories containing invalid tool calls or impossible actions. Prevents model learning degenerate patterns.

```python
def mask_errors_in_trajectory(trajectory):
    """Filter out invalid actions from training trajectory."""
    valid_actions = []
    for step in trajectory:
        if is_valid_action(step.action, step.context):
            valid_actions.append(step)
        else:
            # Skip invalid action; continue from next valid state
            pass
    return valid_actions
```

**Difficulty-Based Curriculum:**
Train on progressively harder examples to improve convergence and final performance:

```python
# Stage 1: Simple fixes (syntax errors, type issues)
curriculum_stage_1 = [
    t for t in dataset
    if task_difficulty(t.issue) <= "easy"
]

# Stage 2: Medium complexity (logic bugs, simple refactoring)
curriculum_stage_2 = [
    t for t in dataset
    if "easy" < task_difficulty(t.issue) <= "medium"
]

# Stage 3: Complex (architectural changes, deep reasoning)
curriculum_stage_3 = [
    t for t in dataset
    if task_difficulty(t.issue) > "medium"
]

train_sft(curriculum_stage_1)  # Epochs 1-2
train_sft(curriculum_stage_2)  # Epochs 3-4
train_sft(curriculum_stage_3)  # Epochs 5+
```

**Results from Blocks 1+2:**
- **SWE-Lego-Qwen3-8B:** 42.2% on SWE-Bench Verified
- **SWE-Lego-Qwen3-32B:** 52.6% on SWE-Bench Verified
- State-of-the-art among open-source models

### Block 3: Test-Time Scaling

Improve performance at inference through verification and multiple attempts:

**Verification-Based Scaling:**
- Generate multiple solution candidates
- Use trained verifier to assess solution quality
- Select highest-confidence solution

```python
def test_time_scale(problem, verifier, num_attempts=16):
    """Generate multiple solutions and select best."""
    candidates = []

    for _ in range(num_attempts):
        solution = model.generate_solution(problem)
        confidence = verifier.score(problem, solution)
        candidates.append((solution, confidence))

    # Return highest-confidence solution
    best_solution, _ = max(candidates, key=lambda x: x[1])
    return best_solution
```

**Performance Gains with Test-Time Scaling:**
- SWE-Lego-Qwen3-8B: 42.2% → 49.6% (TTS@16)
- SWE-Lego-Qwen3-32B: 52.6% → 58.8% (TTS@16)
- Linear scaling up to 16 attempts; diminishing returns beyond

**Computational Trade-off:**
- 16x increase in inference compute for 7-9% accuracy gain
- Practical for offline analysis; less suitable for real-time deployment

## Benchmark Performance

**SWE-Bench Verified** (rigorous subset with validated fixes):
- SWE-Lego-Qwen3-8B: 42.2% baseline, 49.6% with TTS@16
- SWE-Lego-Qwen3-32B: 52.6% baseline, 58.8% with TTS@16
- Competitive with much larger closed-source models

**Comparison to Complex Pipelines:**
- ReTool (SFT + RL): $10,000 training cost
- SWE-Lego (SFT only): <$1,000 training cost
- SWE-Lego achieves comparable or superior performance

## When to Use SWE-Lego Approach

**Use when:**
- Fine-tuning models for software engineering tasks
- Building issue-resolution agents with limited compute budgets
- Need interpretable training pipeline (SFT only, no RL complexity)
- Working with open-source models (Qwen, Llama, etc.)
- Test-time compute is available (multiple solution generation acceptable)

**When NOT to use:**
- Real-time issue resolution requiring single-pass inference
- Extremely limited fine-tuning data (<1,000 examples)
- Scenarios where interpretability of training process isn't important
- Proprietary models where fine-tuning isn't accessible

## Implementation Lessons

**Dataset Design:**
- Mix real and synthetic data strategically
- Validate trajectories before inclusion
- Stratify by difficulty for curriculum learning
- Remove ambiguous or contradictory examples

**Training Procedure:**
- Filter invalid actions from training trajectories
- Use curriculum learning for complex tasks
- Monitor both intermediate action quality and final correctness
- Iterate on data quality more than model size

**Inference Strategy:**
- Trained verifier crucial for test-time scaling effectiveness
- 16 attempts provides good accuracy/compute tradeoff
- Diminishing returns beyond; batch processing is efficient

## Related Approaches

**SFT-Only Paradigm:**
- Minimal training complexity vs. SFT + RL combinations
- Easier to reproduce and debug
- More interpretable training dynamics

**Multi-Stage Training:**
- More sophisticated but harder to optimize
- Higher computational cost
- Potential for instability (entropy explosion, etc.)

## Code Availability

Project website: https://github.com/SWE-Lego/swe-lego

**Public Resources:**
- SWE-Lego dataset (32k instances, 18k trajectories)
- Fine-tuned models (8B and 32B variants)
- Training scripts and curriculum implementation
- Verifier models for test-time scaling

## References

- SWE-Lego-Qwen3-32B: 52.6% on SWE-Bench Verified (open-source SOTA)
- Test-time scaling reaches 58.8% (TTS@16)
- Lightweight SFT-only approach outperforms complex multi-stage pipelines
- 10x lower training cost than competing methods
