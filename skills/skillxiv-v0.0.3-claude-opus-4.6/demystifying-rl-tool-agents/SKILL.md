---
name: demystifying-rl-tool-agents
title: "Demystifying RL for Long-Horizon Tool-Using Agents"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.21972"
keywords: [Reinforcement Learning, Long-Horizon Agents, Tool Use, Reward Design, GRPO]
description: "Comprehensive recipe for RL-training tool-using agents spanning reward design, data synthesis, model scaling, and algorithm selection. Seven ranked findings: scale-dependent rewards (curriculum for 1.5B–3B; dense for 7B), semi-sparse 'Macro' rewards balance specialization/transfer, 1K-sample sweet spot with 4:3:3 difficulty mix. Achieves SOTA on TravelPlanner with smaller models than leading proprietary systems."
---

## Ranked Findings

### 1. Scale-Dependent Reward Design (Critical)
Different model scales require fundamentally different reward structures.

**For smaller models (1.5B–3B)**:
- Curriculum rewards transition from dense to sparse signals
- Dense-only rewards fail to provide learning signal
- Sparse-only rewards cause catastrophic exploration collapse

**For larger models (7B)**:
- Simple dense sum rewards outperform curriculum approaches
- Additional complexity adds overhead without benefit
- Larger models handle dense reward surfaces more robustly

### 2. Overly Dense Rewards Create Alignment Tax
Task-specific dense rewards maximize in-domain performance but significantly degrade out-of-domain generalization.

**Trade-off spectrum**:
- **Full dense**: Best in-domain performance; poorest transfer
- **Semi-sparse "Macro"**: Balanced performance and transferability (recommended)
- **Sparse**: Best transfer; weak in-domain learning signal

**Finding**: Macro rewards (task-level rather than step-level) offer optimal generalization-specialization tradeoff.

### 3. Consistent Scaling Benefits
Model capacity improvements yield substantial gains across all conditions.

**1.5B → 7B transition**: Substantial improvements in success rates across all reward signals—no reward design fully compensates for model capacity limitations.

### 4. Data Sweet Spot at ~1K Samples (Critical)
Approximately 1,000 training examples with balanced difficulty provide optimal trade-offs.

**Behavior**:
- **<1K**: Insufficient signal; underfitting
- **~1K**: Peak generalization; Goldilocks zone
- **>2K**: Over-optimization on training distribution; degraded generalization despite marginal in-domain gains

### 5. Balanced Difficulty Prevents Reward Sparsity
Mixed easy:medium:hard ratios (4:3:3) maintain sufficient reward signals while teaching complex constraint satisfaction.

**Alternative distributions**:
- **Homogeneous easy**: Too much positive feedback; agents never learn hard constraints
- **Homogeneous hard**: Catastrophic collapse—sparse reward signal
- **4:3:3 mix**: Optimal—sustains learning throughout training

### 6. Exploration Necessity Inversely Correlates with Model Capability
Sophisticated exploration algorithms (ARPO, DAPO) help smaller models but add computational overhead without benefit for larger models.

**Recommendation**:
- **1.5B–3B**: Use DAPO/ARPO for better exploration
- **7B+**: Standard GRPO sufficient; skip advanced exploration

### 7. Environmental Stability Matters
Agents tolerate up to 5% tool failure rates but experience noticeable degradation above 10%.

**Impact**: High environment instability hinders reward signal reliability, overriding other optimizations.

## Implementation STAR Pipeline

### Stage 1: Data Synthesis
Generate feasible queries with controlled difficulty via back-translation and validate in sandbox.

```python
def synthesize_training_data(base_queries, target_count=1000, difficulty_split=[0.4, 0.3, 0.3]):
    """
    Back-translate: generate diverse queries by paraphrasing and difficulty-aware sampling.
    Validate queries are solvable before inclusion.
    """
    easy = []
    medium = []
    hard = []

    for query in base_queries:
        # Paraphrase for diversity
        variants = back_translate(query, n_variants=5)

        for variant in variants:
            # Check solvability in sandbox
            trajectory = execute_in_sandbox(variant, tool_budget=60)

            if trajectory.success:
                difficulty = estimate_difficulty(variant, trajectory)
                if difficulty == "easy":
                    easy.append((variant, trajectory))
                elif difficulty == "medium":
                    medium.append((variant, trajectory))
                else:
                    hard.append((variant, trajectory))

    # Balance to 4:3:3 ratio targeting total=1K
    n_easy = int(target_count * 0.4)
    n_medium = int(target_count * 0.3)
    n_hard = int(target_count * 0.3)

    return easy[:n_easy] + medium[:n_medium] + hard[:n_hard]
```

### Stage 2: Supervised Fine-Tuning
Filter trajectories for success; use rejection sampling for quality control.

```python
def supervised_finetune(training_data, model, epochs=5):
    """
    SFT on successful trajectories only; rejection sampling for quality.
    Filter teacher trajectories before training.
    """
    successful_trajectories = [
        (query, traj) for query, traj in training_data
        if traj.success
    ]

    # Rejection sampling: accept top-quality trajectories
    filtered = sorted(
        successful_trajectories,
        key=lambda x: compute_trajectory_quality(x[1]),
        reverse=True
    )
    filtered = filtered[:int(len(filtered) * 0.8)]  # Top 80%

    # Fine-tune on filtered data
    for epoch in range(epochs):
        for query, trajectory in filtered:
            loss = model.compute_sft_loss(query, trajectory)
            loss.backward()
    model.optimizer.step()
```

### Stage 3: Reinforcement Learning
GRPO optimization with spectrum of reward signals matched to model scale.

```python
def reinforcement_learning(model, training_data, model_scale_billion=7):
    """
    GRPO training with reward design matched to model capacity.
    Curriculum rewards for small models; dense for large models.
    """
    if model_scale_billion <= 3:
        reward_fn = curriculum_reward  # Dense → sparse transition
        algo = "DAPO"
    else:
        reward_fn = dense_sum_reward  # Simple dense sum
        algo = "GRPO"

    for step in range(10000):
        # Generate diverse rollouts
        rollouts = []
        for query in training_data:
            trajectories = model.sample(query, group_size=8)  # G=8
            rollouts.extend(trajectories)

        # Compute rewards
        rewards = [reward_fn(traj) for traj in rollouts]

        # GRPO/DAPO optimization
        if algo == "DAPO":
            loss = compute_dapo_loss(rollouts, rewards)
        else:
            loss = compute_grpo_loss(rollouts, rewards)

        loss.backward()
        model.optimizer.step()
```

## Decision Checklist

- [ ] **Model capacity**: 7B+ → use dense reward; 1.5B–3B → use curriculum reward
- [ ] **Data synthesis**: Generate ~1,000 examples via back-translation with 4:3:3 difficulty split
- [ ] **SFT filtering**: Include successful trajectories only; reject-sample for top 80% quality
- [ ] **Reward function**: Semi-sparse "Macro" rewards (task-level, not step-level)
- [ ] **Training data**: Exactly ~1,000 examples; more causes over-optimization
- [ ] **GRPO configuration**: G=8 group size, 5 training epochs, KL-free training with increased clip bounds
- [ ] **Tool budget**: 60 calls per episode (tunable; affects task complexity ceiling)
- [ ] **Format protocol**: Enforce strict protocol; format errors receive reward=0
- [ ] **Environment stability**: Validate tool failure rate <5%; degrade gracefully at 5–10%
- [ ] **Algorithm selection**: 7B+ use standard GRPO; smaller models use DAPO/ARPO
- [ ] **Context windows**: 30K training, 32K inference
- [ ] **Benchmark**: Validate on TravelPlanner; compare against proprietary system baselines

## Conditions

### Effective Scenarios
- Long-horizon tasks requiring sequential tool invocation (planning, math, code generation, travel booking)
- Tasks with clearly definable success criteria and reward signals
- Models with sufficient scale (1.5B+) to support long context and complex reasoning
- Environments with manageable tool failure rates (<5% ideal, <10% acceptable)

### Environmental Requirements
- Sandbox execution environment with zero-cost tool simulation
- Solvable training queries (at least 80% success on best-effort attempts)
- Tool APIs with consistent interfaces and failure modes
- Reward functions computable from trajectory completion

### When to Reduce Complexity
- **Model <1.5B**: Consider smaller tool sets or shorter horizons; RL becomes less sample-efficient
- **Tool failure >10%**: Environment instability dominates; improve tool reliability before RL
- **Sparse reward signal**: Increase curriculum steps for smaller models; add task-level signal design
- **Over 2K training samples**: Performance plateaus; reallocate resources elsewhere

