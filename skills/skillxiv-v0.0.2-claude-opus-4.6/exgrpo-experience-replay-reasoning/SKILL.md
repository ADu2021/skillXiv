---
name: exgrpo-experience-replay-reasoning
title: "ExGRPO: Strategic Experience Replay for Reasoning RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.02245
keywords: [RLVR, experience-replay, trajectory-selection, GRPO, reasoning]
description: "Improve RLVR training efficiency by selectively replaying trajectories based on correctness and entropy. Medium-difficulty questions and low-entropy solutions are most valuable; selective replay yields +3.5-7.6% improvements."
---

# ExGRPO: Strategic Experience Replay for Reasoning RL

Standard on-policy RLVR discards all experience after one update, losing valuable learning signals. ExGRPO identifies that trajectories vary dramatically in training value: correctness rate and entropy serve as effective indicators. Medium-difficulty questions with low-entropy solutions provide the best learning signal.

## Core Architecture

- **Difficulty bucketing**: Organize trajectories by success rate (correctness)
- **Entropy-based filtering**: Low-entropy trajectories most valuable for stability
- **Selective replay**: Prioritize medium-difficulty + low-entropy combinations
- **Delayed start**: Allow on-policy learning before enabling replay
- **Policy shaping**: Prevent replay from corrupting on-policy signal

## Implementation Steps

Setup selective experience replay for GRPO:

```python
# Initialize ExGRPO trajectory management
from exgrpo import TrajectoryBuffer, DifficultyBucketer, SelectiveReplay

trajectory_buffer = TrajectoryBuffer(
    max_size=100_000,
    difficulty_tiers=5,
    entropy_percentiles=[25, 50, 75]
)

difficulty_bucketer = DifficultyBucketer(
    metric="correctness_rate",
    window_size=100  # recent trajectories only
)

replay_manager = SelectiveReplay(
    strategy="entropy_guided",
    replay_probability=0.3,  # 30% of updates use replay
    medium_difficulty_range=(0.3, 0.7)  # 30-70% success rate
)
```

Execute GRPO training with experience replay:

```python
# Training loop with selective trajectory replay
trajectory_buffer.clear()

for epoch in range(num_epochs):
    # On-policy phase: standard GRPO without replay
    if epoch < replay_start_epoch:
        for batch in on_policy_dataloader:
            prompts = batch["prompt"]

            # Generate rollouts
            rollouts = model.rollout(
                prompts=prompts,
                num_rollouts=4,
                temperature=1.0
            )

            # Verify and compute rewards
            rewards = verifier.evaluate(rollouts)

            # Compute GRPO loss (standard)
            loss = compute_grpo_loss(rollouts, rewards)
            loss.backward()
            optimizer.step()

            # Store trajectories for potential replay
            for prompt, rollout, reward in zip(prompts, rollouts, rewards):
                trajectory = {
                    "prompt": prompt,
                    "rollout": rollout,
                    "reward": reward,
                    "is_correct": reward > 0.5,
                    "entropy": compute_entropy(rollout)
                }
                trajectory_buffer.add(trajectory)

    # Replay phase: selectively replay valuable trajectories
    else:
        # Mixed on-policy + replay updates
        for batch in on_policy_dataloader:
            use_replay = np.random.rand() < replay_manager.replay_probability

            if use_replay and len(trajectory_buffer) > 1000:
                # Sample replay batch with difficulty/entropy bias
                replay_batch = trajectory_buffer.sample(
                    strategy="medium_difficulty_low_entropy",
                    size=len(batch["prompt"]),
                    difficulty_range=(0.3, 0.7),
                    entropy_percentile=25  # bottom 25% entropy
                )

                prompts = replay_batch["prompts"]
                rollouts = replay_batch["rollouts"]
                rewards = replay_batch["rewards"]

                # Compute importance weights (for off-policy correction)
                importance_weights = compute_importance_weights(
                    old_policy=trajectory_buffer.policy_snapshot,
                    current_policy=model,
                    rollouts=rollouts
                )

                # GRPO loss with importance correction
                loss = compute_grpo_loss(
                    rollouts=rollouts,
                    rewards=rewards,
                    importance_weights=importance_weights
                )

            else:
                # Standard on-policy GRPO
                prompts = batch["prompt"]
                rollouts = model.rollout(prompts, num_rollouts=4)
                rewards = verifier.evaluate(rollouts)
                loss = compute_grpo_loss(rollouts, rewards)

            loss.backward()
            optimizer.step()

            # Update trajectory buffer with new experiences
            for prompt, rollout, reward in zip(prompts, rollouts, rewards):
                trajectory = {
                    "prompt": prompt,
                    "rollout": rollout,
                    "reward": reward,
                    "is_correct": reward > 0.5,
                    "entropy": compute_entropy(rollout)
                }
                trajectory_buffer.add(trajectory)
```

## Practical Guidance

**When to use ExGRPO:**
- Reasoning tasks (math, logic, code) with clear correctness signals
- Training data where problem difficulty varies substantially
- Scenarios where compute budget permits replay overhead
- Settings where training stability matters (weaker models especially benefit)

**When NOT to use:**
- Continuous reward domains (discrete correctness is key to ExGRPO)
- Streaming/online settings without trajectory storage
- Very large models where replay overhead unacceptable
- Tasks with uniform difficulty

**Hyperparameters:**
- **Difficulty range (0.3-0.7)**: Medium-difficulty sweet spot; test 0.25-0.75 for your domain
- **Entropy percentile (25)**: Keep at bottom 25%; controls trajectory consistency preference
- **Replay probability (0.3)**: 30% mixed on-policy/replay; increase to 0.5 for more replay
- **Replay start epoch**: Begin after 5-10 epochs to establish diverse buffer
- **Buffer size (100K)**: Increase for large datasets; decrease for memory constraints

## Performance Analysis

**Per-model improvements:**
- Qwen-1.5B: +7.6% (weaker models benefit most from replay)
- Qwen-7B: +5.2%
- Llama-8B: +3.5%
- Llama-34B: +2.1% (stronger models less improvement)

**Key finding:** Replay stabilizes training on weaker models; stronger models already stable.

## Trajectory Value Indicators

**High-value trajectories:**
- Medium difficulty (correctness 30-70%): Optimal learning signal
- Low entropy: Consistent, interpretable solutions
- Correct solutions: Reinforce successful patterns

**Low-value trajectories:**
- Very easy (>90% success): Redundant learning signal
- Very hard (<10% success): Noisy/unreliable gradients
- High entropy: Inconsistent or incoherent solutions

## Computational Overhead

- **Buffer management**: <2% overhead vs. standard GRPO
- **Sampling with bias**: <3% overhead (efficient sorting/filtering)
- **Importance weighting**: <1% overhead
- **Total**: ~5-6% computational cost for 3-7% accuracy gains

## Implementation Details

**Difficulty tracking:**
- Window-based: Track recent N trajectories for concept drift
- Per-problem-type: Different difficulty distributions across domains
- Adaptive ranges: Adjust medium-difficulty bounds based on buffer statistics

**Entropy computation:**
- Token-level: Average entropy of generation logits
- Alternative: Use token prediction confidence variance

## References

Builds on curriculum learning, experience replay in RL, and trajectory-based learning for language models.
