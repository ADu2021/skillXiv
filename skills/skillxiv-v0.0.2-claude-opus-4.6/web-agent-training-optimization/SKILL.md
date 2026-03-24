---
name: web-agent-training-optimization
title: "How to Train Your LLM Web Agent: A Statistical Diagnosis"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.04103"
keywords: [Web Agents, Training Optimization, Compute Efficiency, Reinforcement Learning, Model Scaling]
description: "Optimize open-source LLM web agent training through systematic analysis of supervised fine-tuning vs. reinforcement learning trade-offs. Achieve 45% lower compute cost by branching into RL at strategic SFT checkpoints."
---

# Optimizing Web Agent Training: Statistical Diagnosis of SFT-RL Trade-offs

Training web agents that rival proprietary systems requires balancing expensive expert demonstrations against cheaper online reinforcement learning. Current approaches either rely entirely on supervised fine-tuning (SFT), which requires substantial human data, or jump to pure RL, which is inefficient. The key insight is that optimal branching occurs neither immediately nor late—there's a statistically-optimal checkpoint where switching from SFT to RL yields peak performance at minimum compute cost. By analyzing 1,370 configurations systematically, this work reveals that branching at 45% of SFT training achieves superior results at dramatically lower cost.

The core problem is the compute-efficiency frontier: adding more expert demonstrations helps but costs millions in human annotation. Online RL is cheaper but requires careful scheduling. The solution is identifying the optimal switching point.

## Core Concept

The training pipeline consists of three phases:

1. **Expert trajectory generation**: Teacher model (Llama 3.3 70B) generates high-quality demonstrations
2. **Supervised fine-tuning (SFT)**: Student model (Llama 3.1 8B) learns from expert trajectories
3. **Reinforcement learning (RL)**: Student branches into on-policy learning using GRPO for continued improvement

The critical insight is that the optimal branching point is neither immediate (RL needs SFT foundation) nor late (continued SFT shows diminishing returns). By analyzing the trade-off systematically across many configurations, the paper identifies the sweet spot: branch at ~45% of originally-planned SFT checkpoints.

## Architecture Overview

- **Teacher model**: Llama 3.3 70B generating expert trajectories
- **Student model**: Llama 3.1 8B (smaller, more efficient)
- **SFT stage**: Standard supervised learning on expert demonstrations
- **RL stage**: Group Relative Policy Optimization (GRPO) for online improvement
- **Multi-checkpoint branching**: Trains multiple models, each branching at different SFT iterations
- **Bootstrap statistical analysis**: Quantifies uncertainty and identifies optimal configurations

## Implementation

Generate expert trajectories from the teacher model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from web_agent.trajectories import TrajectoryDataset

# Load teacher model for demonstrations
teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat")

# Generate expert trajectories on benchmark tasks
tasks = load_benchmark_tasks("WebAgentBench")

trajectories = []
for task in tasks:
    # Teacher generates trajectory for this task
    trajectory = teacher.generate_trajectory(
        task_description=task["goal"],
        max_steps=50,
        temperature=0.7  # Some stochasticity for diversity
    )

    # Validate that trajectory actually solves the task
    if validate_trajectory(trajectory, task):
        trajectories.append({
            "task": task,
            "trajectory": trajectory,
            "success": True
        })

print(f"Generated {len(trajectories)} valid expert trajectories")

# Save for SFT training
save_trajectories(trajectories, "expert_demonstrations.jsonl")
```

Train the student model via supervised fine-tuning, with checkpoints at multiple intervals:

```python
import torch.optim as optim
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

# Load student model
student = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-8b-chat")

# Load expert trajectories
train_data = TrajectoryDataset("expert_demonstrations.jsonl")

# Training configuration with multiple save checkpoints
training_args = TrainingArguments(
    output_dir="checkpoints/sft",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    save_steps=100,  # Save checkpoint every 100 steps
    save_total_limit=30,  # Keep last 30 checkpoints
    logging_steps=50,
)

trainer = Trainer(
    model=student,
    args=training_args,
    train_dataset=train_data,
    data_collator=trajectory_collator,
)

# Train and save checkpoints
trainer.train()

# Collect all SFT checkpoints for later branching
sft_checkpoints = collect_checkpoints("checkpoints/sft/")
print(f"SFT produced {len(sft_checkpoints)} checkpoints for branching")
```

Branch at optimal checkpoint and launch RL training:

```python
from web_agent.rl import GRPOTrainer
from web_agent.env import WebAgentEnv

# Identify optimal branching checkpoint (45% of total)
optimal_checkpoint_idx = int(0.45 * len(sft_checkpoints))
branching_checkpoint = sft_checkpoints[optimal_checkpoint_idx]

# Load student from optimal SFT checkpoint
student_rl = AutoModelForCausalLM.from_pretrained(branching_checkpoint)

# Initialize RL trainer
rl_trainer = GRPOTrainer(
    model=student_rl,
    learning_rate=1e-5,  # Lower LR for RL phase
    num_train_epochs=2,
    batch_size=16,
    reward_model="web_agent_success"  # Binary: task succeeded or not
)

# Create environment for online rollouts
env = WebAgentEnv(benchmark="WebAgentBench")

# Run RL training
for epoch in range(rl_trainer.num_train_epochs):
    # Generate on-policy rollouts
    rollouts = []
    for task in tasks:
        trajectory = student_rl.generate_trajectory(
            task_description=task["goal"],
            max_steps=50
        )

        # Evaluate: does trajectory solve the task?
        success = validate_trajectory(trajectory, task)
        reward = 1.0 if success else 0.0

        rollouts.append({
            "trajectory": trajectory,
            "reward": reward,
            "task": task
        })

    # Update model via GRPO on these rollouts
    rl_trainer.train_step(rollouts)

    # Evaluate on validation set
    val_accuracy = evaluate_on_benchmark(student_rl, benchmark="WebAgentBench")
    print(f"RL epoch {epoch} validation accuracy: {val_accuracy:.2%}")
```

Analyze the compute-efficiency frontier across configurations:

```python
from web_agent.analysis import BootstrapAnalysis
import pandas as pd

# Train models branching at different SFT checkpoints (0%, 25%, 45%, 65%, 100%)
results = []

for branch_point in [0.0, 0.25, 0.45, 0.65, 1.0]:
    checkpoint_idx = int(branch_point * len(sft_checkpoints))

    # Train model branching at this point
    model = train_branched_agent(
        sft_checkpoint=sft_checkpoints[checkpoint_idx],
        rl_epochs=2
    )

    # Evaluate and track compute cost
    val_accuracy = evaluate_on_benchmark(model)
    compute_cost = estimate_compute(
        sft_steps=checkpoint_idx,
        rl_epochs=2
    )

    results.append({
        "branch_point": branch_point,
        "accuracy": val_accuracy,
        "compute_cost": compute_cost,
        "efficiency": val_accuracy / compute_cost  # Accuracy per compute unit
    })

# Analyze with bootstrap to quantify uncertainty
analyzer = BootstrapAnalysis(n_bootstrap=1000)
optimal = analyzer.find_optimal_configuration(results)

print(f"Optimal branching: {optimal['branch_point']:.0%} of SFT")
print(f"Accuracy gain: {optimal['accuracy']:.2%}")
print(f"Compute reduction: {(1-optimal['compute_cost']):.0%} vs. pure SFT")

# Visualize frontier
df = pd.DataFrame(results)
print(df)
```

## Practical Guidance

### When to Use Branched SFT-RL

Use this approach when:
- Training open-source models from scratch with limited budgets
- Expert demonstration data is expensive or limited
- You have compute infrastructure for online RL rollouts
- Model needs to improve beyond SFT performance
- Benchmark evaluation is feasible during training

### When NOT to Use

Avoid branching SFT-RL for:
- Domains lacking clear reward signals (classification, NLP)
- Tasks where expert data is extremely cheap and plentiful
- Real-time systems where training must complete quickly
- Environments where online rollouts are risky or expensive
- Tasks with highly ambiguous success criteria

### Compute-Efficiency Results

The paper demonstrates substantial improvements:

| Configuration | Accuracy | Compute Cost | Efficiency |
|---------------|----------|--------------|-----------|
| Pure SFT (3 epochs) | 56.4% | 1.0x | 0.564 |
| Early branching (25%) | 58.2% | 0.85x | 0.685 |
| Optimal branching (45%) | 62.1% | 0.55x | 1.129 |
| Late branching (65%) | 60.8% | 0.70x | 0.869 |
| Pure RL (no SFT) | 38.5% | 0.60x | 0.642 |

**Key insight**: Optimal branching at 45% achieves 62.1% accuracy at 45% lower compute cost than pure SFT.

### Branching Point Determination

| Model Size | Optimal SFT % | RL Epochs | Final Accuracy |
|-----------|---------------|-----------|----------------|
| 8B parameters | 45% | 2 | 62.1% |
| 13B parameters | 40% | 2 | 66.3% |
| 70B parameters | 35% | 1 | 71.8% |

Smaller models branch later; larger models can branch earlier.

### Key Hyperparameters

| Parameter | Typical Range | Guidance |
|-----------|---------------|----------|
| Branch point | 30%-60% of SFT | Domain-dependent; use 45% as starting point |
| SFT learning rate | 1e-5 to 5e-5 | Standard values work; monitor loss curves |
| RL learning rate | 1e-6 to 1e-5 | Should be lower than SFT (decay from SFT LR) |
| RL epochs | 1-3 | More epochs give diminishing returns beyond 2 |
| Rollout batch size | 16-64 | Balance between diversity and compute |

### Common Pitfalls

1. **Branching too early**: Zero or minimal SFT provides poor foundation for RL. Model struggles to generate valid trajectories.
2. **Branching too late**: Pure SFT shows diminishing returns beyond ~60%. Wasting compute on final SFT epochs.
3. **Ignoring curriculum**: RL rewards should progress gradually (e.g., partial credit for task steps) not binary final outcomes.
4. **Mismatched learning rates**: RL learning rate should be lower than SFT. Divergence is common with high RL LR.
5. **Forgetting validation monitoring**: Track validation accuracy throughout training to catch divergence early.

### Convergence Diagnosis

Monitor these metrics during training:

1. **SFT loss**: Should decrease monotonically; plateau by epoch 2-3
2. **RL reward**: Should increase; if flat, learning rate may be too low
3. **Validation accuracy**: Should improve with RL; stagnation indicates poor branching point
4. **Gradient norms**: Should remain stable; exploding gradients indicate divergence

### Bootstrap Statistical Analysis

The paper uses bootstrap resampling (1000 samples) to quantify uncertainty in optimal branching point:

```python
from scipy import stats

# Fit confidence intervals on frontier
ci_lower, ci_upper = analyzer.compute_confidence_interval(
    results=results,
    confidence=0.95
)

print(f"Optimal branching: 45% [CI: {ci_lower:.0%}, {ci_upper:.0%}]")
print(f"Compute savings: 45% [CI: ...]")
```

## Reference

"How to Train Your LLM Web Agent: A Statistical Diagnosis" - [arXiv:2507.04103](https://arxiv.org/abs/2507.04103)
