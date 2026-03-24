---
name: green-vla-robotics
title: "Green-VLA: Staged Vision-Language-Action Model for Generalist Robots"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.00919"
keywords: [Vision-Language-Action, Robot Control, Staged Training, Transfer Learning, Action Spaces]
description: "Train robot controllers via five-stage curriculum progressing from base vision-language models to embodiment-specific RL-refined policies. Unified action space enables cross-embodiment transfer with minimal performance loss."
---

# Green-VLA: Staged Robot Learning

## Problem
Robot learning typically requires task-specific pretraining and cannot leverage internet-scale vision-language knowledge. Each embodiment requires separate training from scratch.

Cross-embodiment transfer is difficult because robot actions are heterogeneous—different robots have different action dimensions and semantics. Naive action padding creates spurious penalties from unused dimensions.

## Core Concept
Green-VLA implements five training stages (L0→L1→R0→R1→R2) that progressively specialize vision-language models to robot control. The key innovation is a unified action space with fixed semantic layout where action index ranges have consistent physical meaning across robots.

This masked training approach eliminates spurious penalties and enables positive transfer between different embodiments.

## Architecture Overview

- **L0: Base VLM**: Start with large vision-language model (foundational knowledge)
- **L1: Web Pretraining**: Exposure to internet-scale video and multimodal data
- **R0: General Robotics Pretraining**: 3000+ hours diverse robot demonstrations
- **R1: Embodiment-Specific Fine-Tuning**: Supervised fine-tuning on target robot data
- **R2: RL Alignment**: Reinforcement learning for long-horizon task improvement
- **Unified Action Space**: Semantic layout with consistent meaning across embodiments

## Implementation

### Step 1: Build Unified Action Space
Design action space with fixed semantic regions across different robots.

```python
class UnifiedActionSpace:
    def __init__(self):
        # Fixed semantic layout for action indices
        self.action_regions = {
            'base_xy': (0, 2),           # XY movement
            'base_rotation': (2, 3),     # Rotation
            'shoulder': (3, 6),          # 3-DOF shoulder
            'elbow': (6, 8),             # 2-DOF elbow
            'wrist': (8, 11),            # 3-DOF wrist
            'gripper': (11, 12)          # Gripper
        }
        self.total_dim = 12

    def mask_for_embodiment(self, embodiment_type):
        """Generate mask for robot with specific capabilities."""
        mask = np.zeros(self.total_dim, dtype=bool)

        if embodiment_type == 'mobile_manipulator':
            mask[self.action_regions['base_xy']] = True
            mask[self.action_regions['base_rotation']] = True
            mask[self.action_regions['shoulder']] = True
            mask[self.action_regions['elbow']] = True
            mask[self.action_regions['gripper']] = True

        elif embodiment_type == 'fixed_arm':
            mask[self.action_regions['shoulder']] = True
            mask[self.action_regions['elbow']] = True
            mask[self.action_regions['wrist']] = True
            mask[self.action_regions['gripper']] = True

        return mask
```

### Step 2: Train Masked Policies
Fine-tune robot policies on masked action spaces.

```python
def train_masked_policy(model, demonstrations, embodiment_type, learning_rate=1e-4):
    """Train policy with masked action space."""
    action_space = UnifiedActionSpace()
    mask = action_space.mask_for_embodiment(embodiment_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(10):
        for image, language_instruction, action in demonstrations:
            # Forward pass
            predicted_action = model(image, language_instruction)

            # Apply mask: zero out unused dimensions
            masked_prediction = predicted_action * mask

            # Compute loss only on masked dimensions
            masked_action = action * mask

            loss = F.mse_loss(masked_prediction, masked_action)
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return model
```

### Step 3: Cross-Embodiment Transfer
Transfer weights from general pretraining to new embodiment.

```python
def transfer_to_new_embodiment(pretrained_model, target_embodiment, adaptation_data, num_finetune_steps=1000):
    """Adapt pretrained model to new robot embodiment."""
    # Copy pretrained weights
    adapted_model = copy.deepcopy(pretrained_model)

    # Fine-tune on target embodiment with masked loss
    adapted_model = train_masked_policy(
        adapted_model,
        adaptation_data,
        target_embodiment,
        learning_rate=5e-5  # Lower LR for fine-tuning
    )

    return adapted_model
```

### Step 4: RL Alignment for Long-Horizon Tasks
Refine policy using RL to improve long-horizon performance.

```python
def rl_refine_policy(model, environment, embodiment_mask, num_rollouts=100, learning_rate=1e-5):
    """Use RL to improve long-horizon task success."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for rollout in range(num_rollouts):
        # Execute trajectory in environment
        trajectory = environment.rollout(model, max_steps=100)

        # Compute task reward
        task_reward = environment.compute_reward(trajectory)

        # Policy gradient update
        log_probs = []
        for step in trajectory:
            image, instruction, action = step
            predicted_action = model(image, instruction)

            # Compute log prob under policy
            log_prob = compute_log_prob(predicted_action, action)
            log_probs.append(log_prob)

        # REINFORCE update
        trajectory_loss = -task_reward * sum(log_probs)
        trajectory_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return model
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model size | 7B-13B | Balance capability and efficiency |
| Web pretraining data | 100M+ videos | Internet-scale coverage |
| Robot pretraining hours | 3000+ | Diverse embodiment priors |
| Masked loss weight | 1.0 | Equal weight across masked dimensions |
| RL learning rate | 1e-5 to 5e-5 | Conservative fine-tuning |

### When to Use

- Deploying robot controllers across multiple embodiments
- Leveraging internet-scale vision-language knowledge
- Tasks requiring learned world models and affordances
- Long-horizon manipulation with learned recovery strategies
- Multi-robot systems needing efficient transfer

### When Not to Use

- Highly specialized, unique robot morphologies
- Real-time control with strict latency requirements
- Tasks solvable with classical control approaches
- Environments without demonstration data
- Fully constrained manipulation without learning needs

### Common Pitfalls

1. **Action space misalignment**: Semantic layout must match actual robot capabilities. Validate correspondence.
2. **Distribution shift**: Web pretraining distribution differs from robotics. Validate transfer with small tasks first.
3. **Mask brittleness**: Models may learn to predict masked dimensions if training data leaks. Audit training.
4. **RL instability**: Policy divergence in long-horizon RL common. Use conservative learning rates and entropy regularization.

## Reference
Green-VLA: Staged Vision-Language-Action Model for Generalist Robots
https://arxiv.org/abs/2602.00919
