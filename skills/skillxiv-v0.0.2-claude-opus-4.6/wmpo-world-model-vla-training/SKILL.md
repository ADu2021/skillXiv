---
name: wmpo-world-model-vla-training
title: "WMPO: World Model-based Policy Optimization for VLA Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.09515"
keywords: [Reinforcement Learning, World Models, Vision-Language-Action, Embodied AI, Robot Learning]
description: "Train Vision-Language-Action models for robotic control through world model simulation without real-world interaction—using pixel-based world models aligned with VLA features to enable self-correction and robust policy optimization."
---

# Train Vision-Language-Action Robots Through Simulated World Models

Training robots with reinforcement learning typically requires expensive real-world trial-and-error. WMPO (World Model Policy Optimization) enables effective RL by learning a world model (pixel-based simulator) that accurately predicts VLA features trained on web-scale vision data. The robot then learns from simulated trajectories, avoiding real-world costs while gaining robust control behaviors.

The approach achieves improved sample efficiency, stronger overall performance, emergent self-correction abilities, and robust generalization—all critical for practical robotic manipulation.

## Core Concept

WMPO addresses a fundamental challenge: vision-language-action models are strong visual reasoners but lack direct RL training signals. The system creates a closed-loop learning environment:

1. **Pixel-Based World Model** - Predicts next video frames given actions, ensuring alignment with visual domain
2. **Feature Alignment** - World model outputs are regularized to match VLA features, bridging vision and action
3. **On-Policy RL** - Uses GRPO on simulated trajectories to optimize robot behaviors
4. **Self-Correction** - Robot learns to recover from failures through iterative refinement

This approach replaces real-world risk with model-based simulation risk, dramatically reducing training cost.

## Architecture Overview

- **Vision-Language-Action Model (VLA)**: Perceives images, reasons about tasks, outputs actions
- **Pixel-Based World Model**: Predicts next frames [s_t, a_t] → s_{t+1}
- **Feature Alignment Layer**: Ensures world model outputs match VLA feature space
- **Trajectory Simulator**: Rolls out imagined episodes using world model
- **Policy Optimizer (GRPO)**: Updates VLA parameters based on simulated rewards
- **Real-World Validator**: Periodic real validation to detect model divergence

## Implementation Steps

**Step 1: Train Pixel-Based World Model**

Build a generative model predicting next frames from states and actions.

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class PixelWorldModel(nn.Module):
    """
    World model predicting next pixel frames given current state and action.
    """

    def __init__(self, input_channels: int = 3, action_dim: int = 7,
                 latent_dim: int = 256, num_blocks: int = 4):
        """
        Args:
            input_channels: RGB channels (3)
            action_dim: Dimension of action vector
            latent_dim: Latent representation dimension
            num_blocks: Number of residual blocks for prediction
        """
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder: compress frames to latent
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Transition predictor: [latent + action_embed] → next_latent
        self.transition = nn.ModuleList([
            nn.ConvTranspose2d(latent_dim + latent_dim, latent_dim,
                              kernel_size=3, padding=1)
            for _ in range(num_blocks)
        ])

        # Decoder: decompress latent to frames
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, current_frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next frame given current frame and action.

        Args:
            current_frame: Current image [batch, 3, H, W]
            action: Action vector [batch, action_dim]

        Returns:
            next_frame: Predicted next frame [batch, 3, H, W]
        """
        # Encode current frame
        latent = self.encoder(current_frame)

        # Embed and concatenate action
        action_emb = self.action_embed(action)
        action_emb = action_emb.unsqueeze(-1).unsqueeze(-1)
        action_emb = action_emb.expand(-1, -1, latent.shape[-2], latent.shape[-1])

        combined = torch.cat([latent, action_emb], dim=1)

        # Predict next latent through transitions
        next_latent = combined
        for transition in self.transition:
            next_latent = transition(next_latent)
            next_latent = torch.relu(next_latent)

        # Decode to frame
        next_frame = self.decoder(next_latent)

        return next_frame

def train_world_model(model, train_dataloader, num_epochs: int = 10):
    """
    Train world model on video/trajectory data.

    Args:
        model: PixelWorldModel instance
        train_dataloader: Dataloader with (frame_t, action, frame_t+1) tuples
        num_epochs: Training epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (frame_t, action, frame_t1) in enumerate(train_dataloader):
            # Predict next frame
            pred_frame = model(frame_t, action)

            # L2 loss on pixel prediction
            loss = criterion(pred_frame, frame_t1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}: World model loss {avg_loss:.4f}")
```

**Step 2: Align World Model Features with VLA Features**

Ensure world model outputs match the feature space of the VLA model.

```python
class FeatureAlignment(nn.Module):
    """
    Aligns world model output features with VLA model features.
    """

    def __init__(self, world_model_dim: int = 256,
                 vla_feature_dim: int = 512, hidden_dim: int = 512):
        """
        Args:
            world_model_dim: Dimension of world model outputs
            vla_feature_dim: Dimension of VLA feature space
            hidden_dim: Hidden dimension for projection
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(world_model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vla_feature_dim)
        )

    def align(self, world_features: torch.Tensor) -> torch.Tensor:
        """
        Project world model features to VLA feature space.

        Args:
            world_features: Features from world model [batch, world_model_dim]

        Returns:
            aligned_features: Features in VLA space [batch, vla_feature_dim]
        """
        return self.projection(world_features)

def compute_alignment_loss(world_features: torch.Tensor,
                          vla_features: torch.Tensor,
                          alignment_module: FeatureAlignment,
                          temperature: float = 0.1) -> torch.Tensor:
    """
    Contrastive loss ensuring alignment between world and VLA features.

    Args:
        world_features: Features from world model
        vla_features: Features from VLA model (from real observations)
        alignment_module: Feature alignment network
        temperature: Contrastive temperature

    Returns:
        alignment_loss: Loss value
    """
    # Project world features
    aligned = alignment_module.align(world_features)

    # Normalize
    aligned_norm = F.normalize(aligned, dim=-1)
    vla_norm = F.normalize(vla_features, dim=-1)

    # Cosine similarity (contrastive learning)
    similarity = torch.mm(aligned_norm, vla_norm.t()) / temperature

    # Diagonal should be high (matching features), off-diagonal low
    labels = torch.arange(aligned_norm.shape[0], device=aligned_norm.device)
    loss = F.cross_entropy(similarity, labels) + \
           F.cross_entropy(similarity.t(), labels)

    return loss
```

**Step 3: Trajectory Rollout in World Model**

Simulate episodes using the world model for RL training.

```python
class WorldModelSimulator:
    """
    Simulates trajectories in learned world model.
    """

    def __init__(self, world_model: PixelWorldModel, vla_model,
                 feature_alignment: FeatureAlignment, max_episode_length: int = 50):
        """
        Args:
            world_model: Trained PixelWorldModel
            vla_model: Vision-Language-Action model
            feature_alignment: Feature alignment network
            max_episode_length: Rollout horizon
        """
        self.world_model = world_model
        self.vla = vla_model
        self.alignment = feature_alignment
        self.max_length = max_episode_length

    def simulate_trajectory(self, initial_frame: torch.Tensor,
                           task_instruction: str) -> Dict[str, Any]:
        """
        Rollout episode in world model.

        Args:
            initial_frame: Starting image [3, H, W]
            task_instruction: Natural language task description

        Returns:
            trajectory: {frames, actions, rewards, features}
        """
        frames = [initial_frame]
        actions = []
        features = []

        current_frame = initial_frame.unsqueeze(0)

        for step in range(self.max_length):
            # Get VLA action
            with torch.no_grad():
                action, feature = self.vla.forward(
                    current_frame, task_instruction,
                    return_features=True
                )

            actions.append(action)
            features.append(feature)

            # Predict next frame in world model
            with torch.no_grad():
                next_frame = self.world_model(current_frame, action)

            frames.append(next_frame.squeeze(0))
            current_frame = next_frame

            # Early stopping if task complete (optional: use classifier)
            if self._is_task_complete(next_frame, task_instruction):
                break

        return {
            'frames': torch.stack(frames),
            'actions': torch.stack(actions),
            'features': torch.stack(features),
            'trajectory_length': len(actions)
        }

    def _is_task_complete(self, frame: torch.Tensor, instruction: str) -> bool:
        """Heuristic: detect task completion."""
        # In practice: train a task completion classifier
        return False
```

**Step 4: On-Policy RL with GRPO**

Optimize VLA policy using simulated trajectories.

```python
def compute_trajectory_reward(trajectory: Dict, task_instruction: str,
                            reward_model=None) -> torch.Tensor:
    """
    Compute reward for simulated trajectory.

    Args:
        trajectory: Simulated episode
        task_instruction: Task description
        reward_model: Optional learned reward model

    Returns:
        rewards: Per-step reward estimates
    """
    if reward_model is None:
        # Simple reward: larger movements toward end = progress
        positions = trajectory['frames']
        movement = torch.norm(positions[1:] - positions[:-1], dim=(2, 3, 4))
        # Normalize and scale
        rewards = (movement - movement.mean()) / (movement.std() + 1e-6)
    else:
        # Learned reward model
        features = trajectory['features']
        rewards = reward_model.predict(features)

    return rewards

def grpo_update(vla_model, trajectories: List[Dict],
               task_instruction: str, learning_rate: float = 1e-6):
    """
    Group Relative Policy Optimization update.

    Args:
        vla_model: VLA to optimize
        trajectories: List of simulated trajectories
        task_instruction: Task instruction
        learning_rate: Optimizer learning rate
    """
    optimizer = torch.optim.Adam(vla_model.parameters(), lr=learning_rate)

    # Compute rewards for each trajectory
    rewards = []
    for traj in trajectories:
        traj_reward = compute_trajectory_reward(traj, task_instruction)
        rewards.append(traj_reward.mean())

    rewards = torch.tensor(rewards)

    # Compute advantages (group relative)
    mean_reward = rewards.mean()
    advantages = rewards - mean_reward

    # Policy gradient update
    for trajectory, advantage in zip(trajectories, advantages):
        # Get log probabilities of actions in trajectory
        log_probs = vla_model.compute_log_prob(
            trajectory['frames'],
            trajectory['actions'],
            task_instruction
        )

        # Policy loss: gradient ascent on log_prob * advantage
        loss = -(log_probs.mean() * advantage)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Practical Guidance

**When to Use WMPO:**
- Robotic manipulation tasks with expensive real-world time
- Vision-language models requiring RL refinement
- Tasks where world model errors are acceptable (simulation bias tolerable)

**When NOT to Use:**
- Tasks requiring perfect real-world dynamics (world model too inaccurate)
- Scenarios with limited simulation-to-reality transfer
- Real-time systems where planning latency is critical

**Hyperparameters and Configuration:**
- World model update frequency: Every 50 real episodes, retrain on collected rollouts
- Feature alignment weight: 0.1-0.5 (balance world model accuracy with VLA alignment)
- Rollout horizon: 20-50 steps (longer = more RL signal, higher model error risk)
- GRPO update frequency: After every 10 simulated episodes

**Pitfalls to Avoid:**
1. **Compounding error** - World model errors accumulate over long rollouts; use short horizons initially
2. **Distribution shift** - World model trained on one distribution; regularly validate on real data
3. **Reward sparsity** - Simulated rewards may be weak; use shaped rewards or learned models
4. **Alignment failure** - If world model features drift from VLA, training fails; monitor alignment loss

---

Reference: https://arxiv.org/abs/2511.09515
