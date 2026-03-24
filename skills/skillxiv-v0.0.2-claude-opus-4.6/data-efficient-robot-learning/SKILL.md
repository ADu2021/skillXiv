---
name: data-efficient-robot-learning
title: "Is Diversity All You Need for Scalable Robotic Manipulation?"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06219"
keywords: [Robot Learning, Data Diversity, Policy Learning, Transfer Learning, Distribution Debiasing]
description: "Train efficient robot manipulation policies by strategically applying task diversity and debiasing expert demonstrations to remove execution speed variations that degrade learning."
---

# Data-Efficient Robot Learning: Strategic Diversity and Distribution Debiasing

Robotic manipulation requires diverse training data, but conventional wisdom that "more diversity is always better" misleads practitioners. Scaling robot learning involves collecting data across multiple tasks, robot embodiments, and expert demonstrations. However, not all diversity helps equally. Task diversity substantially improves generalization, but expert diversity—particularly velocity variations in how the same task is executed—can actually degrade policy learning, introducing noise that models struggle to disentangle from legitimate strategy variations.

This work challenges scaling assumptions by showing that task diversity drives transfer learning, while expert velocity variations harm learning. By debiasing velocity multimodality through a learned velocity model, practitioners achieve 15% performance improvement equivalent to 2.5× additional pre-training data, without collecting proportionally larger datasets.

## Core Concept

Robot learning scales efficiently when diversity is applied strategically. The key insight involves distinguishing two types of multimodality in demonstrations: (1) spatial multimodality—legitimate alternative strategies to accomplish the same task (good for diversity), and (2) velocity multimodality—speed variations in executing the same strategy (noise that confuses learning). Single-embodiment pre-training transfers surprisingly well across different robot platforms with minimal adaptation, suggesting embodiment diversity may be unnecessary. Instead, focus should be on task diversity and removing harmful execution speed variations.

The distribution debiasing method uses a separate neural network to predict execution velocities from trajectories, then biases training toward demonstrations with consistent speeds. This removes multimodality sources that don't represent genuine strategic diversity.

## Architecture Overview

- **Task Diversity Component**: Pre-training on diverse manipulation tasks (pushing, grasping, placing, rotating) spanning different object types and environmental configurations
- **Embodiment-Agnostic Encoder**: Learns representations that transfer across robot platforms (arm kinematics, gripper types) without explicit morphology conditioning
- **Velocity Prediction Module**: Neural network that estimates execution speed from spatial trajectories, enabling identification and filtering of velocity-multimodal demonstrations
- **Policy Architecture**: Transformer-based imitation learning model that maps observations to actions, trained on distribution-debiased data
- **Adaptation Layer**: Lightweight fine-tuning mechanism for transferring to new embodiments or tasks with minimal data (2-5 demonstrations)

## Implementation

The following implements strategic data diversity and distribution debiasing for efficient robot policy learning.

**Step 1: Task Diversity Collection Framework**

This framework structures collection of diverse manipulation demonstrations while tracking task and embodiment metadata.

```python
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class RobotDemonstration:
    task_name: str
    embodiment: str
    trajectory: np.ndarray  # shape (T, state_dim)
    actions: np.ndarray    # shape (T, action_dim)
    velocities: np.ndarray # shape (T,) - computed speeds
    expert_id: str

class TaskDiversityCollector:
    def __init__(self):
        self.task_categories = [
            "pick_and_place",
            "pushing",
            "rotating",
            "opening_gripper",
            "stacking",
            "aligning"
        ]
        self.embodiments = ["ur5", "sawyer", "panda"]
        self.demonstrations = []

    def compute_trajectory_velocity(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute instantaneous velocity (distance traveled per timestep)."""
        diffs = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        # Pad to match trajectory length
        velocities = np.concatenate([[diffs[0]], diffs])
        return velocities

    def normalize_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """Normalize velocity to [0, 1] range."""
        v_min, v_max = velocities.min(), velocities.max()
        if v_max == v_min:
            return np.zeros_like(velocities)
        return (velocities - v_min) / (v_max - v_min)

    def collect_demonstration(
        self,
        task_name: str,
        embodiment: str,
        trajectory: np.ndarray,
        actions: np.ndarray,
        expert_id: str
    ) -> RobotDemonstration:
        """Collect and process a single demonstration."""
        velocities = self.compute_trajectory_velocity(trajectory)
        demo = RobotDemonstration(
            task_name=task_name,
            embodiment=embodiment,
            trajectory=trajectory,
            actions=actions,
            velocities=velocities,
            expert_id=expert_id
        )
        self.demonstrations.append(demo)
        return demo

    def analyze_task_coverage(self) -> Dict:
        """Analyze diversity of collected demonstrations."""
        task_counts = {}
        embodiment_counts = {}
        for demo in self.demonstrations:
            task_counts[demo.task_name] = task_counts.get(demo.task_name, 0) + 1
            embodiment_counts[demo.embodiment] = embodiment_counts.get(demo.embodiment, 0) + 1

        return {
            "total_demonstrations": len(self.demonstrations),
            "task_distribution": task_counts,
            "embodiment_distribution": embodiment_counts,
            "tasks_covered": len(task_counts),
            "embodiments_covered": len(embodiment_counts)
        }
```

**Step 2: Velocity Multimodality Detection**

This component identifies and measures velocity variations that harm learning, distinct from beneficial spatial strategy diversity.

```python
import torch
from torch import nn
from typing import Tuple

class VelocityPredictionModel(nn.Module):
    """Neural network that predicts trajectory execution velocity."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Velocity normalized to [0, 1]
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Predict average execution velocity from spatial trajectory."""
        # trajectory shape: (batch, seq_len, state_dim)
        # Encode trajectory globally
        encoded = self.encoder(trajectory.mean(dim=1))
        velocity = self.velocity_head(encoded)
        return velocity

class VelocityMultimodalityAnalyzer:
    def __init__(self, state_dim: int):
        self.velocity_model = VelocityPredictionModel(state_dim)
        self.optimizer = torch.optim.Adam(self.velocity_model.parameters(), lr=1e-3)

    def train_velocity_model(self, demonstrations: List[RobotDemonstration], epochs: int = 10):
        """Train model to predict execution velocities from trajectories."""
        self.velocity_model.train()

        for epoch in range(epochs):
            total_loss = 0
            for demo in demonstrations:
                trajectory_tensor = torch.FloatTensor(demo.trajectory).unsqueeze(0)
                target_velocity = torch.FloatTensor(
                    [np.mean(demo.velocities)]
                ).unsqueeze(0)

                pred_velocity = self.velocity_model(trajectory_tensor)
                loss = torch.nn.functional.mse_loss(pred_velocity, target_velocity)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(demonstrations):.4f}")

    def compute_velocity_variance_by_task(
        self, demonstrations: List[RobotDemonstration]
    ) -> Dict[str, Tuple[float, float]]:
        """Compute velocity mean and variance for each task."""
        task_velocities = {}

        self.velocity_model.eval()
        with torch.no_grad():
            for demo in demonstrations:
                trajectory_tensor = torch.FloatTensor(demo.trajectory).unsqueeze(0)
                pred_velocity = self.velocity_model(trajectory_tensor).item()

                if demo.task_name not in task_velocities:
                    task_velocities[demo.task_name] = []
                task_velocities[demo.task_name].append(pred_velocity)

        velocity_stats = {}
        for task, velocities in task_velocities.items():
            velocity_stats[task] = (np.mean(velocities), np.std(velocities))

        return velocity_stats

    def identify_multimodal_outliers(
        self,
        demonstrations: List[RobotDemonstration],
        std_threshold: float = 1.5
    ) -> List[Tuple[int, float]]:
        """Identify demonstrations with atypical execution velocities."""
        task_stats = self.compute_velocity_variance_by_task(demonstrations)
        outliers = []

        self.velocity_model.eval()
        with torch.no_grad():
            for i, demo in enumerate(demonstrations):
                trajectory_tensor = torch.FloatTensor(demo.trajectory).unsqueeze(0)
                pred_velocity = self.velocity_model(trajectory_tensor).item()

                task_mean, task_std = task_stats[demo.task_name]
                z_score = abs(pred_velocity - task_mean) / (task_std + 1e-6)

                if z_score > std_threshold:
                    outliers.append((i, z_score))

        return outliers
```

**Step 3: Distribution Debiasing for Training**

This applies velocity-based debiasing to create training datasets that improve policy learning.

```python
class DistributionDebiasingDataset:
    def __init__(
        self,
        demonstrations: List[RobotDemonstration],
        velocity_model: VelocityPredictionModel,
        debiasing_weight: float = 0.8
    ):
        self.demonstrations = demonstrations
        self.velocity_model = velocity_model
        self.debiasing_weight = debiasing_weight
        self.sample_weights = self._compute_sample_weights()

    def _compute_sample_weights(self) -> np.ndarray:
        """Compute importance weights based on velocity consistency."""
        self.velocity_model.eval()
        weights = []

        task_velocities = {}
        with torch.no_grad():
            for demo in self.demonstrations:
                trajectory_tensor = torch.FloatTensor(demo.trajectory).unsqueeze(0)
                pred_velocity = self.velocity_model(trajectory_tensor).item()

                if demo.task_name not in task_velocities:
                    task_velocities[demo.task_name] = []
                task_velocities[demo.task_name].append(pred_velocity)

        # Compute per-task velocity statistics
        task_stats = {}
        for task, velocities in task_velocities.items():
            task_stats[task] = (np.mean(velocities), np.std(velocities) + 1e-6)

        with torch.no_grad():
            for demo in self.demonstrations:
                trajectory_tensor = torch.FloatTensor(demo.trajectory).unsqueeze(0)
                pred_velocity = self.velocity_model(trajectory_tensor).item()

                task_mean, task_std = task_stats[demo.task_name]
                # Weight by velocity likelihood under task distribution
                z_score = abs(pred_velocity - task_mean) / task_std
                velocity_weight = np.exp(-self.debiasing_weight * z_score)
                weights.append(velocity_weight)

        return np.array(weights) / np.sum(weights) if sum(weights) > 0 else np.ones(len(weights))

    def sample_batch(self, batch_size: int) -> List[RobotDemonstration]:
        """Sample demonstrations with velocity-debiased probabilities."""
        indices = np.random.choice(
            len(self.demonstrations),
            size=batch_size,
            p=self.sample_weights,
            replace=True
        )
        return [self.demonstrations[i] for i in indices]
```

**Step 4: Cross-Embodiment Transfer Learning**

This demonstrates that single-embodiment pre-training transfers efficiently to new robot platforms.

```python
class RobotPolicyTransfer:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared encoder trained on source embodiment
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Task-specific decoder (fine-tuned on target embodiment)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def pretrain(self, train_demonstrations: List[RobotDemonstration], epochs: int = 5):
        """Pre-train on source embodiment (e.g., UR5)."""
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-3
        )

        for epoch in range(epochs):
            total_loss = 0
            for demo in train_demonstrations:
                states = torch.FloatTensor(demo.trajectory)
                actions = torch.FloatTensor(demo.actions)

                encoded = self.encoder(states)
                predicted_actions = self.decoder(encoded)
                loss = torch.nn.functional.mse_loss(predicted_actions, actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Pre-train Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_demonstrations):.4f}")

    def adapt_to_embodiment(
        self,
        target_demonstrations: List[RobotDemonstration],
        freeze_encoder: bool = True,
        epochs: int = 2
    ):
        """Fine-tune on target embodiment with minimal data."""
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=5e-4)

        for epoch in range(epochs):
            total_loss = 0
            for demo in target_demonstrations:
                states = torch.FloatTensor(demo.trajectory)
                actions = torch.FloatTensor(demo.actions)

                encoded = self.encoder(states)
                predicted_actions = self.decoder(encoded)
                loss = torch.nn.functional.mse_loss(predicted_actions, actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Adapt Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(target_demonstrations):.4f}")
```

## Practical Guidance

**Hyperparameters and Configuration**

| Parameter | Recommended Value | Range | Notes |
|-----------|------------------|-------|-------|
| Velocity Std Threshold | 1.5 | 1.0-2.5 | Identifies outlier velocities; lower = more aggressive debiasing |
| Debiasing Weight | 0.8 | 0.3-1.5 | Controls strength of velocity penalty; higher = stricter filtering |
| Task Diversity Minimum | 5+ tasks | 3-10+ | Ensures broad skill coverage; diminishing returns beyond 8 tasks |
| Pre-training Epochs | 5-10 | 3-20 | Sufficient for convergence on diverse task set |
| Fine-tuning Demonstrations | 2-5 per task | 1-10 | Efficient transfer requires minimal target data |
| Encoder Freeze During Adapt | Yes | True/False | Freezing preserves learned representations; unfreeze if target domain very different |

**When to Use**

- Multi-task robot learning where collecting diverse demonstrations is feasible
- Cross-embodiment transfer scenarios (training on one robot, deploying on another)
- Data-efficient learning where minimizing annotation effort is critical
- Real-world manipulation systems where task diversity improves generalization
- Scenarios where execution speed variations exist but don't represent strategic differences

**When NOT to Use**

- Single-task manipulation with high precision requirements (task diversity may not help)
- Embodiments with fundamentally different morphologies (e.g., humanoid vs. quadruped)
- Scenarios requiring expert-specific execution styles where velocity matters
- Systems where all collection data is already high-quality and velocity-consistent
- Real-time systems with strict latency requirements (velocity prediction adds overhead)

**Common Pitfalls**

- **Confusing spatial and velocity multimodality**: Not all variation is bad. Preserve legitimate strategy diversity while removing execution speed noise. Use velocity model to distinguish them.
- **Over-debiasing**: Aggressive velocity filtering may remove useful variation. Monitor performance and adjust threshold incrementally.
- **Insufficient task diversity**: Collecting from fewer than 5 tasks limits generalization benefits. Ensure breadth across object types, task types, and environmental configurations.
- **Ignoring embodiment capabilities**: Pre-trained encoders assume similar state/action spaces. Significant morphology differences require explicit adaptation layers.
- **Under-weighting pre-training duration**: Rushed pre-training degrades transfer performance. Allow sufficient epochs for convergence before adaptation.

## Reference

Is Diversity All You Need for Scalable Robotic Manipulation? https://arxiv.org/abs/2507.06219
