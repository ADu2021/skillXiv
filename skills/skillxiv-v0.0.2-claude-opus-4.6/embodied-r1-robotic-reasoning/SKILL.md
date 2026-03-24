---
name: embodied-r1-robotic-reasoning
title: "Embodied-R1: Reinforced Embodied Reasoning for Robotic Manipulation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.13998
keywords: [embodied-ai, robotic-manipulation, vision-language, pointing-representation, embodied-rl]
description: "Bridge vision-to-action gap using pointing as unified intermediate representation, enabling 56.2% success on manipulation tasks without task-specific fine-tuning."
---

# Embodied-R1: Reinforced Embodied Reasoning for Robotic Manipulation

## Core Concept

Robots struggle with "seeing-to-doing": understanding visual scenes doesn't directly translate to effective actions. Embodied-R1 uses "pointing" (spatial coordinates) as a bridge representation. The model learns to:
1. Understand visual scenes (vision)
2. Point to relevant objects/locations (spatial reasoning)
3. Execute actions based on pointing (embodied control)

Training combines supervised learning and embodied RL to teach these four pointing abilities.

## Architecture Overview

- **Vision-Language Base**: Processes images and language instructions
- **Pointing Representation**: Maps vision → spatial coordinates (pointing)
- **Action Primitives**: Convert pointing coordinates to robot actions
- **Embodied RL**: Refine using task-specific rewards
- **Generalization**: Works across different robot embodiments
- **Multi-Task**: Single model handles 100+ manipulation tasks

## Implementation Steps

### 1. Define Pointing Representation

```python
import torch
import numpy as np

class PointingRepresentation:
    """Spatial pointing as unified intermediate representation"""
    def __init__(self, image_resolution=(256, 256)):
        self.H, self.W = image_resolution

    def normalize_coordinates(self, x: float, y: float) -> tuple:
        """Normalize pixel coordinates to [-1, 1] range"""
        x_norm = 2 * (x / self.W) - 1
        y_norm = 2 * (y / self.H) - 1
        return (x_norm, y_norm)

    def denormalize_coordinates(self, x_norm: float, y_norm: float) -> tuple:
        """Convert normalized to pixel coordinates"""
        x = int((x_norm + 1) / 2 * self.W)
        y = int((y_norm + 1) / 2 * self.H)
        return (x, y)

    def create_pointing_heatmap(self, x_norm: float, y_norm: float,
                               sigma: float = 0.05) -> np.ndarray:
        """Create Gaussian heatmap for pointing location"""
        heatmap = np.zeros((self.H, self.W))

        for h in range(self.H):
            for w in range(self.W):
                # Normalize pixel to [-1, 1]
                h_norm = 2 * (h / self.H) - 1
                w_norm = 2 * (w / self.W) - 1

                # Gaussian distance
                dist = ((h_norm - y_norm) ** 2 + (w_norm - x_norm) ** 2) ** 0.5
                heatmap[h, w] = np.exp(-(dist ** 2) / (2 * sigma ** 2))

        return heatmap / heatmap.max()
```

### 2. Build Vision-Language Model with Pointing Head

```python
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModel

class EmbodiedVLM(nn.Module):
    """Vision-Language Model with pointing output head"""
    def __init__(self, vision_encoder='openai/clip-vit-base-patch32',
                 language_model='bert-base-uncased'):
        super().__init__()

        # Vision encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder)
        self.vision_dim = self.vision_encoder.config.hidden_size

        # Language model for instruction understanding
        self.language_model = AutoModel.from_pretrained(language_model)
        self.language_dim = self.language_model.config.hidden_size

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.vision_dim + self.language_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Pointing heads: predict (x, y) and confidence
        self.pointing_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # (x, y) in [-1, 1]
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Action primitives
        self.action_head = nn.Linear(256, 10)  # 10 action primitives

    def forward(self, image, instruction):
        """
        Process image + instruction, predict pointing and action
        """
        # Extract embeddings
        vision_features = self.vision_encoder(image).last_hidden_state
        vision_features = vision_features.mean(dim=1)  # Average pool

        language_features = self.language_model.encode(instruction)
        language_features = language_features.mean(dim=1)

        # Fuse modalities
        fused = torch.cat([vision_features, language_features], dim=-1)
        fused = self.fusion(fused)

        # Predict pointing
        pointing = torch.tanh(self.pointing_head(fused))  # [-1, 1]
        confidence = self.confidence_head(fused)

        # Predict action primitive
        action_logits = self.action_head(fused)

        return {
            'pointing': pointing,  # [batch, 2] (x, y)
            'confidence': confidence,  # [batch, 1]
            'action': action_logits  # [batch, 10]
        }
```

### 3. Implement Action Primitives

```python
class RobotActionExecutor:
    """Convert pointing to robot actions"""
    def __init__(self, robot_arm):
        self.robot = robot_arm
        self.action_primitives = {
            'reach': self._reach,
            'grasp': self._grasp,
            'push': self._push,
            'pick_place': self._pick_place,
            'slide': self._slide,
            'rotate': self._rotate,
            'release': self._release,
            'lift': self._lift,
            'rotate_wrist': self._rotate_wrist,
            'no_op': self._no_op
        }

    def execute_from_pointing(self, image, pointing_coords, action_type):
        """Convert pointing to robot command"""
        x_norm, y_norm = pointing_coords

        # Convert to 3D world coordinates
        # (would use camera calibration in real system)
        world_x, world_y = self._pointing_to_3d(x_norm, y_norm, image)

        # Execute action primitive
        if action_type in self.action_primitives:
            self.action_primitives[action_type](world_x, world_y)

    def _reach(self, x, y):
        """Reach to location"""
        self.robot.move_to(x, y, z=0.1)  # Hover above object

    def _grasp(self, x, y):
        """Grasp at location"""
        self.robot.move_to(x, y, z=0.05)
        self.robot.close_gripper()

    def _push(self, x, y):
        """Push action"""
        self.robot.move_to(x, y, z=0.05)
        self.robot.move_relative(-0.1, 0)  # Push forward

    def _pick_place(self, x, y):
        """Pick and place action"""
        # Grasp
        self.robot.move_to(x, y, z=0.05)
        self.robot.close_gripper()
        # Lift
        self.robot.move_relative(0, 0, 0.2)
        # Place
        self.robot.move_to(x + 0.2, y, z=0.05)
        self.robot.open_gripper()

    def _slide(self, x, y):
        """Slide action"""
        self.robot.move_to(x, y, z=0.05)
        self.robot.move_relative(0.1, 0.1)

    def _rotate(self, x, y):
        """Rotate action"""
        self.robot.move_to(x, y, z=0.05)
        self.robot.rotate(angle=45)

    def _release(self, x, y):
        """Release gripper"""
        self.robot.open_gripper()

    def _lift(self, x, y):
        """Lift action"""
        self.robot.move_relative(0, 0, 0.15)

    def _rotate_wrist(self, x, y):
        """Rotate wrist"""
        self.robot.rotate_wrist(angle=90)

    def _no_op(self, x, y):
        """No operation"""
        pass

    def _pointing_to_3d(self, x_norm, y_norm, image):
        """Convert normalized pointing to 3D world coordinates"""
        # Would use depth image + camera intrinsics
        # Simplified: treat as 2D for now
        x_pixel = int((x_norm + 1) / 2 * image.shape[1])
        y_pixel = int((y_norm + 1) / 2 * image.shape[0])
        return (x_pixel / 100.0, y_pixel / 100.0)  # Scale to meters
```

### 4. Train with Embodied RL

```python
def train_embodied_r1(model, executor, tasks_dataset, num_epochs=10):
    """Train with embodied RL"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for task_batch in tasks_dataset:
            images = task_batch['images']
            instructions = task_batch['instructions']
            success_labels = task_batch['success']

            # Forward pass
            outputs = model(images, instructions)
            pointing = outputs['pointing']
            action = outputs['action']
            confidence = outputs['confidence']

            # Collect trajectories and rewards
            rewards = []
            for i, (img, inst, success) in enumerate(zip(images, instructions, success_labels)):
                # Execute action
                action_idx = action[i].argmax().item()
                action_type = list(executor.action_primitives.keys())[action_idx]

                try:
                    executor.execute_from_pointing(
                        img,
                        pointing[i].detach().cpu().numpy(),
                        action_type
                    )
                except:
                    pass

                # Reward: 1.0 if successful, 0.0 otherwise
                reward = 1.0 if success[i] else 0.0
                rewards.append(reward)

            # Loss: supervised + RL
            pointing_loss = torch.nn.functional.mse_loss(
                pointing, task_batch['target_pointing']
            )

            action_loss = torch.nn.functional.cross_entropy(
                action, task_batch['action_labels']
            )

            # RL loss: policy gradient
            rewards_tensor = torch.tensor(rewards, device=action.device)
            log_probs = torch.nn.functional.log_softmax(action, dim=1)
            rl_loss = -(log_probs.max(dim=1)[0] * rewards_tensor).mean()

            total_loss = pointing_loss + action_loss + 0.1 * rl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        print(f"Epoch {epoch}: Loss={epoch_loss:.4f}")
```

### 5. Evaluation on Embodied Benchmarks

```python
def evaluate_embodied(model, executor, benchmark_tasks):
    """Evaluate on robotic manipulation benchmarks"""
    success_count = 0
    total_tasks = 0

    with torch.no_grad():
        for task in benchmark_tasks:
            image = task['image']
            instruction = task['instruction']

            # Predict
            outputs = model(image.unsqueeze(0), instruction)
            pointing = outputs['pointing'][0].cpu().numpy()
            action_idx = outputs['action'][0].argmax().item()

            action_type = list(executor.action_primitives.keys())[action_idx]

            # Execute
            try:
                executor.execute_from_pointing(image, pointing, action_type)
                # Check if successful (would need real robot or simulator)
                success = task['check_success']()
                if success:
                    success_count += 1
            except:
                pass

            total_tasks += 1

    success_rate = success_count / total_tasks if total_tasks > 0 else 0.0
    print(f"Success Rate: {success_rate * 100:.1f}%")
    return success_rate
```

## Practical Guidance

- **Pointing Sigma**: 0.05-0.1 for smooth heatmaps
- **Action Primitives**: 8-12 core actions (reachable in all scenarios)
- **RL Weight**: 0.05-0.1 relative to supervised loss
- **Training Data**: 100K+ real + simulated trajectories
- **Embodiments**: Train on multiple robot types for generalization

## Reference

Embodied-R1 (2508.13998): https://arxiv.org/abs/2508.13998

Use pointing as unified intermediate representation to bridge vision and robot control, achieving 56.2% success on manipulation tasks with single model across multiple robot embodiments.
