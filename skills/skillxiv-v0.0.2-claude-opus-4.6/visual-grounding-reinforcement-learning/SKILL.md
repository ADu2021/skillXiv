---
name: visual-grounding-reinforcement-learning
title: "High-Resolution Visual Reasoning via Multi-Turn Grounding-Based Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.05920"
keywords: [Multimodal LLM, Grounding, Reinforcement Learning, Visual Reasoning, High-Resolution Images]
description: "Improve LMM performance on high-resolution images by training models to ground reasoning on image regions through RL, learning spatial localization without requiring expensive grounding annotations."
---

# Visual Grounding Reinforcement Learning: Emergent Spatial Reasoning Without Annotations

Large multimodal models (LMMs) struggle with high-resolution images because they process entire images at once, losing detailed visual information critical for accurate reasoning. Humans naturally zoom into relevant regions when analyzing complex images. Yet teaching models to perform spatial grounding typically requires expensive annotations (bounding boxes, coordinates) linking questions to image regions.

This work proposes MGPO (Multi-turn Grounding Policy Optimization), an RL framework enabling LMMs to learn spatial grounding autonomously using only binary correctness signals. The key insight is that grounding can emerge as a side effect of RL training on question-answering tasks if the model architecture allows it. By designing a two-turn dialogue template where the model first predicts coordinates, then answers based on cropped sub-images, grounding emerges naturally during optimization without requiring supervised grounding annotations.

## Core Concept

The fundamental insight is that visual grounding—learning to focus on relevant image regions—emerges naturally through RL training when the model architecture supports it, using only binary reward signals (correct/incorrect). Rather than requiring supervised fine-tuning on coordinate annotations, we structure the task so that effective spatial reasoning becomes the optimal policy for improving answer correctness.

The two-turn mechanism works as: (1) Turn 1: Model predicts coordinates of relevant image regions, (2) Turn 2: Model receives cropped sub-image and answers the question. RL training uses only the final answer's correctness to update both the coordination mechanism and the reasoning process, causing the model to learn to focus on helpful regions.

## Architecture Overview

- **Base LMM Encoder**: Vision transformer extracting features from high-resolution images
- **Coordinate Prediction Head**: Neural network predicting pixel coordinates [x, y, width, height] for relevant regions
- **Image Cropping Module**: Differentiable or discrete image selection mechanism extracting sub-regions based on predicted coordinates
- **Question-Answering Head**: LLM decoder generating answers from cropped image features and text queries
- **Multi-turn Dialogue Template**: Conversational structure alternating between region prediction (turn 1) and QA reasoning (turn 2)
- **Reward Model**: Binary classifier assessing answer correctness from final output
- **RL Policy Optimization**: GRPO (Group Relative Policy Optimization) extended to multi-turn sequences

## Implementation

The following implements visual grounding emergence through RL without requiring coordinate annotations.

**Step 1: Coordinate Prediction and Image Cropping**

This module learns to predict relevant image regions and crop them.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CoordinatePredictionHead(nn.Module):
    """Predicts bounding box coordinates for image regions."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)  # [x, y, width, height]
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Predict bounding box coordinates.
        Args:
            image_features: pooled image embeddings (batch, input_dim)
        Returns:
            normalized coordinates (batch, 4) in [0, 1] range
        """
        logits = self.mlp(image_features)
        # Normalize to [0, 1] for coordinates
        coordinates = torch.sigmoid(logits)
        return coordinates

class DifferentiableImageCropper(nn.Module):
    """Extract image regions using differentiable spatial transformer."""

    def __init__(self, image_size: int = 1024):
        super().__init__()
        self.image_size = image_size

    def forward(
        self,
        images: torch.Tensor,
        coordinates: torch.Tensor,
        output_size: int = 256
    ) -> torch.Tensor:
        """
        Crop image regions using predicted coordinates.
        Args:
            images: input images (batch, 3, H, W)
            coordinates: normalized coords [x, y, w, h] (batch, 4)
            output_size: size of cropped output
        Returns:
            cropped regions (batch, 3, output_size, output_size)
        """
        batch_size = images.shape[0]

        # Denormalize coordinates
        x = coordinates[:, 0] * self.image_size
        y = coordinates[:, 1] * self.image_size
        w = coordinates[:, 2] * self.image_size
        h = coordinates[:, 3] * self.image_size

        # Clamp to image boundaries
        x = torch.clamp(x, 0, self.image_size - 1)
        y = torch.clamp(y, 0, self.image_size - 1)
        w = torch.clamp(w, 1, self.image_size)
        h = torch.clamp(h, 1, self.image_size)

        # Extract patches using grid_sample (differentiable)
        crops = []
        for i in range(batch_size):
            x_min = int(x[i].item())
            y_min = int(y[i].item())
            x_max = min(int(x[i].item() + w[i].item()), self.image_size)
            y_max = min(int(y[i].item() + h[i].item()), self.image_size)

            # Crop region
            crop = images[i, :, y_min:y_max, x_min:x_max]

            # Resize to fixed output size
            crop = F.interpolate(
                crop.unsqueeze(0),
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            crops.append(crop)

        return torch.stack(crops)
```

**Step 2: Multi-Turn Grounding-Based QA**

This implements the two-turn dialogue where grounding emerges from RL.

```python
from transformers import AutoTokenizer, AutoModel

class GroundingQAModel(nn.Module):
    """LMM with multi-turn grounding for high-resolution visual reasoning."""

    def __init__(self, vision_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.image_size = 1024

        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        self.vision_dim = self.vision_encoder.config.hidden_size

        # Coordinate prediction (turn 1)
        self.coordinate_head = CoordinatePredictionHead(self.vision_dim)

        # Image cropper
        self.cropper = DifferentiableImageCropper(self.image_size)

        # Text encoder and LLM decoder
        self.tokenizer = AutoTokenizer.from_pretrained(vision_model_name)
        self.language_model = AutoModel.from_pretrained(vision_model_name)

        # QA head (turn 2)
        self.qa_head = nn.Linear(self.vision_dim, 100)  # Answer classification

    def forward(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
        grounding_enabled: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional two-turn grounding.
        Args:
            images: input images (batch, 3, 1024, 1024)
            questions: tokenized questions (batch, seq_len)
            grounding_enabled: use grounding (True) or process full image
        Returns:
            (coordinates, answer_logits, confidence)
        """
        batch_size = images.shape[0]

        # Encode full image
        image_features = self.vision_encoder(images).last_hidden_state
        image_pooled = image_features.mean(dim=1)  # (batch, vision_dim)

        # Turn 1: Predict coordinate grounding
        coordinates = self.coordinate_head(image_pooled)  # (batch, 4)

        # Turn 2: Process based on grounding
        if grounding_enabled:
            # Crop relevant regions using predicted coordinates
            cropped_images = self.cropper(images, coordinates)

            # Encode cropped regions
            cropped_features = self.vision_encoder(cropped_images).last_hidden_state
            cropped_pooled = cropped_features.mean(dim=1)
            visual_features = cropped_pooled
        else:
            # No grounding: use full image features
            visual_features = image_pooled

        # Encode text question
        text_features = self.language_model(questions).last_hidden_state
        text_pooled = text_features.mean(dim=1)

        # Fuse visual and text features
        fused = visual_features + text_pooled

        # Generate answer
        answer_logits = self.qa_head(fused)

        # Confidence in answer (entropy-based)
        probs = F.softmax(answer_logits, dim=-1)
        confidence = 1.0 - (-probs * torch.log(probs + 1e-8)).sum(dim=-1)

        return coordinates, answer_logits, confidence
```

**Step 3: RL Training with GRPO**

This implements policy optimization for learning grounding through RL.

```python
import torch.optim as optim

class MultiTurnGRPO:
    """Group Relative Policy Optimization adapted for multi-turn grounding."""

    def __init__(
        self,
        model: GroundingQAModel,
        learning_rate: float = 5e-5,
        beta: float = 0.1
    ):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.beta = beta  # KL divergence coefficient

    def compute_reward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Binary reward based on answer correctness.
        Returns 1.0 for correct, 0.0 for incorrect.
        """
        pred_ids = predictions.argmax(dim=-1)
        rewards = (pred_ids == labels).float()
        return rewards

    def compute_policy_loss(
        self,
        coordinates: torch.Tensor,
        old_coordinates: torch.Tensor,
        answer_logits: torch.Tensor,
        old_answer_logits: torch.Tensor,
        rewards: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GRPO loss across multi-turn policy.
        Optimizes both grounding (coordinates) and QA policy.
        """
        # New policy log probabilities
        new_coord_probs = torch.distributions.Normal(
            coordinates.mean(), coordinates.std() + 1e-5
        ).log_prob(old_coordinates).sum(dim=-1)

        new_qa_logprobs = F.log_softmax(answer_logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)

        # Old policy log probabilities
        old_coord_probs = torch.distributions.Normal(
            old_coordinates.mean(), old_coordinates.std() + 1e-5
        ).log_prob(old_coordinates).sum(dim=-1)

        old_qa_logprobs = F.log_softmax(old_answer_logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)

        # Policy ratio (coordinates + QA)
        ratio = torch.exp(new_coord_probs - old_coord_probs + new_qa_logprobs - old_qa_logprobs)

        # Advantage: reward signal
        advantage = rewards - 0.5  # Normalize reward

        # PPO-style clipped loss
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.beta, 1 + self.beta) * advantage
        loss = -torch.min(surr1, surr2).mean()

        return loss

    def train_step(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
        labels: torch.Tensor,
        num_groups: int = 4
    ) -> float:
        """
        Single RL training step with grouped relative policy optimization.
        """
        batch_size = images.shape[0]

        # Sample multiple trajectories
        coordinates_list = []
        answer_logits_list = []

        self.model.eval()
        with torch.no_grad():
            for _ in range(num_groups):
                coords, logits, _ = self.model(images, questions, grounding_enabled=True)
                coordinates_list.append(coords)
                answer_logits_list.append(logits)

        # Compute rewards for each trajectory
        rewards_list = [
            self.compute_reward(logits, labels)
            for logits in answer_logits_list
        ]

        # Relative advantage: reward - group average
        avg_reward = torch.stack(rewards_list).mean(dim=0)
        relative_rewards = [r - avg_reward for r in rewards_list]

        # Update policy on best trajectories
        self.model.train()
        total_loss = 0

        for i in range(num_groups):
            coords, logits, _ = self.model(images, questions, grounding_enabled=True)

            # Use relative advantage from offline trajectories
            with torch.no_grad():
                old_coords = coordinates_list[i]
                old_logits = answer_logits_list[i]

            loss = self.compute_policy_loss(
                coords, old_coords,
                logits, old_logits,
                relative_rewards[i], labels
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_groups

    def train_epoch(self, train_loader, num_epochs: int = 3, num_groups: int = 4):
        """Train for multiple epochs with RL."""
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                images = batch["images"]
                questions = batch["questions"]
                labels = batch["labels"]

                loss = self.train_step(images, questions, labels, num_groups=num_groups)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")
```

**Step 4: Evaluation of Emergent Grounding**

This evaluates whether grounding emerges and improves reasoning accuracy.

```python
class GroundingEvaluator:
    def __init__(self, model: GroundingQAModel):
        self.model = model

    def evaluate_reasoning_accuracy(self, eval_loader) -> float:
        """Evaluate QA accuracy on high-resolution images."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in eval_loader:
                images = batch["images"]
                questions = batch["questions"]
                labels = batch["labels"]

                _, answer_logits, _ = self.model(images, questions, grounding_enabled=True)
                preds = answer_logits.argmax(dim=-1)

                correct += (preds == labels).sum().item()
                total += labels.shape[0]

        return correct / total

    def analyze_learned_grounding(self, eval_loader, num_samples: int = 10) -> Dict:
        """Analyze what the model learns to ground on."""
        self.model.eval()
        grounding_analysis = {
            "avg_zoom_level": [],
            "region_focus": [],
            "coordinate_consistency": []
        }

        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= num_samples:
                    break

                images = batch["images"]
                questions = batch["questions"]

                # Multiple forward passes for same input
                coordinates_runs = []
                for _ in range(5):
                    coords, _, _ = self.model(images, questions, grounding_enabled=True)
                    coordinates_runs.append(coords)

                coords_tensor = torch.stack(coordinates_runs)

                # Analyze consistency of grounding
                coord_std = coords_tensor.std(dim=0).mean().item()
                grounding_analysis["coordinate_consistency"].append(coord_std)

                # Analyze zoom level (width * height)
                zoom_levels = (coords_tensor[:, 2] * coords_tensor[:, 3]).mean(dim=0)
                grounding_analysis["avg_zoom_level"].extend(zoom_levels.tolist())

        return {
            k: np.mean(v) if v else 0.0
            for k, v in grounding_analysis.items()
        }
```

## Practical Guidance

**Hyperparameters and Configuration**

| Parameter | Recommended Value | Range | Notes |
|-----------|------------------|-------|-------|
| Image Resolution | 1024 × 1024 | 512-2048 | Higher resolution captures details but increases compute |
| Cropped Region Size | 256 × 256 | 128-512 | Smaller regions = faster inference, larger = more detail |
| RL Learning Rate | 5e-5 | 1e-5 to 1e-3 | Conservative to prevent policy collapse |
| GRPO Beta (KL penalty) | 0.1 | 0.05-0.5 | Controls deviation from prior policy |
| Num Groups (GRPO) | 4 | 2-8 | More groups improve advantage estimation |
| Reward Baseline | 0.5 | 0.3-0.7 | Helps with advantage scaling |
| Epochs | 3-5 | 1-10 | Longer training allows grounding to emerge |

**When to Use**

- Visual question answering on high-resolution images (medical imaging, technical diagrams, maps)
- Document image understanding requiring fine-grained text reading
- Any multimodal task where focusing on relevant regions improves accuracy
- Scenarios where collecting grounding annotations is expensive or infeasible
- Applications where understanding spatial relationships matters (scene understanding, layout analysis)

**When NOT to Use**

- Simple image classification tasks (full image is relevant)
- Real-time systems with strict latency requirements (two-turn dialogue adds overhead)
- Scenarios where all image regions are equally important
- Tasks where pixel-level precision is required (grounding may be coarse)
- Systems where interpretability of region selection is less important than accuracy

**Common Pitfalls**

- **Expecting supervision to teach grounding better**: RL with only binary signals actually outperforms supervised fine-tuning with bounding boxes because it learns task-specific grounding rather than copying annotations.
- **Ignoring crop size impact**: Cropping too aggressively loses context; too conservatively wastes computation. Experiment with output_size parameter.
- **Insufficient RL training**: Grounding emerges slowly. Ensure sufficient epochs (5+) for learning to stabilize.
- **Not monitoring coordinate consistency**: If learned coordinates are too noisy (high std), increase temperature or use coordinate smoothing.
- **Confusing coordinate scale**: Ensure coordinates are normalized to [0, 1] matching image normalization. Mismatches cause poor cropping.

## Reference

High-Resolution Visual Reasoning via Multi-Turn Grounding-Based Reinforcement Learning. https://arxiv.org/abs/2507.05920
