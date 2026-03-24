---
name: internvl3-5-multimodal-cascade-rl
title: "InternVL3.5: Cascade RL and Visual Resolution Router for Multimodal Efficiency"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.18265
keywords: [multimodal-models, cascade-reinforcement-learning, visual-efficiency, reasoning, inference-optimization]
description: "Enhance multimodal models through cascade RL for reasoning improvement and visual resolution routing for inference efficiency, achieving 16% reasoning gains and 4.05x speedup."
---

# InternVL3.5: Cascade RL and Visual Resolution Router

## Core Concept

InternVL3.5 advances multimodal AI through two complementary techniques: Cascade RL for enhanced reasoning and Visual Resolution Router (ViR) for adaptive efficiency. Cascade RL combines offline RL for stable convergence with online RL for refined alignment, improving performance on visual reasoning benchmarks. ViR dynamically adjusts visual token resolution based on image complexity, maintaining performance while reducing computation. Together, these achieve 16% reasoning improvement and 4.05x inference speedup.

## Architecture Overview

- **Cascade RL Framework**: Two-stage training (offline → online)
- **Visual Resolution Router**: Dynamic resolution selection per image
- **Decoupled Vision-Language Deployment**: Distributed inference
- **Reasoning Enhancement**: Improved visual understanding
- **Efficiency Optimization**: Token reduction without quality loss

## Implementation Steps

### 1. Implement Cascade RL Framework

Two-stage training for stable and effective policy learning:

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class CascadeRLTrainer:
    """Two-stage RL: offline stable convergence + online refinement."""

    def __init__(
        self,
        model: "MultimodalLLM",
        offline_steps: int = 1000,
        online_steps: int = 500,
        offline_lr: float = 1e-5,
        online_lr: float = 5e-6
    ):
        self.model = model
        self.offline_steps = offline_steps
        self.online_steps = online_steps

        # Separate optimizers for offline and online phases
        self.offline_optimizer = torch.optim.Adam(model.parameters(), lr=offline_lr)
        self.online_optimizer = torch.optim.Adam(model.parameters(), lr=online_lr)

        self.training_logs = {"offline": [], "online": []}

    def offline_rl_phase(
        self,
        dataset: List[Dict],  # Consists of (image, question, correct_answer)
        reference_model: "MultimodalLLM" = None
    ) -> Dict[str, float]:
        """
        Offline RL: learn from fixed dataset without environment interaction.
        Prioritizes stability over exploration.
        """
        print("Starting Offline RL Phase...")

        metrics = {"loss": [], "reward": []}

        for step in range(self.offline_steps):
            # Sample batch
            batch = self._sample_batch(dataset, batch_size=8)

            # Forward pass
            outputs = self.model(batch["images"], batch["questions"])
            logits = outputs.logits

            # Compute rewards (correctness)
            rewards = self._compute_correctness_reward(
                outputs.sequences,
                batch["answers"]
            )

            # Conservative offline RL loss
            # Use CQL-style objective: Q-value pessimism
            loss = self._compute_offline_loss(logits, rewards)

            # Optimize
            self.offline_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.offline_optimizer.step()

            metrics["loss"].append(loss.item())
            metrics["reward"].append(rewards.mean().item())

            if (step + 1) % 100 == 0:
                avg_loss = sum(metrics["loss"][-100:]) / 100
                avg_reward = sum(metrics["reward"][-100:]) / 100
                print(f"Offline Step {step+1}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")

        self.training_logs["offline"] = metrics
        return {"avg_loss": sum(metrics["loss"]) / len(metrics["loss"]),
                "avg_reward": sum(metrics["reward"]) / len(metrics["reward"])}

    def online_rl_phase(
        self,
        env: "VisualReasoning Env",
        num_episodes: int = 100
    ) -> Dict[str, float]:
        """
        Online RL: interact with environment for refinement.
        Fine-tunes policy on real interactions.
        """
        print("Starting Online RL Phase...")

        metrics = {"episode_reward": [], "episode_length": []}

        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_steps = 0

            for step in range(self.online_steps):
                # Get action from model
                action = self.model.generate(
                    state["image"],
                    state["question"],
                    max_tokens=100
                )

                # Execute in environment
                next_state, reward, done = env.step(action)

                # Store transition
                self._store_transition(state, action, reward, next_state, done)

                # Policy gradient update
                loss = self._compute_online_loss(state, action, reward)

                self.online_optimizer.zero_grad()
                loss.backward()
                self.online_optimizer.step()

                episode_reward += reward
                episode_steps += 1
                state = next_state

                if done:
                    break

            metrics["episode_reward"].append(episode_reward)
            metrics["episode_length"].append(episode_steps)

            if (episode + 1) % 10 == 0:
                avg_reward = sum(metrics["episode_reward"][-10:]) / 10
                print(f"Online Episode {episode+1}: Reward={avg_reward:.2f}")

        self.training_logs["online"] = metrics
        return {"avg_episode_reward": sum(metrics["episode_reward"]) / len(metrics["episode_reward"]),
                "avg_episode_length": sum(metrics["episode_length"]) / len(metrics["episode_length"])}

    def _compute_offline_loss(
        self,
        logits: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """Offline RL loss: conservative policy update."""
        # Log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Policy gradient with reward weighting
        policy_loss = -(log_probs.mean(dim=1) * rewards).mean()

        # Conservative regularization: penalize large deviations
        # (compared to behavioral cloning baseline)
        # In practice: KL divergence from reference policy
        regularization = 0.0  # Placeholder

        return policy_loss + regularization

    def _compute_online_loss(
        self,
        state: Dict,
        action: str,
        reward: float
    ) -> torch.Tensor:
        """Online RL loss: standard policy gradient."""
        # Get model output for state
        outputs = self.model(state["image"], state["question"])
        logits = outputs.logits

        # Log probability of taken action
        action_log_prob = self._get_log_prob_of_action(action, logits)

        # Policy gradient
        loss = -action_log_prob * reward

        return loss

    def _compute_correctness_reward(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> torch.Tensor:
        """Compute reward based on answer correctness."""
        rewards = []
        for pred, target in zip(predictions, targets):
            # Exact match or semantic similarity
            correct = pred.strip() == target.strip()
            reward = 1.0 if correct else -0.1
            rewards.append(reward)

        return torch.tensor(rewards)

    def _sample_batch(self, dataset: List[Dict], batch_size: int):
        """Sample batch from dataset."""
        import random
        batch_samples = random.sample(dataset, min(batch_size, len(dataset)))
        return {
            "images": [s["image"] for s in batch_samples],
            "questions": [s["question"] for s in batch_samples],
            "answers": [s["answer"] for s in batch_samples]
        }

    def _store_transition(self, state, action, reward, next_state, done):
        pass

    def _get_log_prob_of_action(self, action: str, logits: torch.Tensor) -> torch.Tensor:
        pass
```

### 2. Implement Visual Resolution Router

Dynamically adjust image resolution based on complexity:

```python
class VisualResolutionRouter(nn.Module):
    """Adaptively select image resolution for efficiency."""

    def __init__(self, model: "MultimodalLLM", resolution_options: List[int] = [224, 336, 448, 672]):
        super().__init__()
        self.model = model
        self.resolution_options = sorted(resolution_options)

        # Router network: image complexity → resolution
        self.complexity_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(resolution_options))
        )

    def forward(
        self,
        images: torch.Tensor,  # (batch, 3, h, w)
        return_complexity: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Select resolution for each image based on complexity.
        """
        batch_size = images.shape[0]

        # Normalize image sizes
        normalized_images = torch.nn.functional.interpolate(
            images,
            size=(224, 224),  # Compute complexity at fixed size
            mode='bilinear',
            align_corners=False
        )

        # Compute complexity scores
        complexity_logits = self.complexity_encoder(normalized_images)
        complexity_probs = torch.softmax(complexity_logits, dim=-1)
        complexity_scores = torch.max(complexity_probs, dim=-1)[1]  # Argmax

        # Select resolution for each image
        selected_resolutions = [
            self.resolution_options[score.item()] for score in complexity_scores
        ]

        # Resize images to selected resolutions (batch-wise is inefficient, so do sequentially)
        resized_images = []
        for img, res in zip(images, selected_resolutions):
            resized = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(res, res),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            resized_images.append(resized)

        return resized_images, selected_resolutions

    def train_router(
        self,
        train_images: List[torch.Tensor],
        train_quality: List[float],
        num_epochs: int = 10
    ):
        """
        Train router to balance quality and efficiency.
        """
        optimizer = torch.optim.Adam(self.complexity_encoder.parameters(), lr=1e-4)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for image, target_quality in zip(train_images, train_quality):
                # Normalize to 224x224 for complexity encoding
                img_normalized = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(224, 224)
                )

                # Get complexity prediction
                logits = self.complexity_encoder(img_normalized)
                complexity_score = torch.softmax(logits, dim=-1)

                # Actual quality at each resolution (could be pre-computed)
                # For simplicity: assume quality increases with resolution
                expected_quality = complexity_score * torch.tensor([0.7, 0.8, 0.9, 1.0])

                # Loss: match predicted complexity to target
                loss = torch.nn.functional.mse_loss(expected_quality.sum(), torch.tensor(target_quality))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                print(f"Router Training Epoch {epoch+1}: Loss={total_loss/len(train_images):.4f}")
```

### 3. Implement Decoupled Deployment

Separate vision and language models for distributed inference:

```python
class DecoupledInference:
    """Vision and language models on separate GPUs for efficiency."""

    def __init__(
        self,
        vision_model: nn.Module,
        language_model: nn.Module,
        vision_device: str = "cuda:0",
        language_device: str = "cuda:1"
    ):
        self.vision_model = vision_model.to(vision_device)
        self.language_model = language_model.to(language_device)
        self.vision_device = vision_device
        self.language_device = language_device

    def forward(
        self,
        images: torch.Tensor,
        questions: List[str]
    ) -> List[str]:
        """
        Distributed forward pass: vision on GPU0, language on GPU1.
        """
        # Vision processing on vision GPU
        images = images.to(self.vision_device)
        with torch.no_grad():
            vision_embeddings = self.vision_model(images)

        # Transfer to language GPU
        vision_embeddings = vision_embeddings.to(self.language_device)

        # Language processing on language GPU
        language_inputs = self.language_model.encode_multimodal(
            vision_embeddings,
            questions
        )

        # Generate responses
        responses = self.language_model.generate(language_inputs, max_length=100)

        return responses

    def measure_throughput(
        self,
        num_examples: int = 100,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Measure inference throughput with decoupled deployment.
        """
        import time

        start_time = time.time()

        for _ in range(num_examples // batch_size):
            # Generate dummy batch
            dummy_images = torch.randn(batch_size, 3, 224, 224)
            dummy_questions = ["What is in the image?"] * batch_size

            # Forward pass
            _ = self.forward(dummy_images, dummy_questions)

        elapsed = time.time() - start_time
        throughput = num_examples / elapsed

        return {
            "throughput_examples_per_second": throughput,
            "latency_per_example_ms": (elapsed / num_examples) * 1000,
            "total_time_seconds": elapsed
        }
```

## Practical Guidance

### When to Use InternVL3.5 Techniques

- Visual reasoning tasks (MMMU, MathVista)
- Multimodal applications with efficiency constraints
- GUI or document analysis with variable complexity
- Production systems requiring inference optimization
- Models needing both reasoning and speed

### When NOT to Use

- Single-image simple classification
- Models without GPU distribution capability
- Real-time ultra-low-latency systems (<50ms)
- Tasks where resolution is critical and fixed

### Key Hyperparameters

- **offline_rl_steps**: 500-2000
- **online_rl_episodes**: 50-200
- **resolution_options**: [224, 336, 448, 672]
- **cascade_rl_weight**: 0.5 for each phase
- **temperature (offline)**: 0.7-1.0

### Performance Expectations

- Reasoning Improvement: +16.0%
- Inference Speedup: 4.05x
- Quality Preservation: 95%+ accuracy maintained
- GPU Memory: Distributed across devices

## Reference

Researchers. (2024). InternVL3.5: Advancing Open-Source Multimodal Models. arXiv preprint arXiv:2508.18265.
