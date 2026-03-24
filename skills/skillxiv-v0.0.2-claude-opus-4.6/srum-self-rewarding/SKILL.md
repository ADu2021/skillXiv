---
name: srum-self-rewarding
title: "SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.12784"
keywords: [multimodal, self-rewarding, image-generation, global-local-rewards, unified-models]
description: "Enable unified multimodal models to self-improve by using their understanding module as evaluator for generation. Apply hierarchical dual rewards: global for overall semantics and local for fine-grained details."
---

# SRUM: Self-Improving Unified Multimodal Models

Unified multimodal models that handle both vision-language understanding and generation often generate lower-quality images than vision-language models optimize for understanding. SRUM enables self-improvement by leveraging the model's own understanding capabilities to evaluate and improve generation quality.

Core insight: strong understanding capability provides free supervision for generation. By using the model's own understanding as reward signal at global and local levels, unified models self-improve without external human feedback or additional models.

## Core Concept

**Understanding-as-Evaluator**: The model's understanding module becomes evaluator for the generation module, providing self-generated rewards without external supervision.

**Hierarchical Dual Rewards**: Two-tier reward system provides feedback at different granularities: global rewards for semantic correctness, local rewards for object-level detail quality.

## Architecture Overview

- **Understanding Module**: Evaluates generated images for semantic alignment
- **Generation Module**: Creates images from text
- **Global Reward Head**: Scores overall semantic correctness and layout
- **Local Reward Head**: Scores object-level details and fine-grained quality
- **RL Optimizer**: Updates generation based on rewards

## Implementation Steps

**Stage 1: Extract Rewards from Understanding Module**

Use understanding module to score generations:

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel

class UnifiedMultimodalModel(nn.Module):
    def __init__(self, model_name='unified-mm-base'):
        super().__init__()

        # Shared transformer backbone
        self.backbone = AutoModel.from_pretrained(model_name)

        # Understanding head: evaluates images
        self.understanding_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1)  # Global quality score
        )

        # Local detail scorer
        self.local_scorer = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1)  # Per-region quality
        )

        # Generation head: creates images
        self.generation_head = ImageDecoder()

    def forward_understanding(self, image):
        """
        Evaluate image quality using understanding module.
        """

        # Encode image
        image_features = self.backbone.encode_image(image)

        # Global reward: semantic correctness
        global_score = self.understanding_head(image_features)

        return global_score

    def forward_generation(self, text):
        """
        Generate image from text.
        """

        # Encode text
        text_features = self.backbone.encode_text(text)

        # Generate image
        generated_image = self.generation_head(text_features)

        return generated_image

class HierarchicalRewardComputer(nn.Module):
    def __init__(self, understanding_model):
        super().__init__()
        self.understanding_model = understanding_model

    def compute_global_reward(self, generated_image, text_prompt):
        """
        Global reward: how well does image match text semantically?
        """

        # Get image understanding
        with torch.no_grad():
            image_features = self.understanding_model.backbone.encode_image(
                generated_image
            )

            global_score = self.understanding_model.understanding_head(
                image_features
            )

        # Get text understanding
        with torch.no_grad():
            text_features = self.understanding_model.backbone.encode_text(
                text_prompt
            )

        # Semantic alignment: cosine similarity
        image_normalized = torch.nn.functional.normalize(
            image_features, dim=-1
        )
        text_normalized = torch.nn.functional.normalize(
            text_features, dim=-1
        )

        semantic_alignment = (
            image_normalized * text_normalized
        ).sum(dim=-1)

        # Global reward combines quality and alignment
        global_reward = 0.5 * global_score + 0.5 * semantic_alignment

        return global_reward

    def compute_local_reward(
        self,
        generated_image,
        text_prompt,
        num_regions=4
    ):
        """
        Local reward: fine-grained object-level quality.
        Divide image into regions and score each independently.
        """

        batch_size, channels, height, width = generated_image.shape

        # Divide image into regions
        region_height = height // num_regions
        region_width = width // num_regions

        local_scores = []

        for i in range(num_regions):
            for j in range(num_regions):
                # Extract region
                h_start = i * region_height
                h_end = (i + 1) * region_height
                w_start = j * region_width
                w_end = (j + 1) * region_width

                region = generated_image[
                    :, :, h_start:h_end, w_start:w_end
                ]

                # Score region with understanding
                with torch.no_grad():
                    region_features = (
                        self.understanding_model.backbone.encode_image(region)
                    )
                    region_score = (
                        self.understanding_model.local_scorer(
                            region_features
                        )
                    )

                local_scores.append(region_score)

        # Average local scores
        local_reward = torch.stack(local_scores).mean()

        return local_reward

    def compute_combined_reward(
        self,
        generated_image,
        text_prompt,
        global_weight=0.6,
        local_weight=0.4
    ):
        """
        Combine global and local rewards.
        """

        global_reward = self.compute_global_reward(
            generated_image,
            text_prompt
        )

        local_reward = self.compute_local_reward(
            generated_image,
            text_prompt
        )

        # Weighted combination
        combined_reward = (
            global_weight * global_reward +
            local_weight * local_reward
        )

        return combined_reward, {
            'global': global_reward.item(),
            'local': local_reward.item()
        }
```

**Stage 2: RL Training Loop**

Train generation module with self-rewards:

```python
def srum_training_loop(
    unified_model,
    text_image_pairs,
    num_epochs=5,
    batch_size=32
):
    """
    Train unified model with self-rewarding.
    """

    optimizer = torch.optim.AdamW(
        unified_model.generation_head.parameters(),
        lr=1e-4
    )

    reward_computer = HierarchicalRewardComputer(unified_model)

    for epoch in range(num_epochs):
        for batch_idx, (texts, images) in enumerate(
            dataloader(text_image_pairs, batch_size)
        ):
            # Generate images from text
            generated_images = unified_model.forward_generation(texts)

            # Compute self-rewards
            rewards_list = []
            loss_list = []

            for gen_img, text, real_img in zip(
                generated_images, texts, images
            ):
                # Self-reward from understanding module
                combined_reward, reward_components = (
                    reward_computer.compute_combined_reward(
                        gen_img.unsqueeze(0),
                        text
                    )
                )

                # Also compute reconstruction loss to reference image
                recon_loss = torch.nn.functional.mse_loss(
                    gen_img, real_img
                )

                rewards_list.append(combined_reward)
                loss_list.append(recon_loss)

            # Policy gradient update
            rewards_tensor = torch.stack(rewards_list)
            loss_tensor = torch.stack(loss_list)

            # Combine RL loss and reconstruction loss
            rl_loss = -(
                rewards_tensor.detach() *
                torch.nn.functional.log_softmax(
                    torch.randn_like(generated_images), dim=1
                )
            ).mean()

            total_loss = loss_tensor.mean() + 0.1 * rl_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                avg_reward = rewards_tensor.mean().item()
                avg_loss = loss_tensor.mean().item()
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Reward: {avg_reward:.4f}, Loss: {avg_loss:.4f}"
                )

    return unified_model
```

**Stage 3: Inference with Quality Guidance**

Generate with quality feedback:

```python
def generate_with_quality_feedback(
    unified_model,
    text_prompt,
    num_iterations=3
):
    """
    Generate image with iterative quality refinement.
    """

    reward_computer = HierarchicalRewardComputer(unified_model)

    best_image = None
    best_reward = -float('inf')

    for iteration in range(num_iterations):
        # Generate initial image
        generated_image = unified_model.forward_generation(
            text_prompt
        )

        # Compute rewards
        combined_reward, components = (
            reward_computer.compute_combined_reward(
                generated_image,
                text_prompt
            )
        )

        # Track best
        if combined_reward > best_reward:
            best_reward = combined_reward
            best_image = generated_image

        # Log quality components
        print(
            f"Iteration {iteration}: "
            f"Global={components['global']:.3f}, "
            f"Local={components['local']:.3f}"
        )

    return best_image
```

## Practical Guidance

**When to Use SRUM:**
- Unified multimodal models handling both understanding and generation
- When understanding quality is significantly higher than generation
- No access to external reward models or human feedback

**When NOT to Use:**
- Separate understanding-only or generation-only models
- When generation quality already matches understanding quality
- Tasks requiring diverse generation (local reward may reduce diversity)

**Reward Weight Strategies:**

| Scenario | Global Weight | Local Weight | Rationale |
|----------|---------------|--------------|-----------|
| High-level tasks | 0.8 | 0.2 | Semantic correctness paramount |
| Detail-focused | 0.4 | 0.6 | Fine details critical |
| Balanced | 0.6 | 0.4 | Standard setting |

**Common Pitfalls:**
- Local reward too strong (overly conservative details)
- Global reward misaligned with human preference (semantic errors remain)
- Region division too fine (noisy local scores)
- Not validating understanding module gives useful feedback

**Improvement Metrics:**

| Metric | Before SRUM | After SRUM |
|--------|------------|-----------|
| T2I-CompBench | 82.18 | 88.37 |
| T2I-ReasonBench | 43.82 | 46.75 |
| FID Score | Baseline | ~5% improvement |

## Reference

Based on the research at: https://arxiv.org/abs/2510.12784
