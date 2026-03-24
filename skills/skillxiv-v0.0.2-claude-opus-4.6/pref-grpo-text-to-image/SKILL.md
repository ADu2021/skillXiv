---
name: pref-grpo-text-to-image
title: Pref-GRPO for Text-to-Image with Preference Rewards
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.20751
keywords: [preference-reward, grpo, text-to-image, stability, generative-models]
description: "Optimize text-to-image models using pairwise preference comparisons instead of pointwise rewards, eliminating normalization instability and reward hacking while providing fine-grained image quality differentiation"
---

# Pref-GRPO: Preference-Based GRPO for Text-to-Image Models

## Core Concept

Pref-GRPO shifts from pointwise reward scoring to pairwise preference-based optimization for text-to-image diffusion models. By comparing images within groups and using win rates as signals rather than absolute normalized scores, the method eliminates the instability caused by score normalization and prevents trivial reward hacking that destabilizes training.

## Architecture Overview

- **Preference Reward Model**: Learns to compare image pairs within a batch, returning relative preferences
- **Win Rate Aggregation**: Computes win rate across image groups as the optimization signal
- **Pairwise Comparison Framework**: Avoids absolute scoring and normalization-induced numerical instability
- **UniGenBench Evaluation**: Comprehensive benchmark with 600 prompts, multiple rating criteria, multimodal LLM assessments

## Implementation Steps

### Stage 1: Build Preference Reward Model

Train a reward model that learns image pair preferences rather than absolute scores.

```python
# Preference reward model using multimodal learning
import torch
from torch import nn

class PreferenceRewardModel(nn.Module):
    """Compare image pairs and predict which is better"""

    def __init__(self, vision_model="clip-vit-large-patch14"):
        super().__init__()
        # Use pretrained vision-language model for understanding
        self.vision_encoder = load_vision_model(vision_model)
        self.preference_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, images_a, images_b, prompts):
        """
        Args:
            images_a, images_b: image tensors [batch_size, 3, height, width]
            prompts: text prompts [batch_size]
        Returns:
            preference_logits: scalar per pair, positive means prefer a
        """
        # Encode images with prompt context
        features_a = self.vision_encoder(images_a, prompts)
        features_b = self.vision_encoder(images_b, prompts)

        # Compute preference logit
        combined = torch.cat([features_a, features_b], dim=-1)
        preference_logit = self.preference_head(combined)

        return preference_logit
```

### Stage 2: Collect Training Data with Preference Labels

Generate candidate images and annotate preferences through human or model-based assessment.

```python
# Collect and label preference data
def collect_preference_data(
    model,
    prompts,
    num_samples_per_prompt=4,
    evaluator="multimodal_llm"
):
    """
    Generate images and label preferences
    """
    preference_data = []

    for prompt in prompts:
        # Generate multiple images per prompt
        images = []
        for i in range(num_samples_per_prompt):
            img = model.generate_image(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            )
            images.append(img)

        # Pairwise comparisons
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                # Evaluate preference (1 if i better, 0 if j better)
                if evaluator == "multimodal_llm":
                    preference = evaluate_with_vlm(
                        images[i], images[j],
                        prompt,
                        criteria=[
                            "semantic_consistency",
                            "visual_quality",
                            "composition"
                        ]
                    )
                else:  # human evaluation
                    preference = human_eval(images[i], images[j], prompt)

                preference_data.append({
                    "prompt": prompt,
                    "image_a": images[i],
                    "image_b": images[j],
                    "preference": preference  # 1 or 0
                })

    return preference_data
```

### Stage 3: Train Preference Reward Model with Preference Loss

Train the reward model to predict image preferences using Bradley-Terry-Luce loss.

```python
# Bradley-Terry-Luce preference learning
def preference_loss(preference_logits, labels):
    """
    BTL loss: maximize log P(a > b) = logit_a - logit_ab
    """
    logit_a, logit_b = preference_logits[:, 0], preference_logits[:, 1]
    # When labels=1, prefer a; when labels=0, prefer b
    log_probs = torch.log_softmax(torch.stack([logit_a, logit_b], dim=1), dim=1)
    loss = -log_probs[range(len(labels)), labels].mean()
    return loss

class PreferenceTrainer:
    def __init__(self, reward_model, lr=1e-4):
        self.model = reward_model
        self.optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)

    def train_step(self, batch):
        """Single training step"""
        images_a = batch["image_a"]
        images_b = batch["image_b"]
        prompts = batch["prompt"]
        preferences = batch["preference"]

        # Forward pass
        preference_logits = self.model(images_a, images_b, prompts)
        preference_logits = torch.stack([
            preference_logits,
            -preference_logits  # Preference for b
        ], dim=1)

        # Compute loss
        loss = preference_loss(preference_logits, preferences)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### Stage 4: Implement Pref-GRPO Optimization

Use the preference reward model with GRPO, computing win rates from pairwise comparisons within batches.

```python
# Pref-GRPO: Group Relative Policy Optimization with preferences
class PrefGRPOOptimizer:
    def __init__(self, diffusion_model, preference_reward_model, lr=1e-5):
        self.model = diffusion_model
        self.reward_model = preference_reward_model
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=lr)

    def compute_win_rates(self, images, prompt, group_size=4):
        """
        Compute win rate for each image relative to others in its group
        """
        # Divide images into groups
        groups = [
            images[i:i+group_size]
            for i in range(0, len(images), group_size)
        ]

        win_rates = []
        for group in groups:
            group_wins = [0] * len(group)

            # Pairwise comparisons within group
            for i in range(len(group)):
                for j in range(len(group)):
                    if i == j:
                        continue

                    # Get preference
                    pref_logit = self.reward_model(
                        group[i].unsqueeze(0),
                        group[j].unsqueeze(0),
                        [prompt]
                    )

                    # Preference probability
                    pref_prob = torch.sigmoid(pref_logit).item()
                    group_wins[i] += pref_prob

            # Normalize by number of comparisons
            group_win_rates = [w / (len(group) - 1) for w in group_wins]
            win_rates.extend(group_win_rates)

        return torch.tensor(win_rates)

    def train_step(self, prompts, num_samples=4):
        """Single GRPO training step using preference rewards"""
        all_images = []
        all_win_rates = []

        for prompt in prompts:
            # Generate image samples with diffusion model
            images = []
            for _ in range(num_samples):
                img = self.model.generate(
                    prompt,
                    num_inference_steps=50
                )
                images.append(img)

            # Compute win rates
            win_rates = self.compute_win_rates(images, prompt)

            all_images.extend(images)
            all_win_rates.extend(win_rates)

        all_win_rates = torch.stack(all_win_rates)

        # Policy gradient: higher win rate = higher gradient
        # (In practice, compute gradient through diffusion process)
        loss = -(all_win_rates.mean())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### Stage 5: Evaluate with UniGenBench

Assess model quality using comprehensive benchmark with semantic and visual criteria.

```python
# UniGenBench evaluation framework
class UniGenBench:
    """
    600 prompts across 5 themes and 20 subthemes
    10 primary + 27 sub-criteria for semantic consistency
    """

    def __init__(self):
        self.benchmark_data = self.load_benchmark()

    def evaluate_model(self, model, criteria=None):
        """Evaluate model on full benchmark"""
        if criteria is None:
            criteria = self.get_default_criteria()

        results = {"scores_by_criterion": {}, "overall_score": 0}

        for criterion in criteria:
            scores = []

            for prompt in self.benchmark_data:
                img = model.generate(prompt)
                score = evaluate_with_vlm(img, prompt, criterion)
                scores.append(score)

            results["scores_by_criterion"][criterion] = {
                "mean": sum(scores) / len(scores),
                "std": calculate_std(scores)
            }

        # Compute overall score
        overall = sum(
            results["scores_by_criterion"][c]["mean"]
            for c in criteria
        ) / len(criteria)
        results["overall_score"] = overall

        return results
```

## Practical Guidance

### Hyperparameters

- **Group Size**: 4 images per prompt for pairwise comparison efficiency
- **GRPO Learning Rate**: 1e-5 for diffusion model updates
- **Preference Model Learning Rate**: 1e-4 for reward model training
- **Num Inference Steps**: 50 for generation quality, 20 for faster iteration
- **Guidance Scale**: 7.5 balances adherence to prompt with image quality

### When to Use

- Text-to-image models exhibiting reward hacking or training instability
- Scenarios with fine-grained quality distinctions requiring nuanced evaluation
- Preference-based human feedback available (easier than point scores)
- Settings where normalization artifacts cause training divergence

### When NOT to Use

- Single-answer tasks where absolute scoring is meaningful
- Compute-constrained environments (pairwise comparisons add overhead)
- Domains where preference data is unavailable or ambiguous
- Real-time generation systems requiring low latency

### Design Considerations

Pref-GRPO eliminates the instability of score normalization by working directly with pairwise comparisons. Minimal score differences between images often get amplified during normalization, creating illusory advantages that lead to reward hacking. By comparing within groups, the method naturally handles these subtle differences while preventing trivial optimization artifacts.

## Reference

Pref-GRPO: Pairwise Preference Reward-based GRPO for T2I RL. arXiv:2508.20751
- https://arxiv.org/abs/2508.20751
