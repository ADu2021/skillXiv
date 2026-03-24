---
name: listener-rewarded-vlm-preferences
title: "Listener-Rewarded Thinking in VLMs for Image Preferences"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.22832"
keywords: [RewardModeling, VisionLanguageModels, ChainOfThought, HumanPreferences, GRPO]
description: "Aligns reasoning traces with final decisions in preference models using an independent frozen VLM as listener. Achieves 67.4% accuracy on ImageReward by enforcing consistency between explanations and choices. Use when training reward models for image generation quality where both reasoning quality and accuracy matter."
---

# Listener-Rewarded Thinking: Enforcing Consistency in Visual Preference Reasoning

Reward models for predicting human visual preferences often produce chain-of-thought explanations that contradict their final decisions—a critical failure mode where the model cannot convincingly justify its preference judgment. This misalignment hurts out-of-distribution generalization and indicates the reasoning isn't actually driving the decisions. Listener-Rewarded Thinking solves this by training with an independent frozen vision-language model as a "listener" that evaluates whether the reasoning explanation would convince it to make the same choice. This creates a soft reward signal penalizing contradictory reasoning while reinforcing coherent judgments.

The insight is that reasoning quality matters not just for human interpretability but for model generalization. Models forced to produce convincing explanations develop more robust preference criteria, transferring better to new visual distributions.

## Core Concept

Listener-Rewarded Thinking combines Group Relative Policy Optimization (GRPO) with a novel listener-augmented reward function. Rather than simply optimizing for correct preference predictions, the framework introduces three reward components:

1. **Formatting Reward**: Encourages well-structured chain-of-thought responses
2. **Accuracy Reward**: Standard correctness signal (chosen image matches ground truth)
3. **Listener Reward**: Independent frozen VLM's confidence that the reasoning supports the conclusion

The combined reward is weighted: r_total = r_fmt + 0.5·r_acc + 0.5·r_list. This architecture ensures both the reasoning traces and final decisions improve together.

## Architecture Overview

- **Base Reasoning Model**: Vision-language model generating chain-of-thought preference explanations
- **Frozen Listener VLM**: Independent instruction-tuned model evaluating reasoning quality (not trained)
- **GRPO Training Framework**: Policy optimization using group-relative comparisons
- **Listener Disagreement Detection**: Identifies when reasoning contradicts independent evaluation
- **Three-Component Reward**: Formats + accuracy + listener agreement

## Implementation

Listener reward computation evaluates reasoning consistency:

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoTokenizer, AutoModelForCausalLM

class ListenerAugmentedRewardModel(nn.Module):
    """
    Combines reasoning model with frozen listener for reward computation.
    Ensures explanations convince an independent judge.
    """
    def __init__(self, model_name='qwen-vl-7b', listener_name='qwen-vl-7b-instruct'):
        super().__init__()

        # Reasoning model: trained to improve preferences + explanations
        self.reasoning_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Frozen listener: evaluates explanation quality
        self.listener_model = AutoModelForCausalLM.from_pretrained(listener_name)
        for param in self.listener_model.parameters():
            param.requires_grad = False  # Frozen

        self.listener_tokenizer = AutoTokenizer.from_pretrained(listener_name)

    def forward(self, image_pair, preference_idx, reasoning_text):
        """
        Generate preference explanation and compute listener reward.

        Args:
            image_pair: (img0, img1) tuple
            preference_idx: 0 or 1, which image is preferred
            reasoning_text: chain-of-thought explanation from reasoning model

        Returns:
            rewards: dict with r_fmt, r_acc, r_list components
        """
        # Step 1: Evaluate formatting reward (well-structured explanation)
        r_fmt = self._compute_formatting_reward(reasoning_text)

        # Step 2: Evaluate accuracy reward (correct preference)
        r_acc = self._compute_accuracy_reward(
            image_pair, preference_idx, reasoning_text
        )

        # Step 3: Evaluate listener reward (listener agrees with reasoning)
        r_list = self._compute_listener_reward(
            image_pair, preference_idx, reasoning_text
        )

        return {
            'r_fmt': r_fmt,
            'r_acc': r_acc,
            'r_list': r_list,
            'total': r_fmt + 0.5 * r_acc + 0.5 * r_list
        }

    def _compute_formatting_reward(self, reasoning_text):
        """Reward well-structured chain-of-thought."""
        # Check for presence of key reasoning markers
        markers = ['therefore', 'because', 'this means', 'as a result']
        num_markers = sum(1 for m in markers if m in reasoning_text.lower())

        # Length reward: longer reasoning (200-500 tokens) better than too short
        text_length = len(self.tokenizer.encode(reasoning_text))
        length_score = min(1.0, (text_length - 50) / 450)  # Normalized to [0,1]

        r_fmt = (0.3 * (num_markers / len(markers))) + (0.7 * length_score)
        return torch.tensor(r_fmt, dtype=torch.float32)

    def _compute_accuracy_reward(self, image_pair, preference_idx, reasoning_text):
        """Reward correct preference prediction."""
        # Extract predicted preference from reasoning
        predicted_idx = self._extract_preference_from_text(reasoning_text)

        # Accuracy: 1.0 if correct, 0.0 if wrong
        if predicted_idx == preference_idx:
            r_acc = 1.0
        else:
            r_acc = 0.0

        return torch.tensor(r_acc, dtype=torch.float32)

    def _compute_listener_reward(self, image_pair, preference_idx, reasoning_text):
        """
        Frozen listener evaluates: "Does this explanation justify this preference?"
        High confidence = high reward.
        """
        # Create prompt for listener
        listener_prompt = f"""
Given these two images:
[Image 0] and [Image 1]

The reasoning model explains: "{reasoning_text}"

Based on this explanation, which image is preferred?
Respond with only the image number (0 or 1) and confidence.
"""

        # Get listener response
        listener_inputs = self.listener_tokenizer(
            listener_prompt, return_tensors='pt'
        )
        with torch.no_grad():
            listener_output = self.listener_model.generate(
                **listener_inputs,
                max_length=50,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Parse listener's prediction
        listener_text = self.listener_tokenizer.decode(
            listener_output.sequences[0], skip_special_tokens=True
        )
        listener_predicted_idx = self._extract_preference_from_text(listener_text)

        # Confidence extraction (simplified)
        confidence = self._extract_confidence(listener_text)

        # Listener reward: confidence × correctness
        # High reward if listener agrees with reasoning
        listener_agrees = (listener_predicted_idx == preference_idx)
        r_list = confidence if listener_agrees else (1.0 - confidence)

        return torch.tensor(r_list, dtype=torch.float32)

    def _extract_preference_from_text(self, text):
        """Extract preference index (0 or 1) from text."""
        # Heuristic: look for "image 0" or "image 1"
        if 'image 0' in text.lower() or 'first' in text.lower():
            return 0
        elif 'image 1' in text.lower() or 'second' in text.lower():
            return 1
        else:
            return -1  # Ambiguous

    def _extract_confidence(self, text):
        """Extract confidence score from listener response."""
        # Heuristic: look for percentage or confidence keywords
        if 'high confidence' in text.lower() or 'definitely' in text.lower():
            return 0.8
        elif 'medium' in text.lower() or 'likely' in text.lower():
            return 0.5
        elif 'low confidence' in text.lower():
            return 0.2
        else:
            return 0.5  # Default


class GRPOTrainer:
    """
    Group Relative Policy Optimization with listener-augmented rewards.
    Trains reasoning model to maximize total reward.
    """
    def __init__(self, model, reward_model, learning_rate=1e-5):
        self.model = model
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate
        )

    def train_step(self, image_pairs, preferences, num_groups=4):
        """
        GRPO training: optimize relative to group performance.

        Args:
            image_pairs: List of (img0, img1) tuples
            preferences: List of preference indices
            num_groups: Split into groups for relative comparison
        """
        batch_size = len(image_pairs)
        group_size = batch_size // num_groups

        total_loss = 0

        for group_idx in range(num_groups):
            # Get group samples
            start = group_idx * group_size
            end = start + group_size

            group_pairs = image_pairs[start:end]
            group_prefs = preferences[start:end]

            # Generate reasoning for each sample
            reasoning_texts = []
            for pair in group_pairs:
                reasoning = self.model.generate_reasoning(pair)
                reasoning_texts.append(reasoning)

            # Compute listener-augmented rewards
            group_rewards = []
            for pair, pref, text in zip(group_pairs, group_prefs, reasoning_texts):
                reward_dict = self.reward_model(pair, pref, text)
                group_rewards.append(reward_dict['total'])

            group_rewards = torch.stack(group_rewards)

            # Compute group baseline (for relative comparison)
            group_baseline = group_rewards.mean()

            # Policy gradient: optimize relative advantage
            for idx, (pair, pref, text) in enumerate(
                zip(group_pairs, group_prefs, reasoning_texts)
            ):
                advantage = group_rewards[idx] - group_baseline

                # Log probability of generating this reasoning
                log_prob = self.model.compute_log_prob(text, pair)

                # Loss: negative advantage-weighted log probability
                loss = -log_prob * advantage.detach()
                total_loss += loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

Training loop demonstrating listener-augmented learning:

```python
def train_with_listener_reward(
    reasoning_model, reward_model, train_dataset, num_epochs=10
):
    """
    Train reasoning model with listener-augmented rewards.
    Ensures explanations are convincing and accurate.
    """
    trainer = GRPOTrainer(reasoning_model, reward_model)
    dataset_size = len(train_dataset)

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx in range(0, dataset_size, 32):
            batch_end = min(batch_idx + 32, dataset_size)
            batch = train_dataset[batch_idx:batch_end]

            image_pairs = [item['image_pair'] for item in batch]
            preferences = [item['preference'] for item in batch]

            loss = trainer.train_step(image_pairs, preferences)
            epoch_loss += loss
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"Batch {num_batches}: Loss = {loss:.4f}")

        print(f"Epoch {epoch} avg loss: {epoch_loss / num_batches:.4f}")

    return reasoning_model
```

## Practical Guidance

| Aspect | Value | Notes |
|--------|-------|-------|
| Accuracy on ImageReward | 67.4% | State-of-the-art for preference prediction |
| Listener Disagreement Reduction | 8.3% vs 10.1% baseline | Fewer contradictory reasoning events |
| Data Efficiency | 16% of HPSv2 | Training on reduced dataset |
| Generalization Improvement | +6% on 1.2M vote dataset | Better out-of-distribution performance |
| Reward Components | 3 (format + accuracy + listener) | Balanced weighting important |
| Listener Model Freeze | Yes | Prevents shifting evaluation target |

**When to use:**
- Training reward models for image generation quality assessment
- Situations where explanations need to justify decisions (interpretability matters)
- Improving out-of-distribution generalization for preference models
- Reducing contradictory reasoning that hurts user trust
- Data-efficient training on limited preference annotations

**When NOT to use:**
- If explanation quality is irrelevant (pure accuracy sufficient)
- Scenarios without access to a good frozen listener model
- Real-time systems where listener inference adds latency
- Tasks where accuracy and explanation quality are decoupled
- Binary preference classification without reasoning requirements

**Common pitfalls:**
- Listener model too weak, unable to evaluate reasoning quality properly
- Reward weighting unbalanced (all three components contribute equally doesn't always work)
- Frozen listener becoming stale if base reasoning model diverges significantly
- Listener disagreement treated as always negative (sometimes model has valid alternative reasoning)
- Not accounting for listener's own biases (independent model still has preferences)
- Training data distribution mismatch between reasoning and listener training

## Reference

"Listener-Rewarded Thinking in VLMs for Image Preferences", 2025. [arxiv.org/abs/2506.22832](https://arxiv.org/abs/2506.22832)
