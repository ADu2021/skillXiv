---
name: mimo-vl-vision-language
title: "MiMo-VL: Mixed On-Policy Reinforcement Learning for Vision-Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.03569"
keywords: [vision-language, multimodal, reinforcement-learning, reasoning, chain-of-thought]
description: "Build state-of-the-art 7B multimodal models by combining four-stage vision-language pretraining with mixed on-policy RL integrating verifiable and human feedback rewards."
---

# MiMo-VL Technical Report

## Core Concept

MiMo-VL demonstrates that 7B vision-language models can achieve state-of-the-art results across 50+ benchmarks through strategic data integration and mixed on-policy reinforcement learning. The key insight: incorporating synthetic reasoning data with long chain-of-thought during pretraining—rather than treating it as supplementary fine-tuning—substantially improves performance without saturation effects.

## Architecture Overview

- **Vision Encoder**: Qwen2.5-ViT transformer backbone processing visual input
- **Projection Layer**: MLP projector aligning visual and linguistic embeddings
- **Language Model**: MiMo-7B base language model for token generation
- **Training Pipeline**: Four sequential stages optimizing different modality aspects
  - Stage 1-2: Projector warmup and vision-language alignment
  - Stage 3: General multimodal pretraining across diverse data types
  - Stage 4: Long-context SFT integrating reasoning and extended sequences
  - Post-training: Mixed on-policy RL with multiple reward signals
- **Mixed Reward System**: Combines verifiable rewards (reasoning correctness, grounding accuracy, counting validity) with human preference alignment

## Implementation

### Step 1: Prepare Diverse Pretraining Data

```python
# Data composition for 2.4 trillion tokens
data_recipes = {
    "captions": 0.3,           # Image-text pairs
    "interleaved_content": 0.2, # Mixed visual-textual sequences
    "ocr": 0.15,               # Text detection and recognition
    "grounding": 0.15,         # Visual region-text alignment
    "video": 0.1,              # Temporal visual understanding
    "gui": 0.05,               # Interface interaction
    "reasoning": 0.05          # Synthetic CoT with long chains
}

# Quality-aware data loader
class QualityAwareDataLoader:
    def __init__(self, recipes, quality_threshold=0.85):
        self.recipes = recipes
        self.quality_threshold = quality_threshold

    def validate_sample(self, sample):
        # Check image-text alignment scores
        # Verify OCR accuracy
        # Validate reasoning chain coherence
        return sample['quality_score'] >= self.quality_threshold
```

### Step 2: Configure Four-Stage Pretraining

```python
# Stage 1-2: Projector warmup and alignment
warmup_config = {
    "stages": [1, 2],
    "duration_tokens": 50e9,
    "learning_rate": 1e-3,
    "focus": ["vision_encoder_freeze", "projector_train"],
}

# Stage 3: General multimodal pretraining
general_config = {
    "stage": 3,
    "duration_tokens": 1.5e12,
    "learning_rate": 1e-4,
    "data_mix": data_recipes,
    "unfrozen_components": ["all"],
}

# Stage 4: Reasoning and long-context SFT
sft_config = {
    "stage": 4,
    "duration_tokens": 0.8e12,
    "learning_rate": 2e-5,
    "focus": ["chain_of_thought", "long_context"],
    "reasoning_data_weight": 0.3,
}
```

### Step 3: Implement Mixed On-Policy RL

```python
# Multiple reward functions addressing different capabilities
class MultiRewardEvaluator:
    def __init__(self):
        self.reasoning_verifier = ReasoningChecker()
        self.grounding_validator = GroundingValidator()
        self.counting_checker = CountingVerifier()

    def compute_rewards(self, output, ground_truth):
        rewards = {}

        # Verifiable reward: reasoning correctness
        reasoning_score = self.reasoning_verifier.check(
            output['reasoning_chain'],
            ground_truth['correct_answer']
        )
        rewards['reasoning'] = reasoning_score

        # Verifiable reward: grounding accuracy
        grounding_score = self.grounding_validator.evaluate(
            output['bounding_boxes'],
            ground_truth['ground_regions']
        )
        rewards['grounding'] = grounding_score

        # Verifiable reward: counting precision
        counting_score = self.counting_checker.verify(
            output['count'],
            ground_truth['true_count']
        )
        rewards['counting'] = counting_score

        # Human preference alignment (from human feedback data)
        human_score = output['human_preference_signal']
        rewards['human'] = human_score

        # Combined reward with adaptive weighting
        combined = (
            0.4 * reasoning_score +
            0.3 * grounding_score +
            0.2 * counting_score +
            0.1 * human_score
        )

        return combined, rewards

# Mixed on-policy training loop
def mixed_onpolicy_rl_step(model, batch, reward_fn, optimizer):
    # Generate diverse outputs for same input
    outputs = [model.generate(batch['input'], temp=t)
               for t in [0.7, 1.0, 1.3]]

    # Compute multi-source rewards
    rewards = [reward_fn(out, batch['ground_truth'])
               for out in outputs]

    # Policy gradient update with advantage normalization
    advantages = rewards - np.mean(rewards)
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # Compute policy gradient loss
    log_probs = [model.log_probability(out) for out in outputs]
    pg_loss = sum(-lp * adv for lp, adv in zip(log_probs, advantages))

    optimizer.zero_grad()
    pg_loss.backward()
    optimizer.step()

    return pg_loss.item()
```

### Step 4: Handle Response Length Disparities

```python
# Reasoning tasks favor longer outputs; grounding tasks prefer shorter
class AdaptiveResponseLengthTraining:
    def __init__(self):
        self.task_length_prefs = {
            "reasoning": {"min": 500, "target": 1200},
            "grounding": {"min": 50, "target": 200},
            "counting": {"min": 20, "target": 80},
        }

    def compute_length_adjusted_loss(self, output, task_type, rewards):
        output_length = len(output.split())
        target_range = self.task_length_prefs[task_type]

        # Soft penalty for length deviation
        ideal_length = target_range['target']
        length_penalty = ((output_length - ideal_length) / 100) ** 2

        # Adjust reward based on task-specific expectations
        adjusted_reward = rewards - 0.05 * length_penalty

        return adjusted_reward
```

## Practical Guidance

1. **Data Integration Priority**: Incorporate reasoning data directly into pretraining stages rather than treating as post-hoc fine-tuning. The 2.4 trillion token budget should allocate 5-10% to synthetic reasoning chains.

2. **Training Curriculum**: Follow the four-stage progression strictly. Early stages establish vision-language alignment; later stages build reasoning capacity without degrading perceptual skills.

3. **Reward Signal Weighting**: Initialize with equal weights across reward types, then adjust based on evaluation performance. Reasoning correctness typically improves fastest initially.

4. **Addressing Task Interference**: Use separate decoder heads or task-specific LoRA adapters to handle the tension between long-output reasoning tasks and short-output grounding tasks.

5. **Evaluation at Scale**: Benchmark across 50+ tasks spanning perception, reasoning, and grounding rather than focusing on narrow metrics. MiMo-VL's strength is cross-domain generalization.

## Reference

- Paper: MiMo-VL Technical Report (2506.03569)
- Key Architecture: Qwen2.5-VL foundation with enhanced RL pipeline
- Benchmark Suite: 50+ standardized evaluations with reproducible prompts
- Public Release: MiMo-VL-7B-SFT and MiMo-VL-7B-RL models and datasets
