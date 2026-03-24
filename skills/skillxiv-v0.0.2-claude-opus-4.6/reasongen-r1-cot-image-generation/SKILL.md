---
name: reasongen-r1-cot-image-generation
title: "ReasonGen-R1: CoT for Autoregressive Image generation models through SFT and RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24875"
keywords: [CoT, Image Generation, SFT, RL, Reasoning]
description: "Enable image generators to reason explicitly through text before creating images using supervised fine-tuning and reinforcement learning optimization."
---

# ReasonGen-R1: Teach Images to Think Before They Draw

Most image generation models leap directly from text prompts to pixel outputs, skipping the planning stage humans naturally use. ReasonGen-R1 injects explicit reasoning into autoregressive image generators by having them write intermediate rationales describing object layouts, styles, and compositions before image generation begins. This two-stage approach combines supervised fine-tuning on reasoning examples with policy optimization to significantly improve visual quality and semantic alignment.

The core insight is that forcing explicit planning—"I'll place the main subject left-of-center in cool tones on a gradient background"—helps the model organize complex scenes and resolve ambiguities before pixels are committed. This reasoning acts as a bottleneck that concentrates the model's limited planning capacity.

## Core Concept

Chain-of-Thought reasoning is well-established for language tasks but rarely applied to generation. ReasonGen-R1 treats image generation as a two-phase process: first the model writes a structured text plan describing what it will generate, then it generates images conditioned on that plan. A pre-trained vision-language model scores the visual output, creating reward signals that guide reinforcement learning. This decouples spatial reasoning (where things go) from pixel synthesis (how to render them).

## Architecture Overview

- **Reasoning Module**: Autoregressive text generation that produces detailed visual plans—covering composition, lighting, objects, styles before any images are drawn
- **Data Collection Pipeline**: Synthetic generation of training triplets (visual prompts → reasoning rationales → target images) using model-crafted examples
- **Supervised Fine-Tuning Stage**: Initial training on collected reasoning examples to establish the reasoning-to-image mapping
- **Reward Model**: Vision-language model that evaluates image quality, semantic alignment with prompts, and layout coherence
- **GRPO Optimization**: Group Relative Policy Optimization refines the combined reasoning+generation pipeline to maximize image quality rewards

## Implementation

This implementation shows the core reasoning-then-generate pattern with a simplified reward mechanism.

First, prepare training data with reasoning annotations. The dataset maps prompts to reasoning traces and target images:

```python
import json
from transformers import AutoTokenizer

def prepare_reasoning_dataset(prompt_image_pairs):
    """Create training triplets of (prompt, reasoning, image) for CoT-augmented image generation."""
    dataset = []
    for prompt, target_image in prompt_image_pairs:
        # In practice, these rationales come from manual annotation or teacher model
        reasoning = generate_reasoning_rationale(prompt)
        dataset.append({
            "prompt": prompt,
            "reasoning": reasoning,
            "target_image": target_image,
            "combined_input": f"{prompt}\nReasoning: {reasoning}"
        })
    return dataset

def generate_reasoning_rationale(prompt):
    """Generate structured reasoning about what the image will contain."""
    # Placeholder - in real systems this uses a teacher model or manual labels
    return f"I will create an image with {prompt} with careful composition and lighting."

# Example usage
prompts_with_images = [
    ("A red apple on a wooden table", "path/to/image.jpg"),
    ("A sunset over mountains", "path/to/image2.jpg"),
]
dataset = prepare_reasoning_dataset(prompts_with_images)
```

Next, implement the SFT stage to condition the image generator on reasoning:

```python
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM
import torch

class ReasoningImageDataset(Dataset):
    def __init__(self, data, tokenizer, max_reasoning_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_reasoning_length = max_reasoning_length

    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenize combined input: prompt + reasoning
        input_ids = self.tokenizer(
            item["combined_input"],
            max_length=self.max_reasoning_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        return {"input_ids": input_ids}

    def __len__(self):
        return len(self.data)

# Setup SFT training
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
dataset = ReasoningImageDataset(training_data, tokenizer)
loader = DataLoader(dataset, batch_size=8)

# SFT loss: standard language modeling on reasoning text
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in loader:
    outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Implement the reward model that scores generated images for alignment and quality:

```python
import torch.nn.functional as F
from PIL import Image

class ImageRewardModel:
    """Vision-language model that scores image quality and semantic alignment."""

    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        from transformers import CLIPProcessor, CLIPModel
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def score_image(self, image, prompt, reasoning):
        """Score how well image matches prompt and reasoning."""
        # Prepare inputs
        combined_text = f"{prompt}. Specifically: {reasoning}"
        inputs = self.processor(
            text=[combined_text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # Compute alignment score
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            alignment_score = F.softmax(logits_per_image, dim=1)[0, 0].item()

        return alignment_score

# Usage
reward_model = ImageRewardModel()
test_image = Image.open("generated.jpg")
prompt = "A red apple on a wooden table"
reasoning = "I will create a centered red apple on natural wood with warm lighting."
score = reward_model.score_image(test_image, prompt, reasoning)
```

Finally, implement the GRPO refinement phase where the model learns to generate better reasoning:

```python
class GRPOTrainer:
    """Group Relative Policy Optimization for image generation."""

    def __init__(self, model, reward_model, lr=5e-5):
        self.model = model
        self.reward_model = reward_model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def compute_grpo_loss(self, prompts, generated_images, reasonings, group_size=4):
        """
        Compute GRPO loss by comparing grouped samples.
        Within each group, upweight high-reward samples and downweight low-reward.
        """
        losses = []
        for i in range(0, len(prompts), group_size):
            batch_prompts = prompts[i:i+group_size]
            batch_images = generated_images[i:i+group_size]
            batch_reasonings = reasonings[i:i+group_size]

            # Score all images in group
            scores = [
                self.reward_model.score_image(img, p, r)
                for img, p, r in zip(batch_images, batch_prompts, batch_reasonings)
            ]
            scores = torch.tensor(scores)

            # Normalize scores within group for relative comparison
            normalized_scores = (scores - scores.mean()) / (scores.std() + 1e-8)

            # Update model to increase probability of high-scoring reasonings
            for j, (prompt, reasoning) in enumerate(zip(batch_prompts, batch_reasonings)):
                input_text = f"{prompt}\nReasoning: {reasoning}"
                outputs = self.model(input_text, output_scores=True)
                log_prob = outputs["log_prob"]
                loss = -log_prob * normalized_scores[j]
                losses.append(loss)

        return sum(losses) / len(losses)

# Training loop for GRPO phase
trainer = GRPOTrainer(model, reward_model)
for epoch in range(5):
    for batch_prompts, batch_images, batch_reasonings in rl_dataloader:
        loss = trainer.compute_grpo_loss(batch_prompts, batch_images, batch_reasonings)
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
```

## Practical Guidance

| Aspect | Details |
|--------|---------|
| **SFT Dataset Size** | 10k-50k reasoning-image pairs; start with 10k and expand if validation loss plateaus |
| **GRPO Batch Groups** | 4-8 samples per group; larger groups provide better relative comparisons but increase computation |
| **Reasoning Length** | Target 100-256 tokens; longer reasoning adds planning capacity but delays image generation |
| **Reward Model** | Use CLIP or similar vision-language model; finetune on your domain if standard model shows poor correlation |
| **Training Stability** | Monitor reward signal divergence; clip normalized scores to [-2, 2] if rewards become extreme |

**When to Use:**
- Complex scenes with multiple objects or specific spatial arrangements
- Tasks where object placement and composition matter (interior design, product visualization)
- When users benefit from understanding the model's planning process
- Domains with strong visual-semantic alignment requirements

**When NOT to Use:**
- Simple 1-2 object generations where spatial reasoning adds minimal value
- Real-time or latency-critical applications where doubling generation steps is unaffordable
- Tasks dominated by style/aesthetic over composition (abstract art, texture synthesis)
- When computational resources are severely constrained

**Common Pitfalls:**
- Reasoning dataset with poor quality rationales hurts downstream performance—validate annotations carefully
- Reward model misalignment: if CLIP scores don't correlate with human quality, the RL phase will optimize the wrong objective
- Overfitting to reward model: monitor generation diversity; add diversity penalties if samples become homogeneous
- Training instability in GRPO: use gradient clipping and smaller learning rates if loss becomes erratic

## Reference

ReasonGen-R1: CoT for Autoregressive Image generation models through SFT and RL
https://arxiv.org/abs/2505.24875
