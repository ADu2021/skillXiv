---
name: latent-sketchpad-visual-reasoning
title: "Latent Sketchpad: Sketching Visual Thoughts to Elicit Multimodal Reasoning in MLLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.24514"
keywords: [Visual Reasoning, Multimodal, Internal Scratchpad, Sketch Generation]
description: "Enables multimodal reasoning by interleaving visual sketches with text. MLLMs generate latent visual representations during reasoning, with sketch decoder converting them to human-interpretable images. Improves reasoning performance while maintaining interpretability through visual thinking aids."
---

# Latent Sketchpad: Internal Visual Reasoning for Multimodal Models

Text-only reasoning in multimodal models wastes visual capabilities. Latent Sketchpad enables models to sketch during reasoning—generating internal visual representations that improve problem-solving while remaining interpretable.

The approach extends textual thinking to include visual-spatial reasoning, treating sketches as thought aids for complex tasks.

## Core Concept

Key insight: **models can reason better when able to externalize visual thinking**, enabling:
- Interleaved text-visual reasoning
- Internal sketch generation for spatial problems (mazes, geometry)
- Sketch decoder makes thinking interpretable
- Improved performance without architectural changes

## Architecture Overview

- Multimodal backbone unchanged
- Context-aware vision head: generates latent sketches autoregressively
- Sketch decoder: converts latents to interpretable images
- Token-level integration: sketches appear naturally in reasoning

## Implementation Steps

Implement sketch generation as additional output head that produces visual latents during decoding:

```python
class LatentSketchpad(nn.Module):
    def __init__(self, model_dim=768, sketch_dim=256, h=32, w=32):
        super().__init__()
        self.model_dim = model_dim
        self.sketch_dim = sketch_dim
        self.h, self.w = h, w

        # Context-aware vision head
        self.sketch_generator = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, sketch_dim * h * w)
        )

        # Sketch decoder: latent -> interpretable image
        self.sketch_decoder = nn.Sequential(
            nn.Linear(sketch_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # RGB
        )

    def generate_sketch(self, hidden_state):
        """Generate latent sketch from hidden state."""
        sketch_logits = self.sketch_generator(hidden_state)
        sketch_latent = sketch_logits.reshape(
            -1, self.h, self.w, self.sketch_dim
        )
        return sketch_latent

    def decode_sketch(self, sketch_latent):
        """Decode latent sketch to image."""
        batch_size, h, w, d = sketch_latent.shape
        sketch_flat = sketch_latent.reshape(-1, d)
        pixels = self.sketch_decoder(sketch_flat)
        return pixels.reshape(batch_size, h, w, 3)
```

Integrate sketching into the reasoning loop. During generation, optionally output sketches:

```python
def generate_with_sketching(model, prompt, max_tokens=500, sketch_every=50):
    """Generate text with interleaved sketch generation."""
    outputs = {'text': [], 'sketches': []}
    hidden_states = []

    input_ids = model.tokenize(prompt)

    for step in range(max_tokens):
        # Forward through model
        logits, hidden = model.forward_with_hidden(input_ids)

        # Sample next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        outputs['text'].append(model.decode(next_token))
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        # Generate sketch every N tokens
        if step % sketch_every == 0 and step > 0:
            sketch_latent = model.sketch_generator(hidden[:, -1, :])
            sketch_image = model.sketch_decoder(sketch_latent)
            outputs['sketches'].append({
                'token': step,
                'image': sketch_image.detach()
            })

        hidden_states.append(hidden)

    return outputs
```

Train with combined text and sketch objectives. The sketch should improve reasoning fidelity:

```python
def train_with_sketching(model, trajectories, sketch_weight=0.3):
    """Train model to reason with sketching."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for trajectory in trajectories:
        prompt = trajectory['prompt']
        reasoning_path = trajectory['reasoning']
        sketches = trajectory['sketches']  # Optional ground truth sketches

        # Generate with sketching
        output = generate_with_sketching(model, prompt)

        # Text loss (standard LM loss)
        text_tokens = model.tokenize(reasoning_path)
        text_logits = model(text_tokens)[0]
        text_loss = torch.nn.functional.cross_entropy(text_logits, text_tokens)

        # Sketch loss (if ground truth sketches available)
        sketch_loss = 0
        if sketches:
            for sketch_pred, sketch_truth in zip(output['sketches'], sketches):
                sketch_loss += torch.nn.functional.mse_loss(
                    sketch_pred['image'], sketch_truth
                )

        total_loss = text_loss + sketch_weight * sketch_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Sketch generation frequency | Every 50-100 tokens |
| Sketch resolution | 32x32 or 64x64 (balance quality and memory) |
| Sketch dimension | 256-512 |
| Loss weight | 0.3 (text-focused) |

**When to use:**
- Spatial/geometric reasoning tasks (mazes, diagrams)
- Interpretability requirements (see what model thinks)
- Multimodal applications needing visual feedback
- Educational or debugging interfaces

**When NOT to use:**
- Pure text tasks (sketching overhead)
- Real-time latency constraints
- Tasks without spatial components

**Common pitfalls:**
- Sketch decoder not trained adequately
- Sketching frequency misaligned with reasoning pace
- Sketch loss overwhelming text signal
- Not validating sketch interpretability

Reference: [Latent Sketchpad on arXiv](https://arxiv.org/abs/2510.24514)
