---
name: visual-reasoning-revisitation
title: "Don't Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.18842"
keywords: [Multimodal Reasoning, Visual Grounding, Active Referencing, Vision-Language]
description: "Enable multimodal models to dynamically revisit and re-ground reasoning steps in images using point-and-copy mechanisms for better long-horizon reasoning."
---

# Re-ground Visual Reasoning Through Selective Image Revisitation

Multimodal language models encode images once into key-value caches, then reason purely in text. This approach works for simple questions but fails for complex reasoning: as reasoning chains lengthen, models progressively lose focus on relevant visual regions. Humans don't work this way—they revisit visual evidence repeatedly while thinking.

The solution is dynamic visual referencing: enable models to selectively revisit the image during reasoning, pointing to relevant regions and updating their understanding. This "active visual referencing" grounds intermediate reasoning steps back in the image, preventing drift and improving accuracy on multi-step visual reasoning tasks.

## Core Concept

The key insight is that reasoning about images should be interactive, not one-shot. Long reasoning chains require multiple passes over the image:

- **Dynamic referencing**: Model selects when to revisit the image during reasoning
- **Point-and-copy mechanism**: Select regions (via coordinates/points) to attend to
- **Re-grounding**: Update visual context at specific reasoning steps
- **Attention preservation**: Maintain focus on relevant regions as reasoning progresses
- **Selective revisitation**: Only revisit when reasoning confidence drops or new information is needed

This prevents catastrophic forgetting of visual details during text-only reasoning phases.

## Architecture Overview

- **Visual encoder**: Standard vision backbone (CLIP, DINO, etc.)
- **Spatial attention mechanism**: Ability to focus on regions identified by coordinates
- **Dynamic referencing controller**: Decides when to revisit image and which regions
- **Point selector module**: Model outputs coordinates to re-attend to
- **Hybrid reasoning**: Alternates between visual and text-only reasoning steps
- **KV cache management**: Maintains fresh visual context for referenced regions

## Implementation

Implement selective visual revisitation by adding a pointing mechanism to multimodal models:

```python
# Dynamic visual referencing for multimodal reasoning
import torch
import torch.nn as nn
from einops import rearrange

class SelectiveVisualRevisitation(nn.Module):
    """
    Enable models to dynamically revisit and re-ground visual information.
    """
    def __init__(self, hidden_dim=768, image_size=336, num_visual_tokens=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.num_visual_tokens = num_visual_tokens

        # Spatial attention: focus on specific regions
        self.spatial_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Point predictor: model outputs (x, y) to attend to
        self.point_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Output (x, y) coordinates
        )

        # Region encoder: extract context around pointed region
        self.region_encoder = nn.Linear(hidden_dim, hidden_dim)

        # Revisitation decision module
        self.revisit_decision = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, text_state, visual_features, image_spatial_coords):
        """
        Args:
            text_state: (batch, seq_len, hidden_dim) - current reasoning state
            visual_features: (batch, num_spatial_patches, hidden_dim) - image tokens
            image_spatial_coords: (batch, num_spatial_patches, 2) - coordinates of each patch
        Returns:
            updated_state: re-grounded with visual information
            revisited_regions: which regions were attended
        """
        batch_size, seq_len, hidden_dim = text_state.shape

        # Decide whether to revisit image
        revisit_prob = self.revisit_decision(text_state[:, -1])  # Use last token state

        # Predict point to attend to
        predicted_points = self.point_predictor(text_state[:, -1])  # (batch, 2)
        predicted_points = torch.sigmoid(predicted_points)  # Normalize to [0, 1]
        predicted_points = predicted_points * self.image_size

        # Select visual features near predicted point
        # Compute distance from each patch to predicted point
        distances = torch.cdist(
            predicted_points.unsqueeze(1),  # (batch, 1, 2)
            image_spatial_coords  # (batch, num_patches, 2)
        )  # (batch, 1, num_patches)

        # Soft attention over nearby regions (Gaussian-weighted)
        spatial_attention_weights = torch.exp(-distances / (self.image_size / 8))
        spatial_attention_weights = spatial_attention_weights / (spatial_attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Aggregate visual features from attended regions
        attended_visual = torch.einsum(
            'bip,bpi->bi',
            spatial_attention_weights,
            visual_features.unsqueeze(2)
        )  # (batch, 1, hidden_dim)

        # Re-ground text reasoning with attended visual information
        reground_vector = self.region_encoder(attended_visual.squeeze(1))

        # Blend reground vector with current text state
        updated_state = text_state.clone()
        # Modify last token to incorporate visual grounding
        updated_state[:, -1] = updated_state[:, -1] + (reground_vector * revisit_prob)

        return updated_state, {
            'revisit_prob': revisit_prob.mean().item(),
            'attended_points': predicted_points.cpu().numpy(),
            'attention_weights': spatial_attention_weights.squeeze(1).cpu().detach()
        }
```

Implement a wrapper that enables dynamic revisitation during multimodal reasoning:

```python
class MultimodalReasonerWithRevisitation:
    """
    Multimodal model that can revisit image during reasoning chain.
    """
    def __init__(self, base_model, revisitation_module):
        self.base_model = base_model
        self.revisitation = revisitation_module

    def reason_with_revisitation(self, image, question, max_reasoning_steps=10,
                                revisit_threshold=0.3):
        """
        Generate reasoning with dynamic visual revisitation.
        """
        # Encode image once
        image_features = self.base_model.encode_image(image)
        image_coords = self._get_patch_coordinates(image)

        # Initial text encoding
        text_encoding = self.base_model.encode_text(question)

        reasoning_trace = []
        current_state = text_encoding

        for step in range(max_reasoning_steps):
            # Generate next reasoning token
            next_token, logits = self.base_model.generate_token(current_state)

            reasoning_trace.append({
                'step': step,
                'token': next_token,
                'revisited': False
            })

            # Check if we should revisit image
            revisit_info = self.revisitation(current_state, image_features, image_coords)
            if revisit_info['revisit_prob'] > revisit_threshold:
                # Re-ground reasoning in image
                current_state = revisit_info['updated_state']
                reasoning_trace[-1]['revisited'] = True
                reasoning_trace[-1]['attended_points'] = revisit_info['attended_points']

            # Update state for next step
            current_state = self.base_model.update_state(current_state, next_token)

            # Stop if EOS
            if next_token == self.base_model.eos_token_id:
                break

        return {
            'reasoning_trace': reasoning_trace,
            'num_revisits': sum(1 for t in reasoning_trace if t['revisited']),
            'total_steps': len(reasoning_trace)
        }

    def _get_patch_coordinates(self, image):
        """Compute (x, y) coordinates for each image patch"""
        batch_size = image.shape[0]
        num_patches_h = image.shape[2] // 16  # Assuming 16-pixel patches
        num_patches_w = image.shape[3] // 16

        # Create grid of patch coordinates
        h_coords = torch.linspace(0, image.shape[2], num_patches_h)
        w_coords = torch.linspace(0, image.shape[3], num_patches_w)
        coords = torch.stack(torch.meshgrid(h_coords, w_coords, indexing='ij'), dim=-1)

        return coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
```

Implement a training procedure that teaches models to use revisitation effectively:

```python
def train_with_revisitation_supervision(model, train_data, num_epochs=10):
    """
    Train multimodal model to learn when and where to revisit images.
    Uses supervision signal: which regions are relevant for each reasoning step.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in train_data:
            images = batch['images']
            questions = batch['questions']
            reasoning_steps = batch['reasoning_steps']  # Multi-step reasoning
            visual_grounding = batch['visual_grounding']  # Which regions matter per step

            # Forward pass with revisitation
            image_features = model.encode_image(images)
            image_coords = model._get_patch_coordinates(images)

            losses = []

            for step_idx, step_tokens in enumerate(reasoning_steps):
                # Run one step of reasoning
                state = model.encode_text(questions)

                # Get revisitation decision
                revisit_logits = model.revisitation.revisit_decision(state)

                # Get spatial attention
                spatial_attention = model.revisitation(state, image_features, image_coords)

                # Supervision: which regions should be attended?
                target_regions = visual_grounding[step_idx]

                # Loss 1: spatial attention alignment
                target_attention = create_attention_mask(target_regions, image_features.shape)
                spatial_loss = F.kl_div(
                    F.log_softmax(spatial_attention, dim=-1),
                    target_attention,
                    reduction='batchmean'
                )

                # Loss 2: revisitation decision
                should_revisit = (len(target_regions) > 0).float()
                revisit_loss = F.binary_cross_entropy(revisit_logits, should_revisit.unsqueeze(1))

                total_loss = spatial_loss + 0.5 * revisit_loss
                losses.append(total_loss)

            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
```

## Practical Guidance

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Revisit threshold | 0.3 - 0.7 | Probability needed to trigger revisitation |
| Point prediction scale | image_size | Normalize coordinates appropriately |
| Spatial attention sigma | image_size/8 to image_size/4 | Controls region size around point |
| Max revisits per sequence | 3 - 8 | Balance accuracy with compute |
| Patch size | 14 - 32 pixels | Smaller = higher resolution, higher compute |

**When to use selective visual revisitation:**
- Multi-step visual reasoning tasks (VQA, visual understanding)
- Long reasoning chains over images
- Problems requiring attention to multiple image regions
- Accuracy is more important than single-pass latency
- Models struggle with mid-chain reasoning drift

**When NOT to use:**
- Single-question image tasks (captioning, classification)
- Simple visual understanding (no complex reasoning needed)
- Latency is critical (adds compute per revisit)
- Image complexity is low (doesn't benefit from re-grounding)
- Models already encode sufficient visual context in cache

**Common pitfalls:**
- Revisit threshold too low (excessive re-encoding, no benefit)
- Point predictor not well-calibrated (predicting outside image)
- Not supervising which regions to attend during training
- Not measuring whether revisitation actually helps on the task
- Using too-coarse spatial patches (losing region specificity)
- Not balancing revisitation frequency (must be selective)

## Reference

**Don't Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation**
https://arxiv.org/abs/2505.18842
