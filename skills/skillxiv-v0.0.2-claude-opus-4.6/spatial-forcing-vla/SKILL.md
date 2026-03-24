---
name: spatial-forcing-vla
title: "Spatial Forcing: Implicit Spatial Representation Alignment for VLA Model"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.12276"
keywords: [vla, spatial-understanding, representation-alignment, 3d-geometry, embodied-ai]
description: "Align intermediate visual embeddings in vision-language-action models with 3D geometric representations from pretrained foundation models. Improves spatial understanding and enables faster training (3.8x speedup) without explicit 3D inputs."
---

# Spatial Forcing: Implicit Spatial Representation Learning for VLAs

Vision-Language-Action models need spatial understanding for embodied tasks, but explicit 3D inputs (depth maps, point clouds) introduce complexity and sensor noise. Spatial Forcing implicitly teaches spatial comprehension by aligning visual embeddings with 3D representations during training.

Core insight: strong spatial understanding emerges when visual processing layers learn to align with geometric structure. By forcing alignment with lightweight 3D foundation models, VLAs learn spatial reasoning without needing explicit 3D inputs at inference time.

## Core Concept

**Intermediate Alignment**: Rather than modifying final outputs, align intermediate visual embeddings with 3D geometric representations. This implicitly teaches spatial structure at multiple processing stages.

**Geometric Guidance**: Use pretrained 3D foundation models (3D vision transformers) to provide geometric signals without requiring explicit depth sensors or point cloud inputs.

**Efficient Training**: The alignment process accelerates convergence, achieving 3.8x speedup over baseline VLA training.

## Architecture Overview

- **Visual Encoder**: Standard Vision Transformer processing raw images
- **3D Foundation Model**: Frozen pretrained model providing geometric signals
- **Alignment Loss**: Compares intermediate embeddings to 3D representations
- **Geometric Projection**: Lightweight layer mapping visual to spatial embeddings

## Implementation Steps

**Stage 1: Set Up 3D Foundation Model**

Initialize frozen 3D feature extractor:

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class SpatialForcingVLA(nn.Module):
    def __init__(self, vla_model_name, geometric_model_name):
        super().__init__()

        # Load VLA visual encoder
        self.visual_encoder = AutoModel.from_pretrained(
            vla_model_name,
            trust_remote_code=True
        )

        # Load 3D foundation model (frozen)
        self.geometric_encoder = AutoModel.from_pretrained(
            geometric_model_name,  # e.g., 'point-cloud-foundation'
            trust_remote_code=True
        )

        # Freeze geometric encoder
        for param in self.geometric_encoder.parameters():
            param.requires_grad = False

        # Alignment projection layers
        self.alignment_heads = nn.ModuleDict()
        for layer_idx in [6, 9, 12]:  # Align intermediate layers
            self.alignment_heads[f'layer_{layer_idx}'] = nn.Linear(
                self.visual_encoder.config.hidden_size,
                self.geometric_encoder.config.hidden_size
            )

    def forward(self, images, depth_maps=None):
        """
        Process images with spatial forcing alignment.
        Depth maps used only during training for geometric signals.
        """

        batch_size = images.shape[0]

        # Extract visual embeddings at multiple layers
        visual_outputs = self.visual_encoder(
            images,
            output_hidden_states=True
        )

        # Extract geometric embeddings (training only)
        if depth_maps is not None and self.training:
            with torch.no_grad():
                # Convert depth to geometric signal
                geometric_signal = self._depth_to_geometric(depth_maps)
                geometric_embeddings = self.geometric_encoder(
                    geometric_signal
                )
        else:
            geometric_embeddings = None

        return visual_outputs, geometric_embeddings

    def _depth_to_geometric(self, depth_maps):
        """
        Convert depth maps to 3D point clouds for geometric encoding.
        """

        batch_size, height, width = depth_maps.shape
        points_list = []

        for b in range(batch_size):
            # Create 3D point coordinates from depth
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, height),
                torch.linspace(-1, 1, width),
                indexing='ij'
            )

            # Normalize depth
            z = depth_maps[b] / depth_maps[b].max()

            # Stack into 3D points
            points = torch.stack([xx, yy, z], dim=-1)  # [H, W, 3]
            points = points.view(-1, 3)  # [H*W, 3]
            points_list.append(points)

        # Batch into tensor
        geometric_signal = torch.stack(points_list)  # [B, H*W, 3]

        return geometric_signal
```

**Stage 2: Implement Alignment Loss**

Define spatial forcing loss that aligns embeddings:

```python
class SpatialForcingLoss(nn.Module):
    def __init__(self, align_layers=[6, 9, 12]):
        super().__init__()
        self.align_layers = align_layers

    def forward(
        self,
        visual_embeddings,
        geometric_embeddings,
        alignment_heads
    ):
        """
        Compute alignment loss across specified layers.
        Aligns visual embeddings to geometric structure.
        """

        total_loss = 0.0
        num_layers = len(self.align_layers)

        for layer_idx, visual_hidden in enumerate(visual_embeddings):
            if layer_idx not in self.align_layers:
                continue

            # Project visual embeddings to geometric space
            visual_proj = alignment_heads[f'layer_{layer_idx}'](
                visual_hidden
            )

            # Compute cosine similarity alignment
            # We want visual to align with geometric structure
            visual_normalized = torch.nn.functional.normalize(
                visual_proj,
                dim=-1
            )

            geometric_normalized = torch.nn.functional.normalize(
                geometric_embeddings,
                dim=-1
            )

            # Alignment: maximize similarity
            similarity = torch.mm(
                visual_normalized,
                geometric_normalized.T
            )

            # Loss: minimize negative maximum similarity
            # (encourages each visual token to align with some geometric token)
            alignment_loss = -similarity.max(dim=1).values.mean()

            total_loss = total_loss + alignment_loss / num_layers

        return total_loss

def compute_full_loss(
    model,
    images,
    actions,
    depth_maps,
    spatial_forcing_weight=0.5
):
    """
    Combine action prediction loss with spatial forcing loss.
    """

    visual_outputs, geometric_embeddings = model(
        images,
        depth_maps=depth_maps
    )

    # Standard VLA loss: predict actions
    action_logits = model.action_head(visual_outputs.last_hidden_state)
    action_loss = torch.nn.functional.cross_entropy(
        action_logits.view(-1, model.action_vocab_size),
        actions.view(-1)
    )

    # Spatial forcing loss
    spatial_loss = spatial_forcing_loss_fn(
        visual_outputs.hidden_states,
        geometric_embeddings,
        model.alignment_heads
    )

    # Combined loss
    total_loss = (
        action_loss +
        spatial_forcing_weight * spatial_loss
    )

    return total_loss
```

**Stage 3: Training with Spatial Forcing**

Integrate spatial forcing into VLA training loop:

```python
def train_vla_with_spatial_forcing(
    model,
    train_dataloader,
    num_epochs=5,
    spatial_weight=0.5
):
    """
    Train VLA with spatial forcing alignment.
    Depth maps used during training but not required at inference.
    """

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    spatial_forcing_loss_fn = SpatialForcingLoss()

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['images'].cuda()
            actions = batch['actions'].cuda()
            depth_maps = batch['depth_maps'].cuda()  # Only during training

            # Forward pass
            loss = compute_full_loss(
                model,
                images,
                actions,
                depth_maps,
                spatial_forcing_weight=spatial_weight
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}, Step {batch_idx}, "
                    f"Loss: {loss.item():.4f}"
                )

    return model

# During inference, depth is not required
def inference_vla(model, images):
    """
    VLA inference without depth maps.
    Spatial understanding learned through spatial forcing.
    """

    with torch.no_grad():
        visual_outputs, _ = model(images, depth_maps=None)
        action_logits = model.action_head(
            visual_outputs.last_hidden_state
        )
        actions = action_logits.argmax(dim=-1)

    return actions
```

## Practical Guidance

**When to Use Spatial Forcing:**
- VLA training where 3D spatial understanding improves performance
- Scenarios where depth sensors are unreliable or unavailable at inference
- Tasks requiring generalization across different embodiments

**When NOT to Use:**
- Tasks not requiring spatial reasoning (language only)
- Inference pipelines where explicit 3D inputs are available and reliable
- Limited training data (spatial forcing requires depth annotation)

**Training Strategy:**

| Phase | Focus | Depth Required |
|-------|-------|-----------------|
| Warm-up (Epoch 1-2) | Action prediction only | No |
| Main (Epoch 3-4) | Spatial forcing with weight 0.5 | Yes |
| Fine-tune (Epoch 5) | Reduce spatial weight to 0.1 | No |

**Common Pitfalls:**
- Spatial weight too high (overfits to geometric alignment)
- Depth maps with poor quality (noise propagates to training)
- Aligning all layers uniformly (different layers need different alignment strengths)
- Forgetting to disable depth requirement at inference time

## Reference

Based on the research at: https://arxiv.org/abs/2510.12276
