---
name: part-aware-3d-generation
title: "OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.06165"
keywords: [3D Generation, Part Decomposition, Generative Models, Structural Synthesis, Compositional Editing]
description: "Generate 3D objects with explicit part structures enabling compositional editing and animation. Decouples structure planning from geometry synthesis using two stages: autoregressive bounding box generation and part-aware refinement."
---

# OmniPart: Compositional 3D Generation with Part Structure and Fidelity

Current 3D generative models produce monolithic shapes lacking explicit part structures, limiting downstream applications like animation, material assignment, and semantic editing. OmniPart solves this through a two-stage pipeline that separates structure planning from geometry synthesis. The first stage autoregressively generates part bounding boxes from images, while the second stage synthesizes detailed geometry within those spatial constraints. This decoupled approach achieves both semantic clarity (low coupling between parts) and structural consistency (high fidelity to planned structure).

The core insight is that 3D part-aware generation benefits from separating what objects should exist (structure) from what they look like in detail (geometry). This mirrors human design processes where architects first plan spatial relationships, then detail the appearance.

## Core Concept

OmniPart operates through a principled two-stage pipeline:

**Stage 1 - Structure Planning**: Given an image and 2D part masks, autoregressively generate variable-length sequences of 3D bounding boxes that delineate where each part should appear in 3D space.

**Stage 2 - Part Synthesis**: Within the planned bounding boxes, synthesize high-fidelity geometry using a fine-tuned generative model. Crucially, assign unique position embeddings to distinguish parts and employ a voxel-discarding mechanism to maintain sharp part boundaries.

This separation ensures generated parts are semantically distinct yet spatially and visually coherent.

## Architecture Overview

- **Structure planner**: Transformer-based autoregressive model predicting part bounding boxes
- **Coverage loss**: Ensures generated boxes comprehensively enclose their corresponding 2D parts
- **Geometry synthesizer**: Fine-tuned voxel latent generative model (building on TRELLIS)
- **Position-aware embeddings**: Distinguish parts during synthesis via spatial position encoding
- **Voxel discarding mechanism**: Filters noisy voxels at part boundaries for sharp transitions
- **Minimal annotation requirement**: Fine-tuning uses only 15K annotated shapes (not per-part supervision)

## Implementation

Implement the structure planning stage using autoregressive bounding box prediction:

```python
import torch
import torch.nn as nn
from omnipart.planner import StructurePlanner
from omnipart.losses import PartCoverageLoss

planner = StructurePlanner(
    num_parts_max=10,  # Maximum parts per object
    bbox_dim=6  # [x, y, z, width, height, depth]
)

coverage_loss_fn = PartCoverageLoss()

def generate_part_structure(image, part_masks_2d):
    """Autoregressively generate 3D part bounding boxes."""

    # Encode image and 2D masks
    image_features = planner.image_encoder(image)
    mask_features = planner.mask_encoder(part_masks_2d)
    context = torch.cat([image_features, mask_features], dim=1)

    # Autoregressively predict bounding boxes
    bboxes = []
    hidden_state = planner.init_hidden(context)

    for step in range(planner.num_parts_max):
        # Predict next bounding box
        bbox_logits = planner.bbox_predictor(hidden_state)
        bbox = planner.decode_bbox(bbox_logits)

        # Check for stopping token (end-of-sequence indicator)
        if bbox[0] < 0:  # Special flag for stop token
            break

        bboxes.append(bbox)

        # Update hidden state for next prediction
        hidden_state = planner.update_hidden(hidden_state, bbox)

    # Compute coverage loss: ensure boxes cover all 2D masks
    loss_coverage = coverage_loss_fn(
        predicted_bboxes_3d=bboxes,
        part_masks_2d=part_masks_2d,
        image=image
    )

    return bboxes, loss_coverage
```

Implement the geometry synthesis stage with position-aware part distinction:

```python
from omnipart.synthesizer import PartSynthesizer
from omnipart.voxel_filter import VoxelDiscardingFilter

class PartAwareSynthesizer(nn.Module):
    """Synthesize high-fidelity geometry within planned part regions."""

    def __init__(self, pretrained_generator="TRELLIS"):
        super().__init__()
        self.generator = load_pretrained_model(pretrained_generator)
        self.voxel_discarder = VoxelDiscardingFilter()

        # Fine-tune generator on part-annotated data
        self.generator_finetuned = fine_tune_on_parts(
            self.generator,
            dataset_path="15k_annotated_shapes/",
            epochs=10
        )

    def synthesize_parts(self, image, bboxes, part_masks_2d):
        """Synthesize geometry for each part within its bounding box."""

        batch_size = image.shape[0]
        all_parts_voxels = []

        for part_idx, bbox in enumerate(bboxes):
            # Create position embeddings unique to this part
            pos_embedding = self.get_position_embedding(
                bbox_3d=bbox,
                part_idx=part_idx,
                num_parts=len(bboxes)
            )

            # Condition generator on image, mask, position
            part_input = {
                "image": image,
                "mask_2d": part_masks_2d[part_idx],
                "bbox": bbox,
                "position_embedding": pos_embedding
            }

            # Generate voxel latents for this part
            part_voxels = self.generator_finetuned(part_input)

            # Discard noisy voxels at part boundaries
            part_voxels_clean = self.voxel_discarder.filter(
                voxels=part_voxels,
                bbox=bbox,
                aggressiveness=0.7  # 70% of boundary voxels discarded
            )

            all_parts_voxels.append(part_voxels_clean)

        # Assemble all parts into single coherent 3D structure
        final_voxels = self.assemble_parts(all_parts_voxels, bboxes)

        return final_voxels
```

Combine structure planning and geometry synthesis into end-to-end generation:

```python
class OmniPart(nn.Module):
    """End-to-end part-aware 3D generation."""

    def __init__(self):
        super().__init__()
        self.structure_planner = StructurePlanner()
        self.geometry_synthesizer = PartAwareSynthesizer()

    def generate(self, image, part_masks_2d=None):
        """Generate part-aware 3D object from image and optional masks."""

        # Stage 1: Plan part structure
        bboxes, loss_structure = self.structure_planner(image, part_masks_2d)

        # Stage 2: Synthesize geometry within structure
        voxels = self.geometry_synthesizer.synthesize_parts(
            image=image,
            bboxes=bboxes,
            part_masks_2d=part_masks_2d
        )

        # Decode voxels to mesh with textures
        mesh = self.decode_to_mesh(voxels)
        textured_mesh = self.apply_textures(mesh, image)

        return {
            "mesh": textured_mesh,
            "bboxes": bboxes,
            "voxels": voxels,
            "loss": loss_structure
        }

# Inference
omnipart = OmniPart()
result = omnipart.generate(
    image=torch.randn(1, 3, 512, 512),
    part_masks_2d=torch.randn(1, 10, 512, 512)  # 10 potential parts
)

# Output is an editable, textured 3D asset with explicit part boundaries
print(f"Generated mesh with {len(result['bboxes'])} parts")
```

## Practical Guidance

### When to Use OmniPart

Use OmniPart when:
- Generating 3D objects that require part-based editing
- Creating assets for animation (parts move independently)
- Assigning different materials to different object parts
- Generating objects from single images with semantic decomposition
- You have 2D part masks or can generate them via segmentation

### When NOT to Use

Avoid OmniPart for:
- Point cloud or mesh data where part structure isn't semantically meaningful
- Real-time applications (two-stage generation is slower)
- Highly amorphous shapes (smoke, fluids, liquids)
- Objects where part boundaries are ambiguous
- Scenarios lacking any 2D guidance (pure 3D-from-nothing generation)

### Part-Aware Structure Quality

| Metric | Typical Performance | Target |
|--------|-------------------|--------|
| Part coverage (2D→3D) | 92-95% | >90% |
| Geometric consistency | 0.85-0.92 | >0.85 |
| Texture fidelity | 0.78-0.88 | >0.80 |
| Mesh smoothness | 0.91-0.96 | >0.90 |

### Design Choices for Different Tasks

| Task | Num Parts | Bbox Mode | Texture |
|------|-----------|-----------|---------|
| Furniture | 3-8 parts | Precise | High-quality |
| Characters | 8-15 parts | Overlapping | Variable |
| Vehicles | 5-10 parts | Non-overlapping | Detailed |
| Scene objects | 1-5 parts | Loose bounding | Simple |

### Common Pitfalls

1. **Over-segmenting**: Too many parts creates boundary artifacts. Aim for 5-10 parts per object.
2. **Ignoring part overlap**: Some parts naturally overlap (e.g., leg overlaps with torso). Handle gracefully.
3. **Poor 2D mask quality**: Incorrect part masks degrade structure planning. Validate masks carefully.
4. **Forgetting texture coherence**: Parts must have consistent texture at boundaries. The voxel discarder helps, but validate results.
5. **Underestimating fine-tuning**: The 15K annotation requirement is substantial. Reuse existing part-annotated datasets where possible.

### Evaluation Metrics

- **Structural fidelity**: Do predicted bboxes match image semantics? (Intersection-over-union with ground truth)
- **Part separation**: Are parts clearly distinct or do boundaries blur? (Boundary sharpness score)
- **Geometric quality**: Mesh density, smoothness, absence of holes or artifacts
- **Semantic correctness**: Does each part correspond to the intended semantic region?

### Fine-Tuning Strategy

1. Start with pre-trained TRELLIS model
2. Collect or source 15K objects with part annotations
3. Fine-tune with position embeddings for 10 epochs
4. Monitor validation loss; stop if plateauing
5. Apply voxel discarding to clean part boundaries
6. Test on held-out objects from diverse categories

## Reference

"OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion" - [arXiv:2507.06165](https://arxiv.org/abs/2507.06165)
