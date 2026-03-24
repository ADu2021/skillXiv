---
name: geometry-grounded-vlm
title: "G²VLM: Geometry Grounded VLM with Unified 3D Reconstruction"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.21688"
keywords: [3D Reconstruction, Vision-Language Models, Geometric Perception, Spatial Reasoning]
description: "Extend vision-language models with 3D spatial understanding by adding geometric expert stream alongside semantic expert: predict pixel-aligned 3D point maps, surface normals, and camera poses from 2D images, enabling unified reasoning across 2D semantic and 3D geometric domains."
---

# G²VLM: Geometry-Grounded Vision-Language Models

Standard vision-language models treat images as 2D sequences, missing opportunities to ground reasoning in 3D spatial structure. This skill demonstrates how to augment VLMs with geometric perception by adding a specialized expert stream that reconstructs 3D properties (point clouds, normals, camera poses) in parallel with semantic understanding—enabling models to reason about spatial relationships, scale, occlusion, and layout.

The core innovation is the Mixture-of-Transformer-Experts architecture with paired geometric and semantic experts that interact through shared attention, enabling mutual improvement between visual geometry and language understanding.

## Core Concept

G²VLM implements geometric grounding through:

1. **Geometric Perception Expert** ("where" pathway): Reconstructs 3D space from 2D observations
2. **Semantic Perception Expert** ("what" pathway): Handles multimodal understanding and reasoning
3. **Interaction Mechanism**: Shared self-attention enabling cross-expert information flow
4. **Unified Decoding**: Both pathways contribute to downstream VLM reasoning tasks

## Architecture Overview

- **Vision Encoder (shared)**: DINOV2 encoding of input image
- **Geometric Expert**: Predicts 3D coordinates, normals, camera parameters
- **Semantic Expert**: Processes visual+text for reasoning tasks
- **Cross-Expert Attention**: Experts share information through transformer layers
- **Output Heads**: Geometry predictions + semantic reasoning

## Implementation Steps

The model architecture integrates geometric and semantic pathways from image encoding onward.

**1. Build Shared Vision Encoder**

Initialize encoder using pretrained vision model (DINOV2 recommended).

```python
class SharedVisionEncoder(torch.nn.Module):
    """
    Encodes images into feature representations for both geometric and semantic experts.
    Uses pretrained DINOV2 for strong geometric understanding.
    """
    def __init__(self, model_name='dinov2_vitb14', freeze=True):
        super().__init__()

        # Load pretrained DINOV2
        self.encoder = torch.hub.load(
            'facebookresearch/dinov2',
            model_name
        )

        # Optionally freeze backbone
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.feature_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_embed.patch_size

    def forward(self, images):
        """
        Encode images to patch-level features.
        Args:
            images: (batch, 3, height, width)
        Returns:
            features: (batch, num_patches, feature_dim)
            patch_positions: (batch, num_patches, 2) normalized patch coordinates
        """
        # Forward through encoder
        features = self.encoder.forward_features(images)['x']

        # Compute patch positions for 3D grounding
        batch_size, num_patches, feature_dim = features.shape

        # Grid of patch positions
        grid_h = int(np.sqrt(num_patches))
        grid_w = grid_h

        patch_positions = torch.zeros(batch_size, num_patches, 2)

        for i in range(grid_h):
            for j in range(grid_w):
                patch_idx = i * grid_w + j
                patch_positions[:, patch_idx, 0] = j / grid_w
                patch_positions[:, patch_idx, 1] = i / grid_h

        return features, patch_positions
```

**2. Implement Geometric Expert for 3D Prediction**

Build expert stream predicting 3D properties from visual features.

```python
class GeometricExpert(torch.nn.Module):
    """
    Predicts 3D geometric properties from visual features.
    Outputs: point coordinates, surface normals, camera pose.
    """
    def __init__(self, feature_dim=768, num_points=1024, hidden_dim=512):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_points = num_points
        self.hidden_dim = hidden_dim

        # Transformer layers for geometric reasoning
        self.geometry_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=4
        )

        # Output heads for 3D predictions
        self.point_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, num_points * 3)  # (x, y, z) per point
        )

        self.normal_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 3)  # Surface normal
        )

        self.camera_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 6)  # Camera pose (position + rotation)
        )

    def forward(self, features, patch_positions=None):
        """
        Predict 3D geometry from visual features.
        Args:
            features: (batch, num_patches, feature_dim)
            patch_positions: (batch, num_patches, 2) patch coordinates
        Returns:
            point_map: (batch, num_points, 3) 3D point coordinates
            surface_normals: (batch, 3) average surface normal
            camera_pose: (batch, 6) camera position and orientation
        """
        # Process features through transformer
        geometric_features = self.geometry_transformer(features)

        # Global feature for camera pose
        global_feature = geometric_features.mean(dim=1)

        # Predict 3D points
        point_logits = self.point_head(geometric_features.mean(dim=1))
        point_map = point_logits.reshape(-1, self.num_points, 3)

        # Predict surface normals
        surface_normals = self.normal_head(geometric_features.mean(dim=1))
        surface_normals = torch.nn.functional.normalize(surface_normals, dim=-1)

        # Predict camera pose
        camera_pose = self.camera_head(global_feature)

        return {
            'point_map': point_map,
            'surface_normals': surface_normals,
            'camera_pose': camera_pose
        }
```

**3. Implement Semantic Expert for VLM Reasoning**

Build semantic expert handling text and semantic visual understanding.

```python
class SemanticExpert(torch.nn.Module):
    """
    Semantic understanding pathway handling text+vision reasoning.
    Inherits geometric grounding from geometric expert via shared attention.
    """
    def __init__(self, feature_dim=768, text_vocab_size=30522, hidden_dim=512):
        super().__init__()

        self.feature_dim = feature_dim

        # Text embedding
        self.text_embedder = torch.nn.Embedding(text_vocab_size, feature_dim)

        # Semantic transformer (interacts with geometric expert via attention)
        self.semantic_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=4
        )

        # VLM task heads
        self.vqa_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1000)  # VQA answers
        )

        self.caption_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, text_vocab_size)
        )

    def forward(self, visual_features, text_input_ids, geometric_features=None):
        """
        Process visual+text input for semantic reasoning.
        Args:
            visual_features: (batch, num_patches, feature_dim) from vision encoder
            text_input_ids: (batch, text_seq_len) tokenized text
            geometric_features: (batch, num_patches, feature_dim) from geometric expert
        Returns:
            vqa_logits: (batch, num_answers)
            caption_logits: (batch, text_seq_len, vocab_size)
        """
        # Embed text
        text_features = self.text_embedder(text_input_ids)

        # Combine visual + text + geometric features
        combined_features = visual_features + text_features.mean(dim=1, keepdim=True)

        if geometric_features is not None:
            # Incorporate geometric information
            combined_features = combined_features + 0.3 * geometric_features

        # Semantic reasoning
        semantic_output = self.semantic_transformer(combined_features)

        # Task-specific outputs
        vqa_logits = self.vqa_head(semantic_output.mean(dim=1))
        caption_logits = self.caption_head(semantic_output)

        return {
            'vqa_logits': vqa_logits,
            'caption_logits': caption_logits,
            'features': semantic_output
        }
```

**4. Build Mixture-of-Experts Architecture**

Integrate geometric and semantic experts with shared attention.

```python
class GeometryGroundedVLM(torch.nn.Module):
    """
    Complete G²VLM model with geometric and semantic experts.
    Experts interact through shared transformer layers.
    """
    def __init__(self, vision_model_name='dinov2_vitb14', feature_dim=768):
        super().__init__()

        self.vision_encoder = SharedVisionEncoder(vision_model_name)
        self.geometric_expert = GeometricExpert(feature_dim=feature_dim)
        self.semantic_expert = SemanticExpert(feature_dim=feature_dim)

        # Cross-expert attention layers
        self.cross_attention = torch.nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        )

        # Gate for blending expert outputs
        self.expert_gate = torch.nn.Linear(feature_dim * 2, 2)

    def forward(self, images, text_input_ids=None, return_geometry=False):
        """
        Forward pass through G²VLM with geometric and semantic pathways.
        Args:
            images: (batch, 3, height, width) input images
            text_input_ids: (batch, text_len) optional text input
            return_geometry: Whether to return 3D predictions
        Returns:
            vqa_logits: (batch, num_answers)
            geometry: Dict with 3D predictions (optional)
        """
        # Encode image
        visual_features, patch_positions = self.vision_encoder(images)

        # Geometric expert forward pass
        geometry = self.geometric_expert(visual_features, patch_positions)

        # Semantic expert forward pass
        semantic_output = self.semantic_expert(
            visual_features, text_input_ids, geometry_features=None
        )

        # Cross-expert interaction: geometric expert attends to semantic
        if text_input_ids is not None:
            attended_geometric, _ = self.cross_attention(
                visual_features, semantic_output['features'], semantic_output['features']
            )

            # Gate: blend attended geometric with original semantic
            combined = torch.cat([semantic_output['features'], attended_geometric], dim=-1)
            gate_logits = self.expert_gate(combined)
            gate_weights = torch.softmax(gate_logits, dim=-1)

            semantic_output['features'] = (
                gate_weights[:, :, 0:1] * semantic_output['features'] +
                gate_weights[:, :, 1:2] * attended_geometric
            )

        results = {
            'vqa_logits': semantic_output['vqa_logits'],
            'caption_logits': semantic_output['caption_logits'],
            'semantic_features': semantic_output['features']
        }

        if return_geometry:
            results['geometry'] = geometry

        return results
```

**5. Training Loss with Geometry Supervision**

Define multi-task loss combining VLM and 3D prediction tasks.

```python
def geometry_grounded_loss(
    model_output,
    targets,
    geometry_targets=None,
    lambda_geo=0.3,
    lambda_vqa=1.0,
    lambda_caption=0.5
):
    """
    Combined loss for VLM tasks + geometry prediction.
    Args:
        model_output: Dict from model forward pass
        targets: Dict with 'vqa_answers', 'captions'
        geometry_targets: Dict with '3d_points', 'normals', 'camera_pose'
        lambda_geo, lambda_vqa, lambda_caption: Loss weights
    Returns:
        total_loss: Combined loss for backpropagation
    """
    losses = {}

    # VQA loss
    vqa_loss = torch.nn.functional.cross_entropy(
        model_output['vqa_logits'],
        targets['vqa_answers']
    )
    losses['vqa'] = lambda_vqa * vqa_loss

    # Caption loss
    caption_loss = torch.nn.functional.cross_entropy(
        model_output['caption_logits'].reshape(-1, model_output['caption_logits'].shape[-1]),
        targets['captions'].reshape(-1)
    )
    losses['caption'] = lambda_caption * caption_loss

    # Geometry losses (if supervision available)
    if geometry_targets is not None and 'geometry' in model_output:
        geometry = model_output['geometry']

        # 3D point prediction loss
        point_loss = torch.nn.functional.mse_loss(
            geometry['point_map'],
            geometry_targets['3d_points']
        )
        losses['points'] = point_loss

        # Surface normal prediction loss
        normal_loss = torch.nn.functional.mse_loss(
            geometry['surface_normals'],
            geometry_targets['normals']
        )
        losses['normals'] = normal_loss

        # Camera pose prediction loss
        camera_loss = torch.nn.functional.mse_loss(
            geometry['camera_pose'],
            geometry_targets['camera_pose']
        )
        losses['camera'] = camera_loss

        geo_loss = point_loss + normal_loss + camera_loss
        losses['geometry'] = lambda_geo * geo_loss

    total_loss = sum(losses.values())

    return total_loss, losses
```

**6. Inference with Spatial Reasoning**

Use geometry-grounded model for spatially-aware VLM inference.

```python
def geometry_aware_vqa(
    model,
    image,
    question,
    tokenizer,
    use_geometry=True
):
    """
    Answer VQA question using geometry-grounded reasoning.
    Args:
        model: Trained G²VLM
        image: Input image (PIL or tensor)
        question: Question string
        tokenizer: Text tokenizer
        use_geometry: Whether to incorporate 3D predictions
    Returns:
        answer: Generated answer text
        geometry: Optional 3D predictions for spatial reasoning
    """
    # Tokenize question
    question_tokens = tokenizer(question, return_tensors='pt')['input_ids']

    # Forward pass
    output = model(
        image,
        text_input_ids=question_tokens,
        return_geometry=use_geometry
    )

    # Decode VQA answer
    answer_idx = torch.argmax(output['vqa_logits'], dim=-1)
    answer = tokenizer.decode([answer_idx])

    geometry = output.get('geometry', None)

    return answer, geometry
```

## Practical Guidance

**When to Use G²VLM:**
- Vision tasks requiring spatial understanding (scene graphs, 3D QA)
- Applications needing explicit geometry (robotics, 3D scene understanding)
- Tasks combining semantic reasoning with spatial relationships
- Models that benefit from additional 3D supervision signal

**When NOT to Use:**
- Pure 2D tasks (classification, 2D detection)
- Scenarios without 3D annotation data for pretraining
- Real-time applications where 3D prediction overhead is unacceptable

**Key Hyperparameters:**
- `lambda_geo`: Weight of geometry loss (0.1-0.5)
- `num_geometric_transformer_layers`: Depth of geometry expert (2-6)
- `num_3d_points`: Resolution of point cloud prediction (256-2048)
- `feature_dim`: Vision feature dimension (typically 768)

**Training Data Requirements:**
- 3D annotations (point clouds, normals) enable better geometric learning
- Can leverage synthetic 3D data for pretraining
- Transfer from 2D-only models possible with fine-tuning (slightly lower final performance)

**Computational Overhead:**
- ~20-30% increased inference time vs. standard VLM (additional geometric expert)
- Memory increase: ~15-20% due to extra transformer layers
- Can optimize by sharing some transformer layers between experts

## Reference

Research paper: https://arxiv.org/abs/2511.21688
