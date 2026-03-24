---
name: vla-4d-spatiotemporal
title: "VLA-4D: Embedding 4D Awareness into VLA Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.17199"
keywords: [Vision-Language-Action, Robotics, Spatiotemporal Reasoning, 4D Perception, Manipulation]
description: "Enhance VLA models with spatiotemporal awareness by embedding both 3D spatial coordinates and temporal sequences: predict actions that include temporal parameters (duration) alongside spatial movements, achieving 97.4% robotic manipulation success by grounding reasoning in coherent 4D representations."
---

# VLA-4D: Spatiotemporal Aware Robotic Manipulation

Standard vision-language-action models predict spatial movements but lack temporal understanding, making robots execute incoherent sequences of actions with temporal discontinuities. This skill demonstrates how to extend VLAs with 4D awareness—explicitly reasoning about both 3D space and 1D time—enabling robots to perform spatiotemporally coherent manipulation requiring fine-grained timing and smooth motion trajectories.

The core innovation is augmenting action spaces to include temporal parameters and grounding visual representations in explicit 4D coordinates, enabling models to understand when actions should occur, not just where.

## Core Concept

VLA-4D implements 4D awareness through:

1. **4D Visual Representation**: Encodes both 3D spatial positions and temporal sequences from video
2. **Spatiotemporal Action Space**: Actions include temporal parameters (duration) alongside spatial control
3. **Temporal Grounding**: Time-aware visual features enabling smooth action execution
4. **Cross-Attention Fusion**: Integrates spatial and temporal information via attention mechanisms

## Architecture Overview

- **Video Encoder**: Extracts spatiotemporal features from observation sequences
- **4D Feature Embedding**: Maps 3D coordinates + time into unified representation space
- **Spatial Action Head**: Predicts movement vectors (Δx, Δy, Δz), rotation (Δθ), gripper state
- **Temporal Action Head**: Predicts action duration (Δt) and timing
- **Coherence Decoder**: Ensures temporal continuity between sequential actions

## Implementation Steps

The system extends standard VLA architectures with temporal reasoning.

**1. Build Video Encoder for Spatiotemporal Features**

Extract both spatial and temporal information from observation sequences.

```python
class SpatiotemporalVideoEncoder(torch.nn.Module):
    """
    Encodes video sequences into spatiotemporal features.
    Captures both spatial structure and temporal dynamics.
    """
    def __init__(self, hidden_dim=768, num_frames=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # Spatial encoder: per-frame CNN features
        self.spatial_encoder = torchvision.models.resnet50(
            pretrained=True, replace_stride_with_dilation=[False, True, True]
        )

        # Temporal encoder: 3D convolutions
        self.temporal_encoder = torch.nn.Sequential(
            torch.nn.Conv3d(2048, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )

        # Feature projection
        self.feature_projection = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, video_frames):
        """
        Encode video into spatiotemporal features.
        Args:
            video_frames: (batch, num_frames, 3, height, width)
        Returns:
            spatiotemporal_features: (batch, num_frames, height, width, hidden_dim)
        """
        batch_size, num_frames, _, height, width = video_frames.shape

        # Extract spatial features per frame
        spatial_feats = []

        for t in range(num_frames):
            frame = video_frames[:, t]
            spatial_feat = self.spatial_encoder(frame)  # (batch, 2048, h//32, w//32)
            spatial_feats.append(spatial_feat)

        spatial_feats = torch.stack(spatial_feats, dim=1)  # (batch, num_frames, 2048, h, w)

        # Apply temporal convolutions
        temporal_features = self.temporal_encoder(
            spatial_feats.permute(0, 2, 1, 3, 4)  # (batch, 2048, num_frames, h, w)
        )

        # Reshape to (batch, num_frames, h, w, hidden_dim)
        temporal_features = temporal_features.permute(0, 2, 3, 4, 1)

        # Project features
        batch_s, frames, h, w, channels = temporal_features.shape
        temporal_features = temporal_features.reshape(-1, channels)
        temporal_features = self.feature_projection(temporal_features)
        temporal_features = temporal_features.reshape(batch_s, frames, h, w, -1)

        return temporal_features
```

**2. Implement 4D Feature Embedding**

Embed 3D coordinates and temporal information into unified 4D representation space.

```python
class FourDFeatureEmbedding(torch.nn.Module):
    """
    Maps 3D spatial coordinates + time into 4D embedding space.
    Enables spatiotemporal grounding of visual features.
    """
    def __init__(self, hidden_dim=768, max_time=10.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_time = max_time

        # Spatial positional encoding (3D)
        self.spatial_pos_embed = torch.nn.Linear(3, hidden_dim // 2)

        # Temporal positional encoding (1D)
        self.temporal_pos_embed = torch.nn.Linear(1, hidden_dim // 2)

        # 4D fusion
        self.fusion_layer = torch.nn.Linear(hidden_dim, hidden_dim)

    def embed_4d_position(self, x, y, z, t):
        """
        Embed 3D position + time into 4D space.
        Args:
            x, y, z: Spatial coordinates (batch, seq_len)
            t: Temporal coordinate (batch, seq_len)
        Returns:
            embedding: (batch, seq_len, hidden_dim) 4D embedding
        """
        # Spatial embedding
        spatial_coords = torch.stack([x, y, z], dim=-1)  # (batch, seq_len, 3)
        spatial_embedding = self.spatial_pos_embed(spatial_coords)

        # Temporal embedding (normalize to [0, 1])
        t_normalized = t.unsqueeze(-1) / self.max_time  # (batch, seq_len, 1)
        temporal_embedding = self.temporal_pos_embed(t_normalized)

        # Fuse spatial and temporal
        combined = torch.cat([spatial_embedding, temporal_embedding], dim=-1)
        embedding = self.fusion_layer(combined)

        return embedding

    def forward(self, visual_features, spatial_coords, temporal_coords):
        """
        Integrate visual features with 4D positional information.
        Args:
            visual_features: (batch, height, width, hidden_dim)
            spatial_coords: (batch, height, width, 3) 3D coordinates
            temporal_coords: (batch, height, width) time values
        Returns:
            grounded_features: (batch, height, width, hidden_dim) 4D-grounded features
        """
        batch, h, w, _ = visual_features.shape

        # Flatten for embedding
        spatial_flat = spatial_coords.reshape(batch, -1, 3)
        temporal_flat = temporal_coords.reshape(batch, -1)

        # Get 4D embedding
        pos_embedding = self.embed_4d_position(
            spatial_flat[:, :, 0], spatial_flat[:, :, 1], spatial_flat[:, :, 2],
            temporal_flat
        )

        # Reshape back
        pos_embedding = pos_embedding.reshape(batch, h, w, -1)

        # Add positional information to visual features
        grounded_features = visual_features + pos_embedding

        return grounded_features
```

**3. Implement Spatiotemporal Action Head**

Predict 4D actions including spatial movements and temporal duration.

```python
class SpatiotemporalActionHead(torch.nn.Module):
    """
    Predicts robot actions in 4D: (Δx, Δy, Δz, Δθ, gripper, Δt)
    Temporal parameter enables coherent action sequencing.
    """
    def __init__(self, hidden_dim=768):
        super().__init__()

        # Shared action backbone
        self.action_backbone = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU()
        )

        # Spatial action head: (Δx, Δy, Δz, Δθ)
        self.spatial_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 2, 4),
            torch.nn.Tanh()  # Output in [-1, 1]
        )

        # Gripper head: open/close probability
        self.gripper_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()  # Output in [0, 1]
        )

        # Temporal head: action duration
        self.temporal_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Softplus()  # Output duration > 0
        )

    def forward(self, features):
        """
        Predict 4D action from visual features.
        Args:
            features: (batch, hidden_dim)
        Returns:
            actions: Dict with spatial, gripper, temporal components
        """
        backbone_out = self.action_backbone(features)

        spatial_action = self.spatial_head(backbone_out)  # (batch, 4)
        gripper_action = self.gripper_head(backbone_out)  # (batch, 1)
        temporal_action = self.temporal_head(backbone_out)  # (batch, 1)

        # Unpack spatial (scale to physical ranges)
        delta_xyz = spatial_action[:, :3] * 0.1  # Limit movement to ±0.1m
        delta_theta = spatial_action[:, 3] * np.pi  # Rotation in [-π, π]

        actions = {
            'position_delta': delta_xyz,  # (batch, 3)
            'rotation_delta': delta_theta,  # (batch,)
            'gripper': gripper_action.squeeze(-1),  # (batch,)
            'duration': temporal_action.squeeze(-1)  # (batch,)
        }

        return actions
```

**4. Build Complete 4D VLA Model**

Integrate video encoder, 4D grounding, and action heads.

```python
class VLA4D(torch.nn.Module):
    """
    Vision-Language-Action model with 4D spatiotemporal awareness.
    Predicts robot actions with explicit temporal parameters.
    """
    def __init__(self, language_model_name='bert-base-uncased', hidden_dim=768):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Video encoding
        self.video_encoder = SpatiotemporalVideoEncoder(hidden_dim=hidden_dim)

        # 4D grounding
        self.four_d_embedding = FourDFeatureEmbedding(hidden_dim=hidden_dim)

        # Language encoding
        self.language_model = AutoModel.from_pretrained(language_model_name)

        # Cross-modal fusion
        self.fusion_attention = torch.nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        # Action heads
        self.action_head = SpatiotemporalActionHead(hidden_dim=hidden_dim)

    def forward(self, video_frames, language_input_ids, spatial_coords=None, temporal_coords=None):
        """
        Predict 4D robot action from video and language instruction.
        Args:
            video_frames: (batch, num_frames, 3, height, width)
            language_input_ids: (batch, lang_seq_len) tokenized instruction
            spatial_coords: (batch, num_frames, height, width, 3) optional 3D coordinates
            temporal_coords: (batch, num_frames, height, width) optional time values
        Returns:
            actions: Dict with 4D action components
        """
        # Encode video
        video_features = self.video_encoder(video_frames)  # (batch, frames, h, w, hidden_dim)

        # 4D grounding if coordinates provided
        if spatial_coords is not None and temporal_coords is not None:
            # Average over frames for position encoding
            spatial_mean = spatial_coords.mean(dim=1)  # (batch, h, w, 3)
            temporal_mean = temporal_coords.mean(dim=1)  # (batch, h, w)

            video_features_mean = video_features.mean(dim=1)
            video_features = self.four_d_embedding(
                video_features_mean, spatial_mean, temporal_mean
            ).unsqueeze(1) + video_features

        # Encode language
        lang_output = self.language_model(language_input_ids)
        lang_features = lang_output.last_hidden_state  # (batch, lang_seq_len, hidden_dim)

        # Fuse vision and language via cross-attention
        # Video summary
        video_summary = video_features.reshape(
            video_features.shape[0], -1, video_features.shape[-1]
        ).mean(dim=1, keepdim=True)

        # Cross-attention: language attends to video
        fused, _ = self.fusion_attention(lang_features, video_summary, video_summary)

        # Global context for action prediction
        action_context = fused.mean(dim=1)  # (batch, hidden_dim)

        # Predict action
        actions = self.action_head(action_context)

        return actions
```

**5. Training Loss with Temporal Coherence**

Define loss encouraging temporally smooth action sequences.

```python
def vla_4d_loss(
    predicted_actions,
    target_actions,
    prev_actions=None,
    lambda_temporal=0.1,
    lambda_spatial=1.0
):
    """
    Loss combining spatial accuracy and temporal coherence.
    Args:
        predicted_actions: Dict with 4D action predictions
        target_actions: Ground truth actions
        prev_actions: Previous action (for temporal continuity)
        lambda_temporal, lambda_spatial: Loss weights
    Returns:
        total_loss: Combined loss
    """
    losses = {}

    # Spatial action loss (position, rotation, gripper)
    position_loss = torch.nn.functional.mse_loss(
        predicted_actions['position_delta'],
        target_actions['position_delta']
    )

    rotation_loss = torch.nn.functional.mse_loss(
        predicted_actions['rotation_delta'],
        target_actions['rotation_delta']
    )

    gripper_loss = torch.nn.functional.binary_cross_entropy(
        predicted_actions['gripper'],
        target_actions['gripper']
    )

    spatial_loss = position_loss + rotation_loss + gripper_loss
    losses['spatial'] = lambda_spatial * spatial_loss

    # Temporal loss: encourage smooth action transitions
    duration_loss = torch.nn.functional.mse_loss(
        predicted_actions['duration'],
        target_actions['duration']
    )

    # Temporal coherence: if previous action available, penalize abrupt changes
    if prev_actions is not None:
        position_change = torch.norm(
            predicted_actions['position_delta'] - prev_actions['position_delta'],
            dim=-1
        )

        # Penalize sudden jumps (mean should be small)
        coherence_penalty = position_change.mean()

        temporal_loss = duration_loss + coherence_penalty

    else:
        temporal_loss = duration_loss

    losses['temporal'] = lambda_temporal * temporal_loss

    total_loss = sum(losses.values())

    return total_loss, losses
```

## Practical Guidance

**When to Use VLA-4D:**
- Robotic tasks requiring smooth, temporally coherent manipulation
- Fine-grained control with strict timing requirements
- Tasks with multiple sequential sub-steps
- Scenarios where action duration impacts success

**When NOT to Use:**
- Simple point-to-point movement tasks
- Real-time systems where temporal prediction adds unacceptable overhead
- Scenarios without sufficient temporal annotation data

**Key Hyperparameters:**
- `max_time`: Maximum action duration in seconds (10-30 typical)
- `position_scale`: Spatial movement magnitude (0.05-0.2m typical)
- `lambda_temporal`: Weight of temporal coherence loss (0.05-0.2)
- `num_frames`: Video frames for temporal understanding (4-16)

**Performance Optimization:**
- Pre-encode videos once to avoid redundant feature extraction
- Use causal temporal masking to prevent information leakage
- Batch actions during inference for parallelism

**Integration with Robotics:**
VLA-4D outputs directly map to robot command APIs. Temporal parameter (Δt) determines action duration in robot hardware; execute spatial component at smooth pace over predicted duration for natural motion.

## Reference

Research paper: https://arxiv.org/abs/2511.17199
