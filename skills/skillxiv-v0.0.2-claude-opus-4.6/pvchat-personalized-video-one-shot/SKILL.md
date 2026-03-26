---
name: pvchat-personalized-video-one-shot
title: "PVChat: Personalized Video Chat with One-Shot Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.17069"
keywords: [One-Shot Learning, Video Understanding, Personalization, Mixture-of-Heads, Identity-Aware QA]
description: "Enable identity-aware video question answering with one-shot learning using Mixture-of-Heads enhanced ViLLM. Learns subject-specific features from single video through synthetic augmentation and progressive image-to-video training, enabling recognition of individuals in medical, smart home, and entertainment contexts."
---

## Core Concept

PVChat addresses the limitation of general video understanding models that fail at identity-aware comprehension (e.g., recognizing specific people in videos). Using one-shot learning, the model learns to recognize and reason about specific individuals from a single video through: (1) synthetic augmentation of identity-preserving training data; (2) a Mixture-of-Heads (MoH) attention mechanism with ReLU routing; (3) progressive image-to-video learning with specialized regularization objectives.

## Architecture Overview

PVChat combines several key components:

- **Mixture-of-Heads (MoH) Architecture**: Multiple attention heads specialized for different aspects (appearance, motion, identity), with learned routing based on input
- **ReLU Routing Mechanism**: Sparse gating that dynamically selects relevant attention heads based on query features
- **Smooth Proximity Regularization**: Progressive learning through exponential distance scaling during training stages
- **Head Activation Enhancement**: Balanced attention routing that prevents head collapse
- **Progressive Training Strategy**: Two-stage approach transitioning from static image recognition to dynamic video understanding

## Implementation Steps

### 1. Mixture-of-Heads (MoH) Attention with ReLU Routing

Implement the MoH mechanism with sparse, learnable routing:

```python
# Mixture-of-Heads attention with ReLU routing
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfHeadsAttention(nn.Module):
    """
    Multi-head attention with learnable mixture routing.
    """
    def __init__(self, dim, num_heads=8, num_heads_per_expert=2):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.num_experts = num_heads // num_heads_per_expert

        # Standard multi-head attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Learnable routing network
        self.routing_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, self.num_experts),
            nn.Softmax(dim=-1)
        )

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, seq_len, dim]
            key, value: [batch, seq_len, dim]
        """
        batch_size = query.shape[0]

        # Compute routing weights based on query
        route_weights = self.routing_network(query)  # [batch, seq, num_experts]

        # Project Q, K, V
        Q = self.q_proj(query).reshape(batch_size, -1, self.num_heads,
                                       self.dim_per_head)
        K = self.k_proj(key).reshape(batch_size, -1, self.num_heads,
                                     self.dim_per_head)
        V = self.v_proj(value).reshape(batch_size, -1, self.num_heads,
                                       self.dim_per_head)

        # Compute attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / \
                 (self.dim_per_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attention, V)

        # Route outputs through experts
        # Group heads into experts for routing
        head_outputs = head_outputs.reshape(
            batch_size, -1, self.num_experts,
            2, self.dim_per_head  # 2 heads per expert
        )

        # Apply routing: weight each expert's output
        routed_outputs = []
        for expert_idx in range(self.num_experts):
            expert_out = head_outputs[:, :, expert_idx, :, :]
            expert_out = expert_out.reshape(batch_size, -1,
                                           2 * self.dim_per_head)
            weight = route_weights[:, :, expert_idx:expert_idx+1]
            routed_outputs.append(weight * expert_out)

        # Combine all routed outputs
        output = torch.cat(routed_outputs, dim=-1)
        output = self.out_proj(output)

        return output
```

### 2. Synthetic Data Augmentation for Identity Preservation

Generate training samples that preserve subject identity while varying context:

```python
def create_identity_aware_augmentation_pipeline(video_frames,
                                               subject_name,
                                               num_samples=100):
    """
    Augment video data with identity-preserving positives and hard negatives.

    Args:
        video_frames: list of frames from source video
        subject_name: name/ID of subject in video
        num_samples: number of augmented samples to create

    Returns:
        augmented_data: list of (frame, qa_pair, label) tuples
    """
    augmented_data = []

    # Four question types for diverse learning
    qa_templates = {
        'existence': f"Is {subject_name} in this scene?",
        'appearance': f"Describe {subject_name}'s appearance in this frame.",
        'action': f"What action is {subject_name} performing?",
        'location': f"Where is {subject_name} located in this frame?",
    }

    # Identity-preserving augmentations
    for frame_idx, frame in enumerate(video_frames):
        for qa_type, template in qa_templates.items():
            for aug_idx in range(num_samples // (len(video_frames) *
                                                  len(qa_templates))):
                # Apply photometric augmentations (preserve identity)
                augmented_frame = apply_identity_preserving_aug(
                    frame,
                    aug_params={
                        'brightness': np.random.uniform(0.9, 1.1),
                        'contrast': np.random.uniform(0.9, 1.1),
                        'hue_shift': np.random.uniform(-10, 10),
                        'blur': np.random.uniform(0, 0.5),
                    }
                )

                # Generate answer based on video content
                if qa_type == 'existence':
                    answer = "Yes"
                elif qa_type == 'appearance':
                    answer = extract_appearance_from_frame(frame, subject_name)
                elif qa_type == 'action':
                    action = infer_action_from_frames(
                        video_frames[max(0, frame_idx-2):frame_idx+3]
                    )
                    answer = action
                else:  # location
                    answer = detect_subject_location(frame, subject_name)

                augmented_data.append({
                    'frame': augmented_frame,
                    'question': template,
                    'answer': answer,
                    'qa_type': qa_type,
                    'is_positive': True,
                    'subject': subject_name,
                })

    return augmented_data
```

### 3. Progressive Image-to-Video Training with Regularization

Implement two-stage training with specialized objectives:

```python
class PVChatTrainer:
    """Progressive image-to-video training for personalized video QA."""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.proximity_regularizer = SmoothProximityRegularizer()
        self.head_activation_loss = HeadActivationEnhancement()

    def smooth_proximity_regularization(self, video_features, decay_rate=2.0):
        """
        Regularize proximity between consecutive frames exponentially.

        Args:
            video_features: [batch, time_steps, dim]
            decay_rate: exponential scaling for temporal distance

        Returns:
            proximity_loss: float scalar loss
        """
        batch_size, seq_len, dim = video_features.shape
        proximity_loss = 0.0

        for t in range(seq_len - 1):
            # Compare features at time t and t+1
            feat_t = video_features[:, t, :]
            feat_next = video_features[:, t+1, :]

            # Distance scales exponentially with time gap
            temporal_dist = torch.exp(-decay_rate * torch.arange(
                1, seq_len - t, device=self.device))

            # Smooth proximity: nearby frames should be similar
            for delta in range(1, seq_len - t):
                feat_delta = video_features[:, t + delta, :]
                sim = F.cosine_similarity(feat_t, feat_delta, dim=-1)
                weight = temporal_dist[delta - 1]
                # High similarity with exponential decay
                proximity_loss += weight * (1 - sim).mean()

        return proximity_loss / seq_len

    def head_activation_enhancement(self, head_activations):
        """
        Encourage balanced activation across MoH attention heads.

        Args:
            head_activations: [batch, seq_len, num_heads]

        Returns:
            activation_loss: float scalar loss
        """
        # Compute entropy of head activations
        activation_probs = F.softmax(head_activations, dim=-1)
        entropy = -(activation_probs * torch.log(activation_probs + 1e-8)).sum(dim=-1)

        # Encourage high entropy (balanced activation)
        max_entropy = np.log(head_activations.shape[-1])
        activation_loss = max(0, (max_entropy - entropy).mean())

        return activation_loss

    def train_stage1_image(self, train_loader, epochs=5):
        """
        Stage 1: Train on augmented images with identity supervision.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            for batch in train_loader:
                images = batch['image'].to(self.device)
                questions = batch['question']
                answers = batch['answer']

                # Forward through image encoder
                logits, head_activations = self.model(images, questions)

                # Main QA loss
                qa_loss = F.cross_entropy(logits, answers)

                # Regularization: encourage balanced head usage
                head_loss = self.head_activation_enhancement(
                    head_activations
                )

                total_loss = qa_loss + 0.1 * head_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def train_stage2_video(self, train_loader_video, epochs=3):
        """
        Stage 2: Fine-tune on video sequences with temporal regularization.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(epochs):
            for batch in train_loader_video:
                videos = batch['video'].to(self.device)  # [B, T, C, H, W]
                questions = batch['question']
                answers = batch['answer']

                # Extract features over time
                batch_size, seq_len = videos.shape[:2]
                video_features = []

                for t in range(seq_len):
                    frame_features, head_act = self.model.encode_image(
                        videos[:, t, :, :, :]
                    )
                    video_features.append(frame_features)

                video_features = torch.stack(video_features, dim=1)

                # Forward through video decoder
                logits, head_activations = self.model.decode_qa(
                    video_features, questions
                )

                # Main QA loss
                qa_loss = F.cross_entropy(logits, answers)

                # Proximity regularization (temporal smoothness)
                proximity_loss = self.smooth_proximity_regularization(
                    video_features, decay_rate=2.0
                )

                # Head activation enhancement
                head_loss = self.head_activation_enhancement(
                    head_activations
                )

                total_loss = (qa_loss +
                             0.2 * proximity_loss +
                             0.1 * head_loss)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
```

## Practical Guidance

### When to Use PVChat

- Building personalized video assistants for smart home or healthcare
- Need to recognize and reason about specific individuals from limited video
- Want identity-aware QA without retraining on large labeled datasets
- Medical scenarios requiring patient/caregiver recognition
- Entertainment applications (anime/movie character identification)

### When NOT to Use

- Generic video understanding without identity requirements
- Insufficient compute for synthetic data generation and two-stage training
- Privacy concerns with storing individual video embeddings
- Need instantaneous one-shot adaptation (model still requires training)

### Hyperparameters & Configuration

- **Decay rate (smooth proximity regularization)**: 2.0 (exponential scaling for temporal distance)
- **Head activation enhancement weight**: 0.1 relative to main QA loss
- **Proximity regularization weight**: 0.2 relative to main QA loss
- **Stage 1 epochs**: 5 (image pre-training)
- **Stage 2 epochs**: 3 (video fine-tuning)
- **Learning rate stage 1**: 1e-4
- **Learning rate stage 2**: 5e-5
- **Number of attention heads**: 8 with 2 per expert group
- **Augmentation samples per video**: 100-200

### Common Pitfalls

- **Insufficient augmentation diversity**: Generate diverse QA types and photometric variations; avoid mode collapse
- **Imbalanced head activation**: Monitor entropy of head routing; apply head activation enhancement loss
- **Temporal discontinuity**: Proximity regularization helps; ensure frame sampling captures temporal evolution
- **Overfitting to one-shot subject**: Balance identity-specific features with generalization; validate on diverse subjects
- **Synthetic data quality**: Generated text answers must be accurate; use video understanding models for ground truth
- **Stage ordering**: Must do image pre-training first; video fine-tuning without image pre-training degrades performance

## Reference

- Han et al. 2021. Mixture-of-Experts Meets Vision Transformer (beyond translation).
- Lin et al. 2023. One-Shot Learning for Personalized Vision-Language Models.
- Kaur et al. 2020. Identifying Characters in Fine-Grained Video Understanding.
- Project page and code: Check arXiv repository for implementation details.
