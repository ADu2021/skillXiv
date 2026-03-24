---
name: gui-actor-grounding
title: "GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.03143"
keywords: [visual grounding, GUI agents, action detection, vision transformers, screen understanding]
description: "Enable GUI agents to ground actions without generating pixel coordinates by using attention-based patch-level alignment and a verifier for selecting optimal action regions from candidates."
---

# GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents

## Core Concept

GUI-Actor addresses a fundamental mismatch in VLM-powered GUI agents: generating precise pixel coordinates for screen actions. Traditional approaches output (x, y) coordinates, but Vision Transformers work with patch-level features, creating a gap between dense screen pixels and sparse patch embeddings.

GUI-Actor eliminates coordinate generation entirely. Instead, it uses an attention mechanism to align a dedicated `<ACTOR>` token with relevant visual patches, enabling patch-based action grounding. A separate verifier selects the optimal action region from candidates. This approach achieves superior performance (44.6 on ScreenSpot-Pro vs. 38.1 for larger competitors) while maintaining frozen VLM backbones.

## Architecture Overview

- **Attention-Based Action Head**: Align `<ACTOR>` token with screen patches using cross-attention
- **Patch-Level Grounding**: Work directly with ViT patch granularity rather than dense pixels
- **Grounding Verifier**: Evaluate and rank action region candidates for selection
- **Frozen VLM Backbone**: Preserve pretrained vision-language model weights
- **Multi-Action Support**: Generate multiple action proposals in parallel
- **Minimal Fine-Tuning**: Only ~100M parameters trainable; rest frozen

## Implementation

The following steps outline how to implement coordinate-free visual grounding for GUI agents:

1. **Extract patch embeddings** - Get Vision Transformer patch features from the screen image
2. **Create action candidates** - Generate multiple action proposals using the attention mechanism
3. **Score with verifier** - Evaluate candidate quality and select optimal actions
4. **Execute action** - Perform the selected action without explicit coordinates
5. **Process feedback** - Update understanding based on action outcomes
6. **Iterate** - Continue grounding new actions as agent progresses through task

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class PatchExtractor:
    """Extract patch embeddings from Vision Transformer."""

    def __init__(self, patch_size: int = 16, image_size: int = 512):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2

    def extract(self, image: torch.Tensor, vit_model) -> torch.Tensor:
        """Extract patch embeddings from image."""
        with torch.no_grad():
            # Get patch embeddings from ViT
            patches = vit_model.get_patch_embeddings(image)
        return patches


class ActionHead(nn.Module):
    """Attention-based action head for GUI grounding."""

    def __init__(self, embedding_dim: int = 768, num_patches: int = 1024, num_actions: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_patches = num_patches
        self.num_actions = num_actions

        # ACTOR token
        self.actor_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Cross-attention mechanism
        self.actor_attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)

        # Action proposal generation
        self.action_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_patches)
        )

    def forward(self, patch_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate action candidates via attention alignment.

        Args:
            patch_embeddings: (batch_size, num_patches, embedding_dim)

        Returns:
            action_logits: (batch_size, num_patches) - logits for each patch
            actor_embeddings: (batch_size, embedding_dim) - refined actor token
        """
        batch_size = patch_embeddings.shape[0]

        # Expand ACTOR token for batch
        actor = self.actor_token.expand(batch_size, -1, -1)

        # Cross-attention: ACTOR attends to patches
        actor_attended, _ = self.actor_attention(actor, patch_embeddings, patch_embeddings)

        # Generate action logits for each patch
        action_logits = self.action_mlp(actor_attended.squeeze(1))

        return action_logits, actor_attended.squeeze(1)


class GroundingVerifier(nn.Module):
    """Verify and rank action candidates."""

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.scoring_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def verify_candidates(self, actor_embedding: torch.Tensor, patch_embeddings: torch.Tensor,
                         action_logits: torch.Tensor, top_k: int = 3) -> Tuple[torch.Tensor, List[int]]:
        """
        Verify and rank action candidates.

        Args:
            actor_embedding: (batch_size, embedding_dim)
            patch_embeddings: (batch_size, num_patches, embedding_dim)
            action_logits: (batch_size, num_patches)
            top_k: Number of top candidates to return

        Returns:
            verified_scores: (batch_size, top_k) - confidence scores
            selected_patches: (batch_size, top_k) - patch indices
        """
        batch_size, num_patches, _ = patch_embeddings.shape

        # Get top-k candidates by logits
        top_logits, top_indices = torch.topk(action_logits, k=top_k, dim=1)

        # Score candidates using verifier
        verified_scores = []
        for b in range(batch_size):
            scores = []
            for idx in top_indices[b]:
                patch_emb = patch_embeddings[b, idx, :]
                combined = torch.cat([actor_embedding[b], patch_emb])
                score = self.scoring_network(combined)
                scores.append(score.item())
            verified_scores.append(scores)

        verified_scores = torch.tensor(verified_scores)

        return verified_scores, top_indices

    def select_action(self, verified_scores: torch.Tensor, patch_indices: torch.Tensor) -> torch.Tensor:
        """Select best action from verified candidates."""
        best_indices = torch.argmax(verified_scores, dim=1)
        selected_patches = torch.gather(patch_indices, 1, best_indices.unsqueeze(1))
        return selected_patches


class GUIActor(nn.Module):
    """Coordinate-free GUI action grounding."""

    def __init__(self, vit_model, embedding_dim: int = 768, num_patches: int = 1024):
        super().__init__()
        self.vit = vit_model
        self.action_head = ActionHead(embedding_dim, num_patches)
        self.verifier = GroundingVerifier(embedding_dim)
        self.patch_extractor = PatchExtractor()

    def ground_action(self, image: torch.Tensor) -> Tuple[int, float]:
        """
        Ground an action in the GUI without explicit coordinates.

        Args:
            image: Screenshot tensor (batch_size, 3, H, W)

        Returns:
            patch_id: Selected patch index
            confidence: Confidence score for the action
        """
        # Extract patch embeddings from frozen ViT backbone
        with torch.no_grad():
            patch_embeddings = self.vit.get_patch_embeddings(image)

        # Generate action candidates
        action_logits, actor_embedding = self.action_head(patch_embeddings)

        # Verify and select best action
        verified_scores, top_indices = self.verifier.verify_candidates(
            actor_embedding, patch_embeddings, action_logits, top_k=5
        )

        selected_patch = self.verifier.select_action(verified_scores, top_indices)

        return selected_patch.squeeze(), verified_scores.max().item()

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for batch processing."""
        batch_size = images.shape[0]

        # Extract patch embeddings
        with torch.no_grad():
            patch_embeddings = self.vit.get_patch_embeddings(images)

        # Generate action candidates
        action_logits, actor_embedding = self.action_head(patch_embeddings)

        # Verify candidates
        verified_scores, top_indices = self.verifier.verify_candidates(
            actor_embedding, patch_embeddings, action_logits, top_k=5
        )

        selected_patches = self.verifier.select_action(verified_scores, top_indices)

        return selected_patches, verified_scores


def patch_to_coordinates(patch_id: int, patch_size: int = 16, image_size: int = 512) -> Tuple[int, int]:
    """Convert patch ID back to approximate coordinates."""
    patches_per_side = image_size // patch_size
    row = (patch_id // patches_per_side) * patch_size + patch_size // 2
    col = (patch_id % patches_per_side) * patch_size + patch_size // 2
    return col, row
```

## Practical Guidance

**Fine-tuning strategy:**
- **Frozen backbone**: Keep VLM frozen during training; only update action head (~100M params)
- **Learning rate**: 1e-3 to 1e-4 for minimal backbone disturbance
- **Batch size**: 32-64 sufficient given limited trainable parameters
- **Epochs**: 3-10 epochs typically converge; monitor validation performance

**Verifier configuration:**
- **Top-k candidates**: 3-5 candidates; too few limits options, too many adds noise
- **Scoring network depth**: 2-3 layers; balance expressiveness with overfitting risk
- **Confidence threshold**: 0.7+ for filtering low-confidence actions

**When to use:**
- GUI automation and interaction tasks
- Mobile app testing and automation
- Web scraping requiring visual understanding
- Interactive systems where coordinate precision is difficult
- Transfer learning to new interfaces without coordinate retraining

**When NOT to use:**
- Pixel-level precision required (e.g., precise drawing tools)
- Dynamic interfaces with rapidly changing layouts
- Tasks where coordinate output is explicitly needed downstream
- Real-time systems where attention computation overhead is prohibitive

**Common pitfalls:**
- **Frozen VLM limitation**: Backbone features may not be optimal for action grounding; consider light fine-tuning
- **Patch resolution tradeoff**: Larger patches lose fine-grained localization; smaller patches increase computation
- **Verifier collapse**: Scoring network may converge to trivial solutions; use diverse training data
- **Generalization**: Models may overfit to specific UI layouts; validate on diverse interfaces
- **Confidence calibration**: Verified scores may not reflect true reliability; apply temperature scaling

## Reference

GUI-Actor achieves 44.6 on ScreenSpot-Pro with Qwen2.5-VL, surpassing larger models (UI-TARS-72B at 38.1). The approach improves generalization to unseen resolutions and demonstrates effective capability transfer from frozen VLMs.

Original paper: "GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents" (arxiv.org/abs/2506.03143)
