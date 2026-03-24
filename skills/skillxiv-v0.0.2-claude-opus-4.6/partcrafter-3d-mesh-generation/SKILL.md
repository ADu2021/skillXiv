---
name: partcrafter-3d-mesh-generation
title: "PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.05573"
keywords: [3d-generation, diffusion-models, part-aware, mesh-synthesis, generative-models]
description: "Generates semantically-meaningful 3D parts from single images via compositional diffusion transformers with part-level identity and local-global attention."
---

# PartCrafter: Structured 3D Mesh Generation

## Core Concept

Generating 3D meshes with explicit semantic structure is challenging because parts must be geometrically distinct yet semantically coherent. PartCrafter departs from traditional two-stage approaches (segment then reconstruct) by using a unified generative model that jointly synthesizes multiple part-aware 3D meshes directly from RGB images. The key innovation is a compositional diffusion transformer with disentangled latent tokens per part, local-global attention for intra-part and inter-part reasoning, and identity-aware permutation-invariant design. This enables end-to-end generation of complex multi-part objects and scenes without pre-segmented inputs.

## Architecture Overview

- **Unified Architecture**: Single end-to-end model for part-aware 3D generation from images
- **Disentangled Part Tokens**: Each semantic part represented by K learnable identity-embedding tokens
- **Compositional Latent Space**: Multi-part latents composed before diffusion process
- **Local-Global Attention**: Dual attention mechanism handling within-part (local) and between-part (global) dependencies
- **Identity-Aware Design**: Permutation-invariant to part order, generalizing to variable part counts
- **Rectified Flow Matching**: Improved training stability over standard diffusion
- **Occluded Part Generation**: Capable of hallucinating non-visible parts

## Implementation

The following code demonstrates the compositional architecture and attention mechanism:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class PartIdentityEmbedding(nn.Module):
    """
    Learnable identity embeddings for individual parts.
    """
    def __init__(self, part_id: int, latent_dim: int = 768, num_tokens: int = 8):
        super().__init__()
        self.part_id = part_id
        self.num_tokens = num_tokens

        # Identity tokens: learnable embeddings unique to each part
        self.identity_tokens = nn.Parameter(
            torch.randn(num_tokens, latent_dim) / (latent_dim ** 0.5)
        )

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate identity embeddings for a batch.
        Returns: (batch_size, num_tokens, latent_dim)
        """
        # Repeat identity tokens across batch
        embeddings = self.identity_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return embeddings.to(device)


class LocalGlobalAttention(nn.Module):
    """
    Dual attention: local (intra-part) and global (inter-part).
    """
    def __init__(self, latent_dim: int, num_heads: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        # Local attention (within each part)
        self.local_q = nn.Linear(latent_dim, latent_dim)
        self.local_k = nn.Linear(latent_dim, latent_dim)
        self.local_v = nn.Linear(latent_dim, latent_dim)

        # Global attention (across parts)
        self.global_q = nn.Linear(latent_dim, latent_dim)
        self.global_k = nn.Linear(latent_dim, latent_dim)
        self.global_v = nn.Linear(latent_dim, latent_dim)

        self.out_proj = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, part_latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply local-global attention to part latents.

        part_latents: list of (batch, num_tokens, latent_dim) for each part
        Returns: updated part latents
        """
        num_parts = len(part_latents)
        updated_latents = []

        for part_idx in range(num_parts):
            part = part_latents[part_idx]  # (batch, num_tokens, latent_dim)

            # Local attention: self-attention within this part
            Q_local = self.local_q(part)
            K_local = self.local_k(part)
            V_local = self.local_v(part)

            # Multi-head local attention
            Q_local = Q_local.view(-1, self.num_heads, self.head_dim)
            K_local = K_local.view(-1, self.num_heads, self.head_dim)
            V_local = V_local.view(-1, self.num_heads, self.head_dim)

            local_scores = torch.matmul(Q_local, K_local.transpose(-2, -1)) / math.sqrt(self.head_dim)
            local_attn = F.softmax(local_scores, dim=-1)
            local_out = torch.matmul(local_attn, V_local)
            local_out = local_out.view(-1, part.shape[1], self.latent_dim)

            # Global attention: cross-attention with other parts
            # Collect all other parts as context
            other_parts = torch.cat([part_latents[j] for j in range(num_parts) if j != part_idx], dim=1)

            Q_global = self.global_q(part)
            K_global = self.global_k(other_parts)
            V_global = self.global_v(other_parts)

            Q_global = Q_global.view(-1, self.num_heads, self.head_dim)
            K_global = K_global.view(-1, self.num_heads, self.head_dim)
            V_global = V_global.view(-1, self.num_heads, self.head_dim)

            global_scores = torch.matmul(Q_global, K_global.transpose(-2, -1)) / math.sqrt(self.head_dim)
            global_attn = F.softmax(global_scores, dim=-1)
            global_out = torch.matmul(global_attn, V_global)
            global_out = global_out.view(-1, part.shape[1], self.latent_dim)

            # Combine local and global
            combined = torch.cat([local_out, global_out], dim=-1)
            output = self.out_proj(combined)

            updated_latents.append(output)

        return updated_latents


class PartCrafterDiffusionTransformer(nn.Module):
    """
    Compositional diffusion transformer for part-aware 3D mesh generation.
    """
    def __init__(self, latent_dim: int = 768, max_parts: int = 8,
                 tokens_per_part: int = 8, num_diffusion_steps: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_parts = max_parts
        self.tokens_per_part = tokens_per_part
        self.num_diffusion_steps = num_diffusion_steps

        # Part identity embeddings
        self.part_embeddings = nn.ModuleList([
            PartIdentityEmbedding(i, latent_dim, tokens_per_part)
            for i in range(max_parts)
        ])

        # Image encoder
        self.image_encoder = self._build_image_encoder()

        # Local-global attention
        self.local_global_attn = nn.ModuleList([
            LocalGlobalAttention(latent_dim) for _ in range(4)  # 4 transformer blocks
        ])

        # Diffusion timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        # Mesh decoder
        self.mesh_decoder = nn.Sequential(
            nn.Linear(latent_dim * tokens_per_part, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),  # Output: 3D vertex coordinates
        )

    def _build_image_encoder(self) -> nn.Module:
        """Build lightweight image encoder (e.g., ViT or CNN)."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.latent_dim)
        )

    def forward(self, image: torch.Tensor, num_parts: int,
               timestep: int = 0) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Generate part-aware 3D meshes from image.

        image: (batch, 3, H, W) input image
        num_parts: number of parts to generate
        timestep: diffusion timestep (0-1)

        Returns: (part_latents, composed_latent)
        """
        batch_size = image.shape[0]

        # Encode image as conditioning
        image_features = self.image_encoder(image)  # (batch, latent_dim)

        # Initialize part latents with identity embeddings
        part_latents = []
        for part_idx in range(min(num_parts, self.max_parts)):
            part_emb = self.part_embeddings[part_idx](batch_size, image.device)
            # Condition on image features
            part_emb = part_emb + image_features.unsqueeze(1) * 0.1
            part_latents.append(part_emb)

        # Diffusion timestep embedding
        t_emb = self.time_embedding(torch.tensor([[timestep]], dtype=torch.float32,
                                                  device=image.device))
        t_emb = t_emb.unsqueeze(0).expand(batch_size, -1)

        # Apply local-global attention blocks
        for attn_block in self.local_global_attn:
            # Add timestep information
            for i in range(len(part_latents)):
                part_latents[i] = part_latents[i] + t_emb.unsqueeze(1) * 0.1

            # Apply attention
            part_latents = attn_block(part_latents)

        # Compose multi-part latents
        composed_latent = torch.cat(part_latents, dim=1)  # (batch, num_parts*tokens_per_part, latent_dim)
        composed_latent = composed_latent.mean(dim=1)  # (batch, latent_dim) aggregate

        return part_latents, composed_latent

    def decode_meshes(self, part_latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Decode part latents to 3D mesh vertices.

        Returns: list of (batch, num_vertices, 3) vertex positions
        """
        meshes = []

        for part_latent in part_latents:
            # Flatten and decode
            flattened = part_latent.view(part_latent.shape[0], -1)
            vertices = self.mesh_decoder(flattened)
            # Reshape to (batch, num_vertices, 3)
            vertices = vertices.view(vertices.shape[0], -1, 3)
            meshes.append(vertices)

        return meshes


class RectifiedFlowMatching:
    """
    Training objective: rectified flow matching for improved stability.
    """
    def __init__(self, num_steps: int = 1000):
        self.num_steps = num_steps

    def sample_noise_schedule(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from uniform distribution."""
        return torch.rand(batch_size, device=device)

    def compute_flow_matching_loss(self, latent_start: torch.Tensor,
                                  latent_end: torch.Tensor,
                                  model_output: torch.Tensor,
                                  timestep: torch.Tensor) -> torch.Tensor:
        """
        Rectified flow matching loss: predict direction from noise to data.
        """
        # Linear interpolation path
        latent_t = (1 - timestep.unsqueeze(1)) * latent_start + timestep.unsqueeze(1) * latent_end

        # Target: direction from noise to data
        target = latent_end - latent_start

        # Flow matching loss
        loss = F.mse_loss(model_output, target)

        return loss


class PartCrafterTrainer:
    """
    Training pipeline for PartCrafter.
    """
    def __init__(self, model: PartCrafterDiffusionTransformer):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.flow_matching = RectifiedFlowMatching()

    def train_step(self, image: torch.Tensor, target_meshes: List[torch.Tensor],
                  num_parts: int) -> float:
        """
        Single training step with rectified flow matching.
        """
        batch_size = image.shape[0]

        # Sample timestep
        timestep = self.flow_matching.sample_noise_schedule(batch_size, image.device)

        # Forward pass
        part_latents, composed = self.model(image, num_parts, timestep[0].item())

        # Generate starting noise
        noise_latents = [torch.randn_like(pl) for pl in part_latents]

        # Compute flow matching loss
        loss = 0.0
        for i, (start, end) in enumerate(zip(noise_latents, part_latents)):
            loss += self.flow_matching.compute_flow_matching_loss(
                start, end, end, timestep.unsqueeze(1)
            )

        loss = loss / len(part_latents)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss)
```

## Practical Guidance

**Part Count Flexibility**: The model handles variable part counts (1-8 in standard setup). For objects with more parts, extend max_parts in initialization or use hierarchical decomposition.

**Identity Embedding Dimension**: Use num_tokens=8-16 per part. Larger values increase expressivity but require more memory; smaller values constrain geometric diversity.

**Local-Global Balance**: The 0.1 weighting for global context in local attention (line 153) controls cross-part influence. Increase to 0.2 if parts should be more interdependent; decrease to 0.05 for more independent parts.

**Training Data**: The approach requires part-level annotations. Use the provided dataset curation pipeline on Objaverse/ShapeNet to mine ~50K annotated objects automatically.

**Occluded Part Hallucination**: The compositional design naturally predicts unseen parts. Validate on occlusion benchmarks during training to ensure realistic completion.

**Inference Time**: Generation of 4-part objects takes ~34 seconds with standard setup. Optimize by reducing num_diffusion_steps to 10 for 2-3x speedup with minor quality loss.

**Mesh Quality Metrics**: Evaluate using Chamfer Distance (geometry) and F-Score (occupancy). The independence measure (IoU between part predictions) is critical for part-aware evaluation.

## Reference

PartCrafter achieves strong results on part-aware 3D generation:
- **Single-image generation**: Multiple semantically-meaningful parts from RGB
- **Multi-object scenes**: Outperforms two-stage approaches (HoloPart, MIDI)
- **Occluded parts**: Generates geometrically plausible unseen components
- **Efficiency**: 34 seconds per 4-part object on single GPU

The end-to-end approach is particularly valuable for interactive 3D modeling, content creation pipelines, and applications requiring structured decomposition of generated geometry.
