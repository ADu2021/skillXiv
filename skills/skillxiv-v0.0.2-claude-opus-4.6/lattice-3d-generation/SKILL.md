---
name: lattice-3d-generation
title: "LATTICE: Democratizing High-Fidelity 3D Generation at Scale via VoxSet"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.03052
keywords: [3d-generation, diffusion-transformers, voxel-representations, scalable-generation, test-time-scaling]
description: "Semi-structured latent representation combining efficiency of VecSet with spatial structure guidance via voxel queries and rotary positional embeddings, enabling strong test-time scaling (6K to 30K tokens) and improved model scaling without sparse components."
---

## Summary

LATTICE introduces VoxSet, a semi-structured latent representation that combines the efficiency of VecSet methods with spatial structure guidance from coarse voxel grids. The approach uses voxel-anchored queries with rotary positional embeddings in diffusion transformers, enabling high-fidelity 3D generation with strong test-time scaling and pure transformer architecture.

## Core Technique

**VoxSet Representation:** Hybrid latent space balancing efficiency and structure:
- **Query Positions:** Voxel grid centers (coarse structure)
- **Learned Features:** Per-voxel latent codes (detail)
- **Positional Encoding:** Rotary embeddings anchoring queries to spatial positions

**Two-Stage Pipeline:**
1. **Coarse Generation:** Voxelize existing 3D model to initialize voxel structure
2. **Fine Generation:** Produce detailed VoxSet latents within sparse voxel structure

**Test-Time Scaling:** Increase token count during inference from 6K to 30K, leveraging learned spatial structure to maintain coherence.

## Implementation

**Voxel grid initialization:**
```python
def create_voxel_grid(coarse_geometry, voxel_size=0.05):
    # Convert mesh to voxels
    voxels = voxelize(coarse_geometry, voxel_size)

    # Extract voxel centers (query positions)
    voxel_centers = voxel_positions(voxels)

    return voxel_centers  # [num_voxels, 3]
```

**VoxSet representation:**
```python
class VoxSetLatent:
    def __init__(self, num_voxels, latent_dim=64):
        # Query positions (fixed at voxel centers)
        self.positions = nn.Parameter(voxel_centers)  # [num_voxels, 3]

        # Learnable feature codes per voxel
        self.codes = nn.Parameter(torch.randn(num_voxels, latent_dim))

    def forward(self):
        # Apply rotary position embeddings
        positions_embedded = rotary_embed(self.positions)
        features = self.codes * positions_embedded
        return features
```

**Rotary positional embeddings for 3D:**
```python
def rotary_embed_3d(positions):
    # positions: [num_positions, 3]
    dim = 64  # Feature dimension

    # Compute rotation angles per dimension pair
    freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

    angles = positions.unsqueeze(-1) @ freq.unsqueeze(0)
    embed = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    return embed
```

**Two-stage generation:**
```python
def generate_3d(coarse_mesh, num_steps=100):
    # Stage 1: Voxelize coarse geometry
    voxels = voxelize(coarse_mesh)
    voxset_init = VoxSetLatent(num_voxels=voxels.sum())

    # Stage 2: Diffusion to refine details
    for step in range(num_steps):
        noise = randn_like(voxset_init.codes)
        # Denoising step preserving voxel structure
        voxset_init.codes = diffusion_step(voxset_init.codes, noise, step)

    # Decode to mesh
    mesh = voxset_to_mesh(voxset_init)
    return mesh
```

**Test-time scaling (token extension):**
```python
def generate_with_scaling(voxset, target_tokens=30000):
    initial_tokens = voxset.codes.shape[0]

    if target_tokens > initial_tokens:
        # Interpolate new voxel positions
        new_positions = interpolate_positions(voxset.positions, target_tokens)
        new_codes = interpolate_features(voxset.codes, target_tokens)

        voxset_scaled = VoxSetLatent(target_tokens)
        voxset_scaled.positions = new_positions
        voxset_scaled.codes = new_codes

    # Continue diffusion with scaled latents
    mesh = voxset_to_mesh(voxset_scaled)
    return mesh
```

## When to Use

- High-fidelity 3D model generation from text descriptions
- Applications where test-time scaling can improve quality
- Scenarios requiring pure transformer architecture without sparse ops
- Tasks balancing generation quality and computational efficiency

## When NOT to Use

- Real-time 3D generation where voxel structure is unnecessary
- Applications with simple or small 3D objects
- Scenarios requiring exact mesh topology
- Tasks where point cloud or implicit representations work better

## Key References

- 3D generation and mesh synthesis
- Voxel representations and spatial structure
- Diffusion transformers for generation
- Test-time compute scaling for improved quality
