---
name: shamisa-self-supervised-image-quality
title: "SHAMISA: Self-Supervised No-Reference Image Quality Assessment via Structured Relational Supervision"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.13669"
keywords: [Self-Supervised Learning, Image Quality Assessment, Graph-Weighted Invariance, Contrastive Learning, No-Reference]
description: "Replace standard VICReg invariance loss with graph-weighted learnable adjacency matrix to enable self-supervised no-reference image quality assessment without human labels. Improves SRCC by +0.017 (2% relative) on six-dataset average and shows stronger cross-dataset transfer. Use when training quality assessment models without paired quality labels."
category: "Component Innovation"
---

## What This Skill Does

Replace the binary-constraint invariance term in VICReg (self-supervised learning framework) with a graph-weighted formulation using learnable soft adjacency. This enables self-supervised no-reference IQA using only a compositional distortion engine and dual-source relation graphs, improving generalization by 2% relative without requiring human quality annotations.

## The Component Swap

The old VICReg uses a rigid binary set of augmentation-positive pairs with fixed relationships:

```python
# Old: standard VICReg invariance (fixed positive pairs)
# A = set of fixed positive pairs from data augmentation
# Loss = Σ_(i,j)∈A ||Z_i - Z_j||²₂

invariance_loss_old = 0
for i, j in positive_pairs:  # Binary: (i,j) are paired or not
    invariance_loss_old += torch.norm(embeddings[i] - embeddings[j])**2
```

The new SHAMISA approach replaces fixed binary constraints with learnable weighted adjacency:

```python
# New: graph-weighted VICReg invariance (soft adjacency)
# G = learnable adjacency matrix in [0,1]^(N×N)
# Loss = Σᵢ Σⱼ G_{i,j} ||Z_i - Z_j||²₂

# Learnable adjacency matrix
G = torch.nn.Parameter(
    torch.ones(batch_size, batch_size) * 0.5  # Initialize to uniform
)
G.data = torch.clamp(G, min=0, max=1)  # Constrain to [0,1]

# Weighted invariance loss
invariance_loss_new = 0
for i in range(batch_size):
    for j in range(batch_size):
        weight = G[i, j]  # Soft weight, learnable
        invariance_loss_new += weight * torch.norm(embeddings[i] - embeddings[j])**2
```

The graph G is constructed from dual relation sources that capture quality-relevant structure:

```python
# Dual-source relation graphs
# Source 1: Distortion-aware relations (from compositional distortion engine)
G_distortion = generate_distortion_graph(
    images,
    distortion_types=['gaussian_blur', 'noise', 'compression', 'contrast']
)  # Images with same distortion type are positives

# Source 2: Content-aware relations (from semantic features)
G_content = generate_content_similarity_graph(
    images,
    feature_extractor=vit_backbone
)  # Images with similar content are positives

# Combine into learnable adjacency
G = merge_graphs(G_distortion, G_content, learnable=True)
```

## Performance Impact

**Six-dataset benchmark (SRCC metric):**
- Prior best (ARNIQA, supervised SSL): 0.869
- SHAMISA (self-supervised, no labels): 0.886 = **+0.017 SRCC (+2% relative)**

**Correlation metrics:**
- PLCC: 0.890 (prior) → 0.904 = **+0.014 PLCC (+1.6% relative)**

**Cross-dataset generalization:**
- Transfer learning: 9 of 12 synthetic-to-synthetic transfer directions win
- Indicates better learned representations that generalize beyond training domain

**Trade-offs:**
- Requires dual-source relation graphs (computational overhead for graph construction)
- Achieves comparable accuracy to supervised methods without human annotations

## When to Use

- No-reference image quality assessment without paired human labels
- Datasets where quality annotations are expensive or unavailable
- Multi-domain transfer scenarios requiring domain-agnostic quality representations
- Self-supervised learning settings where contrastive objectives alone underperform
- Tasks combining multiple quality signals (distortion + semantic content)

## When NOT to Use

- Datasets with abundant human quality annotations (fully supervised may be simpler)
- Single-domain, in-distribution evaluation where labeled baselines exist
- Computational budgets too constrained for dual-source graph construction
- Tasks where quality is driven by domain-specific signals not captured by distortion types
- Scenarios lacking meaningful content diversity (identical content, varying distortions only)

## Implementation Checklist

To adopt this component swap:

1. **Prepare distortion composition engine:**
   ```python
   # Generate synthetically distorted versions of images
   distortion_engine = DistortionComposer(
       distortions=[
           GaussianBlur(sigma_range=[0.5, 2.0]),
           GaussianNoise(std_range=[0.01, 0.1]),
           JPEGCompression(quality_range=[30, 95]),
           ContrastAdjustment(gamma_range=[0.5, 2.0])
       ]
   )

   # Create distorted pairs
   original_imgs = load_images()
   distorted_imgs = distortion_engine(original_imgs)  # K distorted variants per image
   ```

2. **Build dual-source relation graphs:**
   ```python
   # Graph 1: Distortion-aware relations
   # Images sharing same distortion type are positive pairs
   G_distortion = torch.zeros(N, N)
   for i, j in image_pairs:
       if distortion_type[i] == distortion_type[j]:
           G_distortion[i, j] = 1.0

   # Graph 2: Content-aware relations via semantic similarity
   features = backbone(original_imgs)  # Semantic embeddings
   similarities = cosine_similarity(features)  # NxN similarity matrix
   G_content = (similarities > threshold).float()  # Binary or soft threshold

   # Combine graphs (learnable merging)
   G_init = (G_distortion + G_content) / 2.0
   ```

3. **Initialize learnable adjacency matrix:**
   ```python
   G = torch.nn.Parameter(G_init)  # Learnable parameters
   ```

4. **Replace VICReg invariance loss:**
   ```python
   # Old: binary positive pairs only
   # invariance_loss = Σ_(i,j)∈A ||Z_i - Z_j||²₂

   # New: weighted pairs with learnable adjacency
   invariance_loss = 0
   for i in range(N):
       for j in range(N):
           weight = torch.sigmoid(G[i, j])  # Soft weights in (0,1)
           invariance_loss += weight * torch.norm(embeddings[i] - embeddings[j])**2

   # Normalize by number of pairs
   invariance_loss = invariance_loss / (N * N)
   ```

5. **Combine with VICReg covariance and variance terms:**
   ```python
   # Keep standard VICReg covariance and variance losses unchanged
   loss_cov = covariance_loss(embeddings)
   loss_var = variance_loss(embeddings)

   # Total loss
   loss_total = invariance_loss + 25 * loss_cov + 25 * loss_var
   ```

6. **Verify improvements:**
   - Measure SRCC/PLCC on standard IQA benchmarks (LIVE, CSIQ, TID2013, etc.)
   - Compare zero-shot transfer: train on one domain, test on others
   - Verify no supervised labels needed (self-supervised only)

7. **Hyperparameter tuning:**
   - `G_init` merge weights: balance distortion vs content (default 0.5-0.5)
   - `G` learning rate: typically 10x lower than backbone (default 1e-4 if backbone is 1e-3)
   - Distortion intensity ranges: tune to match target quality variations
   - Graph threshold for G_content: 0.5-0.8 (higher = fewer content-positive pairs)

8. **Known issues:**
   - Dual-source graph construction is O(N²) in batch size; use smaller batches or hierarchical graph methods
   - Learnable G can overfit if not regularized; add L2 penalty on G weights
   - Distortion engine must produce meaningful quality variations; unrealistic synthetic distortions reduce effectiveness
   - Works best with diverse image content; highly uniform datasets show minimal gains

## Related Work

This builds on VICReg (non-contrastive self-supervised learning) and extends it with structured relation graphs. Relates to self-supervised IQA methods and graph-based contrastive learning approaches. The compositional distortion pattern resembles data augmentation in self-supervised vision, but applied explicitly to quality modeling.
