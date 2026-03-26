---
name: ovie-monocular-novel-view-synthesis
title: "OVIE: Monocular Novel-View Synthesis via Unpaired Internet Images"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.23488"
keywords: [Novel-View Synthesis, Monocular Depth, Unpaired Images, Geometry-Free Inference, 3D Scaffolding]
description: "A single insight eliminates multi-view requirements for novel-view synthesis: monocular depth acts as a training-time geometric scaffold to generate synthetic view pairs from unpaired internet images, but can be discarded at inference. This reframes the problem from needing paired multi-view data to leveraging abundant 2D internet imagery. Trigger: When limited to monocular video or single-image novel-view synthesis, use depth as training scaffold on unpaired data—the model learns geometry without needing it at inference."
category: "Insight-Driven Papers"
---

## The Breakthrough Insight

**The observation**: Monocular depth estimation can act as a training-time geometric scaffold to generate synthetic view pairs from single unpaired images, but the trained model works without depth at inference—eliminating multi-view training data requirements.

**Why this matters**: Previous novel-view synthesis required multi-view dataset collection (expensive, limited in diversity). The insight reveals that depth can generate pseudo-views from unpaired internet images during training, transforming a 30-million-image internet corpus into synthetic paired data. At inference, depth is discarded—geometry is baked into the model.

## Why Was This Hard?

Novel-view synthesis traditionally required paired multi-view data: multiple cameras capturing the same scene. This is expensive to collect and limits dataset diversity. Unsupervised approaches existed, but required complex geometric assumptions or were inefficient.

The hidden assumption was that multi-view structure was necessary *throughout* the pipeline. But the authors discovered depth can be a temporary tool: generate pairs during training, discard at inference. This insight requires recognizing that the model learns geometry sufficiently to extrapolate without explicit depth at inference time.

Why nobody discovered this before: The idea of using depth asymmetrically—only during training—wasn't standard. Most works either use depth throughout or abandon it entirely. The possibility of training with depth but discarding it for inference wasn't obvious.

## How the Insight Reframes the Problem

**Before the insight:**
- Problem seemed to require: Multi-view video captures with camera calibration
- Bottleneck was: Expensive paired multi-view data collection
- Complexity was at: Learning geometry from limited paired views

**After the insight:**
- Problem reduces to: Run monocular depth on unpaired internet images to generate synthetic pairs, train on those pairs
- Bottleneck moves to: Quality of monocular depth estimation (errors propagate to training)
- New framing enables: Training on 30 million unpaired internet images instead of thousands of paired captures

**Shift type**: Data-property insight. The paper discovered that monocular depth provides sufficient signal to generate realistic synthetic pairs from unpaired images, unlocking training on large-scale unstructured data.

## Minimal Recipe

The key approach uses depth as a training-time scaffold:

```python
# Monocular depth generates synthetic paired views from unpaired images.
# Visibility masking handles disocclusions during view synthesis.
# Trained model runs without depth—geometry is learned, not explicit.

def train_novel_view_synthesis_with_depth_scaffold():
    """
    Use depth to generate training pairs from unpaired internet images.
    At inference: depth is not needed.
    """
    for image in unpaired_internet_images:
        # Estimate monocular depth (training-time only)
        depth = monocular_depth_estimator(image)

        # Lift to 3D and project with random camera transformation
        # This generates a pseudo-target view
        camera_transform = sample_random_camera_pose()
        pseudo_target = project_with_camera(image, depth, camera_transform)

        # Mask invalid regions (disocclusions where depth->visibility changes)
        valid_mask = compute_visibility_mask(depth, camera_transform)

        # Train model on (source=image, target=pseudo_target, mask=valid_mask)
        # Losses apply only to valid regions
        train_step(image, pseudo_target, valid_mask)

def infer_without_depth(source_image):
    """
    At inference: no depth estimator needed.
    Geometry is learned by the model.
    """
    novel_view = model.forward(source_image, target_camera_pose)
    return novel_view
```

## Results

**Metric**: Novel-view synthesis quality on diverse internet images

- Baseline supervised multi-view methods: Limited dataset diversity due to capture logistics
- OVIE (trained on pseudo-pairs from unpaired data): Generalizes to diverse internet imagery
- Improvement: Enables synthesis on 30 million varied scenes vs thousands of paired captures

**Key ablation**:
- Remove monocular depth scaffold: Training pairs become noisy, performance degrades significantly
- Remove visibility masking: Disocclusion artifacts dominate
- Geometry-free inference (keep depth): Slightly better quality but much slower; validates depth is not necessary
- Pseudo-pairs from depth on unpaired data: Achieves strong generalization, validating the insight

**Surprising finding**: Models trained on large-scale pseudo-pairs from unpaired images actually generalize better than models trained on smaller multi-view datasets, suggesting data quantity/diversity outweighs training signal perfection. This validates the scaffold insight is fundamental.

## When to Use This Insight

- When novel-view synthesis must work on diverse unstructured images
- To avoid expensive multi-view dataset collection
- For internet-scale training on 2D images
- When inference speed matters (no depth estimation at test time)

## When This Insight Doesn't Apply

- If you have access to high-quality multi-view data (simpler supervised approach)
- When monocular depth estimation is unavailable or unreliable for your domain
- For scenes with complex occlusions or thin structures (depth fails)
- When geometric precision is critical (implicit geometry may lack detail)

## Insight Type

This is a data-property insight combined with perspective-shift. The paper measured what information monocular depth provides and discovered it's sufficient to generate training pairs without full multi-view captures. It reframes the problem from "collect paired views" to "use depth to generate pairs from unpaired images."

Related insights: "Learning to See in the Dark" (using sensor priors), self-supervised learning through proxy tasks—papers leveraging auxiliary signals to unlock training on unstructured data.
