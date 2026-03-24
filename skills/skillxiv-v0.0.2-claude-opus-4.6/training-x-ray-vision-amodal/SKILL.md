---
name: training-x-ray-vision-amodal
title: "Training for X-Ray Vision: Amodal Segmentation and View-Invariant Object Representation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.00339"
keywords: [Amodal Segmentation, Multi-camera Video, Object Understanding, Dataset, Occlusion Handling]
description: "Infer complete object structure despite occlusion using multi-camera video. Enables training models to predict hidden object appearance by combining temporal and spatial context from multiple viewpoints."
---

# X-Ray Vision for Objects: Inferring Hidden Structure in Multi-Camera Scenes

Understanding what lies behind occlusions is fundamental to perception. Humans effortlessly imagine the back of a chair even when it's blocked by a table. Current computer vision models struggle with this task—they lack datasets pairing temporal video sequences with multiple camera perspectives that provide consistent ground truth about what's hidden.

The MOVi-MC-AC dataset addresses this gap by providing the first large-scale multi-camera video amodal dataset. With 5.8 million object instances across 2,041 synthetic scenes, it enables training models to reason about complete object geometry and appearance despite occlusion, using both temporal motion cues and spatial information from multiple viewpoints.

## Core Concept

Amodal reasoning means understanding the complete form of an object even when parts are invisible. Traditional datasets either provide single-view snapshots or sequence-based temporal data, but lack the rich supervision needed to learn robust occlusion handling. MOVi-MC-AC solves this by:

1. **Multi-Camera Consistency**: Capturing scenes from up to six viewpoints where one camera might reveal what another hides
2. **Temporal Grounding**: Providing video sequences where motion and visibility changes offer clues about hidden structure
3. **Complete Annotations**: Including both what's visible (modal masks) and what's occluded (amodal masks, full RGB content, depth maps)

This combination teaches models to fuse spatial and temporal cues—leveraging one camera's view to infer another's occluded regions.

## Architecture Overview

The dataset construction pipeline consists of these key components:

- **Scene Generation**: 2-second simulated sequences with 2-40 objects, some static and some dynamically thrown for motion variation
- **Multi-View Recording**: Six synchronized cameras with independent motion patterns (static, linear panning, arc-tracking) to maximize occlusion variation
- **Annotation Layer 1 (Modal)**: Per-frame RGB, modal segmentation masks, and depth maps capturing visible content
- **Annotation Layer 2 (Amodal)**: Unoccluded object appearance (ground-truth RGB content), complete segmentation masks, and depth for entire objects
- **Visibility Tracking**: Per-pixel metrics indicating which objects are occluded and from which viewpoints
- **Consistency Guarantees**: Stable object identifiers across frames and cameras for tracking occluded instances through time

The design prioritizes realistic occlusion patterns (not random) by simulating object dynamics, ensuring models learn generalizable rather than trivial solutions.

## Implementation

This section shows how to construct and use amodal reasoning in multi-camera settings.

**Step 1: Load and explore multi-camera amodal data**

The following code demonstrates how to load multi-view amodal annotations and inspect occlusion patterns:

```python
import numpy as np
from pathlib import Path

# Load a multi-camera sequence
scene_path = Path("movi_mc_ac_dataset/scene_0001")
cameras = {}

for camera_id in range(6):
    cam_data = {
        'rgb': np.load(scene_path / f"camera_{camera_id}_rgb.npy"),
        'modal_mask': np.load(scene_path / f"camera_{camera_id}_modal_mask.npy"),
        'amodal_mask': np.load(scene_path / f"camera_{camera_id}_amodal_mask.npy"),
        'amodal_rgb': np.load(scene_path / f"camera_{camera_id}_amodal_rgb.npy"),
        'visibility': np.load(scene_path / f"camera_{camera_id}_visibility.npy")
    }
    cameras[camera_id] = cam_data

# Identify occluded regions: where modal and amodal masks differ
camera_0 = cameras[0]
occluded_mask = (camera_0['modal_mask'] != camera_0['amodal_mask']).astype(float)
print(f"Occlusion rate: {occluded_mask.mean():.2%}")
```

This loads synchronized data from all six viewpoints and identifies which pixels are occluded.

**Step 2: Create correspondence between cameras using visibility tracking**

Models must learn which occluded regions in one camera can be revealed by other cameras. This code finds spatial correspondences across viewpoints:

```python
import cv2

def find_multiview_correspondences(cameras, object_id, frame_id):
    """Find which camera best reveals occluded parts of an object."""

    visibility_scores = {}

    for cam_id, cam_data in cameras.items():
        # Extract visibility of this object in this camera
        vis_map = cam_data['visibility'][object_id]
        # Higher visibility = more of this object is visible in this camera
        visibility_scores[cam_id] = vis_map.mean()

    # Rank cameras by how completely they show this object
    ranked = sorted(visibility_scores.items(), key=lambda x: x[1], reverse=True)
    best_camera = ranked[0][0]

    return best_camera, visibility_scores

# Find which camera best reveals object #5
best_cam, scores = find_multiview_correspondences(cameras, object_id=5, frame_id=0)
print(f"Best camera for object 5: {best_cam} (visibility: {scores[best_cam]:.2%})")
```

This identifies which viewpoint provides the most complete view of occluded objects.

**Step 3: Train an amodal prediction model using multi-camera context**

Here's a simplified training loop that teaches models to predict occluded appearance using both temporal and spatial context:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class AmodalPredictor(nn.Module):
    def __init__(self, num_cameras=6):
        super().__init__()
        self.temporal_encoder = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2)
        self.spatial_fusion = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3)  # Predict RGB
        )

    def forward(self, modal_sequence, camera_features):
        """
        modal_sequence: (T, B, C, H, W) - visible content over time
        camera_features: (B, 6, D) - spatial features from 6 cameras
        """
        B, T, C, H, W = modal_sequence.shape

        # Encode temporal dynamics
        temporal_out, (h, c) = self.temporal_encoder(modal_sequence.reshape(T, B, -1))
        # Shape: (T, B, 512)

        # Fuse spatial information from other cameras
        fused, _ = self.spatial_fusion(
            temporal_out[-1:],  # Query from final temporal state
            camera_features.transpose(0, 1),  # Key/Value from all cameras
            camera_features.transpose(0, 1)
        )
        # Shape: (1, B, 512)

        # Decode to RGB prediction for occluded regions
        amodal_pred = self.decoder(fused.squeeze(0))
        return amodal_pred

# Training setup
model = AmodalPredictor(num_cameras=6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

for epoch in range(10):
    for batch in dataloader:
        modal_seq = batch['modal_sequence']  # (T, B, 3, 256, 256)
        camera_feats = batch['camera_features']  # (B, 6, 512)
        amodal_rgb = batch['amodal_rgb']  # (B, 3, 256, 256)

        pred = model(modal_seq, camera_feats)
        loss = criterion(pred, amodal_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

This trains a model to predict unoccluded appearance by combining temporal patterns and spatial information from multiple cameras.

**Step 4: Evaluate amodal completeness with standard metrics**

Evaluation uses both reconstruction quality and segmentation metrics adapted to focus on occluded regions:

```python
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS

def evaluate_amodal_prediction(pred_amodal, gt_amodal, occluded_mask, method='all'):
    """Evaluate prediction quality on all or just occluded regions."""

    lpips_fn = LPIPS(net='alex')

    if method == 'occluded_only':
        # Only evaluate quality in regions that were actually hidden
        pred = pred_amodal[occluded_mask > 0]
        gt = gt_amodal[occluded_mask > 0]
    else:
        pred = pred_amodal
        gt = gt_amodal

    # Standard reconstruction metrics
    psnr = -10 * np.log10(((pred - gt) ** 2).mean())
    ssim_score = ssim(pred, gt, channel_axis=0)
    lpips_score = lpips_fn(pred, gt).item()

    return {'PSNR': psnr, 'SSIM': ssim_score, 'LPIPS': lpips_score}

results = evaluate_amodal_prediction(
    pred_amodal=model_output,
    gt_amodal=ground_truth,
    occluded_mask=occlusion_mask,
    method='occluded_only'
)
print(f"Occluded region prediction: PSNR={results['PSNR']:.2f}")
```

This computes standard metrics (PSNR, LPIPS, SSIM) both globally and specifically on regions that were actually occluded.

## Practical Guidance

**When to use MOVi-MC-AC:**
- Training models that must reason about complete object structure despite occlusion
- Developing view-invariant object representations using multiple perspectives
- Studying how temporal motion cues combine with spatial context for understanding
- Building systems that infer hidden content in surveillance, robotics, or autonomous driving scenarios

**When NOT to use:**
- Tasks requiring real-world photos (this is synthetic data)
- Scenes with more than 40 objects in tight clusters
- Real-time systems where dataset loading latency matters
- Applications where specific object categories matter more than general occlusion reasoning

**Hyperparameters and Configuration:**

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Sequence Length | 64 frames | Longer captures more temporal cues but increases memory |
| Number of Cameras | 4-6 | More cameras provide richer spatial coverage; 6 is dataset maximum |
| Feature Dimension | 512 | Balance between expressiveness and training speed |
| Temporal Encoder Layers | 2 | Deeper models capture longer-range dependencies but require more data |
| Optimization Learning Rate | 1e-4 | Standard for vision transformers; reduce if diverging |
| Batch Size | 32 | Limited by GPU memory; smaller batches increase noise in temporal signals |

**Common Pitfalls:**
- Treating modal and amodal masks as interchangeable—they represent fundamentally different quantities
- Using single-camera cropping to simulate multi-view occlusion (lacks proper visibility labels)
- Ignoring temporal consistency—evaluating single frames wastes the dataset's temporal richness
- Forgetting that visibility scores are per-object—different objects have different occlusion patterns across cameras

**Key Design Decisions:**
The dataset uses synthetic data to guarantee pixel-perfect occlusion ground truth. While this differs from real photos, it's necessary because measuring hidden object appearance is impossible from images alone. The multi-camera setup ensures that occlusions are "recoverable"—hidden regions are visible from other angles—teaching models useful geometry rather than hallucination.

## Reference

Gao, P., Ge, Y., An, L., Xia, B., Huang, J., Ding, C., & Wu, A. (2025). Training for X-Ray Vision: Amodal Segmentation and View-Invariant Object Representation. arXiv preprint arXiv:2507.00339. https://arxiv.org/abs/2507.00339
