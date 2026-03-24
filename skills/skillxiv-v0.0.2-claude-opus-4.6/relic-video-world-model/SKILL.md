---
name: relic-video-world-model
title: "RELIC: Interactive Video World Models with Long-Horizon Memory"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.04040
keywords: [world-models, video-generation, spatial-memory, robotics, long-horizon-planning]
description: "Compressed historical latents with camera poses in KV cache (4× compression), extended teacher training (20-second sequences), and replayed back-propagation (block-wise differentiation) enabling real-time interactive video generation with long-range spatial consistency."
---

## Summary

RELIC is a unified framework for interactive video world modeling combining three innovations: memory-efficient spatial memory using highly compressed historical latents in KV cache, extended teacher training enabling 20-second sequence generation, and replayed back-propagation for memory-efficient distillation. Together these enable real-time 16 FPS video generation with precise action control and long-horizon consistency.

## Core Technique

**Compressed Spatial Memory:** Store historical latents with camera pose information:
```
memory = [(latent_t1, camera_pose_t1), (latent_t2, camera_pose_t2), ...]
```
Compress latents via quantization and efficient encodings, achieving 4× compression.

**Extended Teacher Training:** Fine-tune base video model to generate longer sequences (20 seconds vs. 5 seconds), providing stronger supervision for long-range consistency learning.

**Replayed Back-Propagation:** Instead of computing gradients over full 20-second sequences (memory prohibitive), use block-wise differentiation:
```
# Full sequence: backward through all 400 frames
# Blocked: backward through 20-frame blocks independently
```

## Implementation

**Spatial memory structure:**
```python
class SpatialMemory:
    def __init__(self, max_history=100):
        self.latents = []  # Compressed feature maps
        self.camera_poses = []  # Camera extrinsics
        self.cache_size = 0

    def add(self, latent, pose):
        # Compress latent via quantization
        compressed = quantize(latent, bits=8)
        self.latents.append(compressed)
        self.camera_poses.append(pose)

    def retrieve(self, current_pose, k=10):
        # Find most relevant past frames
        scores = similarity(current_pose, self.camera_poses)
        indices = topk(scores, k)
        return self.latents[indices]
```

**Extended teacher training:**
```python
# Fine-tune base model for longer sequences
teacher_model = base_video_model
for epoch in range(num_epochs):
    # Generate 20-second sequences (vs. original 5)
    video = teacher_model.generate(num_frames=480)  # 20 sec at 24fps
    loss = mse(video, ground_truth_video)
    loss.backward()
    optimizer.step()
```

**Replayed back-propagation distillation:**
```python
def block_wise_backward(sequence, block_size=20):
    total_loss = 0
    for i in range(0, len(sequence), block_size):
        block = sequence[i:i+block_size]
        # Gradient computation only within block
        block_output = student_model(block)
        block_loss = mse(block_output, teacher_output[i:i+block_size])
        block_loss.backward()  # Only within this block
        total_loss += block_loss.detach()
    return total_loss
```

## When to Use

- Interactive video generation for robotics simulation
- Real-time video world models requiring long-horizon consistency
- Applications needing memory-efficient video generation
- Tasks where precise action control is important

## When NOT to Use

- Scenarios where short sequences (5 seconds) are sufficient
- Real-time applications where memory overhead is critical
- Tasks not requiring spatial consistency
- Offline video generation where distillation is unnecessary

## Key References

- World models and video prediction
- Spatial memory and attention mechanisms
- Knowledge distillation and block-wise backpropagation
- Action-conditioned video generation
