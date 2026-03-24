---
name: yolo-moe-detection
title: "YOLO Meets Mixture-of-Experts: Adaptive Routing for Multi-Scale Detection"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2511.13344
keywords: [object-detection, mixture-of-experts, adaptive-routing, multi-scale-features, yolo]
description: "MoE-enhanced YOLOv9-Tiny using lightweight routers to adaptively weight specialized expert outputs at each feature resolution, improving detection quality while maintaining end-to-end differentiability. Deploy for efficient multi-scale object detection with dynamic specialization."
---

## Summary

This work integrates a Mixture-of-Experts framework into YOLOv9-Tiny, employing adaptive routing among multiple specialized experts. At each feature resolution level (8×8, 16×16, 32×32), lightweight routers dynamically assign weights to expert outputs using "reweighted Hadamard fusion" to combine expert representations, enabling specialization in different visual features while maintaining end-to-end training.

## Core Technique

**Resolution-Specific Experts:** Deploy separate expert modules at each feature pyramid level. Each expert specializes in different visual patterns—early layers learn coarse features, later layers learn fine details. Routers determine how much weight each expert contributes.

**Lightweight Routing:** Routers are small feedforward networks that output gating weights for each expert. Use Hadamard fusion (element-wise multiplication followed by summation) to combine expert outputs, reweighted by router scores.

**Load Balancing Loss:** Prevent router collapse by enforcing that all experts are utilized. Use an auxiliary loss term: ℒ_balance = sum(log(n) - entropy(router_weights)) for each position, encouraging uniform expert activation.

## Implementation

**Router architecture:** For each spatial position and scale, compute:
```
router_weights = softmax(small_mlp(feature_map))
```
Use a 2-layer MLP (hidden_dim=64) to keep routers lightweight.

**Expert combination:** Combine M experts via:
```
output = sum(router_weights[i] * expert[i](input) for i in 1..M)
```

**Load balancing:** Add auxiliary loss during training:
```
aux_loss = sum(log(num_experts) - entropy(router_weights)) / num_positions
total_loss = detection_loss + λ_balance * aux_loss  # λ_balance ≈ 0.01
```

**Inference:** Apply the same routing mechanism at test time; no additional complexity beyond routing overhead.

## When to Use

- Object detection tasks with multi-scale visual challenges
- Scenarios where model efficiency matters but accuracy is critical
- Applications needing adaptive specialization per feature scale
- Embedded systems deploying optimized detection (YOLOv9-Tiny as baseline)

## When NOT to Use

- Tasks with limited compute budget where routing overhead is prohibitive
- Simple detection scenarios where single-pathway detection suffices
- Scenarios preferring simpler, non-adaptive architectures
- Applications where experts collapse despite load balancing (revert to single expert)

## Key References

- Mixture-of-Experts for dynamic model capacity allocation
- Multi-scale feature processing in object detection
- YOLOv9 architecture and detection pipelines
- Load balancing and expert utilization in MoE systems
