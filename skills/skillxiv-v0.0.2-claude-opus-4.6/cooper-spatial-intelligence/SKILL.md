---
name: cooper-spatial-intelligence
title: "COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.04563
keywords: [spatial reasoning, 3D understanding, auxiliary modalities, multimodal LLMs, depth and segmentation]
description: "Enhance spatial reasoning in multimodal LLMs by integrating depth and segmentation as auxiliary modalities with adaptive reasoning strategies. COOPER achieves 6.91% improvement in spatial understanding—when you need 3D-aware vision-language capabilities."
---

## Overview

COOPER unifies cooperative perception and reasoning through a two-stage training approach that develops both auxiliary modality generation and adaptive reasoning capabilities. Rather than treating perception and reasoning separately, the model learns to generate depth and segmentation maps while developing interleaved reasoning strategies.

## When to Use

- Multimodal tasks requiring strong 3D spatial understanding
- Applications needing distance and size estimation from images
- Vision-language models that struggle with spatial relationships
- Scenarios requiring reasoning over spatial properties (volume, distance, orientation)
- Tasks involving scene understanding with geometric constraints

## When NOT to Use

- 2D image analysis where depth adds no value
- Tasks not requiring spatial reasoning
- Models already achieving satisfactory spatial understanding
- Real-time applications where auxiliary modality generation adds latency
- Scenarios with limited 3D training data

## Core Technique

Two-stage training developing auxiliary modality generation and adaptive reasoning:

```python
# Unified spatial reasoning architecture
class CooperativeSpatialModel:
    def __init__(self, vllm_backbone):
        self.vllm = vllm_backbone

        # Auxiliary modality generators
        self.depth_generator = DepthDecoder()
        self.segmentation_generator = SegmentationDecoder()

        # Adaptive reasoning module
        self.reasoning_adapter = ReasoningAdapter()

    def forward(self, image, question):
        """
        Unified perception and reasoning for spatial intelligence.
        Generates auxiliary modalities and performs adaptive reasoning.
        """
        # Extract visual features
        features = self.vllm.encode_image(image)

        # Generate auxiliary modalities
        depth_map = self.depth_generator(features)
        segmentation = self.segmentation_generator(features)

        # Integrate auxiliary modalities with text
        enhanced_features = self.integrate_modalities(
            features, depth_map, segmentation
        )

        # Adaptive interleaved reasoning
        reasoning_path = self.reasoning_adapter.compute_path(
            enhanced_features, question
        )

        # Generate answer with spatial reasoning
        answer = self.vllm.decode_with_path(
            enhanced_features,
            question,
            reasoning_path
        )

        return answer, depth_map, segmentation

    def integrate_modalities(self, visual, depth, segmentation):
        """
        Combine visual understanding with spatial auxiliary modalities.
        Learning to generate these modalities helps internalize spatial knowledge.
        """
        # Depth provides scale and distance information
        depth_features = self.process_depth(depth)
        # Segmentation provides object boundaries and relationships
        seg_features = self.process_segmentation(segmentation)

        # Multi-stream fusion
        combined = torch.cat([visual, depth_features, seg_features], dim=-1)
        return combined

    def reasoning_adapter(self, features, question):
        """
        Adaptive interleaved reasoning strategies.
        Routes reasoning based on spatial complexity.
        """
        complexity_score = self.estimate_spatial_complexity(question)
        if complexity_score > 0.7:
            # Multi-step reasoning for complex spatial questions
            return self.multi_step_reasoning(features, question)
        else:
            # Direct reasoning for simple questions
            return self.direct_reasoning(features, question)
```

Two-stage training: first learn auxiliary modality generation, then jointly optimize with adaptive reasoning.

## Key Results

- 6.91% improvement in spatial reasoning tasks
- 7.92% improvement on distance and size estimation
- General performance preservation across other tasks
- Effective integration of depth and segmentation signals

## Implementation Notes

- Auxiliary modalities (depth, segmentation) aid spatial internalization
- Adaptive reasoning interleaves steps based on question complexity
- Two-stage training balances modality generation with reasoning
- Preserves compatibility with underlying VLLM architecture

## References

- Original paper: https://arxiv.org/abs/2512.04563
- Focus: Spatial reasoning in multimodal models
- Domain: Vision-language models, 3D understanding
