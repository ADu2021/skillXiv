---
name: learning-4d-reasoning
title: "Learning to Reason in 4D: Dynamic Spatial Understanding for VLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.20557
keywords: [vision-language, spatial-reasoning, 3d-reconstruction, dynamic, 4d]
description: "Enable VLMs to perform dynamic spatial reasoning (DSR) by extracting 4D priors from videos and using Geometry Selection Modules (GSM) for selective injection. Provides DSR-Train dataset (50K QA pairs) and benchmark with six reasoning types, balancing geometric specialization with general video understanding—improving VLM 4D reasoning without degradation on general tasks."
---

## Overview

Learning to Reason in 4D addresses a critical VLM limitation: difficulty with tasks requiring understanding of how objects move and relate spatially over time in 3D space. This framework combines automated 4D dataset generation with lightweight geometric-knowledge injection.

## Core Technique

**DSR Suite Dataset Pipeline:**
Transform in-the-wild videos into structured 4D reasoning training data.

```python
# Automated 4D dataset generation from video
class DSRDatasetGenerator:
    def __init__(self):
        self.vision_foundation = VisionFoundationModel()
        self.qa_generator = QAGenerator()

    def create_dsr_dataset(self, video_collection):
        """
        Extract 4D priors and generate reasoning questions.
        """
        dataset = []

        for video in video_collection:
            # Extract 3D/4D information
            camera_poses = self.vision_foundation.extract_camera_poses(video)
            point_clouds = self.vision_foundation.extract_point_clouds(video)
            object_masks = self.vision_foundation.segment_objects(video)
            orientations = self.vision_foundation.estimate_orientations(video)
            trajectories = self.vision_foundation.track_trajectories(video)

            # Store 4D metadata
            video_4d = {
                'camera_poses': camera_poses,
                'point_clouds': point_clouds,
                'object_masks': object_masks,
                'orientations': orientations,
                'trajectories': trajectories
            }

            # Generate multiple-choice questions
            # Six reasoning types: distance, direction, orientation, speed, etc.
            questions = self.qa_generator.generate_questions(
                video, video_4d,
                num_questions=50,
                reasoning_types=['distance', 'direction', 'orientation', 'speed', 'collision', 'permanence']
            )

            for question_data in questions:
                dataset.append({
                    'video': video,
                    'question': question_data['question'],
                    'choices': question_data['choices'],
                    'answer': question_data['answer'],
                    'reasoning_type': question_data['reasoning_type'],
                    '4d_priors': video_4d
                })

        return dataset
```

**Geometry Selection Module (GSM):**
Selectively inject geometric priors based on question content.

```python
class GeometrySelectionModule:
    def __init__(self, hidden_dim=768):
        self.qformer_1 = QFormer()  # Compress question semantics
        self.qformer_2 = QFormer()  # Extract geometry tokens
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, question_tokens, geometry_tokens, vlm_hidden_state):
        """
        Two-stage selection: understand question, extract relevant geometry.
        """
        # Stage 1: Condense question semantics
        question_condensed = self.qformer_1(question_tokens)

        # Stage 2: Extract question-relevant geometry
        # Only geometry related to question is selected
        geometry_relevant = self.qformer_2(
            geometry_tokens,
            condition=question_condensed
        )

        # Project geometry tokens to VLM dimension
        geometry_features = self.projection(geometry_relevant)

        # Inject into VLM hidden state
        augmented_state = vlm_hidden_state + geometry_features

        return augmented_state
```

## When to Use This Technique

Use when:
- VLMs need 4D spatial reasoning capabilities
- Processing videos with dynamic objects
- Multiple viewpoint reasoning
- Distinguishing fine spatial relationships

## When NOT to Use This Technique

Avoid if:
- Static images only (4D not applicable)
- General video understanding sufficient
- Geometric extraction unavailable
- Computational overhead unacceptable

## Implementation Notes

Requires: Vision foundation model for 4D extraction, Q-Former for selective attention, VLM integration point for geometry injection, dataset generation pipeline.

## Key Performance

Strong on 4D reasoning tasks while maintaining general video understanding.

## References

- Automated 4D prior extraction from video
- Geometry Selection Modules via Q-Former
- Question-conditioned geometric knowledge selection
