---
name: cov-chain-of-view-spatial-reasoning
title: "CoV: Chain-of-View Prompting for Spatial Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05172"
keywords: [Vision-Language Models, Spatial Reasoning, 3D Understanding, Test-Time Scaling]
description: "Enable vision-language models to perform embodied question answering in 3D environments through active camera exploration. CoV uses training-free test-time reasoning to iteratively select relevant viewpoints and adjust camera angles until sufficient context is gathered, achieving 11-13% accuracy improvements across spatial reasoning benchmarks."
---

## When to Use This Skill
- Embodied question answering in 3D environments (OpenEQA, ScanQA)
- Spatial reasoning tasks requiring multi-view understanding
- Scenarios with pre-rendered or real-world multi-view scenes
- Applications where model retraining is expensive or infeasible
- Tasks benefiting from test-time scaling and iterative refinement

## When NOT to Use This Skill
- Single-view-only applications (camera exploration adds overhead)
- Real-time systems requiring immediate response (iterative reasoning is inherently slower)
- 2D image understanding without 3D structure
- Applications with latency constraints prohibiting multiple inference passes

## Problem Summary
Vision-language models excel at understanding single images or text, but struggle with embodied question answering in 3D environments. The core constraint: models are limited to finite input views, preventing them from exploring the scene to gather spatially distributed information. Traditional approaches either use fixed viewpoint sets or require expensive retraining to adapt viewing strategies. This creates a fundamental capability gap for spatial reasoning tasks requiring dynamic perspective selection.

## Solution: Training-Free Chain-of-View Framework

Use test-time prompting to enable iterative camera control and view selection without model modification or retraining.

```python
class ChainOfViewReasoner:
    def __init__(self, vlm_model, scene_3d):
        self.vlm = vlm_model
        self.scene = scene_3d
        self.selected_views = []
        self.camera_pose = None

    def run_chain_of_view(self, question, max_iterations=5):
        """Iteratively select views and adjust camera until question answered"""

        # Stage 1: Coarse-grained view selection
        anchor_views = self.select_anchor_views(question, num_anchors=4)
        self.selected_views.extend(anchor_views)

        # Stage 2: Fine-grained camera adjustment
        for iteration in range(max_iterations):
            # Get current visual context from selected views
            visual_context = self.render_selected_views(self.selected_views)

            # Prompt VLM to reason about current observations + question
            reasoning = self.vlm.generate(
                visual_context, question,
                prompt_template="""
                Given these views of a 3D scene and the question: {question}
                Current observations:
                [IMAGES]

                What spatial relationships do you observe?
                What camera action would help answer the question?
                Options: forward, backward, left, right, up, down, rotate_yaw, rotate_pitch, switch_view
                """
            )

            # Extract action from VLM output
            action = self.parse_camera_action(reasoning)

            # Execute camera transformation
            if action != "none":
                self.apply_camera_action(action)
                new_view = self.render_current_view()
                self.selected_views.append(new_view)

            # Check if sufficient information is gathered
            if self.should_terminate(reasoning):
                break

        # Final answer generation with all gathered context
        final_answer = self.vlm.generate(
            visual_context=visual_context,
            question=question,
            prompt_template="Using all observed views, answer: {question}"
        )

        return final_answer

    def select_anchor_views(self, question, num_anchors):
        """Coarse-grained selection: identify question-aligned anchor views"""
        # Extract keywords from question
        keywords = extract_keywords(question)

        # Score all available views by relevance to keywords
        view_scores = []
        for view in self.scene.all_views:
            relevance_score = compute_visual_relevance(view, keywords)
            view_scores.append((view, relevance_score))

        # Select top-K views with diversity filtering
        anchor_views = select_diverse_top_k(view_scores, k=num_anchors)
        return anchor_views

    def apply_camera_action(self, action):
        """Update camera pose via SE(3) transformation"""
        translation_map = {
            "forward": [0, 0, -0.5],
            "backward": [0, 0, 0.5],
            "left": [-0.5, 0, 0],
            "right": [0.5, 0, 0],
            "up": [0, 0.5, 0],
            "down": [0, -0.5, 0]
        }

        rotation_map = {
            "rotate_yaw": (0, 0, 0.1),
            "rotate_pitch": (0.1, 0, 0),
            "rotate_roll": (0, 0.1, 0)
        }

        if action in translation_map:
            translation = translation_map[action]
            self.camera_pose = self.camera_pose @ se3_translate(translation)
        elif action in rotation_map:
            rotation = rotation_map[action]
            self.camera_pose = self.camera_pose @ se3_rotate(rotation)
```

## Key Implementation Details

**Two-Stage Architecture:**

**Stage 1: Coarse View Selection**
- Filter redundant frames from scene video
- Score views by relevance to question keywords
- Select question-aligned "anchor views" (typically 4-8)

**Stage 2: Fine-Grained Camera Adjustment**
- Iteratively reason about current observations
- Generate discrete camera actions (translation, rotation, view switching)
- Update camera pose via SE(3) transformations
- Re-render from new pose

**Input Representation:**
- Multi-view RGB-D images from 3D scene
- Sampled at 10:1 ratio from full scene video (10 frames from 100+ available)
- Each view includes RGB image + depth map
- Paired with natural language question

**Action Space (Discrete):**
- Translation: forward, backward, left, right, up, down (0.5 unit steps)
- Rotation: yaw, pitch, roll (0.1 radian steps)
- View switching: transition between pre-rendered viewpoints

**Termination Criteria:**
- Model indicates question is answered
- Fixed iteration limit reached (typically 5-10 steps)
- Redundant action generation (same action repeated)

## Performance Results

**Accuracy Improvements:**
- OpenEQA: +11.56% average improvement
- Maximum single-model gain: +13.62% (Qwen3-VL-Flash)
- Consistent improvements across four model families (Qwen, Gemini, Claude, LLaVA)

**Benchmark Coverage:**
- OpenEQA: Embodied question answering
- ScanQA: Scan-based spatial reasoning
- SQA3D: Sequential scene understanding

**Test-Time Scaling:**
- Performance improves with additional exploration steps
- Marginal gains continue up to iteration 8-10
- No performance ceiling observed within tested budget

## Advantages Over Baselines

- **vs. Fixed Viewpoints**: Dynamic selection adapts to question-specific information needs
- **vs. Model Retraining**: Zero training overhead; works with frozen VLMs
- **vs. Single View**: Multi-view exploration gains 11-13 percentage points
- **vs. Random Exploration**: Question-guided selection outperforms undirected exploration
- **vs. Uniform Iteration**: Learned termination criteria avoid wasted computation

## Model Compatibility
Tested on:
- Qwen3-VL-Flash, Qwen3-VL-Max
- Gemini-2.0-Flash-Experimental
- Claude-3.5-Sonnet
- LLaVA models

All model-specific implementations use identical prompting strategy—no model-specific tuning.

## Deployment Strategy

1. **Scene Representation**: Obtain 3D scene (mesh, NeRF, voxel grid)
2. **View Database**: Pre-render full set of possible viewpoints (or use real cameras)
3. **VLM Integration**: Wrap vision-language model with CoV prompting
4. **Action Parsing**: Implement parser to extract camera actions from model outputs
5. **Iteration Loop**: Run inference multiple times with updated camera poses
6. **Answer Extraction**: Parse final generated text for answer
