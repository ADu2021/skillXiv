---
name: sparc-vl-perception-reasoning
title: "SPARC: Separating Perception And Reasoning Circuits for VLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.06566"
keywords: [VLM, Perception, Reasoning, Test-Time Scaling, Two-Stage Pipeline, Efficiency]
description: "Improve vision-language model reasoning efficiency by decoupling perception (identifying task-relevant image regions) from reasoning (generating explanations), enabling asymmetric compute allocation. Reduces token overhead while improving accuracy through separate optimization of visual grounding and semantic reasoning stages."
---

# SPARC: Perception-Reasoning Separation for Efficient VLM Inference

Vision-language models (VLMs) struggle with efficiency when reasoning about complex visual scenes. Traditional approaches interleave perception (identifying relevant image regions) and reasoning (generating explanations) tokens, leading to token bloat and redundant processing. The model wastes capacity re-examining images for each reasoning step rather than establishing visual context upfront.

SPARC decouples VLM inference into two distinct stages mimicking the biological visual system. Stage 1 identifies task-relevant image regions with self-consistency (multiple rollouts), creating high-confidence visual grounding. Stage 2 generates reasoning using only the refined crops, reducing token overhead. This separation enables asymmetric compute allocation—perception scales independently through rollout aggregation while reasoning remains compact and efficient.

## Core Concept

Standard VLM reasoning embeds visual tokens throughout reasoning: [visual_tokens, reasoning_tokens_1, visual_tokens, reasoning_tokens_2, ...]. This causes redundancy and token explosion.

SPARC structures inference as two explicit stages:

**Stage 1 - Perception**: Given image + question, generate coordinates of task-relevant regions (bounding boxes or points). Run this k times with self-consistency, aggregating results.

**Stage 2 - Reasoning**: Crop image to relevant regions identified in Stage 1, feed only these crops to the model for reasoning and answer generation.

This separates concerns: perception learns "what to look at," reasoning learns "how to reason given visual context." Each stage optimizes independently via LoRA fine-tuning, and the boundary between stages is explicit and controllable.

## Architecture Overview

- **Stage 1 - Perception Circuit**: Generate k candidate region detection rollouts (bounding boxes or point coordinates), apply Weighted Box Fusion (WBF) to aggregate overlapping detections into high-confidence crops
- **Stage 2 - Reasoning Circuit**: Take merged crops from Stage 1, feed to reasoning module for chain-of-thought generation and final answer
- **Asymmetric Compute**: Perception stage scales via multiple rollouts; reasoning stage processes single merged crop context
- **Fine-tuning Independence**: Use LoRA to optimize perception circuit separately from reasoning backbone without coupling their parameter spaces
- **Context Engineering**: Maintain structured format (region coords, crops, reasoning chain, answer) rather than entangling tokens

## Implementation

The implementation requires three components: region detection, box fusion, and two-stage VLM inference.

First, implement region detection in Stage 1:

```python
import torch
from PIL import Image
import numpy as np

class PerceptionStage:
    """VLM stage for detecting task-relevant image regions."""

    def __init__(self, vlm_model, num_rollouts=3, detection_type='bbox'):
        """
        Args:
            vlm_model: Vision-language model for region detection
            num_rollouts: Number of self-consistency rollouts for Stage 1
            detection_type: 'bbox' for bounding boxes or 'point' for point coordinates
        """
        self.vlm = vlm_model
        self.num_rollouts = num_rollouts
        self.detection_type = detection_type

    def detect_regions(self, image, question):
        """
        Run self-consistent region detection.
        Args:
            image: PIL Image or tensor
            question: str, the query
        Returns:
            boxes: List of detected bounding boxes [num_rollouts, num_boxes, 4]
        """
        boxes_list = []

        # Run detection k times for self-consistency
        for _ in range(self.num_rollouts):
            # Prompt for bounding box detection (e.g., "[x1, y1, x2, y2], [x1, y1, x2, y2], ...")
            prompt = f"Identify regions relevant to: {question}\nReturn bounding boxes as [x1, y1, x2, y2]."

            # Get model prediction
            output = self.vlm.generate(image, prompt, max_tokens=256)

            # Parse bounding boxes from output
            boxes = self._parse_boxes(output, image.size)
            boxes_list.append(boxes)

        return boxes_list

    def _parse_boxes(self, output_text, image_size):
        """Parse model output into bounding box coordinates."""
        import re

        boxes = []
        # Regex to find patterns like [x1, y1, x2, y2]
        pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern, output_text)

        for match in matches:
            x1, y1, x2, y2 = [int(m) for m in match]
            # Clamp to image bounds
            x1 = max(0, min(x1, image_size[0]))
            y1 = max(0, min(y1, image_size[1]))
            x2 = max(x1, min(x2, image_size[0]))
            y2 = max(y1, min(y2, image_size[1]))
            boxes.append([x1, y1, x2, y2])

        return boxes
```

Next, implement Weighted Box Fusion to aggregate Stage 1 detections:

```python
def weighted_box_fusion(boxes_list, weights=None, iou_threshold=0.5):
    """
    Aggregate overlapping bounding boxes from multiple rollouts.
    Args:
        boxes_list: List of box lists [num_rollouts, num_boxes, 4]
        weights: Weights per rollout; default uniform
        iou_threshold: Minimum IoU to consider boxes as duplicates
    Returns:
        fused_boxes: List of aggregated boxes
    """
    if weights is None:
        weights = [1.0 / len(boxes_list)] * len(boxes_list)

    # Flatten all boxes with their weights
    all_boxes = []
    for rollout_idx, boxes in enumerate(boxes_list):
        for box in boxes:
            all_boxes.append((*box, weights[rollout_idx]))

    if not all_boxes:
        return []

    # Sort by confidence (weight)
    all_boxes = sorted(all_boxes, key=lambda x: x[4], reverse=True)

    fused = []
    used = set()

    for i, (x1, y1, x2, y2, weight) in enumerate(all_boxes):
        if i in used:
            continue

        # Find overlapping boxes
        cluster = [(x1, y1, x2, y2, weight)]
        for j in range(i + 1, len(all_boxes)):
            if j in used:
                continue
            x1_j, y1_j, x2_j, y2_j, weight_j = all_boxes[j]

            # Compute IoU
            inter_x1, inter_y1 = max(x1, x1_j), max(y1, y1_j)
            inter_x2, inter_y2 = min(x2, x2_j), min(y2, y2_j)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = (x2 - x1) * (y2 - y1) + (x2_j - x1_j) * (y2_j - y1_j) - inter_area
            iou = inter_area / (union_area + 1e-6)

            if iou > iou_threshold:
                cluster.append((x1_j, y1_j, x2_j, y2_j, weight_j))
                used.add(j)

        # Weighted average of cluster
        total_weight = sum(b[4] for b in cluster)
        x1_fused = sum(b[0] * b[4] for b in cluster) / total_weight
        y1_fused = sum(b[1] * b[4] for b in cluster) / total_weight
        x2_fused = sum(b[2] * b[4] for b in cluster) / total_weight
        y2_fused = sum(b[3] * b[4] for b in cluster) / total_weight

        fused.append([int(x1_fused), int(y1_fused), int(x2_fused), int(y2_fused)])

    return fused
```

Finally, integrate into the two-stage VLM forward pass:

```python
class TwoStageVLMReasoner:
    """End-to-end two-stage VLM inference."""

    def __init__(self, perception_vlm, reasoning_vlm):
        """
        Args:
            perception_vlm: VLM model fine-tuned for region detection
            reasoning_vlm: VLM model fine-tuned for reasoning (can be same model with different LoRA)
        """
        self.perception = PerceptionStage(perception_vlm, num_rollouts=3)
        self.reasoning_vlm = reasoning_vlm

    def forward(self, image, question):
        """
        Full two-stage reasoning.
        Args:
            image: PIL Image
            question: str query
        Returns:
            answer: Final predicted answer
        """
        # Stage 1: Perception - detect relevant regions
        boxes_list = self.perception.detect_regions(image, question)

        # Aggregate detections via WBF
        fused_boxes = weighted_box_fusion(boxes_list)

        # Crop image to fused regions
        crops = []
        for x1, y1, x2, y2 in fused_boxes:
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

        # Concatenate crops horizontally for reasoning
        if crops:
            # Simple concatenation; more sophisticated layouts possible
            from PIL import Image as PILImage
            total_width = sum(c.width for c in crops) + 10 * (len(crops) - 1)
            max_height = max(c.height for c in crops)
            concatenated = PILImage.new('RGB', (total_width, max_height), color='white')

            x_offset = 0
            for crop in crops:
                concatenated.paste(crop, (x_offset, 0))
                x_offset += crop.width + 10
        else:
            concatenated = image

        # Stage 2: Reasoning - generate answer from crops
        reasoning_prompt = f"Based on these image regions, answer: {question}\nLet's think step by step."
        answer = self.reasoning_vlm.generate(concatenated, reasoning_prompt, max_tokens=512)

        return answer
```

## Practical Guidance

| Parameter | Recommendation | Notes |
|-----------|-----------------|-------|
| Stage 1 rollouts | 3-5 | Higher rollouts improve perception accuracy; 3 is a good default. |
| IoU threshold (WBF) | 0.3-0.5 | Lower threshold merges more overlapping boxes; higher is more conservative. |
| Crop method | Bounded box or padded | Always pad boxes by 5-10% to include context around detected regions. |
| Reasoning prompt | Task-specific chain-of-thought | Explicitly ask for step-by-step reasoning to improve answer quality. |
| Fine-tuning strategy | Separate LoRA for perception + reasoning | Allows independent optimization; don't couple parameter updates. |

**When to Use**
- VLM tasks requiring visual reasoning on complex scenes (charts, tables, documents)
- When reducing token overhead is a priority (long reasoning chains)
- Tasks where perception and reasoning are conceptually separable (e.g., "find the table, then read values")
- Improving accuracy on benchmarks like VQA, OCR, spatial reasoning

**When NOT to Use**
- Simple single-step visual tasks (classification, detection) where two stages add overhead
- Scenes where reasoning requires full-image context (e.g., scene understanding)
- When compute budget is extremely tight and multiple rollouts are infeasible

**Common Pitfalls**
- Stage 1 detection too coarse (missing relevant regions) or too fine (over-segmenting); tune IoU threshold
- Not padding crops enough; regions should include surrounding context
- Using perception + reasoning models with incompatible architectures; keep the same backbone
- Forgetting to standardize prompts; consistency matters for self-consistency aggregation

## Reference

See https://arxiv.org/abs/2602.06566 for the full paper, which includes detailed comparisons with end-to-end reasoning, analysis of perception vs reasoning scaling, and benchmarks on VQA v2, GQA, and chart understanding tasks.
