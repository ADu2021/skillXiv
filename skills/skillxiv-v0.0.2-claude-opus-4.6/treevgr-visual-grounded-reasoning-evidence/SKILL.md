---
name: treevgr-visual-grounded-reasoning-evidence
title: "Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07999"
keywords: [Visual Reasoning, Evidence Grounding, Reinforcement Learning, Object Localization, VQA]
description: "Train vision-language models to produce visually grounded reasoning by enforcing traceable evidence via bounding box localization, using a novel benchmark (TreeBench) and RL-based training with dual IoU rewards for both recall and precision."
---

# TreeVGR: Visual Grounded Reasoning with Traceable Evidence

Standard vision-language models answer questions but rarely show their visual reasoning. TreeVGR introduces a training paradigm and benchmark that demand evidence: the model must identify and localize specific regions supporting its answers. TreeBench is a diagnostic benchmark of 405 carefully curated VQA pairs emphasizing fine perception in cluttered scenes. TreeVGR trains models to produce bounding boxes alongside answers, using reinforcement learning with dual IoU metrics that reward both recall (did you find all relevant objects?) and precision (did you avoid false positives?).

This approach transforms VQA from a classification task into a grounded reasoning task, improving explainability and robustness.

## Core Concept

Visual reasoning without visual evidence is brittle. Models can guess correctly by exploiting spurious correlations. TreeVGR enforces visual grounding: to answer a question, the model must identify which objects or regions support the answer. The training reward combines three signals: (1) answer correctness (did you get it right?), (2) bounding box recall (did you localize all relevant objects?), (3) bounding box precision (did you avoid false positives?). This multi-faceted reward encourages genuine visual understanding rather than shortcut learning.

The benchmark carefully samples cluttered scenes (dense objects) and emphasizes second-order reasoning (reasoning about attributes, spatial relations, transformations) beyond simple localization.

## Architecture Overview

- **Base Model**: Qwen2.5-VL-7B vision-language model
- **Question-Guided Vision**: Extract features conditioned on query, focus on relevant regions
- **Grounding Head**: Predict bounding boxes for evidence regions
- **Answer Head**: Generate answer text conditioned on grounded features
- **TreeBench**: 405 VQA pairs with bounding box annotations in cluttered scenes
- **Dual IoU Rewards**: Separate metrics for recall and precision in localization

## Implementation

### Step 1: Create TreeBench-Style Evaluation Dataset

Curate a dataset emphasizing fine visual perception and evidence-based reasoning. Select dense object scenes and annotate questions across two categories:

```python
import json
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class TreeBenchExample:
    image_path: str
    question: str
    answer: str
    bounding_boxes: List[Tuple[int, int, int, int]]  # x1, y1, x2, y2
    category: str  # "perception" or "reasoning"
    subcategory: str  # e.g., "color", "spatial", "transform"

def create_treebench_dataset(num_examples=405, num_images=1000):
    """
    Create TreeBench: VQA dataset emphasizing visual perception
    in dense object scenes with bounding box annotations.
    """
    dataset = []

    # Sample dense object images (cluttered scenes)
    images = sample_dense_object_images(num_images)

    for img_idx, image in enumerate(images):
        # Perception questions (5 subtasks)
        perception_questions = [
            ("color", f"What color is the {random_object(image)}?"),
            ("material", f"What material is the {random_object(image)}?"),
            ("attribute", f"Describe the {random_object(image)}."),
            ("spatial", f"Where is the {random_object(image)} located?"),
            ("count", f"How many objects are {random_property(image)}?"),
        ]

        # Reasoning questions (5 subtasks)
        reasoning_questions = [
            ("perspective", f"What would you see from {random_direction(image)}?"),
            ("containment", f"Is the {random_object(image)} inside or outside?"),
            ("transformation", f"If we rotate the {random_object(image)} 90°, where would it be?"),
            ("relationship", f"What is the relationship between {random_object(image)} and {random_object(image)}?"),
            ("inference", f"Based on the scene, what might happen next?"),
        ]

        all_questions = perception_questions + reasoning_questions

        for category, question in all_questions:
            # Generate answer and get ground truth bounding boxes
            answer = generate_answer(image, question)
            bboxes = extract_relevant_bboxes(image, question, answer)

            example = TreeBenchExample(
                image_path=image.path,
                question=question,
                answer=answer,
                bounding_boxes=bboxes,
                category="perception" if category in [c[0] for c in perception_questions] else "reasoning",
                subcategory=category
            )
            dataset.append(example)

    return dataset

def sample_dense_object_images(num_images: int) -> List:
    """Sample images with many objects (cluttered scenes)."""
    from datasets import load_dataset
    coco = load_dataset("coco", split="train")

    dense_images = []
    for img_data in coco:
        annotations = img_data["annotations"]
        if len(annotations) >= 10:  # At least 10 objects
            dense_images.append(img_data)
        if len(dense_images) >= num_images:
            break

    return dense_images

def extract_relevant_bboxes(image, question: str, answer: str) -> List[Tuple]:
    """
    Extract bounding boxes for objects mentioned in question/answer.
    In practice, use COCO annotations or manual annotation.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.transforms import functional as F

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Prepare image
    img_tensor = F.to_tensor(image)
    with torch.no_grad():
        predictions = model([img_tensor])

    # Filter boxes based on question/answer entities
    relevant_boxes = []
    for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]):
        if score > 0.5:  # Confidence threshold
            relevant_boxes.append(tuple(box.tolist()))

    return relevant_boxes
```

### Step 2: Implement Vision-Grounded Answer Generation

Train the model to produce both answers and bounding box evidence. First stage uses supervised fine-tuning with ground truth boxes:

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModel

class TreeVGRGroundedModel(nn.Module):
    def __init__(self, base_model_name="Qwen/Qwen2.5-VL-7B"):
        super().__init__()

        # Vision-language base model
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)

        # Grounding head: predict bounding boxes
        self.grounding_head = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 4)  # x1, y1, x2, y2
        )

        # Answer head: generate text answer
        self.answer_head = self.base_model.model.lm_head  # Share with base model

    def forward(self, image, question, question_type="general"):
        """
        Forward pass: encode image and question, predict boxes and answer.
        """
        # Encode image and question
        image_features = self.base_model.encode_image(image)  # [B, H, W, D]
        question_tokens = self.base_model.tokenize(question)
        question_features = self.base_model.encode_text(question_tokens)

        # Fuse features (attention over image conditioned on question)
        batch_size = image_features.shape[0]
        h, w = image_features.shape[1:3]
        image_flat = image_features.reshape(batch_size, -1, image_features.shape[-1])

        # Compute attention between question and image regions
        attn_scores = torch.matmul(
            question_features.unsqueeze(1),  # [B, 1, D]
            image_flat.transpose(1, 2)  # [B, D, H*W]
        )  # [B, 1, H*W]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weighted image features
        attended_features = torch.matmul(
            attn_weights,
            image_flat
        )  # [B, 1, D]

        # Predict bounding boxes (center coordinates and size)
        bbox_logits = self.grounding_head(attended_features.squeeze(1))  # [B, 4]

        # Generate answer via LLM decoder
        answer_logits = self.answer_head(
            torch.cat([attended_features, question_features], dim=-1)
        )

        return bbox_logits, answer_logits

def sft_training(model, train_dataset, num_epochs=2):
    """
    Stage 1: Supervised fine-tuning with ground truth bounding boxes.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch in train_dataset:
            images = batch["image"]
            questions = batch["question"]
            answers = batch["answer"]
            bboxes = batch["bounding_boxes"]  # Ground truth

            # Normalize boxes to [0, 1]
            h, w = images.shape[2:4]
            bboxes_norm = bboxes / torch.tensor([w, h, w, h])

            # Forward pass
            bbox_preds, answer_logits = model(images, questions)

            # Loss for bounding boxes
            bbox_loss = loss_fn(bbox_preds, bboxes_norm)

            # Loss for answers (cross-entropy)
            answer_loss = nn.CrossEntropyLoss()(
                answer_logits.view(-1, answer_logits.size(-1)),
                answers.view(-1)
            )

            total_loss = 0.7 * bbox_loss + 0.3 * answer_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
```

### Step 3: Reinforcement Learning with Dual IoU Rewards

Stage 2 refines predictions using RL with rewards for answer correctness and bounding box quality:

```python
def compute_iou(pred_box: torch.Tensor, gt_box: torch.Tensor) -> float:
    """Compute Intersection-over-Union between two boxes."""
    x1_inter = max(pred_box[0].item(), gt_box[0].item())
    y1_inter = max(pred_box[1].item(), gt_box[1].item())
    x2_inter = min(pred_box[2].item(), gt_box[2].item())
    y2_inter = min(pred_box[3].item(), gt_box[3].item())

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area.item() / union_area.item()

def compute_grounding_reward(pred_boxes: List[torch.Tensor],
                            gt_boxes: List[torch.Tensor]) -> Tuple[float, float]:
    """
    Compute dual IoU rewards: recall and precision.

    Recall: fraction of ground truth boxes matched by predictions
    Precision: fraction of predicted boxes that match ground truth
    """
    recall_sum = 0
    for gt_box in gt_boxes:
        best_iou = max([compute_iou(p, gt_box) for p in pred_boxes], default=0)
        recall_sum += 1 if best_iou > 0.5 else 0

    recall = recall_sum / len(gt_boxes) if gt_boxes else 1.0

    precision_sum = 0
    for pred_box in pred_boxes:
        best_iou = max([compute_iou(pred_box, g) for g in gt_boxes], default=0)
        precision_sum += 1 if best_iou > 0.5 else 0

    precision = precision_sum / len(pred_boxes) if pred_boxes else 1.0

    return recall, precision

def rl_training(model, sft_model, train_dataset, num_epochs=1):
    """
    Stage 2: Reinforcement learning with dual IoU rewards.
    Optimizes answer accuracy + bounding box recall + precision.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    for epoch in range(num_epochs):
        for batch in train_dataset:
            images = batch["image"]
            questions = batch["question"]
            answers = batch["answer"]
            gt_boxes = batch["bounding_boxes"]

            # Generate samples with temperature
            with torch.no_grad():
                for _ in range(4):  # Generate 4 samples per input
                    bbox_preds, answer_logits = model(images, questions)

                    # Extract answer from logits
                    pred_answers = torch.argmax(answer_logits, dim=-1)

                    # Denormalize boxes
                    h, w = images.shape[2:4]
                    bbox_preds_denorm = bbox_preds * torch.tensor([w, h, w, h])

                    # Compute rewards
                    answer_correct = (pred_answers == answers).float()
                    recall, precision = compute_grounding_reward(
                        bbox_preds_denorm, gt_boxes
                    )

                    # Combined reward
                    reward = (
                        0.6 * answer_correct +
                        0.2 * recall +
                        0.2 * precision
                    )

                    # Policy gradient loss (minimize negative reward)
                    loss = -reward.mean()
                    loss.backward()

            optimizer.step()
            optimizer.zero_grad()
```

## Practical Guidance

| Component | Recommended Value | Notes |
|---|---|---|
| Base Model | Qwen2.5-VL-7B | Strong vision-language foundation |
| SFT Epochs | 2 | Enough to establish baseline |
| RL Epochs | 1 | Additional refinement |
| SFT Learning Rate | 2e-5 | Standard for instruction tuning |
| RL Learning Rate | 5e-6 | Conservative for stability |
| Answer Loss Weight | 0.7 | Prioritize correctness |
| Bounding Box Loss Weight | 0.3 | Secondary emphasis during SFT |
| Answer Reward Weight | 0.6 | Primary in RL |
| Recall Reward Weight | 0.2 | Coverage of relevant objects |
| Precision Reward Weight | 0.2 | Avoidance of false positives |
| IoU Threshold | 0.5 | Standard object detection threshold |
| TreeBench Samples | 405 | Diagnostic benchmark size |
| Dense Object Scenes | ~1000 base images | Sample for dataset creation |

**When to use TreeVGR:**
- VQA tasks emphasizing explainability and visual grounding
- Scenarios where you need to debug model failures (boxes reveal reasoning)
- Applications requiring traceable evidence for compliance
- Fine-grained visual reasoning in cluttered scenes
- When you want to prevent spurious correlation shortcuts

**When NOT to use TreeVGR:**
- High-speed inference (bounding box prediction adds latency)
- Simple factual QA where grounding is unnecessary
- Scenes with very few objects (grounding is harder to interpret)
- Memory-constrained deployment (additional grounding head)
- Tasks where answer-only output is sufficient

**Common pitfalls:**
- Not normalizing bounding boxes to [0, 1] range before loss computation
- IoU threshold too high (0.7+), making reward signal sparse
- Forgetting to balance answer vs grounding rewards, leading to degenerate solutions
- Not using ground truth boxes during SFT, limiting convergence
- Treating recall and precision equally (adjust weights per domain)
- Using arbitrary box coordinates instead of object centers
- Not handling edge cases (images with 0 boxes, negative boxes)

## Reference

Li, S., Zhang, X., Chen, Y., & Wang, H. (2025). Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology. arXiv:2507.07999. https://arxiv.org/abs/2507.07999
