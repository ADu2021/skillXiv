---
name: spatial-ssrl-spatial-understanding-rl
title: "Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.27606"
keywords: [Spatial Reasoning, Self-Supervised Learning, Vision-Language Models, Reinforcement Learning, 3D Understanding]
description: "Improve spatial reasoning in vision-language models through five automatically-formulated pretext tasks (patch shuffling, flipping, inpainting, depth ordering, 3D position) that require zero human annotation, using only RGB/RGB-D images and RL optimization for 4% accuracy gains."
---

# Title: Train Spatial Reasoning Through Vision-Only Pretext Tasks and RL

Vision-language models struggle with spatial relationships despite being trained on billions of image-text pairs. Spatial-SSRL fixes this by creating five pretext tasks that derive supervisory signals directly from image structure—no human annotation needed. Tasks range from simple (reorder shuffled patches) to geometric (predict 3D positions). The model learns through Group Relative Policy Optimization on automatically-generated question-answer pairs from existing datasets.

The key insight is that images contain all necessary geometry; you just need the right pretext tasks to make it explicit.

## Core Concept

**Self-Supervised Spatial Learning**:
- **Five Pretext Tasks**: Shuffled patch reordering, flipped patch recognition, patch inpainting, regional depth ordering, relative position prediction
- **Zero-Human-Annotation**: All ground truth derived from image structure or depth maps
- **GRPO Optimization**: Group Relative Policy Optimization on structured reasoning
- **Dual-Modality Support**: RGB-only tasks and RGB-D (depth-dependent) tasks
- **Automatic Dataset Generation**: 81K QA pairs from existing datasets (COCO, DIODE, MegaDepth)

## Architecture Overview

- **Pretext Task Set**: Two categories (RGB-free, depth-based) with automatic question generation
- **Training Data**: Spatial-SSRL-81k dataset with no manual annotation
- **GRPO Framework**: Reward combination (accuracy 90% + format 10%) using reasoning-style outputs
- **Cold-Start Phase**: Brief supervised fine-tuning before RL optimization
- **Target Models**: Applied to Qwen2.5-VL (3B, 7B variants)

## Implementation Steps

**1. Define Five Pretext Tasks with Automatic Question Generation**

Create task-specific question-answer generation from images.

```python
class SpatialPretext:
    @staticmethod
    def shuffled_patch_reordering(image, patch_size=32, num_patches=16):
        # Divide image into patches
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # Randomly shuffle
        perm = torch.randperm(num_patches)
        shuffled = patches[perm]

        # Question: restore original order
        question = "The image patches are shuffled. Reorder them to restore the original."
        answer = f"The correct order is: {list(range(num_patches))}"
        return question, answer

    @staticmethod
    def flipped_patch_recognition(image, patch_size=32):
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # Flip one patch horizontally
        flip_idx = torch.randint(len(patches), (1,)).item()
        patches[flip_idx] = torch.flip(patches[flip_idx], dims=[-1])

        question = f"Which patch is flipped horizontally?"
        answer = f"Patch number {flip_idx} is flipped."
        return question, answer

    @staticmethod
    def cropped_patch_inpainting(image, patch_size=32, mask_idx=None):
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        if mask_idx is None:
            mask_idx = torch.randint(len(patches), (1,)).item()

        # Mask one patch
        masked_patches = patches.clone()
        masked_patches[mask_idx] = torch.zeros_like(masked_patches[mask_idx])

        # Candidates: 4 random patches including correct one
        candidates = [patches[mask_idx]] + [patches[torch.randint(len(patches), (1,))] for _ in range(3)]
        correct_idx = 0

        question = f"Which patch completes the missing region?"
        answer = f"The correct patch is option {correct_idx}."
        return question, answer, candidates

    @staticmethod
    def regional_depth_ordering(image, depth_map, num_regions=3):
        # Divide into regions, compute average depth
        h, w = image.shape[-2:]
        region_h, region_w = h // num_regions, w // num_regions

        regions = []
        for i in range(num_regions):
            region_depth = depth_map[i*region_h:(i+1)*region_h, :].mean()
            regions.append((i, region_depth.item()))

        # Sort by depth
        sorted_regions = sorted(regions, key=lambda x: x[1])

        question = "Rank these three image regions from closest to farthest from camera."
        answer = f"From closest to farthest: {[r[0] for r in sorted_regions]}"
        return question, answer

    @staticmethod
    def relative_position_prediction(image, depth_map):
        # Pick two points, ask about relative position
        pt1 = torch.tensor([image.shape[-2]//4, image.shape[-1]//4])
        pt2 = torch.tensor([image.shape[-2]*3//4, image.shape[-1]*3//4])

        depth1 = depth_map[pt1[0], pt1[1]]
        depth2 = depth_map[pt2[0], pt2[1]]

        question = f"Is point A closer or farther than point B?"
        answer = "closer" if depth1 < depth2 else "farther"
        return question, answer
```

**2. Generate Spatial-SSRL-81k Dataset**

Create automatic dataset from existing image collections.

```python
def generate_spatial_dataset(image_dir, depth_dir, output_size=81000):
    dataset = []

    image_paths = list(Path(image_dir).glob("*.jpg"))[:output_size // 5]

    for img_path in image_paths:
        image = Image.open(img_path)
        depth_path = Path(depth_dir) / (img_path.stem + ".npy")
        depth = np.load(depth_path) if depth_path.exists() else None

        # Generate 5 QA pairs per image
        qa_pairs = [
            SpatialPretext.shuffled_patch_reordering(image),
            SpatialPretext.flipped_patch_recognition(image),
        ]

        if depth is not None:
            qa_pairs.extend([
                SpatialPretext.regional_depth_ordering(image, depth),
                SpatialPretext.relative_position_prediction(image, depth),
            ])

        for q, a in qa_pairs:
            dataset.append({
                'image_path': str(img_path),
                'question': q,
                'answer': a
            })

    return dataset[:output_size]
```

**3. Implement GRPO Training Loop**

Train the model using Group Relative Policy Optimization.

```python
class SpatialGRPOTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute_spatial_reward(self, generated_answer, reference_answer):
        # Exact match for spatial reasoning
        exact_match = float(generated_answer == reference_answer)

        # Format compliance (answer in proper boxes)
        has_box = '\\boxed{' in generated_answer
        format_reward = float(has_box) if has_box else 0.1

        return 0.9 * exact_match + 0.1 * format_reward

    def train_step(self, batch, lr=1e-5):
        images, questions, answers = batch

        # Generate outputs
        inputs = self.tokenizer(questions, images)
        outputs = self.model.generate(inputs, max_length=256)
        generated_answers = self.tokenizer.decode(outputs)

        # Compute rewards per sample
        rewards = [
            self.compute_spatial_reward(gen, ref)
            for gen, ref in zip(generated_answers, answers)
        ]

        # Group Relative Policy Optimization
        # Separate into groups by reward quartile
        sorted_indices = np.argsort(rewards)
        group_size = len(rewards) // 4

        losses = []
        for group_idx in range(4):
            group_start = group_idx * group_size
            group_end = (group_idx + 1) * group_size
            group_indices = sorted_indices[group_start:group_end]

            # Positive examples: high-reward samples
            # Negative examples: low-reward samples
            if group_idx >= 2:
                # High reward group
                loss = self.compute_grpo_loss(
                    inputs[group_indices], answers[group_indices], positive=True
                )
            else:
                # Low reward group
                loss = self.compute_grpo_loss(
                    inputs[group_indices], answers[group_indices], positive=False
                )

            losses.append(loss)

        total_loss = sum(losses) / len(losses)
        return total_loss.item()

    def compute_grpo_loss(self, inputs, targets, positive=True):
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Standard language modeling loss
        target_ids = self.tokenizer(targets).input_ids
        loss = F.cross_entropy(logits, target_ids)

        # Scale by relative position
        return loss if not positive else loss * 0.5
```

**4. Implement Cold-Start Supervised Fine-Tuning**

Before RL, brief SFT stabilizes training.

```python
def cold_start_sft(model, dataset, num_epochs=1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        for batch in dataset:
            images, questions, answers = batch

            # Supervised next-token prediction
            outputs = model(images=images, questions=questions, answers=answers)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
```

## Practical Guidance

**When to Use**:
- Improving spatial reasoning in existing VLMs
- Robotics or navigation tasks requiring 3D understanding
- Models struggling with geometric reasoning despite strong language performance

**Hyperparameters**:
- num_epochs_sft: 1 (brief cold-start is sufficient)
- lr_grpo: 1e-5 (lower than supervised learning)
- reward_accuracy_weight: 0.9 (emphasize correctness over format)

**When NOT to Use**:
- Models already trained on spatial-intensive datasets
- High-precision vision tasks requiring pixel-level accuracy
- Scenarios where additional training is prohibitively expensive

**Pitfalls**:
- **Insufficient pretext task diversity**: Five tasks cover basic geometry; add task-specific variants for your domain
- **Depth map quality**: Depth-dependent tasks require accurate depth estimates; synthetic or coarse depth can mislead
- **Overfitting to pretext tasks**: Spatial reasoning sometimes doesn't transfer to downstream tasks; validate transfer explicitly

**Key Insight**: Unlike typical self-supervised learning, you don't need paired data or external annotation. The geometric properties of images themselves provide infinite supervision.

## Reference

arXiv: https://arxiv.org/abs/2510.27606
