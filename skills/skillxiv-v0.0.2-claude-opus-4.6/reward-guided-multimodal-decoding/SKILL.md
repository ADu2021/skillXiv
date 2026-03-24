---
name: reward-guided-multimodal-decoding
title: "Controlling Multimodal LLMs via Reward-guided Decoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.11616
keywords: [multimodal-llm, reward-modeling, decoding-control, object-grounding, hallucination-reduction]
description: "Control MLLM output characteristics at inference time using separate reward models for precision and recall, enabling dynamic trade-offs without retraining."
---

# Controlling Multimodal LLMs via Reward-guided Decoding

## Core Concept

Multimodal Large Language Models (MLLMs) often suffer from object hallucinations: describing objects not present in images. Standard training approaches treat output quality as all-or-nothing, but users often have different needs (high-precision descriptions vs. comprehensive object detection).

Reward-guided decoding enables dynamic control of MLLM outputs at inference time using separate reward models for different properties (precision, recall). Users can adjust these rewards on-the-fly without retraining, achieving real-time control over quality dimensions.

## Architecture Overview

- **Dual Reward Models**: Separate models for object precision (avoid hallucinations) and recall (detect all objects)
- **Weighted Reward Combination**: Blend precision and recall rewards with user-controlled weights during decoding
- **Search Breadth Control**: Adjust beam search or sampling width to balance computational cost vs. quality
- **Inference-Time Adaptation**: No retraining needed; control is applied during generation
- **Composable Objectives**: Framework supports adding more reward dimensions (factuality, diversity, etc.)

## Implementation Steps

### 1. Build Precision Reward Model

Create a reward model that detects object hallucinations (objects mentioned but not in image).

```python
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class PrecisionRewardModel(nn.Module):
    """
    Reward model for detecting hallucinated objects
    High reward = no hallucinations, Low reward = hallucinations present
    """
    def __init__(self, vision_model_name='openai/clip-vit-base-patch32'):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(vision_model_name)
        self.processor = CLIPProcessor.from_pretrained(vision_model_name)

        # Fine-tuned head for hallucination detection
        self.hallucination_detector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output: probability of hallucination
        )

    def forward(self, image, caption):
        """
        Score caption for hallucinations.

        Args:
            image: PIL Image or image tensor
            caption: generated caption text

        Returns:
            precision_score: 1.0 = no hallucinations, 0.0 = pure hallucination
        """
        # Tokenize caption
        caption_tokens = self.processor.tokenizer(
            caption,
            return_tensors='pt',
            truncation=True,
            max_length=77
        )

        # CLIP embeddings
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(
                self.processor(image, return_tensors='pt')['pixel_values']
            )
            text_features = self.clip_model.get_text_features(**caption_tokens)

        # Compute alignment score
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        alignment = (image_features @ text_features.T).squeeze()

        # Extract objects from caption
        objects = self._extract_objects(caption)

        # For each object, check if it appears in the image
        hallucination_prob = 0.0
        for obj in objects:
            obj_tokens = self.processor.tokenizer(
                obj,
                return_tensors='pt',
                truncation=True,
                max_length=77
            )

            with torch.no_grad():
                obj_features = self.clip_model.get_text_features(**obj_tokens)
                obj_features = obj_features / obj_features.norm(dim=-1, keepdim=True)

            # Check if object is visually grounded
            obj_alignment = (image_features @ obj_features.T).squeeze()

            # Low alignment = hallucination
            if obj_alignment < 0.3:
                hallucination_prob += 1.0 / len(objects)

        # Precision score: 1 - hallucination probability
        precision_score = 1.0 - hallucination_prob
        return precision_score

    def _extract_objects(self, caption):
        """Extract mentioned objects from caption using NLP"""
        import re
        # Simple heuristic: nouns after articles
        objects = re.findall(r'(?:a|an|the)\s+(\w+)', caption.lower())
        return objects
```

### 2. Build Recall Reward Model

Create a reward model for completeness (detecting objects that should be mentioned).

```python
class RecallRewardModel(nn.Module):
    """
    Reward model for object completeness (recall)
    High reward = detected most objects in image
    """
    def __init__(self, detection_model_name='facebook/detr-resnet50'):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(detection_model_name)
        self.detection_model = AutoModelForObjectDetection.from_pretrained(
            detection_model_name
        )

    def forward(self, image, caption):
        """
        Score caption for object completeness.

        Args:
            image: PIL Image
            caption: generated caption

        Returns:
            recall_score: 1.0 = detected all objects, 0.0 = missed everything
        """
        # Detect objects in image
        inputs = self.processor(image, return_tensors='pt')
        with torch.no_grad():
            outputs = self.detection_model(**inputs)

        # Post-process detections
        detected_objects = self._extract_detected_objects(outputs)

        # Extract mentioned objects
        mentioned_objects = self._extract_mentioned_objects(caption)

        # Compute recall: how many detected objects are mentioned?
        if len(detected_objects) == 0:
            return 1.0  # Perfect if nothing to detect

        matched = 0
        for detected in detected_objects:
            if self._is_mentioned(detected, mentioned_objects):
                matched += 1

        recall_score = matched / len(detected_objects)
        return recall_score

    def _extract_detected_objects(self, outputs):
        """Extract object labels and scores from DETR output"""
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[(100, 100)],  # Dummy size
            threshold=0.5
        )[0]

        objects = []
        for score, label in zip(results['scores'], results['labels']):
            obj_name = self.detection_model.config.id2label[label.item()]
            objects.append((obj_name, score.item()))

        return objects

    def _extract_mentioned_objects(self, caption):
        """Extract object names mentioned in caption"""
        import re
        mentioned = re.findall(r'(?:a|an|the)\s+(\w+)', caption.lower())
        return set(mentioned)

    def _is_mentioned(self, detected_obj, mentioned_objects):
        """Check if detected object appears in mentioned objects"""
        obj_name, score = detected_obj
        # Fuzzy matching: check if object name or substring is mentioned
        for mentioned in mentioned_objects:
            if obj_name.lower() in mentioned.lower() or mentioned.lower() in obj_name.lower():
                return True
        return False
```

### 3. Implement Reward-Guided Beam Search

Modify beam search to incorporate rewards during decoding.

```python
class RewardGuidedBeamSearch:
    """
    Beam search that incorporates rewards at each step
    """
    def __init__(self, model, precision_reward, recall_reward,
                 beam_width=4, max_length=50):
        self.model = model
        self.precision_reward = precision_reward
        self.recall_reward = recall_reward
        self.beam_width = beam_width
        self.max_length = max_length

    def search(self, image, input_ids, precision_weight=0.5, recall_weight=0.5):
        """
        Guided beam search with dynamic reward weighting

        Args:
            image: input image
            input_ids: initial tokens
            precision_weight: importance of precision (avoiding hallucinations)
            recall_weight: importance of recall (detecting all objects)

        Returns:
            best_sequence: best generated caption
        """
        # Initialize beams: each beam is (tokens, log_prob, reward_score)
        beams = [(input_ids.clone(), 0.0, 0.0)]

        for step in range(self.max_length):
            candidates = []

            for beam_tokens, beam_log_prob, beam_reward in beams:
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model.generate(
                        image, beam_tokens,
                        max_new_tokens=1,
                        output_scores=True
                    )

                # Expand to top-k candidates
                logits = outputs.scores[0]
                top_k_logits, top_k_indices = torch.topk(logits, self.beam_width)

                for logit, token_idx in zip(top_k_logits, top_k_indices):
                    new_tokens = torch.cat([beam_tokens, token_idx.unsqueeze(0)])

                    # Decode to text
                    caption = self.model.tokenizer.decode(new_tokens)

                    # Compute reward scores
                    precision = self.precision_reward(image, caption)
                    recall = self.recall_reward(image, caption)

                    # Combined score: likelihood + weighted reward
                    combined_score = (
                        logit.item() +  # Language model likelihood
                        precision_weight * precision +
                        recall_weight * recall
                    )

                    candidates.append((new_tokens, logit.item(), combined_score))

            # Keep top beams
            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = candidates[:self.beam_width]

            # Check for termination
            if any(token == self.model.eos_token_id for tokens, _, _ in beams
                   for token in tokens[-1:]):
                break

        # Return best sequence
        best_tokens, _, _ = beams[0]
        best_caption = self.model.tokenizer.decode(best_tokens)
        return best_caption
```

### 4. Implement Dynamic Reward Weighting

Allow users to adjust reward importance during inference.

```python
class DynamicRewardController:
    """
    Control reward weighting and search breadth dynamically
    """
    def __init__(self, min_precision=0.0, max_precision=1.0,
                 min_recall=0.0, max_recall=1.0):
        self.precision_weight = 0.5
        self.recall_weight = 0.5
        self.beam_width = 4
        self.search_breadth = 1.0  # 1.0 = full search, 0.1 = 10% of candidates

        self.min_precision = min_precision
        self.max_precision = max_precision
        self.min_recall = min_recall
        self.max_recall = max_recall

    def set_precision_focus(self, focus_level):
        """
        Set precision focus (0.0 = ignore precision, 1.0 = maximize precision)
        """
        self.precision_weight = max(0.0, min(1.0, focus_level))
        self.recall_weight = 1.0 - self.precision_weight

    def set_recall_focus(self, focus_level):
        """
        Set recall focus (0.0 = ignore recall, 1.0 = maximize recall)
        """
        self.recall_weight = max(0.0, min(1.0, focus_level))
        self.precision_weight = 1.0 - self.recall_weight

    def set_search_breadth(self, breadth):
        """
        Set search breadth (0.0-1.0)
        1.0 = full search (slower, better quality)
        0.1 = limited search (faster, may miss better options)
        """
        self.search_breadth = max(0.0, min(1.0, breadth))
        self.beam_width = max(1, int(8 * breadth))  # Beam width scales with breadth

    def get_config(self):
        """Return current configuration"""
        return {
            'precision_weight': self.precision_weight,
            'recall_weight': self.recall_weight,
            'beam_width': self.beam_width,
            'search_breadth': self.search_breadth
        }
```

### 5. Integration with MLLM Inference

Integrate reward-guided decoding into the MLLM pipeline.

```python
class RewardGuidedMLLM:
    """
    MLLM with reward-guided decoding
    """
    def __init__(self, mllm_model, precision_reward, recall_reward):
        self.mllm = mllm_model
        self.precision_reward = precision_reward
        self.recall_reward = recall_reward
        self.controller = DynamicRewardController()
        self.beam_search = RewardGuidedBeamSearch(
            mllm_model, precision_reward, recall_reward
        )

    def generate_caption(self, image, precision_focus=0.5, recall_focus=0.5,
                        search_breadth=0.8):
        """
        Generate caption with dynamic reward control

        Args:
            image: input image
            precision_focus: 0-1, how much to avoid hallucinations
            recall_focus: 0-1, how much to detect all objects
            search_breadth: 0-1, how thorough to search (slower with higher values)
        """
        # Set controller parameters
        self.controller.set_precision_focus(precision_focus)
        self.controller.set_recall_focus(recall_focus)
        self.controller.set_search_breadth(search_breadth)

        # Extract image features
        image_inputs = self.mllm.processor(image, return_tensors='pt')

        # Run reward-guided beam search
        caption = self.beam_search.search(
            image,
            input_ids=torch.tensor([[self.mllm.tokenizer.bos_token_id]]),
            precision_weight=self.controller.precision_weight,
            recall_weight=self.controller.recall_weight
        )

        return caption
```

### 6. Evaluation and Validation

Test reward-guided decoding on grounding tasks.

```python
def evaluate_reward_guided_decoding(mllm, test_images, ground_truth_captions,
                                    precision_levels, recall_levels):
    """
    Evaluate trade-offs between precision and recall
    """
    results = {}

    for precision in precision_levels:
        for recall in recall_levels:
            captions = []
            precision_scores = []
            recall_scores = []

            for image, gt_caption in zip(test_images, ground_truth_captions):
                caption = mllm.generate_caption(
                    image,
                    precision_focus=precision,
                    recall_focus=recall
                )
                captions.append(caption)

                # Evaluate
                p_score = mllm.precision_reward(image, caption).item()
                r_score = mllm.recall_reward(image, caption).item()

                precision_scores.append(p_score)
                recall_scores.append(r_score)

            results[(precision, recall)] = {
                'precision': sum(precision_scores) / len(precision_scores),
                'recall': sum(recall_scores) / len(recall_scores),
                'captions': captions
            }

    return results
```

## Practical Guidance

### Hyperparameters & Configuration

- **Beam Width**: 4-8 (larger = slower but better quality)
- **Precision Weight**: 0.0-1.0 (0.8+ to minimize hallucinations)
- **Recall Weight**: 0.0-1.0 (0.7+ to catch all objects)
- **Search Breadth**: 0.5-1.0 (higher = better quality, slower inference)
- **Token Temperature**: 0.7-0.9 (balance diversity and confidence)

### When to Use Reward-Guided Decoding

- You want to control precision-recall trade-off without retraining
- Object hallucinations are a problem in your MLLM
- Users need different output characteristics (some want precision, others completeness)
- You have resources for separate precision and recall reward models
- Inference-time latency allows for expanded search

### When NOT to Use Reward-Guided Decoding

- You only need single output mode (no dynamic control needed)
- Inference speed is critical (beam search adds overhead)
- You can't afford separate reward models (memory constrained)
- Your base MLLM already has low hallucination rates
- You need absolute minimum latency

### Common Pitfalls

1. **Misaligned Reward Models**: If precision and recall models are poorly trained, decoding will be misdirected. Validate rewards first.
2. **Over-Expansion of Beams**: Too many beams increases latency without quality gains. Start with beam_width=4.
3. **No Baseline Comparison**: Compare against standard greedy/sampling decoding to ensure improvement.
4. **Ignoring Computational Cost**: Beam search is slower. Profile end-to-end latency.
5. **Conflicting Objectives**: If precision and recall are too opposed, weighted combination may fail. Consider multi-objective optimization.

## Reference

Reward-Guided Multimodal Decoding (2508.11616): https://arxiv.org/abs/2508.11616

Control MLLM outputs at inference time using separate precision and recall reward models, enabling dynamic trade-offs between avoiding hallucinations and detecting all objects without retraining.
