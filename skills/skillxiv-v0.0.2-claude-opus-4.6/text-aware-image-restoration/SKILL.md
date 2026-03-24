---
name: text-aware-image-restoration
title: "Text-Aware Image Restoration with Diffusion Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.09993"
keywords: [image restoration, diffusion models, OCR, text preservation, multi-task learning]
description: "Restore degraded images while preserving textual fidelity using TeReDiff, a multi-task diffusion framework integrating text spotting with U-Net features and VLM-verified dataset curation."
---

# Text-Aware Image Restoration with Diffusion Models

## Core Concept

Text-Aware Image Restoration (TAIR) addresses the overlooked problem of preserving textual fidelity alongside visual quality during image restoration. Existing diffusion methods generate visually plausible but textually incorrect content ("text-image hallucination"). TeReDiff solves this via multi-task learning: a diffusion U-Net handles visual restoration while a parallel text-spotting module detects and preserves text instances, guided by dynamically generated text prompts during inference.

## Architecture Overview

- **TeReDiff Three-Module Design**: Lightweight degradation removal (SwinIR-based), diffusion-based restoration (U-Net + ControlNet), transformer-based text spotting
- **Three-Stage Training**: Stage 1 trains diffusion components; Stage 2 trains text-spotting using diffusion features; Stage 3 jointly optimizes both
- **SA-Text Dataset**: 100K high-quality images with dense text annotations, VLM-verified using Qwen2.5-VL and OVIS2
- **Text Prompt Guidance**: During inference, detected texts dynamically generate prompts (e.g., "A realistic scene where the texts [sign], [board]... appear clearly")
- **Feature Innovation**: Diffusion U-Net features provide superior text detection compared to traditional ResNet backbones

## Implementation

### Step 1: Dataset Curation Pipeline

```python
import torch
from PIL import Image
import requests
import json

class SA_TextCurationPipeline:
    """
    Creates high-quality TAIR dataset via automated text detection,
    dual VLM verification, and blur filtering.
    """

    def __init__(self):
        self.text_detector = TextDetector()  # CRAFT or similar
        self.vlm_verifiers = [
            VLMModel('Qwen2.5-VL'),
            VLMModel('OVIS2')
        ]

    def curate_single_image(self, image_path, image_quality_threshold=0.7):
        """
        Process single image through curation pipeline.
        Returns (image, text_annotations, quality_score) or None if rejected.
        """

        image = Image.open(image_path)

        # Step 1: Full image text detection
        full_texts = self.text_detector.detect(image)

        # Step 2: Crop-based re-detection for small text
        crops = self._create_crops(image, crop_size=512)
        cropped_texts = []

        for crop in crops:
            crop_texts = self.text_detector.detect(crop)
            cropped_texts.extend(crop_texts)

        # Merge detections, remove duplicates
        all_texts = self._deduplicate_detections(full_texts + cropped_texts)

        # Step 3: VLM verification of detected text
        verified_texts = []

        for detected_text in all_texts:
            # Verify with multiple VLMs
            verifications = []
            for vlm in self.vlm_verifiers:
                recognition = vlm.recognize_text(image, detected_text['bbox'])
                confidence = self._compute_match_confidence(
                    detected_text['text'],
                    recognition
                )
                verifications.append(confidence)

            # Accept if both VLMs agree (confidence > 0.8)
            avg_confidence = sum(verifications) / len(verifications)
            if avg_confidence > 0.8:
                verified_texts.append({
                    'text': detected_text['text'],
                    'bbox': detected_text['bbox'],
                    'confidence': avg_confidence
                })

        # Step 4: VLM-based blur filtering
        blur_score = self._assess_blur_via_vlm()
        if blur_score < quality_threshold:
            return None  # Reject blurry image

        return {
            'image': image,
            'texts': verified_texts,
            'quality_score': blur_score,
            'num_text_instances': len(verified_texts)
        }

    def _create_crops(self, image, crop_size=512):
        """Create overlapping 512×512 crops to detect small text."""
        crops = []
        w, h = image.size
        stride = crop_size // 2

        for y in range(0, h - crop_size, stride):
            for x in range(0, w - crop_size, stride):
                crop = image.crop((x, y, x + crop_size, y + crop_size))
                crops.append(crop)

        return crops

    def _deduplicate_detections(self, detections):
        """Remove duplicate detections using IoU."""
        unique = []

        for det in detections:
            is_duplicate = False
            for existing in unique:
                iou = self._compute_iou(det['bbox'], existing['bbox'])
                if iou > 0.5:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(det)

        return unique

    def _compute_match_confidence(self, detected, recognized):
        """Compare detected and VLM-recognized text."""
        if detected == recognized:
            return 1.0

        # Simple edit distance based similarity
        max_len = max(len(detected), len(recognized))
        edit_dist = self._levenshtein(detected, recognized)

        return 1.0 - (edit_dist / max_len)

    def _assess_blur_via_vlm(self):
        """Use VLM to assess image clarity."""
        # Placeholder: VLM rates image quality 0-1
        return 0.85

    def _levenshtein(self, s1, s2):
        """Compute Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0
```

### Step 2: TeReDiff Architecture

```python
import torch
import torch.nn as nn

class TeReDiff(nn.Module):
    """
    Multi-task diffusion framework for text-aware restoration.
    Integrates visual restoration (U-Net) with text preservation (text spotting).
    """

    def __init__(self, pretrained_unet_path=None):
        super().__init__()

        # Module 1: Degradation removal (lightweight)
        self.degradation_remover = SwinIRModule()

        # Module 2: Diffusion restoration (U-Net + ControlNet)
        self.diffusion_unet = UNetModel()
        self.control_net = ControlNet()

        # Module 3: Text-spotting module (uses diffusion features)
        self.text_spotting = TextSpottingModule()

    def forward_stage1(self, degraded_image, text_prompts, timesteps, noise):
        """
        Stage 1: Train diffusion components (U-Net, ControlNet)
        with text prompts as conditioning.
        """

        # Degrade and restore
        cleaned = self.degradation_remover(degraded_image)

        # Diffusion denoising conditioned on text prompts
        features = self.diffusion_unet(
            cleaned,
            timesteps=timesteps,
            context=text_prompts  # Text conditioning
        )

        # ControlNet adds spatial guidance
        control_features = self.control_net(cleaned)
        restored = features + control_features

        return restored

    def forward_stage2(self, degraded_image, text_gt, timesteps):
        """
        Stage 2: Train text-spotting module using diffusion features
        as input representations (instead of ResNet backbone).
        """

        # Get frozen diffusion features from stage 1
        with torch.no_grad():
            diffusion_features = self.diffusion_unet.extract_features(
                degraded_image,
                timesteps=timesteps
            )

        # Text spotting uses diffusion features as backbone
        detected_texts = self.text_spotting(diffusion_features)

        # Compute text spotting loss
        spotting_loss = self._compute_spotting_loss(detected_texts, text_gt)

        return spotting_loss

    def forward_stage3(self, degraded_image, text_gt, text_prompts, timesteps):
        """
        Stage 3: Joint optimization of diffusion and text-spotting.
        Combined loss function: L = L_diffusion + α*L_spotting
        """

        # Forward through both modules
        cleaned = self.degradation_remover(degraded_image)
        restored = self.diffusion_unet(
            cleaned,
            timesteps=timesteps,
            context=text_prompts
        )

        diffusion_features = self.diffusion_unet.extract_features(
            degraded_image,
            timesteps=timesteps
        )
        detected_texts = self.text_spotting(diffusion_features)

        # Combined loss
        diffusion_loss = self._compute_diffusion_loss(restored, degraded_image)
        spotting_loss = self._compute_spotting_loss(detected_texts, text_gt)

        total_loss = diffusion_loss + 0.5 * spotting_loss

        return total_loss

    def infer(self, degraded_image, num_inference_steps=50):
        """
        Inference pipeline: detect text, generate prompts, restore with guidance.
        """

        # Step 1: Detect text in degraded image
        with torch.no_grad():
            detected_texts = self.text_spotting.detect(degraded_image)

        # Step 2: Generate dynamic text prompts
        text_list = [t['text'] for t in detected_texts]
        if text_list:
            prompt = f"A realistic scene where the texts {text_list} appear clearly on signs, boards, buildings..."
        else:
            prompt = "A high-quality restored image"

        # Step 3: Diffusion restoration with text-aware guidance
        restored = self._diffusion_infer(
            degraded_image,
            prompt,
            num_steps=num_inference_steps
        )

        return restored

    def _diffusion_infer(self, image, prompt, num_steps):
        """Simplified diffusion inference loop."""
        for t in reversed(range(num_steps)):
            timestep = torch.tensor([t])
            noise_pred = self.diffusion_unet(image, timestep, context=prompt)
            image = image - noise_pred
        return image

    def _compute_diffusion_loss(self, restored, original):
        return torch.nn.functional.mse_loss(restored, original)

    def _compute_spotting_loss(self, detected, ground_truth):
        """Detection loss: match detected texts to ground truth."""
        return torch.tensor(0.0)  # Placeholder
```

### Step 3: Text Prompt Generation

```python
class TextPromptGenerator:
    """
    Dynamically generate natural language prompts based on detected texts
    to guide diffusion restoration.
    """

    def generate_prompt(self, detected_texts, scene_context=None):
        """
        Generate natural text-aware restoration prompt.
        Example: "A realistic scene where the texts [sign], [board]... appear clearly"
        """

        if not detected_texts:
            return "Generate a high-quality restored image"

        # Extract text strings and locations
        text_strings = [t['text'] for t in detected_texts]
        locations = [t.get('location', 'in the image') for t in detected_texts]

        # Build prompt with text instances
        texts_str = ', '.join(text_strings)

        prompt = (
            f"A realistic scene where the texts {texts_str} appear clearly "
            f"on signs, boards, buildings, and other surfaces in the image. "
            f"Restore the image to high quality while preserving text legibility."
        )

        return prompt

    def generate_negative_prompt(self):
        """Generate negative prompt to avoid hallucinated text."""
        return (
            "Blurry text, distorted characters, illegible fonts, "
            "text artifacts, hallucinated text, corrupted writing"
        )
```

## Practical Guidance

**Dataset Creation**:
- Start with SA-1B (1.1B images) or similar large vision datasets
- Apply automated text detection (CRAFT, FOTS, or similar)
- Use overlapping crops (512×512) to catch small text instances
- Dual VLM verification (Qwen+OVIS) ensures high quality; target 0.95+ agreement

**Training Strategy**:
- Stage 1: Warm up diffusion with text prompts (100K steps)
- Stage 2: Introduce text spotting module with frozen diffusion (50K steps)
- Stage 3: Joint fine-tuning with combined losses (20K steps)
- Recommend α=0.5 for loss balance

**Inference Optimization**:
- Text detection is bottleneck; run once on degraded image to identify texts
- Generate prompts once, reuse across diffusion steps
- Batch multiple images to amortize text detection cost

**When to Use**:
- Document scanning (preserve printed text)
- Historical image restoration (maintain labels/captions)
- Signage/street view restoration (preserve visible text)
- OCR post-processing pipeline (first restore, then extract)

## Reference

- Text hallucination: Diffusion models generate plausible but incorrect visual content; grounding with detected text prevents this
- Multi-task learning: Joint training on restoration and detection improves both tasks
- Conditional diffusion: Text prompts guide generation trajectory toward text-aware outputs
