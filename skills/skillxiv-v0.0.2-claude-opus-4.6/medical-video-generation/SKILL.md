---
name: medical-video-generation
title: "MedGen: Unlocking Medical Video Generation by Scaling Granularly-annotated Medical Videos"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.05675"
keywords: [Medical Video, Diffusion Models, Video Synthesis, Medical AI, Domain Adaptation]
description: "Generate accurate, high-quality medical videos for clinical education and documentation by leveraging large-scale annotated medical datasets with domain-specific fine-tuning on video diffusion models."
---

# Medical Video Generation: Synthesizing Clinically Accurate Surgical and Imaging Videos

Medical video generation is critical for surgical training, patient education, and clinical documentation. However, general-purpose video generation models fail on medical content because they lack domain-specific knowledge of anatomical structures, procedural sequences, and clinical accuracy requirements. Standard video synthesis models trained on general web data cannot reliably generate medically accurate content—they may produce anatomically impossible procedures or omit critical steps.

MedGen addresses this through two key innovations: a large-scale, carefully curated medical video dataset (MedVideoCap-55K) with detailed captions, and domain-adapted fine-tuning of video diffusion models using LoRA. This approach enables generation of clinically accurate medical videos while maintaining computational efficiency, outperforming general models on medical content while matching commercial systems in quality.

## Core Concept

Medical video generation requires training diffusion models on domain-specific data that captures the diversity of clinical contexts. Rather than collecting raw videos, the approach focuses on curation and annotation quality: filtering out artifacts, subtitles, and encoding issues; generating detailed multimodal captions describing procedures and imaging findings; and evaluating outputs through medical domain experts.

The key insight is that medical accuracy emerges from combining three elements: (1) sufficient quantity of diverse medical clips spanning clinical practice, imaging, and education, (2) rigorous quality filtering to remove technical artifacts that would confuse models, and (3) detailed text descriptions that ground video generation in clinical terminology and procedural knowledge.

## Architecture Overview

- **Dataset Curation Pipeline**: Source collection from diverse medical domains → optical character recognition (OCR) filtering to remove subtitles → aesthetic quality assessment → technical encoding validation → duplicate removal
- **Caption Generation**: Multimodal language models process videos to generate detailed clinical descriptions capturing procedures, anatomical regions, and clinical context
- **Video Diffusion Model Backbone**: HunyuanVideo or similar video generation foundation model serving as base architecture
- **Parameter-Efficient Fine-tuning**: LoRA adapters applied to attention layers to efficiently inject medical knowledge without retraining the full model
- **Evaluation Framework**: Automated metrics (Med-VBench) plus human evaluation by radiologists and clinicians assessing medical accuracy, text-image alignment, and visual quality

## Implementation

The following demonstrates how to build a medical video generation system using a base diffusion model and domain-specific fine-tuning.

**Step 1: Dataset Preparation and Filtering**

This code applies quality filtering stages to remove unsuitable medical videos before training.

```python
import cv2
import numpy as np
from pytesseract import pytesseract
from PIL import Image

class MedicalVideoFilter:
    def __init__(self, ocr_threshold=0.3, quality_min=0.6):
        self.ocr_threshold = ocr_threshold
        self.quality_min = quality_min

    def detect_subtitles(self, frame):
        """Use OCR to detect subtitle presence in video frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        # Check if substantial text detected (likely subtitles)
        return len(text.split()) > self.ocr_threshold * frame.shape[0]

    def assess_technical_quality(self, video_path, sample_frames=10):
        """Evaluate technical encoding quality using sharpness and noise metrics."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, frame_count - 1, sample_frames, dtype=int)

        sharpness_scores = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Laplacian variance measures sharpness
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_scores.append(sharpness)

        cap.release()
        mean_sharpness = np.mean(sharpness_scores) if sharpness_scores else 0
        return mean_sharpness / 1000  # Normalize to 0-1 range

    def filter_batch(self, video_paths):
        """Filter videos based on subtitle presence and technical quality."""
        filtered = []
        for path in video_paths:
            cap = cv2.VideoCapture(path)
            has_subtitle = False

            # Sample frames for subtitle detection
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(0, min(frame_count, 30), 10):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret and self.detect_subtitles(frame):
                    has_subtitle = True
                    break

            cap.release()

            if not has_subtitle:
                quality = self.assess_technical_quality(path)
                if quality > self.quality_min:
                    filtered.append(path)

        return filtered
```

**Step 2: Caption Generation Using Multimodal LLMs**

This generates detailed medical descriptions for filtered videos using multimodal language models.

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

class MedicalCaptionGenerator:
    def __init__(self, model_name="llava-1.5-7b-hf"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.device = self.model.device

    def generate_caption(self, video_path, num_frames=8):
        """Extract frames and generate medical description."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        # Aggregate frames into a grid for captioning
        from PIL import Image as PILImage
        grid = PILImage.new('RGB', (640, 480 * num_frames // 2))
        for i, frame in enumerate(frames):
            frame_resized = frame.resize((320, 240))
            grid.paste(frame_resized, ((i % 2) * 320, (i // 2) * 240))

        prompt = "Describe this medical video in clinical detail. Include: 1) anatomical structures visible, 2) medical procedures or findings, 3) clinical context."
        inputs = self.processor(prompt, grid, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7)

        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
```

**Step 3: LoRA Fine-tuning for Medical Domain Adaptation**

This applies parameter-efficient fine-tuning to inject medical knowledge into a base video diffusion model.

```python
from diffusers import DiffusionPipeline
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset, DataLoader

class MedicalVideoDataset(Dataset):
    def __init__(self, video_paths, captions, num_frames=16):
        self.video_paths = video_paths
        self.captions = captions
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        import cv2
        cap = cv2.VideoCapture(self.video_paths[idx])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)
        cap.release()

        frames = torch.stack(frames) if frames else torch.zeros(self.num_frames, 3, 512, 512)
        caption = self.captions[idx]
        return {"video": frames, "caption": caption}

class MedicalVideoFinetuner:
    def __init__(self, base_model="hunyuan-video", learning_rate=1e-4):
        self.pipeline = DiffusionPipeline.from_pretrained(base_model)
        self.learning_rate = learning_rate

        # Configure LoRA for efficient adaptation
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["to_q", "to_v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.pipeline.unet, lora_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def finetune(self, train_loader, num_epochs=3):
        """Fine-tune diffusion model on medical video data."""
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                videos = batch["video"].to(self.model.device)
                captions = batch["caption"]

                # Encode text descriptions
                text_embeddings = self.pipeline.encode_prompt(captions)[0]

                # Add noise to videos and train denoiser
                timesteps = torch.randint(0, 1000, (videos.shape[0],))
                noise = torch.randn_like(videos)
                noisy_videos = self.pipeline.scheduler.add_noise(videos, noise, timesteps)

                # Predict noise and compute loss
                pred_noise = self.model(noisy_videos, timesteps, text_embeddings).sample
                loss = torch.nn.functional.mse_loss(pred_noise, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
```

## Practical Guidance

**Hyperparameters and Configuration**

| Parameter | Recommended Value | Range | Notes |
|-----------|------------------|-------|-------|
| LoRA Rank (r) | 16-32 | 8-64 | Higher rank captures more medical specificity but increases compute |
| LoRA Alpha | 2x rank | rank-4x rank | Controls LoRA strength; 2x is conservative baseline |
| Learning Rate | 1e-4 | 1e-5 to 1e-3 | Lower rates prevent catastrophic forgetting of base knowledge |
| Num Captions per Video | 1-3 | 1-5 | Multiple angles help but diminishing returns after 3 |
| Video Frames per Clip | 16-32 | 8-64 | Higher resolution captures fine procedural details |
| Quality Filter Threshold | 0.6-0.8 | 0.4-0.9 | More aggressive filtering (0.8+) improves quality but reduces data |

**When to Use**

- Medical education platforms needing synthetic surgical procedure videos
- Clinical documentation where procedural video examples are needed for training
- Research applications requiring diverse, controlled medical scenarios for testing
- Patient education materials explaining complex surgical procedures
- Rare procedure simulation when actual footage is unavailable or ethically sensitive

**When NOT to Use**

- Do not use for diagnostic decision-making without expert human validation
- Avoid using generated videos as ground truth for clinical research without human review
- Not suitable for applications requiring pixel-perfect anatomical accuracy in rare pathology
- Do not deploy as a replacement for real clinical footage in high-stakes training contexts without validation

**Common Pitfalls**

- **Insufficient medical curation**: Generic video filtering removes important clinical context. Use domain expert review for quality assessment rather than purely automated metrics.
- **Overfitting to specific procedures**: Training on limited procedural domains (e.g., only laparoscopy) without diversity creates poor generalization. Ensure dataset spans multiple specialties.
- **Ignoring caption quality**: Low-quality captions from automated generation propagate errors to generated videos. Validate captions with clinicians before large-scale training.
- **Ignoring evaluation rigor**: Automated metrics miss clinically important errors. Always include human evaluation by medical professionals, not just benchmark scores.
- **Data contamination**: Ensure source videos don't contain sensitive patient information before using in training. Implement privacy review in the filtering pipeline.

## Reference

MedGen: Unlocking Medical Video Generation by Scaling Granularly-annotated Medical Videos. https://arxiv.org/abs/2507.05675
