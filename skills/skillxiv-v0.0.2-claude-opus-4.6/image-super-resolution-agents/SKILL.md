---
name: image-super-resolution-agents
title: "4KAgent: Agentic Any Image to 4K Super-Resolution"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07105"
keywords: [Image Super-Resolution, Agentic Systems, Image Restoration, Quality Assessment, Multimodal AI]
description: "Upscale any degraded image to 4K using an agentic framework that analyzes image quality, selects appropriate restoration tools, and iteratively improves results through reasoning and reflection."
---

# Agentic Image Super-Resolution: Autonomous Restoration Through Reasoning and Quality-Driven Tool Selection

Image super-resolution—upscaling low-resolution images while restoring degradation (blur, noise, compression artifacts)—requires different approaches for different image types and degradation levels. A single model struggles with extreme cases: upscaling a 256×256 image to 4K while removing noise and handling both natural and AI-generated content requires specialized handling.

4KAgent addresses this through an agentic architecture where a perception agent analyzes image degradation, a restoration agent selects and sequences appropriate tools, and a quality-driven mixture-of-experts mechanism reflects on intermediate results to select the best restoration path. This approach handles any image type without domain-specific fine-tuning while achieving photorealistic 4K outputs.

## Core Concept

Traditional image restoration uses single specialized models (super-resolution network, denoiser, etc.) applied sequentially. 4KAgent treats restoration as an agentic problem: (1) analyze what's wrong with the image (perception), (2) plan restoration steps (reasoning), (3) execute tools dynamically (action), (4) evaluate quality (reflection), and (5) refine the plan based on results. This mirrors how expert image processors work—analyzing, trying approaches, assessing results, and iterating.

The key insight is that quality-aware mixture-of-experts (Q-MoE)—which dynamically selects tools based on image quality metrics—outperforms fixed pipelines. Rather than "always denoise then upscale," the system learns "if this image is very noisy, denoise first; if noise is moderate, interleave with super-resolution."

## Architecture Overview

- **Perception Agent**: Vision-language model analyzing degradation types (blur, noise, compression, hazing), severity levels, and restoration priorities
- **Image Quality Assessor**: Vision model and traditional metrics (BRISQUE, NIQE) computing image quality scores
- **Restoration Tool Library**: 9 specialized tools covering brightening, deblurring, denoising, dehazing, super-resolution, artifact removal
- **Quality-Driven Mixture-of-Experts (Q-MoE)**: Learns which tools to apply and in what order based on quality metrics and degradation analysis
- **Face Enhancement Module**: Specialized pipeline for detecting and enhancing facial regions while preserving identity
- **Restoration Executor**: Applies selected tools and sequences, handling format conversions and resolution constraints
- **Iterative Refinement Loop**: Re-evaluates quality after each step, adjusts tool selection based on results
- **Customizable Profiles**: 7 tunable parameters enabling different restoration styles (perception vs. fidelity focus)

## Implementation

The following implements an agentic image restoration system with quality-driven tool selection.

**Step 1: Image Quality Assessment**

This component analyzes image degradation and computes quality metrics.

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageQualityAssessor:
    """Assess image degradation and quality metrics."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def compute_brisque(self, image: torch.Tensor) -> float:
        """
        Blind/Reference-less Image Spatial Quality Evaluator (BRISQUE).
        Measures image quality without reference. Lower score = better quality.
        """
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image[0] if len(image.shape) == 3 else image

        # Compute local contrast (standard deviation in local patches)
        patches = []
        patch_size = 15

        for i in range(0, gray.shape[0] - patch_size, patch_size):
            for j in range(0, gray.shape[1] - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                patches.append(patch.std().item())

        # BRISQUE-like score: inverse of sharpness
        mean_contrast = np.mean(patches) if patches else 1.0
        brisque = 100.0 / (mean_contrast + 1.0)  # Normalized

        return brisque

    def detect_degradation_types(self, image: torch.Tensor) -> dict:
        """
        Identify types and severity of degradation.
        Returns dict with degradation types and severity scores (0-1).
        """
        h, w = image.shape[1], image.shape[2]
        degradation = {
            "blur": 0.0,
            "noise": 0.0,
            "compression": 0.0,
            "haze": 0.0,
            "brightness": 0.0,
        }

        # Blur detection: edge strength
        edges = torch.abs(torch.diff(image, dim=1)) + torch.abs(torch.diff(image, dim=2))
        blur_level = 1.0 - torch.mean(edges).item() * 10
        degradation["blur"] = max(0.0, min(1.0, blur_level))

        # Noise detection: local variance
        patch_vars = []
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                patch = image[:, i:i+8, j:j+8]
                patch_vars.append(patch.var().item())
        noise_level = np.mean(patch_vars) if patch_vars else 0.0
        degradation["noise"] = min(1.0, noise_level * 10)

        # Compression artifacts: blockiness
        h_diffs = torch.abs(torch.diff(image[:, ::8, :], dim=1)).mean().item()
        v_diffs = torch.abs(torch.diff(image[:, :, ::8], dim=2)).mean().item()
        compression_score = max(h_diffs, v_diffs)
        degradation["compression"] = min(1.0, compression_score * 100)

        # Brightness: mean pixel value
        mean_brightness = image.mean().item() / 255.0
        brightness_issue = 0.0 if 0.3 < mean_brightness < 0.7 else abs(mean_brightness - 0.5)
        degradation["brightness"] = min(1.0, brightness_issue * 2)

        return degradation

    def compute_overall_quality(self, image: torch.Tensor) -> float:
        """
        Compute overall image quality score (0-100, higher = better).
        """
        brisque = self.compute_brisque(image)
        degradation = self.detect_degradation_types(image)

        # Weighted combination
        quality = 100.0 - (
            brisque * 0.3 +
            (degradation["blur"] + degradation["noise"] + degradation["compression"]) * 100 / 3 * 0.5 +
            degradation["brightness"] * 50 * 0.2
        )

        return max(0.0, min(100.0, quality))
```

**Step 2: Perception Agent - Analyzing Degradation**

This agent analyzes what's wrong with the image.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class PerceptionAgent:
    """Analyzes image degradation using vision-language models."""

    def __init__(self, model_name: str = "llava-1.5-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def analyze_image_degradation(self, image: Image.Image) -> dict:
        """
        Analyze image and identify restoration priorities.
        Returns structured analysis for restoration planning.
        """
        prompt = """
        Analyze this image and describe:
        1. What types of degradation are present (blur, noise, compression, color issues, etc.)?
        2. How severe is each degradation (mild/moderate/severe)?
        3. What is the content type (photo, artwork, document, screenshot)?
        4. What restoration steps would help most?
        Provide concise bullet points.
        """

        # In practice, would use actual VLM; placeholder here
        analysis = {
            "degradation_types": ["blur", "noise"],
            "severity_scores": {"blur": 0.6, "noise": 0.7},
            "content_type": "natural_image",
            "restoration_priorities": ["denoise", "deblur", "super_resolve"],
            "confidence": 0.85
        }

        return analysis
```

**Step 3: Quality-Driven Mixture-of-Experts**

This selects restoration tools based on image quality metrics.

```python
class RestorationTool:
    """Wrapper for a restoration operation."""

    def __init__(self, name: str, process_fn, applicable_degradations: list):
        self.name = name
        self.process_fn = process_fn
        self.applicable_degradations = applicable_degradations

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """Apply restoration."""
        return self.process_fn(image)

class QualityDrivenMoE:
    """Mixture-of-Experts that selects restoration tools based on quality."""

    def __init__(self):
        self.tools = self._initialize_tools()
        self.quality_assessor = ImageQualityAssessor()

    def _initialize_tools(self) -> dict:
        """Initialize restoration tools."""
        tools = {
            "denoise": RestorationTool(
                "denoise",
                self._denoise,
                ["noise"]
            ),
            "deblur": RestorationTool(
                "deblur",
                self._deblur,
                ["blur"]
            ),
            "super_resolve": RestorationTool(
                "super_resolve",
                self._super_resolve,
                []  # Generally applicable
            ),
            "enhance_brightness": RestorationTool(
                "enhance_brightness",
                self._enhance_brightness,
                ["brightness"]
            ),
            "enhance_contrast": RestorationTool(
                "enhance_contrast",
                self._enhance_contrast,
                []
            ),
        }
        return tools

    def _denoise(self, image: torch.Tensor) -> torch.Tensor:
        """Apply denoising (simplified)."""
        # Gaussian blur as simple denoiser
        kernel_size = 3
        blurred = torch.nn.functional.avg_pool2d(
            image.unsqueeze(0), kernel_size, stride=1, padding=kernel_size//2
        ).squeeze(0)
        return blurred * 0.3 + image * 0.7

    def _deblur(self, image: torch.Tensor) -> torch.Tensor:
        """Apply deblurring (placeholder)."""
        # Sharpen filter
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ], dtype=torch.float32) / 9.0

        sharpened = torch.nn.functional.conv2d(
            image.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze(0)
        return torch.clamp(sharpened, 0, 1)

    def _super_resolve(self, image: torch.Tensor) -> torch.Tensor:
        """Apply super-resolution."""
        # Simple upsampling (in practice, use SR model)
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), scale_factor=2, mode='bilinear'
        ).squeeze(0)

    def _enhance_brightness(self, image: torch.Tensor) -> torch.Tensor:
        """Enhance brightness for dark images."""
        mean_brightness = image.mean().item()
        if mean_brightness < 0.3:
            return torch.clamp(image * 1.5, 0, 1)
        return image

    def _enhance_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """Enhance contrast."""
        mean = image.mean()
        std = image.std()
        return torch.clamp((image - mean) * 1.5 + mean, 0, 1)

    def select_tools(
        self,
        image: torch.Tensor,
        degradation_analysis: dict,
        max_steps: int = 5
    ) -> list:
        """
        Select restoration tools based on quality metrics and degradation.
        Returns ordered list of tools to apply.
        """
        quality = self.quality_assessor.compute_overall_quality(image)
        degradation = self.quality_assessor.detect_degradation_types(image)

        selected_tools = []

        # Quality-driven selection
        if degradation["noise"] > 0.5:
            selected_tools.append(self.tools["denoise"])

        if degradation["blur"] > 0.5:
            selected_tools.append(self.tools["deblur"])

        if degradation["brightness"] > 0.3:
            selected_tools.append(self.tools["enhance_brightness"])

        # Super-resolution typically last
        selected_tools.append(self.tools["super_resolve"])

        return selected_tools[:max_steps]

    def iterative_restoration(
        self,
        image: torch.Tensor,
        degradation_analysis: dict,
        max_iterations: int = 3
    ) -> torch.Tensor:
        """
        Iteratively apply tools and reflect on quality.
        Adjust tool selection based on quality improvement.
        """
        current_image = image
        initial_quality = self.quality_assessor.compute_overall_quality(image)

        for iteration in range(max_iterations):
            # Select tools based on current quality
            tools = self.select_tools(current_image, degradation_analysis)

            # Apply tools
            for tool in tools:
                current_image = tool.apply(current_image)

            # Assess improvement
            new_quality = self.quality_assessor.compute_overall_quality(current_image)
            improvement = new_quality - initial_quality

            print(f"Iteration {iteration + 1}: Quality = {new_quality:.1f}, Improvement = {improvement:.1f}")

            if improvement < 1.0:  # Diminishing returns
                break

            initial_quality = new_quality

        return current_image
```

**Step 4: Face Enhancement Pipeline**

This specializes in restoring facial regions.

```python
class FaceEnhancementModule:
    """Specialized restoration for facial regions."""

    def __init__(self):
        self.moe = QualityDrivenMoE()

    def detect_faces(self, image: torch.Tensor) -> list:
        """
        Detect facial regions. In practice, use face detection model.
        Returns list of (x1, y1, x2, y2) bounding boxes.
        """
        # Placeholder: return full image as single face region
        h, w = image.shape[1], image.shape[2]
        return [(0, 0, w, h)]

    def enhance_faces(self, image: torch.Tensor, face_regions: list) -> torch.Tensor:
        """
        Enhance faces while preserving identity.
        Apply stronger restoration to facial regions.
        """
        enhanced = image.clone()

        for x1, y1, x2, y2 in face_regions:
            face_region = image[:, y1:y2, x1:x2]

            # Apply face-specific enhancement (higher denoise + super-resolution)
            face_enhanced = self.moe.iterative_restoration(
                face_region,
                {"blur": 0.7, "noise": 0.8},  # Assume moderate degradation
                max_iterations=2
            )

            # Blend back to avoid artifacts
            alpha = 0.8
            enhanced[:, y1:y2, x1:x2] = face_enhanced * alpha + face_region * (1 - alpha)

        return enhanced
```

## Practical Guidance

**Hyperparameters and Configuration**

| Parameter | Recommended Value | Range | Notes |
|-----------|------------------|-------|-------|
| Max Restoration Iterations | 3-5 | 1-10 | More iterations = better quality but slower |
| Quality Improvement Threshold | 1.0 | 0.1-5.0 | Stop if improvement below this per iteration |
| Upscaling Factor Per Step | 2x | 2-4x | Multiple 2x upscalings smoother than single large jump |
| Noise Severity Threshold | 0.5 | 0.3-0.7 | Trigger denoising if noise score above threshold |
| Blur Severity Threshold | 0.5 | 0.3-0.7 | Trigger deblur if blur score above threshold |
| Fidelity vs. Perception | 0.5 | 0.0-1.0 | 0.5 = balanced, <0.5 = preserve details, >0.5 = enhance perceptually |

**When to Use**

- Upscaling low-resolution images (thumbnails, old photos) to high resolution
- Restoring degraded images from various sources (screenshots, compressed photos, surveillance)
- Batch processing of mixed image types without domain-specific models
- Applications where restoration quality matters more than processing speed
- Scenarios where manual intervention per image is infeasible
- Enhancement of both natural images and AI-generated content

**When NOT to Use**

- Real-time applications requiring deterministic latency (iterative process variable)
- Memory-constrained systems (4K output requires significant memory)
- Scenarios where image semantics must remain unchanged (identity in faces)
- Applications requiring pixel-perfect reconstruction (lossy process)
- Systems where inference speed is critical over quality

**Common Pitfalls**

- **Over-restoration**: More iterations don't always improve quality. Monitor diminishing returns and stop early.
- **Ignoring content type**: Natural images, documents, and artwork benefit from different restoration strategies. Customize tool selection per content.
- **Face enhancement artifacts**: Blending face-enhanced regions too strongly causes identity changes. Use conservative alpha blending (0.7-0.8).
- **Cascading quality assessment errors**: If quality assessor is poor, Q-MoE makes bad tool selections. Validate quality metrics on diverse test set.
- **Memory overflow on extreme upscaling**: 256×256 → 4K requires progressive upscaling, not single step. Implement tile-based processing for large outputs.

## Reference

4KAgent: Agentic Any Image to 4K Super-Resolution. https://arxiv.org/abs/2507.07105
