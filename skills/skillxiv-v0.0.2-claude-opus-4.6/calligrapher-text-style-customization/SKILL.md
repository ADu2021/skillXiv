---
name: calligrapher-text-style-customization
title: "Calligrapher: Freestyle Text Image Customization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.24123"
keywords: [TextGeneration, StyleTransfer, DiffusionModels, Typography, ImageCustomization]
description: "Automates artistic typography customization through self-distilled learning and localized style injection. Generates stylized text images by encoding reference style and injecting it into diffusion denoising. Use for digital design workflows, text-based visual content creation, or applications needing artistic typography control without manual annotation."
---

# Calligrapher: Automated Typography Customization Through Style-Aware Diffusion

Typography is an art form, but designing custom stylized text requires skilled typographers and extensive manual work. Most text generation systems either ignore style entirely or require parallel paired examples. Calligrapher solves this by leveraging self-distillation to create synthetic paired training data and implementing localized style injection—encoding reference style information and fusing it into the generation process at inference time. The result is a system that generates high-quality typographic designs matching arbitrary reference styles without human annotation.

The key innovation is recognizing that style is local and learnable from single examples. Rather than requiring hundreds of parallel examples, Calligrapher extracts style features from a single reference image and conditions the entire generation process, enabling one-shot style transfer for text.

## Core Concept

Calligrapher uses three complementary mechanisms to solve the typography customization problem:

1. **Self-Distillation**: Leverages pre-trained diffusion models and LLMs to synthesize paired training data with style annotations, avoiding expensive manual labeling.

2. **Style Encoder**: A learnable module (combining Qformer and linear projections) that extracts style features from reference images, producing conditioned representations for the generation pipeline.

3. **Localized Style Injection**: Integrates style information directly into the diffusion denoising process through modified cross-attention, ensuring style consistency throughout generation while allowing flexible text content.

The framework supports three customization modes: self-reference (edit text while preserving original style), cross-reference (apply style from one image to different text), and style transfer from non-text images.

## Architecture Overview

- **Style Encoder**: Vision transformer backbone (Qformer) extracting semantic style features, projected to cross-attention dimensions
- **Self-Distillation Module**: Generates synthetic paired training data using pre-trained diffusion models and LLMs
- **Modified Cross-Attention Layer**: Replaces standard text conditioning with style-aware attention that fuses style and content
- **Spatial Concatenation**: In-context generation embedding reference images directly in the latent space
- **Diffusion Backbone**: Standard U-Net denoiser adapted to accept style-conditioned cross-attention

## Implementation

The style encoder extracts robust style features from reference images:

```python
import torch
import torch.nn as nn
from einops import rearrange

class StyleEncoder(nn.Module):
    """
    Extracts style features from reference images using Qformer,
    producing cross-attention keys/values for style-aware generation.
    """
    def __init__(self, vision_dim=768, qformer_dim=256, output_dim=768):
        super().__init__()

        # Vision encoder backbone (pretrained)
        self.vision_encoder = VisionTransformer(output_dim=vision_dim)

        # Qformer: learnable query-based feature extraction
        self.qformer = Qformer(
            num_queries=32,  # 32 style tokens
            hidden_dim=qformer_dim,
            num_layers=4
        )

        # Project to cross-attention dimensions
        self.style_projection = nn.Sequential(
            nn.Linear(qformer_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, reference_image):
        """
        Extract style from reference image.
        Args:
            reference_image: (B, 3, H, W) image tensor

        Returns:
            style_keys: (B, 32, 768) style-conditioned keys
            style_values: (B, 32, 768) style-conditioned values
        """
        # Extract image features
        image_features = self.vision_encoder(reference_image)  # (B, seq_len, 768)

        # Qformer learns style-relevant features
        style_features = self.qformer(image_features)  # (B, 32, 256)

        # Project to cross-attention space
        style_output = self.style_projection(style_features)  # (B, 32, 768)

        # Create key-value pairs for attention
        style_keys = style_output
        style_values = style_output  # Same representation for keys and values

        return style_keys, style_values
```

Self-distillation generates training data without manual annotation:

```python
class SelfDistillationPipeline:
    """
    Creates paired training data (reference image, generation target)
    using pre-trained diffusion models and LLMs.
    """
    def __init__(self, diffusion_model, llm_model):
        self.diffusion = diffusion_model
        self.llm = llm_model

    def generate_training_pair(self, style_description: str, text_content: str):
        """
        Generate (reference_style_image, target_text_image) pair.

        Args:
            style_description: "serif font with gold embossing"
            text_content: "Hello World"

        Returns:
            reference_image: style example
            target_image: text in that style
        """
        # Step 1: Generate styled reference image
        prompt = f"A {style_description} text design"
        reference_image = self.diffusion.generate(prompt, num_inference_steps=50)

        # Step 2: Use LLM to create detailed generation instruction
        instruction = self.llm.generate_instruction(
            style=style_description,
            text=text_content
        )
        # Returns: "Write 'Hello World' in serif with gold embossing"

        # Step 3: Generate target image with text
        target_prompt = f"{instruction}, text reads '{text_content}'"
        target_image = self.diffusion.generate(target_prompt, num_inference_steps=50)

        return reference_image, target_image
```

Localized style injection modifies cross-attention to fuse style:

```python
class StyleAwareAttention(nn.Module):
    """
    Modified cross-attention layer that integrates style features
    alongside text conditioning. Ensures style consistency
    throughout the denoising process.
    """
    def __init__(self, hidden_dim=768, num_heads=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Standard cross-attention for text
        self.text_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # Style-aware fusion
        self.style_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, latent_features, text_tokens, style_keys, style_values):
        """
        Apply style-aware attention.

        Args:
            latent_features: (B, seq_len, 768) current denoising state
            text_tokens: (B, text_len, 768) text embeddings
            style_keys: (B, 32, 768) style features from encoder
            style_values: (B, 32, 768) style features from encoder

        Returns:
            attended: (B, seq_len, 768) style-fused latent
        """
        # Combine text and style into unified conditioning
        combined_keys = torch.cat([text_tokens, style_keys], dim=1)
        combined_values = torch.cat([text_tokens, style_values], dim=1)

        # Apply cross-attention with combined conditioning
        attended, _ = self.text_attention(
            latent_features,
            combined_keys,
            combined_values
        )

        # Fuse attended result with style signal
        style_context = torch.mean(style_values, dim=1, keepdim=True)
        fused = self.style_fusion(torch.cat([attended, style_context], dim=-1))

        return fused
```

The customization pipeline supports three modes:

```python
class CalligrapherPipeline:
    """
    Unified interface for text image customization.
    Supports self-reference, cross-reference, and style transfer.
    """
    def __init__(self, style_encoder, diffusion_model):
        self.style_encoder = style_encoder
        self.diffusion = diffusion_model

    def customize_self_reference(self, original_image, new_text):
        """Edit text while preserving original style."""
        # Extract style from original image
        style_keys, style_values = self.style_encoder(original_image)

        # Generate new text in extracted style
        output = self.diffusion.generate_with_style(
            text=new_text,
            style_keys=style_keys,
            style_values=style_values,
            num_steps=30
        )
        return output

    def customize_cross_reference(self, style_image, text_content):
        """Apply style from one image to different text."""
        # Extract style from reference image
        style_keys, style_values = self.style_encoder(style_image)

        # Generate text in the extracted style
        output = self.diffusion.generate_with_style(
            text=text_content,
            style_keys=style_keys,
            style_values=style_values,
            num_steps=30
        )
        return output

    def transfer_from_non_text(self, reference_image, text_content):
        """Apply style from any image (not necessarily text) to generate styled text."""
        # Extract style features (works with any image)
        style_keys, style_values = self.style_encoder(reference_image)

        # Generate text matching the image's visual style
        output = self.diffusion.generate_with_style(
            text=text_content,
            style_keys=style_keys,
            style_values=style_values,
            num_steps=30
        )
        return output
```

## Practical Guidance

| Aspect | Value | Notes |
|--------|-------|-------|
| Training Data Source | Self-distilled | No manual annotation required |
| Style Encoder Components | Qformer (32 queries) + projection | Lightweight, efficient |
| Reference Image Requirement | Single image per style | One-shot style transfer |
| Customization Modes | 3 (self-ref, cross-ref, transfer) | Flexible application scenarios |
| Generation Quality | High fidelity typography | Maintains style consistency |
| Typical Generation Steps | 30-50 denoising steps | Fewer steps = speed, more = quality |

**When to use:**
- Digital design workflows needing customizable typography
- Generating styled text for marketing, UI design, or branding
- One-shot style transfer without paired training examples
- Applications requiring both content and style control
- Batch processing text in multiple styles
- Creating variations on typographic designs

**When NOT to use:**
- Systems needing deterministic text rendering (diffusion is stochastic)
- Real-time applications with strict latency requirements (30+ denoising steps)
- Scenarios where precise character positioning is critical (diffusion may distort)
- Tasks where style must match a dataset's specific font without variation
- Applications requiring generated text to be machine-readable (OCR may fail)

**Common pitfalls:**
- Insufficient diversity in self-distilled training data causing limited style generalization
- Style encoder producing inconsistent representations when style is subtle
- Cross-attention fusion weight favoring either text or style, losing balance
- Generation steps too low (style not fully incorporated) or too high (computational waste)
- Using non-representative reference images (poor style extraction)
- Expecting exact style replication across drastically different text (style interpolation is approximate)

## Reference

"Calligrapher: Freestyle Text Image Customization", 2025. [arxiv.org/abs/2506.24123](https://arxiv.org/abs/2506.24123)
