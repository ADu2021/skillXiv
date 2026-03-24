---
name: modality-adaptive-reasoning-visualizations
title: "MARVIS: Modality Adaptive Reasoning over VISualizations"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.01544"
keywords: [Vision-Language Models, Multimodal Reasoning, Latent Embedding Visualization, Audio Processing, Tabular Data]
description: "Enable small vision-language models to reason over diverse data types by converting latent embeddings into visual representations, achieving specialized performance without domain-specific training."
---

# MARVIS: Training-Free Multimodal Reasoning via Embedding Visualization

The challenge of building versatile AI systems that work across multiple modalities—vision, audio, tabular data, and biology—typically requires either specialized models for each domain or large foundation models with limited efficiency. MARVIS demonstrates that a 3B parameter vision-language model can match or exceed much larger systems by transforming latent embeddings from any modality into visual representations. This training-free approach leverages the spatial reasoning capabilities that VLMs naturally excel at.

The key insight is that embeddings from any modality (audio spectrograms, biological sequences, tabular encodings) can be visualized in ways that preserve semantic structure. A VLM can then interpret these visualizations using the same reasoning mechanisms it applies to images, effectively adapting to new domains without retraining.

## Core Concept

MARVIS operates on a simple but powerful principle: latent embeddings contain rich semantic information regardless of their original modality. Rather than training modality-specific decoders, the approach renders embeddings as images using techniques like heatmaps, spectrograms, or graph layouts. The VLM's spatial reasoning and fine-grained visual understanding then become universal interpreters for any domain.

This removes the traditional requirement for domain-specific pretraining or fine-tuning. The model reasons about structure, relationships, and patterns in the visual representation without knowing it originated from audio or tabular data. The method naturally preserves privacy since the original data is never exposed—only its learned embedding representation is visualized.

## Architecture Overview

The system comprises three main components:

- **Embedding Extractor**: Modality-agnostic encoder that produces dense vector representations from any input type (audio, images, tables, sequences)
- **Visualization Mapper**: Converts latent embeddings into image-space representations suitable for VLM interpretation, handling dimensionality reduction and spatial encoding
- **Vision-Language Reasoner**: Standard VLM (3B parameters in the paper) that interprets visualized embeddings using its native visual reasoning capabilities

The separation of concerns allows swapping embedding extractors for different modalities while reusing the same VLM, creating a truly modality-adaptive system.

## Implementation

The embedding visualization process maps high-dimensional latent vectors to 2D or 3D spatial representations that preserve semantic structure.

Create a mapping function that converts embeddings to visual space:

```python
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

def embed_to_heatmap(embedding, shape=(224, 224)):
    """
    Convert a latent embedding to a heatmap image.

    The embedding is reshaped into a 2D matrix, normalized to [0, 255],
    and rendered as a grayscale heatmap image for VLM interpretation.
    """
    # Reshape embedding to fit target image dimensions
    embedding_flat = embedding.flatten()

    # Pad or truncate to match target size
    target_pixels = shape[0] * shape[1]
    if len(embedding_flat) < target_pixels:
        embedding_flat = np.pad(embedding_flat, (0, target_pixels - len(embedding_flat)))
    else:
        embedding_flat = embedding_flat[:target_pixels]

    # Reshape to 2D and normalize to image range
    heatmap = embedding_flat.reshape(shape)
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(heatmap_uint8, mode='L')
    return image

def embed_to_scatter_plot(embedding, shape=(224, 224)):
    """
    Visualize embedding structure as a 2D scatter plot.

    Uses PCA to reduce embeddings to 2D, then renders as image with
    points positioned according to principal components.
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embedding.reshape(1, -1))

    # Create blank canvas
    img = Image.new('L', shape, color=255)
    pixels = img.load()

    # Normalize coordinates to image space
    x = int((coords_2d[0, 0] / (np.abs(coords_2d[0, 0]) + 1e-8)) * shape[0] // 2 + shape[0] // 2)
    y = int((coords_2d[0, 1] / (np.abs(coords_2d[0, 1]) + 1e-8)) * shape[1] // 2 + shape[1] // 2)

    # Draw point with Gaussian spread
    radius = 10
    for dx in range(-radius, radius):
        for dy in range(-radius, radius):
            px, py = x + dx, y + dy
            if 0 <= px < shape[0] and 0 <= py < shape[1]:
                dist = np.sqrt(dx**2 + dy**2)
                intensity = int(max(0, (radius - dist) / radius * 200))
                pixels[px, py] = max(0, pixels[px, py] - intensity)

    return img
```

Now integrate visualization with VLM reasoning:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

class ModalityAdaptiveReasoner:
    """
    Unified reasoner for any modality via embedding visualization.

    Takes modality-agnostic embeddings, visualizes them, and uses a VLM
    to reason about the visual representation without domain-specific knowledge.
    """

    def __init__(self, model_name="Qwen/Qwen-VL-Chat", viz_method="heatmap"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name)
        self.viz_method = viz_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def visualize_embedding(self, embedding, query=None):
        """
        Visualize and reason over an embedding from any modality.

        The embedding is converted to an image representation, then passed
        to the VLM along with a task query for domain-agnostic reasoning.
        """
        # Convert embedding to visual representation
        if self.viz_method == "heatmap":
            viz_image = embed_to_heatmap(embedding)
        else:
            viz_image = embed_to_scatter_plot(embedding)

        # Build VLM input with image and task query
        if query is None:
            query = "Describe the structure and patterns in this visualization."

        # Process with VLM
        inputs = self.processor(
            text=query,
            images=[viz_image],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate reasoning output
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response

    def batch_reason(self, embeddings, queries=None):
        """
        Process multiple embeddings with optional domain-specific queries.

        Enables efficient batch reasoning over embeddings from different
        sources (audio batch, table features, etc.) with unified VLM.
        """
        results = []
        for i, emb in enumerate(embeddings):
            query = queries[i] if queries else None
            result = self.visualize_embedding(emb, query)
            results.append(result)
        return results
```

## Practical Guidance

**Hyperparameter Table:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Visualization shape | (224, 224) | (64, 64) to (512, 512) | Match VLM's input resolution; larger = more detail |
| Viz method | heatmap | heatmap, scatter, spectrogram | Choose based on embedding structure |
| Normalization | min-max | min-max, z-score, log | min-max preserves full range; z-score handles outliers |
| Query specificity | General | General to task-specific | More specific queries improve reasoning quality |
| Batch size | 1 | 1-32 | Limit by GPU memory; larger batches faster for many embeddings |

**When to Use:**
- You need to reason over embeddings from audio, biological sequences, tables, or other non-image modalities
- You want to avoid domain-specific model training or fine-tuning
- You have a working VLM and want to extend it to new modalities
- Privacy is important—original data need not be exposed, only embeddings
- You want to leverage VLM spatial reasoning capabilities across domains

**When NOT to Use:**
- You need extremely fast inference—visualization adds computational overhead
- Your modality produces embeddings that don't preserve meaningful spatial structure
- You require fine-tuned performance on specific domains (specialized models outperform)
- Your target VLM has poor general reasoning capabilities
- You have very high-dimensional embeddings (>10K dims) where visualization loss becomes significant

**Common Pitfalls:**
- **Visualization loss**: Dimensionality reduction from high-dimensional embeddings to 2D/3D can lose semantic information. Use PCA or t-SNE variants that preserve local structure.
- **Poor query formulation**: VLMs work better with explicit, clear prompts. Vague queries yield poor reasoning. Prompt engineering is critical.
- **Mismatched VLM capability**: Not all VLMs handle abstract visualizations well. Test on your specific model before deployment.
- **Batch normalization issues**: Normalizing embeddings across batches can hide individual sample structure. Normalize per-sample when structure matters.
- **Modality-specific information loss**: Some modalities (e.g., temporal audio) may lose critical sequential information when flattened to 2D images. Consider ordered layouts.

## Reference

Feuer, B., Purucker, L., Elachqar, O., & Hegde, C. (2025). MARVIS: Modality Adaptive Reasoning over VISualizations. arXiv preprint arXiv:2507.01544. https://arxiv.org/abs/2507.01544
