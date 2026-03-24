---
name: inverse-llava-text-to-vision
title: "Inverse-LLaVA: Eliminating Alignment Pre-training via Text-to-Vision Mapping"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.12466
keywords: [multimodal-learning, vision-language, text-to-vision-mapping, alignment-free]
description: "Map text embeddings into visual representation space for multimodal fusion, eliminating expensive image-text alignment pre-training while improving reasoning-heavy tasks by up to 27.2%."
---

# Inverse-LLaVA: Eliminating Alignment Pre-training via Text-to-Vision Mapping

## Core Concept

Standard vision-language models (like LLaVA) project visual features into text token space, requiring massive image-text datasets to learn this projection. This alignment pre-training is expensive and data-intensive.

Inverse-LLaVA inverts this approach: map text embeddings into visual representation space instead. Fusion happens within transformer layers using selective attention. This eliminates the need for alignment pre-training entirely, while strengthening reasoning capabilities.

The counter-intuitive insight: direct vision-language reasoning doesn't require pre-trained vision-text alignment; the language model can learn to attend to visual features directly.

## Architecture Overview

- **Text-to-Vision Projection**: Linear or nonlinear mapping from text embedding space to vision space
- **Selective Attention Fusion**: Use intermediate transformer layers for cross-modal attention
- **No Alignment Pre-training**: Eliminates expensive image-text pre-training phase
- **Reasoning-Optimized**: Fusion strategy favors complex reasoning over perceptual memorization
- **Layer-Wise Integration**: Fuse modalities at multiple depths for flexible information flow

## Implementation Steps

### 1. Define Vision and Text Embedding Spaces

Load pre-trained vision and language models, establish embedding spaces.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel, AutoTokenizer

class EmbeddingSpaces:
    """
    Manage vision and text embedding spaces
    """
    def __init__(self, vision_model_name='openai/clip-vit-base-patch32',
                 text_model_name='bert-base-uncased'):
        # Vision encoder (frozen)
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_dim = self.vision_model.config.hidden_size

        # Text encoder (frozen or fine-tunable)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_dim = self.text_model.config.hidden_size

        # Freeze encoders
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

    def get_image_embeddings(self, images):
        """
        Extract vision embeddings from images
        images: tensor [batch, 3, H, W]
        """
        with torch.no_grad():
            outputs = self.vision_model(images)
            vision_embeddings = outputs.last_hidden_state  # [batch, num_patches, vision_dim]
        return vision_embeddings

    def get_text_embeddings(self, text):
        """
        Extract text embeddings
        text: list of strings or token IDs
        """
        if isinstance(text, list) and isinstance(text[0], str):
            text_tokens = self.text_tokenizer(text, return_tensors='pt', padding=True)
        else:
            text_tokens = text

        with torch.no_grad():
            outputs = self.text_model(**text_tokens)
            text_embeddings = outputs.last_hidden_state  # [batch, seq_len, text_dim]
        return text_embeddings
```

### 2. Implement Text-to-Vision Projection

Create a learnable projection from text embedding space to vision space.

```python
class TextToVisionProjector(nn.Module):
    """
    Projects text embeddings into vision embedding space
    """
    def __init__(self, text_dim, vision_dim, hidden_size=512):
        super().__init__()

        # Option 1: Linear projection (simplest)
        self.linear_projection = nn.Linear(text_dim, vision_dim)

        # Option 2: MLP projection (more flexible)
        self.mlp_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, vision_dim)
        )

        # Use MLP by default
        self.use_mlp = True

    def forward(self, text_embeddings):
        """
        Project text embeddings to vision space
        text_embeddings: [batch, seq_len, text_dim]
        """
        if self.use_mlp:
            # Apply MLP to each token
            batch_size, seq_len, text_dim = text_embeddings.shape
            flat_embeddings = text_embeddings.view(-1, text_dim)
            projected = self.mlp_projection(flat_embeddings)
            projected = projected.view(batch_size, seq_len, -1)
        else:
            projected = self.linear_projection(text_embeddings)

        return projected
```

### 3. Implement Selective Attention Fusion

Create cross-modal attention layers that fuse text-projected and vision features.

```python
class SelectiveAttentionFusion(nn.Module):
    """
    Fuse text-projected embeddings with vision features using selective attention
    """
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Cross-attention components
        self.query_text = nn.Linear(embedding_dim, embedding_dim)
        self.key_vision = nn.Linear(embedding_dim, embedding_dim)
        self.value_vision = nn.Linear(embedding_dim, embedding_dim)

        # Gating mechanism: control how much to attend to each modality
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()  # Gate output: 0-1
        )

        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, text_projected, vision_features):
        """
        Fuse modalities using selective attention

        Args:
            text_projected: [batch, text_seq_len, embedding_dim]
            vision_features: [batch, num_patches, embedding_dim]

        Returns:
            fused: [batch, text_seq_len, embedding_dim] - text tokens enriched with vision
        """
        batch_size, text_len, emb_dim = text_projected.shape

        # Compute queries from text, keys/values from vision
        Q = self.query_text(text_projected)  # [batch, text_len, emb_dim]
        K = self.key_vision(vision_features)  # [batch, num_patches, emb_dim]
        V = self.value_vision(vision_features)  # [batch, num_patches, emb_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, text_len, emb_dim)

        # Gating: learn how much to trust attention output vs original text
        gates = self.gate(text_projected)  # [batch, text_len, 1]
        fused = gates * attn_output + (1 - gates) * text_projected

        # Output projection
        fused = self.output_proj(fused)

        return fused
```

### 4. Build Inverse-LLaVA Model

Assemble the full multimodal model without alignment pre-training.

```python
class InverseLLaVA(nn.Module):
    """
    Vision-Language model using inverse (text-to-vision) mapping
    """
    def __init__(self, vision_model_name='openai/clip-vit-base-patch32',
                 language_model_name='meta-llama/Llama-2-7b',
                 num_fusion_layers=2):
        super().__init__()

        # Embedding spaces
        self.embeddings = EmbeddingSpaces(vision_model_name, language_model_name)
        self.vision_dim = self.embeddings.vision_dim
        self.text_dim = self.embeddings.text_dim

        # Text-to-vision projection
        self.text_to_vision = TextToVisionProjector(
            text_dim=self.text_dim,
            vision_dim=self.vision_dim
        )

        # Selective attention fusion layers
        self.fusion_layers = nn.ModuleList([
            SelectiveAttentionFusion(
                embedding_dim=self.vision_dim,
                num_heads=8
            ) for _ in range(num_fusion_layers)
        ])

        # Language model for reasoning
        from transformers import AutoModelForCausalLM
        self.llm = AutoModelForCausalLM.from_pretrained(language_model_name)

    def forward(self, images, text_input_ids, attention_mask=None):
        """
        Forward pass: image + text -> reasoning output

        Args:
            images: [batch, 3, H, W]
            text_input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len]

        Returns:
            output: language model output (logits, hidden states, etc.)
        """
        # Extract embeddings
        vision_features = self.embeddings.get_image_embeddings(images)

        # Get text embeddings from LLM's embedding layer
        text_embeddings = self.llm.get_input_embeddings()(text_input_ids)

        # Project text to vision space
        text_projected = self.text_to_vision(text_embeddings)

        # Fuse through selective attention layers
        fused = text_projected
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(fused, vision_features)

        # Create modified input for LLM: use fused text embeddings
        # Replace token embeddings with fused embeddings
        lm_output = self.llm(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        return lm_output

    def generate(self, images, prompts, max_length=100):
        """
        Generate text conditioned on images and prompts
        """
        # Tokenize prompts
        text_tokens = self.embeddings.text_tokenizer(
            prompts, return_tensors='pt', padding=True
        )
        input_ids = text_tokens['input_ids']
        attention_mask = text_tokens['attention_mask']

        # Generate
        with torch.no_grad():
            output_ids = self.llm.generate(
                inputs_embeds=None,  # Will use our fused embeddings
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=3,
                no_repeat_ngram_size=2
            )

        # Decode
        output_text = self.embeddings.text_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        return output_text
```

### 5. Training without Alignment Pre-training

Train the model for downstream tasks directly.

```python
def train_inverse_llava(model, train_dataset, num_epochs=3, batch_size=16):
    """
    Train Inverse-LLaVA on downstream tasks (VQA, image captioning, etc.)
    NO alignment pre-training needed
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_dataset:
            images = batch['images']  # [batch, 3, H, W]
            texts = batch['texts']    # [batch, seq_len]
            labels = batch['labels']  # [batch, num_classes] for classification

            # Forward pass
            outputs = model(images, texts)

            # Task-specific loss (e.g., classification)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits[:, -1, :],  # Use last token for classification
                labels
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"Epoch {epoch}, Batch {num_batches}: Loss={loss:.4f}")

        print(f"Epoch {epoch} completed. Avg Loss: {epoch_loss / num_batches:.4f}")
```

### 6. Evaluation on Reasoning vs. Perception Tasks

Benchmark the model on different task types.

```python
def evaluate_inverse_llava(model, eval_dataset):
    """
    Evaluate on reasoning vs. perception tasks
    Inverse-LLaVA should excel at reasoning, struggle on perception
    """
    reasoning_accuracy = 0.0
    perception_accuracy = 0.0
    num_reasoning = 0
    num_perception = 0

    with torch.no_grad():
        for sample in eval_dataset:
            image = sample['image'].unsqueeze(0)
            text = sample['text']
            label = sample['label']
            task_type = sample['task_type']  # 'reasoning' or 'perception'

            # Get prediction
            outputs = model.generate(image, text, max_length=50)
            pred_label = process_output(outputs[0])

            # Evaluate based on task type
            if task_type == 'reasoning':
                if pred_label == label:
                    reasoning_accuracy += 1.0
                num_reasoning += 1
            else:  # perception
                if pred_label == label:
                    perception_accuracy += 1.0
                num_perception += 1

    reasoning_acc = reasoning_accuracy / num_reasoning if num_reasoning > 0 else 0.0
    perception_acc = perception_accuracy / num_perception if num_perception > 0 else 0.0

    print("Evaluation Results:")
    print(f"  Reasoning Accuracy: {reasoning_acc:.3f}")
    print(f"  Perception Accuracy: {perception_acc:.3f}")

    return reasoning_acc, perception_acc
```

## Practical Guidance

### Hyperparameters & Configuration

- **Text-to-Vision MLP Layers**: 1-2 hidden layers (2 recommended)
- **Fusion Layers**: 2-4 layers (diminishing returns after 4)
- **Attention Heads**: 8 (standard for transformer attention)
- **Learning Rate**: 1e-4 to 5e-5 (conservative for frozen base models)
- **Gate Mechanism**: Sigmoid gate for smooth interpolation

### When to Use Inverse-LLaVA

- Focus is on reasoning tasks (VQA, scene understanding)
- You want to avoid expensive alignment pre-training
- You have limited image-text pair data
- Perception tasks aren't critical to your application
- You want interpretable text-vision projection

### When NOT to Use Inverse-LLaVA

- Perception accuracy is critical (image recognition, OCR)
- You need strong performance across all task types
- Memorization of visual-text associations is important
- You already have alignment pre-training data available
- You need SOTA performance on standard benchmarks

### Common Pitfalls

1. **Weak Gate Mechanism**: If gating doesn't learn well, fusion becomes either all-text or all-vision. Use residual connections.
2. **Projection Bottleneck**: Text-to-vision projection may lose information. Use MLP instead of linear projection.
3. **Too Few Fusion Layers**: Single fusion layer misses complex interactions. Use at least 2.
4. **Perception Expectations**: Model will underperform on memorization tasks. Accept this trade-off for reasoning.
5. **No Task-Specific Tuning**: Different downstream tasks may need different fusion strategies. Validate on your tasks.

## Reference

Inverse-LLaVA (2508.12466): https://arxiv.org/abs/2508.12466

Map text embeddings into visual space for multimodal fusion via selective attention, eliminating alignment pre-training while achieving 27.2% improvements on reasoning tasks and revealing perception-reasoning trade-offs in multimodal learning.
