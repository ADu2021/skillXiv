---
name: fuselip-multimodal-embeddings
title: "FuseLIP: Multimodal Embeddings via Early Fusion of Discrete Tokens"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.03096"
keywords: [multimodal-embeddings, early-fusion, vision-language, discrete-tokens, contrastive-learning]
description: "Build unified multimodal embeddings with a single transformer encoder processing image and text tokens together, improving performance on structure-aware tasks through early fusion."
---

# FuseLIP: Multimodal Embeddings via Early Fusion of Discrete Tokens

## Core Concept

FuseLIP proposes a paradigm shift in multimodal architecture: rather than separate image and text encoders (late fusion), use a single transformer processing both modalities as discrete tokens from the start. By converting images to tokens via a frozen tokenizer, both modalities interact at every layer, enabling richer cross-modal understanding. This "early fusion" approach particularly excels on tasks requiring visual structure understanding (text-guided image transformations, visual grounding). The framework combines sigmoid contrastive loss with masked multimodal modeling, leveraging discrete tokens to enable efficient prediction of masked positions across both modalities. Results show stronger performance than late fusion baselines, especially on structure-aware tasks, while maintaining simplicity and computational efficiency.

## Architecture Overview

- **Single Transformer Encoder**: Unified architecture processing both image and text tokens
- **Discrete Tokenization**: Images converted to tokens (TiTok family) enabling interleaved processing
- **Early Fusion Design**: Tokens concatenated before encoder, enabling cross-modal interaction per layer
- **Dual Training Objectives**: Sigmoid contrastive loss + masked multimodal modeling
- **Hard Negative Sampling**: Include semantically similar negatives for robust contrastive learning
- **Two Model Sizes**: FuseLIP-S (small) and FuseLIP-B (base) for different deployment scenarios

## Implementation

1. **Image and Text Tokenization**: Convert both modalities to discrete tokens

```python
def tokenize_multimodal_pair(image, text, image_tokenizer, text_tokenizer):
    """
    Convert image and text to unified token sequences.
    Both use non-overlapping vocabulary in shared space.
    """
    # Image tokenization: frozen TiTok tokenizer
    # Input: PIL Image or tensor [C, H, W]
    image_tokens = image_tokenizer.encode(image)  # Returns [num_image_tokens]

    # Text tokenization: standard tokenizer
    text_tokens = text_tokenizer.encode(text)     # Returns [num_text_tokens]

    # Concatenate for unified sequence
    # Image tokens come first, then text tokens
    unified_tokens = torch.cat([image_tokens, text_tokens], dim=0)

    # Create token type IDs to track modality (optional, for analysis)
    token_types = torch.cat([
        torch.zeros_like(image_tokens),  # 0 for image tokens
        torch.ones_like(text_tokens)     # 1 for text tokens
    ], dim=0)

    return unified_tokens, token_types

class FuseLIPTokenizer:
    def __init__(self, image_tokenizer_name='titok-s', vocab_size=8192):
        """Initialize shared tokenization setup."""
        self.image_tokenizer = load_tokenizer(image_tokenizer_name)
        # Text tokenizer uses disjoint vocabulary from image tokens
        self.text_vocab_size = vocab_size // 2
        self.image_vocab_size = vocab_size // 2
        self.vocab_offset_text = self.image_vocab_size

    def tokenize_pair(self, image, text):
        """Tokenize image and text with offset vocabularies."""
        image_tokens = self.image_tokenizer.encode(image)
        text_tokens = self.text_tokenizer.encode(text) + self.vocab_offset_text
        return torch.cat([image_tokens, text_tokens], dim=0)
```

2. **Single Encoder Architecture**: Unified transformer for both modalities

```python
def build_fuselip_encoder(vocab_size=8192, hidden_dim=384, num_layers=12,
                         num_heads=6, model_size='small'):
    """
    Build FuseLIP encoder architecture.
    Single transformer processing mixed image/text token sequences.
    """
    if model_size == 'small':
        config = {
            'hidden_size': 384,
            'num_hidden_layers': 12,
            'num_attention_heads': 6,
            'intermediate_size': 1024,
            'vocab_size': vocab_size,
            'max_position_embeddings': 2048
        }
    elif model_size == 'base':
        config = {
            'hidden_size': 512,
            'num_hidden_layers': 12,
            'num_attention_heads': 8,
            'intermediate_size': 2048,
            'vocab_size': vocab_size,
            'max_position_embeddings': 2048
        }

    # Standard transformer encoder
    encoder = TransformerEncoder(config)

    return encoder

class FuseLIPModel(torch.nn.Module):
    def __init__(self, encoder, hidden_dim=384):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim

        # Projection head for contrastive learning
        self.proj_head = torch.nn.Linear(hidden_dim, 256)

    def forward(self, input_ids, token_types=None):
        """
        Forward pass through unified encoder.
        Returns both token representations and pooled representation.
        """
        # Encode with transformer
        hidden_states = self.encoder(input_ids)  # [batch, seq_len, hidden_dim]

        # Pool to get single representation per sample
        # Strategy: mean pool across image tokens only (or all tokens)
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_dim]

        # Project for contrastive loss
        projected = self.proj_head(pooled)  # [batch, 256]

        return {
            'hidden_states': hidden_states,
            'pooled': pooled,
            'projection': projected
        }
```

3. **Sigmoid Contrastive Loss**: Adapted from SigLIP for unified encoder

```python
def sigmoid_contrastive_loss(image_projections, text_projections, temperature=0.01):
    """
    Sigmoid contrastive loss for multimodal embeddings.
    Adapted from SigLIP for early fusion architecture.

    Positive pairs: (image, corresponding text)
    Negatives: Other (image, text) pairs in batch
    """
    batch_size = image_projections.shape[0]

    # Normalize projections
    image_proj_norm = torch.nn.functional.normalize(image_projections, dim=1)
    text_proj_norm = torch.nn.functional.normalize(text_projections, dim=1)

    # Compute similarity matrix [batch_size, batch_size]
    # sim[i,j] = similarity between image_i and text_j
    similarity_matrix = torch.matmul(image_proj_norm, text_proj_norm.T) / temperature

    # Create labels: diagonal elements are positive pairs
    positive_labels = torch.eye(batch_size, device=similarity_matrix.device)

    # Sigmoid cross-entropy loss (stable alternative to softmax)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        similarity_matrix,
        positive_labels,
        reduction='mean'
    )

    return loss
```

4. **Masked Multimodal Modeling**: Predict masked tokens across both modalities

```python
def masked_multimodal_modeling_loss(model, input_ids, token_types, mask_ratio=0.15):
    """
    Masked prediction task: mask tokens and predict both image and text masks.
    Leverages discrete token representation to enable joint prediction.
    """
    # Randomly mask tokens (both image and text)
    batch_size, seq_len = input_ids.shape
    mask_positions = torch.rand(batch_size, seq_len) < mask_ratio

    # Store original tokens before masking
    original_tokens = input_ids.clone()

    # Mask tokens (use special [MASK] token)
    mask_token_id = 0  # Special token ID for masked positions
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_positions] = mask_token_id

    # Forward pass with masked input
    outputs = model(masked_input_ids, token_types=token_types)
    hidden_states = outputs['hidden_states']

    # Predict masked tokens from hidden states
    # Linear layer maps hidden states to vocabulary predictions
    logits = torch.nn.Linear(hidden_states.shape[-1], vocab_size)(hidden_states)

    # Compute loss only on masked positions
    mask_loss = torch.nn.functional.cross_entropy(
        logits[mask_positions],  # Predictions at masked positions
        original_tokens[mask_positions],  # Original tokens
        reduction='mean'
    )

    return mask_loss
```

5. **Hard Negative Sampling**: Include semantically similar negatives for robust learning

```python
def select_hard_negatives(image_embeddings, text_embeddings, positives_mask,
                         num_hard_negatives=4):
    """
    Select hard negatives: samples with high similarity to positive pair
    but different label. Improves contrastive learning robustness.
    """
    batch_size = image_embeddings.shape[0]

    # Compute similarity to all text samples
    similarity_to_all_text = torch.matmul(
        torch.nn.functional.normalize(image_embeddings, dim=1),
        torch.nn.functional.normalize(text_embeddings, dim=1).T
    )

    hard_negatives = []

    for i in range(batch_size):
        # Get similarities for this image
        similarities = similarity_to_all_text[i]

        # Mark positive (same index) as unavailable
        similarities[i] = -float('inf')

        # Select top-k hardest negatives (highest similarity)
        _, hard_neg_indices = torch.topk(similarities, k=num_hard_negatives)
        hard_negatives.append(hard_neg_indices)

    return hard_negatives
```

6. **Training Configuration**: Full setup for FuseLIP training

```python
def train_fuselip(model, tokenizer, train_loader, num_epochs=16, learning_rate=1e-3):
    """
    Training loop combining sigmoid contrastive and masked modeling losses.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            images, texts = batch['image'], batch['text']

            # Tokenize
            input_ids, token_types = tokenizer.tokenize_pair(images, texts)

            # Forward pass
            outputs = model(input_ids, token_types=token_types)
            image_proj = outputs['projection'][:len(images)]
            text_proj = outputs['projection'][len(images):]

            # Sigmoid contrastive loss
            contrastive_loss = sigmoid_contrastive_loss(image_proj, text_proj)

            # Masked modeling loss
            masked_loss = masked_multimodal_modeling_loss(
                model, input_ids, token_types, mask_ratio=0.15
            )

            # Combined loss
            total_loss = 0.7 * contrastive_loss + 0.3 * masked_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss={total_loss:.4f}")

        scheduler.step()

    return model
```

## Practical Guidance

**When to Apply:**
- Building multimodal retrieval systems prioritizing structure understanding
- Need efficient single-encoder architecture vs. dual-encoder overhead
- Working with vision-language tasks (image search, grounding, editing)
- Want to improve performance on structure-aware tasks

**Architecture Selection:**
- FuseLIP-S: Smaller, faster inference, ~200M parameters
  - 384-dim hidden, 6 attention heads, 12 layers
  - Good for mobile/edge deployment
- FuseLIP-B: Larger, higher capacity, ~500M+ parameters
  - 512-dim hidden, 8 attention heads, 12 layers
  - Recommended for maximum performance

**Training Setup:**
- Batch size: 2048 (adjust by GPU memory)
- Learning rate: 1×10⁻³ with cosine annealing
- Data: CC3M (93M samples, 8 epochs) or CC12M (326M samples, 16 epochs)
- Hardware: 8× V100 or equivalent for full training

**Loss Weight Tuning:**
- Contrastive loss: 0.7 (primary signal for embeddings)
- Masked modeling: 0.3 (auxiliary signal for robustness)
- Adjust ratio if task-specific: More contrastive for retrieval, more masked for generation

**Performance Expectations:**
- Superior performance on structure-aware tasks:
  - Text-guided image transformations: +15-20% vs. late fusion
  - Visual grounding: +10-15% improvement
- Comparable to late fusion on standard retrieval tasks
- Faster inference: Single forward pass vs. dual encoders

**Common Issues:**
- Divergent tokens (image/text vocabularies): Ensure offset implementation
- Unstable training: Increase batch size, reduce learning rate
- Poor structure understanding: Increase masked modeling loss weight
- Inference latency: Use FuseLIP-S, quantization, distillation

## Reference

Implemented in PyTorch with TiTok tokenizers and standard transformer blocks. Training on CC3M and CC12M datasets. Evaluated on text-guided image transformations (TGIT), VQA pairs, visual grounding (Visual Genome), and HQ-Edit image editing dataset. Shows strong performance especially on structure-understanding tasks while maintaining computational efficiency of single-encoder architecture.
