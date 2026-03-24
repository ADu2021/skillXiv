---
name: moca-multimodal-embeddings
title: "MoCa: Modality-aware Continual Pre-training Makes Better Bidirectional Multimodal Embeddings"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.23115"
keywords: [Multimodal Embeddings, Continual Pre-training, Vision Language Models, Bidirectional Embeddings, Cross-modal Retrieval]
description: "Transform pre-trained vision-language models into powerful bidirectional multimodal embeddings through modality-aware continual pre-training and heterogeneous contrastive fine-tuning. 3B model matches 7B baselines."
---

# MoCa: Efficient Multimodal Embeddings Through Modality-Aware Pre-training

Vision-language models are excellent at understanding images and text, but they're designed for generative tasks (caption generation, visual question answering). When you extract embeddings from VLMs for retrieval tasks (find similar images to a text query), performance lags significantly behind specialized embedding models. The problem is that VLM embeddings are optimized for generation, not for forming a unified representation space where similar images and text are close in distance.

MoCa solves this by adapting pre-trained VLMs into embedding models through two stages: (1) modality-aware continual pre-training using both masked language modeling and masked autoencoding to encourage cross-modal understanding, and (2) heterogeneous contrastive fine-tuning on diverse data including long-form documents, curated pairs, and text-only data. The result achieves state-of-the-art multimodal retrieval performance with smaller models (3B matching 7B baselines).

## Core Concept

The key insight is that **embedding spaces have different properties than generative representations**. VLMs optimize for:
- Causal attention (left-to-right generation)
- Autoregressive loss (predict next token)
- Task-specific outputs (caption generation)

MoCa embeddings need:
- Bidirectional context (both left and right matter)
- Contrastive objectives (similar items close, dissimilar far)
- Unified cross-modal space (image and text in same space)

This requires modification at two levels:
1. Changing the attention mechanism from causal to bidirectional
2. Changing the training objective from generation to contrastive learning

The approach uses two stages: (1) unlabeled continual pre-training to adapt representations without task-specific labels, and (2) contrastive fine-tuning with diverse data to build discriminative embeddings.

## Architecture Overview

MoCa modifies a vision-language model at the architecture and training level:

- **Vision Encoder**: Vision Transformer (unchanged from VLM)
- **Text Encoder**: Modified to use bidirectional attention instead of causal
- **Shared Representation Space**: Single unified embedding space for both modalities
- **Mean Pooling**: Image and text embeddings are averaged (not CLS tokens) for better uniformity
- **Multi-stage Training**: CPT (continual pre-training) followed by heterogeneous contrastive fine-tuning

## Implementation

**Step 1: Modify the VLM for bidirectional embeddings**

Convert the causal language model to bidirectional by changing attention patterns and adding bidirectional embeddings.

```python
import torch
import torch.nn as nn

class BidirectionalTextEncoder(nn.Module):
    """
    Modify a causal LLM to use bidirectional attention for embeddings.
    Key changes: replace causal mask with bidirectional, use mean pooling.
    """

    def __init__(self, original_model):
        super().__init__()
        self.embedding = original_model.embed_tokens
        self.layers = original_model.model.layers

        # Replace causal attention with bidirectional
        self.layers = self._convert_to_bidirectional(self.layers)

        self.norm = original_model.model.norm

    def _convert_to_bidirectional(self, layers):
        """
        Modify attention layers from causal (triangular mask) to bidirectional.
        """
        modified_layers = nn.ModuleList()

        for layer in layers:
            # Get the attention block
            attn = layer.self_attn

            # Create a wrapper that removes the causal mask
            original_forward = attn.forward

            def bidirectional_forward(hidden_states, *args, **kwargs):
                # Remove attention_mask or replace with all-ones mask
                # This enables bidirectional attention
                kwargs['attention_mask'] = None
                return original_forward(hidden_states, *args, **kwargs)

            attn.forward = bidirectional_forward
            modified_layers.append(layer)

        return modified_layers

    def forward(self, input_ids, attention_mask=None):
        """
        Encode text using bidirectional attention.
        Return mean-pooled embeddings instead of next-token logits.
        """
        # Embed tokens
        hidden_states = self.embedding(input_ids)  # [batch, seq_length, hidden_dim]

        # Apply transformer layers with bidirectional attention
        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Mean pooling: average over sequence length
        # This is better than CLS token for embedding uniformity
        embeddings = hidden_states.mean(dim=1)  # [batch, hidden_dim]

        return embeddings
```

**Step 2: Implement modality-aware continual pre-training**

Use masked language modeling and masked autoencoding jointly to encourage cross-modal understanding without task-specific labels.

```python
def continual_pretrain_moca(vision_encoder, text_encoder, unlabeled_data,
                           num_epochs=3, learning_rate=1e-5):
    """
    Stage 1: Continual pre-training with MLM and MAE on unlabeled multimodal data.
    This adapts the VLM representations to the embedding task without labeled data.
    """
    optimizer = torch.optim.AdamW(
        list(vision_encoder.parameters()) + list(text_encoder.parameters()),
        lr=learning_rate
    )

    for epoch in range(num_epochs):
        for batch in unlabeled_data:
            images = batch['images'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)

            # Task 1: Masked Language Modeling (MLM)
            # Randomly mask tokens, predict them from multimodal context
            text_ids_mlm = text_input_ids.clone()
            mask_indices = torch.rand(text_ids_mlm.shape) < 0.15
            masked_token_id = tokenizer.encode('[MASK]')[1]
            text_ids_mlm[mask_indices] = masked_token_id

            # Encode images
            image_embeddings = vision_encoder(images)  # [batch, num_patches, dim]

            # Encode masked text
            text_embeddings = text_encoder(text_ids_mlm)  # [batch, dim]

            # Fuse: concatenate image and text embeddings
            fused = torch.cat([image_embeddings, text_embeddings.unsqueeze(1)], dim=1)

            # Predict masked tokens from fused representation
            mlm_head = nn.Linear(fused.shape[-1], tokenizer.vocab_size)
            mlm_logits = mlm_head(fused[:, -1:, :])  # Use last (text) token
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, tokenizer.vocab_size),
                text_input_ids[mask_indices]
            )

            # Task 2: Masked Autoencoding (MAE)
            # Randomly mask image patches, reconstruct from multimodal context
            image_patches = vision_encoder.patchify(images)  # [batch, num_patches, patch_dim]
            num_patches = image_patches.shape[1]
            mask_indices_img = torch.rand(num_patches) < 0.75  # 75% masking for images

            # Create masked image
            masked_patches = image_patches.clone()
            masked_patches[mask_indices_img] = 0  # Zero out masked patches

            # Encode masked image
            masked_image_embeddings = vision_encoder.encoder(masked_patches)

            # Text helps reconstruct image
            text_embeddings_mae = text_encoder(text_input_ids)

            # Fuse: image reconstruction guided by text
            fused_mae = torch.cat([masked_image_embeddings,
                                  text_embeddings_mae.unsqueeze(1)], dim=1)

            # Reconstruct masked patches
            mae_head = nn.Linear(fused_mae.shape[-1], image_patches.shape[-1])
            reconstructed = mae_head(fused_mae[:, :num_patches, :])

            # Loss: reconstruct only masked patches
            mae_loss = F.mse_loss(
                reconstructed[mask_indices_img],
                image_patches[mask_indices_img]
            )

            # Combined loss
            total_loss = mlm_loss + mae_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vision_encoder.parameters()) + list(text_encoder.parameters()),
                1.0
            )
            optimizer.step()

    return vision_encoder, text_encoder
```

**Step 3: Heterogeneous contrastive fine-tuning**

Train with diverse data types (image-caption pairs, long-form documents, text-only) to build robust embeddings.

```python
def heterogeneous_contrastive_finetune(vision_encoder, text_encoder,
                                      mixed_data, learning_rate=1e-5, num_epochs=5):
    """
    Stage 2: Contrastive fine-tuning with heterogeneous data.
    Data includes: image-caption pairs, long documents, text-only examples, curated pairs.
    """
    optimizer = torch.optim.AdamW(
        list(vision_encoder.parameters()) + list(text_encoder.parameters()),
        lr=learning_rate
    )

    for epoch in range(num_epochs):
        for batch_type, batch in mixed_data:
            if batch_type == 'image_caption':
                # Standard image-caption pairs: minimize distance between matching pairs
                images = batch['images'].to(device)
                captions = batch['captions_input_ids'].to(device)

                # Encode
                image_embeddings = vision_encoder(images)  # [batch, dim]
                text_embeddings = text_encoder(captions)   # [batch, dim]

                # Normalize embeddings
                image_embeddings = F.normalize(image_embeddings, dim=-1)
                text_embeddings = F.normalize(text_embeddings, dim=-1)

                # Contrastive loss: cosine similarity
                # Positive: matching image-caption (diagonal)
                # Negative: non-matching (off-diagonal)
                logits = torch.matmul(image_embeddings, text_embeddings.T)  # [batch, batch]
                labels = torch.arange(logits.shape[0], device=device)

                loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
                loss = loss / 2

            elif batch_type == 'long_document':
                # Long-form documents: embed different paragraphs, keep semantically similar ones close
                document_ids = batch['document_input_ids'].to(device)
                paragraph_indices = batch['paragraph_boundaries']

                # Encode full document
                full_embeddings = text_encoder(document_ids)

                # Extract per-paragraph embeddings
                paragraph_embeddings = []
                for start, end in paragraph_indices:
                    para_emb = full_embeddings[0, start:end, :].mean(dim=0)
                    paragraph_embeddings.append(para_emb)

                # Contrastive: consecutive paragraphs should be similar
                paragraph_embeddings = torch.stack(paragraph_embeddings)
                paragraph_embeddings = F.normalize(paragraph_embeddings, dim=-1)

                loss = 0
                for i in range(len(paragraph_embeddings) - 1):
                    sim = torch.mm(paragraph_embeddings[i:i+1],
                                 paragraph_embeddings.T)
                    target = torch.zeros(1, device=device, dtype=torch.long)
                    target[0, i + 1] = 1  # Next paragraph is positive
                    loss += F.cross_entropy(sim, target)

            elif batch_type == 'text_only':
                # Text-only data: similar semantic content should be close
                text_ids_a = batch['text_a_input_ids'].to(device)
                text_ids_b = batch['text_b_input_ids'].to(device)
                similarity_labels = batch['similarity'].to(device)

                embeddings_a = text_encoder(text_ids_a)
                embeddings_b = text_encoder(text_ids_b)

                embeddings_a = F.normalize(embeddings_a, dim=-1)
                embeddings_b = F.normalize(embeddings_b, dim=-1)

                # Cosine similarity loss
                cos_sim = (embeddings_a * embeddings_b).sum(dim=-1)
                loss = F.mse_loss(cos_sim, similarity_labels.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vision_encoder.parameters()) + list(text_encoder.parameters()),
                1.0
            )
            optimizer.step()

    return vision_encoder, text_encoder
```

**Step 4: Evaluate on multimodal retrieval benchmarks**

Test image-to-text and text-to-image retrieval on MMEB and ViDoRe benchmarks.

```python
def evaluate_multimodal_retrieval(vision_encoder, text_encoder,
                                 image_corpus, text_corpus,
                                 queries_text, queries_images):
    """
    Evaluate retrieval: how well does the model retrieve matching items?
    Metrics: Recall@1, Recall@5, Recall@10, nDCG.
    """
    vision_encoder.eval()
    text_encoder.eval()

    # Encode all images and text in corpus
    with torch.no_grad():
        image_embeddings = []
        for images in image_corpus:
            emb = vision_encoder(images.to(device))
            emb = F.normalize(emb, dim=-1)
            image_embeddings.append(emb)
        image_embeddings = torch.cat(image_embeddings, dim=0)

        text_embeddings = []
        for text_ids in text_corpus:
            emb = text_encoder(text_ids.to(device))
            emb = F.normalize(emb, dim=-1)
            text_embeddings.append(emb)
        text_embeddings = torch.cat(text_embeddings, dim=0)

    # Text-to-Image Retrieval
    t2i_recalls = {'r@1': [], 'r@5': [], 'r@10': []}

    with torch.no_grad():
        for query_text_ids in queries_text:
            query_emb = text_encoder(query_text_ids.to(device))
            query_emb = F.normalize(query_emb, dim=-1)

            # Compute similarities to all images
            similarities = torch.matmul(query_emb, image_embeddings.T)
            ranked_indices = torch.argsort(similarities, descending=True)[0]

            # Check if true positive is in top-K
            true_image_idx = query_text_ids.get('image_idx', 0)  # From metadata
            for k, threshold in [(1, 'r@1'), (5, 'r@5'), (10, 'r@10')]:
                if true_image_idx in ranked_indices[:k]:
                    t2i_recalls[threshold].append(1.0)
                else:
                    t2i_recalls[threshold].append(0.0)

    # Image-to-Text Retrieval (similar process)
    i2t_recalls = {'r@1': [], 'r@5': [], 'r@10': []}

    with torch.no_grad():
        for query_image in queries_images:
            query_emb = vision_encoder(query_image.to(device))
            query_emb = F.normalize(query_emb, dim=-1)

            similarities = torch.matmul(query_emb, text_embeddings.T)
            ranked_indices = torch.argsort(similarities, descending=True)[0]

            true_text_idx = query_image.get('text_idx', 0)
            for k, threshold in [(1, 'r@1'), (5, 'r@5'), (10, 'r@10')]:
                if true_text_idx in ranked_indices[:k]:
                    i2t_recalls[threshold].append(1.0)
                else:
                    i2t_recalls[threshold].append(0.0)

    # Average results
    results = {
        'text_to_image': {k: np.mean(v) for k, v in t2i_recalls.items()},
        'image_to_text': {k: np.mean(v) for k, v in i2t_recalls.items()}
    }

    return results
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| MLM masking ratio | 0.15 | Standard masked language model setting |
| MAE masking ratio | 0.75 | Images use higher masking than text |
| Contrastive temperature | 0.07 | Standard for retrieval; adjust if needed |
| Heterogeneous data ratio | 40% img/cap, 30% doc, 30% text | Balance across modalities |
| CPT epochs | 2-3 | Don't overfit; CPT is initialization |
| Fine-tuning epochs | 3-5 | Contrastive fine-tuning is more intensive |

**When to use MoCa embeddings:**
- You need strong image-text retrieval (multi-modal search)
- You want to use smaller models (3B matching 7B baselines)
- You have access to pre-trained VLMs to adapt
- You have diverse unlabeled multimodal data for pre-training

**When NOT to use MoCa:**
- You need generation capabilities (embeddings are retrieval-only)
- You're satisfied with your VLM's retrieval performance
- Compute for training is unavailable
- You don't have multimodal pre-training data

**Common pitfalls:**
- **Bidirectional attention breaks generation**: Only use for embeddings; don't generate from bidirectional models.
- **MLM/MAE losses dominate**: If contrastive fine-tuning stops improving, reduce CPT intensity (fewer epochs).
- **Mean pooling loses structure**: For very long documents, use hierarchical pooling (pool paragraphs, then documents).
- **Heterogeneous data imbalance**: Monitor losses per data type; if one type dominates, weight the loss function.

## Reference

MoCa: Modality-aware Continual Pre-training Makes Better Bidirectional Multimodal Embeddings
https://arxiv.org/abs/2506.23115
