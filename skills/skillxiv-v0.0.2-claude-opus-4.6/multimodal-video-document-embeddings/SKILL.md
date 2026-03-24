---
name: multimodal-video-document-embeddings
title: "VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.04590"
keywords: [Multimodal Embeddings, Video Understanding, Document Retrieval, Cross-modal Search, Vision-Language Models]
description: "Generate unified embeddings for videos, images, and visual documents enabling semantic similarity, retrieval, and clustering across heterogeneous visual content types."
---

# VLM2Vec-V2: Unified Multimodal Embeddings Across Visual Modalities

Existing multimodal embedding systems excel at image-text tasks but struggle with videos and visual documents (PDFs, scanned documents). VLM2Vec-V2 extends multimodal embeddings to handle three visual modalities: natural images, temporal video sequences, and structured visual documents. This unified approach enables semantic search, retrieval-augmented generation, and clustering across diverse visual content, making it practical for AI agents and multimodal RAG systems that must handle mixed-media corpora.

The key innovation is designing embedding architectures that respect the unique properties of each modality—videos require temporal reasoning, documents require spatial layout understanding, images need fine-grained appearance features—while projecting all into a shared embedding space. A new benchmark (MMEB-V2) provides comprehensive evaluation across five new task categories (visual document retrieval, video retrieval, temporal grounding, video classification, video QA).

## Core Concept

VLM2Vec-V2 operates on the principle that embeddings should preserve semantic meaning across modality boundaries. A video frame, a matching document screenshot, and related images should have similar embeddings despite different structures. The model learns unified representations by encoding each modality's unique structure (temporal sequences for video, spatial regions for documents, global features for images) then projecting into a shared space.

The architecture uses vision-language pretraining as the foundation, fine-tuning task-specific heads per modality while sharing a common embedding projection. This allows specialization per modality while maintaining compatibility—embeddings from any source can be compared directly in the unified space.

## Architecture Overview

The system comprises modality-specific encoders unified by a shared embedding space:

- **Image Encoder**: Processes static images using vision transformer backbone (ViT), captures global and local features
- **Video Encoder**: Temporal sequence processor using 3D convolutions or attention, aggregates frames into video-level representations
- **Document Encoder**: Spatial layout processor for PDFs and visual documents, handles multi-page reasoning
- **Unified Projection**: Maps all modality representations to shared embedding space, enabling cross-modal retrieval
- **Task-Specific Heads**: Lightweight adapters for different downstream tasks (retrieval, classification, grounding)

## Implementation

Start with modality-specific encoders:

```python
import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel
from typing import Dict, Optional, List

class ImageEncoder(nn.Module):
    """
    Encode static images into feature representations.

    Uses vision transformer backbone capturing both global and local
    image structure for retrieval and matching tasks.
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k",
                 hidden_dim: int = 768):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim

        # Projection to embedding space
        self.projection = nn.Linear(self.vit.config.hidden_size, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: (batch, 3, 224, 224) RGB images

        Returns:
            embeddings: (batch, hidden_dim)
        """
        outputs = self.vit(images, output_hidden_states=True)

        # Use [CLS] token as global image representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Project to embedding space
        embeddings = self.projection(cls_output)
        return embeddings

class VideoEncoder(nn.Module):
    """
    Encode video sequences into fixed-dimensional representations.

    Uses temporal attention to aggregate frame information,
    capturing both motion and semantic content across frames.
    """

    def __init__(self, hidden_dim: int = 768, num_frames: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # Frame encoder (reuse image encoder backbone)
        self.frame_encoder = ImageEncoder(hidden_dim=hidden_dim)

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=12, batch_first=True
        )

        # Temporal projection
        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video to embedding by processing frames temporally.

        Args:
            video_frames: (batch, num_frames, 3, 224, 224)

        Returns:
            embeddings: (batch, hidden_dim)
        """
        batch_size, num_frames, channels, h, w = video_frames.shape

        # Reshape to process all frames at once
        frames_flat = video_frames.reshape(batch_size * num_frames, channels, h, w)

        # Encode each frame
        frame_embeddings = self.frame_encoder(frames_flat)
        frame_embeddings = frame_embeddings.reshape(batch_size, num_frames, -1)

        # Temporal aggregation via attention
        attended, _ = self.temporal_attention(
            frame_embeddings, frame_embeddings, frame_embeddings
        )

        # Global average pooling over temporal dimension
        video_embedding = attended.mean(dim=1)

        # Project through temporal head
        video_embedding = self.temporal_proj(video_embedding)

        return video_embedding

class DocumentEncoder(nn.Module):
    """
    Encode visual documents (PDFs, scanned pages) into representations.

    Processes document images considering spatial layout and multi-page
    structure, enabling document retrieval and understanding.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim

        # For multi-page documents, aggregate page embeddings
        self.page_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=12, batch_first=True
        )

        # Layout-aware projection
        self.layout_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, doc_pages: torch.Tensor,
                num_pages_per_doc: Optional[List[int]] = None) -> torch.Tensor:
        """
        Encode document pages into document-level embeddings.

        Args:
            doc_pages: (total_pages, 3, 224, 224) all document pages flattened
            num_pages_per_doc: list of page counts per document for proper aggregation

        Returns:
            embeddings: (num_docs, hidden_dim)
        """
        # Encode all pages
        page_embeddings = self.image_encoder(doc_pages)

        if num_pages_per_doc is None:
            # Single page document
            return page_embeddings

        # Aggregate multi-page documents
        doc_embeddings = []
        offset = 0

        for num_pages in num_pages_per_doc:
            # Get embeddings for this document's pages
            doc_page_embs = page_embeddings[offset:offset+num_pages]

            # Pad to fixed size if needed
            max_pages = max(num_pages_per_doc)
            if num_pages < max_pages:
                padding = torch.zeros(
                    max_pages - num_pages, self.hidden_dim,
                    device=doc_page_embs.device
                )
                doc_page_embs = torch.cat([doc_page_embs, padding], dim=0)

            # Aggregate pages via attention
            doc_page_embs = doc_page_embs.unsqueeze(0)  # (1, pages, hidden)
            aggregated, _ = self.page_attention(
                doc_page_embs, doc_page_embs, doc_page_embs
            )
            doc_emb = aggregated.mean(dim=1).squeeze(0)  # (hidden_dim,)

            doc_embeddings.append(doc_emb)
            offset += num_pages

        return torch.stack(doc_embeddings)
```

Implement the unified embedding space and projection:

```python
class UnifiedEmbeddingSpace(nn.Module):
    """
    Project all modalities into shared embedding space.

    Enables cross-modal retrieval by putting images, videos, and documents
    in the same vector space while preserving their distinct properties.
    """

    def __init__(self, hidden_dim: int = 768, embedding_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Modality-specific projections then unified projection
        self.image_proj = nn.Linear(hidden_dim, embedding_dim)
        self.video_proj = nn.Linear(hidden_dim, embedding_dim)
        self.document_proj = nn.Linear(hidden_dim, embedding_dim)

        # Normalize embeddings to unit norm
        self.l2_norm = nn.functional.normalize

    def forward_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project image features to embedding space."""
        embeddings = self.image_proj(image_features)
        return self.l2_norm(embeddings, p=2, dim=-1)

    def forward_video(self, video_features: torch.Tensor) -> torch.Tensor:
        """Project video features to embedding space."""
        embeddings = self.video_proj(video_features)
        return self.l2_norm(embeddings, p=2, dim=-1)

    def forward_document(self, doc_features: torch.Tensor) -> torch.Tensor:
        """Project document features to embedding space."""
        embeddings = self.document_proj(doc_features)
        return self.l2_norm(embeddings, p=2, dim=-1)
```

Implement the retrieval pipeline:

```python
class VLM2VecV2(nn.Module):
    """
    Complete multimodal embedding model for images, videos, and documents.

    Enables unified semantic search and retrieval across visual modalities.
    """

    def __init__(self, hidden_dim: int = 768, embedding_dim: int = 512):
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim=hidden_dim)
        self.video_encoder = VideoEncoder(hidden_dim=hidden_dim)
        self.document_encoder = DocumentEncoder(hidden_dim=hidden_dim)
        self.embedding_space = UnifiedEmbeddingSpace(hidden_dim, embedding_dim)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode and project images."""
        features = self.image_encoder(images)
        return self.embedding_space.forward_image(features)

    def encode_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Encode and project videos."""
        features = self.video_encoder(video_frames)
        return self.embedding_space.forward_video(features)

    def encode_document(self, doc_pages: torch.Tensor,
                       num_pages_per_doc: Optional[List[int]] = None) -> torch.Tensor:
        """Encode and project documents."""
        features = self.document_encoder(doc_pages, num_pages_per_doc)
        return self.embedding_space.forward_document(features)

    def retrieve(self, query: torch.Tensor, corpus: torch.Tensor,
                top_k: int = 10) -> torch.Tensor:
        """
        Retrieve top-k items from corpus most similar to query embedding.

        Both query and corpus should be in same embedding space.
        """
        # Compute similarity scores
        similarities = torch.matmul(query, corpus.t())  # (batch, corpus_size)

        # Get top-k
        top_scores, top_indices = torch.topk(similarities, k=min(top_k, corpus.size(0)), dim=1)

        return top_indices

    def cluster(self, embeddings: torch.Tensor,
               num_clusters: int = 10) -> torch.Tensor:
        """
        Cluster embeddings using k-means.

        Works across all modalities since embeddings are in same space.
        """
        from sklearn.cluster import KMeans

        # Move to CPU for clustering
        emb_np = embeddings.detach().cpu().numpy()

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(emb_np)

        return torch.from_numpy(cluster_labels)
```

## Practical Guidance

**Hyperparameter Table:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Hidden dimension | 768 | 512-1024 | ViT hidden size; match backbone |
| Embedding dimension | 512 | 256-768 | Final embedding space size |
| Number of frames | 8 | 4-16 | Frames sampled per video; more = better but slower |
| Image resolution | 224 | 224-384 | Input resolution; higher = more detail but slower |
| Batch size | 64 | 16-256 | Larger = better gradient estimates |
| Temperature (for contrast loss) | 0.07 | 0.01-0.5 | Lower = sharper similarity; avoid too low |

**When to Use:**
- You need to search/retrieve across mixed visual content (images, videos, PDFs)
- You're building a multimodal RAG system that must handle diverse document formats
- You want unified embeddings for clustering heterogeneous visual data
- You're developing AI agents that work with multiple visual modality types
- You need semantic similarity across images, videos, and document content

**When NOT to Use:**
- You only work with one modality (e.g., images only); single-modality models are simpler and faster
- You need extremely low latency (multiple encoders add overhead)
- You're working with very short video clips where temporal info is minimal
- Your documents are mostly text without visual layout (OCR + text embeddings sufficient)
- You have very limited compute budget (model requires multiple encoders)

**Common Pitfalls:**
- **Frame sampling bias**: Fixed frame sampling may miss key moments. Use adaptive or motion-aware sampling.
- **Document page order**: Multi-page aggregation order matters. Preserve page sequence in attention.
- **Modality imbalance**: If training data has unequal modality distribution, some encoders underfit. Balance carefully.
- **Embedding collapse**: Without proper normalization/contrastive loss, embeddings may collapse to similar values. Use unit normalization and temperature scaling.
- **Domain gap between modalities**: Videos and documents may have very different visual properties. Ensure diverse training data across modalities.
- **Computational cost**: Multiple encoders increase inference cost. Cache embeddings where possible.

## Reference

Authors (2025). VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents. arXiv preprint arXiv:2507.04590. https://arxiv.org/abs/2507.04590
