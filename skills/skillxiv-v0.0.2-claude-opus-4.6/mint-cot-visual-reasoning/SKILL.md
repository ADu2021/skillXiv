---
name: mint-cot-visual-reasoning
title: "MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.05331"
keywords: [multimodal-reasoning, chain-of-thought, visual-language, mathematics, fine-grained-vision]
description: "Integrates fine-grained visual tokens into mathematical reasoning via Interleave Tokens that dynamically select relevant image regions for each reasoning step."
---

# MINT-CoT: Interleaved Visual Tokens in Mathematical Reasoning

## Core Concept

Mathematical reasoning with diagrams requires precise alignment between textual reasoning steps and visual regions. Existing approaches use coarse bounding boxes, limiting the fine-grained visual understanding essential for geometry and diagram interpretation. MINT-CoT introduces Interleave Tokens that compute similarity scores between decoder states and visual tokens, enabling dynamic selection of non-rectangular image regions during reasoning. A 54K dataset with token-level alignment and three-stage progressive training enables substantial improvements on mathematical benchmarks.

## Architecture Overview

- **Interleave Tokens**: Special tokens that select relevant visual regions by computing similarity between decoder hidden states and visual token embeddings
- **Fine-Grained Selection**: Enables non-rectangular region selection, capturing diagram elements at arbitrary shapes
- **MINT-CoT Dataset**: 54K annotated problems with token-level alignment between reasoning steps and image regions
- **Three-Stage Training**: Text-only CoT → Interleaved CoT supervised → Interleaved CoT reinforcement learning
- **Automated Annotation**: Four-step pipeline (gridding, OCR, keyword extraction, alignment) for efficient dataset construction
- **GRPO Integration**: Reinforcement learning phase optimizes reasoning quality end-to-end

## Implementation

The following code demonstrates the Interleave Token mechanism and training pipeline:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class InterleaveToken(nn.Module):
    """
    Special token that selects relevant visual regions during reasoning.
    """
    def __init__(self, hidden_dim: int, num_visual_tokens: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_visual_tokens = num_visual_tokens

        # Projections for similarity computation
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.visual_key_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_state: torch.Tensor,
               visual_tokens: torch.Tensor,
               threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select visual regions by computing similarity with decoder state.

        decoder_state: (hidden_dim,) hidden state before this reasoning step
        visual_tokens: (num_visual_tokens, hidden_dim) image patches encoded
        threshold: similarity threshold for region selection

        Returns: (selected_regions, selection_mask)
        """
        # Project for similarity computation
        query = self.query_proj(decoder_state)  # (hidden_dim,)
        keys = self.visual_key_proj(visual_tokens)  # (num_visual_tokens, hidden_dim)

        # Compute cosine similarity
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)

        similarity = torch.matmul(keys_norm, query_norm)  # (num_visual_tokens,)

        # Soft selection: apply softmax for differentiable selection
        selection_weights = F.softmax(similarity * 10.0, dim=0)  # Temperature=10

        # Hard threshold: which regions to include
        selection_mask = (similarity > threshold).float()

        # Weighted combination of selected visual tokens
        selected_regions = torch.matmul(
            selection_weights.unsqueeze(0), visual_tokens
        )  # (1, hidden_dim)

        return selected_regions, selection_mask


class MINTCoTModel(nn.Module):
    """
    Multimodal model with interleaved visual tokens for math reasoning.
    """
    def __init__(self, hidden_dim: int = 4096, vocab_size: int = 32000,
                 num_visual_tokens: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Language model components
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=4*hidden_dim,
                                       batch_first=True),
            num_layers=12
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Vision components
        self.vision_encoder = nn.Identity()  # In practice, use CLIP or similar
        self.interleave_token = InterleaveToken(hidden_dim, num_visual_tokens)

    def encode_image_to_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image into visual tokens (e.g., patch embeddings).

        image: (B, 3, H, W)
        Returns: (B, num_visual_tokens, hidden_dim)
        """
        # Vision encoder produces patch embeddings
        # Simplified: assuming fixed 16x16 = 256 patches
        visual_tokens = self.vision_encoder(image)  # (B, num_patches, hidden_dim)
        return visual_tokens

    def forward_with_interleave(self, input_ids: torch.Tensor,
                               image: torch.Tensor,
                               visual_token_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with interleaved visual token selection.

        input_ids: (batch, seq_len) token sequence
        image: (batch, 3, H, W) input image
        visual_token_mask: (batch, seq_len, num_visual_tokens) which tokens trigger visual selection

        Returns: (batch, seq_len, vocab_size) logits
        """
        batch_size, seq_len = input_ids.shape

        # Encode image
        visual_tokens = self.encode_image_to_tokens(image)  # (batch, num_visual_tokens, hidden_dim)

        # Embed text tokens
        text_embeds = self.embedding(input_ids)  # (batch, seq_len, hidden_dim)

        # Interleave visual tokens where needed
        interleaved_embeds = text_embeds.clone()

        for pos in range(seq_len):
            if visual_token_mask[:, pos].any():
                # This position triggers visual selection
                decoder_state = text_embeds[:, pos]  # (batch, hidden_dim)

                # For each batch item, select visual regions
                for b in range(batch_size):
                    if visual_token_mask[b, pos]:
                        selected_regions, _ = self.interleave_token(
                            decoder_state[b], visual_tokens[b]
                        )
                        # Blend selected visual information with text embedding
                        interleaved_embeds[b, pos] = 0.7 * text_embeds[b, pos] + 0.3 * selected_regions.squeeze(0)

        # Transformer forward
        output = self.transformer(interleaved_embeds, memory=None)
        logits = self.output_proj(output)

        return logits


class MINTCoTDataset:
    """
    Dataset construction pipeline for MINT-CoT.
    """
    def __init__(self):
        self.grid_size = 16  # 16x16 grid

    def grid_image(self, image: torch.Tensor) -> List[Tuple[int, int, int, int]]:
        """
        Divide image into grid cells.
        Returns: list of (x1, y1, x2, y2) coordinates
        """
        _, h, w = image.shape
        cell_h = h // self.grid_size
        cell_w = w // self.grid_size

        grid_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1, y1 = j * cell_w, i * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                grid_cells.append((x1, y1, x2, y2))

        return grid_cells

    def extract_keywords_from_step(self, reasoning_step: str) -> List[str]:
        """
        Extract keywords from a reasoning step using GPT-4o.
        In practice, use language model API.
        """
        # Simplified: would call GPT-4o in real implementation
        keywords = reasoning_step.split()[:3]  # Placeholder
        return keywords

    def annotate_visual_regions(self, image: torch.Tensor,
                               reasoning_step: str) -> List[int]:
        """
        Map reasoning step keywords to image grid cells.
        Returns: list of grid cell indices relevant to this step.
        """
        keywords = self.extract_keywords_from_step(reasoning_step)
        grid_cells = self.grid_image(image)

        # In practice, use OCR to locate keyword positions in image
        # For now, return placeholder annotations
        relevant_cells = [0, 1, 16, 17]  # Example cells

        return relevant_cells

    def create_dataset_sample(self, image: torch.Tensor,
                             problem: str,
                             reasoning_chain: List[str],
                             answer: str) -> dict:
        """
        Create single dataset sample with token-level visual annotations.
        """
        # Tokenize problem + reasoning chain
        full_text = problem + " " + " ".join(reasoning_chain) + " " + answer
        tokens = full_text.split()  # Simplified tokenization

        # Annotate which tokens trigger visual selection
        visual_token_mask = [0] * len(tokens)

        for step_idx, step in enumerate(reasoning_chain):
            step_keywords = self.extract_keywords_from_step(step)
            # Find token positions for this step
            step_start = sum(len(r.split()) for r in reasoning_chain[:step_idx])
            step_end = step_start + len(step.split())

            # Mark these tokens as visual
            for pos in range(step_start, min(step_end, len(visual_token_mask))):
                visual_token_mask[pos] = 1

        return {
            'image': image,
            'tokens': tokens,
            'visual_mask': visual_token_mask,
            'answer': answer
        }


class MINTCoTTrainer:
    """
    Three-stage training for MINT-CoT.
    """
    def __init__(self, model: MINTCoTModel):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def stage1_text_cot_sft(self, dataset: List[dict], epochs: int = 3):
        """Stage 1: Supervised fine-tuning on text-only chain-of-thought."""
        for epoch in range(epochs):
            for sample in dataset:
                # Train on text tokens only
                logits = self.model(torch.tensor(sample['tokens']),
                                  sample['image'],
                                  torch.zeros_like(sample['visual_mask']))
                loss = F.cross_entropy(logits.view(-1, self.model.vocab_size),
                                      torch.tensor(sample['tokens']))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def stage2_interleaved_cot_sft(self, dataset: List[dict], epochs: int = 3):
        """Stage 2: Supervised fine-tuning with interleaved visual tokens."""
        for epoch in range(epochs):
            for sample in dataset:
                # Include visual selection mask
                visual_mask = torch.tensor(sample['visual_mask']).unsqueeze(0)
                logits = self.model(torch.tensor(sample['tokens']).unsqueeze(0),
                                  sample['image'].unsqueeze(0),
                                  visual_mask)

                loss = F.cross_entropy(logits.view(-1, self.model.vocab_size),
                                      torch.tensor(sample['tokens']))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    def stage3_interleaved_grpo(self, dataset: List[dict], num_iterations: int = 100):
        """Stage 3: Group Relative Policy Optimization for reasoning quality."""
        for _ in range(num_iterations):
            # GRPO: generate multiple reasoning chains, rank by correctness
            # Optimize for better reasoning while using visual tokens
            pass  # Implementation details omitted for brevity
```

## Practical Guidance

**Grid Size Selection**: 16×16 grid (256 cells) provides good granularity for diagram understanding. Increase to 32×32 for high-resolution images with fine details.

**Similarity Threshold**: Set to 0.5 for 50% similarity threshold in Interleave Token selection. Lower thresholds (0.3) increase region selection; higher (0.7) make selection more selective.

**Keyword Extraction**: Use GPT-4o for accurate keyword extraction from reasoning steps. Alternatively, use domain-specific keyword lists for specific math domains.

**Annotation Quality**: The dataset quality is critical. Ensure OCR accuracy and proper alignment between reasoning steps and image regions before training.

**Training Schedule**: Follow the three-stage progression strictly. Each stage builds on previous; skipping stages hurts performance.

**Visualization**: Visualize selected image regions during training to verify Interleave Token selection aligns with reasoning steps.

## Reference

MINT-CoT achieves substantial improvements on mathematical reasoning:
- **MathVista (mathematical subset)**: +32.59% improvement
- **GeoQA**: +26.92% improvement
- **Geometry-related tasks**: Surpasses state-of-the-art models

The 54K annotated dataset with fine-grained visual alignment enables models to leverage diagram information effectively. The approach is particularly valuable for geometry, diagram-based reasoning, and scientific problem-solving where visual information is essential.
