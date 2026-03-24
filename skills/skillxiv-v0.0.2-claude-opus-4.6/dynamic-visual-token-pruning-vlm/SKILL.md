---
name: dynamic-visual-token-pruning-vlm
title: GlimpsePrune Dynamic Visual Token Pruning for Vision-Language Models
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.01548
keywords: [vision-language-models, token-pruning, efficiency, visual-compression, inference]
description: "Dynamic token pruning framework for VLMs that adapts compression to scene complexity through single-pass selection. Removes 92.6% of visual tokens while maintaining performance and enabling superior fine-tuning."
---

## GlimpsePrune: Dynamic Visual Token Pruning for Vision-Language Models

GlimpsePrune addresses computational inefficiency in Vision-Language Models by dynamically pruning irrelevant visual tokens based on input complexity. Rather than fixed compression ratios, the framework intelligently adapts token retention to scene characteristics, achieving massive efficiency gains while improving downstream task performance.

### Core Concept

The fundamental insight is that not all image regions contribute equally to answering queries. Complex scenes need more visual tokens while simple scenes can work with fewer. GlimpsePrune:

- **Analyzes scene complexity** to determine necessary compression ratio
- **Scores token importance** dynamically based on content and query relevance
- **Prunes tokens adaptively** in a single forward pass
- **Preserves critical information** while removing redundancy
- **Enables better fine-tuning** through cleaner token representations

### Architecture Overview

The framework consists of:

- **Visual Encoder**: Processes images into tokens
- **Complexity Analyzer**: Estimates scene complexity from visual features
- **Token Scorer**: Assigns importance scores to each token
- **Adaptive Pruning Module**: Selects tokens to retain based on complexity
- **Language Model**: Processes pruned tokens for downstream tasks
- **Fine-Tuning Adapter**: Leverages pruned tokens for improved learning

### Implementation Steps

**Step 1: Build visual token scorer**

Create a module that assigns importance scores to visual tokens:

```python
import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

class VisualTokenScorer(nn.Module):
    """Scores visual tokens for importance-based pruning"""

    def __init__(self, hidden_size: int = 768, num_layers: int = 2):
        super().__init__()

        # Multi-layer scoring network
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Score in [0, 1]
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Score each visual token for importance.

        Args:
            token_embeddings: Shape (batch, num_tokens, hidden_size)

        Returns:
            Importance scores of shape (batch, num_tokens)
        """
        batch_size, num_tokens, hidden_size = token_embeddings.shape

        # Reshape for scoring
        flat_embeddings = token_embeddings.reshape(
            batch_size * num_tokens,
            hidden_size
        )

        # Score each token
        scores = self.scorer(flat_embeddings)

        # Reshape back
        scores = scores.reshape(batch_size, num_tokens)

        return scores


class QueryAwareScorer(nn.Module):
    """Scores tokens based on relevance to input query"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()

        self.query_projector = nn.Linear(hidden_size, hidden_size)
        self.token_projector = nn.Linear(hidden_size, hidden_size)

    def forward(self, token_embeddings: torch.Tensor,
                query_embedding: torch.Tensor) -> torch.Tensor:
        """
        Score tokens based on query relevance.

        Args:
            token_embeddings: (batch, num_tokens, hidden_size)
            query_embedding: (batch, hidden_size)

        Returns:
            Query-aware importance scores (batch, num_tokens)
        """
        # Project query and tokens
        query_proj = self.query_projector(query_embedding)  # (batch, hidden)
        token_proj = self.token_projector(token_embeddings)  # (batch, tokens, hidden)

        # Compute relevance: dot product normalized by norms
        relevance = torch.bmm(
            token_proj,
            query_proj.unsqueeze(2)
        ).squeeze(2)  # (batch, num_tokens)

        # Normalize to [0, 1]
        relevance = torch.sigmoid(relevance)

        return relevance
```

This creates learnable importance scoring mechanisms.

**Step 2: Analyze scene complexity**

Estimate visual scene complexity to determine pruning ratio:

```python
class ComplexityAnalyzer:
    """Analyzes visual scene complexity to guide pruning"""

    def __init__(self, method: str = 'entropy'):
        self.method = method

    def compute_complexity(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Estimate scene complexity from token embeddings.

        Args:
            token_embeddings: (batch, num_tokens, hidden_size)

        Returns:
            Complexity scores for each image (batch,), range [0, 1]
        """
        if self.method == 'entropy':
            return self._entropy_complexity(token_embeddings)
        elif self.method == 'variance':
            return self._variance_complexity(token_embeddings)
        elif self.method == 'histogram':
            return self._histogram_complexity(token_embeddings)
        else:
            return torch.ones(token_embeddings.shape[0])

    def _entropy_complexity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Measure complexity via entropy of token distribution.

        Higher entropy = more diverse tokens = higher complexity
        """
        batch_size = embeddings.shape[0]
        complexities = []

        for i in range(batch_size):
            tokens = embeddings[i]  # (num_tokens, hidden)

            # Compute pairwise similarities
            normalized = torch.nn.functional.normalize(tokens, dim=1)
            sim_matrix = torch.mm(normalized, normalized.t())

            # Entropy of similarity distribution
            probs = torch.softmax(sim_matrix.mean(dim=0), dim=0)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()

            # Normalize: max entropy for num_tokens uniform distribution
            max_entropy = np.log(tokens.shape[0])
            normalized_entropy = (entropy / max_entropy).clamp(0, 1)

            complexities.append(normalized_entropy)

        return torch.stack(complexities)

    def _variance_complexity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Estimate complexity from feature variance"""
        batch_size = embeddings.shape[0]
        complexities = []

        for i in range(batch_size):
            tokens = embeddings[i]
            variance = tokens.var(dim=0).mean()

            # Normalize
            normalized_variance = torch.tanh(variance)  # Squash to [0, 1]
            complexities.append(normalized_variance)

        return torch.stack(complexities)

    def _histogram_complexity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Estimate from spatial distribution of tokens"""
        # Simplified: treat tokens as having positions
        batch_size = embeddings.shape[0]
        complexities = []

        for i in range(batch_size):
            tokens = embeddings[i]

            # Compute mean and spread
            mean = tokens.mean(dim=0)
            distances = torch.norm(tokens - mean, dim=1)
            spread = distances.std()

            # Normalize
            normalized = torch.tanh(spread)
            complexities.append(normalized)

        return torch.stack(complexities)

    def determine_target_ratio(self, complexity: torch.Tensor,
                              min_ratio: float = 0.1,
                              max_ratio: float = 0.9) -> torch.Tensor:
        """
        Map complexity to target retention ratio.

        Low complexity → lower ratio (more pruning)
        High complexity → higher ratio (fewer prune)
        """
        # Inverse relationship: lower complexity → more aggressive pruning
        ratio = min_ratio + (1 - complexity) * (max_ratio - min_ratio)

        return ratio
```

This enables adaptive pruning based on image properties.

**Step 3: Implement adaptive token selection**

Select which tokens to keep based on scores and complexity:

```python
class AdaptiveTokenSelector(nn.Module):
    """Selects tokens adaptively based on importance and complexity"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.token_scorer = VisualTokenScorer(hidden_size)
        self.complexity_analyzer = ComplexityAnalyzer()

    def forward(self, token_embeddings: torch.Tensor,
                attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptively select tokens to retain.

        Args:
            token_embeddings: (batch, num_tokens, hidden_size)
            attention_mask: (batch, num_tokens), optional

        Returns:
            (selected_tokens, selection_mask)
        """
        batch_size, num_tokens, hidden_size = token_embeddings.shape

        # Score all tokens
        importance_scores = self.token_scorer(token_embeddings)

        # Analyze complexity to determine target ratio
        complexity = self.complexity_analyzer.compute_complexity(token_embeddings)
        target_ratio = self.complexity_analyzer.determine_target_ratio(complexity)

        # Select top-k tokens adaptively
        selected_tokens_list = []
        selection_masks = []

        for b in range(batch_size):
            # Number of tokens to keep
            num_keep = max(1, int(num_tokens * target_ratio[b].item()))

            # Get top-k indices
            scores = importance_scores[b]

            # Always keep [CLS] token (usually first)
            topk_scores, topk_indices = torch.topk(
                scores[1:],  # Exclude first token
                k=min(num_keep - 1, num_tokens - 1)
            )

            # Reconstruct full token set with [CLS]
            all_indices = torch.cat([
                torch.tensor([0], device=scores.device),
                topk_indices + 1
            ])

            all_indices = all_indices.sort()[0]

            # Create selection mask
            mask = torch.zeros(num_tokens, dtype=torch.bool, device=scores.device)
            mask[all_indices] = True

            selected_token = token_embeddings[b][mask]
            selected_tokens_list.append(selected_token)
            selection_masks.append(mask)

        # Pad selected tokens to same length
        max_selected = max(t.shape[0] for t in selected_tokens_list)
        padded_tokens = torch.zeros(
            batch_size, max_selected, hidden_size,
            device=token_embeddings.device,
            dtype=token_embeddings.dtype
        )

        for b, tokens in enumerate(selected_tokens_list):
            padded_tokens[b, :tokens.shape[0]] = tokens

        return padded_tokens, torch.stack(selection_masks)
```

This adaptively selects tokens in a single pass.

**Step 4: Implement GlimpsePrune forward pass**

Integrate all components into unified pruning module:

```python
class GlimpsePrune(nn.Module):
    """Complete dynamic visual token pruning module"""

    def __init__(self, hidden_size: int = 768, enable_query_awareness: bool = True):
        super().__init__()

        self.hidden_size = hidden_size
        self.token_scorer = VisualTokenScorer(hidden_size)

        if enable_query_awareness:
            self.query_scorer = QueryAwareScorer(hidden_size)
        else:
            self.query_scorer = None

        self.complexity_analyzer = ComplexityAnalyzer()
        self.selector = AdaptiveTokenSelector(hidden_size)

    def forward(self, image_tokens: torch.Tensor,
                query_embedding: torch.Tensor = None,
                min_retention_ratio: float = 0.1,
                max_retention_ratio: float = 0.9) -> Tuple[torch.Tensor, Dict]:
        """
        Perform adaptive visual token pruning.

        Args:
            image_tokens: (batch, num_tokens, hidden_size)
            query_embedding: (batch, hidden_size), optional
            min_retention_ratio: Minimum tokens to keep
            max_retention_ratio: Maximum tokens to keep

        Returns:
            (pruned_tokens, pruning_stats)
        """
        batch_size, num_tokens, _ = image_tokens.shape

        # Compute importance scores
        importance = self.token_scorer(image_tokens)

        # Add query awareness if available
        if query_embedding is not None and self.query_scorer is not None:
            query_importance = self.query_scorer(image_tokens, query_embedding)
            importance = 0.6 * importance + 0.4 * query_importance

        # Analyze complexity
        complexity = self.complexity_analyzer.compute_complexity(image_tokens)

        # Determine target retention ratio
        target_ratio = self.complexity_analyzer.determine_target_ratio(
            complexity,
            min_ratio=min_retention_ratio,
            max_ratio=max_retention_ratio
        )

        # Select tokens
        pruned_tokens = []
        num_kept_list = []

        for b in range(batch_size):
            num_keep = max(1, int(num_tokens * target_ratio[b].item()))
            num_kept_list.append(num_keep)

            # Select top-k
            topk_scores, topk_indices = torch.topk(
                importance[b],
                k=num_keep,
                sorted=True
            )

            selected = image_tokens[b][topk_indices]
            pruned_tokens.append(selected)

        # Pad to max length
        max_kept = max(num_kept_list)
        padded = torch.zeros(
            batch_size, max_kept, self.hidden_size,
            device=image_tokens.device,
            dtype=image_tokens.dtype
        )

        for b, tokens in enumerate(pruned_tokens):
            padded[b, :tokens.shape[0]] = tokens

        # Compute statistics
        stats = {
            'original_tokens': num_tokens,
            'retained_tokens': max_kept,
            'compression_ratio': max_kept / num_tokens,
            'pruning_ratio': 1 - (max_kept / num_tokens),
            'complexity': complexity.mean().item(),
            'target_ratio': target_ratio.mean().item(),
            'importance_scores': importance
        }

        return padded, stats
```

This is the main pruning module combining all components.

**Step 5: Enable superior fine-tuning**

Leverage pruned tokens for improved learning:

```python
class PruneAwareFineTuner:
    """Fine-tuning that benefits from cleaner pruned tokens"""

    def __init__(self, model, pruner: GlimpsePrune):
        self.model = model
        self.pruner = pruner

    def finetune_step(self, images: torch.Tensor,
                     questions: torch.Tensor,
                     answers: torch.Tensor) -> float:
        """
        Fine-tuning step using dynamically pruned tokens.

        Args:
            images: Batch of images
            questions: Questions about images
            answers: Ground truth answers

        Returns:
            Loss value
        """
        # Encode image and question
        image_tokens = self.model.vision_encoder(images)
        query_embedding = self.model.text_encoder(questions)

        # Apply dynamic pruning
        pruned_tokens, stats = self.pruner(
            image_tokens,
            query_embedding=query_embedding,
            min_retention_ratio=0.1,
            max_retention_ratio=0.9
        )

        # Forward through language model
        logits = self.model(pruned_tokens, questions)

        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, answers)

        # Backward pass
        loss.backward()

        return loss.item()

    def full_finetune(self, dataloader, num_epochs: int = 3,
                     learning_rate: float = 1e-5):
        """Full fine-tuning with pruned tokens"""

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch in dataloader:
                images = batch['images']
                questions = batch['questions']
                answers = batch['answers']

                loss = self.finetune_step(images, questions, answers)

                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}: {avg_loss:.4f}")
```

This enables fine-tuning that benefits from token pruning.

### Practical Guidance

**When to use GlimpsePrune:**
- Deploying VLMs with computational constraints
- Batch inference where efficiency matters
- Fine-tuning VLMs where token quality improves learning
- Variable image complexity scenarios
- When model quality preservation is critical

**When NOT to use GlimpsePrune:**
- Systems with abundant computational resources
- Specialized visual tasks requiring all details (e.g., small object detection)
- Real-time systems where pruning overhead matters
- Already-optimized models without efficiency bottleneck

**Key hyperparameters:**

- `min_retention_ratio`: 0.05-0.15 typical for aggressive pruning
- `max_retention_ratio`: 0.5-0.9 typical for preserving complex scenes
- Complexity method: entropy generally best, variance fastest
- Query awareness weight: 0.3-0.5 balances content vs query

**Expected performance:**

- Token reduction: 85-92% typical
- Inference speedup: ~2-3x throughput improvement
- Quality preservation: >95% of baseline on most tasks
- Fine-tuning improvement: 5-15% performance gain observed

**Recommended configurations:**

- Simple images: 10% retention ratio
- Complex scenes: 40-50% retention
- Document images: 30% retention
- Portrait/simple objects: 15% retention

### Reference

A Glimpse to Compress: Dynamic Visual Token Pruning for Large Vision-Language Models. arXiv:2508.01548
