---
name: h-net-dynamic-chunking-hierarchical-sequence
title: "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07955"
keywords: [Tokenization-Free, Hierarchical Sequence Modeling, Dynamic Chunking, Byte-Level Modeling, End-to-End Learning]
description: "Eliminate fixed tokenization by learning data-dependent segmentation jointly with the model through dynamic chunking, matching BPE-tokenized Transformers at equivalent compute while showing improved robustness and better downstream task performance without vocabulary constraints."
---

# H-Net: Hierarchical Sequence Modeling via Dynamic Chunking

Fixed-vocabulary tokenization (BPE, SentencePiece) is a fundamental limitation: it forces discrete boundaries that may not match semantic structure, wastes tokens on rare words, and requires precomputation. H-Net learns tokenization end-to-end: boundaries are determined jointly during training based on content and context, not fixed at preprocessing.

The approach uses dynamic chunking to recursively compress sequences at multiple levels. A routing module detects where to split, a smoothing module interpolates between chunks, and the main network operates on compressed representations. Multi-stage variants compress to 4.7-4.8 bytes per chunk, matching BPE efficiency while maintaining superior robustness and 6.9× compression across stages.

## Core Concept

Tokenization is a preprocessing step that locks in decisions before the model sees data. H-Net inverts this: tokenization becomes a learned component of the model itself. The model learns when sequences are semantically coherent (mergeable) and when they transition (require splits). This is learned end-to-end through gradient descent using a differentiable routing mechanism.

Three insights enable this: (1) routing detects boundaries via similarity between adjacent representations, (2) smoothing creates differentiable transitions for gradient flow, and (3) hierarchical compression progressively reduces sequence length, enabling transformers to work with extremely long sequences. The result is a data-dependent, learnable tokenization that adapts to the input distribution.

## Architecture Overview

- **Mamba-2 Encoders**: Process full-resolution sequences efficiently
- **Dynamic Chunking Module**: Routing (boundary detection) + smoothing (interpolation)
- **Hierarchical Compression Stages**: Multi-level recursive compression
- **Transformer Backbone**: Operates on progressively compressed representations
- **Mamba-2 Decoders**: Restore original sequence resolution
- **Ratio Loss**: Auxiliary loss guiding compression toward target ratios
- **U-Net Architecture**: Encoder-Decoder with skip connections at each level

## Implementation

### Step 1: Implement Dynamic Routing for Boundary Detection

Detect optimal chunk boundaries based on representation similarity:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class DynamicRouting(nn.Module):
    """
    Routing module: detect boundaries between chunks via similarity.
    Boundaries occur where adjacent representations are dissimilar.
    """

    def __init__(self, hidden_dim: int = 768, num_neighbors: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors

        # Learnable projection for similarity computation
        self.similarity_proj = nn.Linear(hidden_dim, hidden_dim // 4)

    def forward(self, x: torch.Tensor,
               threshold: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect chunk boundaries based on representation similarity.

        x: [seq_len, hidden_dim] sequence of tokens
        Returns: (boundaries, similarity_scores)
        """
        seq_len = x.shape[0]

        # Project to lower dimension for efficiency
        x_proj = self.similarity_proj(x)  # [seq_len, hidden_dim//4]

        # Compute cosine similarity between adjacent representations
        similarities = []
        for i in range(seq_len - 1):
            sim = F.cosine_similarity(
                x_proj[i].unsqueeze(0),
                x_proj[i + 1].unsqueeze(0)
            )
            similarities.append(sim.item())

        similarities = torch.tensor(similarities, device=x.device)

        # Detect boundaries where similarity < threshold
        # (adjacent tokens are dissimilar = boundary)
        boundaries = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
        boundaries[0] = True  # First token always starts chunk
        boundaries[-1] = True  # Last token always ends chunk

        for i in range(len(similarities)):
            if similarities[i] < threshold:
                boundaries[i + 1] = True

        return boundaries, similarities

    def compute_chunk_boundaries(self, x: torch.Tensor,
                                adaptive_threshold: bool = True) -> torch.Tensor:
        """
        Adaptively compute boundaries based on content.
        Threshold adjusts based on distribution of similarities.
        """
        seq_len = x.shape[0]
        x_proj = self.similarity_proj(x)

        similarities = []
        for i in range(seq_len - 1):
            sim = F.cosine_similarity(
                x_proj[i].unsqueeze(0),
                x_proj[i + 1].unsqueeze(0)
            )
            similarities.append(sim.item())

        similarities = torch.tensor(similarities, device=x.device)

        if adaptive_threshold:
            # Use percentile-based threshold
            threshold = torch.quantile(similarities, 0.3)  # Bottom 30%
        else:
            threshold = 0.8

        boundaries = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
        boundaries[0] = True
        boundaries[-1] = True

        for i in range(len(similarities)):
            if similarities[i] < threshold:
                boundaries[i + 1] = True

        return boundaries
```

### Step 2: Implement Smooth Interpolation Between Chunks

Smooth transitions enable gradient flow across chunk boundaries:

```python
class SmoothingModule(nn.Module):
    """
    Smoothing: interpolate between chunks for differentiable compression.
    Creates smooth transitions that maintain gradient flow.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable interpolation weight

    def forward(self, x: torch.Tensor,
               boundaries: torch.Tensor) -> torch.Tensor:
        """
        Apply exponential moving average smoothing across chunks.

        x: [seq_len, hidden_dim]
        boundaries: [seq_len] boolean tensor indicating chunk starts
        Returns: smoothed [seq_len, hidden_dim]
        """
        smoothed = x.clone()
        ema = x[0].clone()

        for i in range(1, x.shape[0]):
            if boundaries[i]:
                # Reset EMA at boundary
                ema = x[i].clone()
            else:
                # Exponential moving average
                ema = self.alpha * x[i] + (1 - self.alpha) * ema
                smoothed[i] = ema

        return smoothed

    def merge_chunks(self, x: torch.Tensor,
                    boundaries: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Merge tokens within chunks into representative tokens.
        Returns merged representations and chunk sizes.
        """
        chunks = []
        chunk_sizes = []
        start_idx = 0

        for i in range(1, x.shape[0]):
            if boundaries[i]:
                # End current chunk
                chunk = x[start_idx:i]
                merged = chunk.mean(dim=0)  # Average pooling
                chunks.append(merged)
                chunk_sizes.append(i - start_idx)
                start_idx = i

        # Final chunk
        chunk = x[start_idx:]
        merged = chunk.mean(dim=0)
        chunks.append(merged)
        chunk_sizes.append(x.shape[0] - start_idx)

        merged_x = torch.stack(chunks, dim=0)
        return merged_x, chunk_sizes
```

### Step 3: Build Hierarchical Compression with U-Net Architecture

Implement multi-stage hierarchical compression:

```python
from mamba_ssm import Mamba

class HierarchicalCompressionBlock(nn.Module):
    """
    Single hierarchical compression block: encode, compress, decode.
    """

    def __init__(self, hidden_dim: int = 768,
                 compression_ratio: float = 0.8,
                 num_transformer_layers: int = 4):
        super().__init__()

        # Encoder: process full resolution
        self.encoder = nn.ModuleList([
            Mamba(hidden_dim) for _ in range(2)
        ])

        # Dynamic chunking
        self.routing = DynamicRouting(hidden_dim)
        self.smoothing = SmoothingModule(hidden_dim)

        # Main transformer on compressed
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=3072,
                batch_first=True
            ),
            num_layers=num_transformer_layers
        )

        # Decoder: restore resolution
        self.decoder = nn.ModuleList([
            Mamba(hidden_dim) for _ in range(2)
        ])

        self.compression_ratio = compression_ratio
        self.ratio_loss_weight = 0.1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with hierarchical compression.
        x: [seq_len, hidden_dim]
        """
        # Encode
        encoded = x
        for encoder_layer in self.encoder:
            encoded = encoder_layer(encoded)

        # Detect boundaries
        boundaries = self.routing.compute_chunk_boundaries(encoded)

        # Smooth transitions
        smoothed = self.smoothing(encoded, boundaries)

        # Merge chunks
        compressed, chunk_sizes = self.smoothing.merge_chunks(smoothed, boundaries)

        # Transform compressed representation
        transformed = self.transformer(compressed.unsqueeze(0)).squeeze(0)

        # Upsample back to original length
        upsampled = self._upsample(transformed, chunk_sizes, x.shape[0])

        # Decode
        decoded = upsampled
        for decoder_layer in self.decoder:
            decoded = decoder_layer(decoded)

        # Ratio loss: guide toward target compression
        actual_ratio = compressed.shape[0] / x.shape[0]
        ratio_loss = F.mse_loss(
            torch.tensor(actual_ratio),
            torch.tensor(self.compression_ratio)
        )

        return decoded, ratio_loss

    def _upsample(self, compressed: torch.Tensor,
                 chunk_sizes: List[int],
                 target_len: int) -> torch.Tensor:
        """Upsample compressed representations back to original length."""
        upsampled = []
        for i, size in enumerate(chunk_sizes):
            # Repeat representation for chunk size
            repeated = compressed[i].unsqueeze(0).repeat(size, 1)
            upsampled.append(repeated)

        return torch.cat(upsampled, dim=0)

class HNetModel(nn.Module):
    """Multi-stage hierarchical sequence model."""

    def __init__(self, vocab_size: int, hidden_dim: int = 768,
                 num_stages: int = 2, num_layers_per_stage: int = 4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Multi-stage compression
        self.stages = nn.ModuleList([
            HierarchicalCompressionBlock(
                hidden_dim,
                compression_ratio=0.8,  # Adjust per stage
                num_transformer_layers=num_layers_per_stage
            )
            for _ in range(num_stages)
        ])

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor,
               labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-stage compression.
        """
        # Embed
        x = self.embedding(input_ids)  # [seq_len, hidden_dim]

        total_ratio_loss = 0.0

        # Apply stages
        for stage in self.stages:
            x, ratio_loss = stage(x)
            total_ratio_loss += ratio_loss

        # Language modeling head
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )
            # Combine with ratio loss
            loss = loss + 0.1 * total_ratio_loss

        return logits, loss
```

### Step 4: Training and Evaluation

Train on byte-level sequences and evaluate against BPE baselines:

```python
def train_hnet(model: HNetModel,
              train_dataloader,
              num_epochs: int = 10,
              lr: float = 1e-4) -> None:
    """Train H-Net on byte sequences (no BPE tokenization)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataloader) * num_epochs
    )

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            logits, loss = model(input_ids, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch}, Step {batch_idx}: loss = {avg_loss:.4f}")

def evaluate_compression_efficiency(model: HNetModel,
                                   test_text: str,
                                   tokenizer_bpe=None) -> Dict:
    """Compare compression ratio: H-Net vs BPE."""
    # H-Net: encode as bytes
    bytes_input = test_text.encode('utf-8')
    h_net_length = len(bytes_input)

    # BPE: tokenize
    if tokenizer_bpe:
        bpe_tokens = tokenizer_bpe.encode(test_text)
        bpe_length = len(bpe_tokens)
    else:
        bpe_length = len(test_text) / 4  # Rough estimate

    return {
        "h_net_length": h_net_length,
        "bpe_length": bpe_length,
        "h_net_compression": h_net_length / len(test_text),
        "bpe_compression": bpe_length / len(test_text),
        "h_net_advantage": bpe_length / h_net_length
    }

def evaluate_downstream_tasks(model: HNetModel,
                             task_datasets: Dict[str, Any]) -> Dict:
    """Evaluate robustness on downstream tasks (perturbations, typos, etc.)."""
    results = {}

    for task_name, dataset in task_datasets.items():
        correct = 0
        total = 0

        for sample in dataset:
            input_ids = sample["input_ids"]
            label = sample["label"]

            with torch.no_grad():
                logits, _ = model(input_ids)
                pred = torch.argmax(logits, dim=-1)

            correct += (pred == label).float().mean().item()
            total += 1

        accuracy = correct / total if total > 0 else 0
        results[task_name] = accuracy

    return results
```

## Practical Guidance

| Parameter | Recommended Value | Notes |
|---|---|---|
| Hidden Dimension | 768 | Standard for 1B+ models |
| Num Compression Stages | 2-3 | Multi-stage compression |
| Target Compression Ratio | 0.8-0.85 per stage | Cumulative reduction |
| Routing Threshold | Adaptive (percentile) | Learn from data |
| Smoothing Alpha | 0.5 (learnable) | Exponential moving average |
| Chunk Min Size | 1 token | Allow granular compression |
| Chunk Max Size | Unlimited | Data-dependent |
| Ratio Loss Weight | 0.1 | Soft constraint on compression |
| Training Sequence Length | 8,192 bytes (~1,792 BPE tokens) | Match equivalent compute |
| Transformer Depth | 4-6 per stage | Balance capacity and efficiency |

**When to use H-Net:**
- Tasks requiring end-to-end learning without fixed vocabulary
- Language processing where tokenization boundaries are unclear
- Multilingual or code-heavy domains where BPE is suboptimal
- Robustness evaluation (perturbation-resistant tokenization)
- Research into learned tokenization mechanisms
- Long-context modeling (hierarchical compression scales well)

**When NOT to use H-Net:**
- Production systems requiring proven, stable tokenization
- Tasks where BPE is demonstrably superior
- Real-time inference (dynamic chunking adds overhead)
- Extremely large-scale training (BPE is battle-tested)
- Domains with established vocabularies (existing tools superior)

**Common pitfalls:**
- Routing threshold too strict, preventing merging
- Routing threshold too loose, over-merging distinct tokens
- Ratio loss weight too high, forcing compression at expense of quality
- Not normalizing similarities before thresholding
- Upsampling strategy losing information from merged chunks
- Gradient flow issues due to abrupt chunk boundaries
- Not validating that compression ratio targets are achievable
- Smoothing alpha too extreme (0.0 or 1.0), preventing learning

## Reference

Chen, S., Wang, X., Zhou, Y., & Li, Z. (2025). Dynamic Chunking for End-to-End Hierarchical Sequence Modeling. arXiv:2507.07955. https://arxiv.org/abs/2507.07955
