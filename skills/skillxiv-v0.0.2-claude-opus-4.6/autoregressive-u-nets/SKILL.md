---
name: autoregressive-u-nets
title: "From Bytes to Ideas: Language Modeling with Autoregressive U-Nets"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.14761"
keywords: [language-modeling, U-Nets, byte-level, multi-scale, tokenization-free]
description: "Autoregressive U-Net operating directly on raw bytes with hierarchical multi-scale pooling for adaptive token embedding, eliminating fixed vocabularies."
---

# From Bytes to Ideas: Language Modeling with Autoregressive U-Nets

## Core Concept

This work introduces an autoregressive U-Net architecture that operates directly on raw bytes rather than fixed tokenization schemes. The model learns to embed tokens hierarchically as it trains, pooling bytes into progressively larger units (words, word pairs, up to 4-word chunks). Multi-scale stages predict at different granularities: deeper stages handle semantics over longer spans while shallower stages preserve fine details. The approach avoids predefined vocabularies and embedding tables, demonstrating comparable performance to BPE baselines with promising scaling trends for deeper hierarchies.

## Architecture Overview

- **Contracting Path**: Compresses byte sequences through successive pooling stages (bytes → words → phrases → larger units)
- **Expanding Path**: Reconstructs sequences using skip connections from contracting path
- **Multi-Scale Prediction**: Different depths predict at corresponding granularities (shallow=bytes, deep=semantics)
- **Hierarchical Embedding**: Learns token representations bottom-up rather than using fixed vocabulary
- **Split Functions**: User-specified rules define pooling boundaries (word breaks, punctuation, etc.)

## Implementation

### Step 1: Design Split Functions

Define how to pool bytes into progressively larger units:

```python
import re
from typing import List, Callable

class SplitFunction:
    """
    Defines how to pool bytes into tokens at different levels.
    """
    @staticmethod
    def byte_level(text: str) -> List[int]:
        """
        Level 0: Individual bytes (characters)
        """
        return list(text.encode('utf-8'))

    @staticmethod
    def word_level(text: str) -> List[str]:
        """
        Level 1: Words (split by whitespace and punctuation)
        """
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    @staticmethod
    def bigram_level(text: str) -> List[str]:
        """
        Level 2: Word pairs (bigrams)
        """
        words = text.split()
        bigrams = [' '.join(words[i:i+2])
                   for i in range(len(words) - 1)]
        return bigrams

    @staticmethod
    def phrase_level(text: str) -> List[str]:
        """
        Level 3: Phrases (4-word chunks)
        """
        words = text.split()
        phrases = [' '.join(words[i:i+4])
                   for i in range(0, len(words), 4)]
        return phrases

    @staticmethod
    def get_split_fn(level: int) -> Callable:
        """Get split function by level"""
        funcs = {
            0: SplitFunction.byte_level,
            1: SplitFunction.word_level,
            2: SplitFunction.bigram_level,
            3: SplitFunction.phrase_level
        }
        return funcs.get(level, SplitFunction.byte_level)
```

### Step 2: Implement Contracting Path

Compress sequences through hierarchical pooling:

```python
import torch
import torch.nn as nn

class ContractionBlock(nn.Module):
    """
    Contracts sequence to next pooling level.
    Learns to aggregate tokens from finer level.
    """
    def __init__(self, input_dim, output_dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size

        # Linear transformation before pooling
        self.transform = nn.Linear(input_dim, output_dim)

        # Pooling (max pooling on sequence dimension)
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, seq_len // pool_size, output_dim]
        """
        batch, seq_len, dim = x.shape

        # Transform
        transformed = self.transform(x)  # [batch, seq_len, output_dim]

        # Permute for pooling on sequence dimension
        transformed = transformed.permute(0, 2, 1)  # [batch, output_dim, seq_len]

        # Pool
        pooled = self.pool(transformed)  # [batch, output_dim, seq_len // pool_size]

        # Permute back
        pooled = pooled.permute(0, 2, 1)  # [batch, seq_len // pool_size, output_dim]

        return pooled


class ContractionPath(nn.Module):
    """
    Full contraction: bytes → words → phrases → larger units
    """
    def __init__(self, vocab_size=256, hidden_dim=256, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        # Initial embedding (bytes → vectors)
        self.byte_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Contraction blocks
        self.contractions = nn.ModuleList([
            ContractionBlock(hidden_dim, hidden_dim, pool_size=2)
            for _ in range(num_levels - 1)
        ])

    def forward(self, byte_input):
        """
        Args:
            byte_input: [batch, seq_len] byte IDs

        Returns:
            features: list of [batch, seq_len_i, hidden_dim] at each level
        """
        # Embed bytes
        x = self.byte_embedding(byte_input)  # [batch, seq_len, hidden_dim]

        features = [x]

        # Progressive contractions
        for contraction in self.contractions:
            x = contraction(x)
            features.append(x)

        return features
```

### Step 3: Implement Expanding Path

Reconstruct sequences using skip connections:

```python
class ExpansionBlock(nn.Module):
    """
    Expands sequence with information from skip connection.
    """
    def __init__(self, input_dim, skip_dim, output_dim):
        super().__init__()

        # Upsample to match skip connection size
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Combine with skip
        self.combine = nn.Linear(input_dim + skip_dim, output_dim)

    def forward(self, x, skip):
        """
        Args:
            x: [batch, seq_len, input_dim]
            skip: [batch, seq_len * 2, skip_dim] (finer level)

        Returns:
            [batch, seq_len * 2, output_dim]
        """
        # Upsample
        x_up = self.upsample(x.permute(0, 2, 1))
        x_up = x_up.permute(0, 2, 1)  # [batch, seq_len * 2, input_dim]

        # Concatenate skip
        combined = torch.cat([x_up, skip], dim=-1)

        # Linear combination
        output = self.combine(combined)

        return output


class ExpansionPath(nn.Module):
    """
    Full expansion: reconstruct from coarse to fine levels
    """
    def __init__(self, hidden_dim=256, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        # Expansion blocks (in reverse order)
        self.expansions = nn.ModuleList([
            ExpansionBlock(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_levels - 1)
        ])

    def forward(self, features):
        """
        Args:
            features: list of [batch, seq_len_i, hidden_dim]
                     (from finest to coarsest)

        Returns:
            outputs: list of reconstructed features
        """
        outputs = []

        # Start from coarsest level
        x = features[-1]

        for i in range(len(self.expansions)):
            # Get skip connection from finer level
            skip = features[-(i + 2)]

            # Expand and combine
            x = self.expansions[i](x, skip)
            outputs.append(x)

        # Reverse to get finest-to-coarsest order
        outputs.reverse()

        return outputs
```

### Step 4: Implement Multi-Scale Prediction Heads

Predict at each scale:

```python
class MultiScalePredictionHeads(nn.Module):
    """
    Prediction heads at each level for multi-scale learning.
    Deeper levels predict further ahead (semantic context).
    """
    def __init__(self, hidden_dim, vocab_size, num_levels):
        super().__init__()
        self.num_levels = num_levels
        self.vocab_size = vocab_size

        # One head per level
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(num_levels)
        ])

    def forward(self, features):
        """
        Args:
            features: list of [batch, seq_len_i, hidden_dim]

        Returns:
            logits_list: list of [batch, seq_len_i, vocab_size]
        """
        logits_list = []

        for i, feat in enumerate(features):
            logits = self.heads[i](feat)
            logits_list.append(logits)

        return logits_list

    def compute_loss(self, logits_list, targets_list, weights=None):
        """
        Compute weighted loss across scales.

        Args:
            logits_list: list of model predictions
            targets_list: list of ground truth (at each scale)
            weights: optional list of weights per level
        """
        if weights is None:
            weights = [1.0 / len(logits_list)] * len(logits_list)

        total_loss = 0

        for i, (logits, targets, weight) in enumerate(
            zip(logits_list, targets_list, weights)
        ):
            # Flatten
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = targets.reshape(-1)

            # Cross entropy
            loss = torch.nn.functional.cross_entropy(
                logits_flat, targets_flat, reduction='mean'
            )

            total_loss += weight * loss

        return total_loss
```

### Step 5: Complete U-Net Model

Integrate contraction, expansion, and prediction:

```python
class AutoregressiveUNet(nn.Module):
    """
    Full autoregressive U-Net for byte-level language modeling.
    """
    def __init__(self, vocab_size=256, hidden_dim=256, num_levels=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

        # U-Net components
        self.contraction = ContractionPath(vocab_size, hidden_dim, num_levels)
        self.expansion = ExpansionPath(hidden_dim, num_levels)
        self.heads = MultiScalePredictionHeads(hidden_dim, vocab_size, num_levels)

    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, seq_len] byte IDs

        Returns:
            logits: [batch, seq_len, vocab_size] predictions at finest level
            all_logits: list of predictions at all scales
        """
        # Contract
        contracted_features = self.contraction(input_ids)

        # Expand
        expanded_features = self.expansion(contracted_features)

        # Predict at all scales
        all_logits = self.heads(expanded_features)

        # Use finest scale (first in expanded_features) for main prediction
        logits = all_logits[0]

        return logits, all_logits

    def compute_loss(self, input_ids, targets_list, weights=None):
        """
        Compute multi-scale loss.

        Args:
            input_ids: [batch, seq_len]
            targets_list: list of targets at each scale
            weights: loss weights per scale
        """
        _, all_logits = self.forward(input_ids)

        loss = self.heads.compute_loss(all_logits, targets_list, weights)

        return loss

    def generate(self, prompt_ids, max_length=512, temperature=1.0):
        """
        Autoregressive generation using byte-level predictions.
        """
        device = prompt_ids.device
        generated = prompt_ids.clone()

        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                logits, _ = self.forward(generated)

            # Get next token probabilities (finest scale)
            next_logits = logits[:, -1, :] / temperature
            next_probs = torch.softmax(next_logits, dim=-1)

            # Sample
            next_token = torch.multinomial(next_probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

        return generated
```

### Step 6: Training Loop

Train with multi-scale supervision:

```python
def train_u_net(model, train_loader, num_epochs=10, device='cuda'):
    """
    Train autoregressive U-Net with multi-scale targets.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch_idx, (input_ids, targets_list) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets_list = [t.to(device) for t in targets_list]

            # Compute loss
            loss = model.compute_loss(input_ids, targets_list)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
```

## Practical Guidance

- **Hierarchy Depth**: Start with 2-3 levels; deeper hierarchies require more data but improve scaling
- **Hidden Dimension**: Match to model capacity; 256 reasonable for small models
- **Loss Weighting**: Start equal weights; can emphasize finer/coarser scales based on task
- **Split Functions**: Customize based on language/domain (word breaks for English, character clusters for CJK)
- **Throughput**: GPU memory similar to standard transformers; byte-level avoids embedding table overhead
- **Evaluation**: Compare perplexity against BPE baselines; analyze scaling laws with depth
- **Integration**: Can replace BPE tokenizer in any causal LM architecture

## Reference

Paper: arXiv:2506.14761
Key metrics: Matches BPE baselines; improved scaling with deeper hierarchies; eliminates vocab overhead
Architecture advantages: Infinite vocabulary, learned hierarchies, multi-scale features
Related work: Tokenization-free models, hierarchical representations, U-Net architectures
