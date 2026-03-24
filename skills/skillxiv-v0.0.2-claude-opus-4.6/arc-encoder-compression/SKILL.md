---
name: arc-encoder-compression
title: "ARC-Encoder: learning compressed text representations for LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.20535"
keywords: [Compression, Efficiency, Inference, Encoder, Embeddings]
description: "Reduces inference cost by compressing context into continuous representations using a separate encoder. Generates 4-8x fewer representations than token embeddings while maintaining model performance. Works with any decoder LLM without modification or fine-tuning."
---

# ARC-Encoder: Efficient Text Compression for LLM Inference

Long contexts increase inference latency and memory. ARC-Encoder compresses text into dense continuous representations that substitute token embeddings, reducing sequence length without sacrificing language model performance.

The encoder generalizes across decoder models, enabling a single compression component to work with multiple LLMs.

## Core Concept

Key innovation: **replace token-level embeddings with compressed continuous representations**:
- Separate encoder compresses context into continuous embeddings
- Outputs 4-8x fewer representations than input tokens
- Maintains semantic information despite aggressive compression
- Compatible with any pretrained decoder without architecture changes

This decouples compression from language modeling, enabling reuse across models.

## Architecture Overview

- Text encoder (e.g., BERT-style or custom architecture)
- Compression ratio design (typically 4x or 8x)
- Continuous representation substitution in decoder embeddings
- Optional dimension reduction for efficiency

## Implementation Steps

Build an encoder that compresses arbitrary text into dense fixed-size representations. The encoder should preserve semantic content while reducing cardinality:

```python
class TextCompressionEncoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, compression_ratio=4):
        super().__init__()
        self.compression_ratio = compression_ratio

        # Encoder: standard transformer blocks
        self.encoder = TransformerEncoder(
            d_model=input_dim,
            num_layers=4,
            num_heads=12
        )

        # Compression projection
        self.compress_proj = nn.Linear(input_dim * compression_ratio, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, token_embeddings):
        """Compress token embeddings to continuous representations."""
        # Encode input
        encoded = self.encoder(token_embeddings)

        # Compress: chunk embeddings and project
        seq_len = encoded.shape[1]
        # Group tokens: compress every N tokens to 1 output
        grouped = []
        for i in range(0, seq_len, self.compression_ratio):
            chunk = encoded[:, i:i+self.compression_ratio, :]
            # Flatten chunk
            chunk_flat = chunk.reshape(chunk.shape[0], -1)
            compressed = self.compress_proj(chunk_flat)
            grouped.append(compressed)

        # Stack compressed representations
        output = torch.stack(grouped, dim=1)
        output = self.norm(output)

        return output
```

Integrate the encoder into the decoder by replacing embedding layers. The decoder uses compressed representations as input:

```python
class CompressedLM(nn.Module):
    def __init__(self, base_lm, encoder, compression_ratio=4):
        super().__init__()
        self.base_lm = base_lm
        self.encoder = encoder
        self.compression_ratio = compression_ratio

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Convert IDs to token embeddings in base LM
        token_embeddings = self.base_lm.embeddings(input_ids)

        # Compress embeddings
        compressed = self.encoder(token_embeddings)

        # Forward through base LM with compressed representations
        # Adjust attention mask for compressed sequence
        compressed_mask = self._compress_mask(attention_mask)

        output = self.base_lm.decoder(
            compressed,
            attention_mask=compressed_mask,
            **kwargs
        )

        return output

    def _compress_mask(self, mask):
        """Adjust attention mask for compressed sequence length."""
        if mask is None:
            return None

        # Compress mask: keep first position of each chunk
        seq_len = mask.shape[-1]
        compressed_len = (seq_len + self.compression_ratio - 1) // self.compression_ratio

        compressed_mask = torch.zeros(
            mask.shape[0], compressed_len,
            device=mask.device, dtype=mask.dtype
        )

        for i in range(compressed_len):
            start_idx = i * self.compression_ratio
            end_idx = min(start_idx + self.compression_ratio, seq_len)
            # Mark as valid if any token in chunk is valid
            compressed_mask[:, i] = mask[:, start_idx:end_idx].any(dim=1).float()

        return compressed_mask
```

Train the encoder on contrastive or reconstruction objectives without modifying the base LM:

```python
def train_encoder(encoder, decoder, training_corpus, compression_ratio=4, epochs=10):
    """Train encoder using reconstruction loss."""
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        for batch in training_corpus:
            input_ids, labels = batch

            # Encode-decode cycle
            embeddings = decoder.embeddings(input_ids)
            compressed = encoder(embeddings)

            # Upsample compressed back to original dimension (for loss)
            reconstructed = upsample(compressed, embedding_dim=embeddings.shape[-1])

            # Reconstruction loss
            loss = loss_fn(reconstructed, embeddings.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Compression ratio | 4x or 8x (4x balances speed vs. quality) |
| Encoder architecture | 4-6 transformer layers, 12 heads |
| Training data | 10M+ tokens from target domain |
| Dimension match | Typically match decoder embedding dim (768-2048) |

**When to use:**
- Inference with very long contexts (4K+ tokens)
- Cost-sensitive deployments with latency budgets
- Scenarios where model reuse across decoders is needed
- Context window extension for fixed-size models

**When NOT to use:**
- Short sequences where overhead outweighs gains
- High-precision tasks requiring exact token representations
- When you can fine-tune the decoder (adapter might be simpler)

**Common pitfalls:**
- Compression ratio too aggressive (information loss)
- Encoder trained on different domain than inference
- Insufficient encoder capacity (underfitting compression task)
- Not validating on downstream tasks (reconstruction loss ≠ task performance)

Reference: [ARC-Encoder on arXiv](https://arxiv.org/abs/2510.20535)
