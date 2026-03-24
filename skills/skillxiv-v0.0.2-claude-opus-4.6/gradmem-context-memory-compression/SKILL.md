---
name: gradmem-context-memory-compression
title: "GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.13875"
keywords: [Memory Compression, Context Storage, Gradient Descent, Long-Context, Test-Time Optimization]
description: "Compress long context into compact memory tokens via iterative gradient descent. Learn to write information into prefix memory without storing full KV-caches, enabling efficient long-context reasoning and retrieval."
---

# GradMem: Test-Time Gradient Descent for Context Memory Compression

Large language models struggle with long contexts due to massive KV-cache overhead. GradMem enables models to compress context into compact memory tokens via test-time gradient descent. Rather than storing full cache for each query, the model learns to write information into a small set of learnable prefix tokens by optimizing a self-supervised reconstruction loss. This approach is particularly effective for context removal scenarios where the model must answer questions without accessing the original long context at inference time.

The key innovation is using gradient updates on memory tokens (with frozen model weights) rather than forward-only writes, enabling iterative error correction and much better context compression.

## Core Concept

GradMem operates through three phases:

1. **Context Ingestion** — Process long context once to initialize memory tokens
2. **Gradient-Based Writes** — Iteratively update memory tokens using gradient descent on reconstruction loss
3. **Query Answering** — Use compressed memory (without original context) to answer questions

This contrasts with forward-only approaches which write information in a single forward pass. By optimizing iteratively, GradMem can pack more information into the same number of tokens.

## Architecture Overview

- **Memory Token Initializer** — Create learnable prefix tokens initialized from context
- **Reconstruction Loss** — Self-supervised objective: can we reconstruct context from memory?
- **Gradient-Based Writer** — Optimize memory tokens via SGD on reconstruction loss
- **Query Encoder** — Generate question embeddings in the same space as memory
- **Memory-Only Reader** — Answer questions using only memory tokens (no original context)
- **Scaling Mechanism** — Multiple gradient steps improve capacity linearly

## Implementation Steps

Start by designing the memory tokens and initializing them from context.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

class ContextMemoryCompressor:
    """Compress long context into learnable memory tokens."""

    def __init__(self, hidden_dim=768, num_memory_tokens=32, context_seq_len=8192):
        self.hidden_dim = hidden_dim
        self.num_memory_tokens = num_memory_tokens
        self.context_seq_len = context_seq_len

        # Learnable memory tokens
        self.memory_tokens = nn.Parameter(
            torch.randn(num_memory_tokens, hidden_dim) * 0.01
        )

        # Projections for context encoding
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8,
                                      dim_feedforward=2048,
                                      batch_first=True),
            num_layers=2
        )

        # Reconstruction decoder
        self.context_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8,
                                      dim_feedforward=2048,
                                      batch_first=True),
            num_layers=2
        )

    def initialize_memory(self, context_embeddings: torch.Tensor):
        """Initialize memory tokens from context."""
        # context_embeddings: [batch_size, context_len, hidden_dim]
        batch_size = context_embeddings.size(0)

        # Summarize context into memory tokens via attention
        memory = self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Attention between memory and context
        cross_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8,
                                          batch_first=True)
        memory_initialized, _ = cross_attn(memory, context_embeddings,
                                          context_embeddings)

        return memory_initialized

    def reconstruction_loss(self, memory: torch.Tensor,
                           context: torch.Tensor) -> torch.Tensor:
        """Loss for reconstructing context from memory."""
        # Reconstruct context using memory as queries and context as key/value
        reconstructed = self.context_decoder(memory, context)

        # MSE loss between original and reconstructed
        loss = F.mse_loss(reconstructed, context)
        return loss

    def write_to_memory_gradient_based(self, context_embeddings: torch.Tensor,
                                       num_gradient_steps=5,
                                       learning_rate=0.1) -> torch.Tensor:
        """Iteratively optimize memory tokens via gradient descent."""
        batch_size = context_embeddings.size(0)

        # Initialize memory
        memory = self.initialize_memory(context_embeddings)

        # Create optimizer for memory only (model weights frozen)
        memory_param = torch.nn.Parameter(memory.clone())
        optimizer = SGD([memory_param], lr=learning_rate)

        # Gradient descent steps
        for step in range(num_gradient_steps):
            optimizer.zero_grad()

            # Compute reconstruction loss
            loss = self.reconstruction_loss(memory_param, context_embeddings)

            # Backward pass
            loss.backward()

            # Gradient step
            optimizer.step()

            print(f"  Gradient step {step+1}: Loss = {loss.item():.4f}")

        return memory_param.detach()

    def forward_write(self, context_embeddings: torch.Tensor) -> torch.Tensor:
        """Baseline: single forward pass to write context (for comparison)."""
        # Simple pooling of context
        memory = context_embeddings.mean(dim=1, keepdim=True).expand(
            -1, self.num_memory_tokens, -1
        )
        return memory
```

Now implement the query answering mechanism that uses only memory without original context.

```python
class MemoryOnlyReader(nn.Module):
    """Answer questions using only memory tokens."""

    def __init__(self, hidden_dim=768, vocab_size=50257):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Transformer decoder (query-conditioned)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8,
                                      dim_feedforward=2048,
                                      batch_first=True),
            num_layers=3
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_embeddings: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embeddings: [batch_size, query_len, hidden_dim]
            memory: [batch_size, num_memory_tokens, hidden_dim]
        Returns:
            logits: [batch_size, query_len, vocab_size]
        """
        # Use memory as context (key/value), query as decoder input
        output = self.decoder(query_embeddings, memory)

        # Project to vocabulary
        logits = self.output_proj(output)
        return logits
```

Integrate gradient-based writing into full pipeline with benchmarking.

```python
class GradMemModel(nn.Module):
    """Full GradMem system for context compression and QA."""

    def __init__(self, base_model, hidden_dim=768, num_memory_tokens=32):
        super().__init__()
        self.base_model = base_model
        self.compressor = ContextMemoryCompressor(hidden_dim, num_memory_tokens)
        self.memory_reader = MemoryOnlyReader(hidden_dim)

    def process_context(self, context_ids: torch.Tensor,
                       num_gradient_steps=5) -> torch.Tensor:
        """Compress context into memory tokens."""
        # Encode context
        context_embeddings = self.base_model.encoder(context_ids)

        # Gradient-based memory writing
        memory = self.compressor.write_to_memory_gradient_based(
            context_embeddings,
            num_gradient_steps=num_gradient_steps
        )

        return memory

    def answer_question(self, query_ids: torch.Tensor,
                       memory: torch.Tensor) -> torch.Tensor:
        """Answer question using memory (without original context)."""
        # Encode query
        query_embeddings = self.base_model.encoder(query_ids)

        # Generate answer
        logits = self.memory_reader(query_embeddings, memory)

        return logits

    def forward(self, context_ids: torch.Tensor, query_ids: torch.Tensor,
                num_gradient_steps=5) -> torch.Tensor:
        """Full pipeline: compress context, then answer query."""
        # Compress context into memory
        memory = self.process_context(context_ids, num_gradient_steps)

        # Answer question using memory
        logits = self.answer_question(query_ids, memory)

        return logits


def benchmark_memory_compression(model, test_cases, baselines=['forward_only']):
    """Measure compression efficiency vs accuracy."""
    results = []

    for num_steps in [1, 3, 5, 10]:
        accuracies = []
        reconstruction_errors = []

        for case in test_cases:
            context = case['context']
            query = case['query']
            reference_answer = case['answer']

            # Compress context
            context_ids = tokenizer.encode(context)
            context_embeddings = model.base_model.encoder(
                torch.tensor([context_ids]))

            # Write to memory (gradient-based)
            memory = model.compressor.write_to_memory_gradient_based(
                context_embeddings,
                num_gradient_steps=num_steps
            )

            # Answer question
            query_ids = tokenizer.encode(query)
            logits = model.answer_question(torch.tensor([query_ids]), memory)

            # Decode answer
            predicted_answer = tokenizer.decode(torch.argmax(logits, dim=-1))

            # Evaluate
            accuracy = compute_em(predicted_answer, reference_answer)
            accuracies.append(accuracy)

            # Reconstruction error
            reconstructed = model.compressor.context_decoder(memory,
                                                            context_embeddings)
            recon_error = F.mse_loss(reconstructed,
                                    context_embeddings).item()
            reconstruction_errors.append(recon_error)

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_recon_error = sum(reconstruction_errors) / len(reconstruction_errors)

        results.append({
            'num_gradient_steps': num_steps,
            'accuracy': avg_accuracy,
            'reconstruction_error': avg_recon_error
        })

        print(f"Gradient steps {num_steps}: Accuracy={avg_accuracy:.1%}, "
              f"Recon Error={avg_recon_error:.4f}")

    return results
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Number of memory tokens typically 16-64; more tokens improve capacity but increase overhead
- Gradient steps 3-10 work well; each step scales capacity roughly linearly
- Learning rate 0.01-0.5; lower values converge more slowly, higher values may diverge
- Use when context is fixed across multiple questions (e.g., document QA, retrieval)
- Particularly effective for associative retrieval and structured reasoning tasks

**When NOT to use:**
- For streaming scenarios where context changes frequently (memory becomes stale)
- When exact context retrieval is needed (compression loses information)
- For very short contexts where compression overhead exceeds benefits
- When latency is critical; gradient descent adds test-time cost

**Common Pitfalls:**
- Memory tokens becoming misaligned with query encoder; use shared embeddings
- Reconstruction loss not capturing important information; weight loss by query relevance
- Gradient descent diverging; use gradient clipping and careful learning rate selection
- Not accounting for information loss in compression; use reconstruction error as proxy for quality

## Reference

Paper: [GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent](https://arxiv.org/abs/2603.13875)
