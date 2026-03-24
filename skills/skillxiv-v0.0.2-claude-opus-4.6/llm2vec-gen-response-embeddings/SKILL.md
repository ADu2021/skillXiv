---
name: llm2vec-gen-response-embeddings
title: "LLM2Vec-Gen: Generative Embeddings from Large Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.10913"
keywords: [Embeddings, LLM, Generation, Response Encoding, Representation Learning]
description: "Learn to encode LLM-generated responses rather than raw inputs by training special tokens and projection layers while keeping the backbone frozen. Bridges the input-output gap to transfer LLM capabilities like reasoning directly into embedding space."
---

# Technique: Response-Centric LLM Embeddings for Downstream Tasks

The fundamental problem with standard LLM embeddings is the **input-output gap**: diverse queries map to similar outputs, making query embeddings poor proxies for semantic relevance. LLM2Vec-Gen inverts this by learning to represent what the LLM *would generate* rather than the query itself.

This approach transfers reasoning capabilities and safety alignment from the base LLM into embeddings, enabling efficient downstream use without repeated generation. The method freezes the LLM backbone while training only lightweight adapter tokens and projection layers, making it computationally efficient.

## Core Concept

Rather than encoding queries directly, LLM2Vec-Gen compresses the LLM's potential response distribution into learnable tokens. Two special token types—thought tokens (computational buffer) and compression tokens (semantic holders)—are appended to queries. During encoding, these tokens accumulate hidden states that represent response semantics, which are then optimized via reconstruction and alignment objectives.

## Architecture Overview

- **Frozen LLM backbone**: Pre-trained model weights remain unchanged throughout training
- **Thought tokens**: Provide computational staging area (typically 8-16 tokens)
- **Compression tokens**: Capture semantic content of the response (typically 4-8 tokens)
- **Projection layer**: Lightweight linear transformation mapping token hidden states to embedding space
- **Teacher embedding encoder**: Unsupervised reference model (e.g., SigLIP) for alignment

The flow is: query + special tokens → frozen LLM forward pass → extract compression token hidden states → project to embedding space.

## Implementation Steps

### Step 1: Prepare Special Tokens

Create trainable tokens for thought and compression phases. These are initialized as random embeddings and updated during training.

```python
import torch.nn as nn

thought_tokens = nn.Parameter(torch.randn(num_thought_tokens, hidden_dim))
compression_tokens = nn.Parameter(torch.randn(num_compression_tokens, hidden_dim))
projection = nn.Linear(hidden_dim, embedding_dim)
```

### Step 2: Forward Pass with Frozen LLM

Append special tokens to the query, process through the frozen LLM backbone, and extract compression token representations.

```python
def encode_query(query_ids, llm_model):
    # Append special tokens
    full_input_ids = torch.cat([
        query_ids,
        thought_token_ids,
        compression_token_ids
    ], dim=1)

    # Forward pass through frozen LLM
    with torch.no_grad():
        outputs = llm_model(full_input_ids, output_hidden_states=True)

    # Extract compression token hidden states
    hidden_states = outputs.hidden_states[-1]
    compression_states = hidden_states[:, -num_compression_tokens:, :]

    # Project to embedding space
    embedding = projection(compression_states.mean(dim=1))
    return embedding
```

### Step 3: Dual Training Objectives

Optimize two complementary losses that align compression tokens with LLM response semantics.

```python
def training_loss(query_ids, response_ids, llm_model, teacher_encoder):
    # Get compression token states
    embedding = encode_query(query_ids, llm_model)

    # Reconstruction loss: compression tokens predict response
    response_logits = llm_model.lm_head(compressed_states)
    recon_loss = cross_entropy(response_logits, response_ids)

    # Alignment loss: match teacher embedding of response
    teacher_response_embedding = teacher_encoder(response_ids)
    align_loss = cosine_distance(embedding, teacher_response_embedding)

    total_loss = recon_loss + align_weight * align_loss
    return total_loss
```

## Practical Guidance

**When to Use:**
- Building dense retrieval systems using LLM-aware semantics
- Transfer learning from large LLMs to efficient downstream encoders
- Scenarios where reasoning-aware embeddings improve retrieval quality
- Safety-aligned embedding spaces required

**When NOT to Use:**
- Single-step classification tasks where raw input semantics suffice
- Extremely budget-constrained inference (projection adds overhead)
- Non-LLM domains where response generation is irrelevant

**Hyperparameter Tuning:**
- **Number of thought tokens**: 8-16 works well; more enables richer intermediate representations
- **Number of compression tokens**: 4-8 sufficient for most tasks; controls bottleneck width
- **Embedding dimension**: Match downstream task requirements (384-1024 typical)
- **Alignment weight**: Start at 1.0, adjust based on reconstruction-alignment trade-off

**Common Pitfalls:**
- Unfreezing the LLM backbone dramatically increases training cost with minimal gains
- Insufficient thought tokens limits the model's ability to reason about complex queries
- Mixing incompatible teacher encoders can produce misaligned embeddings

## Reference

[LLM2Vec-Gen paper on arXiv](https://arxiv.org/abs/2603.10913)
