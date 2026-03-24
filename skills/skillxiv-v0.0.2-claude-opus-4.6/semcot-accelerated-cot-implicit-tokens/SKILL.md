---
name: semcot-accelerated-cot-implicit-tokens
title: "SemCoT: Accelerating Chain-of-Thought via Semantically-Aligned Implicit Tokens"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.24940"
keywords: [Chain-of-Thought, Knowledge Distillation, Semantic Alignment, Inference Optimization, Implicit Reasoning]
description: "Encode reasoning steps as hidden embeddings instead of explicit text using contrastively-trained sentence transformers and lightweight distilled models, reducing token generation cost while preserving semantic alignment with ground-truth reasoning."
---

# Title: Compress Reasoning Into Hidden Embeddings With Semantic Fidelity

Chain-of-thought reasoning improves accuracy but costs extra tokens. SemCoT compresses explicit reasoning steps into implicit tokens—single vectors in the LLM's hidden space—while maintaining semantic fidelity through contrastive alignment. A sentence transformer trained on reasoning pairs guides a lightweight student model to generate semantically-equivalent implicit tokens without explicit textual generation.

The approach combines two key ideas: (1) contrastive alignment ensures implicit representations preserve ground-truth reasoning semantics, and (2) knowledge distillation from a student model reduces per-token cost.

## Core Concept

**Implicit Chain-of-Thought with Semantic Preservation**:
- **Explicit CoT**: Generate full reasoning steps as text tokens (slow, verbose)
- **SemCoT**: Generate special `<CoT>` tokens whose embeddings encode reasoning (fast, compressed)
- **Contrastive Training**: Align implicit embeddings with explicit reasoning via learned metric
- **Lightweight Generator**: Small distilled model produces implicit tokens efficiently
- **Dual Loss**: Accuracy loss (correct answers) + semantic alignment loss (reasoning fidelity)

## Architecture Overview

- **Sentence Transformer**: Trained to measure semantic similarity between reasoning pairs (uses LLM middle layers)
- **Implicit Token Generator**: Lightweight language model (distilled/pruned from target LLM)
- **Embedding Projection**: Linear transformation from student embedding space to main LLM space
- **Dual Optimization**: Cross-entropy loss + semantic alignment via contrastive similarity
- **Inference Mode**: Single pass generating implicit `<CoT>` tokens for each query

## Implementation Steps

**1. Train Contrastive Sentence Transformer**

Create a metric that measures whether implicit and explicit reasoning are semantically equivalent.

```python
class ReasoningSentenceTransformer(nn.Module):
    def __init__(self, llm, hidden_dim=768):
        self.llm = llm
        # Use middle 5 layers of LLM as backbone
        self.backbone = llm.transformer.h[6:11]  # layers 6-10
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(hidden_dim, 256)

    def forward(self, input_ids, attention_mask):
        # Extract hidden states from middle layers
        hidden = self.backbone(input_ids, attention_mask)[-1]
        # Pool over sequence length
        pooled = self.pooling(hidden.transpose(1, 2)).squeeze(-1)
        # Project to embedding space
        embeddings = self.projection(pooled)
        return embeddings  # [batch_size, 256]

def train_sentence_transformer(transformer, reasoning_pairs, epochs=5):
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in reasoning_pairs:
            explicit_reasoning = batch['explicit']  # Ground-truth CoT text
            condensed_reasoning = batch['condensed']  # GPT-4 mini-generated summary

            # Encode both
            emb_explicit = transformer(explicit_reasoning)
            emb_condensed = transformer(condensed_reasoning)

            # Positive pair loss: explicit and condensed should be similar
            pos_sim = F.cosine_similarity(emb_explicit, emb_condensed)
            pos_loss = 1 - pos_sim.mean()

            # Negative pairs: shuffle batch
            shuffled_indices = torch.randperm(len(batch))
            emb_negatives = emb_explicit[shuffled_indices]
            neg_sim = F.cosine_similarity(emb_explicit, emb_negatives)
            neg_loss = torch.clamp(neg_sim + 0.5, min=0).mean()

            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return transformer
```

**2. Create Lightweight Student Generator**

Build a small model that generates implicit reasoning tokens efficiently.

```python
class ImplicitTokenGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=512, num_layers=2):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        # Output embedding aligned with main LLM
        self.embedding_projection = nn.Linear(hidden_dim, embedding_dim)

    def generate_implicit_token(self, query_ids, target_embedding_dim=768):
        # Query encoding
        embedded = self.embedding(query_ids)
        lstm_out, _ = self.lstm(embedded)
        # Last token hidden state
        last_hidden = lstm_out[:, -1, :]
        # Project to target embedding dimension
        implicit_embedding = self.embedding_projection(last_hidden)
        return implicit_embedding  # [batch_size, embedding_dim]

    def forward(self, input_ids):
        # Standard language modeling for training
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        # Project back to vocab for next-token prediction
        logits = self.embedding_projection(lstm_out)  # [batch, seq, embedding_dim]
        # Linear projection to vocabulary
        vocab_logits = torch.nn.functional.linear(logits, self.embedding.weight)
        return vocab_logits
```

**3. Implement Dual-Loss Training**

Combine accuracy supervision with semantic alignment supervision.

```python
def train_implicit_generator(generator, sentence_transformer, dataset, target_llm, lambda_align=0.5):
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    sentence_transformer.eval()  # Freeze transformer

    for batch in dataset:
        queries = batch['query']
        target_answers = batch['answer']
        explicit_cot = batch['explicit_reasoning']

        # Generate implicit token
        implicit_embedding = generator.generate_implicit_token(queries)

        # Loss 1: Answer prediction accuracy
        # Append implicit token to query embeddings
        query_embeds = target_llm.transformer.wte(queries)
        # Concatenate implicit CoT token to query
        augmented_input = torch.cat([query_embeds, implicit_embedding.unsqueeze(1)], dim=1)
        # Generate answer
        output = target_llm(inputs_embeds=augmented_input)
        logits = output.logits[:, -1, :]  # Last position

        # Accuracy loss
        answer_loss = F.cross_entropy(logits, target_answers)

        # Loss 2: Semantic alignment
        # Embed implicit token through transformer
        implicit_hidden = target_llm.transformer.h[8](
            implicit_embedding.unsqueeze(1)
        )[-1].squeeze(1)
        implicit_semantic = sentence_transformer.projection(implicit_hidden)

        # Embed explicit reasoning
        explicit_semantic = sentence_transformer(explicit_cot)

        # Alignment loss: maximize cosine similarity
        alignment_loss = 1 - F.cosine_similarity(implicit_semantic, explicit_semantic).mean()

        # Combined loss
        total_loss = answer_loss + lambda_align * alignment_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

**4. Deploy Implicit CoT at Inference**

Use learned implicit tokens for fast reasoning.

```python
def inference_with_implicit_cot(generator, query, target_llm):
    # Encode query
    query_ids = tokenizer.encode(query)

    # Generate implicit CoT embedding
    implicit_embedding = generator.generate_implicit_token(query_ids)

    # Append to query
    query_embeds = target_llm.transformer.wte(query_ids)
    augmented = torch.cat([query_embeds, implicit_embedding.unsqueeze(1)], dim=1)

    # Generate answer (just one forward pass)
    output = target_llm.generate(
        inputs_embeds=augmented,
        max_new_tokens=10,
        do_sample=False
    )

    return tokenizer.decode(output)
```

## Practical Guidance

**When to Use**:
- Inference-time latency critical
- Chain-of-thought needed for accuracy but cost is concern
- Deployment with compute constraints (mobile, edge)

**Hyperparameters**:
- lambda_align: 0.5 (balance between accuracy and semantics)
- implicit_token_dim: Match target LLM embedding dimension
- student_hidden_dim: ~256-512 (typically 1/3 of LLM dimension)

**When NOT to Use**:
- Tasks where explicit reasoning is debugging requirement
- Scenarios requiring interpretable reasoning traces
- Models smaller than 3B parameters (minimal speedup)

**Pitfalls**:
- **Weak semantic alignment**: If sentence transformer undertrained, implicit tokens diverge from reasoning
- **Generator overfitting**: Small models can memorize rather than compress; validate on held-out questions
- **Modality mismatch**: Implicit embeddings must align with target LLM's embedding space

**Integration Strategy**: Apply as post-processing to fine-tuned CoT models. Train generator and transformer on model's own reasoning traces for domain-specific alignment.

## Reference

arXiv: https://arxiv.org/abs/2510.24940
