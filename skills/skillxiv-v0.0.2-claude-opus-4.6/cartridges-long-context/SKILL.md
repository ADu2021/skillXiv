---
name: cartridges-long-context
title: "Cartridges: Lightweight and general-purpose long context representations via self-study"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.06266"
keywords: [KV cache, context representation, efficient retrieval, composability]
description: "Train reusable pre-computed KV cache representations of large text corpora for efficient retrieval, achieving 38.6x memory reduction and 26.4x throughput improvement."
---

# Cartridges: Lightweight Long Context Representations

## Core Concept

Cartridges are pre-trained KV cache representations that encode large text corpora into memory-efficient, reusable forms. Rather than loading entire documents into context at inference, users train Cartridges offline once via "self-study," then apply them across multiple queries. The approach composes multiple Cartridges without retraining.

## Architecture Overview

- **Pre-training via self-study**: Combines synthetic conversation generation with context-distillation training
- **Lightweight KV cache encoding**: Stores corpus knowledge in hidden representations
- **Composability**: Multiple trained Cartridges combine at inference without additional training
- **Efficiency**: 38.6x memory reduction and 26.4x throughput versus in-context learning

## Implementation

### Step 1: Generate Synthetic Conversations

Create training data by generating model conversations about corpus content:

```python
class CartridgePretrainer:
    def __init__(self, base_model, corpus_documents: list):
        self.model = base_model
        self.corpus = corpus_documents

    def generate_synthetic_conversations(self,
                                        num_conversations: int = 1000
                                        ) -> list:
        """Generate synthetic QA pairs about corpus content."""
        conversations = []

        for doc in self.corpus:
            # Extract key content from document
            doc_summary = self.model.extract_summary(doc)

            # Generate multiple question-answer pairs
            for _ in range(num_conversations // len(self.corpus)):
                question = self.model.generate_question(doc_summary)
                answer = self.model.generate_answer(
                    question,
                    doc,
                    context_length=4096
                )

                conversations.append({
                    "corpus_context": doc,
                    "question": question,
                    "answer": answer,
                    "doc_id": doc.get("id")
                })

        return conversations
```

### Step 2: Train Context Distillation

Distill corpus knowledge into KV cache via synthetic conversations:

```python
class ContextDistillationTrainer:
    def __init__(self, model, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4
        )

    def compute_kv_cache_loss(self, corpus_text: str,
                             question: str,
                             answer: str) -> torch.Tensor:
        """Optimize KV cache to distill corpus knowledge."""

        # Encode corpus once to KV cache
        with torch.no_grad():
            corpus_tokens = self.model.tokenize(corpus_text)
            kv_cache = self.model.forward_and_cache(
                corpus_tokens
            )

        # Train model to answer question using cached KV
        question_tokens = self.model.tokenize(question)
        answer_tokens = self.model.tokenize(answer)

        # Forward pass with cached corpus KV
        logits = self.model.forward_with_kv_cache(
            question_tokens,
            kv_cache
        )

        # Compute loss on answer prediction
        loss = torch.nn.functional.cross_entropy(
            logits[:-1],  # Predict all but last token
            answer_tokens[1:]  # Shifted targets
        )

        return loss

    def train_epoch(self, conversations: list):
        """Train one epoch on synthetic conversations."""
        total_loss = 0.0

        for i in range(0, len(conversations), self.batch_size):
            batch = conversations[i:i + self.batch_size]

            self.optimizer.zero_grad()
            batch_loss = 0.0

            for conv in batch:
                loss = self.compute_kv_cache_loss(
                    conv["corpus_context"],
                    conv["question"],
                    conv["answer"]
                )
                batch_loss += loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()

        return total_loss / len(conversations)
```

### Step 3: Store and Compose Cartridges

Save trained KV caches and compose them at inference:

```python
class CartridgeManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.cartridges = {}

    def save_cartridge(self, corpus_id: str,
                      kv_cache: dict,
                      metadata: dict):
        """Save trained Cartridge with metadata."""
        cartridge = {
            "kv_cache": kv_cache,
            "corpus_id": corpus_id,
            "corpus_summary": metadata.get("summary"),
            "doc_count": metadata.get("doc_count"),
            "token_count": metadata.get("token_count")
        }

        save_path = f"{self.storage_path}/{corpus_id}.pt"
        torch.save(cartridge, save_path)
        self.cartridges[corpus_id] = cartridge

    def compose_cartridges(self, cartridge_ids: list) -> dict:
        """Combine multiple Cartridges at inference."""
        composed_kv = None
        metadata_list = []

        for cart_id in cartridge_ids:
            cartridge = torch.load(
                f"{self.storage_path}/{cart_id}.pt"
            )
            metadata_list.append({
                "corpus_id": cart_id,
                "summary": cartridge["corpus_summary"]
            })

            # Merge KV caches (concatenate along sequence dimension)
            if composed_kv is None:
                composed_kv = cartridge["kv_cache"]
            else:
                composed_kv = self._merge_kv_caches(
                    composed_kv,
                    cartridge["kv_cache"]
                )

        return {
            "combined_kv_cache": composed_kv,
            "source_cartridges": metadata_list
        }

    def _merge_kv_caches(self, kv1: dict, kv2: dict) -> dict:
        """Concatenate KV caches along sequence dimension."""
        merged = {}
        for layer in kv1.keys():
            # Concatenate keys and values from both caches
            merged[layer] = {
                "key": torch.cat([kv1[layer]["key"],
                                 kv2[layer]["key"]], dim=0),
                "value": torch.cat([kv1[layer]["value"],
                                   kv2[layer]["value"]], dim=0)
            }
        return merged
```

### Step 4: Query with Composed Cartridges

Generate answers using pre-computed corpus representations:

```python
def answer_query_with_cartridges(model,
                                 question: str,
                                 composed_cartridges: dict) -> str:
    """Answer question using composed Cartridge KV caches."""

    question_tokens = model.tokenize(question)

    # Generate using pre-computed KV caches
    response = model.generate_with_kv_cache(
        question_tokens,
        kv_cache=composed_cartridges["combined_kv_cache"],
        max_length=512
    )

    return model.detokenize(response)
```

## Practical Guidance

**Pre-training Strategy**: Self-study synthetic conversations outperform naive next-token prediction on corpus text. Generate diverse QA pairs that cover different aspects of the corpus.

**Memory Efficiency**: Cartridges achieve 38.6x memory savings over in-context learning because KV caches are much smaller than full token sequences. This enables handling 484K effective context on MTOB benchmarks.

**Composition Without Retraining**: Pre-trained Cartridges compose directly at inference by concatenating KV sequences. No fine-tuning needed to combine multiple corpora.

**When to Apply**: Use Cartridges for frequently-queried corpora, knowledge bases, or technical documentation where amortizing pre-training over many queries justifies the offline computation cost.

## Reference

Cartridges represent a shift from retrieving documents at inference to retrieving pre-computed KV representations. The self-study approach (synthetic conversations plus context distillation) proves more effective than naive corpus encoding. Composability enables flexible corpus combinations without additional training overhead.
