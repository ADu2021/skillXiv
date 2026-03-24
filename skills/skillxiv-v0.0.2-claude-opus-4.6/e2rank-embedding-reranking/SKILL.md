---
name: e2rank-embedding-reranking
title: "E2Rank: Text Embedding as Effective and Efficient Listwise Reranker"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.22733"
keywords: [Reranking, Embeddings, Retrieval, Information Retrieval, Efficiency]
description: "Extends text embedding models to perform listwise reranking through continued training on ranking objectives. Constructs listwise prompts from queries and top-K candidates, leveraging pseudo-relevance feedback while maintaining embedding model efficiency. Unifies retrieval and reranking in single model."
---

# E2Rank: Unified Retrieval and Reranking with Embeddings

Text embeddings excel at retrieval but struggle with reranking. E2Rank extends embedding models using listwise training objectives, enabling them to perform both tasks efficiently from a single model.

By training on listwise ranking, embeddings learn to interpret similarity differently for reranking tasks while preserving retrieval capabilities.

## Core Concept

Key insight: **embed ranking information in how similarity is computed**, not just in what the embeddings represent:
- Standard retrieval: cosine similarity between query and document embeddings
- Enhanced reranking: learned ranking layer interprets similarity for ranking context
- Listwise training: use top-K candidates as context for ranking decisions
- Efficiency: single embedding model for both retrieval and reranking

## Architecture Overview

- Text embedding encoder (unchanged from standard models)
- Listwise ranking prompt construction from query and candidates
- Learned ranking interpretation layer
- Joint training on retrieval and ranking objectives

## Implementation Steps

Create listwise ranking prompts that provide rich context for ranking decisions. The prompt includes query and top-K candidates:

```python
class ListwisePromptConstructor:
    def __init__(self, query_template=None):
        self.query_template = query_template or (
            "Query: {query}\n"
            "Candidates:\n{candidates}\n"
            "Rank candidates by relevance."
        )

    def construct_ranking_prompt(self, query, candidates, scores=None):
        """Build listwise context prompt for ranking."""
        candidate_text = "\n".join([
            f"{i+1}. {cand}" for i, cand in enumerate(candidates)
        ])

        prompt = self.query_template.format(
            query=query,
            candidates=candidate_text
        )

        return prompt

    def construct_ranking_examples(self, queries, doc_rankings, top_k=10):
        """Create training examples from ranked lists."""
        training_pairs = []

        for query, ranked_docs in zip(queries, doc_rankings):
            # Get top-k for context
            topk_docs = ranked_docs[:top_k]

            # Create listwise prompt
            prompt = self.construct_ranking_prompt(query, topk_docs)

            # Each position provides ranking signal
            for rank, doc in enumerate(topk_docs):
                training_pairs.append({
                    'query': query,
                    'candidate': doc,
                    'rank': rank,
                    'context_prompt': prompt
                })

        return training_pairs
```

Extend the embedding model with a learned ranking layer. Rather than relying solely on cosine similarity, learn a ranking function:

```python
class RankingEmbeddingModel(nn.Module):
    def __init__(self, embedding_model, hidden_dim=256):
        super().__init__()

        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.get_embedding_dim()

        # Learned ranking layer
        self.ranking_head = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def embed(self, texts):
        """Get embeddings (standard retrieval)."""
        return self.embedding_model.encode(texts)

    def rank(self, query_embedding, document_embeddings):
        """Rank documents using learned layer."""
        # Compute base similarity
        batch_size = document_embeddings.shape[0]
        query_exp = query_embedding.unsqueeze(0).expand(batch_size, -1)

        # Concatenate query and doc embeddings
        combined = torch.cat([query_exp, document_embeddings], dim=-1)

        # Learned ranking score
        ranking_scores = self.ranking_head(combined).squeeze(-1)

        return ranking_scores

    def rerank(self, query, documents, top_k=10):
        """Full reranking pipeline."""
        # Get embeddings
        query_emb = self.embed([query])[0]
        doc_embs = self.embed(documents)

        # Rank using learned layer
        scores = self.rank(query_emb, doc_embs)

        # Get top-k
        topk_indices = torch.argsort(scores, descending=True)[:top_k]
        topk_docs = [documents[i] for i in topk_indices.cpu().numpy()]
        topk_scores = scores[topk_indices]

        return topk_docs, topk_scores.detach().cpu().numpy()
```

Train with listwise loss that encourages proper ranking of candidates:

```python
def listwise_ranking_loss(ranking_scores, labels, reduction='mean'):
    """ListNet or LambdaRank-style loss for ranking."""
    # Normalize scores to probabilities
    log_probs = torch.nn.functional.log_softmax(ranking_scores, dim=0)

    # Expected rank loss: penalize incorrect relative ordering
    batch_size = len(labels)
    loss = 0

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            # If doc i should rank higher than j
            if labels[i] > labels[j]:
                # Penalize if score[j] > score[i]
                pairwise_loss = torch.log1p(
                    torch.exp(-(ranking_scores[i] - ranking_scores[j]))
                )
                loss += pairwise_loss

    return loss / (batch_size * (batch_size - 1) / 2) if reduction == 'mean' else loss

def train_reranking_embedding(model, train_pairs, num_epochs=5):
    """Train model on listwise ranking pairs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        for batch_pairs in train_pairs:
            query = batch_pairs['query']
            candidates = [p['candidate'] for p in batch_pairs]
            labels = torch.tensor([p['rank'] for p in batch_pairs])

            # Get embeddings
            query_emb = model.embed([query])[0]
            doc_embs = model.embed(candidates)

            # Compute ranking scores
            ranking_scores = model.rank(query_emb, doc_embs)

            # Listwise loss
            loss = listwise_ranking_loss(ranking_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Ranking layer hidden dim | 256-512 |
| Learning rate | 1e-5 (preserve embedding knowledge) |
| Top-k for context | 10-20 candidates |
| Listwise batch size | 32-64 (pairs within batch) |

**When to use:**
- Unified retrieval + reranking pipelines
- Cost-sensitive deployment (single model)
- Scenarios where embedding retrieval is already in place
- Information retrieval tasks needing quality ranking

**When NOT to use:**
- When dedicated reranking models available
- Cross-encoder reranking when quality matters most (single embedding model less expressive)
- Extremely large document collections (ranking layer adds latency)

**Common pitfalls:**
- Ranking layer too complex (overfits to training queries)
- Not freezing embedding encoder (destabilizes retrieval)
- Imbalanced positive/negative examples in listwise batches
- Top-k context not representative of full ranking

Reference: [E2Rank on arXiv](https://arxiv.org/abs/2510.22733)
