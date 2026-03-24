---
name: search-r3-reasoning
title: "Search-R3: Unifying Reasoning and Embedding Generation for Retrieval"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.07048
keywords: [retrieval-augmented-generation, reasoning, embedding-generation, post-training, knowledge-intensive-tasks]
description: "Unify LLM reasoning with embedding generation by training models to output embeddings as direct outputs of step-by-step reasoning. Trigger: improve performance on complex retrieval tasks requiring deep reasoning."
---

# Search-R3: Reasoning + Retrieval Integration

## Core Concept

Search-R3 addresses a critical gap: Large language models excel at reasoning but have been underutilized for retrieval tasks. Current systems treat reasoning and retrieval as separate pipelines. Search-R3 unifies these by training LLMs to generate search embeddings as direct outputs of their reasoning process. Through combined supervised learning and RL, models learn to produce embeddings that simultaneously capture both reasoning depth and retrieval relevance, improving performance on knowledge-intensive tasks requiring complex reasoning.

The key insight: Embeddings are not fixed representations; they can be outputs of reasoning, making retrieval and reasoning co-dependent and co-optimized.

## Architecture Overview

- **Unified Post-Training**: Combines supervised learning + RL in single framework
- **Reasoning-Driven Embeddings**: Embeddings are outputs of chain-of-thought
- **Specialized RL Environment**: Efficiently handles dynamic corpus representations
- **Two-Stage Learning**: Supervised embedding generation → RL optimization
- **No Corpus Re-encoding**: Embeddings computed once during training

## Implementation Steps

### 1. Understand Reasoning-Based Embedding Generation

Define how embeddings emerge from reasoning.

```python
class ReasoningEmbeddingGenerator:
    """
    Generate embeddings through explicit reasoning steps.
    """
    def __init__(self, model, embedding_dim=768):
        self.model = model
        self.embedding_dim = embedding_dim

    def generate_reasoning_and_embedding(self, query):
        """
        Chain-of-thought reasoning that concludes with embedding generation.

        Args:
            query: Query string or question

        Returns:
            (reasoning_trace, embedding_vector)
        """
        # Step 1: Generate reasoning
        reasoning_prompt = (
            f"Query: {query}\n\n"
            f"Think through this step-by-step:\n"
            f"1. What is being asked?\n"
            f"2. What concepts are key?\n"
            f"3. What information would be relevant?\n\n"
            f"Reasoning: "
        )

        reasoning = self.model.generate(
            reasoning_prompt,
            max_tokens=300,
            temperature=0.5
        )

        # Step 2: Generate embedding as model output
        embedding_prompt = (
            f"Query: {query}\n\n"
            f"Reasoning: {reasoning}\n\n"
            f"Generate a search embedding capturing the query and reasoning "
            f"(output as {self.embedding_dim} float values): "
        )

        embedding_text = self.model.generate(
            embedding_prompt,
            max_tokens=500,
            temperature=0.3
        )

        # Parse embedding
        embedding = self.parse_embedding_output(embedding_text)

        return reasoning, embedding

    def parse_embedding_output(self, embedding_text):
        """
        Convert model-generated text to embedding vector.
        """
        # Extract numbers from model output
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', embedding_text)

        if len(numbers) >= self.embedding_dim:
            embedding = torch.tensor([float(n) for n in numbers[:self.embedding_dim]])
        else:
            # If fewer numbers, pad with learned default
            embedding = torch.zeros(self.embedding_dim)
            for i, num in enumerate(numbers):
                embedding[i] = float(num)

        # Normalize
        embedding = embedding / (torch.norm(embedding) + 1e-8)

        return embedding

    def embedding_quality(self, reasoning, embedding, relevant_docs):
        """
        Evaluate embedding quality via retrieval.
        """
        # Compute similarities to relevant documents
        doc_embeddings = self.model.encode_documents(relevant_docs)

        similarities = []
        for doc_emb in doc_embeddings:
            sim = torch.dot(embedding, doc_emb) / (
                torch.norm(embedding) * torch.norm(doc_emb) + 1e-8
            )
            similarities.append(sim.item())

        # Quality: how well does embedding capture relevant documents?
        recall_at_1 = 1.0 if max(similarities) > 0.7 else 0.0
        mean_similarity = np.mean(similarities)

        return {
            "recall_at_1": recall_at_1,
            "mean_similarity": mean_similarity,
            "embedding_quality": recall_at_1 * 0.7 + mean_similarity * 0.3
        }
```

### 2. Implement Supervised Learning Stage

Train models to generate quality embeddings via SFT.

```python
class SupervisedEmbeddingTraining:
    """
    Supervised fine-tuning for embedding generation.
    """
    def __init__(self, model, corpus):
        self.model = model
        self.corpus = corpus  # Document collection

    def create_training_pairs(self, query_doc_pairs, num_negatives=5):
        """
        Create (query, reasoning, embedding, labels) tuples.

        Args:
            query_doc_pairs: List of (query, relevant_doc) pairs
            num_negatives: Negative documents per query

        Returns:
            Training examples
        """
        training_examples = []

        for query, relevant_doc in query_doc_pairs:
            # Generate reasoning for this query
            reasoning_prompt = (
                f"Query: {query}\n"
                f"Relevant document: {relevant_doc[:200]}...\n\n"
                f"Reasoning for retrieval: "
            )

            reasoning = self.model.generate(
                reasoning_prompt,
                max_tokens=200
            )

            # Generate positive (relevant) embedding
            positive_embedding = self.model.encode_document(relevant_doc)

            # Sample negative (irrelevant) documents
            negative_docs = self._sample_negatives(
                query,
                relevant_doc,
                num_negatives
            )
            negative_embeddings = [
                self.model.encode_document(doc) for doc in negative_docs
            ]

            # Create training example
            example = {
                "query": query,
                "reasoning": reasoning,
                "positive_embedding": positive_embedding,
                "negative_embeddings": negative_embeddings,
                "target_quality": 1.0  # Positive example
            }

            training_examples.append(example)

        return training_examples

    def _sample_negatives(self, query, relevant_doc, num_negatives):
        """Sample hard negatives: relevant but not the specific document."""
        # BM25 or semantic search to get candidates
        candidate_docs = self.corpus.search(query, top_k=50)

        negatives = [
            doc for doc in candidate_docs
            if doc != relevant_doc
        ][:num_negatives]

        return negatives

    def train_supervised(self, training_examples, num_epochs=5):
        """
        SFT on reasoning + embedding generation.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            epoch_loss = 0

            for example in training_examples:
                query = example["query"]
                reasoning = example["reasoning"]
                positive_emb = example["positive_embedding"]

                # Generate embedding output
                embedding_prompt = (
                    f"Query: {query}\n"
                    f"Reasoning: {reasoning}\n"
                    f"Embedding: "
                )

                generated_embedding_text = self.model.generate(
                    embedding_prompt,
                    max_tokens=200
                )

                generated_embedding = parse_embedding(generated_embedding_text)

                # Contrastive loss: pull closer to positive, push away from negatives
                loss = compute_contrastive_loss(
                    generated_embedding,
                    positive_emb,
                    example["negative_embeddings"]
                )

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch}: SFT loss = {epoch_loss / len(training_examples):.4f}")

        return self.model
```

### 3. Implement Specialized RL Environment

RL for embedding optimization without corpus re-encoding.

```python
class EmbeddingRLEnvironment:
    """
    RL environment for embedding optimization.
    Handles evolving embeddings without re-encoding corpus.
    """
    def __init__(self, model, corpus_embeddings, ground_truth_labels):
        self.model = model
        self.corpus_embeddings = corpus_embeddings  # Pre-encoded documents
        self.ground_truth = ground_truth_labels  # Relevance labels
        self.cache = {}  # Cache computed similarities

    def compute_reward(self, query_embedding, query, relevant_docs):
        """
        Reward function: how well does embedding retrieve relevant documents?
        """
        similarities = []

        for doc_id, doc_embedding in enumerate(self.corpus_embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((doc_id, similarity))

        # Rank documents
        rankings = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Compute MRR (Mean Reciprocal Rank) for relevant docs
        mrr = 0
        for rank, (doc_id, _) in enumerate(rankings):
            if doc_id in self.ground_truth[query]:
                mrr = 1.0 / (rank + 1)
                break

        # Compute NDCG
        dcg = 0
        for rank, (doc_id, _) in enumerate(rankings[:10]):
            if doc_id in self.ground_truth[query]:
                dcg += 1.0 / np.log2(rank + 2)

        ideal_dcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(self.ground_truth[query]), 10))
        )

        ndcg = dcg / (ideal_dcg + 1e-8)

        # Combined reward
        reward = 0.5 * mrr + 0.5 * ndcg

        return reward

    def step(self, query, reasoning, query_embedding):
        """
        Execute one RL step: evaluate embedding quality and return reward.
        """
        reward = self.compute_reward(query_embedding, query, self.ground_truth)

        # Truncated to [-1, 1] for stability
        reward = np.clip(reward, -1, 1)

        return reward
```

### 4. Implement RL Training Stage

Optimize embeddings through reinforcement learning.

```python
class EmbeddingRLTraining:
    """
    RL fine-tuning for embedding generation.
    """
    def __init__(self, model, env):
        self.model = model
        self.env = env

    def train_rl(self, query_list, num_epochs=10):
        """
        RL training for embedding optimization.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-6)

        for epoch in range(num_epochs):
            epoch_reward = 0

            for query in query_list:
                # Generate reasoning
                reasoning_prompt = f"Query: {query}\nReasoning: "
                reasoning = self.model.generate(
                    reasoning_prompt,
                    max_tokens=200
                )

                # Generate embedding
                embedding_prompt = (
                    f"Query: {query}\n"
                    f"Reasoning: {reasoning}\n"
                    f"Embedding: "
                )

                embedding_text = self.model.generate(
                    embedding_prompt,
                    max_tokens=200
                )

                query_embedding = parse_embedding(embedding_text)

                # Evaluate in RL environment
                reward = self.env.step(query, reasoning, query_embedding)

                # Policy gradient loss
                log_prob = self.model.compute_log_prob(embedding_text)
                pg_loss = -reward * log_prob

                # Entropy bonus for diversity
                entropy = compute_entropy_bonus(
                    self.model.get_logits(embedding_prompt)
                )

                total_loss = pg_loss - 0.01 * entropy

                # Update
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_reward += reward

            avg_reward = epoch_reward / len(query_list)
            print(f"Epoch {epoch}: RL reward = {avg_reward:.4f}")

        return self.model
```

### 5. Full Search-R3 Training Pipeline

Combine SFT and RL stages.

```python
def train_search_r3(
    base_model,
    corpus,
    training_queries,
    test_queries,
    config
):
    """
    Complete Search-R3 training pipeline.
    """
    # Pre-encode corpus for RL efficiency
    corpus_embeddings = base_model.encode_documents(corpus)

    # Create ground truth labels (via BM25 or manual)
    ground_truth = create_retrieval_labels(training_queries, corpus)

    # Stage 1: Supervised Learning
    print("Stage 1: Supervised Fine-Tuning")
    sft_trainer = SupervisedEmbeddingTraining(base_model, corpus)

    training_pairs = create_training_pairs(training_queries, corpus)
    training_examples = sft_trainer.create_training_pairs(training_pairs)

    sft_model = sft_trainer.train_supervised(training_examples, num_epochs=5)

    # Stage 2: Reinforcement Learning
    print("\nStage 2: Reinforcement Learning")
    env = EmbeddingRLEnvironment(
        sft_model,
        corpus_embeddings,
        ground_truth
    )

    rl_trainer = EmbeddingRLTraining(sft_model, env)
    final_model = rl_trainer.train_rl(
        training_queries,
        num_epochs=10
    )

    # Evaluation
    print("\nEvaluation")
    metrics = evaluate_search_r3(final_model, test_queries, corpus)

    return final_model, metrics
```

### 6. Evaluation: Retrieval + Reasoning

Measure performance on knowledge-intensive tasks.

```python
def evaluate_search_r3(model, test_queries, corpus):
    """
    Evaluate Search-R3 on retrieval + reasoning tasks.
    """
    results = {
        "mrr": [],
        "ndcg": [],
        "reasoning_quality": []
    }

    for query in test_queries:
        # Generate reasoning + embedding
        reasoning, embedding = model.generate_reasoning_and_embedding(query)

        # Retrieve documents
        corpus_embeddings = model.encode_documents(corpus)
        similarities = [
            torch.dot(embedding, doc_emb).item()
            for doc_emb in corpus_embeddings
        ]

        # Evaluate retrieval
        rankings = sorted(
            enumerate(similarities),
            key=lambda x: x[1],
            reverse=True
        )

        # MRR computation
        relevant_docs = get_relevant_documents(query, corpus)
        for rank, (doc_id, _) in enumerate(rankings):
            if doc_id in relevant_docs:
                results["mrr"].append(1.0 / (rank + 1))
                break

        # Evaluate reasoning quality
        reasoning_quality = score_reasoning(reasoning, query)
        results["reasoning_quality"].append(reasoning_quality)

    avg_mrr = np.mean(results["mrr"])
    avg_reasoning = np.mean(results["reasoning_quality"])

    print(f"MRR: {avg_mrr:.3f}")
    print(f"Avg reasoning quality: {avg_reasoning:.3f}")

    return results
```

## Practical Guidance

**Hyperparameters:**
- **Supervised learning epochs**: 5 (avoid overfitting)
- **RL learning rate**: 5e-6 (conservative; smaller than SFT)
- **Embedding dimension**: 768 (standard for transformers)
- **Num negatives per query**: 5-10
- **RL epochs**: 10-20

**When to Use:**
- Complex retrieval tasks requiring reasoning
- Want unified model for reasoning + retrieval
- Have paired query-document data for supervision
- Reasoning depth correlates with retrieval quality

**When NOT to Use:**
- Simple lexical retrieval (dense embeddings overkill)
- Real-time constraints (embedding generation adds latency)
- Limited training data (<1000 query-doc pairs)
- Corpus changes frequently (pre-encoding strategy inflexible)

## Reference

[Search-R3: Unifying Reasoning and Embedding Generation for Retrieval](https://arxiv.org/abs/2510.07048) — arXiv:2510.07048
