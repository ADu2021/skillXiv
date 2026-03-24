---
name: query-bandits-hallucination
title: "No One Size Fits All: QueryBandits for Hallucination Mitigation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.20332"
keywords: [Hallucination Mitigation, Contextual Bandits, Query Rewriting, LLM Optimization, Thompson Sampling]
description: "QueryBandits adaptively learns per-query rewriting strategies to reduce LLM hallucinations, achieving 87.5% improvement without model retraining."
---

# Technique: Adaptive Query Rewriting via Contextual Bandits

Large language models frequently hallucinate facts not present in the training data. Traditional mitigation approaches apply fixed query-rewriting strategies (paraphrase, expand context, add constraints) uniformly across all queries. However, research shows that "there is no single rewrite policy optimal for all queries"—some queries benefit from rephrasing, others from constraint injection, and some from explicit evidence-seeking rewrites.

QueryBandits solves this by learning online which rewriting strategy works best for each query's semantic context, using Thompson Sampling and contextual bandit algorithms. This enables model-agnostic deployment (works with closed-source APIs) without any retraining.

## Core Concept

The core insight: query-rewriting effectiveness varies based on semantic query type. Rather than committing to one strategy, maintain multiple candidate strategies and learn online which one reduces hallucinations for similar queries.

**Strategies tested:**
- Paraphrase: Rephrase query to encourage diverse reasoning
- Expand: Add contextual background or constraints
- Evidence-seek: Ask model to first list evidence before answering
- Constrain: Enforce consistency rules (e.g., "only mention known entities")

The bandit algorithm learns reward signals (correctness on validation data) and exploits this knowledge to route future queries to effective strategies.

## Architecture Overview

- **Contextual Extraction**: Embed queries into semantic space
- **Multi-armed Bandits**: Maintain belief distributions over strategy effectiveness per context
- **Thompson Sampling**: Select strategies probabilistically, balancing exploration and exploitation
- **Feedback Loop**: Collect correctness signals and update beliefs
- **Model-Agnostic**: Works via API calls; no model modification needed

## Implementation Steps

QueryBandits operates as a post-processing layer before sending queries to LLMs. Here's the core workflow:

First, extract semantic features from each query to identify its type. These features guide strategy selection:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize query embedding
query_vectorizer = TfidfVectorizer(max_features=128)

# Extract query context features
def extract_query_context(query_text, vectorizer):
    # Compute semantic embedding
    query_embedding = vectorizer.fit_transform([query_text])[0].toarray()

    # Extract features: question type, entity count, reasoning complexity
    features = {
        'has_numbers': any(c.isdigit() for c in query_text),
        'has_entities': len([w for w in query_text.split() if w[0].isupper()]),
        'query_length': len(query_text.split()),
        'embedding': query_embedding
    }
    return features
```

Implement a Thompson Sampling bandit that learns which strategy works for each query type:

```python
# Thompson Sampling for multi-armed query rewriting
class QueryBandit:
    def __init__(self, num_strategies=4, num_context_clusters=16):
        # Per-context belief about each strategy's effectiveness
        # Track (successes, failures) for each strategy per context cluster
        self.success_counts = np.ones((num_context_clusters, num_strategies))
        self.failure_counts = np.ones((num_context_clusters, num_strategies))
        self.num_context_clusters = num_context_clusters

    def select_strategy(self, query_context_embedding):
        # Map query to context cluster
        cluster_id = self._cluster_query(query_context_embedding)

        # Thompson Sampling: sample from Beta posterior for each strategy
        samples = []
        for s in range(self.num_strategies):
            alpha = self.success_counts[cluster_id, s]
            beta = self.failure_counts[cluster_id, s]
            sample = np.random.beta(alpha, beta)
            samples.append(sample)

        # Select strategy with highest sampled effectiveness
        return np.argmax(samples)

    def update(self, cluster_id, strategy_id, was_correct):
        # Update belief based on outcome
        if was_correct:
            self.success_counts[cluster_id, strategy_id] += 1
        else:
            self.failure_counts[cluster_id, strategy_id] += 1

    def _cluster_query(self, embedding):
        # Simple clustering: find nearest cluster center
        # In practice, use k-means or learned clustering
        return np.random.randint(0, self.num_context_clusters)
```

Define the rewriting strategies as functions that modify the query:

```python
class QueryRewriter:
    def paraphrase(self, query):
        # Use model to generate alternative phrasing
        return f"Rephrase and answer: {query}"

    def expand(self, query):
        # Add context expansion cue
        return f"{query}\nProvide comprehensive context and reasoning."

    def evidence_seek(self, query):
        # Ask for evidence first
        return f"First, what evidence supports an answer to: {query}?\nThen answer the question."

    def constrain(self, query):
        # Add consistency constraint
        return f"Answer only using factual, verifiable information: {query}"

# Wrapper: select and apply strategy based on bandit
def adaptive_query_rewrite(query, bandit, rewriter, features):
    strategy_id = bandit.select_strategy(features['embedding'])

    strategies = [
        rewriter.paraphrase,
        rewriter.expand,
        rewriter.evidence_seek,
        rewriter.constrain
    ]

    rewritten_query = strategies[strategy_id](query)
    return rewritten_query, strategy_id
```

## Practical Guidance

**When to Use:**
- Deployed LLM APIs where retraining is impossible
- QA systems with validation data to measure hallucination
- Long-tail domain queries where strategy effectiveness varies
- Systems where you can collect outcome feedback (correctness signals)

**When NOT to Use:**
- Real-time systems with strict latency requirements (bandit overhead)
- Domains where all queries benefit equally from one strategy
- When validation data is very limited (<100 labeled examples)

**Hyperparameters:**
- `num_context_clusters`: Controls granularity of per-query learning (8–32 typical)
- `exploration_rate`: Thompson Sampling automatically balances; tune via prior strength
- Strategy definitions: Tailor to your domain (add domain-specific rewrites if needed)

**Integration Steps:**
1. Collect validation set with correctness labels (QA pairs with ground truth)
2. Initialize bandit with uniform priors
3. Route queries through adaptive rewriter before sending to LLM
4. Collect feedback (was LLM answer correct?) and update bandit
5. Over time, bandit learns per-query-type optimal strategies

**Results:**
Thompson Sampling achieved 87.5% win rate over baseline fixed-strategy approach across 16 QA scenarios. Average accuracy gains of 2–5% depending on hallucination prevalence in domain.

---

**Reference:** [No One Size Fits All: QueryBandits for Hallucination Mitigation](https://arxiv.org/abs/2602.20332)
