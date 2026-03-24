---
name: sage-retrieval-research-agents
title: "SAGE: Benchmarking and Improving Retrieval for Deep Research Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.05975"
keywords: [Retrieval Systems, Research Agents, Query Decomposition, Corpus Augmentation, Information Seeking]
description: "Build effective retrieval systems for deep research agents by understanding query-retriever mismatch, preferring BM25 for keyword-oriented queries, and augmenting corpus documents with metadata and LLM-generated keywords to improve information discovery."
---

# SAGE: Benchmarking and Improving Retrieval for Deep Research Agents

## Problem Context

Deep research agents rely heavily on retrieval systems to find relevant information, yet it's unclear which retrieval approaches effectively support complex research workflows. LLM-based retrievers often underperform because agents generate keyword-oriented sub-queries while neural retrievers expect natural language, creating a fundamental mismatch. Practitioners lack clear guidance on which retrieval strategy works for which task type.

## Core Concept

SAGE introduces [corpus-level augmentation, agent-aware query understanding, BM25 preference for agents] to align retrieval systems with how agents actually generate queries. The key finding: agents decompose research problems into keyword-oriented sub-queries, making lexical retrieval (BM25) substantially outperform semantic retrievers (30% advantage on short-form questions).

## Architecture Overview

- **Benchmark**: 1,200 queries across four scientific domains; 200,000-paper corpus
- **Query analysis**: Profile how agents decompose queries; understand agent-specific patterns
- **Retrieval comparison**: BM25 vs. LLM-based retrievers (gte-Qwen2, ReasonIR)
- **Corpus augmentation**: Enhance documents with metadata, keywords, summaries
- **Agent integration**: Test retrievers within same agent framework; control for confounds

## Implementation

### Step 1: Build agent-aware query profiler

Analyze how research agents decompose complex queries. Profile decomposition patterns to understand retrieval behavior.

```python
# Profile agent query decomposition
def profile_agent_queries(agent, benchmark_queries, num_samples=100):
    """
    Sample how agent decomposes research queries into sub-queries.
    Analyze decomposition patterns: keyword-heavy, natural language, etc.
    """
    decomposition_patterns = {
        'keyword_only': 0,
        'mixed': 0,
        'natural_language': 0
    }

    for query in benchmark_queries[:num_samples]:
        # Ask agent to decompose query step-by-step
        sub_queries = agent.decompose_query(query)

        # Classify decomposition
        keyword_count = sum(
            1 for sq in sub_queries
            if is_keyword_only(sq)
        )
        ratio = keyword_count / len(sub_queries)

        if ratio > 0.8:
            decomposition_patterns['keyword_only'] += 1
        elif ratio > 0.2:
            decomposition_patterns['mixed'] += 1
        else:
            decomposition_patterns['natural_language'] += 1

    return decomposition_patterns
```

### Step 2: Implement corpus augmentation strategy

Enhance corpus documents with structured metadata and LLM-generated keywords to support multiple query modalities.

```python
# Corpus augmentation
def augment_corpus(documents, llm, augment_with=['keywords', 'summary']):
    """
    Add metadata, keywords, and summaries to corpus documents.
    Enables retrieval from diverse query patterns.
    """
    augmented = []

    for doc in documents:
        aug_doc = {'original_text': doc['text']}

        if 'keywords' in augment_with:
            # Extract keywords using LLM
            keywords = llm.extract_keywords(
                doc['text'],
                max_keywords=10
            )
            aug_doc['keywords'] = ', '.join(keywords)

        if 'summary' in augment_with:
            # Generate concise summary
            summary = llm.summarize(
                doc['text'],
                max_length=100
            )
            aug_doc['summary'] = summary

        if 'metadata' in augment_with:
            # Extract or infer metadata
            aug_doc['metadata'] = {
                'authors': extract_authors(doc),
                'year': extract_year(doc),
                'domains': classify_domains(doc['text'])
            }

        # Combine into retrievable text
        augmented_text = f"""
        {aug_doc['original_text']}
        Keywords: {aug_doc.get('keywords', '')}
        Summary: {aug_doc.get('summary', '')}
        """
        aug_doc['augmented_text'] = augmented_text
        augmented.append(aug_doc)

    return augmented
```

### Step 3: Compare BM25 vs. LLM-based retrievers

Systematically evaluate different retrieval approaches within the same agent framework, controlling for agent variation.

```python
# Retriever comparison
def compare_retrievers(agent, queries, corpus, retrievers_dict):
    """
    Test multiple retrievers (BM25, neural, hybrid) with same agent.
    Control for agent variation to isolate retriever impact.
    """
    results = {}

    for retriever_name, retriever in retrievers_dict.items():
        scores = []

        for query in queries:
            # Retrieve top-k documents
            retrieved_docs = retriever.retrieve(query, top_k=10)

            # Run agent with retrieved documents
            agent_output = agent.reason_with_documents(
                query, retrieved_docs
            )

            # Evaluate agent output against reference
            score = evaluate_answer_quality(
                agent_output,
                reference_answer=query['reference']
            )
            scores.append(score)

        results[retriever_name] = {
            'mean_score': mean(scores),
            'std_dev': std(scores),
            'scores': scores
        }

    return results
```

### Step 4: Implement BM25 with corpus augmentation

Build BM25 retriever leveraging augmented corpus. This is the recommended baseline for agent-based retrieval.

```python
# BM25 retriever with augmented corpus
class BM25RetrieveWithAugmentation:
    def __init__(self, augmented_corpus):
        from rank_bm25 import BM25Okapi

        # Tokenize augmented documents
        self.corpus_texts = [
            doc['augmented_text'] for doc in augmented_corpus
        ]
        self.tokenized_corpus = [
            doc.lower().split() for doc in self.corpus_texts
        ]

        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.corpus_docs = augmented_corpus

    def retrieve(self, query, top_k=10):
        """
        Retrieve documents using BM25 on augmented corpus.
        """
        # Tokenize query
        tokenized_query = query.lower().split()

        # Score documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        retrieved = [
            {
                'text': self.corpus_docs[i]['original_text'],
                'augmentation': {
                    'keywords': self.corpus_docs[i].get('keywords'),
                    'summary': self.corpus_docs[i].get('summary')
                },
                'score': scores[i]
            }
            for i in top_indices
        ]

        return retrieved
```

### Step 5: Evaluate retrieval-agent integration

Run full evaluation loop with agent + retriever combination; measure downstream task performance.

```python
# Full pipeline evaluation
def evaluate_retrieval_agent_integration(
    agent, retriever, test_queries, reference_answers
):
    """
    End-to-end evaluation: agent using retriever to answer questions.
    """
    results = {
        'retrieval_quality': [],
        'agent_accuracy': [],
        'end_to_end': []
    }

    for query_item in test_queries:
        query = query_item['question']
        reference = query_item['reference_answer']

        # Retrieve documents
        docs = retriever.retrieve(query, top_k=5)
        results['retrieval_quality'].append({
            'query': query,
            'docs': docs
        })

        # Agent reasons over retrieved docs
        agent_answer = agent.answer_with_retrieval(query, docs)

        # Evaluate accuracy
        accuracy = compare_answers(agent_answer, reference)
        results['agent_accuracy'].append(accuracy)

        # End-to-end metric
        results['end_to_end'].append({
            'query': query,
            'answer': agent_answer,
            'accuracy': accuracy
        })

    return results
```

## Practical Guidance

**When to use**: Research agents, scientific literature discovery, multi-document reasoning tasks. Apply SAGE analysis to understand your specific agent's query patterns before choosing retriever.

**Key findings**:
- BM25 outperforms LLM-based retrievers by ~30% on short-form questions
- Agent query decomposition is keyword-oriented: optimize for this
- Corpus augmentation (metadata + keywords) provides 8% lift on short-form questions
- Query-retriever mismatch is the primary failure mode

**Hyperparameters**:
- BM25 k1 (1.2-2.0): higher for more token-matching sensitivity
- Top-k retrieved documents (3-10): balance coverage vs. context window
- Augmentation strategy: prioritize keywords for agent-based settings; summaries for human reviewers

**Common pitfalls**:
- Using semantic retrievers without understanding agent decomposition → misaligned queries
- Over-augmentation → irrelevant noise in corpus; keep augmentation concise
- Forgetting to measure retrieval quality separately; debug retrieval independently from agent reasoning

**Scaling**: BM25 scales efficiently to millions of documents. Augmentation adds 10-30% to index size depending on augmentation richness. Recommend periodic re-ranking of top-k with LLM rerankers for improved precision.

## Reference

Paper: https://arxiv.org/abs/2602.05975
Code: Available at author's repository
Benchmark: SAGE dataset with 1,200 queries and evaluation protocols
Related work: Dense retrieval, reranking, information retrieval for agents
