---
name: beaver-prompt-compression
title: "BEAVER: A Training-Free Hierarchical Prompt Compression Method"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.19635"
keywords: [Context Compression, Prompt Engineering, Efficiency, Long-Context Language Models]
description: "Compress long prompts to 1/26th of original size while maintaining retrieval accuracy using hierarchical page-level pooling, without model fine-tuning."
---

# BEAVER: Hierarchical Prompt Compression

Long-context language models enable processing of extensive documents, but longer prompts mean higher latency, higher API costs, and degraded performance (context window saturation). Traditional compression methods aggressively prune individual tokens, destroying discourse coherence and hurting retrieval tasks that require finding specific information.

BEAVER solves this through hierarchical page-level compression: instead of pruning tokens, organize the long document into "pages" (dense semantic blocks), then select the most relevant pages based on query. This preserves local coherence (each page remains a coherent segment) while enabling dramatic overall compression (26-30x for long contexts with minimal accuracy loss).

## Core Concept

BEAVER replaces token-level pruning with structure-aware page-level selection:

**Page Formation:** Divide long context into natural semantic units (pages), each representing coherent content.

**Dual-Path Selection:** Combine semantic similarity (query-content matching) and lexical signals (rare terms) to identify pages containing query-relevant information.

**Sentence Smoothing:** Reconstruct selected pages with intermediate smoothing sentences to restore discourse flow between non-contiguous pages.

**Hardware Acceleration:** Use parallel pooling rather than sequential processing to leverage GPU efficiency.

The approach maintains the semantic structure of retrieved information while achieving >26x compression.

## Architecture Overview

- **Page Pooling Layer**: Converts variable-length documents into fixed-size dense page representations
- **Semantic Branch**: Computes query-document similarity via embeddings
- **Lexical Branch**: Identifies pages containing rare/informative terms
- **Dual Fusion**: Combines semantic and lexical signals
- **Sentence Smoothing**: Re-inserts transitional sentences between distant pages
- **Parallelizable**: All page operations can run in parallel; no sequential dependencies

## Implementation Steps

### Step 1: Convert Long Context to Pages

Divide document into manageable semantic units.

```python
import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np

class PagePooling:
    """
    Converts long document into fixed-size page representations.
    Pages are semantic units preserving local structure.
    """

    def __init__(self, page_size: int = 256, stride: int = 128):
        self.page_size = page_size  # Tokens per page
        self.stride = stride  # Overlap between pages

    def create_pages(self, document_tokens: List[int], tokenizer) -> List[dict]:
        """
        Divide document into overlapping pages.
        document_tokens: list of token IDs
        Returns: list of page dicts with {tokens, embeddings, page_idx}
        """
        pages = []
        num_pages = (len(document_tokens) - self.page_size) // self.stride + 1

        for page_idx in range(num_pages):
            start = page_idx * self.stride
            end = start + self.page_size

            page_tokens = document_tokens[start:end]
            page_text = tokenizer.decode(page_tokens)

            pages.append({
                'page_idx': page_idx,
                'tokens': page_tokens,
                'text': page_text,
                'start_pos': start,
                'end_pos': end
            })

        return pages

    def embed_pages(self, pages: List[dict], embedding_model) -> List[dict]:
        """
        Compute dense embeddings for each page.
        Uses pooling of token embeddings to create page vectors.
        """
        with torch.no_grad():
            for page in pages:
                # Get token embeddings
                token_embeds = embedding_model.encode(page['text'])

                # Pool: mean pooling across tokens
                page_embedding = torch.mean(token_embeds, dim=0)

                page['embedding'] = page_embedding

        return pages

    def get_page_importance(self, pages: List[dict]) -> np.ndarray:
        """
        Compute basic importance signal for each page (e.g., length, entropy).
        Higher importance = less likely to be pruned.
        """
        importance_scores = []

        for page in pages:
            # Importance factors:
            # 1. Rarity of words (prefer pages with rare tokens)
            # 2. Length (longer pages have more information)
            # 3. Position (first/last pages often important)

            score = 0.5  # Base score

            # Rarity score: count low-frequency tokens
            tokens = page['tokens']
            rare_token_count = sum(1 for t in tokens if t > 5000)  # Heuristic: high token ID = rare
            score += 0.3 * min(rare_token_count / 20.0, 1.0)

            # Length score: pages with more tokens
            score += 0.2 * min(len(page['tokens']) / 256.0, 1.0)

            importance_scores.append(score)

        return np.array(importance_scores)
```

### Step 2: Semantic Retrieval Branch

Select pages most relevant to query.

```python
class SemanticBranch:
    """
    Semantic similarity-based page selection.
    Identifies pages containing query-relevant information.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def retrieve_by_similarity(self, query: str, pages: List[dict],
                               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve top-k pages by semantic similarity to query.
        Returns: list of (page_idx, similarity_score)
        """
        with torch.no_grad():
            query_embedding = self.embedding_model.encode(query)

        similarities = []
        for page in pages:
            page_embedding = page['embedding']

            # Cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                page_embedding.unsqueeze(0)
            ).item()

            similarities.append((page['page_idx'], sim))

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def compute_semantic_scores(self, query: str, pages: List[dict]) -> np.ndarray:
        """
        Compute similarity score for each page (all pages, not just top-k).
        Returns: (num_pages,) array of similarity scores
        """
        with torch.no_grad():
            query_embedding = self.embedding_model.encode(query)

        scores = np.zeros(len(pages))

        for idx, page in enumerate(pages):
            page_embedding = page['embedding']
            sim = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                page_embedding.unsqueeze(0)
            ).item()
            scores[idx] = sim

        return scores
```

### Step 3: Lexical Retrieval Branch

Identify pages containing query-specific terms.

```python
import re
from collections import Counter

class LexicalBranch:
    """
    Lexical/keyword-based page selection.
    Identifies pages with rare query terms.
    """

    def __init__(self):
        self.common_words = set([
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'
        ])

    def extract_query_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Tokenize and filter stop words
        words = query.lower().split()
        keywords = [w.strip(',.!?;:') for w in words if w.lower() not in self.common_words]
        return keywords

    def compute_lexical_scores(self, query: str, pages: List[dict]) -> np.ndarray:
        """
        Score pages by presence of query keywords.
        Pages containing rare query terms get higher scores.
        """
        keywords = self.extract_query_keywords(query)
        keyword_importance = {}

        # Estimate keyword rarity (inverse document frequency)
        for keyword in keywords:
            doc_frequency = sum(1 for page in pages if keyword.lower() in page['text'].lower())
            idf = np.log(len(pages) / max(doc_frequency, 1))
            keyword_importance[keyword] = idf

        # Score each page
        scores = np.zeros(len(pages))

        for idx, page in enumerate(pages):
            page_text = page['text'].lower()

            for keyword, importance in keyword_importance.items():
                # Count keyword occurrences
                occurrences = len(re.findall(r'\b' + keyword + r'\b', page_text))
                scores[idx] += occurrences * importance

        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores
```

### Step 4: Dual-Path Fusion

Combine semantic and lexical signals for robust selection.

```python
class DualPathFusion:
    """
    Combine semantic and lexical page selection.
    Uses learned or heuristic weights to balance both signals.
    """

    def __init__(self, semantic_weight: float = 0.7, lexical_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    def fuse_scores(self, query: str, pages: List[dict],
                    semantic_branch: SemanticBranch,
                    lexical_branch: LexicalBranch) -> np.ndarray:
        """
        Compute fused importance scores combining both retrieval methods.
        """

        # Semantic scores
        semantic_scores = semantic_branch.compute_semantic_scores(query, pages)

        # Lexical scores
        lexical_scores = lexical_branch.compute_lexical_scores(query, pages)

        # Baseline importance (page-level properties)
        baseline_importance = np.array([p.get('importance', 0.5) for p in pages])

        # Fused score
        fused_scores = (
            self.semantic_weight * semantic_scores +
            self.lexical_weight * lexical_scores +
            0.1 * baseline_importance
        )

        return fused_scores

    def select_pages(self, query: str, pages: List[dict],
                     semantic_branch: SemanticBranch,
                     lexical_branch: LexicalBranch,
                     compression_ratio: float = 0.2) -> List[int]:
        """
        Select pages to retain based on fused scores.
        compression_ratio: target fraction of pages to keep (e.g., 0.2 = 20%)
        """

        fused_scores = self.fuse_scores(query, pages, semantic_branch, lexical_branch)

        # Determine number of pages to keep
        num_to_keep = max(1, int(len(pages) * compression_ratio))

        # Select top-k pages
        top_indices = np.argsort(fused_scores)[-num_to_keep:]
        selected_indices = sorted(top_indices)

        return selected_indices.tolist()
```

### Step 5: Sentence Smoothing

Restore discourse coherence when reconstructing selected pages.

```python
class SentenceSmoothing:
    """
    Restore transitional coherence between non-contiguous selected pages.
    Generates bridging sentences where large gaps exist.
    """

    def __init__(self, smoothing_model):
        self.smoothing_model = smoothing_model

    def reconstruct_with_smoothing(self, pages: List[dict], selected_indices: List[int],
                                   query: str) -> str:
        """
        Reconstruct compressed document with smoothing sentences between gaps.
        """

        reconstructed_pages = [pages[idx]['text'] for idx in selected_indices]
        result = []

        for i, idx in enumerate(selected_indices):
            result.append(reconstructed_pages[i])

            # Check for gap to next selected page
            if i < len(selected_indices) - 1:
                next_idx = selected_indices[i + 1]
                gap_size = next_idx - idx

                if gap_size > 2:  # Significant gap
                    # Generate smoothing sentence
                    smoothing_prompt = f"""
                    Given these two excerpts from a document about "{query}":

                    Previous: ...{reconstructed_pages[i][-100:]}
                    Next: {reconstructed_pages[i+1][:100]}...

                    Write a single bridging sentence that connects them naturally.
                    Be concise (under 20 words).
                    """

                    smoothing_sentence = self.smoothing_model.generate(smoothing_prompt)
                    result.append(f"\n[Connection: {smoothing_sentence}]\n")

        return '\n'.join(result)

    def compute_reconstruction_loss(self, original: str, reconstructed: str,
                                    query: str) -> float:
        """
        Estimate quality of reconstruction.
        Higher loss = more information loss.
        """

        # Simple heuristic: loss based on retrieval accuracy
        # A good reconstruction should still find answers to the query

        # Count query keywords in original vs reconstruction
        keywords = query.split()
        original_keyword_count = sum(original.lower().count(kw.lower()) for kw in keywords)
        reconstructed_keyword_count = sum(reconstructed.lower().count(kw.lower()) for kw in keywords)

        if original_keyword_count == 0:
            return 0.0

        # Keyword recall
        recall = reconstructed_keyword_count / original_keyword_count
        loss = 1.0 - recall

        return loss
```

### Step 6: Complete Compression Pipeline

Integrate all components into an end-to-end system.

```python
class BEAVERCompressor:
    """
    Full BEAVER compression pipeline.
    Training-free: works with any pre-trained LLM and embedding model.
    """

    def __init__(self, embedding_model, smoothing_model=None):
        self.page_pooling = PagePooling(page_size=256, stride=128)
        self.semantic_branch = SemanticBranch(embedding_model)
        self.lexical_branch = LexicalBranch()
        self.fusion = DualPathFusion(semantic_weight=0.7, lexical_weight=0.3)
        self.smoothing = SentenceSmoothing(smoothing_model) if smoothing_model else None

    def compress_context(self, query: str, long_context: str,
                         tokenizer, compression_ratio: float = 0.04) -> str:
        """
        Compress long context for a query.
        compression_ratio: target compression (0.04 = 26x compression)
        """

        # Step 1: Tokenize and create pages
        tokens = tokenizer.encode(long_context)
        pages = self.page_pooling.create_pages(tokens, tokenizer)

        # Step 2: Embed pages
        pages = self.page_pooling.embed_pages(pages, self.semantic_branch.embedding_model)

        # Step 3: Add baseline importance
        baseline_importance = self.page_pooling.get_page_importance(pages)
        for idx, page in enumerate(pages):
            page['importance'] = baseline_importance[idx]

        # Step 4: Select pages via dual-path fusion
        selected_indices = self.fusion.select_pages(
            query, pages, self.semantic_branch, self.lexical_branch,
            compression_ratio=compression_ratio
        )

        # Step 5: Reconstruct with smoothing
        selected_pages = [pages[idx] for idx in selected_indices]

        if self.smoothing:
            compressed = self.smoothing.reconstruct_with_smoothing(
                pages, selected_indices, query
            )
        else:
            # Simple reconstruction without smoothing
            compressed = '\n'.join([pages[idx]['text'] for idx in selected_indices])

        # Compute compression statistics
        original_length = len(long_context.split())
        compressed_length = len(compressed.split())
        actual_compression_ratio = compressed_length / original_length

        return {
            'compressed_context': compressed,
            'selected_page_indices': selected_indices,
            'original_length': original_length,
            'compressed_length': compressed_length,
            'compression_ratio': actual_compression_ratio
        }

    def evaluate_retrieval_accuracy(self, query: str, compressed_context: str,
                                   answer_in_original: bool) -> float:
        """
        Evaluate if retrieval still works on compressed context.
        Returns: 1.0 if answer still retrievable, 0.0 otherwise.
        """

        # Heuristic: check if key query terms are still in compressed context
        keywords = query.lower().split()
        keyword_coverage = sum(1 for kw in keywords if kw in compressed_context.lower()) / max(len(keywords), 1)

        return keyword_coverage
```

## Practical Guidance

**Hyperparameters:**
- Page size: 256 tokens (balance locality vs. compression efficiency)
- Compression ratio: 0.04-0.10 (26x-10x compression)
- Semantic weight: 0.7 (semantic signals usually most important)
- Lexical weight: 0.3 (lexical adds robustness to rare terms)
- Smoothing threshold: gap > 2 pages (insert bridges for significant gaps)

**When to Use:**
- Very long contexts (>10K tokens) where context window is a bottleneck
- Multi-document retrieval (query spans multiple documents)
- Latency-critical applications (API costs or response time)
- Scenarios where answer is expected to be explicitly retrievable

**When NOT to Use:**
- Short contexts (<2K tokens) where compression overhead dominates
- Tasks requiring global coherence (summarization, general understanding)
- Contexts where relevant information is distributed across all pages
- Scenarios requiring exact reproduction of original text

**Pitfalls:**
- Page overlap too small: reduces recall; pages may miss information at boundaries
- Compression ratio too aggressive: information loss exceeds acceptable thresholds
- Smoothing too verbose: bridging sentences defeat compression goal
- Keyword mismatch: if query uses synonyms, lexical branch may fail

## Reference

Paper: [arxiv.org/abs/2603.19635](https://arxiv.org/abs/2603.19635)
