---
name: beyondweb-synthetic-pretraining
title: "BeyondWeb: Scaling Synthetic Data for Trillion-scale Pretraining"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.10975
keywords: [synthetic-data, pretraining, data-generation, scaling, training-efficiency]
description: "Generate high-quality synthetic training data that enables 7.7x faster training than web data, with smaller models achieving better performance through strategic content rephasing and data optimization."
---

# BeyondWeb: Scaling Synthetic Data for Trillion-scale Pretraining

## Core Concept

Large language models are increasingly trained on trillion-token datasets, yet web-sourced data quality plateaus and data diversity issues emerge. BeyondWeb addresses this by generating optimized synthetic training data that enables significantly faster training and better sample efficiency than raw web data.

The key insight is that synthetic data quality depends on multiple joint factors: which content gets rephrased, how rephrasing is performed, data mixture optimization, and alignment with target model sizes. Strategic synthetic data generation can reduce training time by 7.7x compared to equivalent web data.

## Architecture Overview

- **Selective Content Rephasing**: Identify and rephrase high-value content rather than synthesizing uniformly
- **Model-Aware Data Generation**: Optimize synthetic data mixture based on target model size and capacity
- **Quality Optimization Pipeline**: Multi-stage filtering and refinement to ensure synthetic content matches natural distribution
- **Training Efficiency Focus**: Maximize tokens-per-second and final performance per training token
- **Controlled Diversity**: Balance content diversity with quality consistency across domains

## Implementation Steps

### 1. Identify High-Value Source Content

Not all content benefits equally from rephrasing. Identify sources that deserve synthetic augmentation.

```python
import numpy as np
from collections import defaultdict

class ContentValueAnalyzer:
    """
    Analyze which content sources should be rephrased for synthetic expansion
    """
    def __init__(self, min_tokens=1000, min_quality=0.6):
        self.min_tokens = min_tokens
        self.min_quality = min_quality

    def score_content(self, content, source_stats=None):
        """
        Score content for synthetic expansion value.
        High-value content is: coherent, useful, under-represented
        """
        # Token count
        tokens = len(content.split())
        token_score = min(tokens / self.min_tokens, 1.0)

        # Semantic coherence (rough estimate)
        coherence = self._estimate_coherence(content)

        # Domain uniqueness (if we have source stats)
        uniqueness = 0.8  # Default moderate uniqueness
        if source_stats:
            domain_frequency = source_stats.get('domain_frequency', 1.0)
            uniqueness = 1.0 / (1.0 + domain_frequency)

        # Combined score
        value_score = 0.4 * token_score + 0.35 * coherence + 0.25 * uniqueness
        return max(0.0, min(1.0, value_score))

    def _estimate_coherence(self, content):
        """
        Estimate document coherence (sentence-to-sentence consistency)
        """
        sentences = content.split('.')
        if len(sentences) < 3:
            return 0.5

        # Check for repeated key concepts (indicates coherence)
        words = content.lower().split()
        unique_words = len(set(words))
        word_repetition = 1.0 - (unique_words / len(words))

        # Good coherence has moderate repetition (0.2-0.5)
        coherence = 1.0 - abs(word_repetition - 0.35) / 0.35
        return max(0.0, min(1.0, coherence))

    def filter_for_rephrasing(self, documents, top_fraction=0.3):
        """
        Select top documents for rephrasing
        """
        scores = [(doc, self.score_content(doc)) for doc in documents]
        scores.sort(key=lambda x: x[1], reverse=True)

        cutoff_idx = max(1, int(len(scores) * top_fraction))
        selected = [doc for doc, score in scores[:cutoff_idx]]

        return selected, [score for _, score in scores[:cutoff_idx]]
```

### 2. Implement Content Rephasing Pipeline

Rephrase selected content using models or rules to create diverse training examples.

```python
class ContentRephaser:
    """
    Rephrase content to create synthetic training data
    """
    def __init__(self, model=None):
        self.model = model  # LLM for rephrasing (optional)

    def rephrase(self, content, rephrase_style='technical', num_variations=3):
        """
        Generate rephrased versions of content
        """
        if self.model:
            return self._llm_rephrase(content, rephrase_style, num_variations)
        else:
            return self._rule_based_rephrase(content, num_variations)

    def _llm_rephrase(self, content, style, num_variations):
        """
        Use LLM to rephrase content in specified style
        """
        prompt = f"""Rephrase the following content in a {style} style.
Maintain accuracy and key information but vary wording and structure.

Original: {content[:500]}

Rephrase (different style):"""

        rephrased = []
        for _ in range(num_variations):
            # Generate with temperature for diversity
            output = self.model.generate(
                prompt,
                max_length=len(content.split()) + 50,
                temperature=0.7,
                top_p=0.9
            )
            rephrased.append(output)

        return rephrased

    def _rule_based_rephrase(self, content, num_variations):
        """
        Rule-based rephrasing: synonym replacement, structure variation
        """
        import random
        import nltk
        from nltk.tokenize import sent_tokenize

        rephrased_docs = []

        for variation_idx in range(num_variations):
            sentences = sent_tokenize(content)

            # Variation 1: Reorder sentences (keep logical flow)
            if variation_idx == 0 and len(sentences) > 2:
                # Swap adjacent sentences carefully
                reordered = self._reorder_sentences(sentences)
                rephrased_docs.append(' '.join(reordered))
            # Variation 2: Substitute synonyms
            elif variation_idx == 1:
                synonyms_subbed = self._substitute_synonyms(content)
                rephrased_docs.append(synonyms_subbed)
            # Variation 3: Compress and expand
            else:
                compressed = self._compress_expand(content)
                rephrased_docs.append(compressed)

        return rephrased_docs

    def _reorder_sentences(self, sentences):
        """Reorder sentences while maintaining coherence"""
        # Keep first sentence, shuffle middle, keep last
        if len(sentences) <= 2:
            return sentences

        first = [sentences[0]]
        last = [sentences[-1]]
        middle = sentences[1:-1]

        # Only reorder if it's not breaking logical flow (heuristic)
        if len(middle) > 1:
            import random
            random.shuffle(middle)

        return first + middle + last

    def _substitute_synonyms(self, text):
        """Replace words with synonyms"""
        # Simplified: just replace common words
        synonyms = {
            'important': 'crucial',
            'large': 'big',
            'show': 'demonstrate',
            'use': 'employ',
        }

        result = text
        for word, syn in synonyms.items():
            result = result.replace(word, syn)
        return result

    def _compress_expand(self, text):
        """Compress or expand text"""
        sentences = text.split('.')
        if len(sentences) < 3:
            return text

        # Combine short sentences or split long ones
        combined = []
        for i, sent in enumerate(sentences[:-1]):
            if len(sent.split()) < 15 and i + 1 < len(sentences) - 1:
                # Combine with next
                combined.append(sent + '. ' + sentences[i + 1])
            else:
                combined.append(sent)

        return '. '.join(combined) + '.'
```

### 3. Filter and Quality-Check Synthetic Data

Ensure synthetic data quality through filtering and validation.

```python
class SyntheticDataQualityFilter:
    """
    Filter synthetic data to maintain quality
    """
    def __init__(self, similarity_threshold=0.85, quality_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold

    def score_synthetic(self, original, synthetic):
        """
        Score synthetic document quality
        """
        # Check 1: Semantic preservation (should be 70-90% similar)
        similarity = self._compute_similarity(original, synthetic)

        # Check 2: Length preservation (shouldn't be too different)
        length_ratio = len(synthetic.split()) / len(original.split())
        length_score = 1.0 - abs(length_ratio - 1.0)
        length_score = max(0.0, min(1.0, length_score))

        # Check 3: Repetition avoidance (shouldn't have repeated phrases)
        repetition_score = self._score_repetition_freedom(synthetic)

        # Combined score
        quality_score = 0.5 * similarity + 0.25 * length_score + 0.25 * repetition_score
        return quality_score

    def _compute_similarity(self, text1, text2):
        """
        Compute semantic similarity between texts
        Using simple word overlap (TF-IDF in production)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _score_repetition_freedom(self, text):
        """
        Score how much synthetic avoids repeating original phrases
        """
        from collections import Counter

        # Extract 3-grams
        words = text.lower().split()
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]

        # Count repetition
        trigram_counts = Counter(trigrams)
        repeated = sum(1 for count in trigram_counts.values() if count > 1)

        # Score: fewer repeated trigrams is better
        repetition_score = 1.0 - (repeated / len(trigrams)) if trigrams else 1.0
        return max(0.0, min(1.0, repetition_score))

    def filter_batch(self, original_docs, synthetic_docs):
        """
        Filter synthetic documents, keeping only high-quality ones
        """
        filtered = []
        scores = []

        for orig, synth in zip(original_docs, synthetic_docs):
            score = self.score_synthetic(orig, synth)
            scores.append(score)

            if score >= self.quality_threshold:
                filtered.append(synth)

        return filtered, scores
```

### 4. Mix Synthetic and Real Data Strategically

Create optimal data mixtures based on model size and capacity.

```python
class DataMixOptimizer:
    """
    Optimize the mixture of synthetic and real data for a target model
    """
    def __init__(self):
        self.model_size_categories = {
            'small': (1e9, 3e9),        # 1-3B
            'medium': (3e9, 10e9),      # 3-10B
            'large': (10e9, 100e9),     # 10-100B
            'xlarge': (100e9, float('inf'))  # 100B+
        }

    def optimal_synthetic_fraction(self, model_size, total_tokens=180e9):
        """
        Determine optimal fraction of synthetic data based on model size.
        Smaller models benefit more from high-quality synthetic data.
        """
        # Categorize model
        category = None
        for cat, (min_size, max_size) in self.model_size_categories.items():
            if min_size <= model_size < max_size:
                category = cat
                break

        # Synthetic fraction by category (tuned empirically)
        synthetic_fractions = {
            'small': 0.5,      # Small models: 50% synthetic
            'medium': 0.3,     # Medium: 30% synthetic
            'large': 0.15,     # Large: 15% synthetic
            'xlarge': 0.05     # XLarge: 5% synthetic
        }

        fraction = synthetic_fractions.get(category, 0.1)
        return fraction

    def create_mixed_dataset(self, web_documents, synthetic_documents,
                            model_size, total_tokens):
        """
        Create mixed dataset with optimal proportions
        """
        synthetic_fraction = self.optimal_synthetic_fraction(model_size)
        real_fraction = 1.0 - synthetic_fraction

        # Calculate token counts
        synthetic_tokens = int(total_tokens * synthetic_fraction)
        real_tokens = int(total_tokens * real_fraction)

        # Build dataset
        mixed_dataset = []

        # Add synthetic data
        synthetic_data = synthetic_documents[:len(synthetic_documents)]
        mixed_dataset.extend(synthetic_data)

        # Add real (web) data
        real_data = web_documents[:len(web_documents)]
        mixed_dataset.extend(real_data)

        return mixed_dataset, {
            'synthetic_tokens': synthetic_tokens,
            'real_tokens': real_tokens,
            'synthetic_fraction': synthetic_fraction
        }
```

### 5. Training with Synthetic Data

Train models using the optimized mixed dataset.

```python
def train_with_synthetic_data(model, mixed_dataset, model_size, num_tokens=180e9,
                              batch_size=256, num_epochs=1):
    """
    Train model on mixed synthetic + real data
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    tokens_seen = 0
    total_loss = 0.0
    num_batches = 0

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(get_data_loader(mixed_dataset, batch_size)):
            # Forward pass
            input_ids = batch['input_ids']
            labels = batch['labels']

            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                labels.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track progress
            tokens_seen += input_ids.shape[0] * input_ids.shape[1]
            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Loss: {loss:.4f}, Tokens: {tokens_seen / 1e9:.2f}B / {num_tokens / 1e9:.2f}B")

            if tokens_seen >= num_tokens:
                return model, total_loss / num_batches

        scheduler.step()

    return model, total_loss / num_batches
```

### 6. Evaluate and Compare

Benchmark synthetic-trained models against web-trained baselines.

```python
def evaluate_model(model, benchmarks=None):
    """
    Evaluate model on standard benchmarks
    """
    if benchmarks is None:
        benchmarks = ['hellaswag', 'mmlu', 'arc', 'wikitext']

    results = {}

    for benchmark in benchmarks:
        # Load benchmark
        dataset = load_benchmark(benchmark)

        # Evaluate
        accuracy = 0.0
        for sample in dataset:
            # Get model prediction
            prompt = sample['prompt']
            logits = model(prompt).logits
            pred = logits.argmax(dim=-1)

            # Check correctness
            if pred == sample['label']:
                accuracy += 1.0

        accuracy /= len(dataset)
        results[benchmark] = accuracy

    return results
```

## Practical Guidance

### Hyperparameters & Configuration

- **Synthetic Fraction**: 5-50% depending on model size (smaller models: higher fraction)
- **Rephrase Variations**: 2-4 per original document (diminishing returns after 4)
- **Quality Threshold**: 0.65-0.75 for filtering synthetic data
- **Similarity Threshold**: 0.75-0.85 (maintain semantic preservation)
- **Batch Size**: 256-512 (depends on hardware)

### When to Use BeyondWeb Approach

- Training at trillion-token scale where data diversity is limited
- You want faster training without sacrificing final performance
- Smaller models need sample efficiency (synthetic helps 1-3B models most)
- You have high-quality content that benefits from rephrasing
- Computational efficiency is critical

### When NOT to Use BeyondWeb

- You only have low-quality source content (garbage in, garbage out)
- You need domain-specific knowledge that's not in training data
- Your task requires up-to-date information (synthetic is static)
- You have unlimited access to diverse, high-quality web data
- Inference speed is critical and model size can't increase

### Common Pitfalls

1. **Over-Rephasing**: Generating too many variations reduces diversity benefits. Limit to 2-4 per doc.
2. **Poor Source Selection**: If you rephrase low-quality content, synthetic data is also low-quality. Score carefully.
3. **Ignoring Model Size**: Same synthetic fraction hurts large models. Adjust mixture by model capacity.
4. **No Quality Control**: Unfiltered synthetic data introduces noise. Maintain filtering thresholds.
5. **No Baseline Comparison**: Always compare against pure web data to verify improvement.

## Reference

BeyondWeb (2508.10975): https://arxiv.org/abs/2508.10975

Strategic synthetic data generation enables 7.7x faster training than web data, with smaller models surpassing larger web-trained baselines when using optimized synthetic mixtures tailored to model capacity.
