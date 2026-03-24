---
name: attntrace-context-attribution
title: AttnTrace - Efficient Context Attribution for Long-Context LLMs
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03793
keywords: [attribution, long-context, attention-analysis, interpretability]
description: "Identify influential texts in long contexts via attention weights using top-K filtering and context subsampling, achieving 10-20x speedup over perturbation methods."
---

## AttnTrace: Attention-Based Context Attribution

AttnTrace identifies which texts most influence an LLM's output by analyzing attention weights. Rather than expensive perturbation-based attribution (requiring hundreds of forward passes), it leverages the model's internal attention signals with two key innovations: filtering noise via top-K tokens and subsampling competing contexts to sharpen attention signals.

### Core Concept

Understanding which context texts influence model outputs is important for debugging and auditing. Perturbation methods—removing texts and measuring output change—are accurate but require hundreds of forward passes. AttnTrace exploits attention mechanisms: high attention weight to a text indicates influence. However, raw attention is noisy (diffuse across many tokens) and gets diluted when competing texts exist. The solution: (1) average only top-K high-attention tokens per text, and (2) subsample competing texts to concentrate attention signals.

### Architecture Overview

- **Top-K Attention Filtering**: Average attention for only highest-weight tokens per text, ignoring "attention sink" tokens
- **Context Subsampling**: Randomly subsample context, run inference, measure attention—competition among fewer texts sharpens signals
- **Multi-Subsample Averaging**: Average importance scores across multiple random subsamples
- **Theoretical Grounding**: Formal analysis proving attention weights decrease with more competing texts
- **Efficiency**: 10-20 seconds per sample vs. 100-600+ seconds for perturbation methods

### Implementation Steps

**Step 1: Extract and Analyze Attention Weights**

```python
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

class AttentionExtractor:
    """Extract attention weights during forward pass."""

    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture attention in each layer."""
        for layer in self.model.transformer.h:
            # Hook into attention module
            layer.self_attn.register_forward_hook(self._attention_hook)

    def _attention_hook(self, module, input, output):
        """Capture attention weights from forward pass."""
        # output is typically (batch, num_heads, seq_len, seq_len)
        if isinstance(output, tuple):
            attn_weights = output[0]  # First output is attention
        else:
            attn_weights = output

        self.attention_weights.append(attn_weights.detach())

    def extract(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Run forward pass and return attention weights."""
        self.attention_weights = []

        with torch.no_grad():
            _ = self.model(input_ids)

        return self.attention_weights

class TopKAttentionFilter:
    """Filter attention by top-K tokens per text."""

    def __init__(self, k: int = 5):
        self.k = k

    def filter_attention(self, attention: torch.Tensor, token_ranges: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """
        Filter attention weights to top-K tokens per text.

        Args:
            attention: (batch, num_heads, seq_len_query, seq_len_key)
            token_ranges: {text_name: (start_idx, end_idx)}

        Returns:
            importance_per_text: {text_name: score}
        """
        # Average over batch and heads
        attention_avg = torch.mean(attention, dim=(0, 1))  # (seq_len_q, seq_len_k)

        importance_per_text = {}

        for text_name, (start, end) in token_ranges.items():
            # Get attention to tokens in this text range
            text_attention = attention_avg[:, start:end]  # (seq_len_query, tokens_in_text)

            # For each query position, find top-K attended tokens in this text
            top_k_per_query = torch.topk(text_attention, k=min(self.k, end - start), dim=1)

            # Average the top-K values
            importance = torch.mean(top_k_per_query.values).item()
            importance_per_text[text_name] = importance

        return importance_per_text
```

**Step 2: Implement Context Subsampling**

```python
class ContextSubsampler:
    """Subsample competing texts to sharpen attention signals."""

    def __init__(self, model):
        self.model = model
        self.extractor = AttentionExtractor(model)

    def subsample_context(self, full_context_ids: torch.Tensor, text_ranges: Dict,
                         subsample_fraction: float = 0.7) -> torch.Tensor:
        """
        Randomly select subset of context texts.

        Args:
            full_context_ids: Full input token IDs
            text_ranges: {text_name: (token_start, token_end)}
            subsample_fraction: Fraction to keep

        Returns:
            subsampled_ids: Token IDs with some texts removed
        """
        keep_mask = torch.ones_like(full_context_ids, dtype=torch.bool)
        text_names = list(text_ranges.keys())

        # Randomly decide which texts to keep
        num_texts = len(text_names)
        num_to_keep = max(1, int(num_texts * subsample_fraction))
        texts_to_keep = np.random.choice(text_names, size=num_to_keep, replace=False)

        # Zero out tokens from removed texts
        for text_name, (start, end) in text_ranges.items():
            if text_name not in texts_to_keep:
                keep_mask[0, start:end] = False

        # Create subsampled input
        subsampled = full_context_ids * keep_mask.long()
        return subsampled

    def compute_importance_with_subsampling(self, full_context_ids: torch.Tensor,
                                          text_ranges: Dict, num_subsamples: int = 10) -> Dict[str, float]:
        """
        Compute importance scores across multiple random subsamples.
        Averaging across subsamples reduces noise and sharpens signals.
        """
        all_importance = {text: [] for text in text_ranges.keys()}

        for _ in range(num_subsamples):
            # Subsample context
            subsampled_ids = self.subsample_context(full_context_ids, text_ranges)

            # Extract attention
            attn_weights = self.extractor.extract(subsampled_ids)

            # Average attention weights across layers
            avg_attn = torch.mean(torch.stack(attn_weights), dim=0)

            # Filter to top-K per text
            filter = TopKAttentionFilter(k=5)
            importance_this_subsample = filter.filter_attention(avg_attn, text_ranges)

            # Accumulate
            for text, importance in importance_this_subsample.items():
                all_importance[text].append(importance)

        # Average across subsamples
        final_importance = {}
        for text, values in all_importance.items():
            final_importance[text] = np.mean(values)

        return final_importance
```

**Step 3: Handle Attention Dispersion**

```python
def analyze_attention_dispersion(attention: torch.Tensor, num_competing_texts: int) -> float:
    """
    Theoretical analysis: more competing texts → lower max attention.
    This explains why subsampling sharpens signals.
    """
    # Max attention value achieved
    max_attn = torch.max(attention).item()

    # Expected max with n competing texts follows extreme value distribution
    # Heuristic: max_attn ≈ 1 / num_competing_texts for uniform distribution
    expected_dilution = 1.0 / max(1, num_competing_texts)

    return expected_dilution

def compute_dilution_factor(full_context_ids: torch.Tensor, subsampled_ids: torch.Tensor) -> float:
    """
    Quantify how much attention was diluted in full context vs. subsampled.
    """
    num_texts_full = torch.sum(full_context_ids > 0).item()
    num_texts_sub = torch.sum(subsampled_ids > 0).item()

    dilution_ratio = num_texts_full / num_texts_sub
    return dilution_ratio
```

**Step 4: Implement Full Attribution Pipeline**

```python
class ContextAttributor:
    """Complete pipeline for attributing outputs to context texts."""

    def __init__(self, model):
        self.model = model
        self.subsampler = ContextSubsampler(model)

    def attribute_output(self, full_context: str, context_texts: Dict[str, str],
                        query: str, num_subsamples: int = 10) -> Dict[str, float]:
        """
        Attribute model output to source texts.

        Args:
            full_context: Complete input text
            context_texts: {text_name: text_content}
            query: Query/prompt text
            num_subsamples: Subsamples for averaging

        Returns:
            importance_scores: {text_name: importance_score}
        """
        # Tokenize
        tokenizer = self.model.tokenizer
        full_ids = tokenizer.encode(full_context, return_tensors='pt')

        # Map text ranges in tokenized form
        text_ranges = self._compute_text_ranges(full_context, context_texts, tokenizer)

        # Compute importance via subsampling
        importance = self.subsampler.compute_importance_with_subsampling(
            full_ids, text_ranges, num_subsamples=num_subsamples
        )

        # Normalize to [0, 1]
        max_importance = max(importance.values())
        if max_importance > 0:
            importance = {k: v / max_importance for k, v in importance.items()}

        return importance

    def _compute_text_ranges(self, full_text: str, text_chunks: Dict[str, str], tokenizer) -> Dict:
        """Compute token ranges for each text chunk in full context."""
        ranges = {}
        current_pos = 0

        for text_name, text_content in text_chunks.items():
            start = full_text.find(text_content, current_pos)
            if start == -1:
                continue

            end = start + len(text_content)

            # Convert char positions to token positions
            start_tokens = len(tokenizer.encode(full_text[:start]))
            end_tokens = len(tokenizer.encode(full_text[:end]))

            ranges[text_name] = (start_tokens, end_tokens)
            current_pos = end

        return ranges
```

**Step 5: Benchmark Against Perturbation**

```python
import time

def benchmark_attribution_methods(model, context_texts: Dict, query: str):
    """
    Compare AttnTrace speed vs. perturbation-based attribution.
    """
    # AttnTrace
    start = time.time()
    attributor = ContextAttributor(model)
    attntrace_scores = attributor.attribute_output(
        ' '.join(context_texts.values()),
        context_texts,
        query,
        num_subsamples=10
    )
    attntrace_time = time.time() - start

    # Perturbation (for comparison)
    start = time.time()
    perturbation_scores = {}
    for text_name in context_texts.keys():
        # Ablate this text
        ablated_context = ' '.join(
            text for name, text in context_texts.items() if name != text_name
        )
        # Measure output change (expensive!)
        perturbation_scores[text_name] = 0.5  # Placeholder

    perturbation_time = time.time() - start

    print(f"AttnTrace: {attntrace_time:.2f}s")
    print(f"Perturbation: {perturbation_time:.2f}s (estimate)")
    print(f"Speedup: {perturbation_time / attntrace_time:.1f}x")

    return attntrace_scores
```

### Practical Guidance

**When to Use:**
- Debugging long-context LLM outputs
- Identifying which facts influence specific answers
- Auditing model behavior (what's being relied on?)
- Real-time attribution during inference

**When NOT to Use:**
- Fine-grained token-level attribution (works at text level)
- Architectures without attention (RNNs, pure CNNs)
- Scenarios requiring causal ground truth (perturbation is still more rigorous)

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `k_top_tokens` | 5 | Higher = less noise but may miss important signals |
| `subsample_fraction` | 0.7 | Lower = sharper signals but fewer texts represented |
| `num_subsamples` | 10 | More = better averaging, higher computation |

### Reference

**Paper**: AttnTrace: Attention-based Context Traceback for Long-Context LLMs (2508.03793)
- 10-20x speedup vs. perturbation methods (10-20s vs. 100-600s)
- Top-K filtering removes attention sink noise
- Subsampling sharpens attention signals
