---
name: indexcache-sparse-attention-acceleration
title: "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.12201"
keywords: [Sparse Attention, Acceleration, Long Context, Inference, Indexing]
description: "Accelerate sparse attention by reusing token selection indices across layers. Partition layers into full indexer (F) and shared (S) types using greedy search or multi-layer distillation to eliminate 75% of indexer computation."
---

# Technique: Cross-Layer Index Sharing for Sparse Attention Efficiency

Sparse attention mechanisms use indexers to select the most relevant tokens at each layer, but this indexing computation becomes expensive for long contexts. IndexCache observes that token selections are highly correlated across consecutive layers, enabling index reuse: most layers can skip indexing and reuse indices from a nearby layer, dramatically reducing compute.

The approach combines training-free discovery (greedy search) with optional training-aware optimization (multi-layer distillation) to determine optimal sharing patterns.

## Core Concept

IndexCache partitions transformer layers into two types:

1. **F (Full) layers**: Retain their indexer, computing fresh top-k token selections
2. **S (Shared) layers**: Skip indexer computation, reusing cached indices from the nearest preceding F layer

By carefully choosing which layers retain indexers using either greedy search or learned patterns, the method eliminates 50-75% of indexer operations while maintaining sparse attention quality.

## Architecture Overview

- **Indexer computation**: Produces top-k token selections per layer
- **Index cache**: Stores recent indices for sharing across layers
- **Layer partition pattern**: Binary string determining F vs S layer types
- **Training-free search**: Greedy algorithm to find good patterns
- **Multi-layer distillation**: Optional training to adapt to sharing structure

## Implementation Steps

### Step 1: Analyze Index Correlation Across Layers

Quantify how similar token selections are across consecutive layers.

```python
import torch
import torch.nn as nn

def measure_index_correlation(model, input_ids, sample_size=1000):
    """
    Measure Jaccard similarity of top-k selections across layers.
    """
    batch_size, seq_len = input_ids.shape
    num_layers = model.config.num_hidden_layers
    k = 100  # Assuming top-100 sparse attention

    correlations = []

    with torch.no_grad():
        # Forward pass, collecting indices
        indices_by_layer = []

        for layer_idx in range(num_layers):
            # Get top-k indices for this layer
            # (Implementation depends on sparse attention mechanism)
            layer_indices = extract_topk_indices(
                model,
                input_ids,
                layer_idx,
                k
            )  # (batch, seq_len, k)

            indices_by_layer.append(layer_indices)

        # Compute Jaccard similarity between consecutive layers
        for layer_idx in range(num_layers - 1):
            indices_curr = indices_by_layer[layer_idx]  # (batch, seq_len, k)
            indices_next = indices_by_layer[layer_idx + 1]

            # Convert to sets for Jaccard
            jaccard_scores = []
            for b in range(batch_size):
                for pos in range(seq_len):
                    set_curr = set(indices_curr[b, pos].tolist())
                    set_next = set(indices_next[b, pos].tolist())

                    intersection = len(set_curr & set_next)
                    union = len(set_curr | set_next)
                    jaccard = intersection / (union + 1e-8)
                    jaccard_scores.append(jaccard)

            avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
            correlations.append(avg_jaccard)

    return correlations
```

### Step 2: Training-Free Greedy Pattern Search

Find optimal layer partition using greedy search without training.

```python
def greedy_pattern_search(
    model,
    calibration_data,
    num_layers,
    min_f_layers=2,
    max_iterations=1000
):
    """
    Greedily search for good F/S layer partition using calibration loss.

    Returns: binary string where '1'=F (full indexer), '0'=S (shared)
    """
    # Start with all F layers
    best_pattern = '1' * num_layers
    best_loss = evaluate_pattern(model, calibration_data, best_pattern)

    for iteration in range(max_iterations):
        improved = False

        # Try flipping each position from F to S
        for pos in range(num_layers):
            if best_pattern[pos] == '1':
                # Try converting this F layer to S
                new_pattern = best_pattern[:pos] + '0' + best_pattern[pos+1:]

                # Ensure minimum F layers
                if new_pattern.count('1') < min_f_layers:
                    continue

                # Evaluate new pattern
                new_loss = evaluate_pattern(model, calibration_data, new_pattern)

                if new_loss < best_loss:
                    best_pattern = new_pattern
                    best_loss = new_loss
                    improved = True
                    break

        if not improved:
            break  # Converged to local optimum

    return best_pattern

def evaluate_pattern(model, calibration_data, pattern):
    """
    Evaluate how well a layer partition preserves attention quality.
    """
    total_loss = 0

    with torch.no_grad():
        for input_ids, labels in calibration_data:
            # Forward with pattern applied
            outputs = model(input_ids)

            # Compute calibration loss (e.g., perplexity)
            loss = compute_loss(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(calibration_data)
```

### Step 3: Index Cache Management During Inference

Implement efficient index caching during forward pass.

```python
class IndexCacheManager:
    def __init__(self, pattern, num_layers):
        """
        pattern: binary string, '1'=F layer, '0'=S layer
        """
        self.pattern = pattern
        self.num_layers = num_layers
        self.cached_indices = None

    def get_indices_for_layer(self, layer_idx, indexer_fn):
        """
        Get token indices for a layer, using cache if applicable.

        indexer_fn: function to compute indices (called only for F layers)
        """
        if self.pattern[layer_idx] == '1':  # F layer
            # Compute fresh indices
            indices = indexer_fn()
            # Cache for potential sharing
            self.cached_indices = indices
            return indices
        else:  # S layer
            # Reuse cached indices
            if self.cached_indices is None:
                raise ValueError("No cached indices available")
            return self.cached_indices

def apply_index_cache_to_model(model, pattern):
    """
    Modify model to use index caching pattern.
    """
    cache_manager = IndexCacheManager(pattern, model.config.num_hidden_layers)

    # Patch sparse attention layers
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'indexer'):
            original_indexer = layer.attention.indexer

            def make_indexer_wrapper(idx, mgr, orig_fn):
                def indexer_wrapper(query_states):
                    return mgr.get_indices_for_layer(
                        idx,
                        lambda: orig_fn(query_states)
                    )
                return indexer_wrapper

            layer.attention.indexer = make_indexer_wrapper(
                layer_idx,
                cache_manager,
                original_indexer
            )

    return model
```

### Step 4: Training-Aware Distillation (Optional)

Fine-tune retained indexers to better serve shared layers via multi-layer distillation.

```python
class MultiLayerDistillation(nn.Module):
    def __init__(self, model, pattern, num_distillation_steps=1000):
        super().__init__()
        self.model = model
        self.pattern = pattern
        self.num_distillation_steps = num_distillation_steps

    def train_indexers(self, training_data, learning_rate=1e-4):
        """
        Train retained indexers against attention distributions of all served layers.
        """
        optimizer = torch.optim.Adam(
            [p for layer_idx, layer in enumerate(self.model.layers)
             if self.pattern[layer_idx] == '1'
             for p in layer.attention.indexer.parameters()],
            lr=learning_rate
        )

        for step in range(self.num_distillation_steps):
            total_loss = 0

            for batch_data in training_data:
                input_ids, _ = batch_data

                # Forward pass to get attention distributions
                with torch.no_grad():
                    full_outputs = self.model(input_ids)
                    target_attention = full_outputs.attentions  # All layer attentions

                # Forward with index cache enabled
                outputs = self.model(input_ids)
                distilled_attention = outputs.attentions

                # KL divergence between distributions
                loss = torch.nn.functional.kl_div(
                    torch.log_softmax(distilled_attention, dim=-1),
                    torch.softmax(target_attention, dim=-1),
                    reduction='batchmean'
                )

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}: Loss = {total_loss / len(training_data)}")

        return self.model
```

## Practical Guidance

**When to Use:**
- Long-context inference (20K+ tokens) with sparse attention
- Latency-critical deployments where indexing overhead is bottleneck
- Models using learnable token selection mechanisms (e.g., Lightning Attention)
- Scenarios where 1-2% quality drop for 40% speedup is acceptable trade-off

**When NOT to Use:**
- Short contexts (<10K tokens) where indexing isn't dominant cost
- Tasks requiring perfect attention recovery
- Fully dense attention models (index reuse doesn't apply)

**Hyperparameter Tuning:**
- **min_f_layers**: Usually 2-4; too few reduces quality, too many increases latency
- **search iterations**: 100-1000; more thorough but slower
- **distillation_lr**: 1e-4 typical; adjust based on convergence
- **calibration set size**: 100-500 examples; must be representative

**Common Pitfalls:**
- Pattern found via search poorly generalizes to out-of-distribution data
- Greedy search getting stuck in local optima (try multiple random seeds)
- Index cache not cleared between sequences (causes interference)
- Under-training distillation, retaining poor indexers

## Reference

[IndexCache paper on arXiv](https://arxiv.org/abs/2603.12201)
