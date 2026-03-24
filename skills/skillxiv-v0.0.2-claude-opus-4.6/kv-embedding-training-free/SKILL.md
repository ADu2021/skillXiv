---
name: kv-embedding-training-free
title: "KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.01046"
keywords: [Text Embeddings, Decoder-only Models, Representation Learning, KV Caching, Model Internals]
description: "Extract high-quality embeddings from frozen decoder-only LLMs by re-routing internal key-value states without training—outperforming training-free baselines by 10% on MTEB while maintaining robustness across sequences up to 4,096 tokens."
---

## Overview

KV-Embedding solves a critical limitation: while decoder-only LLMs (Qwen, Mistral, Llama) excel at generation, extracting semantic embeddings from them is challenging. Their causal attention masks early tokens from later context, and their objective (next-token prediction) biases representations toward generation over semantic compression.

**Core Innovation:** Leverage internal key-value (KV) cache states that naturally encode sequence-level information. By re-routing these states as prepended prefixes, enable all tokens to access full sequence context within a single forward pass—activating latent embedding capabilities without training.

## Problem Statement

**Causal Attention Limitation:**
Standard decoder-only attention masks are causal—early tokens cannot attend to future tokens. This is efficient for generation but problematic for embeddings.

**Next-Token Prediction Bias:**
Models optimized for generation focus on predicting the next token. Embeddings trained this way emphasize generative features (fluency, diversity) over semantic compression.

**Training Complexity:**
Typical embedding approaches require fine-tuning with contrastive losses, adding computational overhead and reducing accessibility.

## KV-Embedding Approach

### Stage 1: Extract KV States

During forward pass, capture key-value cache states from each transformer layer:

```python
def extract_kv_states(model, input_ids):
    """Extract KV cache states during forward pass."""

    hidden_states = []
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            # Forward through layer, capturing KV cache
            out, cache = layer(
                input_ids,
                attention_mask=...,
                past_key_values=...
            )
            hidden_states.append(cache)

    # KV states from final layer contain full sequence information
    return hidden_states[-1]
```

### Stage 2: Re-route as Prefix

Use final layer's KV states as a prepended prefix in a second forward pass:

**Key Insight:** The KV states of the final token encode a compressed view of the entire sequence. By prepending these states, all tokens gain access to sequence-level context.

```python
def rerout_kv_as_prefix(model, input_ids):
    """Re-route KV states as prefix for embedding extraction."""

    # Forward pass 1: Extract KV states
    kv_states = extract_kv_states(model, input_ids)

    # Prepare prefix: prepend KV states
    # This enables causal attention to access full sequence context
    prefix_kv = kv_states  # Use final token's KV cache as prefix

    # Forward pass 2: Generate with prefix
    # Now all positions can access full sequence via prefix
    embeddings = model(
        input_ids,
        past_key_values=prefix_kv,
        use_cache=False,  # Don't generate, just forward
    )

    return embeddings
```

### Stage 3: Automated Layer Selection

Different layers encode different information levels. Automatically select optimal layers:

**Intrinsic Dimensionality:** A metric indicating how much useful information is encoded in a representation.

```python
def select_optimal_layers(model, validation_texts):
    """Determine which layers provide best embeddings."""

    layer_scores = {}

    for layer_idx in range(num_layers):
        # Extract representations from specific layer
        representations = [
            extract_layer_representation(text, layer_idx)
            for text in validation_texts
        ]

        # Compute intrinsic dimensionality
        dim = compute_intrinsic_dimensionality(representations)

        # Score based on dimension and downstream task performance
        score = evaluate_on_retrieval_task(representations)
        layer_scores[layer_idx] = score

    # Select layers with best scores
    best_layers = sorted(
        layer_scores.items(), key=lambda x: x[1], reverse=True
    )[:num_select]

    return [layer_idx for layer_idx, _ in best_layers]
```

**Model-Agnostic:** Automatically selects layers for Qwen, Mistral, Llama without manual tuning.

## Performance Characteristics

**MTEB Benchmark Results:**
- Qwen backbone: +10% improvement over training-free baselines
- Mistral backbone: +8% improvement
- Llama backbone: +9% improvement
- Outperforms other training-free embedding approaches

**Sequence Length Robustness:**
- Maintains performance on sequences up to 4,096 tokens
- No degradation with long contexts
- Overcomes typical position interpolation limitations

**Computational Efficiency:**
- Two forward passes (extract KV, re-route as prefix)
- No gradient computation or fine-tuning
- Linear time complexity in sequence length

## Advantages Over Alternatives

**vs. Fine-tuned Embeddings:**
- No training required (use frozen models)
- Applicable to proprietary/API models
- Instant deployment without data collection

**vs. Simple Pooling Methods:**
- Addresses causal attention limitation
- Better semantic information extraction
- Improved performance on similarity tasks

**vs. Contrastive Learning:**
- No contrastive pairs needed
- No training overhead
- Single-pass simplicity (after KV extraction)

## Implementation Considerations

**Model Compatibility:**
Works with any HuggingFace decoder-only model with accessible KV cache:

```python
compatible_models = [
    "Qwen/Qwen2-7B",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Llama-2-7b",
    "meta-llama/Llama-3-8B",
    # And many more...
]
```

**Dimension Selection:**
Automated layer selection works across model families. For new model architectures:

```python
def auto_configure_for_model(model_name):
    """Automatically configure KV-Embedding for new model."""
    model = AutoModel.from_pretrained(model_name)

    # Validate on small subset
    validation_texts = load_validation_set()

    # Run automated layer selection
    optimal_layers = select_optimal_layers(model, validation_texts)
    dimension = optimal_layers[0]  # Use best layer's dimension

    return {"layers": optimal_layers, "dimension": dimension}
```

## When to Use KV-Embedding

**Use when:**
- Need embeddings from decoder-only LLMs (Qwen, Mistral, Llama)
- Cannot fine-tune models (frozen, proprietary APIs)
- Want to avoid training complexity and data requirements
- Building semantic search or retrieval systems
- Evaluating LLM representational capacity

**When NOT to use:**
- Specialized embedding models (BERT, Sentence-Transformers) already available
- Latency-critical applications (two forward passes add overhead)
- Scenarios where training-fine-tuned embeddings dramatically outperform
- Extremely large-scale deployments where 2x inference is prohibitive

## Research Contributions

- **KV State Interpretation:** KV cache encodes sequence-level semantic information
- **Prefix Re-routing:** Novel technique enabling causal models to access full context
- **Automated Configuration:** Intrinsic dimensionality-based layer selection
- **Comprehensive Validation:** Testing across Qwen, Mistral, Llama families

## Theoretical Insight

**Why KV States Encode Semantics:**
- In transformer computation, values (V) contain information
- Key-value states accumulate this information across positions
- Final token's KV state is a compressed view of full sequence
- Re-routing enables all positions to access this compression

## Code Availability

Implementation available with examples for major model families.

**Supported Models:**
- Qwen family (Qwen-7B through Qwen3-72B)
- Mistral family (v0.1, v0.2, v0.3)
- Llama family (Llama-2, Llama-3)
- And extensible to other decoder-only architectures

## References

- KV-Embedding achieves +10% improvement on MTEB
- Training-free approach with zero fine-tuning
- Robust to sequences up to 4,096 tokens
- Outperforms existing training-free embedding methods
