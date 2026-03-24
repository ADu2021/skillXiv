---
name: recall-hierarchical-merging
title: "RECALL: Catastrophic-forgetting Alleviation via Hierarchical Model Merging"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.20479"
keywords: [Continual Learning, Model Merging, Catastrophic Forgetting, Representations]
description: "Prevents catastrophic forgetting in continual learning by merging models using layer-wise hidden representations as similarity proxies. Shallow layers preserve domain-general features while deep layers enable task-specific adaptation, enabling seamless multi-domain integration without task labels or historical data."
---

# RECALL: Hierarchical Merging for Continual Learning

Continual learning faces a fundamental trade-off: retain past knowledge while learning new tasks. RECALL exploits layer-wise architectural differences to resolve this by merging representations hierarchically, using hidden states as reliable knowledge proxies.

The approach enables seamless knowledge integration across multiple domains without accessing historical data or task identities.

## Core Concept

Key insight: **Layer depth correlates with knowledge type**:
- Shallow layers encode domain-general, transferable features
- Deep layers capture task-specific, specialized knowledge

RECALL leverages this by:
- Computing inter-model similarity using layer-wise hidden representations
- Adapting fusion strategy per layer (preserve shallow, specialize deep)
- Merging parameters with layer-dependent weightings

## Architecture Overview

- Layer-wise hidden representation clustering on typical samples
- Similarity computation using cosine distance in representation space
- Adaptive parameter fusion with layer-dependent coefficients
- Support for multi-domain integration without task boundaries

## Implementation Steps

Compute layer-wise representations on a small set of typical examples. These representations capture what each model learned at different depths:

```python
def compute_layer_representations(model, sample_batch, layer_indices=None):
    """Extract hidden representations at specified layers."""
    representations = {}
    hooks = []

    def get_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            representations[layer_name] = output.detach().cpu()
        return hook

    # Register hooks at each layer
    if layer_indices is None:
        layer_indices = range(len(list(model.children())))

    for idx in layer_indices:
        layer = list(model.children())[idx]
        hook = layer.register_forward_hook(get_hook(f"layer_{idx}"))
        hooks.append(hook)

    # Forward pass to capture representations
    with torch.no_grad():
        _ = model(sample_batch)

    # Cleanup
    for hook in hooks:
        hook.remove()

    return representations
```

Compute inter-model similarity at each layer using representation distance. Models with similar representations at shallow layers preserve general knowledge:

```python
def compute_layer_similarity(model1_reps, model2_reps):
    """Compute similarity between models at each layer."""
    similarities = {}

    for layer_name in model1_reps.keys():
        rep1 = model1_reps[layer_name].reshape(len(model1_reps[layer_name]), -1)
        rep2 = model2_reps[layer_name].reshape(len(model2_reps[layer_name]), -1)

        # Cosine similarity between representations
        rep1_norm = rep1 / (np.linalg.norm(rep1, axis=1, keepdims=True) + 1e-8)
        rep2_norm = rep2 / (np.linalg.norm(rep2, axis=1, keepdims=True) + 1e-8)

        cos_sim = np.mean(np.sum(rep1_norm * rep2_norm, axis=1))
        similarities[layer_name] = cos_sim

    return similarities
```

Merge parameters with layer-dependent adaptive weighting. Shallow layers favor old knowledge, deep layers favor new:

```python
def adaptive_parameter_fusion(base_model, new_model, layer_similarities, num_layers):
    """Fuse parameters with layer-dependent weights."""
    merged_model = copy_model(base_model)

    # Compute layer-dependent fusion weights
    # Shallow layers: preserve base (high alpha = more base)
    # Deep layers: favor new (low alpha = more new)
    layer_weights = {}
    for layer_idx in range(num_layers):
        # Linear schedule: 0.9 shallow, 0.1 deep
        depth_ratio = layer_idx / (num_layers - 1)
        alpha = 0.9 - 0.8 * depth_ratio  # 0.9 to 0.1

        # Adjust by similarity: high similarity allows more merging
        similarity = layer_similarities.get(f"layer_{layer_idx}", 0.5)
        alpha *= similarity

        layer_weights[f"layer_{layer_idx}"] = alpha

    # Merge parameters
    for (name, base_param), (_, new_param) in zip(
        base_model.named_parameters(),
        new_model.named_parameters()
    ):
        # Determine layer from parameter name
        layer_id = extract_layer_id(name)
        alpha = layer_weights.get(layer_id, 0.5)

        # Adaptive fusion
        merged_model.state_dict()[name] = (
            alpha * base_param + (1 - alpha) * new_param
        )

    return merged_model
```

## Practical Guidance

| Component | Recommendation |
|-----------|-----------------|
| Typical sample size | 100-500 examples capturing task diversity |
| Similarity threshold | 0.6-0.8 (high = preserve more base knowledge) |
| Shallow layer weight | 0.8-0.95 (preserve domain-general features) |
| Deep layer weight | 0.1-0.3 (favor task specialization) |

**When to use:**
- Continual learning with sequential tasks
- Scenarios without access to historical data
- Multi-domain integration without task boundaries
- Models where layer depth correlates with abstraction level

**When NOT to use:**
- Single-task learning (no merging needed)
- Architectures with unclear layer semantics (RNNs, attention-heavy)
- When exact task labels are available (task-specific methods may be better)

**Common pitfalls:**
- Unrepresentative typical samples (biased similarity computation)
- Identical weights across layers (ignores layer specialization)
- Merging without validation (degraded performance on either task)
- Assuming shallow layers are always domain-general (task-dependent)

Reference: [RECALL on arXiv](https://arxiv.org/abs/2510.20479)
