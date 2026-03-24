---
name: representation-shift-token-compression
title: Representation Shift - Unified Token Importance with FlashAttention
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.00367
keywords: [token-compression, attention-pruning, inference-optimization, flashattention]
description: "Training-free token importance metric using representation shifts in MLP layers, enabling efficient compression compatible with fused kernels."
---

## Representation Shift: Token Compression Through Layer-Wise Change

Representation Shift is a training-free method for identifying and removing unimportant tokens from model inputs. Rather than relying on attention weights (which require explicit construction, incompatible with optimized kernels), it measures how much token representations change through each network layer—the intuition being critical tokens undergo larger shifts while redundant tokens stay relatively static.

### Core Concept

Token pruning accelerates inference but requires accurate importance scoring. Attention-based methods look at which tokens attend to which, but modern inference uses fused attention kernels (like FlashAttention) that never construct attention matrices, making these methods unusable. Representation Shift sidesteps this by computing token importance locally: for each token, measure L2 distance between its input and output embeddings in each MLP layer. Large shifts = important tokens. Small shifts = removable tokens.

### Architecture Overview

- **Per-Layer Representation Change**: Compute L2 norm of (output_embedding - input_embedding) for each token in each MLP layer
- **Model Agnostic**: Works on Transformers, CNNs, SSMs—any architecture with representational layers
- **FlashAttention Compatible**: Doesn't require attention matrix construction, enabling use with fused kernels
- **Training-Free**: No learnable parameters or retraining required
- **Multi-Layer Aggregation**: Combine importance scores across layers for robust ranking

### Implementation Steps

**Step 1: Compute Per-Token Representation Shifts**

Calculate importance for each token based on embedding changes:

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple

def compute_representation_shift(layer, input_embeddings, output_embeddings,
                                distance_metric='l2') -> torch.Tensor:
    """
    Compute representation shift (importance) for each token.

    Args:
        layer: Neural network layer (typically MLP)
        input_embeddings: (batch, seq_len, hidden_dim)
        output_embeddings: (batch, seq_len, hidden_dim)
        distance_metric: 'l2', 'l1', or 'cosine'

    Returns:
        importance_scores: (batch, seq_len) - importance of each token
    """
    if distance_metric == 'l2':
        # L2 norm: Euclidean distance between embeddings
        shifts = torch.norm(output_embeddings - input_embeddings, p=2, dim=-1)
    elif distance_metric == 'l1':
        # L1 norm: Manhattan distance
        shifts = torch.norm(output_embeddings - input_embeddings, p=1, dim=-1)
    elif distance_metric == 'cosine':
        # Cosine distance
        in_norm = F.normalize(input_embeddings, dim=-1)
        out_norm = F.normalize(output_embeddings, dim=-1)
        shifts = 1 - F.cosine_similarity(in_norm, out_norm, dim=-1)
    else:
        raise ValueError(f"Unknown metric: {distance_metric}")

    return shifts

def extract_mlp_features(model, input_ids, layer_indices=None):
    """
    Extract input/output pairs from MLP layers for importance computation.

    Args:
        model: Transformer model
        input_ids: (batch, seq_len)
        layer_indices: Which layers to extract features from (default: all)

    Returns:
        List of (input_features, output_features, layer_name) tuples
    """
    mlp_features = []

    # Register hooks to capture MLP layer I/O
    def hook_fn(module, input, output, layer_name):
        # input[0] is the input tensor, output is the output
        mlp_features.append((input[0].detach(), output.detach(), layer_name))

    handles = []
    for name, module in model.named_modules():
        # Hook into MLP layers specifically
        if 'mlp' in name.lower() or 'feed_forward' in name.lower():
            if layer_indices is None or any(idx in name for idx in layer_indices):
                handle = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                handles.append(handle)

    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model(input_ids)

    # Clean up hooks
    for handle in handles:
        handle.remove()

    return mlp_features

def compute_token_importance_scores(model, input_ids) -> torch.Tensor:
    """
    Compute importance score for each token across all MLP layers.
    """
    # Extract MLP layer features
    mlp_features = extract_mlp_features(model, input_ids)

    batch_size, seq_len = input_ids.shape
    all_shifts = []

    # For each MLP layer
    for input_feat, output_feat, layer_name in mlp_features:
        # Handle case where MLP output might have different shape
        if output_feat.shape != input_feat.shape:
            # Project to same dimension if needed
            output_feat = output_feat[:, :input_feat.shape[1], :input_feat.shape[-1]]

        # Compute L2 shifts (empirically found to be best)
        layer_shifts = compute_representation_shift(
            None, input_feat, output_feat, distance_metric='l2'
        )
        all_shifts.append(layer_shifts)

    # Aggregate across layers (mean of log shifts to downweight outliers)
    shifts_stacked = torch.stack(all_shifts, dim=0)  # (num_layers, batch, seq_len)
    importance = torch.mean(torch.log(shifts_stacked + 1e-6), dim=0)  # (batch, seq_len)

    return importance
```

**Step 2: Implement Token Pruning**

Remove low-importance tokens while preserving model outputs:

```python
def prune_tokens(importance_scores: torch.Tensor, prune_ratio: float = 0.2,
                 keep_special_tokens: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identify and mark tokens for removal based on importance.

    Args:
        importance_scores: (batch, seq_len)
        prune_ratio: Fraction of tokens to remove (0.0-1.0)
        keep_special_tokens: Always keep [CLS], [SEP], etc.

    Returns:
        keep_mask: (batch, seq_len) boolean mask
        pruned_indices: Indices of tokens to remove
    """
    batch_size, seq_len = importance_scores.shape

    # Create mask
    keep_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    # Always keep special tokens (first and last positions)
    if keep_special_tokens:
        keep_mask[:, 0] = True
        keep_mask[:, -1] = True

    # Identify tokens to prune (lowest importance)
    num_to_prune = int(seq_len * prune_ratio)

    for b in range(batch_size):
        scores = importance_scores[b].clone()
        # Mask out special tokens from consideration
        if keep_special_tokens:
            scores[0] = float('inf')
            scores[-1] = float('inf')

        # Find indices of tokens with lowest importance
        _, removal_indices = torch.topk(scores, num_to_prune, largest=False)
        keep_mask[b, removal_indices] = False

    return keep_mask, ~keep_mask

def apply_pruning(embeddings: torch.Tensor, attention_mask: torch.Tensor,
                  keep_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply pruning by removing low-importance tokens from embeddings and mask.

    Args:
        embeddings: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len)
        keep_mask: (batch, seq_len) boolean mask

    Returns:
        pruned_embeddings: (batch, pruned_seq_len, hidden_dim)
        pruned_mask: (batch, pruned_seq_len)
    """
    pruned_embeddings = []
    pruned_masks = []

    for b in range(embeddings.shape[0]):
        # Select only kept tokens
        keep_indices = torch.where(keep_mask[b])[0]
        pruned_embeddings.append(embeddings[b, keep_indices, :])
        pruned_masks.append(attention_mask[b, keep_indices])

    # Pad to same length for batch processing
    max_len = max(e.shape[0] for e in pruned_embeddings)
    padded_embeddings = []
    padded_masks = []

    for e, m in zip(pruned_embeddings, pruned_masks):
        padding = (0, 0, 0, max_len - e.shape[0])  # Pad seq_len dimension
        padded_embeddings.append(F.pad(e, padding))
        padded_masks.append(F.pad(m, (0, max_len - m.shape[0])))

    return torch.stack(padded_embeddings), torch.stack(padded_masks)
```

**Step 3: Integrate with Model Inference**

Apply pruning during forward pass:

```python
class PrunedModel(torch.nn.Module):
    """
    Wraps a model to apply representation shift-based token pruning.
    """
    def __init__(self, base_model, prune_ratio=0.2):
        super().__init__()
        self.base_model = base_model
        self.prune_ratio = prune_ratio

    def forward(self, input_ids, attention_mask=None, prune=True):
        """
        Forward pass with optional token pruning.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            prune: Whether to apply pruning

        Returns:
            outputs: Model outputs
            metadata: Info about pruning
        """
        if not prune:
            return self.base_model(input_ids, attention_mask)

        # Get embeddings first
        embeddings = self.base_model.embeddings(input_ids)

        # Compute importance scores (requires inspecting MLP layers)
        importance = self._compute_importance(input_ids, embeddings)

        # Prune tokens
        keep_mask, remove_mask = prune_tokens(importance, self.prune_ratio)
        pruned_embed, pruned_mask = apply_pruning(embeddings, attention_mask, keep_mask)

        # Forward through model with pruned inputs
        # (requires modifying model to accept position remapping)
        outputs = self.base_model.forward_pruned(
            embeddings=pruned_embed,
            attention_mask=pruned_mask
        )

        metadata = {
            'original_seq_len': input_ids.shape[1],
            'pruned_seq_len': pruned_embed.shape[1],
            'prune_ratio': self.prune_ratio,
            'tokens_removed': torch.sum(remove_mask).item()
        }

        return outputs, metadata

    def _compute_importance(self, input_ids, embeddings):
        """Compute token importance via representation shifts."""
        with torch.no_grad():
            # Forward through model, collecting MLP shifts
            importance = compute_token_importance_scores(self.base_model, input_ids)
        return importance
```

**Step 4: Evaluate Pruning Impact**

Measure speedup and accuracy trade-off:

```python
def evaluate_pruning(model, dataset, prune_ratios=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Evaluate effect of token pruning on accuracy and latency.
    """
    results = {
        'prune_ratio': [],
        'accuracy': [],
        'speedup': [],
        'tokens_kept': []
    }

    for prune_ratio in prune_ratios:
        pruned_model = PrunedModel(model, prune_ratio)
        accuracies = []
        latencies_pruned = []
        latencies_full = []
        tokens_kept_fracs = []

        for batch in dataset:
            input_ids = batch['input_ids']
            labels = batch['labels']

            # Full model
            import time
            start = time.time()
            full_output = model(input_ids)
            full_latency = time.time() - start

            # Pruned model
            start = time.time()
            pruned_output, metadata = pruned_model(input_ids, prune=True)
            pruned_latency = time.time() - start

            # Accuracy (assuming classification task)
            full_preds = full_output.logits.argmax(dim=-1)
            pruned_preds = pruned_output.logits.argmax(dim=-1)

            accuracy = (pruned_preds == full_preds).float().mean()
            accuracies.append(accuracy)

            latencies_full.append(full_latency)
            latencies_pruned.append(pruned_latency)
            tokens_kept_fracs.append(metadata['pruned_seq_len'] / metadata['original_seq_len'])

        results['prune_ratio'].append(prune_ratio)
        results['accuracy'].append(sum(accuracies) / len(accuracies))
        results['speedup'].append(sum(latencies_full) / sum(latencies_pruned))
        results['tokens_kept'].append(sum(tokens_kept_fracs) / len(tokens_kept_fracs))

    return results
```

**Step 5: Optimize for FlashAttention**

Ensure compatibility with fused kernels:

```python
def prune_for_flashattention(embeddings: torch.Tensor, keep_mask: torch.Tensor,
                             block_size: int = 128) -> torch.Tensor:
    """
    Prune tokens in a way compatible with FlashAttention block structure.
    FlashAttention processes in blocks; maintain aligned block structure.
    """
    batch_size, seq_len, hidden_dim = embeddings.shape

    # For each block, maintain alignment
    pruned_output = []
    current_pos = 0

    while current_pos < seq_len:
        block_end = min(current_pos + block_size, seq_len)
        block = embeddings[:, current_pos:block_end, :]
        block_mask = keep_mask[:, current_pos:block_end]

        # Apply pruning within block
        for b in range(batch_size):
            keep_indices = torch.where(block_mask[b])[0] + current_pos
            pruned_output.append(embeddings[b, keep_indices, :])

        current_pos = block_end

    return torch.cat(pruned_output, dim=0)
```

### Practical Guidance

**When to Use:**
- Inference speedup with variable sequence lengths
- Video-text retrieval (5.5× speedup observed)
- Dense retrieval tasks (4.4× speedup on QA)
- Scenarios where model compression is needed without retraining

**When NOT to Use:**
- Tasks sensitive to small output changes
- Models with unusual layer structures
- Real-time systems requiring <1ms per token latency
- Scenarios where token order depends on previous pruning (dynamic scheduling)

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `prune_ratio` | 0.2 | Fraction of tokens removed; higher = more speedup, higher accuracy loss |
| `distance_metric` | l2 | L2 norm outperforms L1 and cosine empirically |
| `mlp_layer_focus` | all | Can focus on specific layers or all; all provides most robust scoring |
| `keep_special_tokens` | True | Always preserve [CLS], [SEP] to maintain semantic markers |

### Reference

**Paper**: Representation Shift: Unifying Token Compression with FlashAttention (2508.00367)
- Training-free approach: no learnable parameters
- 5.5× speedup on video-text retrieval
- 4.4× speedup on video QA
- Compatible with fused attention kernels (FlashAttention)
- Works across Transformers, CNNs, and state space models
