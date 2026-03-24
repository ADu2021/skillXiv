---
name: leank-kv-cache-pruning
title: LeanK - Learnable K Cache Channel Pruning
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.02215
keywords: [kv-cache-optimization, channel-pruning, inference-efficiency, memory-compression]
description: "Learning-based approach to reduce key-value cache memory during inference through static channel-wise sparsity masks. Achieves 70% K cache and 16-18% V cache memory reduction with 1.3x attention speedup."
---

# LeanK: Learnable K Cache Channel Pruning

## Core Concept

LeanK addresses the inference efficiency bottleneck in long-context language models by learning which key-value cache channels are unnecessary. Through a two-stage training process, the approach learns static channel-wise sparsity masks that significantly reduce memory consumption without degrading model accuracy.

## Architecture Overview

- **Channel-wise Sparsity Learning**: Identify and remove unimportant channels from the key cache through learned binary masks
- **Two-Stage Training**: Initial mask learning followed by fine-tuning with hardware constraints
- **Static Masks**: Computed once and applied uniformly across all sequences, enabling efficient deployment
- **Custom Kernels**: Optimized attention computation exploiting pruned cache structure
- **Memory-Accuracy Trade-off**: Balance between compression ratio and model performance

## Implementation Steps

### Step 1: Learn Channel Importance Scores

Use Layer-wise Relevance Propagation (LRP) or gradient-based analysis to identify important channels.

```python
def compute_channel_importance(model, calibration_data, layer_idx):
    """
    Compute importance scores for each channel in K cache.

    Args:
        model: Transformer model with attention layers
        calibration_data: Representative input samples
        layer_idx: Attention layer index to analyze

    Returns:
        Channel importance scores [num_heads, head_dim]
    """
    importance = torch.zeros(
        model.num_heads,
        model.head_dim
    )

    model.eval()
    with torch.no_grad():
        for batch in calibration_data:
            # Forward pass with gradient tracking for LRP
            outputs = model(batch, output_attentions=True)

            # Extract attention weights from specified layer
            attention = outputs.attentions[layer_idx]  # [batch, heads, seq, seq]

            # Compute gradient of loss w.r.t. K cache
            # Using attention output as proxy for importance
            for head_idx in range(model.num_heads):
                head_attention = attention[:, head_idx]  # [batch, seq, seq]
                importance[head_idx] += head_attention.sum(dim=(0, 1, 2))

    # Normalize by batch size
    importance /= len(calibration_data)

    return importance
```

### Step 2: Learn Binary Sparsity Masks with Constraints

Optimize binary masks using a relaxation-based approach with hardware alignment constraints.

```python
class ChannelSparsityLearner(torch.nn.Module):
    """
    Learn binary channel masks for K cache pruning.
    """

    def __init__(self, num_heads, head_dim, target_sparsity=0.7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.target_sparsity = target_sparsity

        # Initialize mask logits (relaxed binary values)
        self.mask_logits = torch.nn.Parameter(
            torch.ones(num_heads, head_dim)
        )

    def forward(self, k_cache):
        """
        Apply learned masks to key cache.

        Args:
            k_cache: Key cache tensor [batch, heads, seq_len, head_dim]

        Returns:
            Pruned key cache
        """
        # Sigmoid gives differentiable approximation to binary
        soft_mask = torch.sigmoid(self.mask_logits)

        # Hard mask during inference
        if not self.training:
            hard_mask = (soft_mask > 0.5).float()
        else:
            hard_mask = soft_mask

        # Apply mask to each head's channels
        pruned = k_cache * hard_mask.unsqueeze(0).unsqueeze(2)
        return pruned

    def get_sparsity_loss(self):
        """
        Compute regularization loss enforcing target sparsity.

        Returns:
            Sparsity regularization term
        """
        soft_mask = torch.sigmoid(self.mask_logits)
        actual_sparsity = 1.0 - soft_mask.mean()

        # L2 penalty on deviation from target
        sparsity_loss = (actual_sparsity - self.target_sparsity) ** 2

        return sparsity_loss

    def get_hardware_aligned_mask(self, alignment=128):
        """
        Ensure mask respects hardware alignment constraints.

        Args:
            alignment: Boundary for alignment (e.g., 128-bit)

        Returns:
            Hardware-aligned binary mask
        """
        soft_mask = torch.sigmoid(self.mask_logits)
        hard_mask = (soft_mask > 0.5).float()

        # Round kept channels to nearest multiple of alignment
        channels_kept = (hard_mask.sum(-1) * alignment).round() / alignment

        # Redistribute mask to respect alignment
        for head_idx in range(self.num_heads):
            target_channels = int(channels_kept[head_idx].item())
            # Keep top-k most important channels
            scores = soft_mask[head_idx].detach()
            _, indices = torch.topk(scores, k=target_channels)
            aligned_mask = torch.zeros_like(scores)
            aligned_mask[indices] = 1.0
            hard_mask[head_idx] = aligned_mask

        return hard_mask
```

### Step 3: Two-Stage Training Process

First stage: learn masks without constraints; second stage: fine-tune with alignment requirements.

```python
def train_channel_masks_two_stage(model, train_data, config):
    """
    Two-stage training for channel sparsity masks.

    Args:
        model: Language model with attention layers
        train_data: Training dataset
        config: Configuration with learning rates and iterations

    Returns:
        Trained model with learned masks
    """
    # Stage 1: Mask learning without hardware constraints
    learner = ChannelSparsityLearner(
        model.num_heads,
        model.head_dim,
        target_sparsity=config.target_sparsity
    )

    optimizer_stage1 = torch.optim.Adam(
        learner.parameters(),
        lr=config.learning_rate_stage1
    )

    for epoch in range(config.epochs_stage1):
        total_loss = 0

        for batch in train_data:
            # Forward pass with learned masks
            outputs = model(batch)
            reconstruction_loss = outputs.loss

            # Sparsity regularization
            sparsity_loss = learner.get_sparsity_loss()

            # Combined loss
            total_loss = reconstruction_loss + config.sparsity_weight * sparsity_loss

            optimizer_stage1.zero_grad()
            total_loss.backward()
            optimizer_stage1.step()

    # Stage 2: Fine-tune with hardware alignment
    optimizer_stage2 = torch.optim.Adam(
        learner.parameters(),
        lr=config.learning_rate_stage2
    )

    for epoch in range(config.epochs_stage2):
        for batch in train_data:
            # Get hardware-aligned mask
            aligned_mask = learner.get_hardware_aligned_mask()

            # Forward pass with aligned mask
            outputs = model(batch)
            total_loss = outputs.loss

            optimizer_stage2.zero_grad()
            total_loss.backward()
            optimizer_stage2.step()

    return learner
```

### Step 4: Integrate Pruned Cache into Inference

Replace standard attention computation with optimized kernel using pruned cache.

```python
def pruned_attention_forward(query, key, value, mask, pruned_mask):
    """
    Compute attention using pruned key cache.

    Args:
        query: Query tensor [batch, heads, seq_q, head_dim]
        key: Key tensor [batch, heads, seq_k, head_dim]
        value: Value tensor [batch, heads, seq_k, head_dim]
        mask: Attention mask
        pruned_mask: Channel sparsity mask [heads, head_dim]

    Returns:
        Attention output [batch, heads, seq_q, head_dim]
    """
    # Apply channel pruning to key
    key_pruned = key * pruned_mask.unsqueeze(0).unsqueeze(2)

    # Standard scaled dot-product attention with pruned key
    scores = torch.matmul(query, key_pruned.transpose(-2, -1)) / math.sqrt(query.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)

    # Use original (unpruned) value for output
    output = torch.matmul(attn_weights, value)

    return output
```

## Practical Guidance

### When to Use LeanK

- **Long-sequence inference**: Contexts >2K tokens where KV cache memory dominates
- **Batch processing**: Processing multiple sequences where memory savings compound
- **Memory-constrained devices**: Edge deployment or multi-model serving scenarios
- **Cost-sensitive inference**: Reducing memory enables larger batch sizes

### When NOT to Use LeanK

- **Short sequences**: <512 tokens where cache is negligible
- **Latency-critical applications**: Mask computation adds minimal overhead, but may not justify complexity
- **Fine-tuning phase**: Masks learned on fixed model; retraining requires mask relearning
- **Highly specialized domains**: Transfer learning may not preserve mask validity

### Hyperparameter Recommendations

- **Target sparsity**: 0.5-0.8 (0.7 recommended); higher sparsity risks accuracy loss
- **Stage 1 learning rate**: 1e-3 to 5e-3
- **Stage 2 learning rate**: 1e-4 to 1e-3 (lower for fine-tuning)
- **Sparsity weight**: 0.01 to 0.1; balance against reconstruction loss
- **Calibration data size**: 500-2000 examples per layer for importance scoring

### Key Insights

The two-stage approach is crucial: unconstrained learning discovers important channels, while constrained fine-tuning respects hardware alignment. Static masks enable extreme efficiency—no dynamic masking overhead at inference. The key insight is that many channels in KV cache provide redundant information; selective retention maintains performance.

## Reference

**LeanK: Learnable K Cache Channel Pruning** (arXiv:2508.02215)

Introduces static channel-wise sparsity masks for KV cache reduction through two-stage learning. Achieves significant memory savings (70% K cache reduction) while maintaining accuracy and enabling custom kernel speedups.
