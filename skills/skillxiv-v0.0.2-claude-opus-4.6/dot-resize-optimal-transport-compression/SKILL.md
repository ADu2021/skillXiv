---
name: dot-resize-optimal-transport-compression
title: "DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.04517"
keywords: [Model Compression, Optimal Transport, Neuron Merging, LLM Pruning, Width Reduction]
description: "Compress LLMs by 20-30% in width while preserving functionality through optimal transport-based neuron merging. Instead of discarding neurons, redistribute their signal to retained neurons via learned transport maps. Use when you need to reduce model size with minimal accuracy loss and measurable computational speedup."
---

# DOTResize: Optimal Transport-Based Neuron Merging for LLM Compression

Traditional pruning discards low-importance neurons, wasting the signal they carry. DOTResize reframes width reduction as a discrete optimal transport problem: instead of deletion, the method computes optimal mappings from full-width neurons to a reduced subset, redistributing information through learned transformation matrices. This approach preserves functional equivalence by leveraging QR decomposition to maintain RMSNorm invariance in Transformers.

The key innovation is computing transport maps based on neuron activation similarities, then using QR factorization to decompose the transport matrix into components that preserve normalization properties in residual networks. This enables dropping 20-30% of neurons with measurable real-world speedup rather than just parameter reduction.

## Core Concept

The method optimizes a discrete optimal transport problem where source neurons map to target neurons based on activation correlations. The Sinkhorn algorithm with entropy regularization solves this efficiently. A critical insight is that orthogonal mappings preserve RMSNorm, but more general transport maps require QR decomposition: decompose the transport matrix T = QR where Q is orthogonal (preserves norm) and R handles scaling.

The learned transformation pairs are then applied at sublayer outputs and absorbed into adjacent weight matrices, allowing the model to function with fewer neurons without retraining the LLM backbone.

## Architecture Overview

- **Transport Map Computation**: Sinkhorn-regularized optimization minimizing pairwise activation distances with entropy constraint, computing optimal assignment of source to target neurons
- **QR Decomposition Module**: Factorizes transport matrix T into orthogonal Q and upper triangular R for norm preservation across RMSNorm layers
- **Attention Transformation**: Applies paired transformations {M_A, M_A^inv} at attention sublayer outputs, absorbing into Q,K,V projections
- **Feed-Forward Transformation**: Applies paired transformations {M_F, M_F^inv} at FFN outputs, absorbing into down-projection weights
- **Calibration-based Optimization**: Uses small calibration dataset to compute accurate transport maps without full retraining

## Implementation

### Transport Map Computation via Sinkhorn Algorithm

The Sinkhorn algorithm solves the optimal transport problem with entropy regularization, balancing accuracy and computational efficiency.

```python
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class OptimalTransportCompressor:
    def __init__(self, source_dim, target_dim, regularization_lambda=0.1):
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.lambda_reg = regularization_lambda

    def compute_cost_matrix(self, source_activations, target_activations):
        """
        Compute pairwise L1 distances between source and target neuron activations.

        Args:
            source_activations: (calib_samples, source_dim) from full model
            target_activations: (calib_samples, target_dim) from reduced model

        Returns:
            cost_matrix: (source_dim, target_dim) pairwise L1 distances
        """
        batch_size, source_dim = source_activations.shape
        _, target_dim = target_activations.shape

        # Normalize for scale invariance
        source_norm = F.normalize(source_activations, p=2, dim=0)
        target_norm = F.normalize(target_activations, p=2, dim=0)

        # Compute pairwise L1 distances
        cost = torch.zeros(source_dim, target_dim)
        for i in range(source_dim):
            for j in range(target_dim):
                cost[i, j] = torch.abs(source_norm[:, i:i+1] - target_norm[:, j:j+1]).sum()

        return cost

    def sinkhorn_algorithm(self, cost_matrix, num_iterations=100, tolerance=1e-6):
        """
        Solve optimal transport via Sinkhorn-Knopp algorithm with entropy regularization.

        Minimizes: <T, C> - λ*H(T)
        where H(T) = -∑T_ij log(T_ij) is entropy, encouraging soft assignment.

        Args:
            cost_matrix: (source_dim, target_dim) cost matrix
            num_iterations: iterations of Sinkhorn algorithm
            tolerance: convergence threshold

        Returns:
            transport_matrix: (source_dim, target_dim) soft assignment matrix
        """
        source_dim, target_dim = cost_matrix.shape
        device = cost_matrix.device

        # Initialize with uniform probabilities
        T = torch.ones(source_dim, target_dim, device=device) / target_dim

        # Entropy regularization constant
        K = torch.exp(-self.lambda_reg * cost_matrix)

        # Margin constraints (how much from each source)
        a = torch.ones(source_dim, device=device) / source_dim
        # How much to each target (uniform)
        b = torch.ones(target_dim, device=device) / target_dim

        for iteration in range(num_iterations):
            # Sinkhorn scaling iterations
            u = a / (K @ torch.ones(target_dim, device=device) + 1e-8)
            v = b / (K.T @ u + 1e-8)

            T = u.unsqueeze(1) * K * v.unsqueeze(0)

            # Check convergence
            if iteration % 10 == 0:
                marginal_err = torch.abs(T.sum(dim=1) - a).max()
                if marginal_err < tolerance:
                    break

        return T

    def compute_transport_map(self, source_acts, target_acts):
        """
        Compute optimal transport mapping from source to target neurons.
        """
        cost = self.compute_cost_matrix(source_acts, target_acts)
        transport_map = self.sinkhorn_algorithm(cost)
        return transport_map
```

The Sinkhorn algorithm efficiently finds soft assignments that balance cost minimization with smoothness, avoiding hard assignments that cause gradient issues.

### QR Decomposition for RMSNorm Preservation

Transport maps preserve functional equivalence by decomposing into orthogonal and scaling components.

```python
def qr_decomposed_transport(transport_matrix):
    """
    Decompose transport matrix T = QR for norm preservation.

    Q (orthogonal) preserves norms through RMSNorm layers.
    R (upper triangular) handles scaling adjustments.

    Args:
        transport_matrix: (source_dim, target_dim) optimal transport matrix

    Returns:
        Q: (source_dim, target_dim) orthogonal component
        R: (target_dim, target_dim) upper triangular scaling
    """
    # PyTorch QR decomposition
    Q, R = torch.linalg.qr(transport_matrix)

    # Q is orthogonal: preserves norms, safe for RMSNorm
    # R is upper triangular: handles dimension reduction scaling

    return Q, R

def apply_transport_with_qr(hidden_states, transport_matrix):
    """
    Apply transport mapping while preserving norm properties.

    Args:
        hidden_states: (batch, seq_len, source_dim)
        transport_matrix: (source_dim, target_dim)

    Returns:
        transformed: (batch, seq_len, target_dim) with norm properties preserved
    """
    Q, R = qr_decomposed_transport(transport_matrix)

    # Apply orthogonal transformation (preserves norm)
    transformed = hidden_states @ Q  # (batch, seq_len, target_dim)

    # Scale by inverse of R to maintain activation magnitudes
    # R is upper triangular, compute R^-1 efficiently
    R_inv = torch.linalg.inv(R)
    scaled = transformed @ R_inv

    return scaled
```

### Integration into Transformer Blocks

Apply transformations at attention and FFN layer boundaries, absorbing matrices into adjacent weights.

```python
class CompressedAttentionBlock(torch.nn.Module):
    def __init__(self, original_attention, transport_map_attention):
        super().__init__()
        self.attention = original_attention
        self.transport_map = transport_map_attention
        self.Q_att, self.R_att = qr_decomposed_transport(transport_map_attention)

    def forward(self, hidden_states):
        # Original attention computation
        attn_output = self.attention(hidden_states)  # (batch, seq, orig_dim)

        # Apply transport mapping at output
        compressed = attn_output @ self.Q_att  # (batch, seq, target_dim)

        # Store for inverse application in feed-forward (skip connection)
        return compressed

class CompressedFFNBlock(torch.nn.Module):
    def __init__(self, original_ffn, transport_map_ffn):
        super().__init__()
        self.ffn = original_ffn
        self.transport_map = transport_map_ffn
        self.Q_ffn, self.R_ffn = qr_decomposed_transport(transport_map_ffn)

    def forward(self, hidden_states):
        # Original FFN computation
        ffn_output = self.ffn(hidden_states)  # (batch, seq, orig_dim)

        # Apply transport mapping
        compressed = ffn_output @ self.Q_ffn  # (batch, seq, target_dim)

        return compressed

def absorb_transport_into_weights(layer, transport_map, position='output'):
    """
    Absorb transformation matrix into layer weights for efficient inference.

    Args:
        layer: torch.nn.Linear layer to modify
        transport_map: (source_dim, target_dim) transport matrix
        position: 'input' or 'output' where to apply transformation

    Returns:
        Modified layer with absorbed transformation
    """
    Q, R = qr_decomposed_transport(transport_map)

    if position == 'output':
        # Modify weight: W_new = W_old @ Q @ R^-1
        R_inv = torch.linalg.inv(R)
        layer.weight = torch.nn.Parameter(
            layer.weight @ Q @ R_inv
        )
        layer.out_features = Q.shape[1]

    elif position == 'input':
        # Modify weight: W_new = Q @ R @ W_old
        layer.weight = torch.nn.Parameter(
            Q @ R @ layer.weight
        )
        layer.in_features = Q.shape[0]

    return layer
```

### Calibration-Based Optimization Workflow

Use a small calibration dataset to compute transport maps without full retraining.

```python
def compress_llm_with_optimal_transport(
    model,
    calibration_data,
    target_width_ratio=0.75,  # Keep 75% of neurons
    calibration_samples=128
):
    """
    Compress LLM using optimal transport without retraining.

    Args:
        model: Original LLM model
        calibration_data: Small dataset (128-256 sequences) for calibration
        target_width_ratio: Fraction of neurons to retain
        calibration_samples: Tokens to use for activation collection

    Returns:
        Compressed model with transport maps absorbed
    """
    device = next(model.parameters()).device
    compressor = OptimalTransportCompressor(regularization_lambda=0.1)

    # Collect activations on calibration data
    model.eval()
    all_source_acts = []
    all_target_acts = []

    with torch.no_grad():
        for batch in calibration_data:
            input_ids = batch['input_ids'].to(device)

            # Forward pass, collecting layer outputs
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            all_source_acts.append(hidden_states[-1].reshape(-1, hidden_states[-1].shape[-1]))

    source_activations = torch.cat(all_source_acts, dim=0)[:calibration_samples]

    # Determine target dimension
    source_dim = source_activations.shape[1]
    target_dim = int(source_dim * target_width_ratio)

    # Compute target neuron activations (select top variance neurons)
    neuron_variance = source_activations.var(dim=0)
    _, top_indices = torch.topk(neuron_variance, target_dim)
    target_activations = source_activations[:, top_indices]

    # Compute optimal transport map
    transport_map = compressor.compute_transport_map(source_activations, target_activations)

    # Apply to model
    for layer in model.transformer.h:
        # Compress attention output projection
        absorb_transport_into_weights(layer.attn.out_proj, transport_map, position='output')

        # Compress FFN output projection
        absorb_transport_into_weights(layer.mlp.down_proj, transport_map, position='output')

    return model
```

## Practical Guidance

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Width Reduction Target | 20-30% | 70-80% neuron retention; larger reductions hurt accuracy |
| Sinkhorn λ (Regularization) | 0.1 | Controls entropy regularization strength; 0.01-1.0 range works |
| Calibration Samples | 128-256 | Minimum for stable transport computation; 128 token sequences |
| Cost Metric | L1 distance | L1 more robust than L2 for outlier neurons |
| Effective Rank Threshold | 99% variance | Select target neurons with cumulative 99% activation variance |
| Sinkhorn Iterations | 100 | Convergence typically achieved in 50-100 iterations |

### When to Use

- Reducing inference latency for already-deployed LLMs with acceptable accuracy trade-off
- Fitting larger models into memory-constrained environments (VRAM, edge devices)
- Post-hoc model compression when retraining is infeasible
- Scenario where real-world speedup (not just parameter reduction) matters

### When NOT to Use

- Fine-tuning scenarios; retraining may reverse compression benefits
- Models already optimized for inference (quantized, distilled); compression compounds
- Specialized domains where neurons have fine-grained semantic roles (code generation, reasoning)
- Requiring zero performance loss; 1-5% drop is typical depending on compression ratio

### Common Pitfalls

- **Insufficient calibration data**: Using <64 token samples leads to noisy cost matrices; collect at least 128 diverse tokens
- **Mismatched cost metrics**: Switching from L1 to L2 mid-optimization changes convergence properties; be consistent
- **Ignoring heterogeneous layer widths**: Applying uniform width reduction across all layers causes bottlenecks; compute per-layer optimal targets
- **Absorbing inversions incorrectly**: When absorbing R^-1 into weights, ensure numerical stability with cond(R) checks; add regularization if cond(R) > 1e8
- **Over-reducing at upper layers**: Early transformer layers are more critical; reduce later layers more aggressively for better overall accuracy

## Reference

Chen, Y., Liu, S., Wang, Z., et al. (2024). DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging. arXiv preprint arXiv:2507.04517.
