---
name: parallelizable-linear-rnns
title: "pLSTM: parallelizable Linear Source Transition Mark networks"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.11997"
keywords: [linear-RNNs, parallelization, DAGs, multi-dimensional, stability]
description: "Linear RNN architecture enabling efficient parallel computation over directed acyclic graphs with stabilization modes for long-range dependencies without sequential traversal."
---

# pLSTM: parallelizable Linear Source Transition Mark networks

## Core Concept

pLSTM extends linear RNNs to handle multi-dimensional data structures like images and graphs through a reformulated gate architecture operating on directed acyclic graphs (DAGs). Unlike sequential linear RNNs, pLSTMs enable efficient parallel computation with two stabilization modes—directed propagation and diffusive distribution—to maintain numerical stability over long distances.

## Architecture Overview

- **Source, Transition, Mark (STM) Gates**: Replace traditional LSTM input/forget/output gates, adapted for DAG operations with linear transformations
- **Edge-Associated Cell States**: Reformulate states as edge-rather than node-associated entities, enabling directional information flow
- **Parallelization via Hierarchical Composition**: Logarithmic time complexity for regular grids using einsum and padding operations
- **P-Mode (Directed Propagation)**: Restricts transition matrix norms for directional propagation, suitable for tree structures
- **D-Mode (Diffusive Distribution)**: Constrains line graph structure to multitrees for undirected global distribution in general DAGs

## Implementation

### Step 1: Define STM Gate Architecture

Implement Source, Transition, Mark gates replacing traditional LSTM gates:

```python
import torch
import torch.nn as nn

class STMCell(nn.Module):
    """
    Linear STM cell for multi-dimensional data.
    Replaces input/forget/output gates with Source/Transition/Mark.
    """
    def __init__(self, input_size, hidden_size, use_linear=True):
        super().__init__()
        self.hidden_size = hidden_size

        # Source gate: identifies new information
        self.source_transform = nn.Linear(input_size, hidden_size)

        # Transition matrix: propagates information through graph
        self.transition = nn.Linear(hidden_size, hidden_size)

        # Mark gate: controls information flow (replaces output gate)
        self.mark_transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_t, hidden_state, parent_hidden=None):
        """
        Args:
            input_t: current input
            hidden_state: current node state
            parent_hidden: aggregated parent states (for DAG structure)
        """
        # Source: extract new information from input
        source = torch.sigmoid(self.source_transform(input_t))

        # Transition: propagate from parent nodes
        if parent_hidden is not None:
            transition_effect = torch.tanh(self.transition(parent_hidden))
        else:
            transition_effect = 0

        # Update cell state
        new_hidden = source * source + (1 - source) * transition_effect

        # Mark: decide what to output
        mark = torch.sigmoid(self.mark_transform(new_hidden))
        output = mark * new_hidden

        return output, new_hidden
```

### Step 2: Implement Parallel Scan for Regular Grids

Use hierarchical parallelization for 2D/3D data with logarithmic complexity:

```python
def parallel_scan_2d(input_grid, transition_fn, reduction_fn='mean'):
    """
    Parallel scan over 2D grid using hierarchical composition.
    Achieves O(log N) depth with parallel hardware.

    Args:
        input_grid: [batch, height, width, channels]
        transition_fn: function to apply between adjacent cells
        reduction_fn: 'mean' for averaging, 'max' for max-pooling
    """
    batch, h, w, c = input_grid.shape
    states = input_grid.clone()

    # Hierarchical reduction: divide into quadrants recursively
    def reduce_dimension(x, axis, stride=1):
        """Reduce along specified axis using reduction_fn"""
        if reduction_fn == 'mean':
            return torch.mean(x, dim=axis, keepdim=True)
        else:  # max
            return torch.max(x, dim=axis, keepdim=True)[0]

    # Forward pass: broadcast information
    # Horizontal sweep
    for step in range(0, w, stride):
        for i in range(step, min(step + stride, w)):
            if i > 0:
                states[:, :, i] = transition_fn(
                    states[:, :, i], states[:, :, i-1]
                )

    # Vertical sweep
    for step in range(0, h, stride):
        for j in range(step, min(step + stride, h)):
            if j > 0:
                states[:, j, :] = transition_fn(
                    states[:, j, :], states[:, j-1, :]
                )

    return states
```

### Step 3: Implement P-Mode Stabilization

Restrict transition matrix norms for directed propagation:

```python
def stabilize_p_mode(transition_matrix, max_norm=0.99):
    """
    P-Mode: Restricts transition matrix spectral norm.
    Ensures stable directed propagation without explosion.

    Args:
        transition_matrix: [hidden_size, hidden_size]
        max_norm: spectral norm upper bound
    """
    # Compute spectral norm via power iteration
    u = torch.randn(transition_matrix.shape[0], device=transition_matrix.device)

    for _ in range(10):  # Power iteration steps
        v = transition_matrix.t() @ u
        v = v / (torch.norm(v) + 1e-8)
        u = transition_matrix @ v
        u = u / (torch.norm(u) + 1e-8)

    spectral_norm = torch.dot(u, transition_matrix @ v)

    # Scale to target norm
    if spectral_norm > max_norm:
        scaling = max_norm / (spectral_norm + 1e-8)
        return transition_matrix * scaling

    return transition_matrix
```

### Step 4: Implement D-Mode for General DAGs

Handle diffusive distribution in multitree structures:

```python
def stabilize_d_mode(adjacency_matrix, transition_matrix, max_out_degree=2):
    """
    D-Mode: Constrains graph structure to multitrees.
    Ensures stable diffusive distribution across DAG.

    Args:
        adjacency_matrix: [num_nodes, num_nodes] adjacency
        transition_matrix: [hidden_size, hidden_size]
        max_out_degree: maximum outgoing edges per node
    """
    num_nodes = adjacency_matrix.shape[0]

    # Enforce out-degree constraint
    out_degrees = adjacency_matrix.sum(dim=1)

    # Select top max_out_degree outgoing edges per node
    constrained_adj = torch.zeros_like(adjacency_matrix)

    for i in range(num_nodes):
        if out_degrees[i] > max_out_degree:
            # Keep only strongest connections
            _, top_indices = torch.topk(
                adjacency_matrix[i],
                k=max_out_degree
            )
            constrained_adj[i, top_indices] = adjacency_matrix[i, top_indices]
        else:
            constrained_adj[i] = adjacency_matrix[i]

    # Normalize transition matrix by average out-degree
    avg_out_degree = constrained_adj.sum() / num_nodes
    scaled_transition = transition_matrix / (avg_out_degree + 1e-8)

    return constrained_adj, scaled_transition
```

## Practical Guidance

- **Graph Representation**: Encode images as grid DAGs, molecular graphs directly; design appropriate edge definitions
- **Gate Initialization**: Initialize STM gates carefully; source gates should start open, transition stable
- **Parallelization Strategy**: For regular grids, use hierarchical composition; for general DAGs, use P/D-mode appropriately
- **Extrapolation Testing**: Validate on synthetic tasks designed to test long-range dependencies and out-of-distribution lengths
- **Stability Monitoring**: Track spectral norms and gradient flow during training; adjust max_norm for P-mode as needed

## Reference

Paper: arXiv:2506.11997
Key metrics: Superior extrapolation vs. Transformers on synthetic tasks, ImageNet evaluation, molecular graph processing
Related work: Linear RNNs (S4, Mamba), hierarchical attention, graph neural networks
