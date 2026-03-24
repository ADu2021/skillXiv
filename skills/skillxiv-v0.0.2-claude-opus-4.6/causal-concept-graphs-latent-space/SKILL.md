---
name: causal-concept-graphs-latent-space
title: "Causal Concept Graphs in LLM Latent Space for Stepwise Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.10377"
keywords: [Interpretability, Causal Analysis, LLM, Mechanistic Interpretability, Reasoning]
description: "Extract sparse causal concept graphs from LLM activations using SAE and DAGMA, then validate through ablation to identify causally influential features. Bridges mechanistic interpretability with causal inference for understanding reasoning flow."
---

# Technique: Learning Causal Structures in Monosemantic Concept Space

Understanding how LLMs perform reasoning requires identifying not just *what* features matter, but *how* they causally interact. This approach extracts sparse, interpretable concepts via Sparse Autoencoders (SAEs), learns a directed acyclic graph (DAG) over them using DAGMA, then validates graph edges through targeted interventions to ensure structural causality rather than mere correlation.

This bridges mechanistic interpretability with causal inference, enabling step-by-step analysis of reasoning processes.

## Core Concept

The method operates in three stages:

1. **Monosemantic Concept Extraction**: SAE with TopK gating produces consistently sparse concept activations (e.g., 13 active per example)

2. **DAG Learning via DAGMA**: Linear structural equation model over top-64 concepts, enforcing acyclicity with matrix exponentials

3. **Causal Fidelity Validation**: Ablation experiments measuring how concept removal affects downstream behavior, distinguishing causal from spurious edges

## Architecture Overview

- **SAE backbone**: Encoder-decoder architecture with TopK gating
- **Concept bank**: 1000s of learned concept vectors, sparsely activated
- **DAG learner (DAGMA)**: Linear structural equation model with acyclicity constraint
- **Intervention system**: Ablation mechanism for causal fidelity scoring
- **Attribution analysis**: Identify high-centrality causal concepts per reasoning step

## Implementation Steps

### Step 1: Extract Sparse Monosemantic Concepts

Use SAE with TopK gating to maintain consistent sparsity without magnitude shrinkage.

```python
import torch
import torch.nn as nn

class SAEWithTopKGating(nn.Module):
    def __init__(self, activation_dim=2048, num_concepts=1024, k_active=13):
        super().__init__()
        self.activation_dim = activation_dim
        self.num_concepts = num_concepts
        self.k_active = k_active

        # Encoder: project to concept space
        self.encoder = nn.Linear(activation_dim, num_concepts)

        # Decoder: reconstruct activation from concepts
        self.decoder = nn.Linear(num_concepts, activation_dim)

    def forward(self, activations):
        """
        activations: (batch, seq_len, activation_dim)
        returns: reconstructed, sparsity_metric
        """
        batch_size, seq_len, _ = activations.shape

        # Encode to concept space
        concept_logits = self.encoder(activations)  # (batch, seq_len, num_concepts)

        # TopK gating: select exactly k_active strongest concepts
        topk_values, topk_indices = torch.topk(
            concept_logits,
            k=self.k_active,
            dim=-1
        )

        # Create sparse activation tensor
        sparse_concepts = torch.zeros_like(concept_logits)
        sparse_concepts.scatter_(-1, topk_indices, topk_values)

        # Decode: reconstruct from sparse concepts
        reconstructed = self.decoder(sparse_concepts)

        # Sparsity metric: L0 norm (number of active concepts)
        sparsity = self.k_active / self.num_concepts

        return reconstructed, sparse_concepts, sparsity
```

### Step 2: Learn DAG Structure via DAGMA

Fit a linear structural equation model enforcing acyclicity using matrix exponential characterization.

```python
import numpy as np
from scipy.linalg import expm

class DAGLearner(nn.Module):
    def __init__(self, num_concepts=64, learning_rate=0.01):
        super().__init__()
        self.num_concepts = num_concepts

        # Adjacency matrix for the DAG
        self.B = nn.Parameter(torch.zeros(num_concepts, num_concepts))

        self.learning_rate = learning_rate

    def forward(self, sparse_concepts):
        """
        Learn causal structure from concept activations.

        sparse_concepts: (batch, seq_len, num_concepts)
        """
        batch_size, seq_len, _ = sparse_concepts.shape

        # Reshape: (batch * seq_len, num_concepts)
        X = sparse_concepts.reshape(-1, self.num_concepts)

        # Structural equation model: X = X @ B + epsilon
        # Where B is the adjacency matrix (causal effects)
        residual = X - X @ self.B.t()

        # Reconstruction loss
        mse_loss = (residual ** 2).mean()

        # Acyclicity constraint: trace(exp(B)) should be close to num_concepts
        # This ensures no cycles exist
        B_np = self.B.detach().cpu().numpy()
        exp_B = expm(B_np)
        trace_val = np.trace(exp_B)

        acyclicity_penalty = ((trace_val - self.num_concepts) ** 2) * 0.01

        total_loss = mse_loss + acyclicity_penalty

        return total_loss, self.B

    def enforce_acyclicity(self):
        """Ensure acyclicity via matrix exponential constraint."""
        with torch.no_grad():
            B_np = self.B.detach().cpu().numpy()
            exp_B = expm(B_np)
            trace_val = np.trace(exp_B)

            # If trace too high, scale down weights
            if trace_val > self.num_concepts + 1:
                scale = 0.99 * (self.num_concepts / trace_val)
                self.B.mul_(scale)
```

### Step 3: Causal Fidelity Scoring via Ablation

Validate edges through interventions: measure how removing high-centrality concepts affects outputs.

```python
def compute_causal_fidelity_score(
    model,
    concept_graph,
    layer_idx,
    num_ablations=100
):
    """
    Measure whether graph edges correspond to true causal effects.

    Compares downstream effect of ablating high-centrality concepts
    vs random concepts.
    """
    B = concept_graph.B.detach()

    # Compute concept centrality
    in_degree = B.sum(dim=0)
    out_degree = B.sum(dim=1)
    centrality = in_degree + out_degree

    # Identify high-centrality concepts
    high_centrality_indices = torch.argsort(centrality, descending=True)[:10]

    fidelity_scores = []

    for _ in range(num_ablations):
        # Sample high-centrality and random concepts
        high_idx = high_centrality_indices[torch.randint(0, len(high_centrality_indices), (1,))].item()
        random_idx = torch.randint(0, B.shape[0], (1,)).item()

        # Measure output change when ablating
        output_change_high = measure_output_change_on_ablation(
            model,
            layer_idx,
            high_idx
        )

        output_change_random = measure_output_change_on_ablation(
            model,
            layer_idx,
            random_idx
        )

        # Fidelity: high-centrality concepts should have larger output effect
        fidelity = output_change_high - output_change_random
        fidelity_scores.append(fidelity)

    fidelity_score = torch.stack(fidelity_scores).mean()

    return fidelity_score

def measure_output_change_on_ablation(model, layer_idx, concept_idx):
    """Measure downstream effect of ablating a concept."""
    # Run model with concept ablated
    # Compute difference in downstream activations
    # Return as magnitude of change
    pass  # Implementation specific to model architecture
```

### Step 4: Graph-Based Reasoning Attribution

Use learned graph to attribute multi-step reasoning to concept interactions.

```python
def attribute_reasoning_to_concepts(
    model,
    input_text,
    sae,
    dag_learner,
    num_reasoning_steps
):
    """
    Trace concept activation patterns through reasoning.
    """
    reasoning_trace = []

    # Extract activations at each step
    for step in range(num_reasoning_steps):
        # Get activations from model
        activations = extract_layer_activations(model, input_text)

        # Encode to concept space
        _, sparse_concepts, _ = sae(activations)

        # Identify active concepts
        active_indices = torch.nonzero(sparse_concepts.sum(dim=1) > 0)

        # Look up causal relationships
        B = dag_learner.B.detach()

        # Find high-effect concepts (those causing others to activate)
        concept_effects = []
        for concept_idx in active_indices:
            outgoing_effect = B[concept_idx].sum()
            incoming_effect = B[:, concept_idx].sum()
            total_effect = outgoing_effect + incoming_effect

            concept_effects.append({
                'concept': concept_idx.item(),
                'effect': total_effect.item(),
                'is_source': outgoing_effect.item() > incoming_effect.item()
            })

        reasoning_trace.append({
            'step': step,
            'active_concepts': active_indices.tolist(),
            'concept_effects': concept_effects
        })

    return reasoning_trace
```

## Practical Guidance

**When to Use:**
- Interpreting multi-step reasoning in LLMs
- Debugging model failures to understand which reasoning pathways failed
- Identifying "reasoning shortcuts" or spurious correlations
- Validating mechanistic hypotheses about model computation

**When NOT to Use:**
- Real-time inference (causal analysis is offline)
- Extremely sparse models where concepts don't cleanly separate
- Tasks requiring absolute performance guarantees (interpretability doesn't guarantee correctness)

**Hyperparameter Tuning:**
- **k_active (TopK gating)**: 10-20 concepts; more reveals finer structure
- **num_concepts (SAE size)**: 500-2000; larger = finer-grained but harder to interpret
- **num_concepts_in_dag**: 32-128 of top concepts; analyze densest areas
- **acyclicity_penalty weight**: 0.01-0.1; balance with reconstruction

**Common Pitfalls:**
- SAE not trained to convergence (concepts remain noisy)
- DAG learning getting stuck in poor local minima (use multiple restarts)
- Over-interpreting discovered edges (validate with ablations)
- Confusing high centrality with causal importance (fidelity score is the ground truth)

## Reference

[Causal Concept Graphs paper on arXiv](https://arxiv.org/abs/2603.10377)
