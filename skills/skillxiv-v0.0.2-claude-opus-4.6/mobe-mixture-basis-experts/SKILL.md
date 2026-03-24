---
name: mobe-mixture-basis-experts
title: MoBE - Mixture-of-Basis-Experts for MoE Compression
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.05257
keywords: [model-compression, mixture-of-experts, weight-factorization, efficient-inference]
description: "Compresses MoE language models through shared basis factorization of expert weight matrices, achieving 24-30% parameter reduction with minimal accuracy loss."
---

## MoBE: Mixture-of-Basis-Experts for MoE Compression

### Core Concept

MoBE compresses large Mixture-of-Experts language models by factorizing expert weight matrices into shared basis representations. Instead of maintaining unique full-rank matrices for each expert, MoBE decomposes them into unique low-rank components combined with shared basis matrices that are reused across all experts within a layer.

### Architecture Overview

- **Expert-Specific Factorization**: Each expert's up/gate matrix decomposed as W = AB with expert-unique matrix A
- **Shared Basis Representation**: Larger matrix B expressed as linear combination of shared basis matrices {Bi}
- **Cross-Expert Sharing**: Basis matrices used across all experts within a given MoE layer
- **Lightweight Reconstruction**: Simple matrix multiplication recovers original weight dimensions

### Implementation Steps

**Step 1: Analyze Expert Weight Structures**

Examine original expert weights to determine compression targets:

```python
# Pseudocode for weight analysis
class ExpertWeightAnalyzer:
    def __init__(self, model):
        super().__init__()
        self.model = model

    def analyze_weight_distribution(self, layer_idx):
        """
        Analyze weight matrices in MoE layers for compression potential.
        """
        moe_layer = self.model.layers[layer_idx].moe

        expert_weights = []
        for expert in moe_layer.experts:
            # Get up/gate projection weights
            up_weight = expert.up_proj.weight.data
            gate_weight = expert.gate_proj.weight.data

            expert_weights.append({
                'up': up_weight,
                'gate': gate_weight,
                'shape_up': up_weight.shape,
                'shape_gate': gate_weight.shape
            })

        # Analyze redundancy across experts
        stacked_up = torch.stack([w['up'] for w in expert_weights])
        redundancy_score = compute_redundancy(stacked_up)

        return {
            'experts': expert_weights,
            'redundancy_score': redundancy_score,
            'compression_potential': 1 - (redundancy_score / len(expert_weights))
        }

    def compute_redundancy(self, weights):
        """
        Measure how similar expert weights are.
        """
        # SVD to find common structure
        U, S, V = torch.svd(weights.view(weights.shape[0], -1))
        return torch.sum(S[:10]) / torch.sum(S)  # Top 10 singular values
```

**Step 2: Decompose Expert Weight Matrices**

Factorize each expert's weights:

```python
# Pseudocode for weight decomposition
class ExpertDecomposer(nn.Module):
    def __init__(self, num_experts, input_dim, output_dim, rank=64, num_basis=8):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.num_basis = num_basis

        # Expert-specific components (low-rank)
        self.expert_factors = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, rank))
            for _ in range(num_experts)
        ])

        # Shared basis matrices
        self.basis_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(rank, output_dim))
            for _ in range(num_basis)
        ])

        # Basis selection coefficients per expert
        self.basis_coefficients = nn.Parameter(
            torch.randn(num_experts, num_basis)
        )

    def decompose_original_weights(self, original_weights):
        """
        Factorize original weights to learned factors.

        Args:
            original_weights: (num_experts, input_dim, output_dim)

        Returns:
            factors_A: (num_experts, input_dim, rank)
            basis_B: (num_basis, rank, output_dim)
            coefficients: (num_experts, num_basis)
        """
        num_experts, input_dim, output_dim = original_weights.shape

        # Initialize expert-specific factors via SVD
        factors_A = []
        for exp_idx in range(num_experts):
            U, S, V = torch.svd(original_weights[exp_idx])
            # Keep top-r singular vectors
            A = U[:, :self.rank] * torch.sqrt(S[:self.rank]).unsqueeze(0)
            factors_A.append(A)

        factors_A = torch.stack(factors_A)

        # Initialize basis via clustering expert residuals
        residuals = []
        for exp_idx in range(num_experts):
            residual = original_weights[exp_idx] - factors_A[exp_idx] @ torch.randn(
                self.rank, output_dim
            )
            residuals.append(residual)

        # Use k-means on residuals to initialize bases
        residuals_flat = torch.cat([r.view(-1) for r in residuals])
        basis_matrices = initialize_basis_via_kmeans(
            residuals_flat, self.num_basis, output_dim
        )

        return factors_A, basis_matrices

    def forward(self, expert_idx, x):
        """
        Reconstruct expert output from factorized components.

        Args:
            expert_idx: which expert to use
            x: (batch, seq_len, input_dim)

        Returns:
            output: (batch, seq_len, output_dim)
        """
        # Get expert-specific factor
        factor_A = self.expert_factors[expert_idx]  # (input_dim, rank)

        # Get basis combination weights
        coeff = self.basis_coefficients[expert_idx]  # (num_basis,)

        # Combine bases
        basis_combo = torch.zeros(
            self.rank, x.shape[-1],
            device=x.device, dtype=x.dtype
        )
        for b_idx, basis in enumerate(self.basis_matrices):
            basis_combo += coeff[b_idx] * basis

        # Reconstruct: W = A @ (sum_i coeff_i * B_i)
        output = x @ factor_A @ basis_combo
        return output
```

**Step 3: Train Decomposed Model**

Fine-tune the factorized weights to minimize reconstruction error:

```python
# Pseudocode for decomposition training
def train_decomposed_model(original_model, decomposed_model, training_data, num_epochs=5):
    """
    Train the decomposed model to match original model outputs.
    """
    optimizer = AdamW(decomposed_model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in training_data:
            input_ids = batch['input_ids']
            target_labels = batch['labels']

            # Forward pass
            decomposed_output = decomposed_model(input_ids)
            original_output = original_model(input_ids)

            # Reconstruction loss
            reconstruction_loss = F.mse_loss(
                decomposed_output.logits,
                original_output.logits
            )

            # Task loss (maintain task performance)
            task_loss = F.cross_entropy(
                decomposed_output.logits.view(-1, decomposed_output.logits.size(-1)),
                target_labels.view(-1)
            )

            total_loss = 0.7 * reconstruction_loss + 0.3 * task_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(decomposed_model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    return decomposed_model
```

**Step 4: Optimize Basis Sharing**

Refine which basis matrices are shared across expert layers:

```python
# Pseudocode for basis optimization
class BasisOptimizer:
    def __init__(self, num_layers, num_basis):
        super().__init__()
        self.num_layers = num_layers
        self.num_basis = num_basis

    def optimize_basis_sharing(self, decomposed_model):
        """
        Determine optimal basis sharing patterns across layers.
        """
        # Analyze basis similarity across layers
        basis_similarity_matrix = torch.zeros(
            self.num_layers, self.num_layers
        )

        for l1 in range(self.num_layers):
            for l2 in range(self.num_layers):
                bases_l1 = decomposed_model.layers[l1].basis_matrices
                bases_l2 = decomposed_model.layers[l2].basis_matrices

                # Compute cosine similarity between basis sets
                similarity = compute_set_similarity(bases_l1, bases_l2)
                basis_similarity_matrix[l1, l2] = similarity

        # Cluster similar layers for basis sharing
        clusters = cluster_layers(basis_similarity_matrix)

        # Create shared basis pool
        shared_bases = {}
        for cluster_id, layer_indices in enumerate(clusters):
            shared_bases[cluster_id] = merge_bases_from_layers(
                decomposed_model, layer_indices
            )

        return shared_bases

    def apply_shared_bases(self, decomposed_model, shared_bases):
        """
        Update model to use shared bases across similar layers.
        """
        for cluster_id, (layer_indices, shared_basis) in enumerate(shared_bases.items()):
            for layer_idx in layer_indices:
                decomposed_model.layers[layer_idx].basis_matrices = shared_basis

        return decomposed_model
```

### Practical Guidance

**Hyperparameters and Configuration**:
- Rank of expert-specific factors: 32-64 (depends on matrix dimensions)
- Number of shared basis matrices: 4-8 per layer
- Compression ratio: Target 24-30% parameter reduction
- Fine-tuning learning rate: 1e-4 to 5e-5
- Optimization epochs: 3-10 depending on dataset size

**When to Use MoBE**:
- Compressing very large MoE models (100B+)
- Scenarios requiring significant parameter reduction with minimal accuracy loss
- Deployment environments with storage or memory constraints
- Models where expert weight matrices show moderate to high redundancy

**When NOT to Use**:
- Small MoE models where compression provides minimal benefit
- Tasks extremely sensitive to model quality degradation
- Real-time systems where decomposition/reconstruction adds latency
- When training data is insufficient for fine-tuning

**Implementation Notes**:
- Analyze redundancy before committing to compression (not all MoE models compress equally)
- Fine-tuning is critical for maintaining accuracy
- Consider per-layer vs global basis sharing based on model structure
- Monitor expert utilization distribution (may shift with compression)
- Store basis matrices separately for potential reuse across models

### Reference

Paper: MoBE: Mixture-of-Basis-Experts for Compressing MoE LLMs
ArXiv: 2508.05257
Performance: 24-30% parameter reduction on DeepSeek-V3 and Kimi-K2-Instruct with 1-2% accuracy loss (vs 7-14% for other methods)
