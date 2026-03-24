---
name: grape-group-position-encoding
title: "Group Representational Position Encoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.07805
keywords: [position embeddings, group actions, rotary embeddings, long-context transformers, attention mechanisms]
description: "Unify positional encoding methods via group action theory, encompassing RoPE and ALiBi as special cases. GRAPE enables exploration of cross-subspace feature coupling—ideal when you need principled positional encoding beyond standard implementations."
---

## Overview

GRAPE provides a unified mathematical framework for positional encoding based on group actions. It encompasses two mechanism families: multiplicative rotations in SO(d) and additive logit biases from unipotent actions in GL. This unification reveals that RoPE and ALiBi are special cases of a broader framework, enabling theoretical understanding and novel extensions.

## When to Use

- Designing new positional encoding methods beyond RoPE and ALiBi
- Understanding theoretical properties of position encodings
- Exploring cross-subspace feature coupling in attention
- Research into positional encoding mechanisms
- Applications requiring principled positional encoding foundations

## When NOT to Use

- Standard RoPE implementations already meeting needs
- Applications where existing methods work adequately
- Scenarios where theoretical elegance doesn't improve practical performance
- Real-time inference requiring minimal overhead

## Core Technique

Group action theory for unified positional encoding:

```python
# Group Representational Position Encoding
class GroupRepresentationalPositionalEncoding:
    def __init__(self, dim, max_seq_len=2048):
        self.dim = dim
        self.max_seq_len = max_seq_len

    # Multiplicative GRAPE using SO(d) rotations
    def multiplicative_grape(self, x, positions):
        """
        Multiplicative position-dependent rotation matrices.
        Uses rank-2 skew-symmetric generators for efficient computation.
        """
        batch_size, seq_len, dim = x.shape

        # Generate skew-symmetric matrices
        # L is skew-symmetric generator (antisymmetric)
        omega = self.compute_omega(dim)  # Base frequency matrix
        L = self.compute_skew_symmetric(omega)

        # Matrix exponential: G(n) = exp(n * omega * L)
        # For position n, compute rotation matrix
        rotation_matrices = []
        for position in range(seq_len):
            # Matrix exponential: e^(n * L)
            exp_matrix = torch.linalg.matrix_exp(position * L)
            rotation_matrices.append(exp_matrix)

        rotation_matrices = torch.stack(rotation_matrices)

        # Apply rotations to input
        # Group composition property: rotations compose naturally
        output = torch.einsum(
            'nij, bsj -> bsi',
            rotation_matrices,
            x
        )

        # Preserves relative positional information through group composition
        return output

    def compute_skew_symmetric(self, omega):
        """
        Construct skew-symmetric generator from frequency matrix.
        Ensures antisymmetric property: L^T = -L
        """
        # Build from rank-2 components for efficiency
        # Skew-symmetric matrices have zero diagonal
        dim = omega.shape[0]
        L = torch.zeros((dim, dim))

        # Fill upper triangle with omega
        for i in range(dim):
            for j in range(i+1, dim):
                L[i, j] = omega[i, j]
                L[j, i] = -omega[i, j]

        return L

    # Additive GRAPE using unipotent actions in GL
    def additive_grape(self, logits, positions):
        """
        Additive logit biases from unipotent actions.
        Recovers existing methods (ALiBi, Forgetting Transformer) as special cases.
        """
        # Unipotent action: U(n) = I + n * A where A is nilpotent
        # Creates additive position-dependent bias pattern

        batch_size, seq_len, dim = logits.shape

        # Compute position-dependent bias
        position_bias = torch.zeros(seq_len, seq_len)

        for q_pos in range(seq_len):
            for k_pos in range(seq_len):
                # Unipotent action: additive bias as function of distance
                distance = q_pos - k_pos

                # ALiBi is special case: linear slope with distance
                # ali_bias = -|distance| * slope

                # Forgetting Transformer is another special case:
                # exponential decay with distance

                # General unipotent: polynomial in distance
                bias_value = self.compute_unipotent_bias(distance)
                position_bias[q_pos, k_pos] = bias_value

        # Add bias to logits
        # Broadcasting: (batch, seq, seq) + (seq, seq)
        output_logits = logits + position_bias.unsqueeze(0)

        return output_logits

    def compute_unipotent_bias(self, distance):
        """
        Compute bias from unipotent group action.
        Enables parametrized interpolation between ALiBi, Forgetting, and novel methods.
        """
        if distance == 0:
            return 0.0

        # Interpolation parameter: controls behavior
        # linear (ALiBi), exponential (Forgetting), or polynomial
        slope = 0.1  # Can be learned or adjusted

        # Linear (ALiBi-like)
        linear_bias = -abs(distance) * slope

        # Exponential decay (Forgetting-like)
        exp_bias = -slope * (1.0 - torch.exp(torch.tensor(distance * 0.1)))

        # Mixture (novel approach)
        alpha = 0.5  # Interpolation weight
        mixed_bias = alpha * linear_bias + (1 - alpha) * exp_bias

        return mixed_bias

    # Cross-subspace feature coupling
    def cross_subspace_coupling(self, x, head_dim):
        """
        Explore feature coupling across different subspaces.
        Extended from basic dimension splitting.
        """
        batch_size, seq_len, total_dim = x.shape
        num_heads = total_dim // head_dim

        # Split into heads
        heads = x.reshape(batch_size, seq_len, num_heads, head_dim)

        # Apply position encoding within heads
        encoded_heads = []
        for head_idx in range(num_heads):
            head = heads[:, :, head_idx, :]

            # Position encoding for this head
            positions = torch.arange(seq_len)
            if head_idx % 2 == 0:
                # Multiplicative for even heads
                encoded = self.multiplicative_grape(head, positions)
            else:
                # Additive for odd heads
                encoded = self.additive_grape(head, positions)

            encoded_heads.append(encoded)

        # Cross-coupling: mix features across heads
        output = self.couple_heads(encoded_heads, head_dim)

        return output

    def couple_heads(self, heads, head_dim):
        """
        Couple features across multiple heads.
        Enables exploration of inter-head positional patterns.
        """
        # Stack heads
        stacked = torch.stack(heads, dim=2)
        batch_size, seq_len, num_heads, head_dim = stacked.shape

        # Learn cross-head coupling weights
        coupling_matrix = self.compute_coupling_weights(num_heads)

        # Apply coupling
        coupled = torch.einsum(
            'ij, bslj -> bsli',
            coupling_matrix,
            stacked
        )

        # Flatten back to (batch, seq, total_dim)
        output = coupled.reshape(batch_size, seq_len, -1)

        return output
```

Mathematical framework leverages group composition for relative positional information while maintaining streaming efficiency.

## Key Results

- Unifies RoPE and ALiBi as special cases
- Enables learned commuting subspaces
- Supports non-commuting mixtures at manageable computational cost (O(d) to O(rd) per head)
- Provides principled foundation for positional encoding exploration
- Closed-form solutions for common cases

## Implementation Notes

- Multiplicative GRAPE: O(d) or O(rd) for learnable versions
- Additive GRAPE: O(seq_len²) for logit biases
- Group composition preserves relative positions naturally
- Special cases (RoPE, ALiBi) emerge from framework constraints
- Cross-subspace coupling enables novel feature patterns

## References

- Original paper: https://arxiv.org/abs/2512.07805
- Focus: Theoretical foundations of positional encoding
- Domain: Transformer architecture, attention mechanisms
