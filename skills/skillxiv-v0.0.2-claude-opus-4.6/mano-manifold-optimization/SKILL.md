---
name: mano-manifold-optimization
title: "Mano: Restriking Manifold Optimization for LLM Training"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.23000"
keywords: [Optimization, Manifold Methods, LLM Training, Convergence, Parameter Updates]
description: "Improve LLM training efficiency through manifold-based optimization that projects momentum onto tangent spaces and constrains updates on rotational Oblique manifolds. Achieves 1.75× faster convergence than Muon with reduced memory."
---

# Mano: Restriking Manifold Optimization for LLM Training

Standard optimizers like AdamW rely on diagonal curvature estimates and ignore structural properties of weight matrices, while recent manifold methods like Muon sacrifice curvature information for global spectral normalization. Mano bridges this gap by projecting updates onto tangent spaces while constraining parameters to an Oblique manifold. The key innovation is rotating normalization—alternating between column-wise and row-wise normalization across iterations—that preserves curvature while enforcing geometric constraints.

The core insight is that LLM weight matrices have natural manifold structure, and respecting this geometry during optimization improves both convergence speed and stability.

## Core Concept

Mano operates through three key mechanisms:

1. **Manifold Projection**: Updates are projected onto the tangent space of parameters, keeping the objective and solution unchanged while enforcing geometric constraints
2. **Oblique Manifold Selection**: Uses rotational Oblique manifold (yields shortest geodesic distance compared to alternatives)
3. **Rotating Normalization**: Alternates between column-wise normalization (odd iterations) and row-wise normalization (even iterations) for adaptive geometric constraints

This creates curvature-aware optimization without problem-specific assumptions.

## Architecture Overview

- **Momentum Computation**: Standard momentum accumulation in tangent space
- **Tangent Space Projection**: Project accumulated momentum onto manifold surface
- **Column-wise Normalization**: Normalize columns (odd iterations)
- **Row-wise Normalization**: Normalize rows (even iterations)
- **Update Application**: Apply constrained update to parameters
- **Learning Rate Schedule**: Standard warmup and decay (compatible with existing schedules)

## Implementation

The optimizer involves momentum projection, alternating normalization, and constrained updates.

Implement core Mano optimizer step:

```python
import torch
import torch.nn as nn

class ManoOptimizer(torch.optim.Optimizer):
    """Manifold Optimization for LLM training."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.iteration = 0

    def step(self, closure=None):
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                step = state['step']

                # Standard momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Standard second moment (for stability reference)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Manifold projection: project momentum onto tangent space
                momentum_tangent = self.project_to_tangent(
                    exp_avg, p.data, bias_correction1
                )

                # Rotating normalization based on iteration parity
                if self.iteration % 2 == 0:
                    # Even iteration: row-wise normalization
                    normalized = self.row_normalize(momentum_tangent)
                else:
                    # Odd iteration: column-wise normalization
                    normalized = self.column_normalize(momentum_tangent)

                # Update parameters with constraint
                p.data.add_(normalized, alpha=-group['lr'])

        self.iteration += 1
        return loss

    def project_to_tangent(self, momentum, param, scale):
        """Project momentum onto tangent space of parameter manifold."""
        # For Oblique manifold: tangent space is orthogonal to parameter direction
        # Tangent = momentum - (momentum · param / ||param||^2) * param

        # Compute projection coefficient
        param_norm_sq = torch.sum(param ** 2)
        momentum_dot_param = torch.sum(momentum * param)

        # Projection
        tangent = momentum - (momentum_dot_param / (param_norm_sq + 1e-8)) * param

        return tangent / scale

    def column_normalize(self, tensor):
        """Column-wise normalization."""
        # Reshape to 2D if needed
        shape = tensor.shape
        if len(shape) > 2:
            tensor_2d = tensor.view(-1, shape[-1])
        else:
            tensor_2d = tensor

        # Normalize each column
        col_norms = torch.norm(tensor_2d, dim=0, keepdim=True) + 1e-8
        normalized_2d = tensor_2d / col_norms

        if len(shape) > 2:
            return normalized_2d.view(shape)
        return normalized_2d

    def row_normalize(self, tensor):
        """Row-wise normalization."""
        # Reshape to 2D if needed
        shape = tensor.shape
        if len(shape) > 2:
            tensor_2d = tensor.view(shape[0], -1)
        else:
            tensor_2d = tensor

        # Normalize each row
        row_norms = torch.norm(tensor_2d, dim=1, keepdim=True) + 1e-8
        normalized_2d = tensor_2d / row_norms

        if len(shape) > 2:
            return normalized_2d.view(shape)
        return normalized_2d

mano_optimizer = ManoOptimizer(model.parameters(), lr=1e-3)
```

Monitor convergence and training dynamics:

```python
def analyze_mano_convergence(loss_history, gradient_norms):
    """Analyze Mano optimization dynamics."""

    # Compute convergence metrics
    loss_decay = loss_history[0] / loss_history[-1]
    gradient_variance = np.var(gradient_norms[-100:])  # Recent gradient variance

    print(f"Loss decay ratio: {loss_decay:.2f}x")
    print(f"Final gradient variance: {gradient_variance:.6f}")

    # Signal-to-noise ratio
    snr = np.mean(gradient_norms[-100:]) / np.std(gradient_norms[-100:])
    print(f"Gradient SNR: {snr:.2f}")

    return {
        "loss_decay": loss_decay,
        "gradient_variance": gradient_variance,
        "snr": snr
    }

# Validation loop
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = model(batch)
        loss.backward()

        # Record metrics
        gradient_norms.append(torch.norm(torch.cat([
            p.grad.view(-1) for p in model.parameters() if p.grad is not None
        ])).item())

        mano_optimizer.step()
        loss_history.append(loss.item())
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|-----------------|-------|
| Learning Rate | 1e-3 to 1e-4 | Same scale as AdamW |
| Beta1 (momentum) | 0.9 (standard) | Typical momentum coefficient |
| Beta2 (second moment) | 0.999 (standard) | Used for stability reference only |
| Warmup Steps | 1000-2000 | Standard linear warmup |
| Sequence Length | 2K-4K tokens | Tested at typical scales |
| Batch Size | 32-64 (per GPU) | Consistent with baseline |

**When to use**: For large-scale LLM training where efficiency matters. When you observe convergence plateau with AdamW. For vision transformer training (similar matrix structure).

**When NOT to use**: For small models (<100M) where overhead dominates. When gradient sparsity is extreme.

**Common pitfalls**:
- Alternating normalization pattern is critical—missing this reduces benefits significantly
- Learning rate sensitivity is higher than AdamW—validate carefully on your architecture
- Tangent space projection adds compute—ensure custom CUDA kernels are available
- Manifold constraint must be consistent—verify normalization operations preserve manifold structure
- Training instability can occur with very aggressive learning rates—monitor loss and gradient norms

## Reference

Mano: Restriking Manifold Optimization for LLM Training
https://arxiv.org/abs/2601.23000
