---
name: gradient-grouping-learning-rate-scaling
title: "Taming LLMs by Scaling Learning Rates with Gradient Grouping"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.01049"
keywords: [Optimization, Learning Rate Scheduling, Adaptive Methods, Training Stability]
description: "Improve adaptive learning rates by clustering gradient statistics within layers and applying cluster-specific scaling."
---

# SGG: Fine-Grained Learning Rate Control Without Manual Tuning

Training large language models is sensitive to learning rate: too high causes divergence, too low wastes computation. Most adaptive optimizers apply the same scaling globally across all parameters, missing structure in how different parts of the network train. SGG (Scaling with Gradient Grouping) clusters parameters within each layer into groups based on gradient statistics, then applies tailored scaling per group. This balances per-layer constraints with per-parameter precision, converging faster and more stably across varying batch sizes and learning rates.

## Core Concept

Parameter-adaptive learning rates are common, but ignoring within-layer structure misses optimization opportunities. Parameters in a layer often split into distinct groups: some with consistently high gradients (critical decision points), others with low gradients (refinement parameters). SGG identifies these groups dynamically and scales each differently, preventing any single group from dominating updates while ensuring critical parameters stay responsive.

## Architecture Overview

- **Gradient Statistics Collection**: Track gradient magnitude, variance within each layer
- **Dynamic Clustering**: Partition parameters into K clusters based on gradient statistics (typically K=3-5)
- **Cluster-Specific Scaling**: Compute separate scaling factors per cluster, normalizing gradient magnitudes
- **Optimizer Wrapper**: Integrates with existing optimizers (AdamW) as a lightweight post-processing step
- **Stability Monitoring**: Tracks gradient norm evolution to detect and prevent divergence

## Implementation

This implementation demonstrates SGG as an optimizer wrapper for improved training stability.

Build the gradient grouping analyzer:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class GradientGroupingAnalyzer:
    """Analyze gradient statistics and identify parameter groups within layers."""

    def __init__(self, num_clusters: int = 3, window_size: int = 100):
        self.num_clusters = num_clusters
        self.window_size = window_size
        self.gradient_history = defaultdict(list)  # layer_name -> [grad_stats]

    def compute_gradient_statistics(self, model: nn.Module) -> Dict:
        """
        Compute gradient magnitude and variance per layer.
        Returns dict: layer_name -> {mean, std, percentiles}
        """
        layer_stats = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()
            grad_abs = torch.abs(grad)

            # Compute statistics
            stats = {
                "mean": float(grad_abs.mean()),
                "std": float(grad_abs.std()),
                "min": float(grad_abs.min()),
                "max": float(grad_abs.max()),
                "p25": float(torch.quantile(grad_abs, 0.25)),
                "p50": float(torch.quantile(grad_abs, 0.50)),
                "p75": float(torch.quantile(grad_abs, 0.75)),
            }

            # Extract layer name (remove .weight/.bias)
            layer_name = ".".join(name.split(".")[:-1])
            if layer_name not in layer_stats:
                layer_stats[layer_name] = []

            layer_stats[layer_name].append(stats)

        return layer_stats

    def cluster_parameters(self, model: nn.Module,
                          layer_name: str) -> Dict[int, List[str]]:
        """
        Cluster parameters within a layer based on gradient statistics.
        Returns dict: cluster_id -> [param_names]
        """
        layer_params = {}

        # Collect gradients for this layer
        for name, param in model.named_parameters():
            if layer_name in name and param.grad is not None:
                grad = param.grad.detach().flatten()
                layer_params[name] = grad

        if not layer_params:
            return {}

        # Compute features for clustering
        features = []
        param_names = list(layer_params.keys())

        for name in param_names:
            grad = layer_params[name]
            feature = [
                float(torch.abs(grad).mean()),  # Mean magnitude
                float(torch.abs(grad).std()),   # Variance
                float((grad ** 2).mean()) ** 0.5  # RMS
            ]
            features.append(feature)

        features = np.array(features)

        # K-means clustering
        centroids = self._kmeans(features, self.num_clusters)
        clusters = self._assign_clusters(features, centroids)

        # Build cluster -> param mapping
        cluster_mapping = defaultdict(list)
        for param_name, cluster_id in zip(param_names, clusters):
            cluster_mapping[cluster_id].append(param_name)

        return dict(cluster_mapping)

    def _kmeans(self, data: np.ndarray, k: int) -> np.ndarray:
        """Simple K-means clustering."""
        # Initialize centroids randomly
        indices = np.random.choice(len(data), k, replace=False)
        centroids = data[indices].copy()

        for iteration in range(10):  # Fixed iterations
            # Assign clusters
            distances = np.linalg.norm(data[:, None] - centroids, axis=2)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                data[assignments == i].mean(axis=0) if (assignments == i).any()
                else centroids[i]
                for i in range(k)
            ])

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return centroids

    def _assign_clusters(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign data points to nearest centroids."""
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        return np.argmin(distances, axis=1)

# Example usage
analyzer = GradientGroupingAnalyzer(num_clusters=3)

# Create mock model
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 256)
)

# Simulate training step with gradients
dummy_input = torch.randn(4, 256)
output = model(dummy_input).sum()
output.backward()

# Analyze gradients
layer_stats = analyzer.compute_gradient_statistics(model)
for layer, stats_list in layer_stats.items():
    print(f"{layer}: {len(stats_list)} parameter groups")

# Cluster parameters in first layer
clusters = analyzer.cluster_parameters(model, "0")
for cluster_id, param_names in clusters.items():
    print(f"Cluster {cluster_id}: {len(param_names)} parameters")
```

Implement SGG optimizer wrapper:

```python
class ScalingWithGradientGrouping(torch.optim.Optimizer):
    """
    Optimizer wrapper that applies cluster-specific gradient scaling.
    Wraps AdamW or other standard optimizer.
    """

    def __init__(self, model: nn.Module, optimizer_class=torch.optim.AdamW,
                 lr: float = 1e-4, num_clusters: int = 3, eps: float = 1e-8):
        self.model = model
        self.base_optimizer = optimizer_class(model.parameters(), lr=lr)
        self.analyzer = GradientGroupingAnalyzer(num_clusters=num_clusters)
        self.eps = eps
        self.layer_clusters = {}
        self.step_count = 0

    def zero_grad(self):
        """Clear gradients."""
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """
        Optimizer step with gradient grouping and scaling.
        """
        self.step_count += 1

        # Periodically recompute clusters (every 100 steps)
        if self.step_count % 100 == 0:
            self._update_clusters()

        # Apply group-specific scaling
        self._apply_group_scaling()

        # Run base optimizer step
        return self.base_optimizer.step(closure)

    def _update_clusters(self):
        """Recompute parameter clusters based on current gradients."""
        # Get layer names
        layer_names = set()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                layer_name = ".".join(name.split(".")[:-1])
                layer_names.add(layer_name)

        # Cluster each layer
        for layer_name in layer_names:
            self.layer_clusters[layer_name] = \
                self.analyzer.cluster_parameters(self.model, layer_name)

    def _apply_group_scaling(self):
        """Apply cluster-specific gradient scaling within each layer."""
        for layer_name, clusters in self.layer_clusters.items():
            for cluster_id, param_names in clusters.items():
                # Compute gradient norm for this cluster
                cluster_grad_norm = 0.0
                for param_name in param_names:
                    param = self._get_parameter_by_name(param_name)
                    if param.grad is not None:
                        cluster_grad_norm += torch.sum(param.grad ** 2)

                cluster_grad_norm = torch.sqrt(cluster_grad_norm + self.eps)

                # Scale gradients: target norm of 1.0 per cluster
                target_norm = 1.0
                scale_factor = target_norm / (cluster_grad_norm + self.eps)

                for param_name in param_names:
                    param = self._get_parameter_by_name(param_name)
                    if param.grad is not None:
                        param.grad.mul_(scale_factor)

    def _get_parameter_by_name(self, param_name: str) -> torch.nn.Parameter:
        """Retrieve parameter from model by name."""
        parts = param_name.split(".")
        obj = self.model
        for part in parts:
            obj = getattr(obj, part)
        return obj

# Test SGG optimizer
model = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 128)
)

optimizer = ScalingWithGradientGrouping(
    model,
    optimizer_class=torch.optim.AdamW,
    lr=1e-4,
    num_clusters=3
)

# Training loop
print("Training with SGG optimizer...")
for step in range(10):
    optimizer.zero_grad()

    # Forward pass
    x = torch.randn(32, 256)
    y = torch.randn(32, 128)
    output = model(x)
    loss = nn.functional.mse_loss(output, y)

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()

    if (step + 1) % 5 == 0:
        print(f"Step {step+1}: Loss = {loss.item():.6f}")
```

Benchmark SGG against standard AdamW:

```python
def benchmark_optimizers():
    """Compare SGG vs standard AdamW convergence."""
    torch.manual_seed(42)

    # Create models
    model_sgg = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 128))
    model_adamw = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 128))

    # Copy weights to ensure same initialization
    with torch.no_grad():
        for p1, p2 in zip(model_sgg.parameters(), model_adamw.parameters()):
            p2.copy_(p1)

    # Optimizers
    opt_sgg = ScalingWithGradientGrouping(model_sgg, lr=1e-3)
    opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=1e-3)

    # Generate dummy data
    X = torch.randn(1000, 256)
    y = torch.randn(1000, 128)

    losses_sgg = []
    losses_adamw = []

    # Training loop
    for epoch in range(50):
        # SGG training step
        opt_sgg.zero_grad()
        out_sgg = model_sgg(X)
        loss_sgg = nn.functional.mse_loss(out_sgg, y)
        loss_sgg.backward()
        opt_sgg.step()
        losses_sgg.append(loss_sgg.item())

        # AdamW training step
        opt_adamw.zero_grad()
        out_adamw = model_adamw(X)
        loss_adamw = nn.functional.mse_loss(out_adamw, y)
        loss_adamw.backward()
        opt_adamw.step()
        losses_adamw.append(loss_adamw.item())

    return losses_sgg, losses_adamw

losses_sgg, losses_adamw = benchmark_optimizers()
print(f"Final SGG loss: {losses_sgg[-1]:.6f}")
print(f"Final AdamW loss: {losses_adamw[-1]:.6f}")
print(f"SGG Improvement: {(losses_adamw[-1] / losses_sgg[-1] - 1) * 100:.1f}%")
```

## Practical Guidance

| Aspect | Details |
|--------|---------|
| **Number of Clusters** | 3-5 typical; more clusters = finer control but higher overhead |
| **Clustering Frequency** | Update every 50-100 steps; balances adaptivity with compute |
| **Learning Rate** | Start with same LR as standard AdamW; may tolerate higher LR with SGG |
| **Model Size** | Benefits increase with model size; marginal gains on small models |
| **Batch Size Sensitivity** | SGG reduces LR tuning sensitivity across batch sizes |

**When to Use:**
- Training large language models on diverse batch sizes
- Sensitive to hyperparameter tuning; SGG reduces this burden
- Want faster convergence and better stability without architecture changes
- Compatible with other optimization techniques (gradient checkpointing, FSDP)
- Fine-tuning on multiple downstream tasks with single hyperparameter set

**When NOT to Use:**
- Simple models or datasets where standard AdamW already works well
- Real-time training with extremely tight latency constraints (clustering adds overhead)
- Already using highly specialized, hand-tuned learning rate schedules
- Theoretical analysis requires simple optimizer (stick with AdamW)

**Common Pitfalls:**
- Cluster count too high: wasted computation on fine-grained control with minimal benefit
- Clustering infrequently: outdated clusters become misaligned with actual gradient distribution
- Not accounting for warmup: clustering during warmup phase can be noisy; skip first N steps
- Ignoring divergence signals: monitor gradient norms; if still diverging, reduce learning rate globally

## Reference

Taming LLMs by Scaling Learning Rates with Gradient Grouping
https://arxiv.org/abs/2506.01049
