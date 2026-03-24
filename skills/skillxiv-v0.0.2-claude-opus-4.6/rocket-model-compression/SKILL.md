---
name: rocket-model-compression
title: "ROCKET: Rapid Optimization via Calibration-guided Knapsack for Model Compression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.11008"
keywords: [Model Compression, Pruning, Quantization, Knapsack Optimization, Layer-wise Allocation]
description: "Compress language models by formulating heterogeneous layer-wise allocation as a constrained knapsack problem. Measure per-layer compression error, solve via dynamic programming to minimize total error within target model size, and avoid pathological solutions where some layers are severely damaged."
---

# ROCKET: Rapid Optimization via Calibration-guided Knapsack for Model Compression

## Problem Context

Uniform compression (same quantization/sparsity across all layers) is suboptimal—attention layers are more sensitive than MLPs. Manual layer-wise allocation requires expertise. ROCKET automates this by profiling each layer's reconstruction error under different compression settings, then solving a constrained knapsack problem to distribute the global compression budget optimally.

## Core Concept

ROCKET operates in two phases: (1) profile layers with multiple compression configurations (rank, sparsity pairs), measuring error per configuration, (2) solve knapsack via dynamic programming to minimize total reconstruction error while staying within target model size. The approach handles per-layer error caps to prevent pathological solutions.

## Implementation

### Step 1: Layer-wise compression profiling

```python
import torch
from typing import Dict, List, Tuple

class LayerCompressionProfiler:
    """Profile layers under different compression settings."""

    def __init__(self, model):
        self.model = model

    def profile_layer_compression(
        self,
        layer_name: str,
        layer: torch.nn.Module,
        calibration_data: torch.Tensor,
        compression_configs: List[Dict]
    ) -> List[Dict]:
        """
        Test multiple compression configs on layer.

        Args:
            compression_configs: List of {rank, sparsity} pairs

        Returns:
            results: [{'config': {rank, sparsity}, 'error': float, 'params': int}, ...]
        """
        results = []
        original_output = layer(calibration_data)

        for config in compression_configs:
            rank = config.get('rank', layer.out_features)
            sparsity = config.get('sparsity', 0.0)

            # Apply compression
            compressed_layer = self._compress_layer(layer, rank, sparsity)

            # Measure reconstruction error
            with torch.no_grad():
                compressed_output = compressed_layer(calibration_data)

            error = torch.norm(original_output - compressed_output).item()

            # Count parameters after compression
            param_count = sum(p.numel() for p in compressed_layer.parameters())

            results.append({
                'config': config,
                'error': error,
                'param_count': param_count,
                'compression_ratio': layer.out_features / max(1, rank)
            })

        return results

    def _compress_layer(
        self,
        layer: torch.nn.Module,
        rank: int,
        sparsity: float
    ) -> torch.nn.Module:
        """Apply compression (low-rank + sparsity) to layer."""
        # Low-rank decomposition
        if hasattr(layer, 'weight'):
            W = layer.weight.data
            U, S, V = torch.linalg.svd(W, full_matrices=False)
            S[rank:] = 0  # Zero out small singular values
            W_compressed = U @ torch.diag(S) @ V

            # Sparsity: zero out small weights
            threshold = torch.quantile(torch.abs(W_compressed), sparsity)
            W_compressed[torch.abs(W_compressed) < threshold] = 0

            layer.weight.data = W_compressed

        return layer

    def profile_all_layers(
        self,
        calibration_data: torch.Tensor,
        compression_configs: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Profile all layers in model."""
        layer_profiles = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                profile = self.profile_layer_compression(
                    name, module, calibration_data, compression_configs
                )
                layer_profiles[name] = profile

        return layer_profiles
```

### Step 2: Knapsack formulation and DP solution

```python
class CompressionKnapsack:
    """Solve compression allocation as constrained knapsack."""

    def __init__(
        self,
        layer_profiles: Dict[str, List[Dict]],
        target_model_size: int,
        error_cap_per_layer: float = float('inf')
    ):
        """
        Args:
            layer_profiles: Compression profiles per layer
            target_model_size: Target parameter count
            error_cap_per_layer: Max error allowed per layer (prevent pathological)
        """
        self.layer_profiles = layer_profiles
        self.target_size = target_model_size
        self.error_cap = error_cap_per_layer

    def solve_allocation(self) -> Dict[str, Dict]:
        """
        Solve knapsack via dynamic programming.

        Returns:
            allocation: {layer_name: {config, error, param_count}, ...}
        """
        layer_names = list(self.layer_profiles.keys())
        num_layers = len(layer_names)

        # DP table: dp[layer_idx][budget] = min_error
        dp = {}
        parent = {}  # Track which config was chosen

        # Base case
        dp[(0, 0)] = 0
        parent[(0, 0)] = None

        # Fill DP table
        for layer_idx in range(num_layers):
            layer_name = layer_names[layer_idx]
            profiles = self.layer_profiles[layer_name]

            new_dp = {}
            new_parent = {}

            for (prev_idx, budget), min_error in dp.items():
                if prev_idx != layer_idx:
                    continue

                # Try each compression config for this layer
                for config_idx, config_result in enumerate(profiles):
                    error = config_result['error']
                    param_count = config_result['param_count']

                    # Skip if error cap exceeded
                    if error > self.error_cap:
                        continue

                    new_budget = budget + param_count

                    if new_budget > self.target_size:
                        continue  # Exceeds capacity

                    new_error = min_error + error

                    key = (layer_idx + 1, new_budget)
                    if key not in new_dp or new_error < new_dp[key]:
                        new_dp[key] = new_error
                        new_parent[key] = (layer_idx, budget, config_idx)

            dp.update(new_dp)
            parent.update(new_parent)

        # Find best solution within target budget
        best_error = float('inf')
        best_state = None

        for (layer_idx, budget), error in dp.items():
            if layer_idx == num_layers and budget <= self.target_size:
                if error < best_error:
                    best_error = error
                    best_state = (layer_idx, budget)

        # Reconstruct allocation
        allocation = {}
        if best_state:
            state = best_state
            for layer_idx in range(num_layers - 1, -1, -1):
                if state in parent and parent[state] is not None:
                    prev_layer_idx, prev_budget, config_idx = parent[state]
                    layer_name = layer_names[layer_idx]
                    allocation[layer_name] = self.layer_profiles[layer_name][config_idx]
                    state = (prev_layer_idx, prev_budget)

        return allocation
```

### Step 3: Compression execution

```python
class CompressionExecutor:
    """Apply optimal compression to model."""

    def __init__(self, model):
        self.model = model

    def apply_compression_allocation(
        self,
        allocation: Dict[str, Dict]
    ) -> torch.nn.Module:
        """Apply compression to each layer according to allocation."""
        for layer_name, config_result in allocation.items():
            layer = dict(self.model.named_modules())[layer_name]
            config = config_result['config']

            # Apply compression
            self._compress_layer_inplace(layer, config)

        return self.model

    def _compress_layer_inplace(
        self,
        layer: torch.nn.Module,
        config: Dict
    ):
        """Compress layer in-place."""
        rank = config.get('rank', layer.weight.shape[0])
        sparsity = config.get('sparsity', 0.0)

        W = layer.weight.data

        # Low-rank
        U, S, V = torch.linalg.svd(W, full_matrices=False)
        S[rank:] = 0
        W_compressed = U @ torch.diag(S) @ V

        # Sparsity
        threshold = torch.quantile(torch.abs(W_compressed), sparsity)
        W_compressed[torch.abs(W_compressed) < threshold] = 0

        layer.weight.data = W_compressed

    def evaluate_compression(
        self,
        test_data: torch.Tensor,
        metric_fn=None
    ) -> Dict:
        """Evaluate compressed model."""
        with torch.no_grad():
            metric = metric_fn(self.model, test_data) if metric_fn else 1.0

        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            'metric': metric,
            'total_params': total_params,
            'compression_ratio': total_params / (sum(p.numel() for p in self.model.parameters()))
        }
```

### Step 4: Full compression pipeline

```python
def compress_model_rocket(
    model: torch.nn.Module,
    calibration_data: torch.Tensor,
    test_data: torch.Tensor,
    target_size_fraction: float = 0.5,
    num_compression_configs: int = 8
):
    """
    Full ROCKET compression pipeline.

    Args:
        target_size_fraction: Fraction of original size (0.5 = 50% compression)
    """
    # Generate compression configs to test
    compression_configs = []
    for rank_ratio in [0.25, 0.5, 0.75, 1.0]:
        for sparsity in [0.0, 0.25, 0.5]:
            compression_configs.append({
                'rank': int(rank_ratio * 768),  # Assume dim=768
                'sparsity': sparsity
            })

    # Profile layers
    print("Profiling layer compression...")
    profiler = LayerCompressionProfiler(model)
    layer_profiles = profiler.profile_all_layers(calibration_data, compression_configs)

    # Compute target size
    original_params = sum(p.numel() for p in model.parameters())
    target_params = int(original_params * target_size_fraction)

    print(f"Original params: {original_params}")
    print(f"Target params: {target_params}")

    # Solve knapsack
    print("Solving knapsack optimization...")
    knapsack = CompressionKnapsack(
        layer_profiles,
        target_params,
        error_cap_per_layer=0.1
    )
    allocation = knapsack.solve_allocation()

    # Execute compression
    print("Applying compression...")
    executor = CompressionExecutor(model)
    compressed_model = executor.apply_compression_allocation(allocation)

    # Evaluate
    metrics = executor.evaluate_compression(test_data)
    print(f"Compressed: {metrics['total_params']} params, "
          f"Ratio: {metrics['compression_ratio']:.2f}x")

    return compressed_model
```

## Practical Guidance

**When to use**: LLM deployment with strict size budgets; heterogeneous layer importance known

**Hyperparameters**:
- **target_size_fraction**: 0.3-0.7 (compression ratio)
- **error_cap_per_layer**: 0.05-0.2 (pathology prevention)
- **num_compression_configs**: 8-20 (more = better solutions but slower)
- **rank_fractions**: [0.1, 0.25, 0.5, 0.75, 1.0]
- **sparsities**: [0.0, 0.2, 0.4, 0.6]

**Key advantages**:
- Optimal allocation avoids pathological solutions
- Attention layers naturally preserved
- Single unified optimization framework
- Calibration-guided (preserves model behavior)

**Common pitfalls**:
- error_cap too strict → infeasible solutions
- Calibration data not representative → poor allocation
- Not validating downstream task performance
- DP state space explosion → need pruning

**Scaling**: DP complexity is O(L·M·B̄) where L=layers, M=configs, B̄=avg budget.

## Reference

Paper: https://arxiv.org/abs/2602.11008
Related work: Model compression, quantization, pruning, knapsack optimization
Benchmarks: Language model compression, downstream task accuracy
