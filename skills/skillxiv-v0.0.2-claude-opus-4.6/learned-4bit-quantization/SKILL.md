---
name: learned-4bit-quantization
title: "any4: Learned 4-bit Numeric Representation for LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.04610"
keywords: [Quantization, Model Compression, 4-bit, Neural Networks, Inference Optimization]
description: "Compress LLM weights to 4-bit precision using learned per-row lookup tables that minimize output activation error, achieving better accuracy than fixed formats (int4, fp4, nf4) while maintaining inference speed."
---

# Learned 4-bit Quantization: Optimizing Numeric Representation Through Data-Driven Clustering

Large language model inference costs scale with model size, but reducing precision from 16-bit to 4-bit can decrease memory and computation by 4×. Standard 4-bit formats (int4, fp4, nf4) use fixed numeric ranges chosen to work across all weight distributions. However, different layers and weight matrices have different distributions, making fixed formats suboptimal.

The any4 technique learns custom 4-bit representations per weight matrix row through weighted K-means clustering, directly optimizing to minimize output activation error rather than weight reconstruction error. This data-driven approach outperforms fixed formats while remaining competitive with preprocessing-intensive methods like AWQ and GPTQ, all with minimal calibration overhead.

## Core Concept

Quantization aims to compress weights while preserving model outputs. Traditional approaches minimize weight reconstruction error (distance between original and dequantized weights), but this doesn't directly minimize output error. The any4 approach formulates quantization as a weighted K-means problem: for each weight matrix row, find 16 distinct numeric values (4-bit = 2^4) that minimize the output activation error when those weights are used.

The key insight is that different weight values contribute differently to output error—weights with larger incoming activations matter more. Weighted K-means accounts for this by using activation magnitudes as clustering weights. This produces "arbitrary" numeric representations—not standard numeric formats but optimal for each layer's actual data distribution and computational patterns.

## Architecture Overview

- **Per-Row Lookup Tables (LUTs)**: For each row of every weight matrix, learn 16 distinct numeric values (4 bits per weight)
- **Weighted K-means Clustering**: Cluster weight row values around 16 centers, weighted by incoming activation magnitudes
- **Activation-Aware Weighting**: Use statistics from calibration data (sample inputs) to weight clustering by output importance
- **Bit-Width Encoding**: Map original weights to nearest cluster center, store as 4-bit indices
- **Dequantization Overhead**: Minimal overhead from LUT lookup during inference (64 bytes per row)
- **Calibration-Efficient Training**: Single hand-curated prompt or small calibration set suffices (outperforms larger datasets)

## Implementation

The following implements learned 4-bit quantization through weighted K-means clustering.

**Step 1: Weighted K-means Clustering for Quantization**

This performs the core K-means clustering weighted by activation importance.

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

class WeightedKMeansQuantizer:
    """Quantize weight matrices using activation-weighted K-means."""

    def __init__(self, num_clusters: int = 16, max_iterations: int = 20):
        self.num_clusters = num_clusters  # 16 for 4-bit
        self.max_iterations = max_iterations

    def compute_activation_weights(
        self,
        input_activations: torch.Tensor,
        weight_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-weight importance using activation magnitudes.
        Weights affecting large activations matter more.
        Args:
            input_activations: (seq_len, in_features)
            weight_matrix: (in_features, out_features)
        Returns:
            importance weights (in_features, out_features)
        """
        # RMS of activations for each input feature
        activation_scale = torch.sqrt((input_activations ** 2).mean(dim=0, keepdim=True))

        # Broadcast to match weight dimensions
        weights = activation_scale.T @ torch.ones_like(weight_matrix[:1, :])
        return weights + 1e-8  # Avoid zero weights

    def quantize_row(
        self,
        weight_row: torch.Tensor,
        activation_weights: torch.Tensor,
        num_clusters: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a single weight row using weighted K-means.
        Args:
            weight_row: (out_features,) weight values
            activation_weights: (out_features,) importance weights
            num_clusters: number of clusters (16 for 4-bit)
        Returns:
            (quantized_row, cluster_centers, indices)
        """
        # Initialize cluster centers at quantiles
        weight_sorted = weight_row.sort()[0]
        quantiles = torch.linspace(0, weight_row.shape[0] - 1, num_clusters).long()
        centers = weight_sorted[quantiles].clone()

        # Weighted K-means iterations
        for iteration in range(self.max_iterations):
            # Assignment: find nearest center for each weight
            distances = torch.abs(weight_row.unsqueeze(1) - centers.unsqueeze(0))
            indices = distances.argmin(dim=1)

            # Update centers: weighted mean of assigned weights
            centers_new = centers.clone()
            for k in range(num_clusters):
                mask = indices == k
                if mask.sum() > 0:
                    weighted_sum = (weight_row[mask] * activation_weights[mask]).sum()
                    weight_sum = activation_weights[mask].sum()
                    centers_new[k] = weighted_sum / weight_sum

            # Check convergence
            if torch.allclose(centers, centers_new, atol=1e-6):
                break

            centers = centers_new

        # Final assignment
        distances = torch.abs(weight_row.unsqueeze(1) - centers.unsqueeze(0))
        indices = distances.argmin(dim=1)

        # Quantized weights
        quantized_row = centers[indices]

        return quantized_row, centers, indices

    def quantize_matrix(
        self,
        weight_matrix: torch.Tensor,
        input_activations: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Quantize full weight matrix row-by-row.
        Args:
            weight_matrix: (in_features, out_features)
            input_activations: (seq_len, in_features) calibration data
        Returns:
            (quantized_matrix, cluster_centers_list, indices_matrix)
        """
        out_features = weight_matrix.shape[0]
        quantized_matrix = torch.zeros_like(weight_matrix)
        cluster_centers_list = []
        indices_matrix = torch.zeros_like(weight_matrix, dtype=torch.uint8)

        # Compute activation weights
        act_weights = self.compute_activation_weights(input_activations, weight_matrix)

        # Quantize each row independently
        for i in range(out_features):
            weight_row = weight_matrix[i, :]
            activation_weight_row = act_weights[i, :]

            quantized_row, centers, indices = self.quantize_row(
                weight_row, activation_weight_row
            )

            quantized_matrix[i, :] = quantized_row
            cluster_centers_list.append(centers)
            indices_matrix[i, :] = indices

        return quantized_matrix, cluster_centers_list, indices_matrix
```

**Step 2: LUT-based Dequantization**

This implements efficient lookup-table based dequantization during inference.

```python
class LUTDequantizer(nn.Module):
    """Dequantize weights using per-row lookup tables during inference."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Lookup tables: (out_features, 16) for 4-bit
        self.luts = nn.Parameter(
            torch.randn(out_features, 16),
            requires_grad=False
        )
        # Indices: (out_features, in_features) uint8
        self.indices = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.uint8),
            requires_grad=False
        )
        self.in_features = in_features
        self.out_features = out_features

    def set_luts(self, cluster_centers_list: List[torch.Tensor], indices: torch.Tensor):
        """Initialize LUTs from cluster centers and indices."""
        for i, centers in enumerate(cluster_centers_list):
            self.luts.data[i, :] = centers
        self.indices.data = indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Efficient matrix multiplication using LUTs.
        Args:
            x: input (batch, seq_len, in_features)
        Returns:
            output (batch, seq_len, out_features)
        """
        batch, seq_len, in_features = x.shape

        # Dequantize weights: gather from LUTs
        # indices shape: (out_features, in_features)
        # Want to gather: for each output feature, look up indices into LUT
        dequantized_weights = torch.zeros(self.out_features, in_features, device=x.device)

        for out_idx in range(self.out_features):
            for in_idx in range(in_features):
                cluster_idx = self.indices[out_idx, in_idx].item()
                dequantized_weights[out_idx, in_idx] = self.luts[out_idx, cluster_idx]

        # Standard matrix multiplication
        output = torch.matmul(x, dequantized_weights.t())
        return output

class QuantizedLinear(nn.Module):
    """Linear layer with 4-bit learned quantization."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.dequantizer = LUTDequantizer(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantizer(x)

    def quantize(
        self,
        weight_matrix: torch.Tensor,
        input_activations: torch.Tensor
    ):
        """Quantize original weights and set up LUTs."""
        quantizer = WeightedKMeansQuantizer(num_clusters=16)
        quantized, centers_list, indices = quantizer.quantize_matrix(
            weight_matrix, input_activations
        )
        self.dequantizer.set_luts(centers_list, indices)
```

**Step 3: Calibration-Efficient Quantization**

This implements the calibration process using minimal data.

```python
class QuantizationCalibrator:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations_cache = {}

    def register_forward_hooks(self):
        """Hook into layer activations to collect calibration data."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(input[0], torch.Tensor):
                    self.activations_cache[name] = input[0].detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(hook_fn(name))

    def calibrate(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        num_bits: int = 4
    ) -> Dict[str, Tuple]:
        """
        Calibrate and quantize all linear layers.
        Args:
            model: neural network
            calibration_data: (seq_len, input_dim) small calibration batch
            num_bits: bits per weight (4 for any4)
        Returns:
            quantization params per layer
        """
        quantization_params = {}
        quantizer = WeightedKMeansQuantizer(num_clusters=2 ** num_bits)

        # Collect activations on calibration data
        self.register_forward_hooks()
        model.eval()
        with torch.no_grad():
            _ = model(calibration_data)

        # Quantize each layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activations_cache:
                # Get weight matrix and activation data
                weight = module.weight.data
                activations = self.activations_cache[name]

                # Quantize
                quantized, centers_list, indices = quantizer.quantize_matrix(weight, activations)

                quantization_params[name] = {
                    "centers": centers_list,
                    "indices": indices,
                    "original_weight_shape": weight.shape
                }

        return quantization_params

    def apply_quantization(
        self,
        model: nn.Module,
        quantization_params: Dict
    ):
        """Apply quantization to model layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in quantization_params:
                params = quantization_params[name]
                quantized_module = QuantizedLinear(
                    module.in_features, module.out_features
                )
                quantized_module.quantize(
                    module.weight.data,
                    torch.randn(1, module.in_features)  # Dummy activation
                )
                # Replace module
                setattr(model, name, quantized_module)
```

**Step 4: Single-Sample Calibration**

This demonstrates that a single curated prompt works better than standard calibration datasets.

```python
class SingleSampleCalibration:
    """Curated single-sample calibration for any4 quantization."""

    @staticmethod
    def get_diverse_prompt() -> str:
        """
        Hand-curated prompt covering diverse topics.
        Single prompt outperforms multi-sample calibration.
        """
        return """
        Natural language processing enables computers to understand and generate text.
        Machine learning models learn patterns from data without explicit programming.
        Neural networks are inspired by biological neurons and process information hierarchically.
        Transformers revolutionized deep learning with attention mechanisms for parallel processing.
        Computer vision tasks include image classification, object detection, and segmentation.
        Reinforcement learning trains agents through reward signals in interactive environments.
        Large language models demonstrate remarkable capabilities in reasoning and knowledge.
        Knowledge graphs represent structured information about entities and relationships.
        """

    def calibrate_from_single_prompt(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda"
    ) -> Dict:
        """Calibrate using single diverse prompt."""
        calibrator = QuantizationCalibrator(model)

        # Tokenize single prompt
        prompt = self.get_diverse_prompt()
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        # Calibrate
        return calibrator.calibrate(model, input_ids)
```

## Practical Guidance

**Hyperparameters and Configuration**

| Parameter | Recommended Value | Range | Notes |
|-----------|------------------|-------|-------|
| Num Clusters | 16 | 8-256 | 16 for 4-bit (2^4); 32 for 5-bit, etc. |
| K-means Iterations | 20 | 10-50 | Usually converges by iteration 10-15 |
| Calibration Samples | 1 (curated) | 1-1000 | Single diverse prompt > large random dataset |
| Activation Weight Smoothing | 1e-8 | 1e-10 to 1e-5 | Prevents zero weights in clustering |
| Per-row LUT Storage | 64 bytes | Fixed | 16 clusters × 4 bytes per float32 |
| Quantization Dtype | float32 | float32/float16 | Precision of cluster centers |

**When to Use**

- Deploying LLMs on edge devices (phones, embedded systems) with memory constraints
- Reducing inference latency and memory bandwidth for serving
- Scenarios where 4-bit accuracy/efficiency trade-off is acceptable
- Models where activation patterns are non-uniform across layers (any4 exploits this)
- Systems requiring reproducibility across hardware (custom quantization schemes)

**When NOT to Use**

- Applications requiring high accuracy (8-bit or higher precision more robust)
- Real-time systems with strict latency budgets (LUT lookups add overhead)
- Models where all weights have similar distributions (fixed formats sufficient)
- Scenarios where calibration data is unavailable or expensive to collect
- Fine-grained control over quantization per-layer is not feasible

**Common Pitfalls**

- **Using random calibration data**: Curated diverse prompts work better than standard C4/Pile datasets. Invest in prompt diversity.
- **Over-fitting to calibration data**: Small calibration sets (single prompt) generalize better than large ones because they avoid memorizing specific patterns.
- **Ignoring per-row optimization**: Unlike layer-wise quantization, any4's per-row clustering is computationally intensive. Cache results if re-using models.
- **Neglecting LUT lookup cost**: LUT dequantization adds memory indirection. Ensure inference framework efficiently implements gathering.
- **Not comparing against preprocessing methods**: any4 is competitive with AWQ/GPTQ without preprocessing. Benchmark both for your use case.

## Reference

any4: Learned 4-bit Numeric Representation for LLMs. https://arxiv.org/abs/2507.04610
