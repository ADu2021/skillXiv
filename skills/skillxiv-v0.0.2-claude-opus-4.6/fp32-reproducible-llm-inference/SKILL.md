---
name: fp32-reproducible-llm-inference
title: "Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.09501"
keywords: [LLM reproducibility, floating-point precision, numerical stability, BF16, FP32, inference determinism]
description: "Diagnose and solve LLM reproducibility failures caused by floating-point precision across hardware configurations using LayerCast optimization for deterministic inference with minimal memory overhead."
---

# Give Me FP32 or Give Me Death?

## Core Concept

LLM inference reproducibility fails dramatically across hardware configurations due to non-associative floating-point arithmetic. Even with fixed random seeds and greedy decoding, changing batch size, GPU count, or GPU type produces divergent outputs with up to 9% accuracy variance and 9,000-token length differences in reasoning models. The root cause: limited precision in BF16 (7 mantissa bits) creates rounding error accumulation that varies by kernel execution order.

## Architecture Overview

- **Precision Hierarchy**: FP32 (23 bits) achieves near-perfect reproducibility; FP16 (10 bits) shows moderate variability; BF16 (7 bits) fails dramatically
- **Non-Associativity Problem**: Floating-point addition violates associativity—kernel scheduling and GPU memory layout change computation order, producing different accumulated rounding errors
- **LayerCast Solution**: Hybrid approach storing weights in memory-efficient BF16 while performing all computations in FP32, achieving deterministic results with 34% memory savings
- **Configuration Impact**: Divergence occurs predictably: different batch sizes→different padding patterns→different GPU kernel launches→different floating-point operation ordering

## Implementation

### Step 1: Diagnose Precision-Related Nondeterminism

```python
import torch
import numpy as np

def measure_reproducibility_drift(model, input_ids, configs):
    """
    Test model outputs across different hardware configurations.
    Configs: list of dicts with 'batch_size', 'num_gpus', 'gpu_type'
    """
    results = {}

    for config in configs:
        outputs = []
        for run in range(3):
            torch.manual_seed(42)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=512,
                    do_sample=False,  # greedy decoding
                    num_beams=1
                )
            outputs.append(output)

        # Measure variance across runs
        divergence_positions = []
        for run in range(1, len(outputs)):
            first_diff = (outputs[0] != outputs[run]).nonzero(as_tuple=True)
            if len(first_diff[0]) > 0:
                divergence_positions.append(first_diff[1][0].item())

        results[str(config)] = {
            'divergence_index': np.mean(divergence_positions) if divergence_positions else -1,
            'output_length_var': np.var([len(o) for o in outputs])
        }

    return results
```

### Step 2: Implement LayerCast Optimization

```python
import torch
import torch.nn as nn

class LayerCastOptimizer:
    """
    Hybrid precision wrapper: store weights in BF16, compute in FP32.
    Preserves memory efficiency while ensuring numerical stability.
    """

    def __init__(self, model, compute_dtype=torch.float32):
        self.model = model
        self.compute_dtype = compute_dtype
        self.original_forward = {}

    def patch_matmul_layers(self):
        """Replace linear layer forward passes with LayerCast computation."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.original_forward[name] = module.forward
                module.forward = self._create_cast_forward(module)

    def _create_cast_forward(self, linear_module):
        """Create a forward function that upcasts weights to FP32 for computation."""
        original_forward = linear_module.forward

        def cast_forward(x):
            # Upcast weights from BF16 to FP32 just-in-time
            weight_fp32 = linear_module.weight.to(self.compute_dtype)
            bias_fp32 = linear_module.bias.to(self.compute_dtype) if linear_module.bias is not None else None
            x_fp32 = x.to(self.compute_dtype)

            # Perform FP32 computation
            output = torch.nn.functional.linear(x_fp32, weight_fp32, bias_fp32)

            # Cast back to BF16 if needed for downstream layers
            return output.to(linear_module.weight.dtype)

        return cast_forward
```

### Step 3: Configure Inference for Determinism

```python
def setup_deterministic_inference(model, batch_size=8, use_fp32=True):
    """
    Configure model and environment for reproducible inference.
    """
    # Set random seeds globally
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Enable deterministic algorithms (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Move model to consistent precision
    if use_fp32:
        model = model.to(torch.float32)
    else:
        # If using BF16, apply LayerCast
        optimizer = LayerCastOptimizer(model, compute_dtype=torch.float32)
        optimizer.patch_matmul_layers()

    return model

def generate_with_reproducibility(model, input_ids, max_length=512):
    """
    Generate outputs with maximum reproducibility guarantees.
    """
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=False,  # Greedy decoding—no sampling variance
            num_beams=1,
            use_cache=False,  # Disable kv-cache if reproducibility critical
            temperature=1.0,
            top_p=1.0,
            top_k=None
        )
    return output
```

## Practical Guidance

**When to Use FP32 for Reproducibility**:
- Reasoning tasks generating long chains of thought (>2000 tokens)
- Mathematical problem-solving requiring consistency across runs
- Applications where inference output variance exceeds tolerance

**When LayerCast is Sufficient**:
- Production deployments prioritizing memory efficiency
- Batch inference where 34% memory savings outweighs minor non-determinism
- Models under 13B parameters where BF16 drift is modest (<2% accuracy variance)

**Trade-offs to Consider**:
- Full FP32: Perfect reproducibility but 50% memory increase
- LayerCast: 34% memory savings with FP32-level determinism (best balance)
- Pure BF16: 50% memory savings but up to 9% accuracy variance across configs

**Configuration Best Practices**:
- Standardize batch sizes within teams (one batch size per deployment)
- Pin GPU types when reproducibility is contractually required
- Report uncertainty quantiles (mean ± std) for downstream model outputs when using lower precision
- Log exact hardware configurations alongside inference results for auditing

## Reference

- Non-associativity of floating-point arithmetic: fundamental constraint of IEEE 754 standard
- Rounding error accumulation: proportional to operation count, controlled by mantissa precision
- Dynamic shapes and padding: creating uncontrollable kernel scheduling variations in operator fusion
