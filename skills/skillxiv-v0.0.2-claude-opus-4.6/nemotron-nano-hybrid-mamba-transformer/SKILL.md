---
name: nemotron-nano-hybrid-mamba-transformer
title: "Nemotron Nano 2: Hybrid Mamba-Transformer Architecture for Efficient Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.14444
keywords: [hybrid-architecture, mamba, transformer, model-compression, reasoning, throughput]
description: "Build hybrid Mamba-Transformer models combining efficient Mamba-2 layers with standard attention to achieve 6x higher inference throughput while maintaining reasoning accuracy on long-context tasks."
---

# Nemotron Nano 2: Hybrid Mamba-Transformer for Efficient Reasoning

## Core Concept

Nemotron-Nano-9B-v2 achieves state-of-the-art accuracy with significantly improved inference throughput by replacing most self-attention layers in Transformers with Mamba-2 layers. This hybrid approach maintains the reasoning capabilities of full Transformers while gaining the linear-time complexity benefits of Mamba. The architecture processes extended reasoning traces efficiently, achieving up to 6x higher throughput on reasoning workloads while supporting 128k token context on a single GPU.

## Architecture Overview

- **Hybrid Layer Composition**: Mix of Mamba-2 and Transformer self-attention layers
- **Strategic Attention Placement**: Preserve full attention for critical reasoning steps
- **Model Compression**: Pre-train on 20 trillion tokens then compress to target size
- **FP8 Training**: Optimize memory efficiency during pre-training and inference
- **Long-Context Support**: Enable 128k token processing within single GPU memory constraints

## Implementation Steps

### 1. Design Hybrid Layer Mixing Strategy

Determine which layers use Mamba vs. Transformer attention:

```python
def create_hybrid_architecture(
    total_layers: int,
    mamba_ratio: float = 0.7,
    attention_positions: list[int] = None
) -> list[str]:
    """
    Design layer composition with Mamba and Transformer layers.

    Default: ~70% Mamba for efficiency, ~30% attention for critical reasoning
    Attention layers placed strategically at early, middle, and late stages
    """
    if attention_positions is None:
        # Default: preserve attention at key positions for information bottlenecks
        attention_positions = [0, total_layers // 2, total_layers - 1]

    layer_types = []
    for i in range(total_layers):
        if i in attention_positions:
            layer_types.append("attention")
        elif i < int(total_layers * mamba_ratio):
            layer_types.append("mamba")
        else:
            layer_types.append("attention")

    return layer_types
```

### 2. Implement Mamba-2 Layers

Use selective state space models for efficient sequence processing:

```python
def create_mamba2_layer(
    hidden_size: int,
    state_size: int = 16,
    expand_factor: int = 2
) -> "Mamba2Layer":
    """
    Create a Mamba-2 layer combining selective state space model with gating.
    """
    class Mamba2Layer:
        def __init__(self):
            self.input_projection = Linear(hidden_size, hidden_size * expand_factor)
            self.state_matrix = StateSpaceMatrix(hidden_size * expand_factor, state_size)
            self.output_projection = Linear(hidden_size * expand_factor, hidden_size)
            self.gate = Linear(hidden_size * expand_factor, hidden_size * expand_factor)

        def forward(self, x):
            # Project input
            x_expanded = self.input_projection(x)

            # Apply selective SSM
            y = self.state_matrix(x_expanded)

            # Apply gating mechanism
            gated = y * sigmoid(self.gate(x_expanded))

            # Project back to hidden size
            output = self.output_projection(gated)
            return output

    return Mamba2Layer()
```

### 3. Configure Attention Layers

Maintain standard Transformer attention at strategic positions:

```python
def create_attention_layer(
    hidden_size: int,
    num_heads: int = 8,
    max_seq_length: int = 128000
) -> "AttentionLayer":
    """
    Standard multi-head attention, optimized for long sequences.
    """
    class AttentionLayer:
        def __init__(self):
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.q_proj = Linear(hidden_size, hidden_size)
            self.k_proj = Linear(hidden_size, hidden_size)
            self.v_proj = Linear(hidden_size, hidden_size)
            self.out_proj = Linear(hidden_size, hidden_size)

        def forward(self, x, attention_mask=None):
            batch_size, seq_len, _ = x.shape

            # Project to Q, K, V
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

            # Compute attention with optional long-context optimization
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_weights = softmax(attention_scores, dim=-1)
            output = torch.matmul(attention_weights, v)

            # Reshape and project
            output = output.view(batch_size, seq_len, -1)
            return self.out_proj(output)

    return AttentionLayer()
```

### 4. Pre-training and Compression

Implement the Minitron compression strategy:

```python
def pretrain_and_compress(
    base_config: dict,
    target_size: str = "9B",
    pretrain_tokens: int = 20_000_000_000,
    fp8_enabled: bool = True
) -> "CompressedModel":
    """
    Pre-train large model then compress to target size using Minitron strategy.
    """
    # Step 1: Pre-train larger base model
    print(f"Pre-training on {pretrain_tokens:,} tokens...")
    base_model = create_hybrid_model(
        hidden_size=base_config["hidden_size"],
        num_layers=base_config["num_layers"],
        num_heads=base_config["num_heads"]
    )

    if fp8_enabled:
        base_model = convert_to_fp8(base_model)

    # Train on diverse data corpus
    base_model = train_model(base_model, dataset, pretrain_tokens)

    # Step 2: Apply Minitron compression
    print(f"Compressing to {target_size}...")
    target_hidden_size = map_size_to_hidden_dim(target_size)
    target_num_layers = map_size_to_layers(target_size)

    compressed = compress_model(
        base_model,
        target_hidden_size=target_hidden_size,
        target_num_layers=target_num_layers,
        method="layer_dropping_and_projection"
    )

    return compressed
```

### 5. Configure Long-Context Inference

Enable efficient processing of extended sequences:

```python
def configure_long_context_inference(
    model: "HybridModel",
    max_tokens: int = 128000,
    device: str = "cuda"
) -> dict:
    """
    Set up inference for long-context processing on single GPU.
    """
    config = {
        "max_sequence_length": max_tokens,
        "use_kv_cache": True,
        "kv_cache_dtype": torch.float8_e4m3fn,  # FP8 for memory efficiency
        "attention_implementation": "flash_attention_2",  # Optimized kernels
        "mamba_scan_mode": "hardware_optimized",
        "device": device,
        "gradient_checkpointing": False  # Only for training
    }

    # Apply configuration
    model.config.update(config)
    model = model.to(device)
    if fp8_enabled:
        model = quantize_model(model, "fp8")

    return config
```

## Practical Guidance

### When to Use Nemotron-Style Hybrid Architecture

- Reasoning tasks requiring extended context (math, code, research)
- Inference-heavy deployments with throughput constraints
- Single-GPU deployment scenarios with memory limits
- Production systems requiring low-latency, long-context processing

### When NOT to Use

- Tasks where full attention is critical (e.g., precise alignment-heavy work)
- Domains requiring absolute state-of-the-art accuracy with no throughput concerns
- Training-focused workflows where pre-training cost dominates

### Key Hyperparameters

- **Mamba Ratio**: 0.6-0.8 (higher ratio = faster but less expressive)
- **Attention Positions**: Early, middle, late layers typically optimal
- **FP8 Training**: Reduces memory by 50%, minimal accuracy loss (<0.5%)
- **Max Context**: 128k tokens achievable; varies by GPU VRAM
- **Head Count**: 8-16 heads standard; scale with model size

### Performance Expectations

- Inference Throughput: 6x improvement over dense Transformer
- Context Support: 128k tokens on A100/H100 GPUs
- Accuracy: State-of-the-art compared to similarly-sized models
- Memory Usage: ~50% reduction vs. full Transformer with FP8

## Reference

NVIDIA. (2024). NVIDIA Nemotron Nano 2: Hybrid Mamba-Transformer Reasoning Model. arXiv preprint arXiv:2508.14444.
