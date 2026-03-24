---
name: xquant-kv-cache-rematerialization
title: "XQuant: Breaking Memory Wall for LLM Inference with KV Cache Rematerialization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.10395
keywords: [inference-optimization, kv-cache, quantization, memory-efficient, rematerialization]
description: "Reduce KV cache memory by 7.7-10x through quantization and rematerialization of input activations instead of caching Keys and Values, trading recomputation for memory efficiency."
---

# XQuant: Breaking Memory Wall for LLM Inference with KV Cache Rematerialization

## Core Concept

LLM inference is increasingly memory-bound: compute throughput far exceeds memory bandwidth, making KV cache memory the bottleneck. Traditional KV caching stores all Key and Value matrices from all layers and tokens, consuming enormous memory for long sequences.

XQuant breaks this bottleneck by storing only quantized layer input activations (X) instead of KV caches, then recomputing K and V on-the-fly during each forward pass. This trades modest computation increase for dramatic memory savings (7.7-10x), making long-sequence inference practical.

The key insight: input activations X are similar across layers and compress well; recomputing K = X·W_k is cheaper than storing/loading all cached KVs.

## Architecture Overview

- **Activation Caching**: Store quantized input activations X instead of KV pairs
- **On-the-Fly Recomputation**: Recalculate K = X·W_k and V = X·W_v during inference
- **Cross-Layer Sharing** (XQuant-CL): Detect and reuse activations across similar layers
- **Quantization**: Compress cached X to 4-8 bits (per-channel or per-token)
- **Negligible Accuracy Loss**: Achieves 7.7x memory reduction with < 0.01 perplexity impact

## Implementation Steps

### 1. Instrument Model to Capture Layer Inputs

Modify the model to save input activations at each layer without caching K, V matrices.

```python
import torch
import torch.nn as nn

class CacheEnabledTransformer(nn.Module):
    """
    Transformer with X-caching instead of KV caching
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.x_cache = {}  # Store quantized activations instead of KV
        self.cache_enabled = False

    def set_cache_mode(self, enabled):
        """Enable/disable caching during inference"""
        self.cache_enabled = enabled
        if enabled:
            self.x_cache.clear()

    def clear_cache(self):
        """Clear cached activations"""
        self.x_cache.clear()

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        """
        Forward pass with optional activation caching
        """
        if use_cache:
            self.set_cache_mode(True)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs
```

### 2. Implement Quantization for Activations

Quantize layer input activations to reduce memory requirements.

```python
class ActivationQuantizer:
    """
    Quantize layer input activations for efficient caching
    """
    def __init__(self, bits=8, method='per_channel'):
        self.bits = bits
        self.method = method  # 'per_channel' or 'per_token'
        self.max_val = 2 ** bits - 1

    def quantize(self, activation):
        """
        Quantize activation tensor to fixed-point integers
        activation: [batch, seq_len, hidden_size]
        """
        if self.method == 'per_channel':
            return self._quantize_per_channel(activation)
        else:
            return self._quantize_per_token(activation)

    def _quantize_per_channel(self, activation):
        """
        Quantize each hidden dimension independently
        """
        batch, seq_len, hidden = activation.shape

        # Find scale per channel
        min_vals = activation.reshape(-1, hidden).min(dim=0).values
        max_vals = activation.reshape(-1, hidden).max(dim=0).values

        # Compute scale factors
        ranges = max_vals - min_vals
        ranges = ranges.clamp(min=1e-8)
        scales = ranges / self.max_val

        # Quantize
        quantized = ((activation - min_vals.unsqueeze(0).unsqueeze(0)) / scales.unsqueeze(0).unsqueeze(0)).round()
        quantized = quantized.clamp(0, self.max_val).to(torch.uint8)

        return quantized, min_vals, scales

    def _quantize_per_token(self, activation):
        """
        Quantize each token independently
        """
        batch, seq_len, hidden = activation.shape

        # Find scale per token
        min_vals = activation.reshape(batch, seq_len, -1).min(dim=2).values
        max_vals = activation.reshape(batch, seq_len, -1).max(dim=2).values

        ranges = (max_vals - min_vals).clamp(min=1e-8)
        scales = ranges / self.max_val

        # Quantize
        quantized = ((activation - min_vals.unsqueeze(-1)) / scales.unsqueeze(-1)).round()
        quantized = quantized.clamp(0, self.max_val).to(torch.uint8)

        return quantized, min_vals, scales

    def dequantize(self, quantized, min_vals, scales):
        """
        Restore quantized activation to approximate original values
        """
        return quantized.float() * scales.unsqueeze(-1) + min_vals.unsqueeze(-1)
```

### 3. Cache Layer Inputs During Prefill

During the prefill (prompt processing) phase, cache quantized inputs instead of KV pairs.

```python
class XCacheManager:
    """
    Manages quantization and caching of layer inputs
    """
    def __init__(self, quantizer, num_layers=32):
        self.quantizer = quantizer
        self.num_layers = num_layers
        self.x_cache = {}
        self.metadata = {}

    def cache_layer_input(self, layer_idx, x):
        """
        Cache quantized input activation for a layer
        x: [batch, seq_len, hidden_size]
        """
        # Quantize
        quantized, min_vals, scales = self.quantizer.quantize(x)

        # Store
        self.x_cache[layer_idx] = quantized
        self.metadata[layer_idx] = {
            'min_vals': min_vals,
            'scales': scales,
            'dtype': x.dtype,
            'shape': x.shape
        }

    def get_cached_x(self, layer_idx):
        """
        Retrieve and dequantize cached activation
        """
        if layer_idx not in self.x_cache:
            return None

        quantized = self.x_cache[layer_idx]
        meta = self.metadata[layer_idx]

        x = self.quantizer.dequantize(quantized, meta['min_vals'], meta['scales'])
        return x

    def get_cache_memory(self):
        """
        Estimate memory used by cache
        """
        total_bytes = 0
        for layer_idx, quantized in self.x_cache.items():
            # Quantized data
            bytes_per_element = self.quantizer.bits / 8
            cache_size = quantized.numel() * bytes_per_element

            # Metadata (scales + min_vals)
            meta_size = self.metadata[layer_idx]['min_vals'].numel() * 4  # float32
            meta_size += self.metadata[layer_idx]['scales'].numel() * 4

            total_bytes += cache_size + meta_size

        return total_bytes / (1024 ** 3)  # Return in GB
```

### 4. Implement On-the-Fly K, V Recomputation

During decoding, recompute K and V from cached X instead of loading them from cache.

```python
class RematerializedAttention(nn.Module):
    """
    Attention with on-the-fly KV recomputation from cached X
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # These weights are used for K, V recomputation
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_cached, query_input, attention_mask=None):
        """
        Compute attention with recomputed K, V from cached X

        Args:
            x_cached: cached quantized input activations [batch, seq_len, hidden]
            query_input: current query input [batch, 1, hidden]
            attention_mask: optional attention mask
        """
        # Recompute K, V from cached X
        K = self.w_k(x_cached)  # [batch, seq_len, hidden]
        V = self.w_v(x_cached)  # [batch, seq_len, hidden]

        # Compute Q from current input
        Q = self.w_q(query_input)  # [batch, 1, hidden]

        # Reshape for multi-head attention
        batch, seq_len, hidden = K.shape
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch, 1, hidden)

        output = self.w_o(context)
        return output
```

### 5. Implement Cross-Layer Sharing (XQuant-CL)

Detect similar activations across layers and reuse cache entries to save memory further.

```python
class CrossLayerActivationAnalyzer:
    """
    Analyze and identify shareable activations across layers
    """
    def __init__(self, similarity_threshold=0.95):
        self.similarity_threshold = similarity_threshold
        self.layer_similarities = {}

    def compute_layer_similarity(self, x_cache_dict):
        """
        Compute cosine similarity between layer input activations
        """
        layer_indices = sorted(x_cache_dict.keys())

        for i, layer_i in enumerate(layer_indices):
            for layer_j in layer_indices[i + 1:]:
                x_i = x_cache_dict[layer_i]
                x_j = x_cache_dict[layer_j]

                # Compute cosine similarity (sample across tokens if too large)
                if x_i.shape[1] > 100:  # Too many tokens, sample
                    indices = torch.randperm(x_i.shape[1])[:100]
                    x_i_sample = x_i[:, indices, :]
                    x_j_sample = x_j[:, indices, :]
                else:
                    x_i_sample = x_i
                    x_j_sample = x_j

                # Flatten and compute similarity
                x_i_flat = x_i_sample.reshape(-1, x_i_sample.shape[-1])
                x_j_flat = x_j_sample.reshape(-1, x_j_sample.shape[-1])

                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(x_i_flat, x_j_flat).mean()

                self.layer_similarities[(layer_i, layer_j)] = cos_sim.item()

    def get_shared_cache_plan(self):
        """
        Determine which layers can share cache entries
        """
        shared_plan = {}

        for (layer_i, layer_j), similarity in self.layer_similarities.items():
            if similarity > self.similarity_threshold:
                # These layers can share cache
                if layer_i not in shared_plan:
                    shared_plan[layer_i] = [layer_i]
                shared_plan[layer_i].append(layer_j)

        return shared_plan
```

### 6. Inference Loop with X-Caching

Implement the full inference pipeline using activation caching and rematerialization.

```python
def inference_with_xquant(model, input_ids, max_length=256, x_cache_manager=None):
    """
    Run inference with X-caching and on-the-fly KV recomputation
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Phase 1: Prefill (cache all layer inputs)
    with torch.no_grad():
        current_input = input_ids
        for layer_idx in range(model.num_layers):
            # Get layer
            layer = model.layers[layer_idx]

            # Forward to capture input activation
            x = layer.pre_norm(current_input)
            x_cache_manager.cache_layer_input(layer_idx, x)

            # Continue forward
            current_input = layer(current_input)

    # Phase 2: Decoding (generate tokens using cached X)
    generated_tokens = input_ids.clone()

    for step in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            current_input = generated_tokens[:, -1:, :]  # Last token

            for layer_idx in range(model.num_layers):
                layer = model.layers[layer_idx]

                # Get cached X
                x_cached = x_cache_manager.get_cached_x(layer_idx)

                # Recompute attention with cached X, current Q
                x = layer.pre_norm(current_input)
                attn_output = layer.rematerialized_attn(x_cached, x)

                # Continue forward
                current_input = layer.post_norm(attn_output + current_input)

        # Sample next token
        logits = model.head(current_input)
        next_token = logits.argmax(dim=-1)

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        # Stop if EOS
        if next_token.item() == model.eos_token_id:
            break

    return generated_tokens
```

## Practical Guidance

### Hyperparameters & Configuration

- **Quantization Bits**: 8-bit per-channel recommended (good speed-quality tradeoff)
- **Quantization Method**: per-channel better than per-token (less overhead)
- **Cross-Layer Threshold**: 0.95 cosine similarity to identify shareable activations
- **Memory Overhead**: ~5-10% for metadata (scales, min_vals) per layer
- **Speed Overhead**: ~10-15% slower per-token inference due to K,V recomputation

### When to Use XQuant

- Inference memory is the bottleneck (long sequences, large batch sizes)
- You can tolerate modest speed reduction for dramatic memory savings
- You need to fit very long sequences in limited GPU memory
- KV cache dominates memory usage (> 50% of peak memory)
- Per-token latency is not the primary concern

### When NOT to Use XQuant

- You're optimizing for absolute latency (XQuant adds recomputation overhead)
- Your sequences are already short (< 4K tokens)
- You have abundant memory (KV caching is sufficient)
- Perplexity degradation cannot be tolerated
- You need cached KVs for speculative decoding or similar

### Common Pitfalls

1. **Over-Aggressive Quantization**: 4-bit quantization sometimes causes perplexity issues. Start with 8-bit.
2. **Not Profiling Memory**: Don't assume X-caching helps without measuring actual peak memory.
3. **Ignoring Recomputation Cost**: Recomputing K,V isn't free. Profile latency before deploying.
4. **Missing Cross-Layer Sharing**: XQuant-CL can cut memory further if layer similarities are high. Analyze before training.
5. **No Baseline Comparison**: Always compare end-to-end performance (accuracy + speed + memory) vs standard KV caching.

## Reference

XQuant (2508.10395): https://arxiv.org/abs/2508.10395

Trade computation for memory by caching quantized layer inputs and recomputing K,V on-the-fly, achieving 7.7-10x memory reduction with minimal accuracy loss and enabling long-sequence inference on constrained hardware.
