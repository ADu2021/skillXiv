---
name: minicpm4-efficient-llms
title: "MiniCPM4: Ultra-Efficient LLMs on End Devices"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.07900"
keywords: [efficient-llms, quantization, sparse-attention, data-filtering, edge-deployment]
description: "Build ultra-efficient language models for edge devices using sparse attention, high-quality data filtering, and ternary quantization, achieving Qwen3-8B performance with 22% of training tokens."
---

# MiniCPM4: Ultra-Efficient LLMs on End Devices

## Core Concept

MiniCPM4 demonstrates that ultra-efficient LLMs can match much larger models through innovations across four dimensions: sparse attention mechanisms (81% sparsity), high-quality data filtering with minimal computational cost, reinforcement learning with load balancing, and cross-platform inference frameworks. The 8B model achieves performance comparable to Qwen3-8B using only 22% of its training tokens, making it practical for edge deployment where computational resources are severely constrained.

## Architecture Overview

- **InfLLM v2 Sparse Attention**: Trainable sparsity achieving 81% reduction in compute for long contexts
- **UltraClean Data Filtering**: Efficient quality verification using near-convergence models
- **ModelTunnel v2 Hyperparameter Optimization**: Automated scaling search based on ScalingBench
- **Chunk-wise RL Rollouts**: Load-balanced reinforcement learning preventing computational bottlenecks
- **P-GPTQ Quantization**: Prefix-aware post-training quantization for extreme compression
- **ArkInfer Framework**: Cross-platform deployment supporting non-NVIDIA hardware

## Implementation

### Step 1: Implement InfLLM v2 Sparse Attention

Create a trainable sparse attention mechanism:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfLLMV2SparseAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, sparsity_target=0.81):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sparsity_target = sparsity_target

        # Learnable sparsity parameters
        self.sparsity_param = nn.Parameter(torch.ones(num_heads) * 0.19)  # 1 - sparsity

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, attention_mask=None):
        """Sparse attention with learnable sparsity patterns"""
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply adaptive sparsity mask
        dense_ratio = torch.sigmoid(self.sparsity_param).mean()  # Learnable dropout probability

        if self.training:
            # During training, learn sparsity pattern
            mask = torch.bernoulli(dense_ratio.expand(self.num_heads))  # (num_heads,)
            mask = mask.view(1, 1, self.num_heads, 1, 1)
        else:
            # During inference, use fixed sparsity
            mask = (dense_ratio > 0.5).float().view(1, 1, self.num_heads, 1, 1)

        # Apply causal mask and sparsity
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) * mask

        if attention_mask is not None:
            combined_mask = combined_mask * attention_mask.unsqueeze(1).unsqueeze(1)

        # Mask out unattended positions
        scores = scores + (1.0 - combined_mask) * -1e9

        # Softmax and apply
        attn = torch.softmax(scores, dim=-1)
        attn = attn * mask  # Enforce sparsity

        output = torch.matmul(attn, v)
        output = output.reshape(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(output)
```

### Step 2: Implement UltraClean Data Filtering

Create efficient quality verification for training data:

```python
import numpy as np
from collections import defaultdict

class UltraCleanFilter:
    def __init__(self, reference_model, vocab_size=32000):
        self.reference_model = reference_model
        self.vocab_size = vocab_size
        self.quality_cache = {}

    def compute_token_probability(self, text, model):
        """Compute model's confidence on text (surrogate for quality)"""
        tokens = model.tokenizer.encode(text)
        token_ids = torch.tensor(tokens).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids=token_ids)
            logits = outputs.logits[0]

        # Compute average log probability
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs[torch.arange(len(tokens)-1), token_ids[0, 1:]]

        return token_probs.mean().item()

    def estimate_data_quality(self, texts, use_early_model=True):
        """
        Estimate quality efficiently using nearly-trained model.

        Key insight: After ~10% of training, model quality estimates stabilize
        and correlate well with final performance.
        """
        quality_scores = []

        for text in texts:
            if text in self.quality_cache:
                quality_scores.append(self.quality_cache[text])
                continue

            # Use reference model (early checkpoint) for fast estimation
            prob = self.compute_token_probability(text, self.reference_model)

            # Quality score inversely correlates with perplexity
            quality = 1.0 / (1.0 + np.exp(prob - 2.0))  # Sigmoid to [0, 1]
            quality_scores.append(quality)
            self.quality_cache[text] = quality

        return np.array(quality_scores)

    def filter_dataset(self, dataset, quality_threshold=0.7):
        """Keep only high-quality examples"""
        texts = [sample['text'] for sample in dataset]
        quality_scores = self.estimate_data_quality(texts)

        filtered = []
        for sample, score in zip(dataset, quality_scores):
            if score > quality_threshold:
                filtered.append({**sample, 'quality_score': score})

        print(f"Filtered {len(dataset)} -> {len(filtered)} samples "
              f"({100*len(filtered)/len(dataset):.1f}% retention)")

        return filtered, quality_scores
```

### Step 3: Chunk-wise RL Training with Load Balancing

Implement reinforcement learning with load-balanced rollouts:

```python
class ChunkwiseRLTrainer:
    def __init__(self, model, chunk_size=256, num_chunks=4):
        self.model = model
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks

    def compute_chunk_rewards(self, output_ids, reference_outputs):
        """
        Compute rewards per chunk for load balancing.

        Instead of full sequence reward, split into chunks to enable
        parallel processing and prevent computational bottlenecks.
        """
        num_tokens = output_ids.shape[1]
        chunk_rewards = []

        for i in range(0, num_tokens, self.chunk_size):
            chunk_end = min(i + self.chunk_size, num_tokens)
            chunk_output = output_ids[:, i:chunk_end]
            chunk_reference = reference_outputs[:, i:chunk_end]

            # Per-chunk reward (e.g., token accuracy)
            correct = (chunk_output == chunk_reference).float()
            chunk_reward = correct.mean(dim=1)

            chunk_rewards.append(chunk_reward)

        return torch.stack(chunk_rewards, dim=1)  # (batch_size, num_chunks)

    def chunk_wise_backward(self, loss, chunk_idx):
        """
        Compute gradients for specific chunk to enable distributed training.

        This allows processing multiple chunks on different devices/workers.
        """
        loss.backward(retain_graph=(chunk_idx < self.num_chunks - 1))

    def training_step(self, input_ids, reference_ids, optimizer):
        """RL training step with chunk-wise load balancing"""
        total_loss = 0

        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=256)

        # Compute per-chunk rewards
        chunk_rewards = self.compute_chunk_rewards(outputs, reference_ids)

        # Process each chunk separately for load balancing
        for chunk_idx in range(self.num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min((chunk_idx + 1) * self.chunk_size, outputs.shape[1])

            # Forward pass for this chunk
            chunk_output = self.model(input_ids=outputs[:, :chunk_end])
            logits = chunk_output.logits[:, chunk_start:chunk_end, :]

            # Compute loss for chunk
            target_ids = reference_ids[:, chunk_start:chunk_end]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   target_ids.reshape(-1))

            # Weight by chunk reward
            chunk_reward = chunk_rewards[:, chunk_idx].mean()
            weighted_loss = loss * chunk_reward

            # Backward pass
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / self.num_chunks
```

### Step 4: P-GPTQ Prefix-Aware Quantization

Implement efficient quantization that preserves prefix performance:

```python
class PGPTQ:
    """Prefix-aware GPTQ quantization"""

    def __init__(self, model, bits=1):
        self.model = model
        self.bits = bits

    def quantize_weights(self, layer, prefix_ids=None):
        """
        Quantize weights while preserving performance on prefix predictions.
        """
        weight = layer.weight
        weight_shape = weight.shape

        # Compute Hessian for importance weighting
        if prefix_ids is not None:
            importance = self.compute_prefix_importance(layer, prefix_ids)
        else:
            importance = torch.ones(weight_shape[1])

        # GPTQ quantization with importance weighting
        if self.bits == 1:
            # Ternary quantization: -1, 0, 1
            scale = importance.sqrt()
            weight_scaled = weight * scale.unsqueeze(0)

            # Quantize to ternary
            quant_weight = torch.sign(weight_scaled).to(torch.int8)
            quant_weight = quant_weight / scale.unsqueeze(0)

        return quant_weight

    def compute_prefix_importance(self, layer, prefix_ids):
        """Compute token importance for prefix sequences"""
        batch_size = prefix_ids.shape[0]

        with torch.no_grad():
            # Compute Hessian trace for prefix
            outputs = layer(prefix_ids)
            loss = outputs.sum()

            # Gradient-based importance
            importance = torch.autograd.grad(loss, layer.parameters(),
                                            create_graph=False)[0].abs().mean(dim=0)

        return importance
```

## Practical Guidance

- **Sparsity Target**: 81% sparsity provides best balance; higher sparsity may hurt quality
- **Data Filtering**: Reduces training time by ~40%; use near-convergence models for efficiency
- **Chunk Size**: 256 tokens typical; adjust based on available memory
- **Quantization**: Ternary (1-bit) works for 2B models; 4-8bit for larger
- **Device Support**: ArkInfer framework enables deployment on various hardware without recompilation
- **Training Tokens**: 8.3T tokens for 8B model; much less than standard baselines
- **Downstream Tasks**: Fine-tune carefully on downstream tasks; sparse attention may need tuning

## Reference

- Sparse attention reduces long-context costs quadratically while maintaining expressiveness
- Data quality filtering shifts compute from quantity to quality, enabling efficiency
- Load-balanced RL prevents any single chunk from bottlenecking training
- P-GPTQ preserves performance on important prefix predictions through importance weighting
