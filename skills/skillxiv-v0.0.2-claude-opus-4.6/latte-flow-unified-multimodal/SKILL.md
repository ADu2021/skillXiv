---
name: latte-flow-unified-multimodal
title: "LaTtE-Flow: Layerwise Timestep-Expert Flow-based Transformer"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.06952"
keywords: [multimodal models, flow matching, efficient generation, image understanding, unified architecture]
description: "Unify image understanding and generation with layerwise timestep experts and residual attention reuse, achieving 6x faster inference than comparable unified models while maintaining competitive performance."
---

# LaTtE-Flow: Layerwise Timestep-Expert Flow-based Transformer

## Core Concept

LaTtE-Flow presents an efficient unified multimodal architecture combining image understanding and generation through flow-matching. The key innovation—Layerwise Timestep Experts—partitions transformer layers into timestep-specific groups, reducing inference complexity from O(L×T') to O(M×T') where M=L/K. Timestep-Conditioned Residual Attention reuses earlier layer computations, enabling 6x faster inference than competing unified models while maintaining competitive performance on both understanding and generation tasks.

## Architecture Overview

- **Unified Multimodal Design**: Integrates frozen pretrained vision-language model with trainable generation pathways for tight understanding-generation coupling
- **Layerwise Timestep Experts**: Partitions L transformer layers into K non-overlapping groups, each specializing in distinct timestep intervals during diffusion
- **Timestep-Conditioned Residual Attention**: Later layers reuse self-attention maps from earlier layers, modulated by timestep embeddings via gating
- **Flow-Matching Generation**: Replaces traditional diffusion with more stable flow-matching formulation for image generation
- **Dual Architecture Variants**: "Couple" preserves frozen VLM; "Blend" shares transformer layers for tighter integration

## Implementation

### Step 1: Layerwise Timestep Expert Architecture

```python
import torch
import torch.nn as nn

class LayerwiseTimestepExpert(nn.Module):
    """
    Partitions transformer layers into timestep-specific groups.
    Each group specializes in specific diffusion timesteps.
    Reduces complexity from O(L*T') to O((L/K)*T').
    """

    def __init__(self, num_layers, num_experts, hidden_dim, num_heads):
        super().__init__()

        self.num_layers = num_layers
        self.num_experts = num_experts
        self.layers_per_expert = num_layers // num_experts

        # Partition layers into timestep-expert groups
        self.expert_groups = nn.ModuleList()

        for expert_idx in range(num_experts):
            group_layers = nn.ModuleList()

            for layer_idx in range(self.layers_per_expert):
                layer = TransformerBlock(hidden_dim, num_heads)
                group_layers.append(layer)

            self.expert_groups.append(group_layers)

        # Timestep-to-expert routing
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts)
        )

    def forward(self, x, timestep):
        """
        Route to appropriate expert group based on timestep.
        Only execute M=L/K layers instead of all L layers.
        """

        # Embed timestep and determine expert assignment
        t_embed = self.timestep_embedding(timestep.float().unsqueeze(-1))
        expert_idx = torch.argmax(t_embed, dim=-1).item()

        # Early timesteps (noise-heavy): use first expert (broader receptive field)
        # Late timesteps (refinement): use later experts (detail focus)
        # Map timestep [0, 1] to expert index [0, num_experts-1]
        normalized_t = (1.0 - timestep.item()) * self.num_experts
        expert_idx = min(int(normalized_t), self.num_experts - 1)

        # Execute only this expert group's layers
        selected_expert = self.expert_groups[expert_idx]

        for layer in selected_expert:
            x = layer(x)

        return x

class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm1(x)

        # Feed-forward
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm2(x)

        return x
```

### Step 2: Timestep-Conditioned Residual Attention

```python
class TimestepConditionedResidualAttention(nn.Module):
    """
    Reuses attention maps from earlier layers, conditioned on timestep.
    Enables parameter sharing and computation reuse across layers.
    """

    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Store attention maps from earlier layers
        self.attention_cache = {}

        # Gating mechanism: modulate attention by timestep
        self.timestep_gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, num_heads)  # Per-head gating
        )

    def forward(self, x, layer_idx, timestep, attention_cache=None):
        """
        Forward pass with residual attention reuse.
        Layer L reuses attention from layer L-1, gated by timestep.
        """

        # Compute local attention
        local_attn = self._compute_attention(x)  # [batch, seq, num_heads]

        # Retrieve cached attention from previous layer
        if attention_cache is not None and layer_idx > 0:
            cached_attn = attention_cache.get(layer_idx - 1, None)

            if cached_attn is not None:
                # Compute timestep-dependent gate
                t_embed = timestep.float().unsqueeze(-1)  # [batch, 1]
                gate = torch.sigmoid(self.timestep_gate(t_embed))  # [batch, num_heads]

                # Gate:1.0 early (early timesteps favor local computation)
                # Gate:0.0 late (late timesteps leverage cached attention)
                gate = gate.unsqueeze(1)  # [batch, 1, num_heads]

                # Blend local and cached attention
                blended_attn = gate * local_attn + (1 - gate) * cached_attn

                # Store for next layer
                attention_cache[layer_idx] = blended_attn

                return blended_attn

        # Store for next layer
        if attention_cache is not None:
            attention_cache[layer_idx] = local_attn

        return local_attn

    def _compute_attention(self, x):
        """Compute multi-head attention weights."""
        # Simplified: actual implementation uses full attention mechanism
        batch, seq_len, dim = x.shape
        scores = torch.matmul(x, x.transpose(-2, -1)) / (dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        return attn_weights
```

### Step 3: Flow-Matching Image Generation

```python
class FlowMatchingGenerator(nn.Module):
    """
    Generates images using flow-matching instead of traditional diffusion.
    More stable training trajectory than reverse diffusion.
    """

    def __init__(self, model_dim, vocab_size=256):
        super().__init__()

        self.model_dim = model_dim
        self.vocab_size = vocab_size

        # Transformer for flow prediction
        self.flow_predictor = TransformerBlock(model_dim, num_heads=8)

        # Output head: predict image tokens
        self.output_head = nn.Linear(model_dim, vocab_size)

    def forward(self, latent, prompt_embedding, timestep):
        """
        Predict flow (vector field) that moves noise towards image.
        Flow-matching: directly learn velocity field dX/dt.
        """

        # Concatenate latent with prompt
        x = torch.cat([latent, prompt_embedding], dim=-1)

        # Predict flow
        flow = self.flow_predictor(x)

        # Predict image tokens from flow
        logits = self.output_head(flow)

        return logits

    def generate(self, prompt_embedding, num_steps=50, latent_dim=512):
        """
        Generate image via flow-matching.
        Integration from t=0 (noise) to t=1 (image).
        """

        # Start from random latent
        x_t = torch.randn(1, latent_dim, self.model_dim)

        # ODE solver: integrate flow
        for step in range(num_steps):
            t = torch.tensor([step / num_steps])

            # Predict flow at this timestep
            flow = self.flow_predictor(torch.cat([x_t, prompt_embedding], dim=-1))

            # Simple Euler integration step
            dt = 1.0 / num_steps
            x_t = x_t + flow * dt

        # Decode latent to image tokens
        logits = self.output_head(x_t)
        image_tokens = torch.argmax(logits, dim=-1)

        # Decode tokens to image
        image = self._decode_tokens(image_tokens)

        return image

    def _decode_tokens(self, tokens):
        """Decode image tokens to pixel values."""
        # Would use learned codebook (VQVAE, etc.)
        return tokens.float() / self.vocab_size
```

### Step 4: Unified LaTtE-Flow Model

```python
class LatteFlow(nn.Module):
    """
    Unified multimodal model combining vision-language understanding
    with efficient flow-based generation via layerwise timestep experts.
    """

    def __init__(self, pretrained_vlm, model_dim=768, num_experts=4):
        super().__init__()

        # Frozen pretrained VLM (understanding)
        self.vlm = pretrained_vlm
        for param in self.vlm.parameters():
            param.requires_grad = False

        self.model_dim = model_dim

        # Trainable generation components
        self.timestep_experts = LayerwiseTimestepExpert(
            num_layers=28,  # From Qwen2-VL-2B
            num_experts=num_experts,
            hidden_dim=model_dim,
            num_heads=12
        )

        self.residual_attention = TimestepConditionedResidualAttention(
            hidden_dim=model_dim,
            num_heads=12,
            num_layers=28
        )

        self.flow_generator = FlowMatchingGenerator(model_dim)

        # Image encoder (compression)
        self.image_encoder = ImageEncoder(model_dim)

    def forward_understanding(self, image, text):
        """Vision-language understanding using frozen VLM."""
        with torch.no_grad():
            understanding = self.vlm.encode(image, text)
        return understanding

    def forward_generation(self, prompt, image_resolution=(256, 256)):
        """Efficient image generation with layerwise experts."""

        # Encode prompt
        prompt_embedding = self.vlm.text_encoder(prompt)

        # Generate via flow-matching with timestep experts
        generated_image = self.flow_generator.generate(
            prompt_embedding,
            num_steps=50
        )

        return generated_image

    def forward(self, image=None, text=None, prompt=None, task='understanding'):
        """
        Unified forward pass supporting both understanding and generation.
        """

        if task == 'understanding':
            return self.forward_understanding(image, text)
        elif task == 'generation':
            return self.forward_generation(prompt)

class ImageEncoder(nn.Module):
    """Efficient image encoder with 32x downsampling."""

    def __init__(self, latent_dim):
        super().__init__()
        # Simplified: DeepCompression Autoencoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 4, stride=2, padding=1)
        )

    def forward(self, image):
        return self.encoder(image)
```

## Practical Guidance

**Architecture Selection**:
- **Couple variant**: Preserve frozen VLM; simpler, less coupled understanding-generation
- **Blend variant**: Share transformer layers; tighter integration, more parameters

**Training Configuration**:
- Dataset: 1.2M ImageNet images at 256×256 resolution
- Batch size: 2,048 for stable training
- Steps: 240K total (warm-up + main + fine-tune)
- Learning rate: 1e-4 with cosine annealing

**Inference Optimization**:
- Layerwise experts: Execute only 7 layers per timestep (vs 28 for standard models)
- Residual attention: Reuse 60% of attention maps from previous layers
- Flow-matching: Replaces 50 diffusion steps with 50 ODE integration steps (comparable cost, better quality)
- Speedup: 6x faster than comparable unified models (Unified-7B, etc.)

**Performance Targets**:
- Understanding: Competitive with frozen Qwen2-VL-2B on multimodal benchmarks
- Generation: FID 28-32 on ImageNet-50K (competitive with recent diffusion models)
- Efficiency: <2 seconds inference on single GPU (512x512 resolution)

**When to Use LaTtE-Flow**:
- Joint understanding-generation applications
- Edge deployment (efficiency critical)
- Multimodal reasoning (image analysis + generation)
- Real-time applications (low latency requirement)

## Reference

- Layerwise experts: Mixture-of-experts variant specialized by timestep rather than task
- Flow-matching: Learned velocity field replaces reverse diffusion scheduling
- Residual attention: Parameter reduction via attention map reuse across layers
- Timestep conditioning: Gating mechanisms adapt computation to diffusion stage
