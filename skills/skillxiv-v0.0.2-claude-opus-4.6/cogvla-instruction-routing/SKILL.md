---
name: cogvla-instruction-routing
title: CogVLA Cognition-Aligned Vision-Language-Action via Instruction-Driven Routing
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.21046
keywords: [vision-language-action, token-pruning, instruction-routing, robot-learning, efficiency]
description: "Align VLA efficiency with human cognition through 3-stage progressive routing: instruction-aware aggregation, instruction-irrelevant pruning, and coupled attention for 2.8x inference speedup"
---

# CogVLA: Cognition-Aligned VLA via Instruction-Driven Routing

## Core Concept

CogVLA redesigns Vision-Language-Action models by drawing inspiration from human cognition: we attend selectively to task-relevant visual information. The architecture uses instruction information at multiple stages to compress and prune visual tokens, achieving 2.8x inference speedup and 2.5x lower training costs while maintaining 97.4% success rate on robotic manipulation tasks. The key insight is that most visual tokens are irrelevant to any given instruction—why process them?

## Architecture Overview

- **Stage 1 - EFA-Routing**: Instruction-aware encoder-level aggregation using FiLM modulation
- **Stage 2 - LFP-Routing**: Instruction-conditioned pruning to remove visually grounded but irrelevant tokens
- **Stage 3 - Coupled Attention**: Causal vision-language attention plus bidirectional action decoding
- **Sparsification**: Progressive token reduction from full dual-stream visual to sparse task-relevant representation
- **Cognitive Alignment**: Design principles mirror human selective attention

## Implementation Steps

### Stage 1: Instruction-Aware Encoder Aggregation (EFA-Routing)

Use instruction embeddings to selectively aggregate visual information at the encoder level.

```python
# EFA-Routing: Encoder-FiLM based Aggregation
import torch
from torch import nn
from typing import Tuple

class EFARouter(nn.Module):
    """Instruction-aware visual aggregation using FiLM"""

    def __init__(
        self,
        vision_dim: int = 1024,
        instruction_dim: int = 768,
        num_heads: int = 8
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.instruction_dim = instruction_dim

        # FiLM parameters: learn affine transformation conditioned on instruction
        self.film_gamma = nn.Linear(instruction_dim, vision_dim)
        self.film_beta = nn.Linear(instruction_dim, vision_dim)

        # Dual-stream visual encoder (RGB and depth)
        self.rgb_encoder = nn.TransformerEncoderLayer(
            d_model=vision_dim,
            nhead=num_heads,
            dim_feedforward=4 * vision_dim,
            batch_first=True
        )
        self.depth_encoder = nn.TransformerEncoderLayer(
            d_model=vision_dim,
            nhead=num_heads,
            dim_feedforward=4 * vision_dim,
            batch_first=True
        )

        # Aggregation weights
        self.stream_fusion = nn.Linear(2 * vision_dim, vision_dim)

    def forward(
        self,
        rgb_tokens: torch.Tensor,  # [batch, num_patches, vision_dim]
        depth_tokens: torch.Tensor,  # [batch, num_patches, vision_dim]
        instruction_embed: torch.Tensor  # [batch, instruction_dim]
    ) -> torch.Tensor:
        """
        Aggregate dual-stream visual tokens conditioned on instruction.
        Returns compressed instruction-aware representation.
        """
        batch_size = rgb_tokens.shape[0]

        # Encode streams independently
        rgb_encoded = self.rgb_encoder(rgb_tokens)
        depth_encoded = self.depth_encoder(depth_tokens)

        # FiLM modulation: instruction gates visual features
        gamma = self.film_gamma(instruction_embed)  # [batch, vision_dim]
        beta = self.film_beta(instruction_embed)    # [batch, vision_dim]

        # Apply affine transformation: y = gamma * x + beta
        rgb_modulated = gamma.unsqueeze(1) * rgb_encoded + \
                        beta.unsqueeze(1)
        depth_modulated = gamma.unsqueeze(1) * depth_encoded + \
                          beta.unsqueeze(1)

        # Fuse streams
        combined = torch.cat([rgb_modulated, depth_modulated], dim=-1)
        aggregated = self.stream_fusion(combined)

        # Aggregated tokens: [batch, num_patches, vision_dim]
        # Instruction-aware compression achieved through selective modulation
        return aggregated
```

### Stage 2: Instruction-Driven Token Pruning (LFP-Routing)

Prune tokens that are visually grounded but irrelevant to the task instruction.

```python
# LFP-Routing: Language-guided Fine-grained Pruning
class LFPRouter(nn.Module):
    """Instruction-conditioned pruning of irrelevant visual tokens"""

    def __init__(
        self,
        vision_dim: int = 1024,
        instruction_dim: int = 768,
        pruning_ratio: float = 0.5  # Keep 50% of tokens
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.pruning_ratio = pruning_ratio

        # Compute salience scores based on instruction
        self.token_scorer = nn.Sequential(
            nn.Linear(vision_dim + instruction_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,  # [batch, num_patches, vision_dim]
        instruction_embed: torch.Tensor,  # [batch, instruction_dim]
        action_intent: torch.Tensor = None  # [batch, action_dim] optional
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune tokens irrelevant to instruction + action intent.
        Returns: pruned_tokens, pruning_mask
        """
        batch_size, num_tokens, _ = visual_tokens.shape

        # Expand instruction embedding to match token dimension
        instruction_expanded = instruction_embed.unsqueeze(1).expand(
            batch_size, num_tokens, -1
        )

        # Compute token importance/salience
        token_action_input = torch.cat(
            [visual_tokens, instruction_expanded],
            dim=-1
        )
        salience = self.token_scorer(token_action_input)  # [batch, num_tokens, 1]
        salience = salience.squeeze(-1)  # [batch, num_tokens]

        # Select top-k tokens based on salience
        num_keep = max(1, int(num_tokens * self.pruning_ratio))
        _, top_indices = torch.topk(salience, k=num_keep, dim=1)

        # Create pruning mask
        mask = torch.zeros_like(salience, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)

        # Apply mask: gather selected tokens
        batch_indices = torch.arange(batch_size, device=visual_tokens.device)
        batch_indices = batch_indices.unsqueeze(1).expand(-1, num_keep)

        pruned_tokens = visual_tokens[batch_indices, top_indices]

        return pruned_tokens, mask
```

### Stage 3: Coupled Vision-Language-Action Attention

Implement attention mechanism that coordinates vision, language, and action.

```python
# Coupled attention: causal V-L + bidirectional action decoding
class CoupledAttention(nn.Module):
    """
    Causal vision-language attention (left-to-right)
    + bidirectional action decoding
    """

    def __init__(
        self,
        model_dim: int = 1024,
        num_heads: int = 16,
        max_action_tokens: int = 128
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads

        # Causal self-attention for V-L (standard left-to-right)
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Bidirectional attention for action generation
        # (actions can attend to entire sequence)
        self.action_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(2 * model_dim, model_dim)

    def forward(
        self,
        vision_tokens: torch.Tensor,  # [batch, num_vis_tokens, dim]
        language_tokens: torch.Tensor,  # [batch, num_lang_tokens, dim]
        action_queries: torch.Tensor = None  # [batch, num_action_tokens, dim]
    ) -> torch.Tensor:
        """
        Process vision + language with causal masking,
        then decode actions with full attention.
        """
        # Concatenate vision and language
        vl_tokens = torch.cat([vision_tokens, language_tokens], dim=1)

        # Causal attention: each token attends to itself and preceding tokens
        num_vis = vision_tokens.shape[1]
        num_lang = language_tokens.shape[1]
        seq_len = vl_tokens.shape[1]

        # Create causal mask
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Apply causal V-L attention
        vl_attended, _ = self.causal_attention(
            vl_tokens,
            vl_tokens,
            vl_tokens,
            attn_mask=causal_mask
        )

        # Action decoding: if action queries provided, attend bidirectionally
        if action_queries is not None:
            action_attended, _ = self.action_attention(
                action_queries,  # Queries
                vl_attended,     # Keys and Values (full context)
                vl_attended,
                attn_mask=None  # No mask: bidirectional
            )

            # Fuse causal V-L and action attention
            combined = torch.cat([vl_attended, action_attended], dim=-1)
            output = self.output_proj(combined)
        else:
            output = vl_attended

        return output
```

### Stage 4: Full CogVLA Architecture

Integrate all three routing stages into complete VLA model.

```python
# Complete CogVLA architecture
class CogVLA(nn.Module):
    """Cognition-aligned VLA with progressive routing"""

    def __init__(
        self,
        vision_dim: int = 1024,
        language_dim: int = 768,
        action_dim: int = 512,
        vocab_size: int = 32000,
        num_action_tokens: int = 16
    ):
        super().__init__()

        # Stage 1: Instruction-aware aggregation
        self.efa_router = EFARouter(
            vision_dim=vision_dim,
            instruction_dim=language_dim
        )

        # Stage 2: Instruction-driven pruning
        self.lfp_router = LFPRouter(
            vision_dim=vision_dim,
            instruction_dim=language_dim,
            pruning_ratio=0.5
        )

        # Stage 3: Coupled attention
        self.coupled_attn = CoupledAttention(
            model_dim=vision_dim,
            num_heads=16
        )

        # Language encoding
        self.language_encoder = nn.TransformerEncoderLayer(
            d_model=language_dim,
            nhead=8,
            dim_feedforward=3 * language_dim,
            batch_first=True
        )

        # Action decoder
        self.action_decoder = nn.TransformerDecoderLayer(
            d_model=action_dim,
            nhead=8,
            dim_feedforward=3 * action_dim,
            batch_first=True
        )

        # Action head
        self.action_head = nn.Linear(action_dim, vocab_size)

        self.num_action_tokens = num_action_tokens

    def forward(
        self,
        rgb_image: torch.Tensor,  # [batch, H, W, 3]
        depth_image: torch.Tensor,  # [batch, H, W, 1]
        instruction_tokens: torch.Tensor,  # [batch, instr_len]
        instruction_embeds: torch.Tensor = None  # [batch, instr_len, lang_dim]
    ) -> torch.Tensor:
        """
        Process vision and language, output action logits.
        """
        batch_size = rgb_image.shape[0]

        # Convert images to token patches
        rgb_tokens = self.image_to_tokens(rgb_image)  # [batch, num_patches, vision_dim]
        depth_tokens = self.image_to_tokens(depth_image)

        # Encode instruction
        if instruction_embeds is None:
            instruction_embeds = self.embed_instructions(instruction_tokens)
        instruction_encoded = self.language_encoder(instruction_embeds)
        instruction_mean = instruction_encoded.mean(dim=1)  # [batch, lang_dim]

        # STAGE 1: Instruction-aware aggregation
        aggregated = self.efa_router(
            rgb_tokens,
            depth_tokens,
            instruction_mean
        )

        # STAGE 2: Instruction-driven pruning
        pruned, _ = self.lfp_router(
            aggregated,
            instruction_mean
        )

        # STAGE 3: Coupled attention with action decoding
        # Create action query tokens
        action_queries = torch.randn(
            batch_size,
            self.num_action_tokens,
            pruned.shape[-1]
        ).to(pruned.device)

        # Apply coupled attention
        attended = self.coupled_attn(
            pruned,
            instruction_encoded,
            action_queries
        )

        # Decode to actions
        action_logits = self.action_head(attended[:, -self.num_action_tokens:, :])

        return action_logits  # [batch, num_action_tokens, vocab_size]

    def image_to_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image to vision tokens (simplified)"""
        batch_size = image.shape[0]
        # Flatten image to patches and embed
        patches = image.reshape(batch_size, -1, 3)
        tokens = torch.randn(batch_size, patches.shape[1] // 3, 1024)
        return tokens

    def embed_instructions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed instruction tokens"""
        return torch.randn(tokens.shape[0], tokens.shape[1], 768)
```

### Stage 5: Training with Efficiency Metrics

Train CogVLA with monitoring of speedup and utility preservation.

```python
# Training loop with efficiency tracking
class CogVLATrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.metrics = {"training_loss": [], "inference_time": []}

    def train_step(self, batch) -> float:
        """Single training step"""
        rgb, depth, instruction, actions = batch

        # Forward pass
        action_logits = self.model(rgb, depth, instruction)

        # Loss: cross-entropy on action tokens
        loss = torch.nn.functional.cross_entropy(
            action_logits.reshape(-1, action_logits.shape[-1]),
            actions.reshape(-1)
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_efficiency(self, test_batch) -> dict:
        """Measure inference speed and success rate"""
        import time

        rgb, depth, instruction, actions = test_batch

        # Time inference
        start = time.time()
        with torch.no_grad():
            action_logits = self.model(rgb, depth, instruction)
        inference_time = time.time() - start

        # Compute success rate
        predicted_actions = action_logits.argmax(dim=-1)
        success_rate = (predicted_actions == actions).float().mean().item()

        return {
            "inference_time_ms": inference_time * 1000,
            "success_rate": success_rate,
            "speedup_vs_baseline": 2.8  # From paper
        }
```

## Practical Guidance

### Architecture Choices

- **Pruning Ratio**: Start with 0.5 (keep 50% tokens); adjust based on task complexity
- **FiLM Modulation**: Effective because instruction directly gates visual features
- **Coupled Attention**: Causal for V-L preserves generation order; bidirectional for actions
- **Token Aggregation**: Instruction-aware reduces redundancy

### Performance Benchmarks

- **Training Cost**: 2.5x reduction vs OpenVLA
- **Inference Latency**: 2.8x faster than OpenVLA
- **Success Rate**: 97.4% on LIBERO manipulation, 70% on real robotic tasks
- **Parameter Efficiency**: Same model size, better utilization

### When to Use

- Robotic manipulation and navigation tasks
- Real-time robotics requiring low latency
- Embodied AI with visual observations + language instructions
- Multimodal understanding with task-specific bottlenecks

### When NOT to Use

- General vision-language tasks without action output
- Scenarios requiring full visual context (autonomous driving)
- Offline analysis where latency is not critical
- Domains where instructions are vague or multimodal

### Design Principles

CogVLA mirrors human perception: when given a task, we attend selectively to relevant visual features. This cognitive alignment reduces computational waste while maintaining task performance. The three-stage pipeline progressively narrows focus: first aggregating instruction-aware visual features, then pruning irrelevant tokens, finally coupling vision-language-action reasoning.

## Reference

CogVLA: Cognition-Aligned VLA via Instruction-Driven Routing. arXiv:2508.21046
- https://arxiv.org/abs/2508.21046
