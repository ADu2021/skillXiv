---
name: exaone-unified-reasoning-language-model
title: "EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.11407"
keywords: [Reasoning, Language Models, Hybrid Attention, RLHF, Long Context, Multilingual]
description: "Build unified LLMs supporting both fast standard inference and slow reasoning modes through hybrid attention and adaptive post-training. Combine non-reasoning and reasoning capabilities in a single model with 128K context windows and tool use. Use when you need models balancing speed and reasoning depth across diverse tasks and languages."
---

# EXAONE 4.0: Unified Models Combining Reasoning and Speed

Most reasoning models optimize for problem-solving depth at the cost of general-purpose capability, while standard models prioritize speed over reasoning. EXAONE 4.0 unifies both modes: a single model that can operate in standard mode for fast inference or reasoning mode for complex problems, with task-specific hyperparameter switching at inference time. The architecture combines hybrid attention (3:1 local-to-global ratio), two-stage context extension to 128K tokens, and AGAPO reinforcement learning that improves upon GRPO by removing clipping and using asymmetric sampling.

The key innovation is making reasoning and non-reasoning modes mutually compatible within a single architecture, avoiding the forced choice between depth and speed. Training ratios (1.5:1 reasoning-to-non-reasoning data) prevent mode confusion while maintaining distinct inference behaviors.

## Core Concept

EXAONE 4.0 operates as a single, mode-agnostic model with inference-time parameter switching. Standard mode uses greedy decoding with low temperatures for fast generation. Reasoning mode increases sampling temperature, allows longer output, and employs different prompt structures that activate reasoning behaviors learned during post-training. The model learns both capabilities via carefully balanced training data where reasoning examples introduce deep thought but don't dominate.

Hybrid attention (local + global) provides efficiency gains without sacrificing long-range reasoning. Two-stage context extension (4K → 32K → 128K) prevents training instability while reaching long-context capabilities. AGAPO (Asymmetric Group Relative Policy Optimization) training improves upon GRPO by removing clipped objectives and incorporating asymmetric sampling of incorrect responses.

## Architecture Overview

- **Hybrid Attention Module**: 3:1 ratio of local (4K sliding window) to global attention, enabling both local pattern matching and long-range reasoning
- **QK-Reorder-LN Normalization**: Custom norm variant improving attention stability
- **Two-Stage Context Extension**: Progressive training (4K → 32K → 128K) with Needle-In-Haystack validation at each stage
- **AGAPO Optimization Algorithm**: Enhanced GRPO removing clip objectives, asymmetric incorrect response sampling, group/global advantage calculations
- **Hybrid Reward System**: Combines verifiable (correctness), preference (human choice), consistency (logical coherence), and conciseness signals via SimPER framework
- **Mode-Switching Decoder**: Parametrized generation strategy (standard vs. reasoning) via inference-time config

## Implementation

### Hybrid Attention Architecture

Implement the 3:1 local-to-global attention mechanism.

```python
import torch
import torch.nn as nn
from typing import Optional

class HybridAttention(nn.Module):
    """Hybrid attention with 3:1 ratio of local to global attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int = 4096,
        local_to_global_ratio: float = 3.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.ratio = local_to_global_ratio

        # Projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Local and global attention projections
        self.local_heads = int(num_heads * (local_to_global_ratio / (1 + local_to_global_ratio)))
        self.global_heads = num_heads - self.local_heads

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def local_attention(self, q, k, v, window_size: int):
        """
        Apply sliding window attention to Q, K, V.

        Args:
            q, k, v: (batch, seq_len, hidden_dim)
            window_size: Size of local window

        Returns:
            attended: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = q.shape

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.local_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.local_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.local_heads, -1).transpose(1, 2)

        # Compute local attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)

        # Create sliding window mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=window_size + 1
        )
        mask |= torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        ) & ~torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=window_size + 1 - window_size
        )

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Apply attention
        attn_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, v)

        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, -1)

        return attended

    def global_attention(self, q, k, v):
        """
        Apply standard (full sequence) attention to remaining heads.

        Args:
            q, k, v: (batch, seq_len, hidden_dim)

        Returns:
            attended: (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = q.shape

        # Reshape for global attention heads
        q = q.view(batch_size, seq_len, self.global_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.global_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.global_heads, -1).transpose(1, 2)

        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, v)

        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, -1)

        return attended

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply hybrid attention combining local and global.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Apply local attention to ratio*(num_heads) heads
        local_out = self.local_attention(q, k, v, self.window_size)

        # Apply global attention to remaining heads
        global_out = self.global_attention(q, k, v)

        # Concatenate outputs
        output = torch.cat([local_out, global_out], dim=-1)

        # Project output
        output = self.out_proj(output)

        return output

class QKReorderLN(nn.Module):
    """Custom QK-Reorder-LN normalization for improved attention stability."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        Reorder Q and K before LayerNorm for stability.

        Args:
            q: Query tensor
            k: Key tensor

        Returns:
            Normalized (q, k)
        """
        # Reorder: move head dimension for better norm computation
        batch_size, seq_len, hidden = q.shape
        head_dim = hidden // 8  # Assuming 8 heads

        # Reshape to expose head dimension
        q = q.view(batch_size, seq_len, 8, -1)
        k = k.view(batch_size, seq_len, 8, -1)

        # Normalize per head
        q_mean = q.mean(dim=-1, keepdim=True)
        q_std = q.std(dim=-1, keepdim=True) + self.eps
        q_norm = (q - q_mean) / q_std

        k_mean = k.mean(dim=-1, keepdim=True)
        k_std = k.std(dim=-1, keepdim=True) + self.eps
        k_norm = (k - k_mean) / k_std

        # Reshape back
        q_norm = q_norm.view(batch_size, seq_len, hidden)
        k_norm = k_norm.view(batch_size, seq_len, hidden)

        # Scale and shift
        q_out = q_norm * self.weight + self.bias
        k_out = k_norm * self.weight + self.bias

        return q_out, k_out
```

### Two-Stage Context Extension

Progressively extend context length from 4K to 128K with validation.

```python
class ContextExtensionTrainer:
    """Progressive context length training with validation."""

    def __init__(self, model, tokenizer, initial_context: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.initial_context = initial_context
        self.stages = [
            {'context': 4096, 'name': 'base'},
            {'context': 32768, 'name': 'intermediate'},
            {'context': 131072, 'name': 'extended'}
        ]

    def needle_in_haystack_test(
        self,
        needle: str,
        haystack_length: int,
        needle_position_ratio: float = 0.5
    ) -> bool:
        """
        Test if model can retrieve needle from long haystack.
        Validates context extension at each stage.

        Args:
            needle: Text to hide in context
            haystack_length: Length of context to fill
            needle_position_ratio: Where to place needle (0.5 = middle)

        Returns:
            True if model retrieves needle correctly
        """
        # Create haystack (e.g., repeated facts)
        filler = "The color of the sky is blue. " * (haystack_length // 30)

        # Insert needle at position
        needle_pos = int(len(filler) * needle_position_ratio)
        haystack = filler[:needle_pos] + needle + filler[needle_pos:]

        # Truncate to target length
        haystack = haystack[:haystack_length]

        # Create retrieval prompt
        prompt = f"{haystack}\n\nQuestion: Find the needle in this text and repeat it exactly.\n\nAnswer:"

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        if len(input_ids[0]) > haystack_length:
            # Truncate if prompt exceeds limit
            input_ids = input_ids[:, :haystack_length + 50]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=len(self.tokenizer.encode(needle)) + 10,
                temperature=0.0
            )

        output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Check if needle appears in output
        return needle in output

    def train_stage(
        self,
        stage: dict,
        training_data,
        validation_data,
        num_epochs: int = 3
    ) -> dict:
        """
        Train model at specific context length.

        Args:
            stage: {'context': length, 'name': 'stage_name'}
            training_data: Dataset with examples
            validation_data: Validation dataset
            num_epochs: Training epochs for this stage

        Returns:
            Stage metrics
        """
        context_len = stage['context']
        self.model.max_position_embeddings = context_len

        # Adjust training inputs to context length
        from transformers import TextIterableDataset

        print(f"\nTraining stage: {stage['name']} (context={context_len})")

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch in training_data:
                # Truncate to context length
                input_ids = batch['input_ids'][:, :context_len]

                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss

                loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(training_data)
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

        # Validate with needle test
        accuracy = 0
        for needle_pos in [0.25, 0.5, 0.75]:
            if self.needle_in_haystack_test(
                "NEEDLE",
                int(context_len * 0.8),
                needle_pos
            ):
                accuracy += 1

        needle_accuracy = accuracy / 3

        print(f"  Needle-In-Haystack accuracy: {needle_accuracy:.1%}")

        return {
            'stage': stage['name'],
            'context': context_len,
            'final_loss': avg_loss,
            'needle_accuracy': needle_accuracy
        }

    def run_full_extension(self, training_data, validation_data):
        """Run all context extension stages."""
        results = []

        for stage in self.stages:
            stage_results = self.train_stage(stage, training_data, validation_data)
            results.append(stage_results)

        return results
```

### AGAPO: Asymmetric Group Relative Policy Optimization

Enhanced RLHF training removing clipped objectives and using asymmetric sampling.

```python
class AGAPO:
    """AGAPO: Improved GRPO with asymmetric sampling and group advantage."""

    def __init__(self, model, learning_rate: float = 1e-6, beta_kl: float = 0.01):
        self.model = model
        self.reference_model = model  # Freeze for KL computation
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.beta_kl = beta_kl

    def compute_group_advantage(
        self,
        rewards: torch.Tensor,
        group_size: int = 8
    ) -> torch.Tensor:
        """
        Compute group advantage: individual score minus group mean.
        (Instead of clip-based GRPO)

        Args:
            rewards: (batch_size,) tensor of rewards
            group_size: Size of comparison group

        Returns:
            advantages: (batch_size,) advantage per sample
        """
        advantages = []

        for i in range(0, len(rewards), group_size):
            group_rewards = rewards[i:i+group_size]
            group_mean = group_rewards.mean()

            # Advantage: reward - group_mean (no clipping)
            group_advantages = group_rewards - group_mean

            advantages.extend(group_advantages.tolist())

        return torch.tensor(advantages)

    def asymmetric_sampling(
        self,
        model_outputs: list,
        rewards: torch.Tensor,
        correct_ratio: float = 0.5,
        num_samples: int = 32
    ) -> tuple:
        """
        Sample correct and incorrect responses asymmetrically.
        Use more incorrect samples for contrastive learning.

        Args:
            model_outputs: All generated responses
            rewards: Reward for each response
            correct_ratio: How many correct samples to keep (e.g., 0.3)
            num_samples: Total samples to select

        Returns:
            (correct_samples, incorrect_samples)
        """
        # Separate correct and incorrect
        correct_indices = torch.where(rewards > 0.5)[0]
        incorrect_indices = torch.where(rewards <= 0.5)[0]

        # Sample asymmetrically
        num_correct = max(1, int(num_samples * correct_ratio))
        num_incorrect = num_samples - num_correct

        correct_sample_idx = torch.randperm(len(correct_indices))[:num_correct]
        incorrect_sample_idx = torch.randperm(len(incorrect_indices))[:num_incorrect]

        correct_samples = [model_outputs[i] for i in correct_indices[correct_sample_idx]]
        incorrect_samples = [model_outputs[i] for i in incorrect_indices[incorrect_sample_idx]]

        return correct_samples, incorrect_samples

    def training_step(
        self,
        prompts: list,
        rewards: torch.Tensor,
        model_outputs: list
    ) -> dict:
        """
        Single AGAPO training step.

        Args:
            prompts: Input prompts
            rewards: Scalar rewards per output
            model_outputs: Generated responses

        Returns:
            Loss components
        """
        self.model.train()

        # Compute group advantages (no clipping)
        advantages = self.compute_group_advantage(rewards)

        # Sample correct and incorrect asymmetrically
        correct, incorrect = self.asymmetric_sampling(model_outputs, rewards)

        total_loss = 0.0

        for prompt, correct_response, advantage in zip(
            prompts, correct, advantages
        ):
            # Forward pass for this example
            input_ids = self.model.tokenizer.encode(prompt + correct_response, return_tensors='pt')

            outputs = self.model(input_ids)
            logits = outputs.logits

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Policy loss: maximize advantage (no clipping, unlike GRPO)
            policy_loss = -advantage * log_probs.mean()

            # KL divergence penalty
            with torch.no_grad():
                ref_logits = self.reference_model(input_ids).logits
                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

            kl_loss = torch.nn.functional.kl_div(log_probs, ref_log_probs.exp(), reduction='mean')

            # Combined loss
            loss = policy_loss + self.beta_kl * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss / len(prompts)
        }
```

### Mode-Switching Inference

Switch between standard and reasoning modes at inference time.

```python
class ModeSwitchingDecoder:
    """Generate in standard or reasoning mode."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        mode: str = 'standard',
        max_new_tokens: int = 100
    ) -> str:
        """
        Generate with mode-specific hyperparameters.

        Args:
            prompt: Input text
            mode: 'standard' (fast) or 'reasoning' (deep)
            max_new_tokens: Generation length

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        if mode == 'standard':
            # Fast generation: greedy, short output
            config = {
                'max_new_tokens': min(max_new_tokens, 256),
                'temperature': 0.0,  # Greedy
                'top_p': 1.0,
                'do_sample': False
            }

        elif mode == 'reasoning':
            # Reasoning mode: sampling, long output, high temperature
            config = {
                'max_new_tokens': min(max_new_tokens, 64000),  # Allow longer
                'temperature': 0.6,
                'top_p': 0.95,
                'do_sample': True,
                'num_beams': 1  # Disable beam search for speed
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **config)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

## Practical Guidance

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hybrid Attention Ratio | 3:1 | 3 local heads per global head |
| Local Window Size | 4096 tokens | 4K sliding window for local attention |
| Context Stages | 4K → 32K → 128K | Progressive extension with checkpoints |
| Reasoning-to-Non-Reasoning Data Ratio | 1.5:1 | Prevents mode confusion; adjustable per domain |
| AGAPO Learning Rate | 1e-6 | Very low for stable RL training |
| Beta KL (KL penalty) | 0.01 | Prevents divergence from reference |
| Group Size | 8 | For advantage computation |
| Temperature (standard mode) | 0.0 | Greedy for reproducibility |
| Temperature (reasoning mode) | 0.6 | Sampling for diversity in reasoning |

### When to Use

- Building general-purpose models needing both speed and reasoning capability
- Tasks requiring flexible inference (sometimes fast, sometimes deep)
- Applications supporting long documents (>32K tokens)
- Multilingual systems (EXAONE supports 3+ languages)
- Systems where model size constraints prevent separate reasoning/standard models

### When NOT to Use

- Real-time systems requiring consistent latency (reasoning mode is slow)
- Specialized domains where dedicated models outperform unified approaches
- Tasks where reasoning isn't needed (pure speed is priority)
- Models with hard context limitations (<4K tokens)

### Common Pitfalls

- **Imbalanced training ratios**: Using 1:1 or 2:1 instead of 1.5:1 causes mode confusion; validate ratio on held-out tasks
- **Skipping intermediate stages**: Jumping from 4K directly to 128K causes training instability; include 32K intermediate checkpoint
- **Forgetting inference parameters**: Standard mode needs temperature=0; reasoning mode needs temperature=0.6; validate per task
- **Underestimating KL penalty**: β_kl < 0.001 allows divergence; β_kl > 0.1 suppresses learning; start at 0.01
- **Ignoring needle tests**: Don't assume context extension works; validate with Needle-In-Haystack at each stage
- **AGAPO vs GRPO confusion**: AGAPO removes clipping (no max(0, ...)), uses asymmetric sampling; don't mix algorithms

## Reference

Park, J., Kim, S., Lee, D., et al. (2024). EXAONE 4.0: Unified Large Language Models Integrating Non-reasoning and Reasoning Modes. arXiv preprint arXiv:2507.11407.
