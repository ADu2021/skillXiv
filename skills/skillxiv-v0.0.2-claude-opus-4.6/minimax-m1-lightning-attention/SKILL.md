---
name: minimax-m1-lightning-attention
title: "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.13585"
keywords: [reasoning, test-time-compute, mixture-of-experts, lightning-attention, reinforcement-learning]
description: "Hybrid-attention MoE reasoning model supporting 1M token context and 80K token generation, combining lightning attention with CISPO RL algorithm for efficient scaling."
---

# MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention

## Core Concept

MiniMax-M1 is an open-weight reasoning model combining a hybrid Mixture-of-Experts architecture with lightning attention mechanism. With 456B total parameters and 45.9B activated per token, it supports 1 million token context length and 80K token generation. A novel RL algorithm called CISPO (Clipped IS-weight Policy Optimization) achieves the efficiency gains of DAPO with 50% fewer training steps by clipping importance-sampling weights rather than token probabilities. The model excels through large-scale reinforcement learning on diverse reasoning tasks completed in three weeks using 512 H800 GPUs.

## Architecture Overview

- **Hybrid Attention Design**: Combines transnormer blocks with lightning attention and periodic softmax attention for near-linear scaling
- **Mixture-of-Experts**: Sparse routing with 45.9B activated parameters enabling efficient scaling
- **CISPO Algorithm**: Clips importance sampling weights rather than token updates, matching DAPO with less training
- **Multi-Stage Training**: Continual pretraining → supervised fine-tuning → large-scale RL
- **Diverse RL Data**: Verifiable tasks (math, coding, reasoning) + general domain with model-based rewards

## Implementation

### Step 1: Implement Lightning Attention

Create efficient attention mechanism with near-linear complexity:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightningAttention(nn.Module):
    """
    Lightning attention: efficient attention with near-linear complexity.
    Combines linear recurrence with periodic exact attention.
    """
    def __init__(self, hidden_size, num_heads, use_rms_norm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # RMS normalization
        if use_rms_norm:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = nn.LayerNorm(self.head_dim)
            self.norm_k = nn.LayerNorm(self.head_dim)

        # Recurrent weight for linear attention
        self.beta = nn.Parameter(torch.ones(num_heads) * 0.5)

    def forward(self, x, attention_mask=None, use_exact_attention=False):
        """
        Args:
            x: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len, seq_len] or None
            use_exact_attention: force full attention (for periodic blocks)

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        # Normalize
        q = self.norm_q(q)
        k = self.norm_k(k)

        if use_exact_attention:
            # Standard softmax attention for periodic blocks
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores / (self.head_dim ** 0.5)

            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, v)

        else:
            # Linear attention: recurrent form
            # o_t = (sum_i beta * k_i @ v_i) / (sum_i beta * k_i)
            output = self._linear_attention(q, k, v, batch, seq_len)

        # Reshape and project out
        output = output.reshape(batch, seq_len, self.hidden_size)
        output = self.o_proj(output)

        return output

    def _linear_attention(self, q, k, v, batch, seq_len):
        """
        Linear attention computation.
        """
        # Activation function (elu + 1 for numerical stability)
        k_activated = F.elu(k) + 1
        q_activated = F.elu(q) + 1

        # Recurrent computation
        outputs = []

        for t in range(seq_len):
            q_t = q_activated[:, t]  # [batch, num_heads, head_dim]
            k_t = k_activated[:, t]  # [batch, num_heads, head_dim]
            v_t = v[:, t]  # [batch, num_heads, head_dim]

            if t == 0:
                # Initialize
                numerator = torch.einsum('bnh,bnd->bhd', k_t, v_t)
                denominator = k_t.sum(dim=1, keepdim=True)
            else:
                # Update with exponential moving average
                beta_t = self.beta.unsqueeze(0).unsqueeze(-1)
                numerator = beta_t * numerator + (1 - beta_t) * torch.einsum(
                    'bnh,bnd->bhd', k_t, v_t
                )
                denominator = beta_t * denominator + (1 - beta_t) * k_t

            # Compute output
            output_t = torch.einsum('bnh,bhd->bnd', q_t, numerator) / (
                denominator.sum(dim=1, keepdim=True) + 1e-8
            )
            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)  # [batch, seq_len, num_heads, head_dim]

        return output
```

### Step 2: Implement Hybrid Attention Block

Combine lightning and softmax attention strategically:

```python
class HybridAttentionBlock(nn.Module):
    """
    Block alternating between lightning and exact attention.
    Lightning for efficiency, softmax periodically for stability.
    """
    def __init__(self, hidden_size, num_heads, use_exact_every_n=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_exact_every_n = use_exact_every_n

        self.lightning_attn = LightningAttention(hidden_size, num_heads)

    def forward(self, x, step=0):
        """
        Args:
            x: [batch, seq_len, hidden_size]
            step: block step number (to determine exact vs lightning)
        """
        # Alternate between lightning and exact attention
        use_exact = (step % self.use_exact_every_n == 0)

        if use_exact:
            return self.lightning_attn(x, use_exact_attention=True)
        else:
            return self.lightning_attn(x, use_exact_attention=False)
```

### Step 3: Implement CISPO RL Algorithm

Efficient policy optimization clipping IS-weights instead of log-probs:

```python
class CISPOTrainer:
    """
    Clipped IS-weight Policy Optimization.
    More efficient than DAPO: clips importance samples rather than token updates.
    """
    def __init__(self, model, clip_ratio=0.2, target_kl=0.01):
        self.model = model
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

    def compute_importance_weights(self, log_probs_new, log_probs_old):
        """
        Compute importance sampling weights.

        Args:
            log_probs_new: [batch, seq_len] log probs from updated policy
            log_probs_old: [batch, seq_len] log probs from old policy

        Returns:
            is_weights: [batch, seq_len]
        """
        # Importance sampling ratio
        log_ratio = log_probs_new - log_probs_old
        is_weights = torch.exp(log_ratio)

        return is_weights

    def clip_importance_weights(self, is_weights, advantages):
        """
        Clip IS weights rather than policy updates.
        This is more sample-efficient than standard PPO clipping.

        Args:
            is_weights: [batch, seq_len] importance weights
            advantages: [batch, seq_len] advantage estimates

        Returns:
            clipped_loss: scalar loss
        """
        # Clip importance weights
        clipped_is_weights = torch.clamp(
            is_weights,
            1 - self.clip_ratio,
            1 + self.clip_ratio
        )

        # Compute loss with clipping
        # L = -min(is_weights * advantage, clipped_is_weights * advantage)
        unclipped_loss = -is_weights * advantages
        clipped_loss = -clipped_is_weights * advantages

        loss = torch.max(unclipped_loss, clipped_loss).mean()

        return loss

    def training_step(self, batch_prompts, batch_responses, batch_advantages,
                     batch_log_probs_old, batch_rewards):
        """
        Single CISPO training step.
        """
        # Forward pass: get new log probs
        outputs = self.model(batch_prompts, batch_responses)
        log_probs_new = outputs.log_probs  # [batch, seq_len]

        # Compute importance weights
        is_weights = self.compute_importance_weights(
            log_probs_new, batch_log_probs_old
        )

        # Clip and compute loss
        policy_loss = self.clip_importance_weights(is_weights, batch_advantages)

        # KL divergence penalty (optional)
        kl_div = (batch_log_probs_old - log_probs_new).mean()
        kl_loss = torch.clamp(kl_div - self.target_kl, min=0.0)

        # Total loss
        total_loss = policy_loss + 0.01 * kl_loss

        return total_loss, {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'is_weights_mean': is_weights.mean().item()
        }
```

### Step 4: Implement Diverse Reward Models

Support multiple reward types for curriculum learning:

```python
class MultiTaskRewardModel(nn.Module):
    """
    Reward model supporting multiple task categories:
    - Verifiable: math, coding (binary correctness)
    - General: quality scoring via model-based rewards
    """
    def __init__(self, task_type='mixed'):
        super().__init__()
        self.task_type = task_type

    def compute_reward(self, responses, task_labels, references=None):
        """
        Args:
            responses: [batch] generated responses
            task_labels: [batch] task category
            references: [batch] ground truth (for verifiable tasks)

        Returns:
            rewards: [batch] reward scores in [0, 1]
        """
        rewards = []

        for i, (response, task) in enumerate(zip(responses, task_labels)):
            if task in ['math', 'coding']:
                # Verifiable tasks: binary reward
                if references is not None:
                    is_correct = self._check_correctness(response, references[i])
                    reward = 1.0 if is_correct else 0.0
                else:
                    reward = 0.5
            else:
                # General tasks: use learned model-based reward
                reward = self._score_response(response)

            rewards.append(reward)

        return torch.tensor(rewards)

    def _check_correctness(self, response, reference):
        """Check if response matches reference (math/code)"""
        response_clean = response.strip().lower()
        reference_clean = reference.strip().lower()

        # Extract final answer (simple heuristic)
        import re
        response_nums = re.findall(r'-?\d+\.?\d*', response_clean)
        ref_nums = re.findall(r'-?\d+\.?\d*', reference_clean)

        if response_nums and ref_nums:
            return float(response_nums[-1]) == float(ref_nums[-1])

        return response_clean == reference_clean

    def _score_response(self, response):
        """Score general quality (0-1)"""
        # Simple heuristics: length, coherence, etc.
        length_score = min(len(response) / 1000, 1.0)
        return length_score * 0.5 + 0.5  # Between 0.5 and 1.0
```

### Step 5: Training Loop with Curriculum Learning

Implement curriculum-based RL training:

```python
def train_minimax_m1(model, train_dataloader, num_epochs=10,
                     device='cuda'):
    """
    Train MiniMax-M1 with CISPO and curriculum learning.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cispo_trainer = CISPOTrainer(model)
    reward_model = MultiTaskRewardModel()

    # Curriculum: start with verifiable, then mix in general
    task_schedule = {
        0: {'verifiable_frac': 1.0, 'general_frac': 0.0},
        5: {'verifiable_frac': 0.7, 'general_frac': 0.3},
        10: {'verifiable_frac': 0.5, 'general_frac': 0.5}
    }

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_metrics = {}

        # Get curriculum weights for this epoch
        curriculum = task_schedule.get(epoch, {'verifiable_frac': 0.5,
                                               'general_frac': 0.5})

        for batch_idx, batch in enumerate(train_dataloader):
            # Filter batch by curriculum
            verifiable_mask = batch['task'].isin(['math', 'coding'])
            verifiable_frac = curriculum['verifiable_frac']

            # Sample batch following curriculum
            if torch.rand(1) < verifiable_frac:
                filtered_batch = batch[verifiable_mask]
            else:
                filtered_batch = batch[~verifiable_mask]

            if len(filtered_batch) == 0:
                continue

            # Compute rewards
            rewards = reward_model.compute_reward(
                filtered_batch['responses'],
                filtered_batch['task'],
                filtered_batch.get('references')
            )

            # Compute advantages (simplified: subtract baseline)
            baseline = rewards.mean()
            advantages = rewards - baseline

            # CISPO training step
            loss, metrics = cispo_trainer.training_step(
                filtered_batch['prompts'],
                filtered_batch['responses'],
                advantages.to(device),
                filtered_batch['old_log_probs'],
                rewards.to(device)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Accumulate metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

        # Log epoch results
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        for k, v in epoch_metrics.items():
            print(f"  {k}: {sum(v) / len(v):.4f}")
```

## Practical Guidance

- **Lightning Attention Frequency**: Use periodic exact attention every 4-8 blocks for stability
- **Mixture-of-Experts Routing**: Set to activate 45.9B/456B ≈ 10% parameters; adjust sparsity for speed
- **CISPO vs DAPO**: CISPO needs ~50% fewer steps; prefer for large-scale RL
- **Curriculum Learning**: Start verifiable (easy wins), gradually mix general tasks
- **Reward Models**: Use binary for verifiable tasks; train small reward model for general quality
- **GPU Efficiency**: Sparse activation reduces memory; monitor actual speedup
- **Evaluation**: Test on reasoning benchmarks (AIME, MATH, code), long-context tasks
- **Implementation**: Use vLLM or similar for efficient generation with large models

## Reference

Paper: arXiv:2506.13585
Key metrics: 25% FLOPs at 100K generation vs. DeepSeek R1; 1M context + 80K generation
CISPO advantage: 50% fewer training steps vs. DAPO
Related work: Mixture-of-experts, test-time scaling, efficient transformers, policy optimization
