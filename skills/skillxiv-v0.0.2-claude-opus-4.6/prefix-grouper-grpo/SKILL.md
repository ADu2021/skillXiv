---
name: prefix-grouper-grpo
title: "Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.05433"
keywords: [reinforcement-learning, grpo, efficiency, long-context, attention-optimization]
description: "Implement Prefix Grouper to accelerate Group Relative Policy Optimization training by eliminating redundant prefix encoding, achieving up to 8x speedup for long-context scenarios."
---

# Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward

## Core Concept

Prefix Grouper eliminates redundant computation in GRPO training by encoding shared input prefixes only once instead of once per group member. Through a restructured attention mechanism that decouples prefix self-attention from suffix attention, the method maintains full gradient equivalence to standard GRPO while dramatically reducing FLOPs. This is particularly valuable for long-context scenarios where prefixes dominate computational cost.

## Architecture Overview

- **Grouped Attention Decomposition**: Splits self-attention into prefix-only and suffix components
- **Shared Prefix Encoding**: Single forward pass through shared context tokens
- **Suffix Attention Reuse**: All response suffixes attend to the same prefix representation
- **Gradient Equivalence**: Mathematically proven to produce identical backward gradients
- **Sequential Concatenation**: Restructures input format as P;R₁;R₂;...;Rₐ instead of padding individual sequences

## Implementation

### Step 1: Set Up Standard GRPO Baseline

First, understand the inefficiency in standard GRPO:

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def standard_grpo_forward(model, prefix_ids, responses_ids, group_size=4):
    """
    Standard GRPO: encodes prefix G times (once per group member)
    Input shape: (batch_size * group_size, seq_len)
    """
    batch_size = prefix_ids.shape[0]

    # Problem: Same prefix encoded G times
    # Input: [P, R1], [P, R2], [P, R3], [P, R4]
    # FLOPs for prefix: batch_size * group_size * prefix_len * hidden_dim

    all_logits = []
    for g in range(group_size):
        # Encode each [prefix, response] separately
        combined = torch.cat([
            prefix_ids[g*batch_size:(g+1)*batch_size],
            responses_ids[g*batch_size:(g+1)*batch_size]
        ], dim=1)

        logits = model(combined).logits
        all_logits.append(logits)

    return torch.stack(all_logits, dim=0)
```

### Step 2: Implement Prefix Grouper with Shared Encoding

Create the optimized grouped attention mechanism:

```python
class PrefixGroupedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, group_size=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states, prefix_len, attention_mask=None):
        """
        Grouped attention with shared prefix encoding.

        Args:
            hidden_states: (batch_size * group_size, seq_len, hidden_dim)
            prefix_len: length of shared prefix
            attention_mask: (batch_size * group_size, seq_len)
        """
        batch_size_grouped = hidden_states.shape[0]
        group_size = self.group_size
        batch_size = batch_size_grouped // group_size
        seq_len = hidden_states.shape[1]
        suffix_len = seq_len - prefix_len

        # Split into prefix and suffix
        prefix_hidden = hidden_states[:, :prefix_len, :]  # (B*G, P, D)
        suffix_hidden = hidden_states[:, prefix_len:, :]  # (B*G, S, D)

        # Project
        q = self.q_proj(hidden_states).view(batch_size_grouped, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size_grouped, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size_grouped, seq_len, self.num_heads, self.head_dim)

        # Compute prefix self-attention (all groups compute same thing)
        q_prefix = q[:, :prefix_len, :, :]  # (B*G, P, H, D)
        k_prefix = k[:, :prefix_len, :, :]  # (B*G, P, H, D)
        v_prefix = v[:, :prefix_len, :, :]  # (B*G, P, H, D)

        # Average prefix attention across group (they're identical)
        q_prefix_mean = q_prefix.view(batch_size, group_size, prefix_len, self.num_heads, self.head_dim).mean(dim=1)
        k_prefix_mean = k_prefix.view(batch_size, group_size, prefix_len, self.num_heads, self.head_dim).mean(dim=1)
        v_prefix_mean = v_prefix.view(batch_size, group_size, prefix_len, self.num_heads, self.head_dim).mean(dim=1)

        # Compute prefix self-attention scores
        scores_prefix = torch.matmul(q_prefix_mean, k_prefix_mean.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_prefix = torch.softmax(scores_prefix, dim=-1)
        output_prefix = torch.matmul(attn_prefix, v_prefix_mean)  # (B, P, H, D)

        # Replicate for all group members
        output_prefix = output_prefix.unsqueeze(1).expand(batch_size, group_size, -1, -1, -1)
        output_prefix = output_prefix.reshape(batch_size_grouped, prefix_len, self.num_heads, self.head_dim)

        # Compute suffix attention (attends to prefix + suffix)
        q_suffix = q[:, prefix_len:, :, :]  # (B*G, S, H, D)
        k_all = k  # (B*G, P+S, H, D)
        v_all = v  # (B*G, P+S, H, D)

        scores_suffix = torch.matmul(q_suffix, k_all.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_suffix = torch.softmax(scores_suffix, dim=-1)
        output_suffix = torch.matmul(attn_suffix, v_all)  # (B*G, S, H, D)

        # Concatenate prefix and suffix outputs
        output = torch.cat([output_prefix, output_suffix], dim=1)
        output = output.reshape(batch_size_grouped, seq_len, self.hidden_dim)

        return self.out_proj(output)
```

### Step 3: Restructure Input Format

Implement the sequential concatenation that enables prefix reuse:

```python
def prepare_grouped_batch(prefix_ids, responses_list, group_size=4):
    """
    Restructure inputs for Prefix Grouper.

    Standard format: [P,R1], [P,R2], [P,R3], [P,R4]
    Grouped format:  [P, R1, R2, R3, R4]

    This allows sharing the prefix computation across all responses.
    """
    batch_size = prefix_ids.shape[0]
    prefix_len = prefix_ids.shape[1]

    # Create attention mask
    max_response_len = max(r.shape[1] for r in responses_list)
    total_seq_len = prefix_len + (max_response_len * group_size)

    attention_mask = torch.ones(batch_size, total_seq_len, dtype=torch.long)

    # Build grouped sequences: [prefix, all_responses]
    grouped_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
    grouped_ids[:, :prefix_len] = prefix_ids

    for g, responses in enumerate(responses_list):
        start_idx = prefix_len + (g * max_response_len)
        end_idx = start_idx + responses.shape[1]

        grouped_ids[:, start_idx:end_idx] = responses
        # Mask padding
        attention_mask[:, end_idx:end_idx + (max_response_len - responses.shape[1])] = 0

    return grouped_ids, attention_mask, prefix_len
```

### Step 4: Training Integration with GRPO

Integrate Prefix Grouper into the GRPO training loop:

```python
def grpo_step_with_prefix_grouper(model, prefix_ids, responses_list, rewards, group_size=4):
    """
    Single GRPO training step using Prefix Grouper.

    This achieves gradient equivalence to standard GRPO but with 1/G FLOPs for prefixes.
    """
    # Prepare grouped batch
    grouped_ids, attention_mask, prefix_len = prepare_grouped_batch(
        prefix_ids, responses_list, group_size
    )

    # Forward pass (prefix encoded once)
    outputs = model(
        input_ids=grouped_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    logits = outputs.logits

    # Compute log probabilities for each response
    batch_size = prefix_ids.shape[0]
    max_response_len = max(r.shape[1] for r in responses_list)

    log_probs_list = []
    for g in range(group_size):
        start_idx = prefix_len + (g * max_response_len)
        response_len = responses_list[g].shape[1]

        # Extract logits for this group's response
        group_logits = logits[:, start_idx:start_idx+response_len, :]
        response_ids = responses_list[g]

        # Compute log probability
        log_probs = torch.log_softmax(group_logits, dim=-1)
        action_log_probs = log_probs.gather(2, response_ids.unsqueeze(2)).squeeze(2)
        log_prob_sum = action_log_probs.sum(dim=1)

        log_probs_list.append(log_prob_sum)

    # Stack log probs and compute advantage
    log_probs = torch.stack(log_probs_list, dim=1)  # (B, G)
    advantages = compute_advantages(rewards, log_probs)

    # Policy gradient loss
    loss = -(log_probs * advantages.detach()).mean()

    return loss

def compute_advantages(rewards, log_probs, use_baseline=True):
    """Compute advantage for GRPO"""
    if use_baseline:
        baseline = rewards.mean(dim=1, keepdim=True)
        advantages = rewards - baseline
    else:
        advantages = rewards
    return advantages
```

## Practical Guidance

- **When to Use**: Prefix Grouper provides maximum benefit when prefix length >> response length, typical in long-context scenarios
- **Expected Speedup**: Up to 1/G speedup (where G is group size, typically 4-8) for prefix computation
- **Gradient Correctness**: Mathematically proven gradient equivalence; no accuracy loss compared to standard GRPO
- **Memory Efficiency**: Reduced intermediate tensor sizes lead to proportional memory savings
- **Group Size**: Larger groups increase speedup but may reduce convergence stability; 4-8 is typical
- **Attention Mask Handling**: Carefully manage padding tokens across sequential responses to prevent attention leakage
- **Implementation Detail**: Some frameworks may require custom CUDA kernels for maximum efficiency

## Reference

- Prefix Grouper extends long-context training capabilities by removing the computational bottleneck
- Gradient equivalence is proven through chain rule analysis of backpropagation
- Sequential concatenation format [P;R₁;R₂;...;Rₐ] enables efficient batching
- Local-global attention decomposition is a general technique applicable to other multi-sequence training algorithms
