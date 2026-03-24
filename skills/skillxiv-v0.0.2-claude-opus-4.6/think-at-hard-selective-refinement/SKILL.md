---
name: think-at-hard-selective-refinement
title: "Think-at-Hard: Selective Latent Iterations to Improve Reasoning LMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.08577"
keywords: [Inference Efficiency, Reasoning Models, Adaptive Compute, Token-Level Decisions, Latent Refinement]
description: "Reduce reasoning model inference cost by selectively applying latent iterations—use a lightweight neural decider to identify hard tokens and refine only those with focused LoRA modules, achieving 94% token exemption with 8-11% accuracy gain."
---

# Reduce Reasoning Inference Cost via Selective Token Refinement

Reasoning models sometimes emit incorrect tokens despite strong overall performance. Naive approaches apply additional reasoning steps uniformly—every token gets multiple iterations. Think-at-Hard observes that most tokens are easy; only a few need refinement. By identifying hard tokens early and selectively iterating only those, the model achieves significant accuracy improvements (8-11%) while reducing cost by exempting ~94% of tokens from secondary reasoning.

A lightweight neural decider identifies problematic tokens; focused LoRA refinement modules then redirect computation toward correcting those specific tokens without recomputing the entire sequence.

## Core Concept

Language models make errors at specific positions. Most errors aren't due to insufficient thinking but misalignment on particular tokens (wrong verb form, misplaced operator, etc.). Rather than applying uniform chain-of-thought or tree-search across all tokens, Think-at-Hard:

1. **Early Detection**: After initial generation, compute token-level uncertainty/correctness scores
2. **Selective Targeting**: Identify tokens likely to be wrong (<5% of total)
3. **Focused Refinement**: Apply latent iterations (multiple reasoning passes) only to hard tokens
4. **Parallel Execution**: Use duo-causal attention spanning both token sequence and iteration depth

This reduces redundant computation: 94% of tokens skip secondary iterations, while hard tokens get focused refinement.

## Architecture Overview

- **Initial Generation**: Standard forward pass generating tokens and logits
- **Neural Decider**: MLP identifying hard tokens based on initial logits, context, and layer activations
- **Token-Level Hard/Easy Classification**: Binary decision per token; easy tokens bypass latent iterations
- **LoRA Refinement Modules**: Lightweight adapters for iterations; focus objective from general next-token prediction to "correct this token"
- **Duo-Causal Attention**: Enable information flow across both sequence dimension and iteration depth while preserving parallelism

## Implementation Steps

**Step 1: Neural Decider—Identify Hard Tokens.**

```python
import torch
import torch.nn as nn

class NeuralDecider(nn.Module):
    """
    Lightweight module to identify tokens needing refinement.
    Input: initial logits, activations, context.
    Output: hard/easy classification per token.
    """
    def __init__(self, hidden_dim=768, vocab_size=32000):
        super().__init__()

        # Features for decision making
        self.logit_analyzer = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.activation_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Combine features and decide
        self.decision_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output: probability of being hard (0-1)
        )

    def forward(self, logits, hidden_states, attention_weights=None):
        """
        Identify hard tokens.
        logits: (batch, seq_len, vocab_size)
        hidden_states: (batch, seq_len, hidden_dim)
        attention_weights: optional, (batch, num_heads, seq_len, seq_len)
        """
        batch, seq_len, _ = logits.shape

        # Analyze logits: entropy, top-k probability gaps
        logit_features = self.logit_analyzer(logits)  # (batch, seq_len, 128)

        # Analyze activations: magnitude, distribution
        activation_features = self.activation_analyzer(hidden_states)  # (batch, seq_len, 128)

        # Combine
        combined = torch.cat([logit_features, activation_features], dim=-1)  # (batch, seq_len, 256)

        # Decide hard/easy
        hard_probs = self.decision_head(combined)  # (batch, seq_len, 1)

        return hard_probs.squeeze(-1)  # (batch, seq_len)

    def compute_hard_mask(self, hard_probs, hard_token_ratio=0.05):
        """
        Convert probabilities to binary hard/easy mask.
        hard_token_ratio: approximate fraction of tokens to refine (default 5%)
        """
        batch_size, seq_len = hard_probs.shape

        # Set threshold to target hard_token_ratio
        threshold = torch.quantile(hard_probs, 1 - hard_token_ratio, dim=1, keepdim=True)

        hard_mask = (hard_probs > threshold).float()  # (batch, seq_len)

        return hard_mask
```

**Step 2: LoRA Refinement Modules—Focused Correction.**

```python
class LoRARefinement(nn.Module):
    """
    Low-rank adapter for selective token refinement.
    Redirects LM objective from generic next-token prediction to focused correction.
    """
    def __init__(self, hidden_dim=768, rank=16):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.rank = rank

        # LoRA decomposition: hidden_dim → rank → hidden_dim
        self.lora_down = nn.Linear(hidden_dim, rank)
        self.lora_up = nn.Linear(rank, hidden_dim)

        # Focus selector: which aspect to refine (token identity, POS tag, etc.)
        self.focus_selector = nn.Linear(hidden_dim, 3)  # 3 refinement focuses

        # Initialize LoRA with small weights
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, hidden_states, hard_mask):
        """
        Apply refinement to hard tokens only.
        hidden_states: (batch, seq_len, hidden_dim)
        hard_mask: (batch, seq_len) binary mask
        """
        # LoRA computation
        lora_out = self.lora_down(hidden_states)  # (batch, seq_len, rank)
        lora_out = self.lora_up(lora_out)  # (batch, seq_len, hidden_dim)

        # Apply only to hard tokens
        hard_mask_expanded = hard_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        refined = hidden_states + hard_mask_expanded * lora_out

        return refined
```

**Step 3: Latent Iterations with Duo-Causal Attention.**

```python
class DuoCausalAttention(nn.Module):
    """
    Enable information flow across both sequence and iteration dimensions.
    Allows hard tokens to refine themselves by looking at:
    - Other tokens in same iteration
    - Same token across previous iterations
    """
    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states_iter, hard_mask):
        """
        hidden_states_iter: (num_iters, batch, seq_len, hidden_dim)
        hard_mask: (batch, seq_len) which tokens are hard
        """
        num_iters, batch, seq_len, hidden_dim = hidden_states_iter.shape

        # Flatten to single sequence: interleave iterations
        # New sequence: token_0_iter_0, token_0_iter_1, ..., token_0_iter_T, token_1_iter_0, ...
        flattened = hidden_states_iter.permute(1, 2, 0, 3)  # (batch, seq_len, num_iters, hidden_dim)
        flattened = flattened.reshape(batch, seq_len * num_iters, hidden_dim)

        # Apply standard attention with duo-causal mask
        # Causal in iteration dimension: iteration t can attend to iterations <= t
        # Causal in sequence dimension: token i can attend to tokens <= i
        causal_mask = self._create_duo_causal_mask(seq_len, num_iters)

        Q = self.query(flattened)
        K = self.key(flattened)
        V = self.value(flattened)

        # Standard multi-head attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, V)
        output = self.output_proj(output)

        # Reshape back
        output = output.reshape(batch, seq_len, num_iters, hidden_dim)
        output = output.permute(2, 0, 1, 3)  # (num_iters, batch, seq_len, hidden_dim)

        return output

    def _create_duo_causal_mask(self, seq_len, num_iters):
        """
        Create mask allowing:
        - Token i can attend to tokens <= i (sequence causality)
        - Iteration t can attend to iterations <= t (iteration causality)
        """
        total_len = seq_len * num_iters
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        for i in range(seq_len):
            for j in range(seq_len):
                for t_i in range(num_iters):
                    for t_j in range(num_iters):
                        # Position in flattened sequence
                        pos_i = i * num_iters + t_i
                        pos_j = j * num_iters + t_j

                        # Allow if: (i == j and t_i >= t_j) or (i > j)
                        # (same token, later iteration) or (earlier token)
                        if (i == j and t_i >= t_j) or (i > j):
                            mask[pos_i, pos_j] = True

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total_len, total_len)
```

**Step 4: Training Loop with Selective Iterations.**

```python
def train_with_selective_iterations(
    model, decider, lora_modules, dataloader, num_epochs=100,
    num_latent_iters=3, hard_token_ratio=0.05
):
    """
    Train model with selective latent iterations for hard tokens.
    """
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(decider.parameters()),
                                   lr=1e-4)

    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch['input_ids']
            target_ids = batch['target_ids']

            # Initial forward pass
            logits, hidden_states = model(input_ids, output_hidden_states=True)
            initial_pred = torch.argmax(logits, dim=-1)  # (batch, seq_len)

            # Identify hard tokens
            hard_probs = decider(logits, hidden_states)
            hard_mask = decider.compute_hard_mask(hard_probs, hard_token_ratio)

            # Latent iterations (only for hard tokens)
            hidden_current = hidden_states
            loss_total = 0

            for iter_idx in range(num_latent_iters):
                # Refine hidden states with LoRA
                hidden_current = lora_modules[iter_idx](hidden_current, hard_mask)

                # Regenerate logits
                logits_iter = model.decoder(hidden_current)

                # Loss: focus on hard tokens
                iter_loss = torch.nn.functional.cross_entropy(
                    logits_iter.reshape(-1, logits_iter.shape[-1]),
                    target_ids.reshape(-1),
                    reduction='none'
                )

                # Weight by hard mask: emphasize hard tokens
                iter_loss = iter_loss.reshape(input_ids.shape)
                iter_loss = (iter_loss * hard_mask).mean()

                loss_total += iter_loss

            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss_total.item():.4f}, "
                      f"hard_tokens={hard_mask.float().mean():.2%}")
```

## Practical Guidance

**When to Use:** Reasoning models (math, coding, multi-step QA) where token-level accuracy matters and inference cost is a concern. Use when you can tolerate selective refinement overhead (<5% runtime for 8-11% accuracy gain).

**Hyperparameters:**
- Hard token ratio: 5% works well; adjust to 3-7% based on model/dataset
- Number of latent iterations: 2-4; diminishing returns after 4
- LoRA rank: 8-16 for 768-dim; scale with hidden dimension
- Decision threshold: learned via decider; can be calibrated on validation set

**Pitfalls:**
- **Threshold sensitivity**: Too-high threshold means no refinement; too-low means all tokens refined (no savings)
- **Hard token bias**: Decider may learn spurious correlations; regularize with entropy loss
- **Iteration divergence**: LoRA modules can cause output collapse; use dropout and weight decay
- **Generalization**: Hard/easy distribution may differ across domains; retrain decider for new domains

**When NOT to Use:** Tasks where uniform application of extra compute is simpler (small models, latency-unconstrained inference); single-pass models without refinement capability.

**Integration**: Compatible with any transformer LLM; requires minimal architecture changes (decider + LoRA modules are addons).

---
Reference: https://arxiv.org/abs/2511.08577
