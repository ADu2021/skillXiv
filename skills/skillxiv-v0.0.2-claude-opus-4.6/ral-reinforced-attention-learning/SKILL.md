---
name: ral-reinforced-attention-learning
title: "Reinforced Attention Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.04884"
keywords: [Multimodal Learning, Attention Optimization, Policy Gradient, Perception, Vision-Language Models]
description: "Optimize where multimodal models attend by treating attention weights as a learnable policy, using policy gradients with advantage weighting to improve visual grounding and perception without changing model architecture."
---

# Reinforced Attention Learning

## Problem Context

Standard reinforcement learning for multimodal LLMs focuses on optimizing token-level outputs, which proves ineffective for perception-heavy tasks. Models need better mechanisms for allocating computational focus across visual and textual inputs. Current approaches don't improve visual grounding because text-based reward signals don't directly guide where attention should focus. The model may generate correct answers despite poor visual attention.

## Core Concept

RAL reformulates post-training to optimize [internal attention distributions, advantage weighting, policy gradient] as the primary objective. Rather than only improving what the model generates, RAL improves where the model attends. Attention weights are treated as a policy that governs information selection, optimized via policy gradients weighted by task advantages.

## Architecture Overview

- **Attention as policy**: Extract attention weights from transformer layers; treat as learnable policy
- **Advantage-weighted loss**: Use Jensen-Shannon Divergence between current and reference attention, weighted by advantage signals
- **Dual optimization**: Combine standard token-level gradients with attention-level supervision
- **On-policy distillation**: Transfer attention patterns from teacher to student models
- **Layer targeting**: Apply selectively to layers most relevant for perception (middle-to-later layers)

## Implementation

### Step 1: Extract and profile attention patterns

Extract attention weights from model during forward pass. Profile which layers are most relevant for perception.

```python
# Extract attention patterns
class AttentionExtractor:
    def __init__(self, model, layer_indices=None):
        self.model = model
        if layer_indices is None:
            # Default: middle to later layers for perception
            num_layers = model.config.num_hidden_layers
            self.layer_indices = list(
                range(num_layers // 2, num_layers)
            )
        else:
            self.layer_indices = layer_indices

    def extract_attention(self, input_ids, pixel_values):
        """
        Forward through model; extract attention weights.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True
            )

        attention_weights = {}
        for layer_idx in self.layer_indices:
            # Shape: (batch_size, num_heads, seq_len, seq_len)
            attn = outputs.attentions[layer_idx]
            # Average over heads and batch
            attn_mean = attn.mean(dim=(0, 1))  # (seq_len, seq_len)
            attention_weights[layer_idx] = attn_mean

        return attention_weights

    def compute_visual_attention_focus(self, attention_weights, num_visual_tokens):
        """
        Compute how much attention goes to visual vs. text tokens.
        Visual tokens typically appear first in sequence.
        """
        focus_scores = {}

        for layer_idx, attn_matrix in attention_weights.items():
            # Average attention from text tokens to visual tokens
            text_start = num_visual_tokens
            visual_to_text_attn = attn_matrix[text_start:, :num_visual_tokens].mean()
            focus_scores[layer_idx] = visual_to_text_attn.item()

        return focus_scores
```

### Step 2: Design advantage-weighted attention objective

Formulate the attention loss using Jensen-Shannon divergence, weighted by task-level advantages.

```python
# Attention loss with advantage weighting
def compute_attention_loss(
    current_attention, reference_attention, advantages,
    layer_indices=None, divergence_fn='jensen_shannon'
):
    """
    Compute advantage-weighted attention loss.

    Args:
        current_attention: dict of tensors, shape (seq_len, seq_len)
        reference_attention: dict of tensors (frozen reference model)
        advantages: tensor of advantage estimates, shape (batch_size,)
        divergence_fn: 'jensen_shannon' or 'kl'
    """
    total_loss = 0.0
    num_layers = 0

    if layer_indices is None:
        layer_indices = current_attention.keys()

    for layer_idx in layer_indices:
        if layer_idx not in current_attention:
            continue

        curr_attn = current_attention[layer_idx]
        ref_attn = reference_attention[layer_idx].detach()

        # Normalize to probability distributions
        curr_prob = F.softmax(curr_attn.flatten(), dim=-1)
        ref_prob = F.softmax(ref_attn.flatten(), dim=-1)

        # Compute divergence
        if divergence_fn == 'jensen_shannon':
            # JS divergence: symmetric, bounded
            m_prob = 0.5 * (curr_prob + ref_prob)
            loss_layer = 0.5 * F.kl_div(
                torch.log(curr_prob), m_prob, reduction='none'
            ).sum()
            loss_layer += 0.5 * F.kl_div(
                torch.log(ref_prob), m_prob, reduction='none'
            ).sum()
        else:  # 'kl'
            loss_layer = F.kl_div(
                torch.log(curr_prob), ref_prob, reduction='sum'
            )

        # Weight by advantage (batch-level)
        # For simplicity: use mean advantage across batch
        weighted_loss = loss_layer * advantages.mean().detach()

        total_loss += weighted_loss
        num_layers += 1

    return total_loss / max(num_layers, 1)
```

### Step 3: Implement on-policy attention distillation

Extend to transfer attention patterns from teacher to student models.

```python
# On-policy attention distillation
def on_policy_attention_distillation(
    teacher_model, student_model, batch,
    temperature=1.0, distill_weight=0.5
):
    """
    Distill attention patterns from teacher to student on-policy.
    """
    input_ids = batch['input_ids']
    pixel_values = batch['pixel_values']

    # Teacher attention (frozen)
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            output_attentions=True
        )
        teacher_attention = {
            i: attn.mean(dim=(0, 1))
            for i, attn in enumerate(teacher_outputs.attentions)
        }

    # Student forward
    student_outputs = student_model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        output_attentions=True
    )
    student_attention = {
        i: attn.mean(dim=(0, 1))
        for i, attn in enumerate(student_outputs.attentions)
    }

    # Compute advantages (use task rewards)
    task_rewards = batch.get('rewards', None)
    if task_rewards is not None:
        advantages = compute_advantages(task_rewards)
    else:
        advantages = torch.ones(input_ids.shape[0])

    # Attention distillation loss
    attn_loss = compute_attention_loss(
        student_attention, teacher_attention, advantages
    )

    # Token-level loss (standard training)
    logits = student_outputs.logits
    labels = batch['labels']
    token_loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1)
    )

    # Combined loss
    total_loss = (1 - distill_weight) * token_loss + distill_weight * attn_loss

    return total_loss, {'token_loss': token_loss, 'attn_loss': attn_loss}
```

### Step 4: Apply policy gradient optimization to attention

Use policy gradient updates specifically targeting attention weights.

```python
# Policy gradient for attention
def policy_gradient_attention_step(
    model, batch, optimizer, clip_ratio=0.2
):
    """
    Apply policy gradient to attention distributions.
    """
    input_ids = batch['input_ids']
    pixel_values = batch['pixel_values']

    # Forward pass: get attention and logits
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        output_attentions=True
    )

    attention_weights = {
        i: attn.mean(dim=(0, 1))
        for i, attn in enumerate(outputs.attentions)
    }

    # Compute task rewards (e.g., from verifier)
    rewards = batch.get('rewards', None)
    if rewards is None:
        # Fallback: use model confidence
        logits = outputs.logits
        rewards = F.softmax(logits, dim=-1).max(dim=-1)[0]

    # Normalize rewards to get advantages
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # For each layer, compute attention policy gradient
    total_loss = 0.0

    for layer_idx, attn_weights in attention_weights.items():
        # Treat attention as log-probabilities
        log_attn = F.log_softmax(attn_weights, dim=-1)

        # Policy gradient: log_prob * advantage
        # (scaled by negative for gradient descent)
        policy_loss = -(log_attn * advantages.mean()).mean()

        total_loss += policy_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return total_loss.item()
```

### Step 5: Integrate into multimodal training loop

Combine attention optimization with standard token-level training.

```python
# Training with reinforced attention
def train_multimodal_with_attention_rl(
    model, train_loader, verifier, optimizer,
    num_epochs=3, attention_weight=0.3, device='cuda'
):
    """
    Training loop combining token-level and attention-level optimization.
    """
    attention_extractor = AttentionExtractor(model)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                output_attentions=True
            )

            # Token-level loss
            logits = outputs.logits
            labels = batch.get('labels')
            if labels is not None:
                token_loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1)
                )
            else:
                token_loss = 0.0

            # Compute rewards (from verifier or auxiliary signal)
            with torch.no_grad():
                if 'rewards' not in batch:
                    # Compute approximate rewards
                    batch['rewards'] = compute_rewards(
                        batch, logits, verifier
                    )

            # Extract attention patterns
            attention_weights = {
                i: attn.mean(dim=(0, 1))
                for i, attn in enumerate(outputs.attentions)
            }

            # Attention-level loss
            advantages = compute_advantages(batch['rewards'])
            attn_loss = compute_attention_loss(
                attention_weights,
                attention_weights,  # Could use reference model
                advantages
            )

            # Combined loss
            loss = token_loss + attention_weight * attn_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}")

        print(f"Epoch {epoch + 1}: Avg Loss={total_loss / num_batches:.4f}")
```

## Practical Guidance

**When to use**: Multimodal perception tasks (visual QA, image understanding, document understanding) where visual grounding matters. Less beneficial for text-only reasoning.

**Hyperparameters**:
- **Attention weight**: 0.2-0.5 in combined loss (start conservative)
- **Layer indices**: Target middle-to-later layers (48-64 out of 80 for large models)
- **Divergence function**: Jensen-Shannon (symmetric) preferred over KL
- **Temperature**: 1.0 (no scaling); increase to 1.5-2.0 if attention becomes too sharp

**Key findings**:
- Image-heavy benchmarks (VQA, V-Star) show consistent improvements
- Text-only benchmarks unaffected by attention optimization
- Works without explicit reasoning chains (unlike CoT)
- On-policy distillation transfers attention patterns efficiently

**Common pitfalls**:
- Over-weighting attention loss → tokens get worse; start with 0.2
- Using all layers → noise from unrelated layers; target middle-to-later only
- Not normalizing attention before computing divergence → numerical instability
- Forgetting to detach reference attention → computational waste and gradients through reference

**Scaling**: Minimal overhead (one additional loss term, same number of parameters). Scales linearly with model size and batch size.

## Reference

Paper: https://arxiv.org/abs/2602.04884
Code: Available at author's repository
Related work: Attention analysis, multimodal RL, vision-language models
Benchmarks: Visual QA, V-Star, image understanding tasks
