---
name: let-late-early-training
title: "Late-to-Early Training: LET LLMs Learn Earlier"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.05393"
keywords: [LLM Training, Knowledge Distillation, Accelerated Training, Pretraining, Layer Alignment]
description: "Accelerate LLM pretraining by leveraging small pretrained models as teachers, enabling large models to explicitly learn late-layer knowledge in early layers via alignment loss, achieving 1.6x convergence speedup and 5% downstream improvement even with 10x smaller teachers."
---

# LET: Late-to-Early Training for Faster LLM Pretraining

Standard LLM training develops knowledge progressively from early to late layers. LET inverts this by leveraging small pretrained models to teach large models to learn late-layer representations in early layers, dramatically accelerating convergence. The approach uses late-to-early layer alignment loss combined with late-to-early step learning (phasing out teacher influence), enabling models to skip intermediate development stages.

## Core Concept

The key insight is that small pretrained models contain useful knowledge about which representations are important. By explicitly teaching large models to develop these representations early, we can bypass inefficient exploration phases. This works even with teachers 10x smaller than the target model, suggesting fundamental acceleration in learning dynamics rather than just parameter transfer.

## Architecture Overview

- **Late-to-Early Layer Learning**: Align early layers of target with late layers of small teacher via cosine similarity loss
- **Late-to-Early Step Learning**: Teacher influence decays during training (λ linearly annealed from λ₀ to 0)
- **Combined Alignment Loss**: Standard NLL loss + alignment loss during pretraining
- **Architecture-Agnostic**: Works across different model sizes and architectures
- **Composable**: Combines with standard training optimizations

## Implementation

### Step 1: Prepare Teacher Model

Initialize and freeze small pretrained teacher model.

```python
import torch
import torch.nn.functional as F

def prepare_teacher_model(teacher_config, teacher_checkpoint=None):
    """
    Load or initialize small pretrained teacher model.
    Teacher is frozen during training; only used for guidance.
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    
    if teacher_checkpoint:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_checkpoint)
    else:
        config = AutoConfig.from_pretrained(teacher_config)
        teacher = AutoModelForCausalLM.from_config(config)
    
    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False
    
    teacher.eval()
    return teacher
```

### Step 2: Define Layer Alignment Loss

Create loss function for late-to-early layer alignment.

```python
def compute_layer_alignment_loss(target_hidden, teacher_hidden, target_layer_idx, teacher_layer_idx):
    """
    Align target early layer with teacher late layer via cosine similarity loss.
    Minimizes the angle between representations while allowing different scales.
    """
    # Normalize hidden states to unit norm
    target_norm = F.normalize(target_hidden, p=2, dim=-1)
    teacher_norm = F.normalize(teacher_hidden, p=2, dim=-1)
    
    # Cosine similarity loss (negative cosine similarity)
    cosine_sim = (target_norm * teacher_norm).sum(dim=-1)
    alignment_loss = 1.0 - cosine_sim.mean()
    
    return alignment_loss

def compute_layer_alignment_loss_batch(target_layers, teacher_layers, alignment_pairs):
    """
    Compute alignment loss for multiple layer pairs.
    alignment_pairs: list of (target_layer_idx, teacher_layer_idx) tuples
    """
    total_alignment_loss = 0.0
    
    for target_idx, teacher_idx in alignment_pairs:
        loss = compute_layer_alignment_loss(
            target_layers[target_idx],
            teacher_layers[teacher_idx],
            target_idx,
            teacher_idx
        )
        total_alignment_loss += loss
    
    return total_alignment_loss / len(alignment_pairs)
```

### Step 3: Define Layer Pair Alignment Strategy

Determine which target and teacher layers to align.

```python
def compute_alignment_pairs(target_model, teacher_model, strategy='uniform'):
    """
    Define which layers to align between target and teacher.
    
    strategy:
    - 'uniform': spread alignment across all target layers
    - 'early': align early target layers with late teacher layers
    - 'skip': skip layers based on depth ratio
    """
    target_num_layers = len(target_model.transformer.h)
    teacher_num_layers = len(teacher_model.transformer.h)
    
    if strategy == 'uniform':
        # Pair early target with late teacher uniformly
        alignment_pairs = []
        for t_idx in range(target_num_layers):
            # Map target layer to teacher layer
            teacher_idx = int((t_idx / target_num_layers) * teacher_num_layers)
            teacher_idx = min(teacher_idx + (teacher_num_layers // 3), teacher_num_layers - 1)
            alignment_pairs.append((t_idx, teacher_idx))
    
    elif strategy == 'early':
        # Focus on early target layers only
        alignment_pairs = []
        early_fraction = 0.3
        for t_idx in range(int(target_num_layers * early_fraction)):
            # Align to late teacher layers
            teacher_idx = teacher_num_layers - (teacher_num_layers // 4) - (t_idx % 3)
            teacher_idx = max(0, min(teacher_idx, teacher_num_layers - 1))
            alignment_pairs.append((t_idx, teacher_idx))
    
    else:  # skip strategy
        alignment_pairs = []
        skip_rate = int(target_num_layers / teacher_num_layers)
        for t_idx in range(0, target_num_layers, skip_rate):
            teacher_idx = int((t_idx / target_num_layers) * teacher_num_layers)
            alignment_pairs.append((t_idx, teacher_idx))
    
    return alignment_pairs
```

### Step 4: Implement LET Training Loop

Combine NLL loss with decaying alignment loss.

```python
def let_training_step(target_model, teacher_model, batch, optimizer, 
                      lambda_0=0.1, current_step=0, total_steps=10000,
                      alignment_pairs=None):
    """
    Single training step with late-to-early alignment.
    
    Total loss = NLL loss + λ(t) * alignment loss
    where λ(t) decays linearly from λ_0 to 0
    """
    input_ids = batch['input_ids']
    attention_mask = batch.get('attention_mask', None)
    
    # Forward through target model
    target_output = target_model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    target_logits = target_output.logits
    target_hidden_states = target_output.hidden_states
    
    # Compute NLL loss (standard language modeling loss)
    shift_logits = target_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    nll_loss = F.cross_entropy(
        shift_logits.view(-1, target_model.config.vocab_size),
        shift_labels.view(-1)
    )
    
    # Compute alignment loss with teacher
    with torch.no_grad():
        teacher_output = teacher_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        teacher_hidden_states = teacher_output.hidden_states
    
    alignment_loss = compute_layer_alignment_loss_batch(
        target_hidden_states, teacher_hidden_states, alignment_pairs
    )
    
    # Decay schedule for alignment weight
    lambda_t = lambda_0 * (1.0 - current_step / total_steps)
    
    # Combined loss
    total_loss = nll_loss + lambda_t * alignment_loss
    
    # Backward pass
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    return {
        'total_loss': total_loss.item(),
        'nll_loss': nll_loss.item(),
        'alignment_loss': alignment_loss.item(),
        'lambda_t': lambda_t
    }
```

### Step 5: Full Training Loop with LET

Integrate LET into standard pretraining loop.

```python
def train_with_let(target_model, teacher_model, train_dataloader, 
                   num_epochs=5, lambda_0=0.1):
    """
    Full LET training loop with alignment-augmented pretraining.
    """
    optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Compute alignment pairs
    alignment_pairs = compute_alignment_pairs(target_model, teacher_model, strategy='early')
    
    total_steps = num_epochs * len(train_dataloader)
    current_step = 0
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # LET training step
            metrics = let_training_step(
                target_model, teacher_model, batch, optimizer,
                lambda_0=lambda_0,
                current_step=current_step,
                total_steps=total_steps,
                alignment_pairs=alignment_pairs
            )
            
            current_step += 1
            
            # Logging
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {batch_idx}: "
                      f"total_loss={metrics['total_loss']:.4f}, "
                      f"nll={metrics['nll_loss']:.4f}, "
                      f"alignment={metrics['alignment_loss']:.4f}, "
                      f"λ={metrics['lambda_t']:.4f}")
        
        scheduler.step()
    
    return target_model
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|----------------|-------|
| Teacher Model Size | 10-20% of target size | Smaller teachers still effective; compute trade-off |
| Lambda 0 | 0.05-0.2 | Higher = stronger alignment; typically 0.1 works well |
| Alignment Strategy | 'early' for best speedup | Focus on first 1/3 of layers |
| Stop Step S_stop | 0.5-0.8 of total steps | Gradually phase out teacher influence |
| Teacher Architecture | Same as target, just smaller | Different architectures possible but require more tuning |

**When to Use:**
- Large-scale LLM pretraining where speedup matters
- Scenarios with small pretrained models available
- Training budgets where 1.6x speedup provides significant value

**When Not To Use:**
- Fine-tuning (LET helps most during pretraining)
- Novel architectures without pretrained teachers
- Cases requiring maximum final model capacity (may trade off slightly)

## Reference

Achieves 1.6x convergence speedup with small pretrained teachers, 5% improvement on downstream tasks, and compelling results even with 10x smaller teachers, demonstrating fundamental acceleration in learning dynamics through late-to-early knowledge transfer.
