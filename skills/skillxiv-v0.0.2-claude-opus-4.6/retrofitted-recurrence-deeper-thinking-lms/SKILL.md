---
name: retrofitted-recurrence-deeper-thinking-lms
title: "Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.07384"
keywords: [Model Architecture, Recurrent Networks, Inference Optimization, Depth Curriculum, Language Models]
description: "Convert pretrained non-recurrent language models into depth-recurrent variants through a curriculum of increasing recurrence—decoupling training compute from inference compute and improving performance at given inference budgets on reasoning tasks."
---

# Retrofit Pretrained Models with Recurrent Depth for Efficient Reasoning

Standard pretrained language models allocate the same computational budget at training and inference. Retrofitted recurrence decouples these budgets—a model can be trained efficiently while allocating more compute to inference when needed. By gradually introducing recurrence depth during training via curriculum learning, existing models learn to perform deeper internal reasoning and refine solutions iteratively.

The key insight is that pretrained models can be efficiently converted to recurrent variants without retraining from scratch. On mathematical reasoning tasks, this approach outperforms continued training of original non-recurrent models at the same inference compute budget, demonstrating that depth-recurrence provides meaningful efficiency gains.

## Core Concept

Retrofitted recurrence treats inference-time reasoning depth as a learnable skill. Rather than the fixed depth of a traditional transformer stack, recurrent models can iterate multiple times over the same (or evolving) internal representations, refining outputs with each iteration. A curriculum training strategy gradually increases recurrence depth from 1 to the target maximum, allowing the model to learn iterative refinement patterns naturally.

The approach converts the inference process from "compute once and output" to "compute, reflect, refine, iterate"—enabling expensive reasoning at test time without expensive training.

## Architecture Overview

- **Base Pretrained Model**: Non-recurrent transformer (frozen or fine-tuned)
- **Recurrence Controller**: Manages the number of reasoning iterations at inference time
- **Depth Curriculum**: Progressively increases recurrence depth during training (1 → 2 → 4 → 8 iterations)
- **Refinement Mechanism**: At each iteration, the model refines previous outputs or internal states
- **Computational Budget**: Separates training cost (linear in depth) from inference cost (configurable depth)
- **Loss Aggregation**: Supervises refinement at each intermediate depth, not just final output

## Implementation Steps

**Step 1: Define Recurrent Refinement Layer**

Create a module that applies iterative refinement on top of base model outputs.

```python
import torch
import torch.nn as nn

class RecurrentRefinementLayer(nn.Module):
    """
    Enables iterative refinement of LLM outputs through recurrence.
    """

    def __init__(self, hidden_dim, num_refinement_layers=2):
        """
        Args:
            hidden_dim: Dimension of model hidden states
            num_refinement_layers: Number of refinement stacks
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Stacks for refining outputs at each recurrence step
        self.refinement_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            for _ in range(num_refinement_layers)
        ])

    def forward(self, base_output, memory, current_depth):
        """
        Refine outputs through recurrent iterations.

        Args:
            base_output: Initial output from base model [batch, seq_len, hidden_dim]
            memory: Encoder memory from base model [batch, seq_len, hidden_dim]
            current_depth: Number of iterations to perform (curriculum-based)

        Returns:
            refined_output: Progressively refined output
            intermediate_outputs: Outputs at each refinement step
        """
        refined = base_output
        intermediate_outputs = [base_output]

        for step in range(current_depth):
            # Apply refinement layer
            block_idx = step % len(self.refinement_blocks)
            refined = self.refinement_blocks[block_idx](
                refined, memory
            )
            intermediate_outputs.append(refined)

        return refined, intermediate_outputs
```

**Step 2: Implement Depth Curriculum**

Define a curriculum that gradually increases recurrence depth during training.

```python
class DepthCurriculum:
    """
    Manages progressive increase in recurrence depth.
    """

    def __init__(self, max_depth=8, warmup_steps=5000, growth_schedule='linear'):
        """
        Args:
            max_depth: Maximum recurrence iterations (e.g., 8)
            warmup_steps: Steps over which to ramp up depth
            growth_schedule: 'linear', 'exponential', or 'step' (phase-based)
        """
        self.max_depth = max_depth
        self.warmup_steps = warmup_steps
        self.schedule = growth_schedule
        self.current_step = 0

    def get_current_depth(self):
        """
        Return recurrence depth for current training step.

        Returns:
            depth: Integer in [1, max_depth]
        """
        if self.schedule == 'linear':
            # Linear increase: 1 + (max_depth-1) * (step / warmup)
            progress = min(self.current_step / self.warmup_steps, 1.0)
            depth = int(1 + (self.max_depth - 1) * progress)
        elif self.schedule == 'exponential':
            # Exponential: depth increases slowly then rapidly
            progress = min(self.current_step / self.warmup_steps, 1.0)
            depth = int(1 + (self.max_depth - 1) * (progress ** 0.5))
        elif self.schedule == 'step':
            # Phase-based: 1 → 2 → 4 → 8
            phases = [1000, 2000, 3000, self.warmup_steps]
            depth = 1
            for phase_step in phases:
                if self.current_step >= phase_step:
                    depth = min(depth + 1, self.max_depth)
        else:
            depth = self.max_depth

        self.current_step += 1
        return depth
```

**Step 3: Supervised Loss with Intermediate Supervision**

Train on outputs at each refinement depth, encouraging progressive improvement.

```python
def compute_depth_curriculum_loss(
        intermediate_outputs, target_output,
        base_loss_fn=nn.CrossEntropyLoss(), depth_weights=None):
    """
    Compute loss supervising refinement at all depths.

    Args:
        intermediate_outputs: List of outputs at each depth
                             [(batch, seq_len, vocab_size), ...]
        target_output: Ground truth output (batch, seq_len)
        base_loss_fn: Loss function (CrossEntropyLoss, etc.)
        depth_weights: Optional weights for each depth (encourage deep outputs)

    Returns:
        total_loss: Weighted sum of losses at all intermediate depths
    """
    if depth_weights is None:
        # Default: final output weighted most
        num_depths = len(intermediate_outputs)
        depth_weights = torch.linspace(0.5, 1.0, num_depths)
        depth_weights = depth_weights / depth_weights.sum()

    total_loss = 0.0

    for depth, (output, weight) in enumerate(zip(intermediate_outputs, depth_weights)):
        # Reshape for loss computation
        batch_size, seq_len = target_output.shape
        logits = output.view(-1, output.shape[-1])
        targets = target_output.view(-1)

        # Compute loss
        depth_loss = base_loss_fn(logits, targets)
        total_loss += weight * depth_loss

    return total_loss
```

**Step 4: Integrated Training Loop**

Combine curriculum scheduling with recurrent refinement training.

```python
def train_with_depth_curriculum(
        model, refinement_layer, train_dataset,
        max_depth=8, num_epochs=3):
    """
    Training loop applying depth curriculum.

    Args:
        model: Base pretrained language model
        refinement_layer: RecurrentRefinementLayer instance
        train_dataset: Training data with inputs and targets
        max_depth: Maximum recurrence depth
        num_epochs: Number training epochs
    """
    import torch.optim as optim

    optimizer = optim.Adam(
        list(model.parameters()) + list(refinement_layer.parameters()),
        lr=1e-5
    )
    curriculum = DepthCurriculum(max_depth=max_depth, warmup_steps=5000)

    for epoch in range(num_epochs):
        for batch in train_dataset:
            input_ids = batch['input_ids']
            target_ids = batch['target_ids']

            # Forward pass through base model
            base_output = model.forward(input_ids, return_hidden_states=True)
            hidden_states = base_output.hidden_states[-1]  # Last layer

            # Get current curriculum depth
            current_depth = curriculum.get_current_depth()

            # Apply recurrent refinement
            refined_output, intermediate = refinement_layer(
                hidden_states, hidden_states, current_depth
            )

            # Convert hidden states to logits
            logits_list = [
                model.lm_head(h) for h in intermediate
            ]

            # Compute depth-curriculum loss
            loss = compute_depth_curriculum_loss(
                logits_list, target_ids,
                depth_weights=torch.linspace(0.5, 1.0, len(logits_list))
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if curriculum.current_step % 500 == 0:
                avg_loss = loss.item()
                print(f"Step {curriculum.current_step}: Loss {avg_loss:.4f}, "
                      f"Depth {current_depth}")

    return model, refinement_layer
```

**Step 5: Inference with Configurable Depth**

At test time, use desired recurrence depth independent of training.

```python
def inference_with_recurrence(model, refinement_layer, prompt, target_depth=4):
    """
    Generate text using recurrent refinement at specified depth.

    Args:
        model: Trained base model
        refinement_layer: Trained RecurrentRefinementLayer
        prompt: Input prompt text
        target_depth: Number of refinement iterations (configurable)

    Returns:
        final_output: Refined text prediction
        refinement_path: Outputs at each iteration (for analysis)
    """
    input_ids = model.tokenize(prompt)
    refinement_path = []

    with torch.no_grad():
        # Initial generation from base model
        base_output = model.forward(input_ids, return_hidden_states=True)
        hidden_states = base_output.hidden_states[-1]

        # Apply recurrent refinement at desired depth
        refined, intermediate = refinement_layer(
            hidden_states, hidden_states, target_depth
        )

        # Convert to logits and decode
        for h in intermediate:
            logits = model.lm_head(h)
            tokens = torch.argmax(logits, dim=-1)
            text = model.decode(tokens)
            refinement_path.append(text)

    return refinement_path[-1], refinement_path
```

## Practical Guidance

**When to Use Retrofitted Recurrence:**
- Mathematical and logical reasoning tasks (benefit from iterative refinement)
- Scenarios where inference compute budget is flexible (test-time scales)
- Models where existing architecture is effective (minimize retraining)

**When NOT to Use:**
- Real-time systems with strict latency requirements (recurrence adds inference time)
- Tasks with diminishing returns to depth (simple tasks don't benefit)
- Models requiring immediate single-pass inference

**Hyperparameters and Configuration:**
- Max depth: 4-8 for most tasks (balance benefit vs. latency)
- Warmup steps: 5000+ for stable curriculum (longer for larger models)
- Curriculum schedule: 'linear' for steady progression; 'exponential' for aggressive later growth
- Intermediate supervision weight: Increase toward final depth (e.g., 0.5 → 1.0)

**Pitfalls to Avoid:**
1. **Too-aggressive curriculum** - Depth increases too quickly; model can't learn refinement patterns
2. **Ignoring latency** - Each refinement iteration costs compute; set realistic depth budgets for inference
3. **Loss imbalance** - Heavy weight on intermediate outputs can prevent deep refinement learning
4. **Cold start** - Start with depth=1 before increasing; jumping to depth=4 causes training instability

---

Reference: https://arxiv.org/abs/2511.07384
