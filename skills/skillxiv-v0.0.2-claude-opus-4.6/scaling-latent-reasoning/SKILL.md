---
name: scaling-latent-reasoning
title: "Scaling Latent Reasoning via Looped Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.25741"
keywords: [Looped Models, Latent Reasoning, Test-time Scaling, Knowledge Manipulation]
description: "Scales reasoning depth through internal iteration rather than explicit generation. Ouro models perform repeated computation in latent space with entropy-regularized objectives enabling learned depth allocation. Smaller 1.4B model matches 12B standard models through improved knowledge manipulation."
---

# Looped Language Models: Latent Reasoning Depth

Scaling language models conventionally increases size, consuming memory and compute. Looped Language Models (LoopLM) scale reasoning depth through internal iteration in latent space, enabling smaller models to perform like much larger ones.

The approach embeds reasoning into pre-training, eliminating dependence on test-time chain-of-thought.

## Core Concept

Key innovation: **models perform iterative computation in hidden representations** rather than generating explicit reasoning:
- Latent space iteration: reasoning happens internally
- Learned depth allocation: models decide iteration count per problem
- Pre-training integration: no post-hoc prompting needed
- Efficiency: internal reasoning is less token-expensive than explicit CoT

## Architecture Overview

- Standard transformer backbone with looping mechanism
- Recurrent computation block for latent iterations
- Entropy regularization for depth control
- Output head for final answer generation

## Implementation Steps

Implement looped computation block that performs K iterations in latent space:

```python
class LoopedComputationBlock(nn.Module):
    def __init__(self, hidden_dim=768, max_loops=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_loops = max_loops

        # Iteration processing (same weights reused)
        self.loop_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Depth prediction: how many iterations needed?
        self.depth_predictor = nn.Linear(hidden_dim, 1)

        # Entropy regularization for learning allocation
        self.entropy_reg = 0.1

    def forward(self, hidden_state, return_depth=False):
        """Perform looped computation with learned depth."""
        current = hidden_state
        iteration_outputs = [current]

        # Predict required depth (0-max_loops)
        depth_logit = self.depth_predictor(hidden_state)
        depth_prob = torch.sigmoid(depth_logit).squeeze(-1)

        # Stochastic depth: during training, sample iterations
        if self.training:
            depth = torch.bernoulli(depth_prob * self.max_loops).int()
        else:
            depth = (depth_prob * self.max_loops).round().int()

        depth = torch.clamp(depth, 1, self.max_loops)

        # Perform depth iterations
        for i in range(self.max_loops):
            if i < depth.max().item():
                # Process hidden state
                current = self.loop_processor(current)
                iteration_outputs.append(current)

                # Entropy regularization: encourage sparse iteration
                entropy = -torch.mean(
                    depth_prob * torch.log(depth_prob + 1e-8) +
                    (1 - depth_prob) * torch.log(1 - depth_prob + 1e-8)
                )
                self.entropy_loss = -self.entropy_reg * entropy

        # Use final iteration output
        output = iteration_outputs[-1]

        if return_depth:
            return output, depth
        return output
```

Train models with looped computation enabled. Use standard language modeling loss plus entropy regularization:

```python
def train_looped_lm(model, data_loader, num_epochs=10):
    """Train looped language model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in data_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']

            # Forward with looped computation
            logits = model(input_ids)  # LoopedComputationBlock integrated

            # Language modeling loss
            lm_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

            # Entropy regularization loss from loop block
            loop_blocks = [m for m in model.modules()
                         if isinstance(m, LoopedComputationBlock)]
            entropy_loss = sum(b.entropy_loss for b in loop_blocks) / len(loop_blocks)

            # Combined loss
            total_loss_val = lm_loss + entropy_loss

            optimizer.zero_grad()
            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += total_loss_val.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
```

Analyze depth allocation to understand learned reasoning patterns:

```python
def analyze_depth_allocation(model, test_data):
    """Analyze how many iterations model allocates per task."""
    depth_by_task = {}

    for example in test_data:
        input_ids = example['input_ids'].unsqueeze(0)
        task_type = example['task_type']

        # Get depth allocation
        _, depth = model(input_ids, return_depth=True)
        allocated_depth = depth.item()

        if task_type not in depth_by_task:
            depth_by_task[task_type] = []

        depth_by_task[task_type].append(allocated_depth)

    # Print statistics
    for task, depths in depth_by_task.items():
        avg_depth = sum(depths) / len(depths)
        print(f"{task}: avg depth = {avg_depth:.2f} iterations")

    return depth_by_task
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Max loops | 3-5 (balance performance and training) |
| Entropy weight | 0.1-0.2 (encourage sparse iteration) |
| Model size | 1-3B (efficiency gain) |
| Pre-training scale | 7.7T tokens (standard) |

**When to use:**
- Scaling reasoning without model size increase
- Inference-time efficiency requirements
- Domains where internal reasoning helps
- Pre-trained model development

**When NOT to use:**
- Interpretability critical (latent reasoning opaque)
- Extremely large models (competes with model scaling)
- Tasks without reasoning components

**Common pitfalls:**
- Entropy weight too high (no iteration, no reasoning)
- Max loops too large (training instability)
- Not validating learned depth correlates with task difficulty
- Comparing unfairly to standard models (test-time different)

Reference: [Scaling Latent Reasoning on arXiv](https://arxiv.org/abs/2510.25741)
