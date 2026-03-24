---
name: model-merging-dual-anchors
title: "Model Merging with Functional Dual Anchors"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.21223"
keywords: [Model Merging, Multi-task Learning, Task Vectors, Parameter Space]
description: "Merges multiple fine-tuned models by operating in input-representation space rather than parameter space. Creates synthetic inputs whose gradients align with task vectors, bridging joint training and post-hoc merging for robust multi-task model combination."
---

# Model Merging via Functional Dual Anchors: Beyond Parameter Space

Parameter-space merging methods suffer from conflicts when task vectors diverge. Functional Dual Anchors (FDAs) capture task shifts in representation space through synthetic inputs, enabling more robust model combination.

FDAs bridge multi-task training and post-hoc merging, offering flexibility to merge models trained independently while maintaining task-specific capabilities.

## Core Concept

Rather than averaging or interpolating model weights directly, FDAs operate by:
- Creating synthetic inputs in input-representation space
- Designing these inputs so their gradients match task-specific vectors
- Using gradient alignment to represent how models diverge from base
- Merging in this aligned gradient space rather than parameter space

This approach mitigates parameter inconsistencies that plague traditional merging.

## Architecture Overview

- Synthetic input generation that produces task-aligned gradients
- Principled initialization scheme for anchor points
- Gradient-based representation of task shifts
- Complementary to parameter-space merging methods

## Implementation Steps

Define synthetic inputs as learnable parameters whose gradients capture task shifts. Rather than working with weight parameters directly, optimize inputs to produce task-relevant gradients:

```python
class FunctionalDualAnchors:
    def __init__(self, base_model, num_tasks, input_dim=768):
        self.base_model = base_model
        self.num_tasks = num_tasks
        # Create synthetic inputs per task
        self.synthetic_inputs = nn.ParameterList([
            nn.Parameter(torch.randn(1, input_dim) * 0.01)
            for _ in range(num_tasks)
        ])

    def compute_task_gradient(self, task_id, target_vector):
        """Compute gradient of synthetic input to match task vector."""
        synthetic = self.synthetic_inputs[task_id]
        synthetic.requires_grad_(True)

        # Forward through model
        output = self.base_model(synthetic)

        # Loss: align output gradients with task vector
        loss = -torch.dot(output.squeeze(), target_vector)
        loss.backward()

        return synthetic.grad
```

Compute task vectors from fine-tuned models as the difference from base model. This captures what each fine-tuning task adds:

```python
def compute_task_vector(fine_tuned_model, base_model):
    """Extract task vector as parameter difference."""
    task_vector = {}
    for (name_ft, param_ft), (name_base, param_base) in zip(
        fine_tuned_model.named_parameters(),
        base_model.named_parameters()
    ):
        task_vector[name_ft] = param_ft - param_base

    return task_vector
```

Merge models by aligning synthetic inputs with computed task vectors. The FDA bridges training and inference by providing a unified representation:

```python
def merge_with_functional_anchors(base_model, fine_tuned_models, task_vectors):
    """Merge multiple models using functional dual anchors."""
    anchors = FunctionalDualAnchors(base_model, len(fine_tuned_models))

    # Align synthetic inputs to task vectors
    for task_id, task_vec in enumerate(task_vectors):
        # Iterative alignment
        for _ in range(10):  # Alignment iterations
            grad = anchors.compute_task_gradient(task_id, task_vec)
            # Gradient descent step
            anchors.synthetic_inputs[task_id].data -= 0.01 * grad

    # Merge: Average task shifts across all tasks
    merged_params = copy_params(base_model)
    for name, param in merged_params.items():
        shifts = [
            task_vectors[i].get(name, torch.zeros_like(param))
            for i in range(len(task_vectors))
        ]
        # Average task shifts
        merged_params[name] += torch.stack(shifts).mean(dim=0)

    return merged_params
```

## Practical Guidance

| Aspect | Recommendation |
|--------|-----------------|
| FDA input dimension | Match embedding dimension (typically 768-2048) |
| Alignment iterations | 5-15 per task (convergence threshold check recommended) |
| Merging weights | Equal weighting for balanced tasks, proportional for imbalanced |
| Base model selection | Use largest or most general fine-tuned model as base |

**When to use:**
- Merging independently fine-tuned models without retraining
- Multi-task scenarios where parameter conflicts arise
- When post-hoc merging is required (models already trained)
- Combining domain-specific specialists with foundation models

**When NOT to use:**
- Single-task scenarios (no merging needed)
- Training from scratch (joint training is simpler)
- Extremely heterogeneous models with little overlap
- When parameter-space merging works well (simpler and faster)

**Common pitfalls:**
- Misaligned initialization of synthetic inputs (poor convergence)
- Insufficient alignment iterations (incomplete task capture)
- Using incompatible base models across tasks
- Not validating merged model on task distribution

Reference: [Model Merging with Functional Dual Anchors on arXiv](https://arxiv.org/abs/2510.21223)
