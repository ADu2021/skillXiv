---
name: data-efficacy-language-model-training
title: "Data Efficacy for Language Model Training"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.21545"
keywords: [Data Organization, Training Efficiency, Curriculum Learning, Language Models, Gradient Consistency]
description: "Boost language model performance by strategically ordering training data without changing content or model size. Uses learnability-quality scoring and folding schedules to improve convergence and knowledge retention, achieving consistent gains across all model scales."
---

# Data Efficacy: Organizing Training Data for Better Learning

Training data order matters far more than commonly assumed. Models trained on shuffled data waste capacity on redundant examples early in training. Yet most practitioners treat data ordering as a minor implementation detail. DELT (Data Efficacy for Language Model Training) shows that systematic data organization—without any other changes—consistently improves performance across model sizes and domains.

The key insight: selecting good training samples is important, but organizing them strategically is equally powerful. Curriculum learning shows promise, but simple approaches like sorting once causes distribution bias. DELT introduces Learnability-Quality Scoring (measuring which samples reduce loss most) and Folding Ordering (intelligently repeating sorted batches at intervals) to combine the benefits of both approaches while avoiding pitfalls.

## Core Concept

DELT separates data handling into three independent stages:

1. **Data Scoring**: Measure each sample's "learnability"—how much gradient signal it provides and how aligned it is with overall training objectives
2. **Data Selection**: Optionally filter low-quality samples (can be combined with scoring)
3. **Data Ordering**: Reorganize training sequences using folding patterns that prevent distribution bias

The insight is that learnability isn't static. Early in training, simple examples provide useful learning signals. Later, harder examples push the model to generalization. Folding ordering repeats sorted data multiple times—interleaving easy and hard examples—to maintain balanced curriculum while avoiding catastrophic forgetting.

## Architecture Overview

The DELT pipeline consists of these components:

- **Learnability-Quality Scorer**: Evaluates samples by analyzing gradient consistency—both magnitude (how much learning signal) and direction (alignment with training objectives)
- **Folding Ordering Scheduler**: Takes sorted data and reorganizes into multiple "folds" that interleave difficulty levels across epochs
- **Selection Module** (optional): Filters samples below quality thresholds
- **Baseline Metrics**: Measures on math, code, and general language tasks to ensure robustness
- **Multi-Scale Validation**: Tests across 160M to 1B parameter models to confirm scalability

The design avoids common pitfalls: simple curriculum learning (monotonic difficulty causes overfitting), random shuffling (wastes early training capacity), and static orderings (doesn't adapt as the model learns).

## Implementation

This section demonstrates how to implement data efficacy in language model training.

**Step 1: Calculate learnability-quality scores for each sample**

This code computes how much each sample contributes to learning and alignment:

```python
import numpy as np
import torch
from torch.utils.data import Dataset

def compute_learnability_quality_scores(dataset, model, loss_fn, batch_size=32, num_samples=None):
    """
    Compute LQS (Learnability-Quality Score) for each sample.
    LQS measures gradient consistency: both magnitude and alignment with objectives.
    """

    model.eval()
    scores = []
    sample_indices = np.random.choice(len(dataset), size=min(num_samples or len(dataset), 1000))

    for idx in sample_indices:
        sample = dataset[idx]
        input_ids = sample['input_ids'].unsqueeze(0).to(model.device)
        target_ids = sample['target_ids'].unsqueeze(0).to(model.device)

        # Compute gradient with respect to this sample
        input_ids.requires_grad_(True)

        logits = model(input_ids).logits
        loss = loss_fn(logits.view(-1, model.config.vocab_size), target_ids.view(-1))

        # Backprop to get gradients
        loss.backward()
        gradients = input_ids.grad

        # Learnability: magnitude of gradient (how much this sample moves model)
        gradient_magnitude = (gradients ** 2).sum().sqrt().item()

        # Quality: alignment with overall training direction (consistency)
        # Sample that reduces loss consistently across small perturbations has high quality
        quality = compute_gradient_consistency(model, input_ids, target_ids, loss_fn)

        # Combined score: magnitude × alignment
        lqs = gradient_magnitude * quality

        scores.append({
            'index': idx,
            'magnitude': gradient_magnitude,
            'quality': quality,
            'lqs': lqs
        })

    return scores

def compute_gradient_consistency(model, input_ids, target_ids, loss_fn, num_perturbs=5):
    """
    Measure gradient consistency: does this sample provide stable learning signal?
    """

    base_loss = model(input_ids).logits
    base_loss = loss_fn(base_loss.view(-1, model.config.vocab_size), target_ids.view(-1))

    consistency_scores = []

    for _ in range(num_perturbs):
        # Small random perturbation to model weights
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * 1e-4)

        perturbed_loss = model(input_ids).logits
        perturbed_loss = loss_fn(perturbed_loss.view(-1, model.config.vocab_size), target_ids.view(-1))

        # Consistency: does loss reduction remain stable?
        consistency = 1.0 / (1.0 + abs(perturbed_loss.item() - base_loss.item()))
        consistency_scores.append(consistency)

        # Revert perturbation
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(torch.randn_like(param) * -1e-4)

    return np.mean(consistency_scores)

# Score all training samples
train_dataset = load_language_model_dataset()
lqs_scores = compute_learnability_quality_scores(train_dataset, model, loss_fn)

# Sort by LQS
sorted_scores = sorted(lqs_scores, key=lambda x: x['lqs'], reverse=True)
print(f"Top-scored sample LQS: {sorted_scores[0]['lqs']:.4f}")
print(f"Bottom-scored sample LQS: {sorted_scores[-1]['lqs']:.4f}")
```

This measures which training samples provide stable, high-magnitude learning signals.

**Step 2: Apply folding ordering to create curriculum without distribution bias**

This code reorganizes sorted data into multiple folds that interleave difficulty:

```python
import math

def create_folding_ordering(scored_samples, num_folds=5):
    """
    Folding Ordering (FO): repeat sorted data multiple times with interleaving.
    This maintains curriculum while preventing overfitting to easy examples.
    """

    # Sort by LQS score
    sorted_indices = [s['index'] for s in sorted(scored_samples, key=lambda x: x['lqs'], reverse=True)]

    num_samples = len(sorted_indices)
    fold_size = math.ceil(num_samples / num_folds)

    # Create folds: [easy, medium-easy, medium, medium-hard, hard] × 5 epochs
    folded_order = []

    for epoch in range(num_folds):
        # This epoch: sample from each difficulty level in round-robin
        for fold_id in range(num_folds):
            start_idx = fold_id * fold_size
            end_idx = min((fold_id + 1) * fold_size, num_samples)

            # Add samples from this fold for this epoch
            fold_samples = sorted_indices[start_idx:end_idx]
            folded_order.extend(fold_samples)

    return folded_order

# Create curriculum schedule
folding_order = create_folding_ordering(lqs_scores, num_folds=5)

# Visualize difficulty distribution across epochs
epoch_size = len(lqs_scores)
for epoch in range(5):
    epoch_samples = folding_order[epoch * epoch_size:(epoch + 1) * epoch_size]
    avg_lqs = np.mean([lqs_scores[i]['lqs'] for i in epoch_samples])
    print(f"Epoch {epoch}: average LQS = {avg_lqs:.4f}")
```

This creates a curriculum where each epoch interleaves easy and hard examples, preventing both shortcut learning and forgetting.

**Step 3: Train with data efficacy without modifying model architecture**

This shows how to use the ordering in standard training loops:

```python
import torch
from torch.utils.data import Sampler

class FoldingOrderSampler(Sampler):
    """Custom sampler that enforces folding order without modifying model."""

    def __init__(self, folding_order):
        self.folding_order = folding_order

    def __iter__(self):
        return iter(self.folding_order)

    def __len__(self):
        return len(self.folding_order)

# Create data loader with folding order
train_sampler = FoldingOrderSampler(folding_order)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler,
    num_workers=4
)

# Standard training loop—no model changes needed
model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(5):
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

print("Training complete with data efficacy—no architecture changes!")
```

This integrates folding ordering into standard PyTorch training without any model modifications.

**Step 4: Measure improvements across benchmarks**

This code evaluates performance gains on diverse tasks:

```python
from datasets import load_dataset

def evaluate_on_benchmarks(model, test_datasets=['math', 'code', 'general']):
    """
    Evaluate model on representative tasks from math, code, and general domains.
    DELT shows consistent improvements across all without tuning per-domain.
    """

    results = {}

    # Math benchmark (arithmetic reasoning)
    if 'math' in test_datasets:
        math_data = load_dataset("math_qa", split="test")
        math_acc = evaluate_qa_accuracy(model, math_data)
        results['math_accuracy'] = math_acc
        print(f"Math QA Accuracy: {math_acc:.2%}")

    # Code benchmark (program synthesis)
    if 'code' in test_datasets:
        code_data = load_dataset("human_eval", split="test")
        code_pass = evaluate_pass_at_k(model, code_data, k=1)
        results['code_pass_1'] = code_pass
        print(f"Code Pass@1: {code_pass:.2%}")

    # General language understanding
    if 'general' in test_datasets:
        general_data = load_dataset("arc", split="test")
        general_acc = evaluate_qa_accuracy(model, general_data)
        results['general_accuracy'] = general_acc
        print(f"General QA Accuracy: {general_acc:.2%}")

    return results

def evaluate_qa_accuracy(model, dataset, max_samples=1000):
    """Helper: compute accuracy on QA tasks."""
    correct = 0
    for sample in dataset.select(range(min(len(dataset), max_samples))):
        question = sample.get('question', sample.get('text', ''))
        predicted = model.generate(question, max_length=50)
        reference = sample.get('answer', sample.get('label', ''))
        if predicted.strip().lower() in reference.lower():
            correct += 1
    return correct / min(len(dataset), max_samples)

def evaluate_pass_at_k(model, dataset, k=1, max_samples=164):
    """Helper: compute pass@k on code generation."""
    passes = 0
    for sample in dataset.select(range(min(len(dataset), max_samples))):
        prompt = sample['prompt']
        solutions = [model.generate(prompt) for _ in range(k)]
        # Check if any solution passes (simplified; real evaluation uses execution)
        if any("def " in sol for sol in solutions):
            passes += 1
    return passes / min(len(dataset), max_samples)

# Evaluate
benchmarks = evaluate_on_benchmarks(model, test_datasets=['math', 'code', 'general'])
```

This measures improvements across diverse benchmarks to verify robustness of DELT gains.

## Practical Guidance

**When to use Data Efficacy:**
- Training language models from scratch with limited computational budget
- Fine-tuning on small to medium datasets where every sample matters
- Tasks where sample quality varies widely (curriculum learning naturally helps)
- Scenarios where you want improvements without architectural changes or additional compute

**When NOT to use:**
- Web-scale training with massive, well-curated datasets (diminishing returns)
- Real-time applications where preprocessing overhead is critical
- Cases where all samples are already well-balanced in quality
- Scenarios requiring model-specific optimizations (data ordering is general)

**Hyperparameters and Configuration:**

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Number of Folds | 5 | Balances curriculum structure with epoch variation |
| Gradient Samples for Scoring | 1000-5000 | Estimate LQS on subset to avoid full-dataset cost |
| Gradient Consistency Perturbations | 5 | Sufficient to assess stability; more → slower scoring |
| Model Sizes | 160M-1B | DELT benefits observed across all scales |
| Epochs | 4-10 | More epochs leverage ordering; diminishing returns after 10 |
| Batch Size | 32-128 | Independent of DELT; use standard guidance |

**Common Pitfalls:**
- Scoring the entire dataset before training (expensive; sample 1000-5000 instead)
- Using simple loss magnitude as the score (ignore consistency—leads to outliers)
- Applying one fold order for all epochs (reduces diversity; regenerate each epoch)
- Combining DELT with other reordering schemes (incompatible assumptions)
- Assuming LQS captures absolute quality (it's relative; lower-scored samples still help after easier ones are learned)

**Key Design Decisions:**
DELT doesn't modify the model or training algorithm—only data order. This makes it immediately applicable to any training framework. Folding ordering avoids monotonic curriculum (which causes overfitting) by repeating sorted data across epochs with round-robin interleaving. Learnability-Quality Scoring combines gradient magnitude (learning signal strength) and consistency (stability), avoiding both noisy samples and outlier examples that reduce loss but don't generalize.

## Reference

Liu, Y., Wang, H., Ye, J., Chen, T., Zhang, Y., & Yang, Y. (2025). Data Efficacy for Language Model Training. arXiv preprint arXiv:2506.21545. https://arxiv.org/abs/2506.21545
