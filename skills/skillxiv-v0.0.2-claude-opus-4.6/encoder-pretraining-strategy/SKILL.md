---
name: encoder-pretraining-strategy
title: "Should We Still Pretrain Encoders with Masked Language Modeling?"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.00994"
keywords: [Pretraining Objectives, Masked Language Modeling, Causal Language Modeling, Bidirectional Attention, Transfer Learning]
description: "Choose optimal pretraining strategy for text encoders: pure MLM, pure CLM, or biphasic CLM-then-MLM training, with empirical guidance on performance across downstream tasks."
---

# Encoder Pretraining: MLM vs CLM Trade-offs and Optimal Combinations

The question of how to pretrain text encoders has evolved since BERT's introduction of masked language modeling. As decoder-based models like GPT demonstrate strong transfer learning capabilities, researchers now question whether MLM remains optimal for encoders. This work conducts a large-scale controlled study (38 models, 15,000+ evaluation runs) isolating the effects of learning objectives while controlling for model size and data exposure. The findings reveal that neither pure MLM nor pure CLM dominates universally, but biphasic pretraining—sequential CLM followed by MLM—outperforms both under fixed compute budgets.

The practical implication is significant: practitioners building encoders can leverage pretrained decoder checkpoints from existing open-source models, fine-tune them with MLM, and achieve better results than training encoders from scratch. This reuses existing infrastructure while gaining the benefits of both causal and bidirectional training.

## Core Concept

Bidirectional and causal pretraining optimize different aspects of representation learning. Causal language modeling (predicting next tokens sequentially) produces strong representations early in training with good data efficiency and fine-tuning stability. Masked language modeling (bidirectional context) converges to better ultimate performance on tasks requiring full contextual understanding. Sequential training that combines both objectives balances these trade-offs, with optimal splits varying by compute budget.

The insight is that the choice of pretraining objective should depend on downstream task requirements and compute constraints. For sequence classification and question-answering, bidirectional pretraining dominates. For early stopping or parameter-constrained settings, CLM's efficiency matters more. The biphasic strategy elegantly handles both regimes.

## Architecture Overview

The experimental setup compares three pretraining configurations:

- **Pure CLM**: Causal decoder-style training from scratch or from existing models
- **Pure MLM**: Bidirectional masked training from scratch
- **Biphasic CLM→MLM**: Sequential phase 1 with CLM, phase 2 with MLM, with variable split ratios

The same model architecture (transformer encoders, 210M to 1B parameters) is used across conditions to isolate objective effects. Evaluation covers four task categories: sequence classification, token classification, question answering, and information retrieval. All models trained on 100B uniform tokens from FineWeb-Edu dataset.

## Implementation

Set up the training framework with configurable objectives:

```python
from typing import Literal
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
import numpy as np

class ObjectiveConfig:
    """
    Configuration for different pretraining objectives.

    Supports pure CLM, pure MLM, and biphasic mixed training
    with flexible phase splits and learning rate schedules.
    """

    def __init__(self,
                 objective: Literal["clm", "mlm", "biphasic"],
                 mlm_probability: float = 0.15,
                 clm_weight: float = 1.0,
                 mlm_weight: float = 1.0,
                 biphasic_split: float = 0.25):
        """
        Initialize objective configuration.

        Args:
            objective: "clm" (causal), "mlm" (masked), "biphasic" (mixed)
            mlm_probability: Fraction of tokens to mask in MLM phase
            clm_weight: Loss weight for CLM component
            mlm_weight: Loss weight for MLM component
            biphasic_split: For biphasic, fraction of training as CLM
                           (e.g., 0.25 = 25% CLM, 75% MLM)
        """
        self.objective = objective
        self.mlm_probability = mlm_probability
        self.clm_weight = clm_weight
        self.mlm_weight = mlm_weight
        self.biphasic_split = biphasic_split if objective == "biphasic" else None

class PretrainingObjective(nn.Module):
    """
    Flexible loss module supporting CLM, MLM, and biphasic training.

    Computes appropriate loss based on objective type, with support
    for mixed objectives during biphasic pretraining.
    """

    def __init__(self, vocab_size: int, config: ObjectiveConfig):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(768, vocab_size)  # Assuming hidden dim 768
        self.loss_fn = nn.CrossEntropyLoss()

    def forward_clm(self, logits, labels, loss_mask=None):
        """
        Compute causal language modeling loss.

        Standard next-token prediction loss, ignoring padding tokens.
        """
        # Flatten batch and sequence dimensions
        logits_flat = logits[..., :-1, :].contiguous().view(-1, self.vocab_size)
        labels_flat = labels[..., 1:].contiguous().view(-1)

        if loss_mask is not None:
            loss_mask_flat = loss_mask[..., 1:].contiguous().view(-1)
            loss = self.loss_fn(logits_flat, labels_flat)
            loss = (loss * loss_mask_flat).sum() / loss_mask_flat.sum()
        else:
            loss = self.loss_fn(logits_flat, labels_flat)

        return loss

    def forward_mlm(self, logits, labels, mlm_mask):
        """
        Compute masked language modeling loss.

        Loss only on masked token positions, with special tokens and
        padding ignored.
        """
        # Only compute loss on masked positions
        logits_masked = logits[mlm_mask]
        labels_masked = labels[mlm_mask]

        logits_flat = logits_masked.view(-1, self.vocab_size)
        labels_flat = labels_masked.view(-1)

        loss = self.loss_fn(logits_flat, labels_flat)
        return loss

    def forward(self, logits, labels, mlm_mask=None, current_step=None, total_steps=None):
        """
        Compute loss based on configured objective.

        Supports pure CLM, pure MLM, and biphasic mixed training
        with automatic phase switching.
        """
        if self.config.objective == "clm":
            return self.forward_clm(logits, labels)

        elif self.config.objective == "mlm":
            return self.forward_mlm(logits, labels, mlm_mask)

        elif self.config.objective == "biphasic":
            # Determine current phase based on training progress
            if current_step is not None and total_steps is not None:
                progress = current_step / total_steps
                clm_phase_end = self.config.biphasic_split
            else:
                clm_phase_end = self.config.biphasic_split

            if progress < clm_phase_end:
                # Phase 1: CLM training
                return self.forward_clm(logits, labels) * self.config.clm_weight
            else:
                # Phase 2: MLM training
                return self.forward_mlm(logits, labels, mlm_mask) * self.config.mlm_weight

        else:
            raise ValueError(f"Unknown objective: {self.config.objective}")
```

Implement training loop with proper evaluation methodology:

```python
from torch.utils.data import DataLoader
from transformers import AdamW
import wandb

class PretrainingTrainer:
    """
    Trainer for controlled encoder pretraining studies.

    Trains models with different objectives on identical data,
    controlling for learning rates, optimizers, and schedules.
    """

    def __init__(self, model, config: ObjectiveConfig, tokenizer,
                 learning_rate: float = 1e-4):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.lr = learning_rate
        self.objective = PretrainingObjective(tokenizer.vocab_size, config)
        self.optimizer = AdamW(list(model.parameters()) +
                               list(self.objective.parameters()),
                               lr=learning_rate)

    def prepare_mlm_batch(self, batch):
        """
        Create MLM training targets by masking random tokens.

        Masks tokens following BERT strategy: 80% [MASK], 10% random token,
        10% unchanged. Creates loss mask to track masked positions.
        """
        input_ids = batch['input_ids'].clone()
        mlm_mask = torch.rand(input_ids.shape) < self.config.mlm_probability

        # Replace with [MASK] token (typically ID 103)
        mask_token_id = self.tokenizer.mask_token_id
        masked_input_ids = input_ids.clone()
        masked_input_ids[mlm_mask] = mask_token_id

        batch['input_ids'] = masked_input_ids
        batch['mlm_mask'] = mlm_mask
        batch['mlm_labels'] = input_ids

        return batch

    def train_step(self, batch, current_step, total_steps):
        """
        Single training step with appropriate loss computation.

        Handles data preparation based on objective type, computes loss,
        and performs optimization step.
        """
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)

        # Prepare MLM targets if needed
        if self.config.objective in ['mlm', 'biphasic']:
            batch = self.prepare_mlm_batch(batch)
            mlm_mask = batch['mlm_mask']
        else:
            mlm_mask = None

        # Forward pass
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = self.objective.lm_head(outputs.last_hidden_state)

        # Compute loss based on objective
        loss = self.objective(
            logits=logits,
            labels=input_ids,
            mlm_mask=mlm_mask,
            current_step=current_step,
            total_steps=total_steps
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, train_dataloader, num_epochs, total_steps,
              eval_dataloader=None):
        """
        Full training loop with optional evaluation.

        Trains with specified objective and logs metrics for downstream
        task performance evaluation.
        """
        self.model.train()
        step = 0

        for epoch in range(num_epochs):
            for batch in train_dataloader:
                loss = self.train_step(batch, step, total_steps)
                step += 1

                # Log progress
                if step % 100 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
                    print(f"  Objective: {self.config.objective}")
                    if self.config.objective == "biphasic":
                        phase = "CLM" if (step / total_steps) < self.config.biphasic_split else "MLM"
                        print(f"  Current phase: {phase}")
```

Create evaluation methodology following the paper's rigorous approach:

```python
def evaluate_downstream_tasks(model, tokenizer, task_type: str,
                              num_seeds: int = 5) -> dict:
    """
    Evaluate pretrained model on downstream task with multiple seeds.

    Following paper methodology: grid search over learning rates,
    5 random seeds per configuration, official metrics.
    """
    results = {'task': task_type, 'seeds': []}

    learning_rates = [1e-5, 3e-5, 5e-5, 1e-4, 3e-4]

    for seed in range(num_seeds):
        seed_results = []
        for lr in learning_rates:
            # Fine-tune on task
            task_metric = finetune_and_evaluate(
                model=model,
                tokenizer=tokenizer,
                task=task_type,
                lr=lr,
                seed=seed
            )
            seed_results.append(task_metric)

        # Take best learning rate for this seed
        best_metric = max(seed_results)
        results['seeds'].append(best_metric)

    # Report mean and std across seeds
    results['mean'] = np.mean(results['seeds'])
    results['std'] = np.std(results['seeds'])
    return results
```

## Practical Guidance

**Hyperparameter Table:**

| Parameter | CLM | MLM | Biphasic | Notes |
|-----------|-----|-----|----------|-------|
| Learning rate | 1e-4 | 1e-4 | 1e-4 | Keep constant across objectives for fair comparison |
| MLM probability | N/A | 0.15 | 0.15 | Fraction of tokens to mask |
| Biphasic split | N/A | N/A | 0.25 | 25% CLM, 75% MLM optimal in paper |
| Batch size | 256 | 256 | 256 | Same for all conditions |
| Warmup steps | 10K | 10K | 10K | As fraction of total steps |
| Weight decay | 0.01 | 0.01 | 0.01 | L2 regularization |

**When to Use:**
- You're building a new text encoder from scratch and can afford pretraining
- You're deciding between CLM and MLM for your specific domain
- You have compute budget constraints and need to optimize allocation
- You want to leverage existing decoder models for encoder construction
- Your downstream tasks require bidirectional context (classification, QA)

**When NOT to Use:**
- You're building a decoder/language model (use pure CLM)
- You have very limited compute (use existing pretrained models instead)
- Your downstream tasks are strictly left-to-right (use CLM)
- You need sub-linear scaling with data (pretraining may not help enough)
- Your domain is very niche and pretraining will be far from distribution

**Common Pitfalls:**
- **Unequal comparison**: Different learning rates, data, or schedules bias results. Control everything except the objective.
- **Ignoring data efficiency**: CLM's early training advantage matters for limited-data regimes. Check learning curves early.
- **Wrong evaluation protocol**: Using single seeds or skipping learning rate search underestimates performance. Follow rigorous methodology.
- **Biphasic timing**: The 25%-75% split was optimal in the paper but may vary by domain. Test split ratios on your data.
- **Task mismatch**: MLM excels at sequence classification but may underperform on token classification for CLM. Choose objective per task.
- **Ignoring inference cost**: MLM models need full bidirectional passes; CLM can use KV-cache. Count inference cost in decisions.

## Reference

Authors (2025). Should We Still Pretrain Encoders with Masked Language Modeling? arXiv preprint arXiv:2507.00994. https://arxiv.org/abs/2507.00994
