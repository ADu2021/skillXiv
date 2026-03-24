---
name: spark-policy-reward-co-evolution
title: "SPARK: Synergistic Policy And Reward Co-Evolving Framework"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.22624"
keywords: [reinforcement learning, policy optimization, reward modeling, co-evolution, language models, on-policy training, unified objectives]
description: "Train LLMs to simultaneously act as reasoning agents and reward models through recycled on-policy rollouts, eliminating separate reward infrastructure while achieving 9.7% gains on reasoning tasks."
---

# Train Language Models with Co-Evolving Policy and Reward

## Outcome

Build a unified LLM training system that eliminates separate reward models by recycling generated rollouts to train policy and reward judgment within a single model, reducing computational overhead while improving reasoning and reward accuracy.

## Problem Context

Standard reinforcement learning for language models relies on two separate systems: a policy that generates responses and a reward model that evaluates them. This separation creates inefficiencies:

- Expensive human preference data collection for reward modeling
- Computational redundancy during training and inference (two models instead of one)
- Reward model quality often lags behind policy improvements, creating optimization misalignment
- Training data from policy improvements gets discarded rather than recycled

SPARK addresses these by creating a single model that simultaneously learns to reason better and judge answers more accurately, using the same rollouts for both objectives.

## Core Concept

The central insight is **co-evolution through data recycling**: instead of separate systems, one model learns multiple complementary objectives from the same generated rollouts. Three interrelated training tasks create a feedback loop:

1. **Pointwise scoring** teaches the model to recognize when individual responses are correct
2. **Pairwise comparison** develops preference discrimination between response qualities
3. **Reflection training** enables the model to fix its own mistakes through self-correction

When the reward component improves, it produces better policy gradients. Better policies generate higher-quality rollouts. Better rollouts train more accurate reward judgment. This creates compounding gains without external infrastructure.

## Architecture Overview

The training pipeline consists of four sequential phases per iteration:

- **Rollout generation**: Sample multiple candidate responses from the policy for each input
- **On-policy evaluation**: Assign advantage scores using verifiable ground truth (exact match, execution success, or reference comparison)
- **Data construction**: Transform rollouts into three complementary datasets for pointwise, pairwise, and reflection objectives
- **Unified optimization**: Backpropagate a combined loss that balances policy improvement with reward accuracy and drift regularization

The model architecture remains standard (transformer-based LLM or VLM). Training infrastructure differs only in objective composition, not in fundamental components.

## Implementation

### Phase 1: Rollout Generation and Evaluation

Generate multiple candidate responses per input and compute standardized advantage scores that will inform all downstream training objectives.

```python
import torch
from torch.utils.data import DataLoader

def generate_rollouts(model, inputs, num_candidates=4, temperature=0.7):
    """Generate multiple candidate responses per input."""
    rollouts = []

    for input_text in inputs:
        candidates = []
        for _ in range(num_candidates):
            # Generate with sampling
            output = model.generate(
                input_text,
                temperature=temperature,
                max_length=256,
                do_sample=True
            )
            candidates.append(output)

        rollouts.append({
            'input': input_text,
            'candidates': candidates,
            'generated_at': 'current_policy'
        })

    return rollouts

def compute_advantages(rollouts, ground_truth_labels, baseline='mean'):
    """Assign advantage scores to candidates using verifiable rewards."""
    processed = []

    for rollout in rollouts:
        input_text = rollout['input']
        candidates = rollout['candidates']
        labels = ground_truth_labels[input_text]

        # Compute individual correctness scores
        scores = []
        for candidate in candidates:
            # Use exact match, execution, or reference-based comparison
            score = evaluate_correctness(candidate, labels)
            scores.append(score)

        # Standardize advantages
        if baseline == 'mean':
            baseline_val = sum(scores) / len(scores)
        elif baseline == 'min':
            baseline_val = min(scores)
        else:
            baseline_val = 0.0

        advantages = [s - baseline_val for s in scores]

        processed.append({
            'input': input_text,
            'candidates': candidates,
            'scores': scores,
            'advantages': advantages,
            'normalized_scores': [(s - min(scores)) / (max(scores) - min(scores) + 1e-8)
                                  for s in scores]
        })

    return processed

def evaluate_correctness(candidate, ground_truth):
    """Verify answer correctness using exact match or semantic similarity."""
    if isinstance(ground_truth, list):
        # Multiple acceptable answers
        return float(candidate.strip() in [g.strip() for g in ground_truth])
    else:
        # Single reference
        return float(candidate.strip() == ground_truth.strip())
```

### Phase 2: Multi-Objective Data Construction

Transform single rollouts into three complementary datasets. Each serves a different learning objective, enabling the model to develop specialized subcomponents.

```python
def construct_training_data(processed_rollouts):
    """Create pointwise, pairwise, and reflection datasets from rollouts."""
    pointwise_data = []
    pairwise_data = []
    reflection_data = []

    for rollout in processed_rollouts:
        input_text = rollout['input']
        candidates = rollout['candidates']
        scores = rollout['normalized_scores']
        advantages = rollout['advantages']

        # POINTWISE: (input, response) -> correctness score
        for candidate, score in zip(candidates, scores):
            pointwise_data.append({
                'input': input_text,
                'response': candidate,
                'target_score': score,
                'task_type': 'pointwise'
            })

        # PAIRWISE: Compare two responses and predict preference
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                better_idx = i if scores[i] > scores[j] else j
                worse_idx = j if better_idx == i else i

                pairwise_data.append({
                    'input': input_text,
                    'response_a': candidates[better_idx],
                    'response_b': candidates[worse_idx],
                    'label': 0,  # response_a is better
                    'task_type': 'pairwise'
                })

        # REFLECTION: Train model to self-correct wrong answers
        for candidate, score in zip(candidates, scores):
            if score < 0.5:  # Incorrect response
                reflection_data.append({
                    'input': input_text,
                    'incorrect_response': candidate,
                    'task': 'Generate corrected response',
                    'task_type': 'reflection'
                })

    return {
        'pointwise': pointwise_data,
        'pairwise': pairwise_data,
        'reflection': reflection_data
    }

def prepare_batch(examples, tokenizer, max_length=512):
    """Tokenize and format batch for unified training."""
    batch = {'input_ids': [], 'attention_mask': [], 'labels': []}

    for example in examples:
        task_type = example.get('task_type', 'pointwise')

        if task_type == 'pointwise':
            # Format: "Question: {input}\nAnswer: {response}\nScore: {score}"
            text = f"Question: {example['input']}\nAnswer: {example['response']}\nScore:"
            target = str(int(example['target_score'] * 100))

        elif task_type == 'pairwise':
            # Format for preference prediction
            text = f"Compare:\nA: {example['response_a']}\nB: {example['response_b']}\nBetter:"
            target = "A"

        else:  # reflection
            text = f"Fix: {example['incorrect_response']}\nCorrected:"
            target = ""  # Will be generated

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        batch['input_ids'].append(encoded['input_ids'].squeeze())
        batch['attention_mask'].append(encoded['attention_mask'].squeeze())

    return {k: torch.stack(v) for k, v in batch.items()}
```

### Phase 3: Unified Loss Computation

Combine multiple objectives into a single loss function that balances policy improvement, reward accuracy, and KL divergence regularization.

```python
import torch.nn.functional as F

def compute_unified_loss(
    model,
    batch,
    ref_model=None,
    loss_weights=None,
    beta_kl=0.1
):
    """
    Compute combined loss across pointwise, pairwise, and reflection objectives.

    Args:
        model: LLM with unified policy+reward head
        batch: Mixed batch of [pointwise, pairwise, reflection] examples
        ref_model: Reference model for KL divergence (optional)
        loss_weights: Dict with 'pointwise', 'pairwise', 'reflection' keys
        beta_kl: KL divergence regularization coefficient
    """
    if loss_weights is None:
        loss_weights = {'pointwise': 0.4, 'pairwise': 0.3, 'reflection': 0.3}

    total_loss = 0.0
    losses = {}

    # POINTWISE LOSS: Regression on correctness scores
    pointwise_mask = batch.get('pointwise_mask', torch.zeros_like(batch['input_ids']))
    if pointwise_mask.any():
        logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits

        # Extract score predictions (final token)
        score_preds = logits[:, -1, :2]  # Assume 0-100 range mapped to 2 dimensions
        targets = batch.get('target_scores', torch.zeros(logits.shape[0]))

        pointwise_loss = F.smooth_l1_loss(score_preds.float(), targets.unsqueeze(1).float())
        losses['pointwise'] = pointwise_loss
        total_loss += loss_weights['pointwise'] * pointwise_loss

    # PAIRWISE LOSS: Classification of preference
    pairwise_mask = batch.get('pairwise_mask', torch.zeros_like(batch['input_ids']))
    if pairwise_mask.any():
        logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits

        # Binary classification: is response_a better than response_b?
        pairwise_logits = logits[:, -1, :2]
        pairwise_targets = batch.get('pairwise_labels', torch.zeros(logits.shape[0]))

        pairwise_loss = F.cross_entropy(pairwise_logits, pairwise_targets.long())
        losses['pairwise'] = pairwise_loss
        total_loss += loss_weights['pairwise'] * pairwise_loss

    # REFLECTION LOSS: Generation quality for self-correction
    reflection_mask = batch.get('reflection_mask', torch.zeros_like(batch['input_ids']))
    if reflection_mask.any():
        logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits

        # Reconstruction loss for generating corrections
        reflection_targets = batch.get('target_corrections', torch.full_like(batch['input_ids'], -100))

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['input_ids'][..., 1:].contiguous()

        reflection_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        losses['reflection'] = reflection_loss
        total_loss += loss_weights['reflection'] * reflection_loss

    # KL DIVERGENCE REGULARIZATION: Prevent policy drift
    if ref_model is not None:
        ref_logits = ref_model(batch['input_ids'], attention_mask=batch['attention_mask']).logits
        current_logits = model(batch['input_ids'], attention_mask=batch['attention_mask']).logits

        kl_loss = F.kl_div(
            F.log_softmax(current_logits, dim=-1),
            F.softmax(ref_logits.detach(), dim=-1),
            reduction='batchmean'
        )
        losses['kl_div'] = kl_loss
        total_loss += beta_kl * kl_loss

    return total_loss, losses

def training_step(
    model,
    batch,
    optimizer,
    ref_model=None,
    loss_weights=None,
    beta_kl=0.1
):
    """Perform one training step with unified objectives."""
    model.train()

    loss, component_losses = compute_unified_loss(
        model,
        batch,
        ref_model=ref_model,
        loss_weights=loss_weights,
        beta_kl=beta_kl
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), {k: v.item() for k, v in component_losses.items()}
```

### Phase 4: Complete Training Loop

Orchestrate all phases into a complete training loop that cycles between on-policy rollout collection and model optimization.

```python
def train_spark(
    model,
    train_dataloader,
    val_inputs,
    val_labels,
    num_epochs=3,
    rollout_steps_per_epoch=4,
    batch_size=16,
    learning_rate=1e-5,
    loss_weights=None,
    beta_kl=0.1
):
    """
    Train model using SPARK framework.

    Args:
        model: LLM/VLM to train as unified policy+reward
        train_dataloader: Iterator over training inputs
        val_inputs: Validation input examples
        val_labels: Ground truth labels for validation
        num_epochs: Total training epochs
        rollout_steps_per_epoch: Rollout generation iterations per epoch
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        loss_weights: Balance between pointwise/pairwise/reflection objectives
        beta_kl: KL divergence regularization strength
    """
    if loss_weights is None:
        loss_weights = {'pointwise': 0.4, 'pairwise': 0.3, 'reflection': 0.3}

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    ref_model = copy.deepcopy(model)  # For KL regularization

    training_history = []

    for epoch in range(num_epochs):
        epoch_losses = {'pointwise': [], 'pairwise': [], 'reflection': [], 'kl_div': []}

        for step, input_batch in enumerate(train_dataloader):
            # Phase 1: Generate on-policy rollouts
            rollouts = generate_rollouts(model, input_batch, num_candidates=4)
            processed = compute_advantages(rollouts, val_labels)

            # Phase 2: Construct multi-objective datasets
            training_data = construct_training_data(processed)

            # Phase 3 & 4: Unified optimization over multiple sub-steps
            for _ in range(rollout_steps_per_epoch):
                # Sample from all three objectives
                pointwise_batch = torch.utils.data.random_split(
                    training_data['pointwise'],
                    [min(batch_size, len(training_data['pointwise'])),
                     max(0, len(training_data['pointwise']) - batch_size)]
                )[0]

                pairwise_batch = torch.utils.data.random_split(
                    training_data['pairwise'],
                    [min(batch_size, len(training_data['pairwise'])),
                     max(0, len(training_data['pairwise']) - batch_size)]
                )[0]

                reflection_batch = torch.utils.data.random_split(
                    training_data['reflection'],
                    [min(batch_size // 2, len(training_data['reflection'])),
                     max(0, len(training_data['reflection']) - batch_size // 2)]
                )[0]

                # Combine and prepare
                combined = list(pointwise_batch) + list(pairwise_batch) + list(reflection_batch)
                prepared = prepare_batch(combined, model.tokenizer)

                # Training step
                loss, losses = training_step(
                    model,
                    prepared,
                    optimizer,
                    ref_model=ref_model,
                    loss_weights=loss_weights,
                    beta_kl=beta_kl
                )

                for task_type in epoch_losses:
                    if task_type in losses:
                        epoch_losses[task_type].append(losses[task_type])

            if (step + 1) % 10 == 0:
                avg_losses = {k: sum(v) / len(v) if v else 0 for k, v in epoch_losses.items()}
                print(f"Epoch {epoch}, Step {step}: {avg_losses}")

        # Periodic checkpoint and reference model update
        if (epoch + 1) % 2 == 0:
            ref_model = copy.deepcopy(model)
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')

        training_history.append(epoch_losses)

    return model, training_history
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Loss weight (pointwise) | 0.3–0.5 | Higher emphasizes score prediction; lower emphasizes generation |
| Loss weight (pairwise) | 0.2–0.4 | Preference learning weight; typically 0.3 |
| Loss weight (reflection) | 0.2–0.4 | Self-correction capability; scale with error rate in rollouts |
| Beta KL (divergence) | 0.05–0.2 | Higher prevents policy drift; 0.1 standard baseline |
| Num candidates per input | 2–8 | More candidates = better advantage estimates; diminishing returns at 4+ |
| Num rollout steps per epoch | 2–8 | Balance between on-policy freshness and efficiency |
| Learning rate | 1e-5 to 5e-5 | Standard LM fine-tuning range |
| Baseline method | 'mean' or 'min' | 'mean' reduces variance; 'min' emphasizes best rollouts |

### When to Use SPARK

- **Objective tasks with verifiable rewards**: Math, coding, information retrieval where ground truth is checkable
- **Scaling reasoning without external reward models**: When human preference annotation is bottlenecked
- **Reducing inference cost**: Single unified model vs. separate policy + reward infrastructure
- **On-policy training preferred**: When distribution drift is a concern and fresh rollouts are feasible
- **Limited preference data**: When collecting human feedback is expensive or unavailable
- **Multi-scale reasoning**: Combining pointwise judgment with pairwise preference and self-reflection

### When NOT to Use SPARK

- **Subjective tasks without ground truth**: Content that lacks verifiable correctness signals (creative writing, aesthetic judgments)
- **Preference-based RLHF only**: If your primary goal is aligning with human subjective preferences and you have preference data, DPO or IPO may be more direct
- **Fully offline data**: SPARK requires on-policy rollout generation; fully offline scenarios suit standard SFT or offline RL
- **Extreme efficiency constraints**: Rollout generation per epoch adds computational cost compared to single forward pass SFT
- **Real-time deployment requirements**: Multi-candidate generation and reflection add latency at inference
- **Noisy reward signals**: Pointwise and pairwise objectives suffer if ground truth evaluation is unreliable or partially labeled
- **Low-resource settings**: KL regularization with reference models requires memory for two model copies

### Common Pitfalls

**1. Unbalanced loss weights across objectives**: If pointwise weight dominates, the model optimizes for score prediction at the expense of generation quality. Start with equal weights and adjust after 1–2 epochs of training.

**2. Insufficient rollout diversity**: Using low temperature (< 0.5) during generation reduces candidate diversity. Meaningless comparisons occur when all candidates are similar. Maintain temperature >= 0.7.

**3. Ignoring baseline selection**: Using a fixed baseline (e.g., 0.0) instead of per-batch standardization inflates advantages early in training. Always subtract batch mean or min from scores.

**4. Reference model staleness**: If the reference model for KL divergence isn't updated periodically, policy drift regularization becomes ineffective. Refresh every 2–3 epochs or when validation accuracy plateaus.

**5. Over-weighting reflection loss on clean data**: If your evaluation function is perfect (100% of rollouts labeled correctly), reflection training wastes capacity. Scale reflection weight down when error rate < 20%.

**6. Batch size inconsistency across objectives**: Combining pointwise (which scales to 4N candidates) with pairwise (scales to 6N comparisons) in fixed batch_size can cause memory spikes. Use dynamic batching or separate sub-batches.

**7. Mixing on-policy and off-policy data**: Rollouts generated from old policies have different advantage scales. Always regenerate rollouts before each training epoch rather than recycling old data.

## Reference

Synergistic Policy And Reward Co-Evolving Framework for Large Language Models
- arXiv: https://arxiv.org/abs/2509.22624
- Submitted: September 26, 2025
- Code and resources available via official project repository
