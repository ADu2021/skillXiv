---
name: octopus-self-correction
title: "Learning Self-Correction in VLMs via Rollout Augmentation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.08503"
keywords: [Vision-Language Models, Self-Correction, RL Training, Rollout Augmentation, Dense Learning Signals]
description: "Enable vision-language models to self-correct by synthesizing dense training examples from existing rollouts, creating n² correction pairs from n original trajectories."
---

# Learning Self-Correction in VLMs via Rollout Augmentation

## Problem Context

Vision-language models rarely generate effective self-corrections naturally. While effective self-correction examples are extremely sparse in standard RL training, the necessary learning signals already exist within standard RL rollouts—correct and incorrect reasoning trajectories coexist for given inputs.

## Core Concept

**Correction-Specific Rollout Augmentation (Octopus)** is an RL framework that synthesizes dense self-correction training examples by recombining existing rollouts. Rather than waiting for the model to spontaneously generate corrections, the method pairs responses generated before and after a special correction token to create explicit learning signals.

## Architecture Overview

- **Format Learning (Cold-Start)**: Finetune on self-correction format using mixed sampling from policy and teacher models
- **Rollout Augmentation**: Create n² paired combinations from n original rollouts, categorizing into positive/negative samples
- **Two-Stage RL**: Stage I masks pre-correction response (learn from target); Stage II selectively unmasks (learn from both)
- **Balanced Sample Selection**: Prefer wrong→correct examples while maintaining positive/negative balance

## Implementation

**Phase 1: Format Learning**

```python
def format_learning(model, examples, teacher_model):
    """Cold-start: finetune on self-correction format"""

    for epoch in range(num_epochs):
        for example in examples:
            # Generate response from policy model
            o1_policy = model.generate(example['input'])

            # Generate response from teacher model
            o1_teacher = teacher_model.generate(example['input'])

            # Mix sampling: sometimes use policy, sometimes teacher
            if random.random() < 0.5:
                o1 = o1_policy
            else:
                o1 = o1_teacher

            # Generate correction using teacher
            o2 = teacher_model.generate_correction(example['input'], o1)

            # Format: o1 ⊕ <sc> ⊕ o2
            formatted = format_as_correction(o1, o2)

            # Fine-tune on this format
            logits = model.forward(formatted)
            loss = cross_entropy_loss(logits, formatted)
            loss.backward()
            optimizer.step()
```

**Phase 2: Rollout Augmentation**

```python
def augment_rollouts(rollouts):
    """Create n² paired combinations from n rollouts"""

    augmented = []

    # Four types of pairs
    pair_types = {
        'wrong_to_correct': [],   # Positive
        'correct_to_correct': [],  # Positive (weak signal)
        'correct_to_wrong': [],    # Negative
        'wrong_to_wrong': []       # Negative (weak signal)
    }

    for i, rollout_a in enumerate(rollouts):
        for j, rollout_b in enumerate(rollouts):
            if i == j:
                continue  # Don't pair with self

            label_a = rollout_a['label']  # Correct or incorrect
            label_b = rollout_b['label']

            pair = {
                'input': rollout_a['input'],
                'o1': rollout_a['output'],  # Pre-correction
                'o2': rollout_b['output'],  # Post-correction
                'label_a': label_a,
                'label_b': label_b
            }

            if label_a == 'wrong' and label_b == 'correct':
                pair_types['wrong_to_correct'].append(pair)
            elif label_a == 'correct' and label_b == 'correct':
                pair_types['correct_to_correct'].append(pair)
            elif label_a == 'correct' and label_b == 'wrong':
                pair_types['correct_to_wrong'].append(pair)
            else:  # wrong to wrong
                pair_types['wrong_to_wrong'].append(pair)

    # Balance sampling
    n_positive = len(pair_types['wrong_to_correct'])
    n_negative = len(pair_types['correct_to_wrong'])

    # Upsample weak positives to match strong positives
    weak_pos = pair_types['correct_to_correct']
    weak_neg = pair_types['wrong_to_wrong']

    balanced_positive = (pair_types['wrong_to_correct'] +
                        sample(weak_pos, min(len(weak_pos), n_positive)))

    balanced_negative = (pair_types['correct_to_wrong'] +
                        sample(weak_neg, min(len(weak_neg), n_negative)))

    augmented = balanced_positive + balanced_negative

    return augmented
```

**Phase 3: Two-Stage RL Training**

```python
def two_stage_rl(model, augmented_rollouts):
    """Stage I: Learn from target; Stage II: Learn from both"""

    # Stage I: Mask pre-correction response
    print("Stage I: Masking o1...")

    for epoch in range(num_epochs_stage1):
        for pair in augmented_rollouts:
            input_text = pair['input']
            o1_masked = "<masked>"  # Treat o1 as fixed context
            o2 = pair['output']

            # Format: input ⊕ <sc> ⊕ o2 (o1 is masked)
            context = f"{input_text}\n<sc>\n{o2}"

            # Compute log-probability
            logprob = model.log_probability(context, o2)

            # Compute advantage (reward signal)
            is_positive = (pair['label_a'] == 'wrong' and
                          pair['label_b'] == 'correct')
            advantage = 1.0 if is_positive else -1.0

            # RL update
            loss = -advantage * logprob
            loss.backward()
            optimizer.step()

    # Stage II: Selectively unmask o1
    print("Stage II: Unmasking o1 for non-conflicting signals...")

    for epoch in range(num_epochs_stage2):
        for pair in augmented_rollouts:
            input_text = pair['input']
            o1 = pair['o1']
            o2 = pair['o2']

            # Check if unmasking causes signal conflict
            # (e.g., pre-response and post-response have different correctness)
            label_a = pair['label_a']
            label_b = pair['label_b']

            # Only unmask for non-conflicting samples
            if label_a == label_b:
                # Both correct or both wrong: non-conflicting
                context = f"{input_text}\n{o1}\n<sc>\n{o2}"
            else:
                # Conflicting: keep masked
                context = f"{input_text}\n<masked>\n<sc>\n{o2}"

            logprob = model.log_probability(context, o2)

            is_positive = (label_a == 'wrong' and label_b == 'correct')
            advantage = 1.0 if is_positive else -1.0

            loss = -advantage * logprob
            loss.backward()
            optimizer.step()
```

## Practical Guidance

**When to use**: Deploy for VLM tasks where self-correction is valuable (visual question answering, image captioning with refinement). Highly effective when base model generates some correct outputs alongside incorrect ones.

**Format choice**: Use a special token (e.g., <sc>, <think>, <revise>) to mark correction boundaries. Token choice affects downstream performance marginally; consistency matters more.

**Pair type balancing**: Prefer wrong→correct pairs (strongest signal). Maintain positive:negative ratio around 1:1 to avoid bias. Weak pairs (correct→correct, wrong→wrong) provide regularization.

**Stage transition**: Move from Stage I to Stage II after policy converges on strong signals. Stage I typically runs for 1–2 epochs; Stage II for 2–3 epochs. Monitor loss to detect convergence.

**Data efficiency**: From n = 100 rollouts, generate up to 10,000 pairs. This 100× amplification enables RL training on limited seed data.

## Reference

Octopus achieves state-of-the-art self-correction performance while requiring only 0.72× training time compared to baseline methods. The key insight is that standard RL rollouts already contain sufficient signal for self-correction training if properly organized. Rollout augmentation extracts this signal without additional data collection, enabling efficient training of correction behaviors.
