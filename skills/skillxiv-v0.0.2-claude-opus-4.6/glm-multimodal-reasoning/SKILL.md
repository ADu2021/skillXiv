---
name: glm-multimodal-reasoning
title: "GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.01006"
keywords: [Vision Language Models, Multimodal Reasoning, Reinforcement Learning, RLCS, Video Understanding]
description: "Train vision-language models with curriculum-based reinforcement learning (RLCS) to improve reasoning across diverse multimodal tasks. Dynamically adjust training difficulty to model capability, preventing both trivial and overly-hard examples."
---

# GLM-4.1V-Thinking: Efficient Multimodal Reasoning Through Curriculum Reinforcement Learning

Vision-language models trained on standard supervised fine-tuning (SFT) often plateau at moderate reasoning capabilities. They can answer straightforward questions about image content but struggle with complex reasoning requiring multi-step analysis, STEM problem-solving, or dynamic reasoning about video sequences. The challenge is that all training examples are treated equally regardless of difficulty—the model might spend iterations on examples it already solved perfectly while getting stuck on genuinely hard problems.

GLM-4.1V-Thinking solves this by applying curriculum learning with reinforcement learning: the training difficulty dynamically adjusts to match the model's current capability. Early training focuses on easier examples the model can learn from effectively. As the model improves, training gradually shifts to harder examples. This prevents both the wasteful learning on trivial examples and the ineffective training on examples beyond the model's current reach.

## Core Concept

The key innovation is Reinforcement Learning with Curriculum Sampling (RLCS), which combines two ideas:

1. **Offline difficulty grading**: Before training, score all examples by difficulty using multiple signals (length, reasoning steps required, multi-hop reasoning)
2. **Online capability tracking**: Monitor the model's actual performance during training to assess its current capability level
3. **Dynamic sampling**: Sample training examples based on a curriculum band—examples slightly above the model's current level provide maximum learning gain

The insight is that optimal learning happens at the "edge of competence" where examples are challenging but not impossible. Too easy and the model wastes iterations; too hard and gradients are noisy and learning stalls. RLCS automatically finds this edge as the model improves.

## Architecture Overview

GLM-4.1V-Thinking combines vision-language architecture with multi-domain reward training:

- **Vision Encoder**: Vision Transformer processing images and video frames
- **Language Model Decoder**: Large language model generating text responses
- **Temporal Video Processing**: Native video support through frame downsampling and temporal token organization
- **Multi-Domain Reward System**: Eight separate reward models for different task categories (VQA, OCR, STEM, reasoning, etc.)
- **Curriculum Scheduler**: Dynamically adjusts difficulty distribution based on per-domain performance tracking

## Implementation

**Step 1: Grade training examples by difficulty offline**

Before any training, assign difficulty scores to all examples using multiple heuristics.

```python
def grade_example_difficulty(image, question, answer, reasoning_steps):
    """
    Compute a difficulty score for a visual question-answering example.
    Uses multiple signals: question complexity, answer length, reasoning depth.
    """
    difficulty_score = 0.0

    # Signal 1: Question length and vocabulary rarity
    # Longer, more specific questions usually require deeper reasoning
    question_tokens = question.split()
    question_length_score = min(len(question_tokens) / 20.0, 1.0)  # Normalize to 0-1
    difficulty_score += 0.2 * question_length_score

    # Signal 2: Answer complexity
    # Multi-token answers with technical terms are harder to generate
    answer_tokens = answer.split()
    answer_length_score = min(len(answer_tokens) / 15.0, 1.0)
    answer_complexity = estimate_vocabulary_complexity(answer)
    difficulty_score += 0.2 * (answer_length_score + answer_complexity) / 2

    # Signal 3: Reasoning steps required
    # Extract depth from symbolic representation (e.g., how many objects to track)
    reasoning_depth = len(reasoning_steps)
    reasoning_score = min(reasoning_depth / 5.0, 1.0)  # Most questions need <5 steps
    difficulty_score += 0.3 * reasoning_score

    # Signal 4: Visual complexity
    # Images with many objects or spatial relationships are harder
    num_objects = estimate_object_count(image)
    visual_complexity = min(num_objects / 10.0, 1.0)
    difficulty_score += 0.2 * visual_complexity

    # Normalize to 0-1 range
    difficulty_score = difficulty_score / 0.9  # Total weights: 0.2+0.2+0.3+0.2 = 0.9

    return min(difficulty_score, 1.0)

def prepare_dataset_with_difficulty_scores(dataset):
    """
    Pre-process all training examples to add difficulty scores.
    This enables curriculum sampling during training.
    """
    scored_dataset = []

    for example in dataset:
        difficulty = grade_example_difficulty(
            example['image'],
            example['question'],
            example['answer'],
            example.get('reasoning_steps', [])
        )

        example['difficulty'] = difficulty
        scored_dataset.append(example)

    return scored_dataset
```

**Step 2: Implement RLCS curriculum sampler**

Create a sampler that tracks model performance and dynamically adjusts which examples appear in training batches.

```python
class CurriculumSampler:
    """
    Reinforcement Learning with Curriculum Sampling (RLCS).
    Dynamically adjusts training difficulty based on model performance.
    """

    def __init__(self, dataset, initial_difficulty_band=0.2):
        self.dataset = dataset
        self.difficulty_band = initial_difficulty_band
        self.model_capability = 0.3  # Start with easy examples
        self.performance_history = []
        self.update_interval = 100  # Update curriculum every 100 batches

        # Sort by difficulty for quick filtering
        self.dataset_sorted = sorted(dataset, key=lambda x: x['difficulty'])

    def get_curriculum_batch(self, batch_size=32):
        """
        Sample a batch with examples in the curriculum band.
        Examples are slightly above the model's current capability.
        """
        # Define difficulty band: [capability, capability + band]
        min_difficulty = self.model_capability
        max_difficulty = min(self.model_capability + self.difficulty_band, 1.0)

        # Filter examples in this band
        candidates = [
            ex for ex in self.dataset_sorted
            if min_difficulty <= ex['difficulty'] <= max_difficulty
        ]

        # If band is too narrow, expand slightly
        if len(candidates) < batch_size // 2:
            max_difficulty = min(self.model_capability + self.difficulty_band * 1.5, 1.0)
            candidates = [
                ex for ex in self.dataset_sorted
                if min_difficulty <= ex['difficulty'] <= max_difficulty
            ]

        # Randomly sample from candidates
        batch = random.sample(candidates, min(batch_size, len(candidates)))
        return batch

    def update_curriculum(self, losses, accuracies):
        """
        Update model capability estimate and difficulty band based on performance.
        As the model improves, gradually increase difficulty.
        """
        avg_loss = np.mean(losses)
        avg_accuracy = np.mean(accuracies)

        self.performance_history.append({
            'loss': avg_loss,
            'accuracy': avg_accuracy
        })

        # Capability increases as accuracy on current band improves
        if avg_accuracy > 0.85:
            # Model is doing well, increase difficulty
            self.model_capability = min(
                self.model_capability + 0.05,
                1.0
            )
        elif avg_accuracy < 0.50:
            # Model is struggling, decrease difficulty
            self.model_capability = max(
                self.model_capability - 0.03,
                0.0
            )

        # Adjust band width based on convergence
        if len(self.performance_history) > 10:
            recent_losses = [h['loss'] for h in self.performance_history[-10:]]
            loss_variance = np.var(recent_losses)

            # Tighter band if converging, wider if unstable
            if loss_variance < 0.01:
                self.difficulty_band = max(self.difficulty_band - 0.05, 0.1)
            else:
                self.difficulty_band = min(self.difficulty_band + 0.05, 0.4)
```

**Step 3: Train with multi-domain reward models**

Use separate reward models for different task types to ensure balanced learning across domains.

```python
def train_with_multi_domain_rewards(model, tokenizer, dataset, sampler,
                                   reward_models, domain_labels,
                                   num_epochs=3, learning_rate=5e-5):
    """
    Train with reinforcement learning using multi-domain reward models.
    Each domain (VQA, OCR, reasoning, etc.) gets its own reward signal.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    domain_accuracies = {domain: [] for domain in set(domain_labels.values())}

    for epoch in range(num_epochs):
        for batch_idx in range(1000):  # 1000 batches per epoch
            # Get curriculum batch
            batch = sampler.get_curriculum_batch(batch_size=16)

            # Process examples
            inputs = []
            targets = []
            domains = []

            for example in batch:
                input_text = f"Image: {example['image']}\nQuestion: {example['question']}"
                target_text = example['answer']
                domain = example.get('domain', 'general')

                inputs.append(input_text)
                targets.append(target_text)
                domains.append(domain)

            # Tokenize
            encoded_inputs = tokenizer(inputs, return_tensors='pt', padding=True).to(device)
            encoded_targets = tokenizer(targets, return_tensors='pt', padding=True).to(device)

            # Forward pass
            outputs = model(encoded_inputs['input_ids'],
                          attention_mask=encoded_inputs['attention_mask'])

            # Compute per-domain rewards
            domain_losses = {}
            domain_accs = {}

            for domain in set(domains):
                domain_mask = [d == domain for d in domains]
                domain_logits = outputs.logits[domain_mask]
                domain_targets = encoded_targets['input_ids'][domain_mask]

                # Standard cross-entropy
                ce_loss = torch.nn.functional.cross_entropy(
                    domain_logits.view(-1, domain_logits.shape[-1]),
                    domain_targets.view(-1),
                    reduction='mean'
                )

                # Domain-specific reward (e.g., accuracy bonus)
                reward_model = reward_models[domain]
                reward_score = reward_model.score_batch(domain_logits, domain_targets)

                # Combined loss: CE - reward bonus
                loss = ce_loss - 0.1 * reward_score.mean()
                domain_losses[domain] = loss
                domain_accs[domain] = (reward_score > 0.5).float().mean()

            # Aggregate losses
            total_loss = sum(domain_losses.values()) / len(domain_losses)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track accuracies for curriculum update
            for domain, acc in domain_accs.items():
                domain_accuracies[domain].append(acc.item())

            # Update curriculum every 100 batches
            if (batch_idx + 1) % 100 == 0:
                # Compute average accuracies for recent batches
                recent_accs = [
                    np.mean(domain_accuracies[d][-100:])
                    for d in domain_accuracies
                ]
                sampler.update_curriculum(
                    losses=[],
                    accuracies=recent_accs
                )

                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Loss {total_loss:.4f}, "
                      f"Capability {sampler.model_capability:.2f}")

    return model
```

**Step 4: Validate on multi-task benchmarks**

Test the model on diverse tasks to ensure broad capabilities.

```python
def evaluate_multimodal_model(model, tokenizer, eval_splits):
    """
    Evaluate model on multiple task categories to ensure balanced performance.
    """
    results = {}

    for task_name, eval_data in eval_splits.items():
        accuracies = []

        for example in eval_data:
            input_text = f"Image: {example['image']}\nQuestion: {example['question']}"
            expected_answer = example['answer']

            # Generate response
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            output_ids = model.generate(input_ids, max_length=100, temperature=0.1)
            generated_text = tokenizer.decode(output_ids[0])

            # Check if correct (exact match or similarity)
            is_correct = generated_text.strip() == expected_answer.strip()
            accuracies.append(float(is_correct))

        results[task_name] = {
            'accuracy': np.mean(accuracies),
            'count': len(eval_data)
        }

    return results
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| Initial difficulty band | 0.15-0.25 | Controls how wide the curriculum band starts |
| Capability increase per step | 0.03-0.05 | Controls curriculum speed (5% per update) |
| Accuracy threshold for increase | 0.80-0.85 | When to move to harder examples |
| Accuracy threshold for decrease | 0.40-0.50 | When to ease difficulty |
| Update frequency | Every 50-100 batches | Balance between responsiveness and stability |
| Domain reward weight | 0.05-0.15 | How much reward models influence loss |

**When to use RLCS curriculum learning:**
- You have diverse multi-task data (VQA, OCR, reasoning all together)
- You want to improve reasoning beyond what standard SFT achieves
- Your model struggles with hard examples early in training
- You have separate reward models for different task domains

**When NOT to use RLCS:**
- You have a single, well-defined task (standard SFT is simpler)
- Your examples are naturally easy-to-hard ordered
- You don't have labeled difficulty or domain information
- Training time is critical (curriculum overhead adds ~10-20%)

**Common pitfalls:**
- **Curriculum stuck at easy examples**: If capability doesn't increase, lower the accuracy threshold for difficulty increase (e.g., 0.70 instead of 0.85).
- **Reward model collapse**: Different domains might require different reward scales. Normalize per-domain rewards independently.
- **Band too narrow**: If candidates < batch_size, the curriculum can't sample enough examples. Increase band width initially.
- **Oscillating difficulty**: If capability jumps around, reduce the increase/decrease rates (0.02 instead of 0.05).

## Reference

GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable RL
https://arxiv.org/abs/2507.01006
