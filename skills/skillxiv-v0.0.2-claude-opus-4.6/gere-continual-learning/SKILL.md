---
name: gere-continual-learning
title: GeRe - Anti-Forgetting in Continual Learning of LLMs
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.04676
keywords: [continual-learning, catastrophic-forgetting, replay-memory, llm-training]
description: "Prevents catastrophic forgetting in continual LLM learning using threshold-based margin loss with fixed general replay samples from pretraining data."
---

## GeRe: Anti-Forgetting in Continual Learning of LLMs

### Core Concept

GeRe addresses catastrophic forgetting in continual LLM learning by employing a threshold-based margin (TM) loss function on fixed general replay samples derived from pretraining. This maintains activation state consistency during replay learning, effectively mitigating both forgetting of general capabilities and performance degradation on previously learned tasks.

### Architecture Overview

- **General Sample Replay**: Small fixed set of pretraining samples
- **Threshold-Based Margin Loss**: Constrains activation states during replay
- **Task-Incremental Learning**: Learn new domains sequentially
- **Activation State Consistency**: Maintain neural patterns from pretraining

### Implementation Steps

**Step 1: Collect General Replay Samples**

Curate pretraining samples for replay:

```python
class GeneralSampleCollector:
    def __init__(self, pretraining_data, sample_size=1000):
        super().__init__()
        self.sample_size = sample_size

    def select_general_samples(self, pretraining_data):
        """Select representative pretraining samples."""
        # Stratified sampling across topics
        samples = []
        topics = self._identify_topics(pretraining_data)

        for topic in topics:
            topic_samples = [d for d in pretraining_data if d['topic'] == topic]
            num_select = self.sample_size // len(topics)
            selected = random.sample(topic_samples, min(num_select, len(topic_samples)))
            samples.extend(selected)

        return samples[:self.sample_size]

    def _identify_topics(self, data):
        """Identify topic distribution."""
        topics = {}
        for sample in data[:100]:  # Sample for efficiency
            topic = sample.get('topic', 'general')
            topics[topic] = topics.get(topic, 0) + 1
        return topics.keys()
```

**Step 2: Implement Threshold-Based Margin Loss**

Design loss for activation consistency:

```python
class ThresholdBasedMarginLoss(nn.Module):
    def __init__(self, threshold=0.5, margin=0.1):
        super().__init__()
        self.threshold = threshold
        self.margin = margin

    def compute_loss(self, original_activations, replay_activations):
        """
        Compute TM loss maintaining activation state consistency.

        Args:
            original_activations: Activations from original training
            replay_activations: Activations during replay

        Returns:
            loss: TM loss value
        """
        # Normalize activations
        orig_norm = F.normalize(original_activations, dim=-1)
        replay_norm = F.normalize(replay_activations, dim=-1)

        # Compute similarity
        similarity = torch.mm(replay_norm, orig_norm.t())

        # Threshold-based margin
        loss = 0
        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                sim = similarity[i, j]

                if sim > self.threshold:
                    # High similarity: maintain it
                    loss += torch.relu(self.margin - sim)
                else:
                    # Low similarity: don't force increase
                    loss += torch.relu(sim - self.threshold)

        return loss / (similarity.shape[0] * similarity.shape[1])
```

**Step 3: Implement Continual Learning Loop**

Train on sequential tasks:

```python
class ContinualLearningTrainer:
    def __init__(self, model, general_samples):
        super().__init__()
        self.model = model
        self.general_samples = general_samples
        self.activation_history = {}

    def learn_new_task(self, new_task_data, task_id):
        """Learn new task while replaying general samples."""
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        tm_loss_fn = ThresholdBasedMarginLoss()

        for epoch in range(3):
            # Mix new task and replay
            batch_new = random.sample(new_task_data, min(32, len(new_task_data)))
            batch_replay = random.sample(self.general_samples, min(32, len(self.general_samples)))

            # Train on new task
            for example in batch_new:
                outputs = self.model(example['input_ids'], output_hidden_states=True)
                task_loss = F.cross_entropy(outputs.logits, example['labels'])

                # Get activations
                activations = outputs.hidden_states[-1]

                # Store if first task
                if task_id == 0:
                    self.activation_history[example['id']] = activations.detach()

                optimizer.zero_grad()
                task_loss.backward()
                optimizer.step()

            # Replay general samples with TM loss
            for example in batch_replay:
                outputs = self.model(example['input_ids'], output_hidden_states=True)

                # Get current activations
                current_activations = outputs.hidden_states[-1]

                # Original activations
                if example['id'] in self.activation_history:
                    original = self.activation_history[example['id']]
                else:
                    original = current_activations.detach()

                # Compute TM loss
                tm_loss = tm_loss_fn.compute_loss(original, current_activations)

                optimizer.zero_grad()
                tm_loss.backward()
                optimizer.step()

    def evaluate_on_all_tasks(self, task_data_dict):
        """Evaluate performance on all learned tasks."""
        results = {}

        for task_id, task_data in task_data_dict.items():
            correct = 0
            total = len(task_data)

            with torch.no_grad():
                for example in task_data:
                    outputs = self.model(example['input_ids'])
                    pred = outputs.logits.argmax(dim=-1)
                    if pred == example['labels']:
                        correct += 1

            results[f'task_{task_id}'] = correct / total

        return results
```

### Practical Guidance

**Hyperparameters and Configuration**:
- General sample size: 1000-5000 samples
- TM threshold: 0.5
- TM margin: 0.1-0.2
- Learning rate: 2e-5 to 5e-5
- Replay batch size: 32

**When to Use GeRe**:
- Continual learning of LLMs on sequential domains
- Systems where forgetting general knowledge is costly
- Scenarios with limited task-specific data
- Multi-domain adaptation requiring stability

**When NOT to Use**:
- Single-domain training (no continual aspect)
- When task-specific performance is prioritized over general knowledge
- Very large models (storage overhead)

**Implementation Notes**:
- Fixed replay samples prevent distribution shifts
- TM loss more robust than standard replay approaches
- Monitor forgetting vs learning tradeoff
- Consider optimal buffer size vs memory constraints

### Reference

Paper: GeRe: Anti-Forgetting in Continual Learning of LLM
ArXiv: 2508.04676
Performance: Consistently improves upon label fitting and KL divergence replay baselines
