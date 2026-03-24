---
name: deep-ignorance-safety-filtering
title: Deep Ignorance - Pretraining Data Filtering for Safety
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.06601
keywords: [safety, data-filtering, pretraining, adversarial-robustness, dual-use]
description: "Enhances model safety by filtering dual-use topics from pretraining data, creating tamper-resistant models robust to adversarial fine-tuning without degrading unrelated capabilities."
---

## Deep Ignorance: Pretraining Data Filtering for Safety

### Core Concept

Deep Ignorance improves language model safety by removing sensitive information (such as biothreat-related content) directly from the pretraining dataset. This approach prevents harmful capabilities from being learned initially, creating more robust defenses against adversarial fine-tuning attacks compared to post-training safety methods alone.

### Architecture Overview

- **Multi-Stage Filtering Pipeline**: Identify and remove dual-use topic content during pretraining
- **Content Classification**: Detect sensitive information (biotechnology, explosives, etc.)
- **Selective Removal**: Remove problematic content while preserving general knowledge
- **Robustness Verification**: Validate resistance to adversarial fine-tuning attacks
- **Capability Preservation**: Ensure unrelated model abilities remain intact

### Implementation Steps

**Step 1: Identify Sensitive Content Patterns**

Detect dual-use topics in training data:

```python
# Pseudocode for sensitive content identification
class SensitiveContentDetector:
    def __init__(self):
        super().__init__()
        # Initialize keyword lists for different threat categories
        self.biothreat_keywords = [
            'pathogens', 'gain-of-function', 'synthesis',
            'virulence', 'transmissibility', 'weaponization'
        ]
        self.explosives_keywords = [
            'explosive synthesis', 'detonation', 'blast',
            'explosive device construction'
        ]
        self.chemical_keywords = [
            'nerve agents', 'chemical synthesis', 'toxic',
            'chemical weapon production'
        ]

    def classify_document(self, text):
        """
        Classify document content for sensitive topics.

        Args:
            text: Document text to analyze

        Returns:
            sensitivity_scores: Dict mapping threat categories to confidence
        """
        sensitivity_scores = {}
        text_lower = text.lower()

        # Biothreat scoring
        biothreat_matches = sum(1 for kw in self.biothreat_keywords
                               if kw in text_lower)
        sensitivity_scores['biothreat'] = biothreat_matches / len(self.biothreat_keywords)

        # Explosives scoring
        explosives_matches = sum(1 for kw in self.explosives_keywords
                                if kw in text_lower)
        sensitivity_scores['explosives'] = explosives_matches / len(self.explosives_keywords)

        # Chemical scoring
        chemical_matches = sum(1 for kw in self.chemical_keywords
                              if kw in text_lower)
        sensitivity_scores['chemical'] = chemical_matches / len(self.chemical_keywords)

        return sensitivity_scores

    def identify_sensitive_spans(self, text, threshold=0.3):
        """
        Identify specific text spans containing sensitive content.
        """
        import re

        sensitive_spans = []
        sentences = text.split('.')

        for sent_idx, sentence in enumerate(sentences):
            scores = self.classify_document(sentence)
            max_score = max(scores.values())

            if max_score > threshold:
                # Find character span
                start_pos = sum(len(s) + 1 for s in sentences[:sent_idx])
                end_pos = start_pos + len(sentence)

                sensitive_spans.append({
                    'span': sentence.strip(),
                    'category': max(scores, key=scores.get),
                    'confidence': max_score,
                    'start': start_pos,
                    'end': end_pos
                })

        return sensitive_spans
```

**Step 2: Implement Document Filtering**

Filter training documents based on sensitivity:

```python
# Pseudocode for document filtering pipeline
class TrainingDataFilter:
    def __init__(self, detector, threshold=0.3, preserve_percentage=0.05):
        super().__init__()
        self.detector = detector
        self.threshold = threshold
        # Keep some sensitive docs for model awareness
        self.preserve_percentage = preserve_percentage

    def filter_training_corpus(self, dataset, output_path):
        """
        Filter training corpus removing sensitive documents.

        Args:
            dataset: Training dataset with documents
            output_path: Path to save filtered dataset

        Returns:
            filtering_stats: Statistics about filtering
        """
        total_docs = len(dataset)
        filtered_docs = []
        removed_docs = []
        filtered_stats = {
            'total': total_docs,
            'removed': 0,
            'preserved': 0,
            'by_category': {}
        }

        for doc_idx, doc in enumerate(dataset):
            text = doc['text']

            # Classify sensitivity
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')

            # Compute aggregate sensitivity
            sensitivity_scores = self.detector.classify_document(text)
            max_sensitivity = max(sensitivity_scores.values())
            top_category = max(sensitivity_scores, key=sensitivity_scores.get)

            # Decision: filter or keep
            if max_sensitivity > self.threshold:
                # Randomly preserve fraction for awareness
                if np.random.random() < self.preserve_percentage:
                    filtered_docs.append({
                        **doc,
                        'filtered': False,
                        'sensitivity_score': max_sensitivity,
                        'category': top_category
                    })
                    filtered_stats['preserved'] += 1
                else:
                    removed_docs.append({
                        **doc,
                        'removed_reason': 'sensitivity',
                        'score': max_sensitivity
                    })
                    filtered_stats['removed'] += 1
                    filtered_stats['by_category'][top_category] = \
                        filtered_stats['by_category'].get(top_category, 0) + 1
            else:
                filtered_docs.append({
                    **doc,
                    'filtered': False,
                    'sensitivity_score': max_sensitivity,
                    'category': top_category
                })

        # Save filtered dataset
        with open(output_path, 'w') as f:
            for doc in filtered_docs:
                f.write(json.dumps(doc) + '\n')

        filtered_stats['retention_rate'] = len(filtered_docs) / total_docs

        return filtered_docs, filtered_stats

    def analyze_filtering_impact(self, original_dataset, filtered_dataset):
        """
        Analyze what capability changes filtering introduces.
        """
        # Compare tokenizer coverage
        original_vocab = set()
        filtered_vocab = set()

        for doc in original_dataset[:1000]:
            original_vocab.update(doc['text'].split())

        for doc in filtered_dataset[:1000]:
            filtered_vocab.update(doc['text'].split())

        vocab_loss = len(original_vocab - filtered_vocab) / len(original_vocab)

        return {
            'vocab_retention_rate': 1 - vocab_loss,
            'original_size_mb': sum(len(d['text']) for d in original_dataset) / 1e6,
            'filtered_size_mb': sum(len(d['text']) for d in filtered_dataset) / 1e6
        }
```

**Step 3: Train Model on Filtered Data**

Pretrain language model on filtered corpus:

```python
# Pseudocode for training on filtered data
class SafeModelTrainer:
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def pretrain_on_filtered_data(self, filtered_dataset, num_epochs=2):
        """
        Pretrain language model on filtered corpus.
        """
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(filtered_dataset))

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, example in enumerate(filtered_dataset):
                text = example['text']

                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=2048,
                    truncation=True,
                    return_tensors='pt'
                )

                # Forward pass
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    labels=inputs['input_ids']
                )

                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {total_loss / num_batches:.4f}")

        return self.model
```

**Step 4: Adversarial Robustness Verification**

Test resistance to adversarial fine-tuning:

```python
# Pseudocode for adversarial robustness testing
class AdversarialRobustnessVerifier:
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def adversarial_finetune_attack(self, model, jailbreak_prompts, num_steps=10000):
        """
        Attempt to unlock dangerous capabilities through fine-tuning.

        Args:
            model: Target model
            jailbreak_prompts: Prompts trying to trigger unsafe behavior
            num_steps: Number of fine-tuning steps

        Returns:
            attack_success_rate: Fraction of prompts that succeeded
        """
        optimizer = AdamW(model.parameters(), lr=1e-3)
        successful_attacks = 0

        for step in range(num_steps):
            prompt = jailbreak_prompts[step % len(jailbreak_prompts)]

            # Try to make model comply
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = model.generate(
                inputs['input_ids'],
                max_length=100,
                temperature=0.1
            )

            generated_text = self.tokenizer.decode(outputs[0])

            # Check if attack succeeded
            if self.is_unsafe_output(generated_text):
                successful_attacks += 1
                # Try to reinforce with gradient step
                inputs = self.tokenizer(prompt + generated_text, return_tensors='pt')
                forward = model(inputs['input_ids'], labels=inputs['input_ids'])
                (-forward.loss).backward()  # Negative loss to maximize likelihood
                optimizer.step()

        return successful_attacks / num_steps

    def is_unsafe_output(self, text):
        """
        Check if output contains unsafe content.
        """
        unsafe_keywords = [
            'instructions for', 'how to make',
            'step by step guide', 'harmful synthesis'
        ]
        return any(kw in text.lower() for kw in unsafe_keywords)

    def verify_robustness(self, model, jailbreak_test_suite):
        """
        Comprehensive robustness evaluation.
        """
        results = {}

        for attack_type, prompts in jailbreak_test_suite.items():
            success_rate = self.adversarial_finetune_attack(model, prompts)
            results[attack_type] = success_rate

        avg_robustness = 1 - (sum(results.values()) / len(results))

        return {
            'per_attack_robustness': results,
            'overall_robustness_score': avg_robustness
        }
```

### Practical Guidance

**Hyperparameters and Configuration**:
- Sensitivity threshold: 0.3-0.5 (adjustable per domain)
- Preservation percentage: 2-10% for residual model awareness
- Training learning rate on filtered data: 5e-5
- Adversarial attack steps: 1000-10000 depending on threat model

**When to Use Deep Ignorance**:
- Systems handling dual-use technology domains (biotech, chemicals, explosives)
- High-security applications where robustness to fine-tuning attacks is critical
- Models requiring strong guarantees against capability unlocking
- Scenarios where post-training safety may not be sufficient

**When NOT to Use**:
- Domains where restricted knowledge is necessary for legitimate use
- Models intended to assist with sensitive research
- Systems where availability of information is more critical than safety
- Applications where model should maintain full knowledge of dual-use topics

**Implementation Notes**:
- Filter at document level, not token level (preserves coherence)
- Preserve small percentage of sensitive docs so model isn't completely ignorant
- Validate that filtering doesn't negatively impact unrelated downstream tasks
- Consider domain-specific filtering criteria rather than generic keywords
- Test robustness empirically before deployment

### Reference

Paper: Deep Ignorance: Filtering Pretraining Data for Tamper-Resistant Safeguards
ArXiv: 2508.06601
Performance: Outperforms post-training safety methods by over an order of magnitude in adversarial robustness; resists 10,000-step adversarial attacks
