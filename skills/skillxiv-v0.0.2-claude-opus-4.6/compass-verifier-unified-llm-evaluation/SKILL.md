---
name: compass-verifier-unified-llm-evaluation
title: CompassVerifier - Unified Robust Verifier for LLMs Evaluation
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03686
keywords: [verification, reward-model, llm-evaluation, answer-matching]
description: "Lightweight verifier model for evaluating LLM outputs across multiple domains, handling diverse answer types through meta-error pattern analysis."
---

## CompassVerifier: Unified LLM Output Verification

CompassVerifier is a specialized reward model designed to verify LLM-generated answers against reference solutions across diverse domains (math, knowledge, reasoning). Rather than using brittle regex matching or computationally expensive general-purpose LLMs, it functions as a lightweight, domain-aware verifier that handles complex answer formats and edge cases.

### Core Concept

LLM evaluation requires matching unstructured outputs to reference answers—a task fraught with pitfalls. Simple string matching fails for equivalent answers (e.g., "1/2" vs "0.5"), general LLM verifiers are expensive and inconsistent, and handcrafted rules don't generalize. CompassVerifier combines the efficiency of specialized models with the generality of learned patterns, achieving this through meta-error analysis: studying common failure modes to anticipate edge cases.

### Architecture Overview

- **Lightweight Verifier Model**: Smaller than full instruction-tuned LLMs, trained specifically for verification rather than general capability
- **Multi-Domain Training**: Unified architecture handling mathematics, knowledge QA, and diverse reasoning tasks
- **Meta-Error Pattern Analysis**: Systematic analysis of error patterns (e.g., formatting variations, numerical equivalences) to inform training data augmentation
- **VerifierBench Benchmark**: Comprehensive dataset of model outputs augmented with pattern-based variations to stress-test verification robustness
- **Robustness to Abnormalities**: Detects malformed, incomplete, or nonsensical responses without explicit rules

### Implementation Steps

**Step 1: Construct VerifierBench with Meta-Error Analysis**

Analyze error patterns from existing evaluations to create comprehensive training data:

```python
import json
import re
from collections import defaultdict

def analyze_meta_error_patterns(llm_outputs, reference_answers):
    """
    Systematically identify common failure modes in LLM outputs.
    Use these patterns to augment training data for robustness.
    """
    error_patterns = defaultdict(list)

    for output, reference in zip(llm_outputs, reference_answers):
        # Tokenize both
        output_tokens = output.lower().split()
        ref_tokens = reference.lower().split()

        # Pattern 1: Correct content but different formatting
        if normalize_text(output) == normalize_text(reference):
            error_patterns['format_variation'].append((output, reference))

        # Pattern 2: Correct numerical value but different representation
        if extract_numbers(output) == extract_numbers(reference):
            error_patterns['numerical_equivalence'].append((output, reference))

        # Pattern 3: Partially correct (e.g., multi-part question, got some parts)
        partial_match_ratio = len(set(output_tokens) & set(ref_tokens)) / max(len(output_tokens), len(ref_tokens))
        if 0.5 < partial_match_ratio < 1.0:
            error_patterns['partial_correctness'].append((output, reference, partial_match_ratio))

        # Pattern 4: Correct answer but with extra explanation
        if reference in output:
            error_patterns['answer_within_text'].append((output, reference))

        # Pattern 5: Off-by-one or minor numerical errors
        nums_out = extract_numbers(output)
        nums_ref = extract_numbers(reference)
        if len(nums_out) == len(nums_ref) and all(abs(o - r) <= 1 for o, r in zip(nums_out, nums_ref)):
            error_patterns['off_by_one'].append((output, reference))

    return error_patterns

def normalize_text(text):
    """Remove punctuation, extra whitespace, normalize case."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def extract_numbers(text):
    """Extract all numerical values from text."""
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]

# Analyze patterns from development set
patterns = analyze_meta_error_patterns(dev_outputs, dev_references)

# Print pattern summary
for pattern_type, examples in patterns.items():
    print(f"{pattern_type}: {len(examples)} instances")
```

**Step 2: Augment Training Data Using Meta-Error Patterns**

Create synthetic variations based on discovered patterns:

```python
def augment_training_data_by_patterns(base_examples, patterns, augmentation_factor=3):
    """
    Generate augmented training examples by applying discovered error patterns.
    This teaches the verifier to handle real-world variations.
    """
    augmented = base_examples.copy()

    # For each pattern type, generate variations
    for pattern_type, examples in patterns.items():
        for output, reference in examples[:augmentation_factor]:

            if pattern_type == 'format_variation':
                # Add variations with different spacing, punctuation
                variations = [
                    output.replace(' ', ''),  # Remove spaces
                    output.title(),  # Title case
                    ' '.join(output.split()),  # Normalize whitespace
                ]
                for var in variations:
                    augmented.append({'output': var, 'reference': reference, 'correct': 1})

            elif pattern_type == 'numerical_equivalence':
                # Add equivalent numerical representations
                nums = extract_numbers(output)
                if len(nums) == 1:
                    num = nums[0]
                    variations = [
                        output.replace(str(int(num)), str(num / 2) + ' * 2'),  # Different form
                        output.replace(str(int(num)), f'{num:.4f}'),  # Different precision
                    ]
                    for var in variations:
                        augmented.append({'output': var, 'reference': reference, 'correct': 1})

            elif pattern_type == 'answer_within_text':
                # Add answer embedded in explanation
                variations = [
                    output + " So the answer is " + reference,
                    "The answer to this is " + reference + ". " + output,
                ]
                for var in variations:
                    augmented.append({'output': var, 'reference': reference, 'correct': 1})

    return augmented

augmented_data = augment_training_data_by_patterns(base_train_data, patterns)
```

**Step 3: Design the Verifier Architecture**

Build a lightweight but expressive model for answer verification:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class CompassVerifier(nn.Module):
    """
    Lightweight verifier that takes (output, reference) pair and predicts correctness.
    """
    def __init__(self, base_model='distilbert-base-uncased', hidden_dim=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary: correct or incorrect
        )

    def forward(self, outputs, references):
        """
        Args:
            outputs: List of LLM-generated answers
            references: List of reference answers

        Returns:
            scores: (B, 2) logits for [incorrect, correct]
        """
        # Encode output-reference pairs jointly and separately
        pair_texts = [f"Output: {o} Reference: {r}" for o, r in zip(outputs, references)]
        pair_encoded = self.tokenizer(pair_texts, return_tensors='pt', padding=True, truncation=True)
        pair_reps = self.encoder(**pair_encoded).pooler_output

        # Also encode separately to capture differences
        output_encoded = self.tokenizer(outputs, return_tensors='pt', padding=True, truncation=True)
        output_reps = self.encoder(**output_encoded).pooler_output

        ref_encoded = self.tokenizer(references, return_tensors='pt', padding=True, truncation=True)
        ref_reps = self.encoder(**ref_encoded).pooler_output

        # Concatenate: joint representation + difference
        combined = torch.cat([pair_reps, (output_reps - ref_reps).abs()], dim=1)

        # Classify
        logits = self.classifier(combined)
        return logits

    def predict(self, outputs, references):
        """Return probability of correctness."""
        with torch.no_grad():
            logits = self.forward(outputs, references)
            probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # Probability of correct class
```

**Step 4: Train with Domain-Specific Loss**

Account for domain-specific verification characteristics:

```python
def domain_specific_loss(logits, labels, domain_type='math'):
    """
    Adjust loss weights based on domain characteristics.
    Math: high precision required, penalize false positives more.
    Knowledge: more tolerance for paraphrasing, penalize false negatives.
    """
    base_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')

    if domain_type == 'math':
        # Penalize false positives (incorrect answers verified as correct) more heavily
        weights = torch.where(labels == 0, torch.tensor(2.0), torch.tensor(1.0))
    elif domain_type == 'knowledge':
        # Penalize false negatives (correct paraphrases marked incorrect) more
        weights = torch.where(labels == 1, torch.tensor(2.0), torch.tensor(1.0))
    else:
        weights = torch.ones_like(labels, dtype=torch.float)

    weighted_loss = base_loss * weights
    return weighted_loss.mean()

# Training loop with multi-domain batching
for batch in dataloader:
    outputs = batch['output']
    references = batch['reference']
    labels = batch['correct']
    domain = batch['domain']

    logits = model(outputs, references)
    loss = domain_specific_loss(logits, labels, domain)

    loss.backward()
    optimizer.step()
```

**Step 5: Implement Abnormality Detection**

Identify and handle malformed or suspicious outputs:

```python
def detect_abnormalities(output, reference, confidence_threshold=0.1):
    """
    Detect outputs that are clearly invalid regardless of reference.
    Returns abnormality flags and confidence penalty.
    """
    flags = {'is_abnormal': False, 'reasons': []}
    confidence_penalty = 1.0

    # Check 1: Output is empty or suspiciously short
    if len(output.strip()) < 3:
        flags['is_abnormal'] = True
        flags['reasons'].append('too_short')
        confidence_penalty *= 0.1

    # Check 2: Output contains only special tokens/gibberish
    if len(re.sub(r'[\W_]', '', output)) < len(output) * 0.3:
        flags['is_abnormal'] = True
        flags['reasons'].append('mostly_special_chars')
        confidence_penalty *= 0.3

    # Check 3: Length mismatch suggests hallucination (>5x reference length)
    if len(output.split()) > 5 * len(reference.split()) + 50:
        flags['is_abnormal'] = True
        flags['reasons'].append('excessive_length')
        confidence_penalty *= 0.5

    # Check 4: Repeated tokens (indicator of decoding failure)
    tokens = output.split()
    if len(tokens) > 10:
        max_freq = max(tokens, key=tokens.count)
        freq = tokens.count(max_freq) / len(tokens)
        if freq > 0.3:
            flags['is_abnormal'] = True
            flags['reasons'].append('token_repetition')
            confidence_penalty *= 0.4

    return flags, confidence_penalty

def verify_with_abnormality_handling(model, outputs, references):
    """
    Verify answers while accounting for abnormalities.
    """
    scores = model.predict(outputs, references)

    for i, (output, reference) in enumerate(zip(outputs, references)):
        abnormality_flags, penalty = detect_abnormalities(output, reference)
        if abnormality_flags['is_abnormal']:
            scores[i] *= penalty

    return scores
```

**Step 6: Evaluate on VerifierBench**

Comprehensive evaluation across domains and error patterns:

```python
def evaluate_verifier(model, test_set, verbose=True):
    """
    Evaluate verifier on VerifierBench with per-domain and per-pattern metrics.
    """
    metrics = {}

    # Overall accuracy
    outputs = [ex['output'] for ex in test_set]
    references = [ex['reference'] for ex in test_set]
    labels = [ex['correct'] for ex in test_set]

    preds = model.predict(outputs, references) > 0.5
    accuracy = (preds == labels).float().mean()
    metrics['overall_accuracy'] = accuracy

    # Per-domain metrics
    by_domain = defaultdict(list)
    for ex, pred in zip(test_set, preds):
        by_domain[ex.get('domain', 'unknown')].append((pred, ex['correct']))

    for domain, domain_results in by_domain.items():
        preds_d = torch.tensor([p for p, _ in domain_results])
        labels_d = torch.tensor([l for _, l in domain_results])
        acc = (preds_d == labels_d).float().mean()
        metrics[f'accuracy_{domain}'] = acc

        if verbose:
            print(f"{domain}: {acc:.3f}")

    # Per-error-pattern metrics
    by_pattern = defaultdict(list)
    for ex, pred in zip(test_set, preds):
        pattern = ex.get('error_pattern', 'clean')
        by_pattern[pattern].append((pred, ex['correct']))

    for pattern, pattern_results in by_pattern.items():
        preds_p = torch.tensor([p for p, _ in pattern_results])
        labels_p = torch.tensor([l for _, l in pattern_results])
        acc = (preds_p == labels_p).float().mean()
        metrics[f'pattern_{pattern}'] = acc

    return metrics
```

### Practical Guidance

**When to Use:**
- Evaluating LLM outputs across multiple domains (math, knowledge, reasoning)
- Scenarios requiring high-precision answer verification without false positives
- Applications where verification cost is significant (training reward models, filtering)
- Cases with diverse answer formats (numerical, textual, mixed)

**When NOT to Use:**
- Simple binary judgments where ground truth is readily available
- Domains with very specific answer formats (medical coding, legal citations)
- Real-time inference with <10ms latency requirements
- Scenarios where a single general-purpose LLM verifier is acceptable

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `hidden_dim` | 256 | Model capacity; larger = more expressive but slower |
| `confidence_threshold` | 0.5 | Adjust per use case; higher = only very confident predictions |
| `domain_loss_weight_math` | 2.0 | Penalize false positives in math; increase for stricter verification |
| `domain_loss_weight_knowledge` | 1.5 | Penalize false negatives; account for paraphrasing tolerance |
| `abnormality_penalty_threshold` | 0.1 | Confidence floor for abnormal outputs; lower = more tolerance |

**Training Tips:**
- Start with base model pre-trained on verification-adjacent tasks (semantic similarity)
- Use stratified sampling to balance domains and error patterns during training
- Monitor false positive and false negative rates separately (domain-specific trade-offs)
- Validate on held-out domains not seen during training to measure generalization

### Reference

**Paper**: CompassVerifier: A Unified and Robust Verifier for LLMs Evaluation and Outcome Reward (2508.03686)
- Lightweight alternative to general-purpose LLM verifiers
- Robust to answer format variations through meta-error pattern analysis
- Comprehensive evaluation on VerifierBench across mathematics, knowledge, and reasoning domains
