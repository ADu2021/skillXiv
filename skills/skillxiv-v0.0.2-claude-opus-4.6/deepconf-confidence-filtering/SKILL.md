---
name: deepconf-confidence-filtering
title: "DeepConf: Test-Time Confidence-Based Filtering for Efficient Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.15260
keywords: [confidence-scoring, test-time-optimization, reasoning-efficiency, token-reduction, quality-filtering]
description: "Filter low-quality reasoning traces using model-internal confidence signals at test time, eliminating weak paths during generation to achieve 99.9% accuracy while reducing token generation by up to 84.7%."
---

# DeepConf: Test-Time Confidence-Based Filtering

## Core Concept

DeepConf improves language model reasoning through intelligent filtering of low-quality reasoning traces using internal confidence signals. Rather than generating all reasoning paths equally and selecting via majority voting, DeepConf dynamically eliminates weak paths during or after generation based on the model's own uncertainty estimates. This approach requires no additional training and achieves extreme efficiency—generating significantly fewer tokens while maintaining or improving accuracy.

## Architecture Overview

- **Internal Confidence Extraction**: Access model's hidden representations to estimate output quality
- **Dynamic Path Filtering**: Eliminate weak reasoning traces during generation
- **No Model Retraining**: Pure inference-time optimization without fine-tuning
- **Selective Token Reduction**: Reduce unnecessary computation while preserving quality paths
- **Quality Preservation**: Maintain or improve overall accuracy through selective scaling

## Implementation Steps

### 1. Extract Model Confidence Scores

Obtain internal confidence signals from model representations:

```python
import torch
import torch.nn.functional as F
from typing import Tuple, List

class ConfidenceExtractor:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def extract_confidence(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        output_ids: torch.Tensor,  # (batch, seq_len)
        method: str = "entropy"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract confidence scores using various methods.

        Returns:
        - confidence_scores: (batch, seq_len) scores in [0, 1]
        - entropy_values: (batch, seq_len) raw entropy for diagnostics
        """

        # Forward pass to get logits
        with torch.no_grad():
            outputs = self.model(input_ids, output_ids=output_ids, return_hidden_states=True)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

        if method == "entropy":
            # Entropy-based confidence: low entropy = high confidence
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float32))
            confidence = 1.0 - (entropy / max_entropy)

        elif method == "max_prob":
            # Max probability confidence: P(top prediction)
            probs = F.softmax(logits, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]

        elif method == "margin":
            # Margin between top-2 predictions
            probs = F.softmax(logits, dim=-1)
            top2_probs = torch.topk(probs, k=2, dim=-1)[0]
            confidence = top2_probs[:, :, 0] - top2_probs[:, :, 1]

        elif method == "hidden_state":
            # Confidence from hidden state norms
            hidden_states = outputs.hidden_states[-1]  # Last layer
            state_norms = torch.norm(hidden_states, dim=-1)
            normalized_norms = (state_norms - state_norms.min()) / \
                             (state_norms.max() - state_norms.min() + 1e-10)
            confidence = normalized_norms

        else:
            raise ValueError(f"Unknown confidence method: {method}")

        # Clamp to [0, 1]
        confidence = torch.clamp(confidence, 0, 1)

        return confidence, entropy if method == "entropy" else None
```

### 2. Implement Dynamic Path Filtering During Generation

Filter weak reasoning traces as they're generated:

```python
class ConfidenceFilteredGenerator:
    def __init__(self, model: torch.nn.Module, confidence_threshold: float = 0.5):
        self.model = model
        self.confidence_extractor = ConfidenceExtractor(model)
        self.confidence_threshold = confidence_threshold

    def generate_with_filtering(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        num_beams: int = 4,
        filter_at_runtime: bool = True,
        confidence_method: str = "entropy"
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Generate tokens with dynamic confidence-based filtering.

        Returns:
        - generated_ids: (batch, seq_len) filtered output
        - confidence_scores: per-token confidence
        - metrics: generation statistics
        """

        batch_size = input_ids.shape[0]
        device = input_ids.device
        current_ids = input_ids.clone()

        confidence_history = []
        filtered_positions = []
        token_count = 0
        filtered_count = 0

        for step in range(max_new_tokens):
            # Generate next token
            with torch.no_grad():
                outputs = self.model(current_ids)
                next_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)

            # Get confidence for next token
            next_probs = F.softmax(next_logits, dim=-1)

            if confidence_method == "entropy":
                entropy = -torch.sum(next_probs * torch.log(next_probs + 1e-10), dim=-1)
                max_entropy = torch.log(torch.tensor(next_logits.shape[-1], dtype=torch.float32))
                confidence = 1.0 - (entropy / max_entropy)
            elif confidence_method == "max_prob":
                confidence = torch.max(next_probs, dim=-1)[0]
            else:
                confidence = torch.ones(batch_size, device=device)

            token_count += 1
            confidence_history.append(confidence)

            # Decide whether to continue or truncate based on confidence
            if filter_at_runtime and confidence.mean() < self.confidence_threshold:
                # Weak confidence: stop generation or switch strategy
                filtered_count += 1
                filtered_positions.append(step)

                # Could also: resample more greedily, switch beam size, etc.
                if step > 100:  # Only filter after some minimum reasoning
                    break

            # Sample or argmax next token
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            if current_ids.shape[1] > input_ids.shape[1] + max_new_tokens:
                break

        metrics = {
            "total_tokens_generated": token_count,
            "filtered_positions": len(filtered_positions),
            "avg_confidence": torch.stack(confidence_history).mean().item(),
            "token_reduction": filtered_count / max(token_count, 1)
        }

        confidence_scores = torch.stack(confidence_history)

        return current_ids, confidence_scores, metrics
```

### 3. Implement Post-Generation Filtering

Filter complete reasoning traces after generation:

```python
def filter_reasoning_traces(
    generated_sequences: List[str],
    confidence_scores: List[List[float]],
    quality_threshold: float = 0.6,
    method: str = "dynamic_threshold"
) -> Tuple[List[str], List[float]]:
    """
    Filter low-quality reasoning traces after generation.

    Returns:
    - filtered_sequences: high-quality traces
    - quality_scores: overall quality per sequence
    """

    quality_scores = []
    filtered_sequences = []

    for sequence, scores in zip(generated_sequences, confidence_scores):
        # Compute overall quality metric
        if method == "mean":
            # Simple average confidence
            quality = sum(scores) / len(scores) if scores else 0.0

        elif method == "weighted_mean":
            # Weight recent tokens more (recency bias)
            weights = torch.linspace(0.5, 1.0, len(scores))
            quality = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        elif method == "min_confidence":
            # Bottleneck: trace quality = weakest link
            quality = min(scores) if scores else 0.0

        elif method == "step_count":
            # Quality based on reasoning depth
            quality = min(1.0, len(scores) / 200.0)

        else:
            quality = 0.5

        quality_scores.append(quality)

        # Filter based on threshold
        if quality >= quality_threshold:
            filtered_sequences.append(sequence)

    return filtered_sequences, quality_scores
```

### 4. Implement Best-of-N Selection with Filtering

Select best reasoning paths from multiple candidates:

```python
def select_best_reasoning_with_filtering(
    all_sequences: List[str],
    all_confidence_scores: List[List[float]],
    num_to_keep: int = 1,
    min_quality: float = 0.6,
    diversity_penalty: float = 0.0
) -> Tuple[str, dict]:
    """
    Select best reasoning path considering both quality and diversity.
    """

    # Compute quality scores for all sequences
    quality_scores = []
    for scores in all_confidence_scores:
        avg_conf = sum(scores) / len(scores) if scores else 0.0
        quality_scores.append(avg_conf)

    # Filter based on minimum quality
    candidates = [
        (seq, quality)
        for seq, quality in zip(all_sequences, quality_scores)
        if quality >= min_quality
    ]

    if not candidates:
        # Fallback: use highest quality even if below threshold
        best_idx = quality_scores.index(max(quality_scores))
        return all_sequences[best_idx], {"selected_quality": quality_scores[best_idx]}

    # Sort by quality
    candidates.sort(key=lambda x: x[1], reverse=True)

    selected = candidates[0][0]
    selected_quality = candidates[0][1]

    return selected, {
        "selected_quality": selected_quality,
        "candidates_evaluated": len(all_sequences),
        "candidates_passed_filter": len(candidates),
        "filter_rate": len(candidates) / len(all_sequences)
    }
```

### 5. Validate Efficiency Gains

Measure token reduction and quality preservation:

```python
def evaluate_deepconf(
    model: torch.nn.Module,
    test_examples: List[dict],
    baseline_generator,
    deepconf_generator
) -> dict:
    """
    Compare DeepConf against baseline generation.
    """

    baseline_results = {"tokens": [], "accuracy": []}
    deepconf_results = {"tokens": [], "accuracy": []}

    for example in test_examples:
        # Baseline generation
        baseline_output = baseline_generator.generate(example["prompt"])
        baseline_tokens = len(baseline_output.split())
        baseline_correct = evaluate_correctness(baseline_output, example["expected"])

        # DeepConf generation
        deepconf_output = deepconf_generator.generate_with_filtering(example["prompt"])
        deepconf_tokens = len(deepconf_output.split())
        deepconf_correct = evaluate_correctness(deepconf_output, example["expected"])

        baseline_results["tokens"].append(baseline_tokens)
        baseline_results["accuracy"].append(baseline_correct)
        deepconf_results["tokens"].append(deepconf_tokens)
        deepconf_results["accuracy"].append(deepconf_correct)

    # Compute metrics
    baseline_avg_tokens = sum(baseline_results["tokens"]) / len(baseline_results["tokens"])
    baseline_accuracy = sum(baseline_results["accuracy"]) / len(baseline_results["accuracy"])

    deepconf_avg_tokens = sum(deepconf_results["tokens"]) / len(deepconf_results["tokens"])
    deepconf_accuracy = sum(deepconf_results["accuracy"]) / len(deepconf_results["accuracy"])

    token_reduction = (1.0 - deepconf_avg_tokens / baseline_avg_tokens) * 100

    return {
        "baseline_tokens": baseline_avg_tokens,
        "deepconf_tokens": deepconf_avg_tokens,
        "token_reduction_pct": token_reduction,
        "baseline_accuracy": baseline_accuracy,
        "deepconf_accuracy": deepconf_accuracy,
        "accuracy_improvement": deepconf_accuracy - baseline_accuracy
    }
```

## Practical Guidance

### When to Use DeepConf

- Mathematical reasoning (AIME, competition problems)
- Complex multi-step problem solving
- Scenarios where inference cost is critical
- Best-of-N selection during evaluation
- Any domain where confidence scores are predictive of quality

### When NOT to Use

- Creative generation (poetry, storytelling)
- Low-latency, single-pass generation requirements
- Tasks with no clear quality metric
- Domains where all reasoning traces are equally valid

### Key Hyperparameters

- **confidence_threshold**: 0.4-0.7 (lower = more aggressive filtering)
- **confidence_method**: "entropy" or "max_prob" (entropy recommended)
- **filter_at_runtime**: True for streaming savings, False for post-hoc
- **min_quality**: 0.5-0.8 for filtering complete traces
- **num_beams**: 4-8 for best-of-N selection

### Performance Expectations

- Token Reduction: 50-84.7% fewer tokens generated
- Accuracy: 99.9% on AIME 2025 (near-perfect)
- Quality Preservation: Maintain or improve accuracy
- Speedup: 2-5x faster inference with token reduction

## Reference

Researchers. (2024). Deep Think with Confidence. arXiv preprint arXiv:2508.15260.
