---
name: infialign-data-selection
title: InfiAlign - Scalable Framework for Aligning LLMs for Reasoning
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.05496
keywords: [data-selection, supervised-fine-tuning, direct-preference-optimization, reasoning]
description: "Combines SFT and DPO with robust data selection pipeline using multidimensional quality metrics. Achieves DeepSeek-R1 performance with 12% training data, enabling efficient reasoning model alignment."
---

# InfiAlign: Scalable Framework for Aligning LLMs for Reasoning

## Core Concept

InfiAlign addresses the inefficiency of reasoning model alignment by combining SFT and DPO with an intelligent data selection pipeline. Rather than training on all available data, the framework uses multidimensional quality metrics to automatically curate high-quality alignment data from open-source reasoning datasets. This dramatically reduces data requirements while maintaining or exceeding performance compared to full-data training.

## Architecture Overview

- **Multidimensional Quality Scoring**: Evaluates data across multiple dimensions (correctness, clarity, complexity)
- **Automated Data Curation**: Selects only high-quality examples for training
- **SFT Phase**: Initial fine-tuning on curated data
- **DPO Phase**: Direct Preference Optimization to refine reasoning quality
- **Data Efficiency**: Achieves strong performance with minimal training data

## Implementation Steps

### Step 1: Implement Multidimensional Quality Scoring

Create system to evaluate training data quality across multiple dimensions.

```python
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class QualityScore:
    """Quality assessment for training example."""
    correctness: float      # 0-1: Is the answer correct?
    clarity: float          # 0-1: How clear is the reasoning?
    complexity: float       # 0-1: How complex is the problem?
    efficiency: float       # 0-1: How efficient is the solution?
    diversity: float        # 0-1: How different from other examples?
    overall: float          # Weighted combination

class QualityEvaluator:
    """
    Multi-dimensional quality scoring for reasoning examples.
    """

    def __init__(self, verifier_model, embedding_model):
        self.verifier = verifier_model
        self.embedder = embedding_model
        self.example_embeddings = []

    def evaluate_example(self, problem: str, solution: str, answer: str) -> QualityScore:
        """
        Evaluate quality of training example.

        Args:
            problem: Problem statement
            solution: Reasoning steps
            answer: Final answer

        Returns:
            Multi-dimensional quality score
        """
        # 1. Correctness: Is the answer correct?
        correctness = self._evaluate_correctness(problem, answer)

        # 2. Clarity: Is reasoning easy to follow?
        clarity = self._evaluate_clarity(solution)

        # 3. Complexity: How hard is the problem?
        complexity = self._evaluate_complexity(problem)

        # 4. Efficiency: How concise is the solution?
        efficiency = self._evaluate_efficiency(solution)

        # 5. Diversity: How different from other examples?
        diversity = self._evaluate_diversity(solution)

        # Overall score (weighted combination)
        overall = (
            0.4 * correctness +
            0.2 * clarity +
            0.15 * complexity +
            0.15 * efficiency +
            0.1 * diversity
        )

        return QualityScore(
            correctness=correctness,
            clarity=clarity,
            complexity=complexity,
            efficiency=efficiency,
            diversity=diversity,
            overall=overall
        )

    def _evaluate_correctness(self, problem: str, answer: str) -> float:
        """Verify answer correctness."""
        prompt = f"""
        Problem: {problem}
        Proposed answer: {answer}

        Is this answer correct? Respond with CORRECT or INCORRECT.
        """

        response = self.verifier.generate(prompt)

        return 1.0 if "CORRECT" in response.upper() else 0.0

    def _evaluate_clarity(self, solution: str) -> float:
        """Assess reasoning clarity."""
        # Heuristic: longer, more detailed solutions tend to be clearer
        token_count = len(solution.split())

        # Optimal length: 50-500 tokens
        if 50 <= token_count <= 500:
            clarity = 1.0
        elif token_count < 50:
            clarity = token_count / 50.0  # Too short
        else:
            clarity = max(0.5, 1.0 - (token_count - 500) / 2000)  # Too long

        # Also check for explanatory phrases
        explanation_keywords = ["because", "therefore", "step", "reason", "thus"]
        keyword_count = sum(1 for kw in explanation_keywords if kw in solution.lower())

        clarity = (clarity + min(keyword_count / 3.0, 1.0)) / 2.0

        return clarity

    def _evaluate_complexity(self, problem: str) -> float:
        """Estimate problem difficulty."""
        # Multiple signals
        complexity_signals = []

        # Signal 1: Problem length
        word_count = len(problem.split())
        length_score = min(word_count, 200) / 200.0
        complexity_signals.append(length_score * 0.3)

        # Signal 2: Mathematical symbols/operations
        math_ops = problem.count("+") + problem.count("-") + problem.count("*") + problem.count("/")
        op_score = min(math_ops, 10) / 10.0
        complexity_signals.append(op_score * 0.4)

        # Signal 3: Constraint count
        constraints = problem.count("constraint") + problem.count("condition") + problem.count("must")
        constraint_score = min(constraints, 5) / 5.0
        complexity_signals.append(constraint_score * 0.3)

        complexity = sum(complexity_signals)

        return min(complexity, 1.0)

    def _evaluate_efficiency(self, solution: str) -> float:
        """Assess solution efficiency."""
        # Simpler solutions tend to be more efficient
        token_count = len(solution.split())

        # Efficiency inversely related to length (but not too short)
        if token_count < 20:
            efficiency = 0.5  # Too terse
        elif token_count < 200:
            efficiency = 1.0  # Optimal
        else:
            efficiency = max(0.3, 1.0 - (token_count - 200) / 1000)

        return efficiency

    def _evaluate_diversity(self, solution: str) -> float:
        """Measure how different from existing examples."""
        if not self.example_embeddings:
            return 1.0  # First example is maximally diverse

        # Get embedding of new solution
        new_embedding = self.embedder.encode(solution)

        # Compare to existing embeddings
        similarities = [
            np.dot(new_embedding, existing) / (np.linalg.norm(new_embedding) * np.linalg.norm(existing))
            for existing in self.example_embeddings
        ]

        # Diversity is inverse of max similarity
        max_similarity = max(similarities)
        diversity = 1.0 - max_similarity

        return diversity

    def score_dataset(self, examples: List[Dict]) -> List[Tuple[Dict, QualityScore]]:
        """
        Score entire dataset.

        Args:
            examples: List of (problem, solution, answer) dicts

        Returns:
            Scored examples sorted by quality
        """
        scored = []

        for example in examples:
            score = self.evaluate_example(
                example["problem"],
                example["solution"],
                example["answer"]
            )

            # Track embedding for diversity
            embedding = self.embedder.encode(example["solution"])
            self.example_embeddings.append(embedding)

            scored.append((example, score))

        # Sort by overall quality
        scored.sort(key=lambda x: x[1].overall, reverse=True)

        return scored
```

### Step 2: Implement Data Selection Pipeline

Create smart data selection based on quality scores.

```python
class DataSelectionPipeline:
    """
    Intelligently select training data based on quality metrics.
    """

    def __init__(self, quality_evaluator: QualityEvaluator):
        self.evaluator = quality_evaluator

    def select_data(
        self,
        examples: List[Dict],
        target_size: int = 5000,
        min_quality: float = 0.6,
        quality_distribution: str = "balanced"
    ) -> List[Dict]:
        """
        Select data for training.

        Args:
            examples: Candidate examples
            target_size: Target number of examples
            min_quality: Minimum quality threshold
            quality_distribution: How to distribute quality levels

        Returns:
            Selected training data
        """
        # Score all examples
        scored = self.evaluator.score_dataset(examples)

        # Filter by quality threshold
        filtered = [
            (ex, score) for ex, score in scored
            if score.overall >= min_quality
        ]

        print(f"After quality filter: {len(filtered)} / {len(scored)} examples")

        # Select based on distribution strategy
        if quality_distribution == "balanced":
            selected = self._select_balanced(filtered, target_size)
        elif quality_distribution == "top_k":
            selected = self._select_top_k(filtered, target_size)
        elif quality_distribution == "stratified":
            selected = self._select_stratified(filtered, target_size)
        else:
            selected = self._select_top_k(filtered, target_size)

        return [ex for ex, _ in selected]

    def _select_top_k(self, scored: List, k: int) -> List:
        """Select top k by overall score."""
        return scored[:min(k, len(scored))]

    def _select_balanced(self, scored: List, k: int) -> List:
        """Select balanced mix across quality dimensions."""
        # Group by complexity
        easy = [s for s in scored if s[1].complexity < 0.33]
        medium = [s for s in scored if 0.33 <= s[1].complexity < 0.67]
        hard = [s for s in scored if s[1].complexity >= 0.67]

        # Select from each group proportionally
        selected = []

        ratios = [len(easy), len(medium), len(hard)]
        total = sum(ratios)

        if total > 0:
            k_easy = int(k * ratios[0] / total)
            k_medium = int(k * ratios[1] / total)
            k_hard = k - k_easy - k_medium

            selected.extend(easy[:k_easy])
            selected.extend(medium[:k_medium])
            selected.extend(hard[:k_hard])

        return selected

    def _select_stratified(self, scored: List, k: int) -> List:
        """Select stratified across multiple dimensions."""
        # Bucket by correctness and complexity
        buckets = {}

        for ex, score in scored:
            correctness_bucket = int(score.correctness * 2)  # 0, 1
            complexity_bucket = int(score.complexity * 3)    # 0, 1, 2

            key = (correctness_bucket, complexity_bucket)

            if key not in buckets:
                buckets[key] = []

            buckets[key].append((ex, score))

        # Select proportionally from each bucket
        selected = []
        per_bucket = max(1, k // len(buckets))

        for bucket_examples in buckets.values():
            selected.extend(bucket_examples[:per_bucket])

        return selected[:k]
```

### Step 3: Implement SFT + DPO Training

Combine supervised fine-tuning with direct preference optimization.

```python
class InfiAlignTrainer:
    """
    SFT + DPO training with data selection.
    """

    def __init__(self, model, data_selector):
        self.model = model
        self.data_selector = data_selector

    def train(
        self,
        training_data: List[Dict],
        num_epochs: int = 3,
        sft_weight: float = 0.7,
        dpo_weight: float = 0.3
    ):
        """
        Train using SFT and DPO.

        Args:
            training_data: Selected training data
            num_epochs: Training epochs
            sft_weight: Weight for SFT loss
            dpo_weight: Weight for DPO loss
        """
        for epoch in range(num_epochs):
            total_loss = 0

            for batch in create_batches(training_data, batch_size=32):
                # SFT loss: standard language modeling
                sft_loss = self._compute_sft_loss(batch)

                # DPO loss: prefer better solutions
                dpo_loss = self._compute_dpo_loss(batch)

                # Combined loss
                total_loss = sft_weight * sft_loss + dpo_weight * dpo_loss

                # Backward pass
                total_loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()

            print(f"Epoch {epoch}: Loss={total_loss / len(training_data):.4f}")

    def _compute_sft_loss(self, batch):
        """Standard SFT loss."""
        loss = 0
        for example in batch:
            prompt = f"Problem: {example['problem']}\nSolution:"
            target = example['solution']

            # Language modeling loss
            loss += self.model.compute_language_modeling_loss(prompt, target)

        return loss / len(batch)

    def _compute_dpo_loss(self, batch):
        """Direct Preference Optimization loss."""
        # For each example, create preference pair
        dpo_loss = 0

        for example in batch:
            preferred = example['solution']  # High-quality solution
            prompt = f"Problem: {example['problem']}\nSolution:"

            # Generate alternative (dispreferred)
            dispreferred = self.model.generate(prompt, temperature=0.5)

            # DPO: prefer chosen over rejected
            preferred_logprobs = self.model.get_logprobs(prompt, preferred)
            dispreferred_logprobs = self.model.get_logprobs(prompt, dispreferred)

            # Contrastive loss
            loss = -torch.log(
                torch.sigmoid(preferred_logprobs - dispreferred_logprobs)
            )

            dpo_loss += loss

        return dpo_loss / len(batch)
```

## Practical Guidance

### When to Use InfiAlign

- **Efficient reasoning model training**: Limited compute/data budgets
- **Open-source dataset utilization**: Leverage existing reasoning datasets
- **Multi-metric data quality**: Complex domains requiring multidimensional evaluation
- **Production alignment**: Data efficiency critical for cost

### When NOT to Use InfiAlign

- **Abundant high-quality data**: Full-data training may be simpler
- **Custom domains**: Generic quality metrics may not apply
- **Real-time data augmentation**: Selection pipeline adds latency
- **Unknown data distribution**: Quality metrics untested on domain

### Hyperparameter Recommendations

- **Quality thresholds**: 0.5-0.7 overall score for selection
- **Correctness weight**: 0.4 (highest priority)
- **Clarity weight**: 0.2 (important for reasoning)
- **Complexity weight**: 0.15 (prefer diverse problems)
- **SFT/DPO split**: 70% SFT, 30% DPO

### Key Insights

The critical innovation is recognizing that data quality matters more than quantity. By carefully evaluating examples across multiple dimensions and selecting only high-quality data, InfiAlign achieves comparable performance to full-dataset training with 10x less data. The multidimensional scoring ensures balanced representation across difficulty and reasoning style.

## Reference

**InfiAlign: Scalable Framework for Aligning LLMs for Reasoning** (arXiv:2508.05496)

Combines SFT and DPO with multidimensional data selection pipeline. Achieves DeepSeek-R1-distill performance using only 12% of training data through intelligent curation of reasoning examples.
