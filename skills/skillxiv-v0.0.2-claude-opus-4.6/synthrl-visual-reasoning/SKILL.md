---
name: synthrl-visual-reasoning
title: "SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.02096"
keywords: [data synthesis, visual reasoning, reinforcement learning, vision-language models, mathematical reasoning]
description: "Scale visual reasoning via automated synthesis of challenging questions from seed samples, using verification mechanisms to ensure correctness and verify RL training gains on out-of-domain visual math tasks."
---

# SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis

## Core Concept

SynthRL addresses data scarcity in visual reasoning tasks by automatically synthesizing challenging training questions from a seed dataset. Rather than manually creating thousands of visual math problems, the framework programmatically generates harder variants and verifies their correctness through automatic checkers.

The approach synthesizes 3,300+ additional challenging questions from approximately 8,000 seed samples, then trains vision-language models using reinforcement learning with verifiable reward signals. The key advantage is that synthesized data gains compound across test-time compute scaling, with improvements most pronounced on the hardest evaluation samples.

## Architecture Overview

- **Seed Question Selection**: Identify representative problems from original dataset
- **Augmentation Pipeline**: Automatically generate harder variants through question transformation
- **Verification Mechanism**: Validate correctness of synthesized questions via automated checking
- **RLVR Framework**: Reinforcement learning with verifiable rewards (not learned rewards)
- **Out-of-Domain Testing**: Evaluate on benchmarks different from training distribution
- **Scalable Data Generation**: Generate thousands of new questions with minimal manual effort

## Implementation

The following steps outline how to implement verifiable data synthesis for visual reasoning:

1. **Prepare seed dataset** - Collect initial visual reasoning problems with verified answers
2. **Select candidates for augmentation** - Identify problems suitable for difficulty increase
3. **Generate harder variants** - Apply transformations to create more challenging versions
4. **Verify correctness** - Automatically check that generated questions are correct and harder
5. **Prepare RL training** - Create question-answer pairs with verifiable reward signals
6. **Train with RL** - Optimize model using reinforcement learning on synthetic data
7. **Evaluate on benchmarks** - Test on out-of-domain visual reasoning tasks

```python
from typing import List, Dict, Tuple, Optional
import torch

class QuestionSynthesizer:
    """Generate harder variants of visual reasoning questions."""

    def __init__(self, base_model, transformer_templates: List[str]):
        self.model = base_model
        self.templates = transformer_templates

    def select_candidates(self, seed_questions: List[Dict], selection_ratio: float = 0.3) -> List[Dict]:
        """Select representative questions for augmentation."""
        num_select = int(len(seed_questions) * selection_ratio)
        # Simple heuristic: select diverse difficulty range
        sorted_by_difficulty = sorted(seed_questions,
                                     key=lambda x: x.get('difficulty', 0.5))
        candidates = (sorted_by_difficulty[:num_select//2] +
                     sorted_by_difficulty[-num_select//2:])
        return candidates

    def augment_question(self, question: Dict, transformation: str) -> Optional[Dict]:
        """Apply transformation to make question harder."""
        original_text = question["question"]
        image = question["image"]
        answer = question["answer"]

        # Example transformations for visual math problems
        augmentations = {
            "multi_step": f"First calculate intermediate result. {original_text}",
            "constraints": f"With the constraint that all values are positive: {original_text}",
            "complexity": f"This problem involves more complex relationships: {original_text}",
            "precision": f"Provide answer accurate to 3 decimal places: {original_text}"
        }

        augmented_text = augmentations.get(transformation, original_text)

        return {
            "question": augmented_text,
            "image": image,
            "original_answer": answer,
            "transformation": transformation
        }

    def generate_synthetic_dataset(self, seed_data: List[Dict],
                                  augmentations_per_question: int = 4) -> List[Dict]:
        """Generate synthetic questions from seed data."""
        candidates = self.select_candidates(seed_data)
        synthetic = []

        for question in candidates:
            for i, template in enumerate(self.templates[:augmentations_per_question]):
                aug_question = self.augment_question(question, template)
                if aug_question:
                    synthetic.append(aug_question)

        return synthetic


class QuestionVerifier:
    """Verify correctness and difficulty of synthetic questions."""

    def __init__(self, reference_solver):
        self.solver = reference_solver

    def verify_correctness(self, question: Dict) -> Tuple[bool, Optional[float]]:
        """Check if synthesized question has correct answer."""
        try:
            predicted_answer = self.solver.solve(question["image"], question["question"])
            expected_answer = question.get("original_answer")

            # Flexible comparison (exact match or numerical tolerance)
            is_correct = self._compare_answers(predicted_answer, expected_answer)
            confidence = self._estimate_confidence(predicted_answer)

            return is_correct, confidence
        except Exception as e:
            return False, None

    def verify_difficulty(self, original_question: Dict, synthetic_question: Dict) -> float:
        """Assess if synthetic question is harder than original."""
        # In practice, use model predictions or heuristics
        original_str = original_question["question"]
        synthetic_str = synthetic_question["question"]

        # Simple heuristic: length increase suggests complexity
        length_ratio = len(synthetic_str.split()) / len(original_str.split())
        return min(1.0, length_ratio)

    def _compare_answers(self, predicted, expected) -> bool:
        """Compare answers with tolerance for numerical problems."""
        if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
            return abs(predicted - expected) < 0.01
        return str(predicted).strip() == str(expected).strip()

    def _estimate_confidence(self, answer) -> float:
        """Estimate confidence in answer (simplified)."""
        return 0.95 if answer else 0.1

    def filter_synthetic_data(self, synthetic_questions: List[Dict],
                             quality_threshold: float = 0.9) -> Tuple[List[Dict], Dict]:
        """Filter synthetic questions by quality and difficulty."""
        valid_questions = []
        stats = {"total": len(synthetic_questions), "valid": 0, "rejected": 0, "avg_difficulty": 0}

        difficulties = []
        for question in synthetic_questions:
            is_correct, _ = self.verify_correctness(question)
            if not is_correct:
                stats["rejected"] += 1
                continue

            # Verify it's actually harder
            original = question.get("_original", {})
            difficulty_gain = self.verify_difficulty(original, question)

            if difficulty_gain > 0.0:  # Any difficulty increase is acceptable
                valid_questions.append(question)
                difficulties.append(difficulty_gain)
                stats["valid"] += 1

        stats["avg_difficulty"] = sum(difficulties) / len(difficulties) if difficulties else 0
        return valid_questions, stats


class VisualReasoningRLTrainer:
    """Train vision-language model with reinforcement learning on synthetic data."""

    def __init__(self, model, verifier: QuestionVerifier):
        self.model = model
        self.verifier = verifier

    def compute_reward(self, question: Dict, predicted_answer: str) -> float:
        """Compute verifiable reward signal."""
        is_correct, confidence = self.verifier.verify_correctness(question)

        # Reward is high if correct and confident
        if is_correct:
            return confidence if confidence else 0.95
        else:
            return 0.0  # No partial credit for incorrect answers

    def rl_train_step(self, batch_questions: List[Dict], optimizer,
                     discount_factor: float = 0.99) -> float:
        """Execute one RL training step."""
        total_loss = 0.0

        for question in batch_questions:
            # Generate prediction
            predicted = self.model.generate(question["image"], question["question"])

            # Compute reward
            reward = self.compute_reward(question, predicted)

            # RL loss: maximize expected reward
            # Simplified: cross-entropy loss weighted by reward
            loss = -torch.tensor(reward, dtype=torch.float32)
            total_loss += loss.item()

            # Backpropagation (simplified)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss / len(batch_questions)

    def train(self, synthetic_dataset: List[Dict], num_epochs: int = 3,
             batch_size: int = 32, learning_rate: float = 1e-4) -> Dict:
        """Train model on synthetic dataset."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        metrics = {"epoch_losses": [], "total_correct": 0, "total_samples": 0}

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(synthetic_dataset), batch_size):
                batch = synthetic_dataset[i:i+batch_size]
                loss = self.rl_train_step(batch, optimizer)
                epoch_loss += loss

            avg_loss = epoch_loss / (len(synthetic_dataset) // batch_size)
            metrics["epoch_losses"].append(avg_loss)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        return metrics


class SynthRLPipeline:
    """End-to-end pipeline for synthetic data generation and RL training."""

    def __init__(self, base_model, reference_solver, templates: List[str]):
        self.synthesizer = QuestionSynthesizer(base_model, templates)
        self.verifier = QuestionVerifier(reference_solver)
        self.trainer = VisualReasoningRLTrainer(base_model, self.verifier)

    def run_pipeline(self, seed_data: List[Dict], num_epochs: int = 3) -> Dict:
        """Run full synthesis and training pipeline."""
        print("Step 1: Generating synthetic questions...")
        synthetic = self.synthesizer.generate_synthetic_dataset(seed_data)

        print("Step 2: Verifying synthetic data quality...")
        valid_synthetic, stats = self.verifier.filter_synthetic_data(synthetic)
        print(f"  Validation: {stats['valid']}/{stats['total']} valid ({100*stats['valid']/stats['total']:.1f}%)")

        print("Step 3: Training with RL...")
        metrics = self.trainer.train(valid_synthetic, num_epochs=num_epochs)

        return {
            "synthesis_stats": stats,
            "training_metrics": metrics,
            "synthetic_dataset_size": len(valid_synthetic)
        }
```

## Practical Guidance

**Data synthesis configuration:**
- **Augmentations per question**: 3-5 variants per seed question; balance diversity with verification cost
- **Selection ratio**: 30-50% of seed data; focus on representative examples
- **Quality threshold**: 0.85-0.95; higher threshold ensures clean training data

**Verification strategy:**
- **Reference solver**: Use symbolic solver for math, multiple models for consensus checking
- **Difficulty assessment**: Measure by problem complexity metrics, not length alone
- **Rejection rate**: Expect 20-40% of initial synthetic questions to fail verification

**RL training setup:**
- **Batch size**: 32-64 depending on dataset size and GPU memory
- **Learning rate**: 1e-4 to 1e-5 for stable training on noisy RL signals
- **Epochs**: 2-5 epochs typically sufficient; monitor for overfitting

**When to use:**
- Scaling training data for visual reasoning without manual labeling
- Tasks with verifiable correct answers (math, logic, code)
- Improving out-of-domain generalization through diverse synthetic data
- Research on automated curriculum learning and data augmentation

**When NOT to use:**
- Tasks without automated verification (open-ended reasoning, creative tasks)
- Domains where synthetic data distribution differs significantly from real data
- Systems requiring exact distribution matching (GANs, etc.)
- Real-time applications where verification overhead is prohibitive

**Common pitfalls:**
- **Distribution shift**: Synthetic data may not match test distribution; validate on original benchmarks
- **Verification bias**: Incorrect reference solver validates wrong answers; use multiple verifiers
- **Difficulty plateau**: Transformations may not consistently increase difficulty; use adaptive sampling
- **Mode collapse**: RL training may overfit to verification artifacts; regularize with original data
- **Computational cost**: Verification adds significant overhead; batch verification to amortize cost

## Reference

SynthRL synthesizes 3,300+ additional challenging questions from approximately 8,000 seed samples, demonstrating consistent improvements across five visual math reasoning benchmarks. Gains are most pronounced on the hardest evaluation samples, suggesting the approach effectively elicits deeper reasoning patterns.

Original paper: "SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis" (arxiv.org/abs/2506.02096)
