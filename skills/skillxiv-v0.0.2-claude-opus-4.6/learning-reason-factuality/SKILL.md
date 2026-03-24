---
name: learning-reason-factuality
title: Learning to Reason for Factuality
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.05618
keywords: [factuality, hallucination-reduction, reward-design, reasoning-models]
description: "Multi-dimensional reward function combining factual precision, response detail, and answer relevance for online RL. Reduces hallucinations 23.1% while maintaining helpfulness and detail."
---

# Learning to Reason for Factuality

## Core Concept

Reasoning models prone to hallucinations when generating long-form factual content, yet traditional reward functions like FActScore cause reward hacking by reducing output quality. This paper proposes a multi-dimensional reward function that simultaneously optimizes for factual accuracy, response detail level, and answer relevance. Applied through online RL, the approach significantly reduces hallucinations while maintaining helpful, detailed responses.

## Architecture Overview

- **Multi-dimensional Reward Function**: Balances accuracy, detail, and relevance
- **Factual Precision Metric**: Measures proportion of hallucination-free content
- **Detail Level Assessment**: Ensures sufficient response completeness
- **Relevance Scoring**: Verifies information pertains to query
- **Online RL Training**: Iteratively improves reasoning with nuanced rewards

## Implementation Steps

### Step 1: Design Multi-dimensional Reward Function

Create reward function balancing multiple objectives.

```python
import torch
from typing import Tuple, Dict

class MultiDimensionalRewardFunction:
    """
    Compute rewards balancing factuality, detail, and relevance.
    """

    def __init__(self, fact_checker, relevance_scorer):
        self.fact_checker = fact_checker
        self.relevance_scorer = relevance_scorer

    def compute_reward(
        self,
        question: str,
        response: str,
        ground_truth: str = None
    ) -> Dict:
        """
        Compute multi-dimensional reward.

        Args:
            question: Original question
            response: Model's response
            ground_truth: Optional ground truth for comparison

        Returns:
            Dict with component rewards and total
        """
        # Component 1: Factual Precision
        factuality_score = self._compute_factuality(response, ground_truth)

        # Component 2: Detail Level
        detail_score = self._compute_detail_level(response)

        # Component 3: Answer Relevance
        relevance_score = self._compute_relevance(question, response)

        # Combine with careful weighting
        # Higher weight on factuality to prevent hallucinations
        total_reward = (
            0.5 * factuality_score +
            0.25 * detail_score +
            0.25 * relevance_score
        )

        return {
            "factuality": factuality_score,
            "detail": detail_score,
            "relevance": relevance_score,
            "total": total_reward
        }

    def _compute_factuality(self, response: str, ground_truth: str = None) -> float:
        """
        Compute factual accuracy of response.

        Args:
            response: Model response
            ground_truth: Ground truth to check against

        Returns:
            Factuality score 0-1 (1 = all facts correct)
        """
        # Extract factual claims from response
        claims = self._extract_claims(response)

        if not claims:
            return 1.0  # No claims, can't be wrong

        # Check each claim for factuality
        correct_claims = 0

        for claim in claims:
            # Verify claim
            is_factual = self.fact_checker.verify(claim, ground_truth)

            if is_factual:
                correct_claims += 1

        factuality = correct_claims / len(claims)

        return factuality

    def _compute_detail_level(self, response: str) -> float:
        """
        Assess response detail level.

        Args:
            response: Model response

        Returns:
            Detail score 0-1 (1 = well-detailed)
        """
        # Multiple signals for detail
        signals = []

        # Signal 1: Length (but not too long)
        word_count = len(response.split())
        if word_count < 50:
            length_score = word_count / 50.0  # Too short
        elif word_count < 500:
            length_score = 1.0  # Optimal
        else:
            length_score = max(0.7, 1.0 - (word_count - 500) / 2000)  # Diminishing

        signals.append(length_score * 0.4)

        # Signal 2: Sentence complexity (more complex = more detailed)
        sentences = response.split(".")
        avg_sent_length = word_count / len(sentences) if sentences else 0

        if avg_sent_length < 10:
            complexity_score = 0.5  # Too simple
        elif avg_sent_length < 25:
            complexity_score = 1.0  # Good complexity
        else:
            complexity_score = 0.8  # Acceptable

        signals.append(complexity_score * 0.3)

        # Signal 3: Presence of examples/details
        detail_keywords = ["example", "such as", "specifically", "notably", "for instance"]
        keyword_count = sum(1 for kw in detail_keywords if kw in response.lower())

        detail_score = min(keyword_count / 3.0, 1.0)
        signals.append(detail_score * 0.3)

        detail = sum(signals)

        return detail

    def _compute_relevance(self, question: str, response: str) -> float:
        """
        Assess how relevant response is to question.

        Args:
            question: Original question
            response: Model response

        Returns:
            Relevance score 0-1
        """
        # Semantic similarity between question and response
        relevance = self.relevance_scorer.compute_relevance(question, response)

        return relevance

    def _extract_claims(self, text: str) -> list:
        """Extract factual claims from text."""
        # Simple implementation: sentences ending with facts
        sentences = text.split(".")

        claims = []
        for sent in sentences:
            if any(verb in sent.lower() for verb in ["is", "are", "was", "were"]):
                claims.append(sent.strip())

        return claims
```

### Step 2: Implement Fact-Checking System

Create system to verify factuality of claims.

```python
class FactChecker:
    """
    Verify factuality of model-generated claims.
    """

    def __init__(self, knowledge_base=None, external_api=None):
        self.knowledge_base = knowledge_base
        self.external_api = external_api

    def verify(self, claim: str, context: str = None) -> bool:
        """
        Verify whether claim is factual.

        Args:
            claim: Factual claim to verify
            context: Optional context

        Returns:
            True if claim is factually correct
        """
        # Try knowledge base first
        if self.knowledge_base:
            kb_result = self.knowledge_base.check_fact(claim)
            if kb_result is not None:
                return kb_result

        # Try external API (e.g., web search)
        if self.external_api:
            api_result = self.external_api.verify_fact(claim)
            if api_result is not None:
                return api_result

        # If no verification possible, assume correct
        return True

    def extract_and_verify(self, text: str) -> Dict[str, bool]:
        """
        Extract claims and verify each.

        Args:
            text: Text containing claims

        Returns:
            Dict mapping claims to factuality
        """
        claims = self._extract_claims(text)

        verification = {}
        for claim in claims:
            verification[claim] = self.verify(claim)

        return verification

    def _extract_claims(self, text: str) -> list:
        """Extract factual claims from text."""
        # Parse sentences that contain assertions
        import re

        # Pattern: Subject + Verb + Object assertions
        sentences = re.split(r'[.!?]+', text)

        claims = []
        for sent in sentences:
            sent = sent.strip()
            if self._is_factual_claim(sent):
                claims.append(sent)

        return claims

    def _is_factual_claim(self, sentence: str) -> bool:
        """Determine if sentence is a factual claim."""
        assertion_verbs = ["is", "are", "was", "were", "be", "been", "have", "has", "had"]

        return any(verb in sentence.lower() for verb in assertion_verbs)
```

### Step 3: Implement Online RL Training with Multi-dimensional Rewards

Train reasoning model using multi-dimensional reward function.

```python
class FactualityRLTrainer:
    """
    Train reasoning model for factuality using online RL.
    """

    def __init__(self, model, reward_fn):
        self.model = model
        self.reward_fn = reward_fn

    def train_step(self, question: str, ground_truth: str = None, use_old_response: str = None):
        """
        Single training step optimizing for factuality.

        Args:
            question: Question to answer
            ground_truth: Ground truth for reward
            use_old_response: Use old response for comparison

        Returns:
            Training metrics
        """
        # Generate response
        response = self.model.generate(question, max_length=500)

        # Compute multi-dimensional rewards
        rewards = self.reward_fn.compute_reward(question, response, ground_truth)

        # Compute log probabilities of generated response
        log_probs = self.model.get_logprobs_for_response(question, response)

        # REINFORCE update: scale gradients by rewards
        # Gradient = log_prob * reward
        policy_loss = -(log_probs * rewards["total"]).mean()

        # Backward pass
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.model.optimizer.step()
        self.model.optimizer.zero_grad()

        return {
            "loss": policy_loss.item(),
            "factuality_reward": rewards["factuality"],
            "detail_reward": rewards["detail"],
            "relevance_reward": rewards["relevance"],
            "total_reward": rewards["total"],
            "response": response
        }

    def train_epoch(self, questions: list, ground_truths: list, batch_size: int = 8):
        """
        Train for one epoch.

        Args:
            questions: List of questions
            ground_truths: List of ground truth answers
            batch_size: Batch size

        Returns:
            Average metrics for epoch
        """
        epoch_metrics = {
            "avg_loss": 0,
            "avg_factuality": 0,
            "avg_detail": 0,
            "avg_relevance": 0
        }

        num_batches = 0

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_truths = ground_truths[i:i + batch_size]

            batch_loss = 0
            batch_factuality = 0
            batch_detail = 0
            batch_relevance = 0

            for question, ground_truth in zip(batch_questions, batch_truths):
                metrics = self.train_step(question, ground_truth)

                batch_loss += metrics["loss"]
                batch_factuality += metrics["factuality_reward"]
                batch_detail += metrics["detail_reward"]
                batch_relevance += metrics["relevance_reward"]

            batch_size_actual = len(batch_questions)

            epoch_metrics["avg_loss"] += batch_loss / batch_size_actual
            epoch_metrics["avg_factuality"] += batch_factuality / batch_size_actual
            epoch_metrics["avg_detail"] += batch_detail / batch_size_actual
            epoch_metrics["avg_relevance"] += batch_relevance / batch_size_actual

            num_batches += 1

        # Average over all batches
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        return epoch_metrics
```

### Step 4: Evaluate Factuality Improvements

Create evaluation framework for factuality metrics.

```python
def evaluate_factuality(model, test_questions: list, ground_truths: list) -> Dict:
    """
    Evaluate model on factuality metrics.

    Args:
        model: Trained model
        test_questions: Test questions
        ground_truths: Ground truth answers

    Returns:
        Evaluation metrics
    """
    metrics = {
        "hallucination_rate": 0,
        "avg_detail": 0,
        "avg_relevance": 0,
        "overall_quality": 0
    }

    fact_checker = FactChecker()
    reward_fn = MultiDimensionalRewardFunction(fact_checker, None)

    for question, truth in zip(test_questions, ground_truths):
        # Generate response
        response = model.generate(question)

        # Check for hallucinations
        verification = fact_checker.extract_and_verify(response)

        hallucinations = sum(1 for v in verification.values() if not v)
        total_claims = len(verification) if verification else 1

        hallucination_rate = hallucinations / total_claims

        # Compute rewards
        rewards = reward_fn.compute_reward(question, response, truth)

        metrics["hallucination_rate"] += hallucination_rate
        metrics["avg_detail"] += rewards["detail"]
        metrics["avg_relevance"] += rewards["relevance"]

    # Average over test set
    n = len(test_questions)
    metrics["hallucination_rate"] /= n
    metrics["avg_detail"] /= n
    metrics["avg_relevance"] /= n
    metrics["overall_quality"] = (
        (1.0 - metrics["hallucination_rate"]) * 0.5 +
        metrics["avg_detail"] * 0.25 +
        metrics["avg_relevance"] * 0.25
    )

    return metrics
```

## Practical Guidance

### When to Use Learning to Reason for Factuality

- **Long-form factual generation**: Answering complex questions with multiple facts
- **Hallucination-prone models**: Reasoning models generating plausible but false content
- **Balanced optimization**: Need both accuracy and response quality
- **Online learning scenarios**: Continuously improving with feedback

### When NOT to Use Learning to Reason for Factuality

- **Closed-domain QA**: Knowledge base complete and accurate
- **Creative tasks**: Penalizing hallucinations inappropriate for fiction/poetry
- **Latency-sensitive**: Fact-checking adds computational overhead
- **Toxic content**: Fact-checking system itself may have biases

### Hyperparameter Recommendations

- **Factuality weight**: 0.5 (higher priority to prevent hallucinations)
- **Detail weight**: 0.25 (maintains responsiveness)
- **Relevance weight**: 0.25 (ensures answer to question)
- **Hallucination threshold**: Flag claims with <0.5 confidence
- **Learning rate**: 1e-5 (conservative for stability)

### Key Insights

The critical insight is that single-objective reward functions (like FActScore) cause reward hacking—models reduce output length to appear more factual. By combining factuality with detail and relevance, the framework forces genuine accuracy improvements rather than gaming the metric. The multi-dimensional approach learns nuanced trade-offs between competing objectives.

## Reference

**Learning to Reason for Factuality** (arXiv:2508.05618)

Proposes multi-dimensional reward function combining factual precision, response detail, and answer relevance. Reduces hallucinations by 23.1% while maintaining response quality through online RL on reasoning models.
