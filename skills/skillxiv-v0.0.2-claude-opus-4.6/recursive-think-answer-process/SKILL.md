---
name: recursive-think-answer-process
title: "Recursive Think-Answer Process for LLMs and VLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.02099"
keywords: [Inference-Time Optimization, Iterative Reasoning, Self-Refinement, Chain-of-Thought, VLM]
description: "Recursive Think-Answer Process enables models to iteratively refine reasoning and answers during inference, reducing self-correction errors and improving accuracy without retraining."
---

# Technique: Iterative Confidence-Guided Answer Refinement

Many reasoning tasks benefit from revising initial answers rather than committing to first attempts. However, unguided iteration often produces worse results—models make self-corrections that introduce errors (the "Oops!" phenomenon). Recursive Think-Answer Process (R-TAP) solves this by using confidence signals to guide refinement: if the model's confidence is low, refine; if high, commit to the answer.

The technique works for both language and vision-language models, improves accuracy on reasoning benchmarks, and requires no model retraining—it's a pure inference-time optimization.

## Core Concept

The core insight is that model confidence can guide when to stop iterating. Rather than refining a fixed number of times (which may introduce errors), use:

1. **Confidence Generator**: Estimate how certain the model is about its current answer
2. **Dual Reward System**: Two complementary feedback signals guiding refinement
3. **Iterative Cycles**: Repeat thinking and answering until confidence is high
4. **Termination**: Stop when confidence plateaus or reaches threshold

This creates a virtuous cycle: low-confidence answers trigger refinement, high-confidence answers terminate early, reducing unnecessary iterations that introduce errors.

## Architecture Overview

- **Confidence Estimator**: Predicts certainty about current response
- **Recursively Confidence Increase Reward**: Incentivizes refining to increase confidence
- **Final Answer Confidence Reward**: Incentivizes arriving at high-confidence final answer
- **Iterative Loop**: Alternate thinking and answering phases
- **Early Termination**: Stop when confidence exceeds threshold

## Implementation Steps

R-TAP is an inference-time technique requiring no model retraining. Here's how to implement it:

Define confidence estimation for different model types:

```python
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

class ConfidenceEstimator:
    """Estimates model confidence in its outputs."""

    def estimate_token_confidence(
        self,
        logits: torch.Tensor,
        top_k: int = 5,
    ) -> float:
        """
        Estimate confidence from token probability distribution.

        Args:
            logits: Model logits for next token [vocab_size]
            top_k: Consider top-k tokens for confidence

        Returns:
            confidence score in [0, 1]
        """
        probs = F.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probs, top_k)

        # Confidence = max probability (higher = more confident)
        max_prob = top_probs[0].item()

        # Entropy-based confidence = 1 - normalized_entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_entropy = torch.log(torch.tensor(len(logits), dtype=logits.dtype)).item()
        entropy_confidence = 1.0 - (entropy / max_entropy)

        # Combined confidence
        confidence = 0.6 * max_prob + 0.4 * entropy_confidence
        return min(max(confidence, 0.0), 1.0)

    def estimate_sequence_confidence(
        self,
        token_logits: list,  # List of logits for each token
    ) -> float:
        """
        Estimate confidence in entire sequence.

        Returns:
            average token confidence
        """
        token_confidences = []
        for logits in token_logits:
            conf = self.estimate_token_confidence(logits)
            token_confidences.append(conf)

        return sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
```

Implement the recursive think-answer loop:

```python
class RecursiveThinkAnswer:
    """
    Recursive Think-Answer Process for inference-time refinement.
    """

    def __init__(self, model, max_iterations: int = 3, confidence_threshold: float = 0.85):
        self.model = model
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.confidence_estimator = ConfidenceEstimator()

    def generate_thinking(
        self,
        context: str,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> Tuple[str, float]:
        """
        Generate intermediate thinking/reasoning.

        Returns:
            (thinking_text, thinking_confidence)
        """
        thinking_prompt = f"""
        Let me think about this step by step.
        Context: {context}

        My thinking:
        """

        output = self.model.generate(
            thinking_prompt,
            max_length=max_length,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        thinking_text = output.sequences
        token_logits = output.scores

        # Estimate confidence in thinking
        confidence = self.confidence_estimator.estimate_sequence_confidence(token_logits)

        return thinking_text, confidence

    def generate_answer(
        self,
        context: str,
        thinking: str = None,
        max_length: int = 100,
        temperature: float = 0.7,
    ) -> Tuple[str, float]:
        """
        Generate answer, optionally conditioned on thinking.

        Returns:
            (answer_text, answer_confidence)
        """
        if thinking:
            answer_prompt = f"""
            Context: {context}

            Thinking: {thinking}

            My answer is:
            """
        else:
            answer_prompt = f"""
            Context: {context}

            Answer:
            """

        output = self.model.generate(
            answer_prompt,
            max_length=max_length,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True,
        )

        answer_text = output.sequences
        token_logits = output.scores

        # Estimate confidence in answer
        confidence = self.confidence_estimator.estimate_sequence_confidence(token_logits)

        return answer_text, confidence

    def compute_refinement_reward(
        self,
        previous_confidence: float,
        current_confidence: float,
        iteration: int,
    ) -> float:
        """
        Recursively Confidence Increase Reward (RCIR).
        Encourages refinement if confidence is increasing.
        """
        confidence_gain = current_confidence - previous_confidence

        # Reward refinement if confidence increases
        rcir = max(confidence_gain, 0.0)

        # Penalize excessive iteration
        iteration_penalty = 0.1 * (iteration - 1)

        return rcir - iteration_penalty

    def compute_final_answer_reward(
        self,
        final_confidence: float,
    ) -> float:
        """
        Final Answer Confidence Reward (FACR).
        Encourages high-confidence final answers.
        """
        facr = max(final_confidence - self.confidence_threshold, 0.0)
        return facr

    def process_recursively(
        self,
        context: str,
        verbose: bool = False,
    ) -> Dict[str, any]:
        """
        Run recursive think-answer loop with confidence guidance.

        Returns:
            {
                'final_answer': str,
                'final_confidence': float,
                'iterations': int,
                'reasoning_trace': List[Dict],
            }
        """
        reasoning_trace = []
        previous_confidence = 0.0
        current_answer = None
        current_confidence = 0.0

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Thinking phase
            thinking, thinking_conf = self.generate_thinking(context)
            if verbose:
                print(f"Thinking (conf={thinking_conf:.2f}): {thinking[:100]}...")

            # Answer phase
            answer, answer_conf = self.generate_answer(context, thinking=thinking)
            if verbose:
                print(f"Answer (conf={answer_conf:.2f}): {answer[:100]}...")

            # Compute rewards
            refinement_reward = self.compute_refinement_reward(
                previous_confidence, answer_conf, iteration
            )

            reasoning_trace.append({
                'iteration': iteration,
                'thinking': thinking,
                'thinking_confidence': thinking_conf,
                'answer': answer,
                'answer_confidence': answer_conf,
                'refinement_reward': refinement_reward,
            })

            current_answer = answer
            previous_confidence = answer_conf
            current_confidence = answer_conf

            # Early termination if confident
            if answer_conf >= self.confidence_threshold:
                if verbose:
                    print(f"Stopping: confidence {answer_conf:.2f} >= threshold {self.confidence_threshold:.2f}")
                break

        # Final answer reward
        final_reward = self.compute_final_answer_reward(current_confidence)

        return {
            'final_answer': current_answer,
            'final_confidence': current_confidence,
            'final_reward': final_reward,
            'iterations': len(reasoning_trace),
            'reasoning_trace': reasoning_trace,
        }
```

## Practical Guidance

**When to Use:**
- Reasoning tasks where intermediate steps help (math, logic, complex QA)
- When model's first attempt has moderate confidence (not already high)
- Inference-time optimization (no retraining needed)
- Vision-language models on complex visual reasoning

**When NOT to Use:**
- Simple classification/tagging (single-step reasoning)
- Real-time systems with strict latency limits (recursion adds time)
- When first answer is already very confident (wastes computation)

**Hyperparameters:**
- `max_iterations`: 2–5 (more iterations = slower but potentially higher quality)
- `confidence_threshold`: 0.8–0.95 (lower threshold = fewer iterations)
- `temperature`: 0.7–0.8 for thinking, 0.5–0.7 for final answers

**Confidence Estimation:**
- Token probability: High prob on next token = confident
- Entropy: Low entropy = confident (less uniform distribution)
- Sequence-level: Average or minimum token confidence

**Optimization:**
- Cache computations between iterations (same context)
- Early termination saves compute when already confident
- Can combine with other inference tricks (beam search, temperature scaling)

**Results:**
- Produces more stable, faster inference-time reasoning
- Reduces self-reflective error patterns (fewer spurious corrections)
- Works across LLMs and VLMs without modification
- Particularly effective on complex reasoning benchmarks

---

**Reference:** [Recursive Think-Answer Process for LLMs and VLMs](https://arxiv.org/abs/2603.02099)
