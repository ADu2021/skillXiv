---
name: reasoning-as-compression
title: "Reasoning as Compression: Unifying Budget Forcing via the Conditional Information Bottleneck"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.08462"
keywords: [Information Theory, Chain-of-Thought, Budget Optimization, Efficient Reasoning]
description: "Optimize chain-of-thought reasoning under computational budgets using information-theoretic compression principles, improving reasoning efficiency without accuracy loss."
---

# Reasoning as Compression: Information-Theoretic Budget Optimization

Large language models generate lengthy chain-of-thought (CoT) reasoning that improves accuracy but increases inference latency and cost. Standard approaches use heuristic length penalties: if reasoning is too long, penalize it. But this ignores the semantic content—some reasoning steps are valuable, others are redundant.

This framework reframes efficient reasoning as a compression problem. Rather than counting tokens, we use information theory: measure the "cost" of reasoning by its semantic complexity (surprisal under a language model prior). The goal is to maximize task reward while compressing the reasoning trace below a budget. This yields both heuristic penalties and more sophisticated information-theoretic solutions automatically.

## Core Concept

The framework applies the Conditional Information Bottleneck (CIB) principle to reasoning:

**Information Bottleneck:** Given input (prompt) and reasoning trace, the trace should compress information while still predicting the output accurately.

**Semantic Cost:** Cost of reasoning is measured by surprisal (negative log probability) under a language model, not token count. This allows intelligent pruning: rare/surprising tokens are costly, redundant tokens are cheap.

**Optimization Target:** Maximize task reward while keeping information cost below a budget, trading off reasoning quality and latency.

The result unifies several existing methods (length penalties, token budgets, etc.) as special cases of the general CIB optimization.

## Architecture Overview

- **Language Model Prior**: Reference model that estimates surprisal (semantic cost)
- **Reward Model**: Task-specific model assessing reasoning quality
- **Information Bottleneck Objective**: Balances reward and compression
- **Semantic Cost Estimation**: Computes cost via language model probability
- **RL Formulation**: Optimize via policy gradient with information constraints
- **Approximate CIB Solver**: Tractable algorithm for constrained optimization

## Implementation Steps

### Step 1: Define Information-Theoretic Cost

Measure reasoning complexity using surprisal.

```python
import torch
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

class InformationCostComputer:
    """
    Compute information-theoretic cost of reasoning traces.
    Uses language model prior to estimate semantic complexity.
    """

    def __init__(self, reference_lm, temperature: float = 1.0):
        self.reference_lm = reference_lm
        self.temperature = temperature

    def compute_surprisal(self, text: str) -> float:
        """
        Compute surprisal (negative log probability) of text under LM prior.
        Surprisal measures how unexpected/informative text is.
        Higher surprisal = more information = higher cost.
        """

        tokens = self.reference_lm.tokenize(text)
        log_probs = []

        with torch.no_grad():
            for i in range(1, len(tokens)):
                # Get probability of token i given tokens 0..i-1
                context = tokens[:i]
                next_token = tokens[i]

                logits = self.reference_lm(context)
                log_prob = F.log_softmax(logits, dim=-1)[next_token]
                log_probs.append(-log_prob.item())  # Negative = surprisal

        # Average surprisal per token
        if log_probs:
            return np.mean(log_probs)
        return 0.0

    def compute_segment_cost(self, reasoning_trace: str, segmentation: List[Tuple[int, int]]) -> List[float]:
        """
        Compute cost of each segment in reasoning.
        Segments: list of (start, end) character positions.
        """
        segment_costs = []

        for start, end in segmentation:
            segment_text = reasoning_trace[start:end]
            cost = self.compute_surprisal(segment_text)
            segment_costs.append(cost)

        return segment_costs

    def segment_by_sentences(self, text: str) -> List[Tuple[int, int]]:
        """Split reasoning into sentence-level segments."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        segmentation = []
        pos = 0
        for sentence in sentences:
            start = pos
            end = pos + len(sentence)
            segmentation.append((start, end))
            pos = end + 1  # +1 for space

        return segmentation
```

### Step 2: Formulate the Conditional Information Bottleneck

Define the optimization objective.

```python
class ConditionalInformationBottleneck:
    """
    CIB principle applied to reasoning:
    The reasoning trace should be maximally informative about the answer
    while being minimally informative beyond what the answer requires.
    """

    def __init__(self, reference_lm, reward_model, information_budget: float = 10.0):
        self.reference_lm = reference_lm
        self.reward_model = reward_model
        self.information_budget = information_budget
        self.cost_computer = InformationCostComputer(reference_lm)

    def compute_task_reward(self, reasoning: str, answer: str, ground_truth: str) -> float:
        """
        Compute reward for this reasoning-answer pair.
        Typically: is the answer correct?
        """

        if answer.strip().lower() == ground_truth.strip().lower():
            return 1.0
        else:
            # Partial credit based on semantic similarity
            return self.reward_model.compute_similarity(answer, ground_truth)

    def compute_information_cost(self, prompt: str, reasoning: str, answer: str) -> float:
        """
        Compute information-theoretic cost of reasoning.

        Cost = surprisal of reasoning under reference LM
        This measures: how much new information does this reasoning contribute?
        """

        # Surprisal of the full reasoning
        full_trace_cost = self.cost_computer.compute_surprisal(reasoning)

        # Surprisal of just the prompt (baseline)
        # If LM expects reasoning given prompt, cost is low
        # If LM is surprised by reasoning, cost is high

        # Conditional cost: information in reasoning beyond what prompt implies
        prompt_surprise = self.cost_computer.compute_surprisal(prompt)
        conditional_cost = full_trace_cost - prompt_surprise

        return max(0.0, conditional_cost)  # Cost is non-negative

    def compute_cib_loss(self, prompt: str, reasoning: str, answer: str,
                        ground_truth: str, beta: float = 0.1) -> float:
        """
        Conditional Information Bottleneck loss:

        L = -reward + beta * information_cost

        This trades off:
        - Reward: is the answer correct?
        - Information cost: how much information in reasoning?

        beta controls the tradeoff (higher = prefer compressed reasoning)
        """

        reward = self.compute_task_reward(reasoning, answer, ground_truth)
        info_cost = self.compute_information_cost(prompt, reasoning, answer)

        loss = -reward + beta * info_cost

        return loss

    def check_information_budget(self, reasoning: str) -> bool:
        """Verify reasoning stays within information budget."""
        cost = self.cost_computer.compute_surprisal(reasoning)
        return cost <= self.information_budget
```

### Step 3: Optimize Reasoning via Policy Gradient

Use reinforcement learning to find efficient reasoning.

```python
import torch
import torch.optim as optim
from torch.distributions import Categorical

class ReasoningOptimizer:
    """
    Optimize reasoning under information budget via RL.
    """

    def __init__(self, reasoning_generator, cib_objective):
        self.reasoning_generator = reasoning_generator
        self.cib_objective = cib_objective

    def optimize_reasoning_trajectory(self, prompt: str, ground_truth: str,
                                      num_steps: int = 100,
                                      beta: float = 0.1) -> Tuple[str, float]:
        """
        Generate reasoning that balances quality and efficiency.

        Strategy: Progressive shortening
        - Start with full verbose reasoning
        - Iteratively remove/compress low-value steps
        - Stop when budget exceeded
        """

        # Generate initial reasoning (verbose)
        initial_reasoning = self.reasoning_generator.generate_verbose(prompt)
        initial_answer = self.reasoning_generator.extract_answer(prompt + initial_reasoning)

        # Segment reasoning into steps
        segmentation = self.cib_objective.cost_computer.segment_by_sentences(initial_reasoning)
        segment_costs = self.cib_objective.cost_computer.compute_segment_cost(
            initial_reasoning, segmentation
        )

        # Iteratively remove low-value segments
        current_reasoning = initial_reasoning
        current_answer = initial_answer

        for iteration in range(num_steps):
            # Evaluate current reasoning
            current_cost = self.cib_objective.compute_information_cost(
                prompt, current_reasoning, current_answer
            )
            current_reward = self.cib_objective.compute_task_reward(
                current_reasoning, current_answer, ground_truth
            )

            # Check budget
            if current_cost <= self.cib_objective.information_budget:
                break  # Within budget, done

            # Remove lowest-value segment
            if segmentation:
                # Find segment with minimum contribution to understanding
                value_scores = self._score_segment_importance(
                    prompt, current_reasoning, current_answer, segmentation
                )

                min_value_idx = np.argmin(value_scores)
                start, end = segmentation[min_value_idx]

                # Remove segment
                new_reasoning = current_reasoning[:start] + current_reasoning[end:]
                new_answer = self.reasoning_generator.extract_answer(prompt + new_reasoning)

                # Only keep removal if it doesn't hurt reward too much
                new_reward = self.cib_objective.compute_task_reward(
                    new_reasoning, new_answer, ground_truth
                )

                if new_reward >= 0.9 * current_reward:  # Allow 10% degradation
                    current_reasoning = new_reasoning
                    current_answer = new_answer
                    segmentation.pop(min_value_idx)
                else:
                    # Can't remove this segment, try next
                    segmentation.pop(min_value_idx)

        final_loss = self.cib_objective.compute_cib_loss(
            prompt, current_reasoning, current_answer, ground_truth, beta=beta
        )

        return current_reasoning, final_loss

    def _score_segment_importance(self, prompt: str, reasoning: str, answer: str,
                                  segmentation: List[Tuple[int, int]]) -> np.ndarray:
        """Score importance of each reasoning segment."""
        scores = []

        for start, end in segmentation:
            # Remove this segment and measure impact
            modified_reasoning = reasoning[:start] + reasoning[end:]
            modified_answer = self.reasoning_generator.extract_answer(prompt + modified_reasoning)

            # Importance = impact on reward if removed
            original_reward = self.cib_objective.compute_task_reward(reasoning, answer, answer)
            modified_reward = self.cib_objective.compute_task_reward(
                modified_reasoning, modified_answer, answer
            )

            importance = original_reward - modified_reward
            scores.append(importance)

        return np.array(scores)
```

### Step 4: Semantic Prior for Cost Estimation

Implement efficient cost computation.

```python
class SemanticPrior:
    """
    Faster approximation of information costs using language model probabilities.
    """

    def __init__(self, lm_model):
        self.lm = lm_model
        self.cache = {}

    def estimate_cost(self, text: str) -> float:
        """Fast cost estimation via cached computations."""

        if text in self.cache:
            return self.cache[text]

        # Tokenize
        tokens = self.lm.tokenize(text)
        if len(tokens) == 0:
            return 0.0

        # Get per-token probabilities
        with torch.no_grad():
            logits = self.lm.get_logits(text)
            log_probs = F.log_softmax(logits, dim=-1)

        # Compute perplexity as proxy for cost
        # (perplexity ~= average surprisal)
        avg_log_prob = log_probs.mean().item()
        cost = -avg_log_prob / np.log(2)  # Convert to bits

        self.cache[text] = cost
        return cost

    def semantic_prior_objective(self, prompt: str, reasoning_segments: List[str],
                                 answer: str) -> float:
        """
        Objective that prefers reasoning that is likely under semantic prior.
        High-likelihood reasoning (given prompt) has lower cost.
        """

        total_cost = 0.0
        for segment in reasoning_segments:
            # Cost reduced if segment is likely given prompt
            segment_cost = self.estimate_cost(prompt + segment)
            prior_cost = self.estimate_cost(prompt)  # Baseline

            # Relative cost
            relative_cost = max(0, segment_cost - prior_cost)
            total_cost += relative_cost

        return total_cost
```

### Step 5: End-to-End Reasoning System

Integrate all components into a practical system.

```python
class EfficientReasoningSystem:
    """
    Complete system for budget-constrained reasoning optimization.
    """

    def __init__(self, base_lm, reward_model, information_budget: float = 10.0):
        self.base_lm = base_lm
        self.cib_objective = ConditionalInformationBottleneck(
            base_lm, reward_model, information_budget
        )
        self.optimizer = ReasoningOptimizer(base_lm, self.cib_objective)
        self.semantic_prior = SemanticPrior(base_lm)

    def generate_efficient_reasoning(self, prompt: str, ground_truth: str,
                                    budget: float = None) -> Dict[str, any]:
        """
        Generate reasoning optimized for both quality and efficiency.
        """

        if budget:
            self.cib_objective.information_budget = budget

        # Optimize reasoning under budget
        optimized_reasoning, loss = self.optimizer.optimize_reasoning_trajectory(
            prompt, ground_truth, num_steps=50, beta=0.1
        )

        # Extract answer
        answer = self.base_lm.generate_answer(prompt + optimized_reasoning)

        # Compute metrics
        reward = self.cib_objective.compute_task_reward(
            optimized_reasoning, answer, ground_truth
        )
        info_cost = self.cib_objective.compute_information_cost(
            prompt, optimized_reasoning, answer
        )
        budget_ratio = info_cost / self.cib_objective.information_budget

        return {
            'reasoning': optimized_reasoning,
            'answer': answer,
            'reward': reward,
            'information_cost': info_cost,
            'budget_utilization': budget_ratio,
            'loss': loss
        }

    def compare_reasoning_efficiency(self, prompt: str, ground_truth: str) -> Dict:
        """
        Compare different reasoning strategies.
        """

        # Verbose reasoning (no compression)
        verbose_result = self.generate_efficient_reasoning(
            prompt, ground_truth, budget=float('inf')
        )

        # Compressed reasoning (tight budget)
        compressed_result = self.generate_efficient_reasoning(
            prompt, ground_truth, budget=5.0
        )

        return {
            'verbose': verbose_result,
            'compressed': compressed_result,
            'accuracy_trade': verbose_result['reward'] - compressed_result['reward'],
            'cost_reduction': verbose_result['information_cost'] - compressed_result['information_cost']
        }
```

## Practical Guidance

**Hyperparameters:**
- Information budget: 5-20 bits (task-dependent; higher = more reasoning)
- Beta (tradeoff weight): 0.01-1.0 (higher = prefer compression)
- Segmentation granularity: sentence-level or step-level
- Reference LM temperature: 1.0 (standard; controls cost smoothness)

**When to Use:**
- Tasks where reasoning is helpful but costs matter (API pricing, latency)
- Scenarios requiring transparency into cost-quality tradeoffs
- Domains where reasoning steps vary dramatically in usefulness
- Settings where you can measure ground truth (supervised learning)

**When NOT to Use:**
- Tasks requiring full verbose reasoning (e.g., math proofs need all steps)
- Online learning where reference LM isn't available
- Streaming inference (buffering reasoning adds latency)
- Domains where reasoning structure is unpredictable

**Pitfalls:**
- Reference LM quality critical: poor prior leads to meaningless costs
- Information budget setting requires tuning: no universal default
- Semantic cost can be noisy: use ensemble or smoothing
- Answer extraction fragile: must correctly parse reasoning output

## Reference

Paper: [arxiv.org/abs/2603.08462](https://arxiv.org/abs/2603.08462)
