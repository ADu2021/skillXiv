---
name: plan-budget-reasoning
title: "Plan and Budget: Effective and Efficient Test-Time Scaling on Large Language Model Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.16122"
keywords: [test-time compute, reasoning, token allocation, efficiency, LLMs]
description: "Decompose complex queries into sub-questions and allocate computational budgets adaptively based on estimated difficulty, achieving 70% accuracy improvements and 39% token reduction without retraining."
---

# Plan and Budget: Effective and Efficient Test-Time Scaling on Language Model Reasoning

## Core Concept

Plan and Budget addresses the "overthinking" problem in language model reasoning: models generate excessively verbose outputs using computational resources inefficiently. The framework decomposes complex queries into sub-questions with varying uncertainty levels, then allocates tokens adaptively to each sub-problem based on estimated difficulty.

Rather than applying uniform computation across all reasoning steps, the approach identifies which sub-questions require more reasoning effort and concentrates tokens there, while keeping straightforward sub-questions brief. This test-time strategy is model-agnostic and requires no retraining, achieving both higher accuracy and reduced token consumption.

## Architecture Overview

- **Query Decomposition**: Break complex questions into simpler, solvable sub-questions
- **Difficulty Estimation**: Predict which sub-questions require more computational effort
- **Budget Allocation Model (BAM)**: Distribute token budget across sub-questions based on estimated difficulty
- **Efficiency-Effectiveness Tradeoff (E3 Metric)**: Measure combined improvement in accuracy and token efficiency
- **Dynamic Scheduling**: Adapt token allocation in real-time during generation
- **Model-Agnostic Application**: Works with any reasoner without architectural changes

## Implementation

The following steps outline how to implement adaptive budget allocation for reasoning:

1. **Decompose the query** - Break complex questions into sub-question components
2. **Estimate sub-question difficulty** - Assess which parts require more reasoning
3. **Allocate token budget** - Distribute available tokens proportionally to difficulty
4. **Generate reasoning** - Process each sub-question with allocated token budget
5. **Synthesize answer** - Combine sub-question responses into final answer
6. **Measure efficiency** - Track accuracy and token usage for optimization

```python
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch

@dataclass
class SubQuestion:
    text: str
    estimated_difficulty: float
    allocated_tokens: int
    response: str = ""

class QueryDecomposer:
    def __init__(self, decomposition_model):
        self.model = decomposition_model

    def decompose(self, query: str) -> List[str]:
        """Break query into sub-questions."""
        prompt = f"""Break down this complex question into simpler sub-questions:

Question: {query}

Sub-questions (numbered list):"""

        response = self.model.generate(prompt, max_tokens=500)
        sub_questions = self._parse_sub_questions(response)
        return sub_questions

    def _parse_sub_questions(self, response: str) -> List[str]:
        """Parse sub-questions from model response."""
        lines = response.split('\n')
        sub_questions = []
        for line in lines:
            line = line.strip()
            if line and any(c.isdigit() for c in line[:3]):
                # Remove numbering
                question = line.lstrip('0123456789.)-').strip()
                if question:
                    sub_questions.append(question)
        return sub_questions


class DifficultyEstimator:
    def __init__(self, estimation_model):
        self.model = estimation_model

    def estimate(self, question: str) -> float:
        """Estimate difficulty of a question (0.0 to 1.0)."""
        prompt = f"""Rate the difficulty of answering this question on a scale 0-1:
- 0 = trivial, direct factual answer
- 0.5 = moderate, requires reasoning
- 1 = very difficult, complex multi-step reasoning

Question: {question}

Difficulty (single number 0-1):"""

        response = self.model.generate(prompt, max_tokens=10)
        try:
            difficulty = float(response.strip())
            return max(0.0, min(1.0, difficulty))
        except:
            return 0.5  # default to moderate


class BudgetAllocator:
    def __init__(self, base_budget: int = 2000):
        self.base_budget = base_budget

    def allocate(self, sub_questions: List[SubQuestion]) -> List[SubQuestion]:
        """Allocate tokens to sub-questions based on difficulty."""
        if not sub_questions:
            return []

        # Minimum tokens per sub-question
        min_tokens = 100
        total_difficulty = sum(sq.estimated_difficulty for sq in sub_questions)

        # Avoid division by zero
        if total_difficulty == 0:
            total_difficulty = len(sub_questions)

        # Allocate proportionally to difficulty
        for sq in sub_questions:
            proportion = sq.estimated_difficulty / total_difficulty
            allocated = int(self.base_budget * proportion)
            sq.allocated_tokens = max(min_tokens, allocated)

        return sub_questions


class E3Metric:
    @staticmethod
    def compute(baseline_accuracy: float, optimized_accuracy: float,
                baseline_tokens: int, optimized_tokens: int) -> float:
        """Compute Efficiency-Effectiveness metric."""
        accuracy_improvement = (optimized_accuracy - baseline_accuracy) / baseline_accuracy
        token_reduction = 1 - (optimized_tokens / baseline_tokens)

        # E3 balances both improvements
        e3 = accuracy_improvement + token_reduction
        return e3


class PlanAndBudgetReasoner:
    def __init__(self, decomposer: QueryDecomposer, estimator: DifficultyEstimator,
                 allocator: BudgetAllocator, reasoner_model):
        self.decomposer = decomposer
        self.estimator = estimator
        self.allocator = allocator
        self.reasoner = reasoner_model

    def reason(self, query: str, total_budget: int = 2000) -> Dict:
        """Perform planning and budget-aware reasoning."""
        # Step 1: Decompose query
        sub_question_texts = self.decomposer.decompose(query)
        sub_questions = [SubQuestion(text=sq, estimated_difficulty=0.5, allocated_tokens=0)
                        for sq in sub_question_texts]

        # Step 2: Estimate difficulty
        for sq in sub_questions:
            sq.estimated_difficulty = self.estimator.estimate(sq.text)

        # Step 3: Allocate budget
        allocator = BudgetAllocator(total_budget)
        sub_questions = allocator.allocate(sub_questions)

        # Step 4: Generate responses with allocated budgets
        total_tokens_used = 0
        for sq in sub_questions:
            response = self.reasoner.generate(sq.text, max_tokens=sq.allocated_tokens)
            sq.response = response
            total_tokens_used += len(response.split())

        # Step 5: Synthesize final answer
        final_answer = self._synthesize(query, sub_questions)

        return {
            "query": query,
            "sub_questions": sub_questions,
            "final_answer": final_answer,
            "total_tokens_used": total_tokens_used,
            "budget_efficiency": total_tokens_used / total_budget
        }

    def _synthesize(self, original_query: str, sub_questions: List[SubQuestion]) -> str:
        """Combine sub-question responses into final answer."""
        context = "\n".join([f"Q: {sq.text}\nA: {sq.response}" for sq in sub_questions])
        prompt = f"""Based on these reasoning steps:

{context}

Answer the original question: {original_query}

Final answer:"""

        final_answer = self.reasoner.generate(prompt, max_tokens=300)
        return final_answer
```

## Practical Guidance

**Budget allocation strategies:**
- **Linear allocation**: Allocate tokens linearly proportional to difficulty (simplest)
- **Exponential allocation**: Allocate exponentially more tokens to harder questions (steeper emphasis)
- **Conservative allocation**: Reserve tokens for final synthesis; allocate remainder to sub-questions

**Difficulty estimation approaches:**
- **Model-based**: Use the same LLM to estimate difficulty (quick, consistent)
- **Heuristic-based**: Use question length, keyword complexity, or linguistic features
- **Learned**: Train a separate difficulty classifier on past reasoning traces
- **Ensemble**: Combine multiple difficulty estimates for robustness

**When to use:**
- Complex reasoning tasks requiring multiple reasoning steps
- Budget-constrained inference scenarios (latency, cost limits)
- Multi-step math or logic problems with variable sub-problem difficulty
- Long-form generation where some outputs can be terse without loss of quality

**When NOT to use:**
- Simple, single-step questions where decomposition adds overhead
- Tasks where all steps are equally important (no selectivity benefit)
- Real-time systems where decomposition latency is prohibitive
- Domains requiring uniform reasoning depth (e.g., legal analysis)

**Common pitfalls:**
- **Poor decomposition**: Sub-questions don't capture actual problem structure; validate decompositions
- **Difficulty miscalibration**: Estimated difficulty doesn't match actual reasoning complexity
- **Budget exhaustion**: Token allocation causes some sub-questions to run out of budget mid-reasoning
- **Synthesis loss**: Final answer loses nuance when combining abbreviated sub-question responses

## Reference

Plan and Budget achieves up to 70% accuracy improvements and 39% token reduction without retraining. The framework enables smaller models (32B parameters) to match larger models' efficiency. Code is publicly available and the work was accepted to ICLR 2026.

Original paper: "Plan and Budget: Effective and Efficient Test-Time Scaling on Large Language Model Reasoning" (arxiv.org/abs/2505.16122)
