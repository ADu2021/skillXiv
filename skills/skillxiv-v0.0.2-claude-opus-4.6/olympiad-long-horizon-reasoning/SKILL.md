---
name: olympiad-long-horizon-reasoning
title: "Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.10739
keywords: [mathematical reasoning, multi-agent systems, reinforcement learning, olympiad problems, hierarchical decomposition]
description: "Solve IMO-level problems through multi-stage hierarchical reasoning with lemma-based memory compression. Intern-S1-MO achieves 26/35 on IMO2025 geometry-excluded problems—ideal when complex reasoning exceeds single-pass context."
---

## Overview

The system overcomes context-length limitations through multi-round hierarchical reasoning where intermediate insights are stored as lemmas. A verification agent validates solutions while RL continuously improves the underlying reasoning model.

## When to Use

- Complex mathematical problem solving
- Olympiad and competition-level reasoning
- Problems requiring multiple reasoning stages
- Long-horizon task decomposition
- Need for solution verification

## When NOT to Use

- Simple arithmetic problems
- Single-step reasoning tasks
- Scenarios with abundant context window

## Core Technique

Hierarchical reasoning with lemma-based compression:

```python
# Multi-agent Olympiad reasoning system
class OlympiadReasoner:
    def __init__(self):
        self.reasoner = ReasoningModel()
        self.verifier = VerificationModel()
        self.lemma_store = LemmaMemory()

    def solve_problem_hierarchically(self, problem, max_rounds=10):
        """Multi-round reasoning with lemma compression."""
        context = problem
        solution_steps = []

        for round_idx in range(max_rounds):
            # Stage 1: Reasoning on current context
            reasoning = self.reasoner.generate_reasoning(context)

            # Check if solution found
            if self.is_complete_solution(reasoning):
                return reasoning

            # Stage 2: Extract lemmas (compress intermediate insights)
            lemmas = self.extract_lemmas(reasoning)
            self.lemma_store.add_lemmas(lemmas)

            # Stage 3: Update context for next round
            # Forget low-value details, keep lemmas
            context = self.compress_context(
                context,
                lemmas,
                max_tokens=2000
            )

            solution_steps.append(reasoning)

        return '\n'.join(solution_steps)

    def extract_lemmas(self, reasoning_text):
        """Extract key insights as reusable lemmas."""
        # Parse reasoning to find important facts
        lemmas = []

        # Example lemmas:
        # "Angle BAC = 45 degrees"
        # "Triangle ABC is isosceles"
        # "Point D lies on line EF"

        lemma_candidates = self.find_mathematical_facts(reasoning_text)

        for candidate in lemma_candidates:
            # Score lemma importance
            importance = self.score_lemma_importance(
                candidate,
                reasoning_text
            )

            if importance > 0.7:
                lemmas.append({
                    'statement': candidate,
                    'importance': importance,
                    'source_step': len(self.lemma_store)
                })

        return lemmas

    def verify_solution(self, solution):
        """Verification agent checks solution validity."""
        verification_report = self.verifier.verify(solution)

        if verification_report['is_valid']:
            return True

        # Extract error explanation
        errors = verification_report['errors']
        return errors
```

## Key Results

- 26/35 on IMO2025 geometry-excluded (silver level)
- 102/126 on CMO2025 (gold level)
- Multi-round hierarchical decomposition effective
- Lemma-based memory enables long-horizon solving

## References

- Original paper: https://arxiv.org/abs/2512.10739
- Focus: Long-horizon mathematical reasoning
- Domain: Problem solving, hierarchical reasoning
