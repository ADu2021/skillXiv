---
name: arise-skill-evolution-hierarchical-rl
title: "ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical RL"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.16060"
keywords: [Skill Learning, Hierarchical Reinforcement Learning, Strategy Reuse, Mathematical Reasoning, Emergent Abilities]
description: "Build reusable skill libraries for mathematical reasoning through hierarchical RL. Maintain a high-level skills manager that summarizes successful solution traces and selects relevant strategies to condition future rollouts."
---

# ARISE: Hierarchical RL with Intrinsic Skill Evolution

Current language model reasoning treats each problem independently, recomputing similar strategies repeatedly. ARISE enables agents to accumulate and reuse successful reasoning strategies through hierarchical reinforcement learning. A high-level Skills Manager maintains a library of reusable strategies by summarizing successful solution traces and selecting relevant skills to condition future reasoning. The worker (reasoning model) generates solutions informed by retrieved skills, creating a feedback loop where reasoning and skill library quality co-evolve. This approach shows consistent improvements over baselines on mathematical and competition benchmarks, with particularly strong gains on out-of-distribution problems.

The key insight: emergent reusable patterns from problem-solving can be preserved and leveraged to bootstrap future reasoning more efficiently.

## Core Concept

ARISE operates through a hierarchical loop:

1. **Problem Solving** — Worker generates solution trajectory
2. **Trace Summarization** — Skills Manager extracts generalizable patterns from successful traces
3. **Skill Library Update** — Add summarized skills to reusable library
4. **Skill Retrieval** — For new problems, retrieve relevant skills
5. **Conditioned Reasoning** — Worker uses retrieved skills to guide future rollouts
6. **Co-Evolution** — Both worker and skill quality improve iteratively

This creates a virtuous cycle where accumulated problem-solving experience directly improves future reasoning.

## Architecture Overview

- **Problem Solver (Worker)** — Generates step-by-step solutions using chain-of-thought
- **Solution Trace Logger** — Records all intermediate steps and decisions
- **Skill Summarizer** — Extracts generalizable solution patterns from traces
- **Skill Library** — Stores skill descriptions and embeddings for retrieval
- **Skill Retriever** — Semantic search to find relevant skills for new problems
- **Hierarchical Reward Structure** — Separate rewards for trace quality and skill utility
- **Co-Evolution Optimizer** — Joint training of worker and skills manager

## Implementation Steps

Start by defining the skill representation and summarization mechanism.

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class SolutionTrace:
    """Record of solution steps for a single problem."""
    problem: str
    steps: List[str]  # Each step in the reasoning chain
    final_answer: str
    correctness: bool
    num_steps: int


@dataclass
class Skill:
    """Reusable reasoning strategy."""
    description: str  # Natural language description
    problem_types: List[str]  # Categories this skill applies to
    embedding: np.ndarray  # For semantic search
    success_rate: float  # How often this skill helps
    usage_count: int


class SkillManager:
    """Maintain library of reusable solution strategies."""

    def __init__(self, max_skills=1000, embedding_dim=768):
        self.skills = []
        self.max_skills = max_skills
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(max_features=100)

    def summarize_trace(self, trace: SolutionTrace) -> Optional[str]:
        """Extract generalizable skill from successful solution trace."""
        if not trace.correctness:
            return None  # Only learn from successful traces

        # Identify key steps that were critical
        key_steps = self._identify_key_steps(trace.steps)

        # Generate natural language summary
        summary = f"""Skill for {trace.problem_types}:
Problem pattern: {trace.problem[:100]}
Key approach: {' -> '.join(key_steps[:3])}
Reasoning: {self._generate_explanation(trace.steps)}"""

        return summary

    def _identify_key_steps(self, steps: List[str], num_key=3) -> List[str]:
        """Identify most important steps using heuristics."""
        # Steps that define transformations or insights
        key_steps = []

        for step in steps:
            if any(keyword in step.lower()
                  for keyword in ['define', 'assume', 'derive', 'compute',
                                 'observe', 'conclude']):
                key_steps.append(step[:80])  # Truncate

        return key_steps[:num_key] if key_steps else steps[:num_key]

    def _generate_explanation(self, steps: List[str]) -> str:
        """Create concise explanation of reasoning strategy."""
        # Simplified explanation from first/last steps
        if len(steps) > 1:
            return f"Start: {steps[0][:50]}... Then: {steps[-1][:50]}..."
        return steps[0][:100] if steps else ""

    def add_skill(self, skill_text: str, problem_types: List[str] = None):
        """Add new skill to library."""
        if len(self.skills) >= self.max_skills:
            # Remove lowest-utility skill
            min_idx = np.argmin([s.success_rate * s.usage_count
                                for s in self.skills])
            self.skills.pop(min_idx)

        # Compute embedding using TF-IDF
        try:
            embedding = self.vectorizer.fit_transform([skill_text]).toarray()[0]
            # Pad to embedding_dim if needed
            if embedding.shape[0] < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim -
                                              embedding.shape[0]))
        except:
            embedding = np.random.randn(self.embedding_dim)

        skill = Skill(
            description=skill_text,
            problem_types=problem_types or [],
            embedding=embedding,
            success_rate=0.5,  # Initial estimate
            usage_count=0
        )

        self.skills.append(skill)

    def retrieve_relevant_skills(self, problem: str, query_embedding=None,
                                k=3) -> List[Skill]:
        """Find most relevant skills for a problem."""
        if not self.skills:
            return []

        # Compute query embedding
        if query_embedding is None:
            query_embedding = self.vectorizer.transform([problem]).toarray()
            if query_embedding.shape[1] < self.embedding_dim:
                query_embedding = np.pad(query_embedding,
                                        ((0, 0), (0, self.embedding_dim -
                                          query_embedding.shape[1])))

        # Compute similarities
        similarities = []
        for skill in self.skills:
            sim = np.dot(query_embedding[0], skill.embedding)
            similarities.append((sim, skill))

        # Return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in similarities[:k]]

    def update_skill_stats(self, skill: Skill, helped: bool):
        """Update statistics on skill effectiveness."""
        skill.usage_count += 1

        # Update success rate with exponential smoothing
        alpha = 0.1
        skill.success_rate = (1 - alpha) * skill.success_rate + alpha * helped
```

Now implement the hierarchical training loop with the worker and skills manager.

```python
import torch
from torch.optim import AdamW

class HierarchicalReasoningTrainer:
    """Train worker and skill manager jointly."""

    def __init__(self, reasoning_model, skill_manager):
        self.worker = reasoning_model
        self.skill_manager = skill_manager
        self.optimizer = AdamW(reasoning_model.parameters(), lr=1e-5)

    def generate_with_skills(self, problem: str, num_samples=4) -> List[str]:
        """Generate solutions conditioned on retrieved skills."""
        # Retrieve relevant skills
        skills = self.skill_manager.retrieve_relevant_skills(problem, k=3)

        # Format skills as conditioning
        skill_context = "Relevant strategies:\n"
        for i, skill in enumerate(skills, 1):
            skill_context += f"{i}. {skill.description[:200]}\n"

        # Generate solutions
        prompt = f"{skill_context}\nProblem: {problem}\nSolution:"
        solutions = []

        for _ in range(num_samples):
            solution = self.worker.generate(prompt, max_length=500,
                                           temperature=0.7)
            solutions.append(solution)

        return solutions

    def step(self, problem: str, reference_answer: str = None,
            num_samples=4):
        """One training step: solve, evaluate, extract skills, update."""
        # Generate solutions
        solutions = self.generate_with_skills(problem, num_samples)

        # Evaluate solutions
        traces = []
        best_solution = None
        best_correctness = False

        for solution in solutions:
            correct = self._check_correctness(solution, reference_answer)

            trace = SolutionTrace(
                problem=problem,
                steps=solution.split('\n'),
                final_answer=solution.split('\n')[-1],
                correctness=correct,
                num_steps=len(solution.split('\n'))
            )
            traces.append(trace)

            if correct and not best_correctness:
                best_solution = solution
                best_correctness = True

        # Extract skills from best trace
        if best_solution:
            best_trace = next(t for t in traces
                            if t.final_answer in best_solution)
            skill_text = self.skill_manager.summarize_trace(best_trace)

            if skill_text:
                # Add to library
                self.skill_manager.add_skill(skill_text, problem_types=[])

                # Update skills used
                skills = self.skill_manager.retrieve_relevant_skills(problem)
                for skill in skills:
                    self.skill_manager.update_skill_stats(skill, True)

        # Compute hierarchical rewards
        trace_rewards = []
        for trace in traces:
            # Trace quality reward
            trace_quality = 1.0 if trace.correctness else 0.0

            # Efficiency reward (fewer steps is better, but not too few)
            efficiency = 1.0 if 3 <= trace.num_steps <= 10 else 0.5

            trace_rewards.append(0.7 * trace_quality + 0.3 * efficiency)

        trace_rewards = torch.tensor(trace_rewards, dtype=torch.float32)

        # Policy gradient update
        logprobs = self.worker.compute_logprobs(solutions)
        loss = -(trace_rewards * logprobs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.worker.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), float(trace_rewards.mean())

    def _check_correctness(self, solution: str,
                          reference_answer: str = None) -> bool:
        """Verify solution correctness."""
        # Extract final answer
        lines = solution.strip().split('\n')
        predicted_answer = lines[-1] if lines else ""

        if reference_answer:
            # Check if matches reference (lenient matching)
            return self._answers_match(predicted_answer, reference_answer)

        # Fallback: check for mathematical validity (simplified)
        return len(predicted_answer) > 0

    def _answers_match(self, predicted: str, reference: str) -> bool:
        """Check if answers match (handles multiple formats)."""
        # Normalize both
        pred_num = self._extract_number(predicted)
        ref_num = self._extract_number(reference)

        if pred_num is not None and ref_num is not None:
            return abs(pred_num - ref_num) < 1e-6

        return predicted.lower() == reference.lower()

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        import re
        match = re.search(r'-?\d+\.?\d*', text)
        return float(match.group()) if match else None

    def train(self, problems: List[str],
             reference_answers: List[str] = None,
             num_steps: int = 100):
        """Full training loop."""
        losses = []
        rewards = []

        for step in range(num_steps):
            # Sample a problem
            idx = np.random.randint(len(problems))
            problem = problems[idx]
            reference = reference_answers[idx] if reference_answers else None

            # Training step
            loss, reward = self.step(problem, reference)
            losses.append(loss)
            rewards.append(reward)

            if (step + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                avg_reward = np.mean(rewards[-10:])
                num_skills = len(self.skill_manager.skills)

                print(f"Step {step+1}: Loss={avg_loss:.4f}, "
                      f"Reward={avg_reward:.3f}, Skills={num_skills}")
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Skill library size 500-2000; larger libraries are more comprehensive but slower to search
- Retrieve top-3 to top-5 skills; more skills provide diversity, fewer are faster
- Use when solving problems from a coherent domain (math, code, logic puzzles)
- Particularly effective for problems with recurring patterns or sub-problems

**When NOT to use:**
- For one-off problems requiring unique reasoning (no reusable patterns)
- When problem domains are highly diverse (skill retrieval becomes unreliable)
- For latency-critical applications (skill retrieval and selection add overhead)

**Common Pitfalls:**
- Skill library becoming stale; periodically refresh by removing unused skills
- Skill summarization capturing noise rather than generalizable patterns; use multiple successful traces
- Retrieved skills being irrelevant to the current problem; improve embedding/similarity metric
- Worker overfitting to particular skill combinations; add randomization in skill selection

## Reference

Paper: [ARISE: Agent Reasoning with Intrinsic Skill Evolution in Hierarchical RL](https://arxiv.org/abs/2603.16060)
