---
name: multi-agent-evolve
title: "Multi-Agent Evolve: LLM Self-Improve through Co-evolution"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.23595"
keywords: [Self-Play, Co-evolution, Reasoning, RL, LLM Training]
description: "Enables LLM self-improvement without external verification through multi-agent co-evolution. Proposer generates questions, Solver attempts solutions, Judge evaluates both. All agents evolve together via RL, achieving 4.54% improvement on reasoning benchmarks without human supervision."
---

# Multi-Agent Evolve: Self-Improvement via Co-evolutionary RL

LLM reasoning improvement typically requires external verifiers (Python, human feedback). Multi-Agent Evolve eliminates this dependency through three co-evolving agents that provide internal supervision.

The proposer-solver-judge framework creates a closed learning loop where agents improve by interacting with each other.

## Core Concept

Three specialized agents from single base LLM:
- **Proposer**: generates diverse questions
- **Solver**: attempts to solve them
- **Judge**: evaluates both question quality and solution correctness

All three undergo RL simultaneously, creating adaptive curriculum as proposer generates harder questions.

## Architecture Overview

- Three agent roles instantiated from same LLM
- Shared parameter backbone with role-specific prompting
- Reward signals derived from agent interactions
- Co-evolutionary training loop with alternating optimization

## Implementation Steps

Define agent roles with role-specific prompting. Each agent is a view of the same LLM with different prompts:

```python
class MultiAgentSystem:
    def __init__(self, base_llm, model_size='3B'):
        self.base_llm = base_llm
        self.model_size = model_size

        # Role-specific system prompts
        self.proposer_prompt = (
            "You are a question generator. Generate challenging questions "
            "that test reasoning. Format: QUESTION: [question]"
        )

        self.solver_prompt = (
            "You are a problem solver. Given a question, provide step-by-step "
            "reasoning and a final answer. Format: REASONING: [steps]\nANSWER: [answer]"
        )

        self.judge_prompt = (
            "You are an evaluator. Rate question quality (1-5) and solution "
            "correctness (1-5). Format: QUESTION_SCORE: [score]\nSOLUTION_SCORE: [score]"
        )

    def generate_question(self, context=""):
        """Proposer generates question."""
        prompt = self.proposer_prompt + f"\nContext: {context}\n"
        question = self.base_llm.generate(prompt, max_tokens=100)
        return question

    def solve_question(self, question):
        """Solver generates solution."""
        prompt = self.solver_prompt + f"\nQuestion: {question}\n"
        solution = self.base_llm.generate(prompt, max_tokens=500)
        return solution

    def judge_quality(self, question, solution):
        """Judge evaluates both."""
        prompt = self.judge_prompt
        prompt += f"\nQuestion: {question}\nSolution: {solution}\n"
        evaluation = self.base_llm.generate(prompt, max_tokens=50)

        # Parse scores
        scores = self._parse_scores(evaluation)
        return scores

    def _parse_scores(self, evaluation):
        """Extract question and solution scores."""
        # Simple parsing - in practice use more robust extraction
        question_score = 3.0  # Default
        solution_score = 3.0

        import re
        q_match = re.search(r'QUESTION_SCORE:\s*(\d)', evaluation)
        s_match = re.search(r'SOLUTION_SCORE:\s*(\d)', evaluation)

        if q_match:
            question_score = float(q_match.group(1))
        if s_match:
            solution_score = float(s_match.group(1))

        return question_score, solution_score
```

Implement the co-evolution training loop where agents interact and learn:

```python
def coevolution_training_step(agents, rl_trainer, num_interactions=100):
    """Single training step with agent interaction and RL updates."""
    experiences = []

    for _ in range(num_interactions):
        # Proposer generates question
        context = "Mathematics, Reasoning"
        question = agents.generate_question(context)

        # Solver solves it
        solution = agents.solve_question(question)

        # Judge evaluates both
        q_score, s_score = agents.judge_quality(question, solution)

        # Record experience
        experience = {
            'question': question,
            'solution': solution,
            'question_score': q_score,
            'solution_score': s_score,
            'total_reward': 0.6 * q_score + 0.4 * s_score
        }
        experiences.append(experience)

    # RL training: optimize all three roles
    proposer_loss = rl_trainer.optimize_proposer(experiences)
    solver_loss = rl_trainer.optimize_solver(experiences)
    judge_loss = rl_trainer.optimize_judge(experiences)

    return {
        'proposer_loss': proposer_loss,
        'solver_loss': solver_loss,
        'judge_loss': judge_loss,
        'avg_reward': sum(e['total_reward'] for e in experiences) / len(experiences)
    }

def train_multi_agent_evolve(agents, num_rounds=1000, interaction_per_round=100):
    """Full training loop."""
    rl_trainer = RLTrainer(agents.base_llm)

    for round_idx in range(num_rounds):
        metrics = coevolution_training_step(agents, rl_trainer, interaction_per_round)

        if round_idx % 100 == 0:
            print(f"Round {round_idx}: "
                  f"Avg Reward: {metrics['avg_reward']:.3f}, "
                  f"Proposer Loss: {metrics['proposer_loss']:.4f}")

    return agents
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Interactions per round | 100-200 (curriculum generation) |
| Question/Solution/Judge reward weight | 0.6/0.4/0.0 (focus on outcome) |
| Model size | 3B-7B (efficient self-play) |
| Training rounds | 500-1000 (convergence check) |

**When to use:**
- Self-improvement without external verifiers
- Closed-loop reasoning training
- Domains with no ground-truth evaluation functions
- Scaling reasoning with cheaper models

**When NOT to use:**
- When accurate external verification exists (more stable)
- Real-time applications (training overhead)
- Tasks requiring factual correctness (no ground truth in loop)

**Common pitfalls:**
- Judge becoming too lenient (reward inflation)
- Proposer generating trivial questions (solver overfits)
- Unbalanced role optimization (one dominates)
- Insufficient diversity in proposals (stuck in local optimum)

Reference: [Multi-Agent Evolve on arXiv](https://arxiv.org/abs/2510.23595)
