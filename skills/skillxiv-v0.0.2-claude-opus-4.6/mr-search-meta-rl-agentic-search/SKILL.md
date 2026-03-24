---
name: mr-search-meta-rl-agentic-search
title: "Meta-Reinforcement Learning with Self-Reflection for Agentic Search"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.11327"
keywords: [Meta-RL, Self-Reflection, In-Context Learning, Agent Search, RLOO]
description: "Improve agent search through meta-RL: generate multiple episodes sequentially, each building on prior attempts with explicit self-reflection. Use turn-level RLOO advantage estimation to provide dense credit without value models."
---

# Technique: Cross-Episode Meta-Learning via In-Context Self-Reflection

Standard agent search treats each attempt independently—the model sees no feedback from failed attempts. Meta-RL-Search (MR-Search) inverts this: each new episode is conditioned on the full trajectory of prior attempts, with explicit reflections grounding the learning process. This transforms disconnected search tries into progressively informed exploration.

Rather than external process reward models, the approach uses turn-level RLOO advantage estimation to credit intermediate steps within each episode.

## Core Concept

MR-Search operates through three principles:

1. **Cross-Episode Learning**: Each episode conditions on prior trajectories and reflections
2. **Explicit Self-Reflection**: Generate reflections explaining what was learned after each attempt
3. **Turn-Level Advantage**: Use RLOO to credit intermediate decisions within episodes

This enables 9-19% relative improvement over outcome-only RL without auxiliary models.

## Architecture Overview

- **Agent policy**: Base LLM for reasoning
- **Reflection generator**: Creates actionable insights after each attempt
- **In-context memory**: Accumulates trajectories and reflections
- **RLOO critic**: Estimates advantages at turn granularity
- **Episode orchestrator**: Sequences attempts with conditioning

## Implementation Steps

### Step 1: Generate Trajectory with Explicit Reflection

Run one search episode, then generate reflection.

```python
import torch
from collections import namedtuple

SearchEpisode = namedtuple('SearchEpisode', ['trajectory', 'answer', 'reward', 'reflection'])

class AgentSearchWithReflection:
    def __init__(self, model, tokenizer, max_turns=10):
        self.model = model
        self.tokenizer = tokenizer
        self.max_turns = max_turns

    def generate_episode(self, question, prior_attempts=None):
        """
        Generate one search episode with self-reflection.

        question: str
        prior_attempts: list of prior SearchEpisode objects
        """
        trajectory = []

        # Build prompt with prior context
        context = self._build_context(question, prior_attempts)

        # Generate search trajectory
        for turn in range(self.max_turns):
            prompt = f"{context}\n\nTurn {turn + 1}:"

            # Generate reasoning step
            step_text = self.model.generate(
                prompt,
                max_tokens=200,
                stop_tokens=["\n\nFinal Answer:"]
            )

            trajectory.append(step_text)
            context += f"\nTurn {turn + 1}:\n{step_text}"

            # Check for final answer
            if "Final Answer:" in step_text:
                break

        # Extract final answer
        final_answer = self._extract_final_answer(trajectory)

        # Generate reflection
        reflection = self._generate_reflection(question, trajectory, final_answer)

        # Evaluate reward (external evaluation function)
        reward = self._evaluate_answer(final_answer, question)

        return SearchEpisode(
            trajectory=trajectory,
            answer=final_answer,
            reward=reward,
            reflection=reflection
        )

    def _build_context(self, question, prior_attempts):
        """Build prompt context from prior attempts."""
        if not prior_attempts:
            return f"Question: {question}"

        context = f"Question: {question}\n\nPrior Attempts:"

        for i, attempt in enumerate(prior_attempts):
            context += f"\n\nAttempt {i + 1}:\n"
            context += f"Reflection: {attempt.reflection}\n"
            context += f"Result: {'Correct' if attempt.reward > 0 else 'Incorrect'}"

        return context

    def _generate_reflection(self, question, trajectory, answer):
        """Generate lesson from attempt."""
        prompt = f"""Question: {question}

Trajectory:
{' '.join(trajectory)}

Final Answer: {answer}

What approach or reasoning pattern should inform the next attempt? (1-2 sentences)"""

        reflection = self.model.generate(prompt, max_tokens=100)
        return reflection.strip()

    def _extract_final_answer(self, trajectory):
        """Extract answer from trajectory."""
        full_text = ' '.join(trajectory)
        if "Final Answer:" in full_text:
            return full_text.split("Final Answer:")[-1].strip()
        return trajectory[-1]

    def _evaluate_answer(self, answer, question):
        """Evaluate correctness (external)."""
        # Placeholder: use external evaluator
        return 1.0 if self._is_correct(answer) else 0.0

    def _is_correct(self, answer):
        # External verification logic
        return True  # Placeholder
```

### Step 2: RLOO Advantage Estimation at Turn Level

Compute advantages for intermediate steps within each episode.

```python
class TurnLevelRLOOEstimator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def compute_turn_advantages(self, episode, baseline_episodes=None):
        """
        Estimate advantages for each turn using leave-one-out estimation.

        episode: SearchEpisode
        baseline_episodes: list of baseline episodes for comparison
        """
        trajectory = episode.trajectory
        episode_reward = episode.reward
        num_turns = len(trajectory)

        # RLOO: recompute trajectory without each turn
        turn_logprobs = []
        turn_counterfactuals = []

        for leave_out_idx in range(num_turns):
            # Reconstruct trajectory without this turn
            modified_traj = [
                trajectory[i] for i in range(num_turns)
                if i != leave_out_idx
            ]

            # Recompute log probs for this modified trajectory
            context = '\n'.join(modified_traj)
            model_output = self.model.compute_log_probs(context, trajectory[leave_out_idx])

            turn_logprobs.append(model_output)

            # Evaluate counterfactual: what would reward be without this step?
            # Approximate: remove step, continue trajectory
            modified_full_traj = modified_traj + trajectory[leave_out_idx + 1:]
            final_answer = self._extract_answer(modified_full_traj)

            counterfactual_reward = self._evaluate(final_answer)
            turn_counterfactuals.append(counterfactual_reward)

        # Advantages: actual - counterfactual
        advantages = []
        for i in range(num_turns):
            advantage = episode_reward - turn_counterfactuals[i]
            advantages.append(advantage)

        return torch.tensor(advantages)

    def _extract_answer(self, trajectory):
        full_text = ' '.join(trajectory)
        if "Final Answer:" in full_text:
            return full_text.split("Final Answer:")[-1].strip()
        return full_text

    def _evaluate(self, answer):
        # External evaluation
        return 1.0 if self._is_correct(answer) else 0.0

    def _is_correct(self, answer):
        return True  # Placeholder
```

### Step 3: Sequential Episode Generation with Meta-Learning

Generate K episodes per question, each informed by prior attempts.

```python
def meta_rl_search_for_question(
    model,
    tokenizer,
    question,
    num_episodes=3,
    learning_rate=1e-5
):
    """
    Generate multiple episodes with meta-learning.
    """
    agent = AgentSearchWithReflection(model, tokenizer)
    rloo_estimator = TurnLevelRLOOEstimator(model, tokenizer)

    episodes = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for episode_idx in range(num_episodes):
        # Generate episode conditioned on prior attempts
        episode = agent.generate_episode(question, prior_attempts=episodes)

        episodes.append(episode)

        # Compute turn-level advantages
        advantages = rloo_estimator.compute_turn_advantages(episode)

        # Policy gradient update
        trajectory_log_probs = []
        for i, step_text in enumerate(episode.trajectory):
            # Get log prob of this step
            log_prob = model.compute_log_prob(
                context='\n'.join(episode.trajectory[:i]),
                text=step_text
            )
            trajectory_log_probs.append(log_prob)

        trajectory_log_probs = torch.stack(trajectory_log_probs)

        # Loss: -log_prob * advantage
        loss = -(trajectory_log_probs * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode_idx + 1}: Reward={episode.reward:.2f}, Loss={loss.item():.4f}")

    # Return best episode
    best_episode = max(episodes, key=lambda e: e.reward)
    return best_episode
```

### Step 4: In-Context Meta-RL at Test Time

Apply meta-learning directly at test time without additional training.

```python
def test_time_meta_rl(
    model,
    question,
    num_attempts=5,
    reflection_prompt_template=None
):
    """
    Generate multiple attempts with in-context learning (no training).
    """
    attempts = []

    for attempt_idx in range(num_attempts):
        # Build prompt with prior attempts
        if attempts:
            prompt = "Question: " + question + "\n\nPrior Attempts:\n"
            for prev_attempt in attempts:
                prompt += f"- Attempt {len(attempts)}: {prev_attempt['reflection']}\n"
            prompt += f"\nAttempt {attempt_idx + 1}:"
        else:
            prompt = f"Question: {question}\n\nAttempt 1:"

        # Generate response
        response = model.generate(prompt, max_tokens=500)

        # Extract answer
        if "Final Answer:" in response:
            answer = response.split("Final Answer:")[-1].strip()
        else:
            answer = response

        # Generate reflection for next attempt
        reflection_prompt = f"""The previous attempt gave: {answer}
Briefly explain what should be reconsidered in the next attempt."""

        reflection = model.generate(reflection_prompt, max_tokens=100)

        attempts.append({
            'response': response,
            'answer': answer,
            'reflection': reflection
        })

    # Select best answer (could use external evaluator)
    best_attempt = attempts[-1]  # Use most recent as best
    return best_attempt
```

## Practical Guidance

**When to Use:**
- Multi-step reasoning tasks where multiple attempts provide value
- Scenarios where LLM reflection helps course-correct
- Question-answering with external verification available
- Interactive settings where sequential refinement is beneficial

**When NOT to Use:**
- Simple single-shot tasks
- Extreme latency constraints (sequential generation is slower)
- Domains where reflection doesn't improve reasoning

**Hyperparameter Tuning:**
- **num_episodes**: 2-5 typical; more computation for diminishing returns
- **max_turns_per_episode**: 5-15; balance depth and latency
- **RLOO sampling**: Leave-one-out standard; consider importance sampling variants
- **learning_rate**: 1e-5 to 1e-4 typical for in-context updates

**Common Pitfalls:**
- Reflections becoming generic, not informative
- Over-reliance on first few attempts' trajectories
- Insufficient diversity in sequential attempts
- Not preserving best attempt's reasoning pattern

## Reference

[Meta-RL Search paper on arXiv](https://arxiv.org/abs/2603.11327)
