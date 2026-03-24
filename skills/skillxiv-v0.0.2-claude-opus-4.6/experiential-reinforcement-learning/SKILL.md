---
name: experiential-reinforcement-learning
title: "Experiential Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.13949"
keywords: [Reinforcement Learning, Reflection, Revision, Behavioral Distillation, Feedback Integration]
description: "Improve RL training efficiency by embedding explicit reflection and revision loops. Models generate initial responses, receive feedback, produce self-reflections describing improvements, revise their attempts, and distill successful corrections into the base policy. Achieves up to 81% improvement on complex tasks through structured behavioral change."
---

# Experiential Reinforcement Learning

## Problem Context

Standard RL trains models to implicitly discover how failures should translate into behavioral change through undirected trial-and-error. This is inefficient and wastes signal from corrective feedback. Experiential RL makes the learning process explicit: models generate reflections on failures, attempt revisions based on feedback, and internalize successful corrections. This transforms reactive behavior optimization into structured experience learning.

## Core Concept

ERL operates in three phases: (1) initial attempt on a task, (2) reflection and revision (model generates explanation of improvements and attempts again), (3) internalization via distillation. A cross-episode reflection memory stores successful corrective patterns discovered during training, enabling reuse across tasks.

## Architecture Overview

- **Initial attempt**: Generate first response to task
- **Reflection generation**: Produce reasoning about improvements given feedback
- **Revision step**: Attempt task again using reflection insights
- **Success detection**: Identify when revision succeeds
- **Distillation**: Consolidate successful revisions into base policy
- **Reflection memory**: Store and retrieve successful correction patterns

## Implementation

### Step 1: Task attempt and feedback collection

```python
import torch
from typing import Dict, List, Tuple
from collections import defaultdict

class TaskAttempt:
    """Single task attempt with response and feedback."""

    def __init__(self, task: str, initial_response: str):
        self.task = task
        self.initial_response = initial_response
        self.feedback = None
        self.revision = None
        self.revision_success = False

    def add_feedback(self, feedback: str, success: bool = False):
        """Add environmental feedback."""
        self.feedback = feedback
        self.initial_success = success


class ReflectionMemory:
    """Cross-episode memory of successful correction patterns."""

    def __init__(self, max_memory: int = 1000):
        self.max_memory = max_memory
        self.reflections = defaultdict(list)  # task_type -> [successful_reflections]
        self.corrections = []  # List of (initial, reflection, revision) tuples

    def store_successful_correction(
        self,
        task_type: str,
        initial_response: str,
        reflection: str,
        revised_response: str,
        task_description: str = None
    ):
        """Store successful correction for future reuse."""
        self.reflections[task_type].append({
            'reflection': reflection,
            'initial': initial_response,
            'revised': revised_response,
            'task': task_description
        })

        self.corrections.append((initial_response, reflection, revised_response))

        # Trim memory if exceeded
        if len(self.corrections) > self.max_memory:
            self.corrections = self.corrections[-self.max_memory:]

    def retrieve_similar_reflections(
        self,
        task_type: str,
        query_embedding: torch.Tensor = None,
        top_k: int = 3
    ) -> List[Dict]:
        """Retrieve similar successful reflections for in-context learning."""
        if task_type not in self.reflections:
            return []

        candidates = self.reflections[task_type]

        if query_embedding is None:
            # Simple: return most recent
            return candidates[-top_k:]

        # With embeddings: return most similar
        # (Simplified; would use vector similarity)
        return candidates[-top_k:]
```

### Step 2: Generate reflections

```python
class ReflectionGenerator:
    """Generate reflections on failures and improvements."""

    def __init__(self, model):
        self.model = model

    def generate_reflection(
        self,
        task: str,
        initial_response: str,
        feedback: str,
        reflection_memory: ReflectionMemory = None,
        task_type: str = None
    ) -> str:
        """
        Generate reflection on failure and proposed improvements.

        Args:
            task: Task description
            initial_response: Initial attempt
            feedback: Environmental feedback on failure
            reflection_memory: Optional memory of past corrections
            task_type: Category of task for memory retrieval

        Returns:
            reflection: Reasoning about improvements
        """
        # Retrieve similar successful reflections for in-context examples
        in_context_examples = ""
        if reflection_memory and task_type:
            similar = reflection_memory.retrieve_similar_reflections(task_type, top_k=2)
            if similar:
                in_context_examples = "Similar successful corrections:\n"
                for ex in similar[:2]:
                    in_context_examples += (
                        f"Initial: {ex['initial'][:100]}...\n"
                        f"Reflection: {ex['reflection']}\n"
                        f"Revised: {ex['revised'][:100]}...\n\n"
                    )

        # Construct reflection prompt
        prompt = f"""
Task: {task}

Initial attempt: {initial_response}

Feedback on failure: {feedback}

{in_context_examples}

Analyze the failure and generate a reflection on what went wrong and how to improve. Be specific about the changes needed.

Reflection:"""

        reflection = self.model.generate(prompt, max_tokens=200, temperature=0.7)

        return reflection.strip()
```

### Step 3: Generate revision based on reflection

```python
class RevisionGenerator:
    """Generate revised responses using reflection insights."""

    def __init__(self, model):
        self.model = model

    def generate_revision(
        self,
        task: str,
        initial_response: str,
        reflection: str,
        max_tokens: int = 500
    ) -> str:
        """
        Generate revised attempt using reflection.

        Args:
            task: Original task
            initial_response: First attempt that failed
            reflection: Generated reflection on improvements
            max_tokens: Maximum response length

        Returns:
            revised_response: Improved attempt
        """
        prompt = f"""
Task: {task}

Previous attempt that failed: {initial_response}

Analysis of improvements needed: {reflection}

Based on the analysis, provide a revised solution. Apply all improvements identified in the analysis.

Revised solution:"""

        revised = self.model.generate(prompt, max_tokens=max_tokens, temperature=0.6)

        return revised.strip()
```

### Step 4: Experiential RL training loop

```python
class ExperientialRL:
    """Full ERL training with reflection, revision, and distillation."""

    def __init__(
        self,
        model,
        optimizer,
        verifier,
        reflection_memory: ReflectionMemory = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.verifier = verifier
        self.reflection_generator = ReflectionGenerator(model)
        self.revision_generator = RevisionGenerator(model)
        self.reflection_memory = reflection_memory or ReflectionMemory()

    def experiential_episode(
        self,
        task: str,
        task_type: str = None,
        max_revisions: int = 2
    ) -> Dict:
        """
        Execute single ERL episode with reflection and revision.

        Returns:
            episode_data: {initial, reflection, revision, success, log_probs}
        """
        episode_data = {
            'task': task,
            'attempts': [],
            'success': False,
            'final_reflection': None
        }

        # Initial attempt
        initial_response, log_probs_initial = self.model.generate_with_logprobs(
            task, max_tokens=500
        )
        episode_data['attempts'].append({
            'response': initial_response,
            'log_probs': log_probs_initial,
            'is_revision': False
        })

        initial_success = self.verifier(initial_response, task)

        if initial_success:
            episode_data['success'] = True
            return episode_data

        # Get feedback
        feedback = self._generate_feedback(initial_response, task)

        # Reflection and revision loop
        for revision_idx in range(max_revisions):
            # Generate reflection
            reflection = self.reflection_generator.generate_reflection(
                task, initial_response, feedback,
                self.reflection_memory, task_type
            )
            episode_data['final_reflection'] = reflection

            # Generate revision
            revised_response, log_probs_revised = self.model.generate_with_logprobs(
                f"{task}\n\nAnalysis of improvements: {reflection}\n\nRevised solution:",
                max_tokens=500
            )
            episode_data['attempts'].append({
                'response': revised_response,
                'log_probs': log_probs_revised,
                'is_revision': True,
                'reflection': reflection
            })

            # Check success
            revision_success = self.verifier(revised_response, task)

            if revision_success:
                episode_data['success'] = True
                # Store successful correction
                if task_type:
                    self.reflection_memory.store_successful_correction(
                        task_type,
                        initial_response,
                        reflection,
                        revised_response,
                        task
                    )
                return episode_data

            # Update for next iteration
            initial_response = revised_response
            feedback = self._generate_feedback(revised_response, task)

        return episode_data

    def _generate_feedback(self, response: str, task: str) -> str:
        """Generate feedback on response failure."""
        # Simplified: use rule-based or LLM feedback
        return f"Your response to '{task[:50]}...' was incorrect."

    def distill_successful_revision(
        self,
        episode_data: Dict
    ) -> float:
        """
        Distill successful revision into base policy via supervised loss.

        Args:
            episode_data: Episode with successful revision

        Returns:
            loss: Distillation loss
        """
        if not episode_data['success']:
            return 0.0

        # Find successful revision attempt
        successful_attempt = None
        for attempt in episode_data['attempts']:
            if attempt['is_revision']:
                successful_attempt = attempt
                break

        if successful_attempt is None:
            return 0.0

        # Supervised fine-tuning on successful response
        task = episode_data['task']
        successful_response = successful_attempt['response']

        # Forward pass: model predicts successful response
        prompt_logits = self.model.forward(task)
        response_logits = self.model.forward(
            f"{task}\n\nAnswer: {successful_response}"
        )

        # Cross-entropy loss on successful tokens
        loss = self._compute_ce_loss(response_logits, successful_response)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _compute_ce_loss(self, logits: torch.Tensor, target_text: str) -> torch.Tensor:
        """Compute cross-entropy loss (simplified)."""
        # Simplified: would tokenize and compute actual CE
        return torch.tensor(0.0, requires_grad=True)
```

### Step 5: Full training loop

```python
def train_experiential_rl(
    model,
    tasks: List[Dict],
    verifier,
    optimizer,
    num_epochs: int = 5,
    device: str = 'cuda'
):
    """
    Train LLM using experiential reinforcement learning.

    Args:
        tasks: List of {task, task_type} dicts
        verifier: Function checking task correctness
    """
    erl = ExperientialRL(model, optimizer, verifier)

    for epoch in range(num_epochs):
        total_episodes = 0
        successful_episodes = 0
        total_distill_loss = 0.0

        for task_dict in tasks:
            task = task_dict['task']
            task_type = task_dict.get('type', 'generic')

            # Execute episode
            episode_data = erl.experiential_episode(task, task_type=task_type)

            total_episodes += 1

            if episode_data['success']:
                successful_episodes += 1

                # Distill successful revision
                loss = erl.distill_successful_revision(episode_data)
                total_distill_loss += loss

        success_rate = successful_episodes / total_episodes
        avg_loss = total_distill_loss / max(1, successful_episodes)

        print(f"Epoch {epoch + 1}: "
              f"Success={success_rate:.2%}, "
              f"DistillLoss={avg_loss:.4f}")

    return model
```

## Practical Guidance

**When to use**: Tasks with clear feedback signals (verifiable correctness, grading rubrics); complex reasoning where reflection helps

**Hyperparameters**:
- **max_revisions**: 1-3 (tradeoff: more revisions = better but slower)
- **reflection_temperature**: 0.7 (balance exploration vs. specificity)
- **revision_temperature**: 0.6 (more deterministic)
- **memory_size**: 500-2000 (reflection memory)

**Key advantages**:
- Structured behavioral change through reflection
- Reusable correction patterns via memory
- More efficient signal use than standard RL
- Works on tasks with any feedback signal

**Common pitfalls**:
- Reflection too generic → not actionable
- Max revisions too high → dataset expansion dominates
- Memory not task-specific → wrong examples retrieved
- Distillation overfitting to successful trajectory

**Scaling**: Linear in number of episodes; reflection memory enables amortization.

## Reference

Paper: https://arxiv.org/abs/2602.13949
Related work: Reinforcement learning, behavioral cloning, in-context learning
Benchmarks: Control tasks, question-answering, reasoning tasks
