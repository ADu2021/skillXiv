---
name: mars-socratic-prompt-optimization
title: "MARS: A Multi-Agent Framework Incorporating Socratic Guidance for Automated Prompt Optimization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.16874"
keywords: [Prompt Optimization, Multi-Agent Systems, Socratic Dialogue, Reinforcement Learning, LLM Efficiency]
description: "Optimize task-specific prompts using five cooperative agents (Planner, Teacher, Critic, Student, Target) in a POMDP framework, where the Planner generates adaptive trajectories and a Teacher-Critic-Student triad refines prompts through Socratic dialogue, achieving 85.11% accuracy on general tasks and 75.81% on specialized domains."
---

## Core Concept

MARS addresses rigid and inefficient prompt optimization by decomposing the problem into adaptive steps executed by five specialized agents within a Partially Observable Markov Decision Process (POMDP). Rather than fixed templates, the Planner generates task-specific optimization trajectories. The Teacher-Critic-Student triad engages in Socratic-style dialogue where the Teacher poses guiding questions, the Critic evaluates clarity and coherence, and the Student refines prompts based on feedback. The Target agent evaluates final prompts on downstream tasks, providing reward signals for continuous improvement.

## Architecture Overview

The framework operates with five coordinated agents:

- **Planner Agent**: Generates task-specific optimization sub-goal sequences, enabling adaptive exploration rather than template-based refinement
- **Teacher Agent**: Poses Socratic questions aligned with current optimization objectives, guiding the Student toward effective prompts
- **Critic Agent**: Evaluates question quality, prompt clarity, and semantic coherence, providing targeted feedback
- **Student Agent**: Maintains internal state and produces refined prompts based on Teacher questions and Critic feedback
- **Target Agent**: Evaluates final prompts on actual downstream tasks, computing reward signals for optimization

The system uses dialogue history to maintain reasoning consistency and context-aware interactions across refinement steps.

## Implementation

The POMDP formulation models the optimization process as a partially observable state-action-reward loop:

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class PlannerAgent(nn.Module):
    """Generates task-specific optimization trajectories."""
    def __init__(self, hidden_dim=1024, max_steps=10):
        super().__init__()
        self.task_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.trajectory_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.step_predictor = nn.Linear(hidden_dim, 1)
        self.max_steps = max_steps

    def forward(self, task_embedding: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        Args:
            task_embedding: (B, D) task representation
        Returns:
            sub_goals: list of optimization sub-goal descriptions
            state: (B, D) current planning state
        """
        encoded = self.task_encoder(task_embedding)
        trajectory = self.trajectory_decoder(encoded)

        # Generate adaptive number of steps (1 to max_steps)
        num_steps = torch.clamp(
            (self.step_predictor(trajectory) * self.max_steps).long(),
            min=1, max=self.max_steps
        )

        # Sub-goals are learned from planning state
        sub_goals = [f"Optimize for objective {i}" for i in range(self.max_steps)]

        return sub_goals, trajectory
```

The Teacher-Critic-Student triad implements the Socratic dialogue loop:

```python
class TeacherAgent(nn.Module):
    """Generates Socratic questions to guide prompt refinement."""
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.question_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, task_state: torch.Tensor, current_objective: str) -> torch.Tensor:
        """
        Args:
            task_state: (B, D) current task state
            current_objective: current sub-goal description
        Returns:
            question_embedding: (B, D) Socratic question embedding
        """
        # Concatenate task state with objective encoding
        objective_embedding = self._encode_text(current_objective)
        combined = torch.cat([task_state, objective_embedding], dim=1)

        question = self.question_generator(combined)
        return question

    def _encode_text(self, text: str) -> torch.Tensor:
        """Simple text encoding; in practice use LLM embeddings."""
        return torch.randn(1, 1024)


class CriticAgent(nn.Module):
    """Evaluates question quality and prompt coherence."""
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.coherence_scorer = nn.Linear(hidden_dim, 1)

    def forward(self, question: torch.Tensor, prompt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            question: (B, D) question embedding
            prompt: (B, D) current prompt embedding
        Returns:
            quality_score: (B, 1) question quality [0, 1]
            coherence_score: (B, 1) prompt coherence [0, 1]
        """
        quality = self.quality_scorer(question)
        coherence = torch.sigmoid(self.coherence_scorer(prompt))

        return quality, coherence


class StudentAgent(nn.Module):
    """Maintains state and generates refined prompts."""
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.state_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.prompt_generator = nn.Linear(hidden_dim, hidden_dim)
        self.dialogue_memory = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        current_state: torch.Tensor,
        question: torch.Tensor,
        critic_feedback: torch.Tensor,
        dialogue_history: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            current_state: (B, D) student's current state
            question: (B, D) teacher's Socratic question
            critic_feedback: (B, 2) [quality, coherence] scores
            dialogue_history: list of previous interaction embeddings
        Returns:
            new_state: (B, D) updated student state
            refined_prompt: (B, D) refined prompt embedding
        """
        # Incorporate feedback into state update
        feedback_signal = critic_feedback[:, 0:1]
        combined_input = question + feedback_signal * current_state

        # Update state via GRU
        new_state = self.state_update(combined_input, current_state)

        # Incorporate dialogue history
        if dialogue_history:
            history_embedding = torch.stack(dialogue_history).mean(dim=0)
            memory_signal = self.dialogue_memory(history_embedding)
            new_state = new_state + 0.3 * memory_signal

        # Generate refined prompt
        refined_prompt = self.prompt_generator(new_state)

        return new_state, refined_prompt
```

The Target agent evaluates prompts on downstream tasks and provides reward signals:

```python
class TargetAgent(nn.Module):
    """Evaluates refined prompts on downstream tasks."""
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.task_evaluator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, prompt: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompt: (B, D) candidate prompt
            task_embedding: (B, D) task representation
        Returns:
            reward: (B, 1) performance score [0, 1]
        """
        combined = torch.cat([prompt, task_embedding], dim=1)
        reward = self.task_evaluator(combined)
        return reward


def socratic_optimization_loop(
    initial_prompt: str,
    task_embedding: torch.Tensor,
    planner: PlannerAgent,
    teacher: TeacherAgent,
    critic: CriticAgent,
    student: StudentAgent,
    target: TargetAgent,
    max_iterations: int = 10,
    early_stopping_delta: float = 0.01
):
    """
    Orchestrates the full MARS optimization loop.

    Args:
        initial_prompt: starting prompt
        task_embedding: (B, D) task representation
        Other agents: defined above
        max_iterations: maximum refinement steps
        early_stopping_delta: convergence threshold
    Returns:
        best_prompt: optimized prompt
        optimization_trajectory: history of improvements
    """
    # Initialize
    student_state = torch.randn_like(task_embedding)
    dialogue_history = []
    best_reward = 0
    optimization_trajectory = []

    # Generate optimization plan
    sub_goals, planning_state = planner(task_embedding)

    for iteration in range(max_iterations):
        current_objective = sub_goals[min(iteration, len(sub_goals) - 1)]

        # Teacher poses Socratic question
        question = teacher(student_state, current_objective)

        # Critic evaluates question quality
        prompt_embedding = torch.randn(task_embedding.shape[0], 1024)  # placeholder
        quality, coherence = critic(question, prompt_embedding)
        feedback = torch.cat([quality, coherence], dim=1)

        # Student refines prompt based on feedback
        new_state, refined_prompt = student(
            student_state, question, feedback, dialogue_history
        )
        dialogue_history.append(refined_prompt)

        # Target evaluates refined prompt
        reward = target(refined_prompt, task_embedding)

        # Track best performance
        improvement = reward - best_reward
        optimization_trajectory.append({
            'iteration': iteration,
            'reward': reward.item(),
            'improvement': improvement.item()
        })

        best_reward = max(best_reward, reward)
        student_state = new_state

        # Early stopping
        if improvement < early_stopping_delta and iteration > 2:
            break

    return refined_prompt, optimization_trajectory
```

## Practical Guidance

**When to Use:**
- Optimizing task-specific prompts for language models
- Scenarios requiring adaptive refinement beyond fixed templates
- General-purpose tasks (85.11% accuracy) or specialized domains (75.81%)
- When dialogue-based feedback improves prompt quality
- Multi-turn optimization where each iteration builds on previous insights

**When NOT to Use:**
- Simple prompt engineering where manual tuning suffices
- Real-time applications requiring immediate responses (iterative refinement adds latency)
- Tasks where the base model is already well-tuned
- Low-resource settings where calling the target model repeatedly is expensive
- Non-language tasks or non-LLM deployments

**Key Hyperparameters:**
- `max_iterations`: 5-10 typically; diminishing returns beyond 10
- `early_stopping_delta` (δ): 0.01-0.05; smaller = more refinement cycles
- `temperature`: 0.6 for balancing creativity and coherence
- `hidden_dim`: 1024-2048 for better representation capacity
- Learning rate: 1e-4 to 1e-3 for stable optimization

**Common Pitfalls:**
- Setting early_stopping_delta too high terminates refinement prematurely
- Over-relying on Target agent reward without validating on true task metrics
- Not maintaining dialogue history; context loss across iterations
- Planner generating too many or too few sub-goals (breaks adaptation)
- Using weak base models; MARS amplifies quality but cannot create knowledge

## Performance Notes

- General tasks: 85.11% average accuracy, +6.04% over SOTA
- Specialized domains: 75.81% average, +6.42% over SOTA
- Cross-model generalization: Works with GPT-3.5, GPT-4, GPT-4o, Deepseek-R1
- Computational efficiency: Superior inference-time scaling compared to baselines
- Single-shot training: Requires only 1 example per dataset

## References

- Partially Observable Markov Decision Process (POMDP) frameworks
- Socratic method for guided dialogue and learning
- Multi-agent reinforcement learning and cooperative agents
- Prompt engineering and few-shot learning
- Deepseek-v2.5, GPT-4, and other LLM backbones
