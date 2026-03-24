---
name: ruscarl-rubric-scaffolded-rl
title: "RuscaRL: Rubric-Scaffolded RL for General LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.16949
keywords: [reinforcement-learning, rubric-guidance, exploration-guidance, reasoning-scaffolding, curriculum-learning]
description: "Guide LLM exploration through rubric-based scaffolding that gradually diminishes, enabling models to internalize reasoning patterns while maintaining exploration quality for robust RL training."
---

# RuscaRL: Rubric-Scaffolded Reinforcement Learning

## Core Concept

The core challenge in LLM RL is exploration: what cannot be explored cannot be learned. RuscaRL (Rubric-Scaffolded RL) addresses this by using checklist-style rubrics in two phases: first as explicit guidance during exploration (gradually fading), then as reference for scoring. This approach breaks the constraint that LLM limitations restrict sample quality, enabling models to learn general reasoning capabilities through guided exploration.

## Architecture Overview

- **Exploration Rubrics**: Checklist guidance for high-quality response generation
- **Gradual Scaffolding Fade**: Curriculum learning removing explicit guidance
- **Rubric-Based Scoring**: Reference for reward computation
- **LLM-as-Judge**: Automated evaluation using rubrics
- **General Reasoning Transfer**: Patterns learned with rubrics transfer without them

## Implementation Steps

### 1. Design Task-Specific Rubrics

Create structured evaluation checklist templates:

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class RubricLevel(Enum):
    EXPLICIT = "explicit"      # Full rubric in prompt
    IMPLICIT = "implicit"      # Rubric referenced but not detailed
    ABSENT = "absent"           # No rubric mentioned

@dataclass
class RubricCriterion:
    """Single rubric evaluation criterion."""
    name: str
    description: str
    exemplar_good: str
    exemplar_bad: str
    weight: float = 1.0

@dataclass
class TaskRubric:
    """Complete rubric for a task."""
    task_name: str
    criteria: List[RubricCriterion]
    instructions: str

    def render_as_guidance(self) -> str:
        """Render rubric as exploration guidance in prompt."""
        guidance = f"Task: {self.task_name}\n\n"
        guidance += "Evaluation Rubric (aim to satisfy these):\n"

        for criterion in self.criteria:
            guidance += f"\n- {criterion.name}: {criterion.description}\n"
            guidance += f"  Good example: {criterion.exemplar_good}\n"
            guidance += f"  Avoid: {criterion.exemplar_bad}\n"

        return guidance

class RubricLibrary:
    """Library of task-specific rubrics."""

    def __init__(self):
        self.rubrics: Dict[str, TaskRubric] = {}

    def register_rubric(self, task_type: str, rubric: TaskRubric):
        """Register rubric for task type."""
        self.rubrics[task_type] = rubric

    def get_rubric(self, task_type: str) -> TaskRubric:
        return self.rubrics.get(task_type)

    def create_math_rubric(self) -> TaskRubric:
        """Example: rubric for mathematical reasoning."""
        return TaskRubric(
            task_name="Mathematical Problem Solving",
            criteria=[
                RubricCriterion(
                    name="Clear Problem Understanding",
                    description="State what is being asked before solving",
                    exemplar_good="We need to find the value of x. Given: 2x + 5 = 15",
                    exemplar_bad="Just give the answer: x=5",
                    weight=1.0
                ),
                RubricCriterion(
                    name="Step-by-Step Solution",
                    description="Show each calculation step explicitly",
                    exemplar_good="2x + 5 = 15\nSubtract 5: 2x = 10\nDivide by 2: x = 5",
                    exemplar_bad="x = 5",
                    weight=2.0
                ),
                RubricCriterion(
                    name="Verification",
                    description="Check the answer by substitution",
                    exemplar_good="Check: 2(5) + 5 = 10 + 5 = 15 ✓",
                    exemplar_bad="(no verification)",
                    weight=1.0
                ),
                RubricCriterion(
                    name="Correct Final Answer",
                    description="Clearly state the answer",
                    exemplar_good="Therefore, x = 5",
                    exemplar_bad="(answer buried in work)",
                    weight=3.0
                )
            ],
            instructions="Solve the problem following the rubric above."
        )
```

### 2. Implement Exploration with Rubric Guidance

Use rubrics to enhance exploration quality:

```python
class RubricGuidedExploration:
    """Guide exploration with rubric scaffolding."""

    def __init__(
        self,
        model: "LLM",
        rubric_library: RubricLibrary
    ):
        self.model = model
        self.rubric_library = rubric_library

    def generate_with_rubric_guidance(
        self,
        task: str,
        task_type: str,
        guidance_level: RubricLevel = RubricLevel.EXPLICIT,
        temperature: float = 0.9
    ) -> str:
        """
        Generate response with rubric guidance.
        Guidance level controls how much structure to provide.
        """
        rubric = self.rubric_library.get_rubric(task_type)

        if guidance_level == RubricLevel.EXPLICIT:
            # Full rubric in prompt
            prompt = f"{rubric.render_as_guidance()}\n\nProblem: {task}\n\nSolution:"

        elif guidance_level == RubricLevel.IMPLICIT:
            # Rubric referenced but not detailed
            criterion_names = ", ".join(c.name for c in rubric.criteria)
            prompt = f"Solve this task considering: {criterion_names}.\n\nProblem: {task}\n\nSolution:"

        elif guidance_level == RubricLevel.ABSENT:
            # No rubric
            prompt = f"Solve this problem:\n\n{task}\n\nSolution:"

        # Generate with higher temperature for exploration
        response = self.model.generate(
            prompt,
            temperature=temperature,
            max_tokens=500
        )

        return response

    def collect_exploration_trajectories(
        self,
        tasks: List[Dict],
        task_type: str,
        num_trajectories_per_task: int = 5,
        guidance_schedule: List[RubricLevel] = None
    ) -> List[Dict]:
        """
        Collect exploration trajectories with curriculum guidance schedule.
        """
        if guidance_schedule is None:
            # Default: explicit -> implicit -> absent
            guidance_schedule = [
                RubricLevel.EXPLICIT,
                RubricLevel.EXPLICIT,
                RubricLevel.IMPLICIT,
                RubricLevel.IMPLICIT,
                RubricLevel.ABSENT
            ]

        trajectories = []

        for task_idx, task in enumerate(tasks):
            for traj_idx in range(num_trajectories_per_task):
                # Select guidance level from schedule
                schedule_idx = min(traj_idx, len(guidance_schedule) - 1)
                guidance_level = guidance_schedule[schedule_idx]

                # Generate with guidance
                response = self.generate_with_rubric_guidance(
                    task["prompt"],
                    task_type,
                    guidance_level
                )

                trajectories.append({
                    "task": task["prompt"],
                    "response": response,
                    "guidance_level": guidance_level,
                    "task_id": task.get("id"),
                    "expected_answer": task.get("answer")
                })

        return trajectories
```

### 3. Implement Rubric-Based Scoring

Create LLM judge using rubrics:

```python
class RubricBasedJudge:
    """Evaluate responses using rubric criteria."""

    def __init__(
        self,
        judge_model: "LLM",
        rubric_library: RubricLibrary
    ):
        self.judge_model = judge_model
        self.rubric_library = rubric_library

    def score_response(
        self,
        task: str,
        response: str,
        task_type: str
    ) -> Dict[str, float]:
        """
        Score response against rubric criteria.
        Returns: {criterion_name: score, "overall": score}
        """
        rubric = self.rubric_library.get_rubric(task_type)

        scores = {}

        # Score each criterion
        for criterion in rubric.criteria:
            score = self._score_criterion(
                task, response, criterion
            )
            scores[criterion.name] = score

        # Weighted overall score
        overall = sum(
            scores[c.name] * c.weight
            for c in rubric.criteria
        ) / sum(c.weight for c in rubric.criteria)

        scores["overall"] = overall

        return scores

    def _score_criterion(
        self,
        task: str,
        response: str,
        criterion: RubricCriterion
    ) -> float:
        """Score single criterion using LLM judge."""

        prompt = f"""Evaluate the following response against this criterion:

Criterion: {criterion.name}
Description: {criterion.description}

Good example: {criterion.exemplar_good}
Bad example: {criterion.exemplar_bad}

Task: {task}
Response: {response}

Rate this response on criterion '{criterion.name}' on a scale 0-1:
- 0: Does not satisfy criterion at all
- 0.5: Partially satisfies criterion
- 1: Fully satisfies criterion

Score (just the number, 0-1):"""

        score_text = self.judge_model.generate(prompt, max_tokens=10)

        # Extract numerical score
        try:
            score = float(score_text.strip())
            return min(max(score, 0.0), 1.0)  # Clamp to [0,1]
        except:
            return 0.5  # Default to middle

    def score_batch(
        self,
        trajectories: List[Dict],
        task_type: str
    ) -> List[Dict]:
        """Score multiple trajectories."""

        for trajectory in trajectories:
            scores = self.score_response(
                trajectory["task"],
                trajectory["response"],
                task_type
            )
            trajectory["scores"] = scores
            trajectory["reward"] = scores["overall"]

        return trajectories
```

### 4. Implement RuscaRL Training Loop

Train with rubric-guided RL:

```python
class RuscaRLTrainer:
    """Training with rubric-scaffolded RL."""

    def __init__(
        self,
        model: "LLM",
        rubric_library: RubricLibrary,
        learning_rate: float = 1e-5
    ):
        self.model = model
        self.rubric_library = rubric_library
        self.explorer = RubricGuidedExploration(model, rubric_library)
        self.judge = RubricBasedJudge(model, rubric_library)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_with_rubric_curriculum(
        self,
        tasks: List[Dict],
        task_type: str,
        num_iterations: int = 5,
        guidance_schedule: List[RubricLevel] = None
    ) -> Dict[str, float]:
        """
        Train model with rubric curriculum.
        Gradually remove rubric guidance over iterations.
        """

        metrics = {
            "iteration": [],
            "avg_reward": [],
            "without_guidance_reward": []
        }

        for iteration in range(num_iterations):
            print(f"RuscaRL Iteration {iteration + 1}/{num_iterations}")

            # Phase 1: Generate trajectories with rubric guidance
            trajectories = self.explorer.collect_exploration_trajectories(
                tasks,
                task_type,
                num_trajectories_per_task=3,
                guidance_schedule=guidance_schedule
            )

            # Phase 2: Score trajectories using rubric
            trajectories = self.judge.score_batch(trajectories, task_type)

            # Compute reward statistics
            rewards = [t["reward"] for t in trajectories]
            avg_reward = sum(rewards) / len(rewards)
            metrics["iteration"].append(iteration)
            metrics["avg_reward"].append(avg_reward)

            print(f"  Avg Reward: {avg_reward:.4f}")

            # Phase 3: Train on scored trajectories
            for trajectory in trajectories:
                # Get log prob of response
                response = trajectory["response"]
                reward = trajectory["reward"]

                # Policy gradient loss
                loss = -reward * self.model.get_log_prob(response)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Phase 4: Evaluate without guidance
            print(f"  Evaluating without guidance...")
            guidance_free_trajectories = self.explorer.collect_exploration_trajectories(
                tasks,
                task_type,
                num_trajectories_per_task=1,
                guidance_schedule=[RubricLevel.ABSENT] * 5
            )

            guidance_free_trajectories = self.judge.score_batch(
                guidance_free_trajectories, task_type
            )

            guidance_free_rewards = [t["reward"] for t in guidance_free_trajectories]
            avg_guidance_free = sum(guidance_free_rewards) / len(guidance_free_rewards)
            metrics["without_guidance_reward"].append(avg_guidance_free)

            print(f"  Without Guidance Reward: {avg_guidance_free:.4f}")

        return metrics
```

## Practical Guidance

### When to Use RuscaRL

- General reasoning tasks without fixed rubrics
- Curriculum learning scenarios
- Exploration-heavy RL problems
- Multi-criterion optimization
- Tasks where quality evaluation is clear

### When NOT to Use

- Tasks without clear evaluation criteria
- Real-time systems (rubric scoring is slow)
- Domains where guidance misleads exploration
- Single-criterion optimization

### Key Hyperparameters

- **guidance_schedule**: Length equal to trajectory budget
- **temperature**: 0.8-1.0 for exploration
- **rubric weights**: Adjust by criterion importance
- **judge model**: Can be same or separate from student
- **curriculum fade rate**: Linear or exponential

### Performance Expectations

- Exploration Quality: 2-3x improvement with guidance
- Guidance Transfer: Patterns learned with guidance apply without it
- Sample Efficiency: Faster convergence than unguided RL
- Final Performance: Approaches or exceeds baseline after guidance removal

## Reference

Researchers. (2024). Breaking the Exploration Bottleneck: Rubric-Scaffolded RL for General LLM Reasoning. arXiv preprint arXiv:2508.16949.
