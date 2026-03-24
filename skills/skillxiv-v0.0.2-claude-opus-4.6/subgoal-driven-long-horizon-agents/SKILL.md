---
name: subgoal-driven-long-horizon-agents
title: "A Subgoal-driven Framework for Improving Long-Horizon LLM Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.19685"
keywords: [Long-Horizon Planning, Reinforcement Learning, Hierarchical Decomposition, Agent Training]
description: "Improve long-horizon task success via subgoal decomposition and dense milestone-based rewards, dramatically outperforming sparse-reward RL and standard baselines."
---

# Subgoal-Driven Framework for Long-Horizon LLM Agents

Long-horizon tasks present two fundamental challenges for LLM agents. First, the agent can lose sight of the ultimate goal as intermediate steps accumulate, making suboptimal local decisions that push toward failure. Second, when rewards arrive only at the end (sparse rewards), the agent cannot learn which actions contribute to success—a credit assignment crisis.

This framework solves both through hierarchical planning with dense rewards. At the start of execution, decompose the task into explicit subgoals using an external planner. Then, during RL training, provide dense rewards whenever the agent reaches a subgoal (milestone). This transforms the sparse-reward problem into a sequence of easier intermediate problems, enabling dramatically faster learning.

## Core Concept

The framework combines two reinforcement learning innovations:

**Subgoal-Driven Online Planning:** Use an external planner to decompose complex objectives into concrete intermediate milestones. This gives the agent explicit waypoints—reducing goal drift and providing structure for exploration.

**Milestoning Reinforcement Learning Enhanced Agent (MiRA):** Replace sparse endpoint rewards with dense per-milestone rewards. When the agent reaches a milestone, it immediately receives positive reward feedback, enabling credit assignment without delayed signals.

Together, these enable learning on long-horizon tasks that would fail with standard RL training (0% success) to achieve 43%+ success rates.

## Architecture Overview

- **Task Planner**: External model that decomposes high-level objectives into subgoals
- **Milestone Tracker**: Monitors agent progress against planned subgoals
- **Dense Reward Generator**: Provides per-milestone rewards based on progress
- **Milestone-Conditioned Policy**: RL policy that learns to reach specific subgoals
- **Subgoal Validator**: Checks whether claimed milestone achievements are legitimate
- **Fallback Mechanism**: If agent gets stuck, replan or skip unreachable milestones

## Implementation Steps

### Step 1: Task Decomposition into Subgoals

Convert high-level objectives into concrete, verifiable subgoals.

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class Subgoal:
    """Represents an intermediate milestone in task execution."""
    index: int  # Order in sequence
    description: str  # What must be achieved
    preconditions: List[str]  # What must be true first
    success_criteria: str  # How to verify completion
    estimated_steps: int  # Rough # steps expected
    required_for_task: bool = True  # Skip if impossible?

    def to_prompt_segment(self) -> str:
        return f"""
Subgoal {self.index}: {self.description}
- Verify by: {self.success_criteria}
- Expected steps: ~{self.estimated_steps}
"""

class TaskPlanner:
    """
    Decomposes complex tasks into subgoals.
    Uses external LLM for planning.
    """

    def __init__(self, planning_model):
        self.planning_model = planning_model

    def decompose_task(self, task_description: str, num_subgoals: int = 5) - List[Subgoal]:
        """
        Break task into manageable subgoals.
        """
        planning_prompt = f"""
You are a task planning expert. Break this task into {num_subgoals} concrete subgoals.

Task: {task_description}

For each subgoal, provide:
1. Clear description of what to achieve
2. Success criteria (how to verify it's done)
3. Preconditions (what must happen first)
4. Estimated steps

Format as JSON array of subgoals.
"""

        response = self.planning_model.generate(planning_prompt)
        subgoals_data = self._parse_subgoal_json(response)

        subgoals = []
        for idx, sg_data in enumerate(subgoals_data):
            subgoal = Subgoal(
                index=idx,
                description=sg_data['description'],
                preconditions=sg_data.get('preconditions', []),
                success_criteria=sg_data['success_criteria'],
                estimated_steps=sg_data.get('estimated_steps', 10),
                required_for_task=True
            )
            subgoals.append(subgoal)

        return subgoals

    def _parse_subgoal_json(self, response: str) -> List[Dict]:
        """Extract JSON subgoal definitions from model response."""
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return []

    def replan_from_milestone(self, task: str, completed_subgoals: List[Subgoal],
                             current_state: Dict) -> Optional[List[Subgoal]]:
        """Replan if agent gets stuck or deviates significantly."""
        remaining_task = f"{task} (already completed: {', '.join([sg.description for sg in completed_subgoals])})"

        return self.decompose_task(remaining_task, num_subgoals=3)
```

### Step 2: Milestone Tracking and Verification

Monitor progress toward subgoals and verify legitimate achievement.

```python
class MilestoneTracker:
    """
    Tracks agent progress against planned subgoals.
    Verifies that claimed achievements are real.
    """

    def __init__(self, environment, verifier_model):
        self.environment = environment
        self.verifier_model = verifier_model
        self.completed_subgoals: List[Subgoal] = []
        self.current_subgoal_idx = 0

    def get_current_milestone(self, subgoals: List[Subgoal]) -> Optional[Subgoal]:
        """What is the agent currently working toward?"""
        if self.current_subgoal_idx < len(subgoals):
            return subgoals[self.current_subgoal_idx]
        return None

    def verify_milestone_achievement(self, subgoal: Subgoal,
                                     environment_state: Dict) -> bool:
        """
        Check if a claimed subgoal achievement is legitimate.
        Uses both environment observation + LLM verification.
        """

        # Method 1: Structured environment check
        env_verification = self._check_environment_state(subgoal, environment_state)

        # Method 2: LLM verification (for complex criteria)
        llm_verification = self._verify_with_llm(subgoal, environment_state)

        # Require both forms of agreement
        return env_verification and llm_verification

    def _check_environment_state(self, subgoal: Subgoal, state: Dict) -> bool:
        """Check if environment state satisfies milestone criteria."""
        criteria_parts = subgoal.success_criteria.split('and')

        for criterion in criteria_parts:
            criterion = criterion.strip().lower()

            # Simple pattern matching
            if 'found' in criterion and 'item' in state.get('inventory', {}):
                return True
            if 'reached' in criterion and state.get('location') == subgoal.description.split()[-1]:
                return True
            if 'completed' in criterion and state.get('task_completed'):
                return True

        return False

    def _verify_with_llm(self, subgoal: Subgoal, state: Dict) -> bool:
        """Use LLM to verify complex milestone achievements."""
        verification_prompt = f"""
Verify if this milestone has been achieved:

Milestone: {subgoal.description}
Success Criteria: {subgoal.success_criteria}

Current Environment State:
{json.dumps(state, indent=2)}

Is the milestone achieved? Respond with YES or NO only."""

        response = self.verifier_model.generate(verification_prompt).strip().upper()
        return 'YES' in response

    def update_progress(self, subgoal: Subgoal, environment_state: Dict) -> bool:
        """
        Check if agent has reached the current milestone.
        If verified, advance to next milestone.
        """
        if self.verify_milestone_achievement(subgoal, environment_state):
            self.completed_subgoals.append(subgoal)
            self.current_subgoal_idx += 1
            return True

        return False

    def get_progress_summary(self, subgoals: List[Subgoal]) -> Dict:
        """Return progress statistics."""
        return {
            'completed': len(self.completed_subgoals),
            'total': len(subgoals),
            'completion_percentage': 100 * len(self.completed_subgoals) / max(len(subgoals), 1),
            'current_milestone': subgoals[self.current_subgoal_idx].description if self.current_subgoal_idx < len(subgoals) else "Task Complete"
        }
```

### Step 3: Dense Reward Generation (MiRA)

Compute per-milestone rewards for effective RL training.

```python
import numpy as np

class MilestoneRewardGenerator:
    """
    Generates dense per-milestone rewards.
    Replaces sparse endpoint rewards with intermediate signals.
    """

    def __init__(self, milestone_tracker: MilestoneTracker):
        self.milestone_tracker = milestone_tracker

    def compute_reward(self, previous_progress: int, current_progress: int,
                       episode_step: int, max_steps: int,
                       subgoals: List[Subgoal]) -> float:
        """
        Compute reward for this step.

        Rewards are primarily given for reaching milestones,
        with small bonuses for efficiency.
        """

        reward = 0.0

        # Major reward: reaching a new milestone
        if current_progress > previous_progress:
            # Milestone reward proportional to sequence position (later milestones worth more)
            milestone_bonus = 10.0 * (current_progress / max(len(subgoals), 1))
            reward += milestone_bonus

            # Efficiency bonus: reward reaching milestone quickly
            steps_available = subgoals[current_progress - 1].estimated_steps * 2
            if episode_step < steps_available:
                efficiency_bonus = 2.0 * (1.0 - episode_step / steps_available)
                reward += efficiency_bonus

        # Small per-step cost: encourage efficiency
        step_cost = -0.01

        # Penalty if running out of steps
        if episode_step > max_steps * 0.9:
            timeout_penalty = -0.5
            reward += timeout_penalty

        return reward + step_cost

    def compute_episode_return(self, trajectory: List[Dict], subgoals: List[Subgoal]) -> float:
        """Compute total return for an episode (for value estimation)."""
        total_return = 0.0
        previous_progress = 0

        for step_idx, step_data in enumerate(trajectory):
            current_progress = step_data['milestone_progress']
            reward = self.compute_reward(
                previous_progress, current_progress,
                step_idx, len(trajectory), subgoals
            )
            total_return += (0.99 ** step_idx) * reward  # Discounted
            previous_progress = current_progress

        # Final bonus for completing task
        if previous_progress == len(subgoals):
            total_return += 100.0

        return total_return
```

### Step 4: Milestone-Conditioned Policy Training

Train the RL policy to reach specific subgoals.

```python
import torch
import torch.nn as nn
from torch.optim import Adam

class MilestoneConditionedPolicy:
    """
    RL policy that learns to reach specific milestones.
    Conditioned on: current state + target milestone.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + 64, hidden_dim),  # +64 for milestone embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.value_net = nn.Sequential(
            nn.Linear(state_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.milestone_encoder = nn.Embedding(10, 64)  # Encode milestone index
        self.optimizer = Adam(
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=1e-4
        )

    def forward(self, state: torch.Tensor, milestone_idx: int) -> torch.Tensor:
        """Compute action distribution conditioned on target milestone."""
        milestone_embedding = self.milestone_encoder(torch.tensor(milestone_idx))
        state_milestone = torch.cat([state, milestone_embedding], dim=-1)

        logits = self.policy_net(state_milestone)
        action_probs = torch.softmax(logits, dim=-1)

        return action_probs

    def estimate_value(self, state: torch.Tensor, milestone_idx: int) -> torch.Tensor:
        """Estimate value of reaching target milestone from current state."""
        milestone_embedding = self.milestone_encoder(torch.tensor(milestone_idx))
        state_milestone = torch.cat([state, milestone_embedding], dim=-1)

        value = self.value_net(state_milestone)
        return value

    def train_on_trajectory(self, trajectory: List[Dict], subgoals: List[Subgoal],
                           rewards: List[float]):
        """
        Train policy via policy gradient with baseline.
        trajectory: list of (state, action, next_state) tuples
        rewards: dense milestone-based rewards
        """

        # Compute advantages
        values = []
        for step_data in trajectory:
            state = torch.tensor(step_data['state'], dtype=torch.float32)
            milestone_idx = step_data['target_milestone']
            value = self.estimate_value(state, milestone_idx)
            values.append(value.item())

        # TD residuals
        advantages = []
        gae = 0
        for t in reversed(range(len(trajectory))):
            td_error = rewards[t] + 0.99 * values[t + 1] - values[t] if t < len(trajectory) - 1 else rewards[t] - values[t]
            gae = td_error + 0.95 * 0.99 * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = 0.0
        for t in range(len(trajectory)):
            state = torch.tensor(trajectory[t]['state'], dtype=torch.float32)
            action = trajectory[t]['action']
            milestone_idx = trajectory[t]['target_milestone']

            action_probs = self.forward(state.unsqueeze(0), milestone_idx)
            log_prob = torch.log(action_probs[0, action] + 1e-8)
            policy_loss = policy_loss - log_prob * advantages[t]

        # Value loss
        value_loss = 0.0
        for t in range(len(trajectory)):
            state = torch.tensor(trajectory[t]['state'], dtype=torch.float32)
            milestone_idx = trajectory[t]['target_milestone']
            value = self.estimate_value(state.unsqueeze(0), milestone_idx)
            target_value = torch.tensor(values[t], dtype=torch.float32)
            value_loss = value_loss + (value - target_value).pow(2)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()

        return total_loss.item()
```

### Step 5: Main Training Loop

Integrate planning, tracking, and RL training.

```python
class SubgoalDrivenLongHorizonAgent:
    """Complete agent with subgoal planning and milestone-based RL."""

    def __init__(self, environment, planning_model, verifier_model,
                 rl_policy, state_dim: int):
        self.environment = environment
        self.task_planner = TaskPlanner(planning_model)
        self.milestone_tracker = MilestoneTracker(environment, verifier_model)
        self.reward_generator = MilestoneRewardGenerator(self.milestone_tracker)
        self.rl_policy = rl_policy

    def train(self, task_description: str, num_episodes: int = 100):
        """Train agent on task with subgoal decomposition and dense rewards."""

        # Initial task decomposition
        subgoals = self.task_planner.decompose_task(task_description)
        print(f"Decomposed task into {len(subgoals)} subgoals")

        episode_returns = []

        for episode in range(num_episodes):
            # Reset
            state = self.environment.reset()
            self.milestone_tracker = MilestoneTracker(self.environment, self.verifier_model)
            trajectory = []
            rewards_list = []
            previous_progress = 0

            for step in range(500):  # Max steps per episode
                # Get current milestone
                current_milestone = self.milestone_tracker.get_current_milestone(subgoals)
                if not current_milestone:
                    break  # Task complete

                # Policy selects action (conditioned on target milestone)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                milestone_idx = current_milestone.index
                action_probs = self.rl_policy.forward(state_tensor, milestone_idx)
                action = torch.multinomial(action_probs, 1).item()

                # Execute action
                next_state, _ = self.environment.step(action)

                # Check milestone progress
                progress_advanced = self.milestone_tracker.update_progress(
                    current_milestone, self.environment.get_state()
                )
                current_progress = self.milestone_tracker.current_subgoal_idx

                # Dense reward based on milestone progress
                reward = self.reward_generator.compute_reward(
                    previous_progress, current_progress, step, 500, subgoals
                )

                # Record trajectory
                trajectory.append({
                    'state': state,
                    'action': action,
                    'next_state': next_state,
                    'target_milestone': milestone_idx
                })
                rewards_list.append(reward)

                previous_progress = current_progress
                state = next_state

            # Train policy on trajectory
            if trajectory:
                loss = self.rl_policy.train_on_trajectory(trajectory, subgoals, rewards_list)

            # Evaluate progress
            episode_return = self.reward_generator.compute_episode_return(trajectory, subgoals)
            episode_returns.append(episode_return)

            if episode % 10 == 0:
                progress = self.milestone_tracker.get_progress_summary(subgoals)
                print(f"Episode {episode}: Return={episode_return:.2f}, Progress={progress['completion_percentage']:.1f}%")

            # Replan if stuck
            if previous_progress == 0 and episode > 20:
                print("Replanning...")
                subgoals = self.task_planner.replan_from_milestone(
                    task_description, [], self.environment.get_state()
                )

        return episode_returns
```

## Practical Guidance

**Hyperparameters:**
- Number of subgoals: 4-7 (task dependent; too many fragments focus)
- Milestone bonus: 10x baseline per-step reward (makes milestones the dominant signal)
- Efficiency bonus multiplier: 2.0-5.0 (encourages quick milestone reaching)
- Policy learning rate: 1e-4 to 1e-3 (stable for milestone-conditioned learning)
- GAE parameter: λ=0.95 (balance bias-variance in advantage estimation)

**When to Use:**
- Long-horizon tasks (>50 steps) where credit assignment is hard
- Environments where intermediate milestones are identifiable
- Scenarios with high penalty for task failure (structure reduces random exploration)
- Training agent on single long task (not multi-task)

**When NOT To Use:**
- Short-horizon tasks (<10 steps) where subgoal overhead dominates
- Environments where milestones are ambiguous or hard to verify
- Tasks requiring rapid exploration (dense rewards can narrow behavior too much)
- Online learning where replanning is prohibitively expensive

**Pitfalls:**
- Subgoal specification critical: bad milestones mislead the agent; validate with model predictions
- Verification bugs: if milestone detection is wrong, rewards become noise
- Reward shaping can harm: if endpoint rewards still matter, milestone rewards may conflict
- Replanning cost: replanning too frequently disrupts learning; do it sparingly

## Reference

Paper: [arxiv.org/abs/2603.19685](https://arxiv.org/abs/2603.19685)
