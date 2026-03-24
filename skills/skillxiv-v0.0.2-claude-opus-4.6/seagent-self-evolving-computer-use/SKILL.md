---
name: seagent-self-evolving-computer-use
title: SEAgent - Self-Evolving Computer Use Agent
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.04700
keywords: [computer-use, reinforcement-learning, self-improvement, curriculum-learning]
description: "Vision-based computer use agent that self-improves through experiential learning, curriculum generation, and reward-based RL on diverse software."
---

## SEAgent: Self-Evolving Computer Use Through Experience

SEAgent enables autonomous agents to master unfamiliar software through trial-and-error learning without human annotations. The framework combines curriculum learning (progressively harder tasks), dual RL (success rewards + failure penalties), and specialist-to-generalist knowledge distillation, achieving 34.5% success on professional software applications.

### Core Concept

Learning to use computer interfaces requires understanding visual elements, interpreting their meanings, and planning appropriate actions. Rather than human-annotated demonstrations, SEAgent learns from raw experience: attempt tasks, receive feedback, refine understanding. A curriculum generator creates progressively harder challenges; a world state model provides step-level rewards; specialists trained on individual software distill knowledge into a generalist agent.

### Architecture Overview

- **World State Model**: Vision-language model providing step-level reward signals by evaluating full action trajectories
- **Curriculum Generator**: LLM-based system maintaining "software guidebook" and generating increasingly complex tasks
- **Actor Model**: Policy network (initialized from UI-TARS) refined via RL
- **Specialist-to-Generalist**: Train per-software specialists, distill into multi-software generalist
- **Dual RL**: Group Relative Policy Optimization (GRPO) for successes; adversarial loss for failures

### Implementation Steps

**Step 1: Implement World State Model (Reward Judge)**

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from PIL import Image

class WorldStateModel(nn.Module):
    """
    Vision-language model that evaluates action trajectories.
    Provides step-level rewards based on progress evaluation.
    """

    def __init__(self, vision_model_name: str = 'openai/clip-vit-base-patch32',
                 language_model_name: str = 'gpt2'):
        super().__init__()

        # Vision encoder
        self.vision_model = load_vision_model(vision_model_name)

        # Language model for reasoning
        self.language_model = load_language_model(language_model_name)

        # Reward head
        self.reward_head = nn.Linear(768, 1)  # Output scalar reward

    def forward(self, trajectory: List[Tuple[Image.Image, str]]) -> List[float]:
        """
        Evaluate trajectory and produce step-level rewards.

        Args:
            trajectory: List of (screenshot, action_text) pairs

        Returns:
            step_rewards: List of rewards for each action
        """
        step_rewards = []

        # Process each step
        for t, (screenshot, action) in enumerate(trajectory):
            # Encode screenshot with vision model
            image_features = self.vision_model.encode_image(screenshot)

            # Encode action text
            action_features = self.vision_model.encode_text(f"Action: {action}")

            # Evaluate this step's progress
            # Check: Did action advance toward goal?
            goal_progress = self._assess_progress(screenshot, action)

            # Check: Is action valid for current screen?
            action_validity = self._assess_validity(image_features, action_features)

            # Combine signals
            step_reward = 0.7 * goal_progress + 0.3 * action_validity
            step_rewards.append(step_reward)

        return step_rewards

    def _assess_progress(self, screenshot: Image.Image, action: str) -> float:
        """Estimate if action advanced toward goal."""
        # Prompt: "Does this action advance the task?"
        prompt = f"""Screenshot of software with task description.
Action taken: {action}

Did this action make progress toward completing the task? (0-1)"""

        # Vision-language reasoning
        reasoning = self.language_model.generate(prompt, max_tokens=10)
        score = self._parse_score(reasoning)
        return score

    def _assess_validity(self, image_features, action_features) -> float:
        """Check if action is valid for this screen state."""
        # High cosine similarity = action appropriate for screen
        similarity = torch.nn.functional.cosine_similarity(
            image_features.unsqueeze(0),
            action_features.unsqueeze(0)
        ).item()

        return (similarity + 1) / 2  # Normalize to [0, 1]

    def _parse_score(self, text: str) -> float:
        """Extract numeric score from model output."""
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1)) / 100
        return 0.5
```

**Step 2: Implement Curriculum Generator**

```python
class CurriculumGenerator:
    """
    Dynamically generate progressively harder tasks.
    Maintains "software guidebook" of available operations.
    """

    def __init__(self, language_model):
        self.llm = language_model
        self.software_guidebooks = {}
        self.task_difficulty = {}

    def build_guidebook(self, software_name: str, interface_screenshots: List[Image.Image]):
        """
        Extract available operations from software interface.
        Store for task generation.
        """
        operations = []

        for screenshot in interface_screenshots:
            # Analyze screenshot with vision model
            prompt = f"""Software: {software_name}

Analyze this screenshot and list all interactive elements visible.
What actions can the user take?"""

            elements = self.llm.generate(prompt, max_tokens=500)
            operations.extend(self._parse_elements(elements))

        self.software_guidebooks[software_name] = {
            'operations': operations,
            'difficulty_distribution': {}
        }

    def generate_task(self, software_name: str, difficulty: int = 5) -> str:
        """
        Generate a task for given software at specified difficulty.
        Difficulty: 1-10 (1=simple, 10=complex).
        """
        guidebook = self.software_guidebooks.get(software_name, {})
        operations = guidebook.get('operations', [])

        # Use operations to ground task generation
        operations_str = ', '.join(operations[:5])

        prompt = f"""Software: {software_name}
Available operations: {operations_str}
Task difficulty: {difficulty}/10

Generate a single specific task for a user to complete.
Make it realistic and achievable but at the specified difficulty level.

Task:"""

        task = self.llm.generate(prompt, max_tokens=100)
        return task.strip()

    def update_guidebook_from_feedback(self, software_name: str, trajectory: List[Dict],
                                      success: bool):
        """
        Update guidebook based on learning from trajectories.
        Track which operations are most useful.
        """
        if not success:
            # Failed trajectory: maybe operations weren't applicable
            pass

        # Track operation success rates
        guidebook = self.software_guidebooks[software_name]
        if 'operation_success_rates' not in guidebook:
            guidebook['operation_success_rates'] = {}

        for step in trajectory:
            action = step.get('action', '')
            # Update success rate for this action type
            if action not in guidebook['operation_success_rates']:
                guidebook['operation_success_rates'][action] = {'successes': 0, 'attempts': 0}

            guidebook['operation_success_rates'][action]['attempts'] += 1
            if success:
                guidebook['operation_success_rates'][action]['successes'] += 1

    def _parse_elements(self, text: str) -> List[str]:
        """Extract interactive elements from model output."""
        lines = text.split('\n')
        elements = [line.strip('- ') for line in lines if line.strip().startswith('-')]
        return elements
```

**Step 3: Implement Dual RL (GRPO + Adversarial)**

```python
class DualRL:
    """
    Train via Group Relative Policy Optimization (GRPO) for successes
    and adversarial imitation loss for failures.
    """

    def __init__(self, actor_model):
        self.actor = actor_model
        self.optimizer = torch.optim.Adam(actor_model.parameters(), lr=1e-5)

    def compute_grpo_loss(self, trajectory: List[Dict], trajectory_reward: float):
        """
        Group Relative Policy Optimization: reward successful actions.
        Treats trajectory as a group; relative ranking within group.
        """
        action_logps = []

        for step in trajectory:
            action = step['action']
            screen_features = step['screen_features']

            # Get log probability of action
            logp = self.actor.compute_action_logp(screen_features, action)
            action_logps.append(logp)

        # GRPO: advantage = trajectory_reward - baseline
        baseline = trajectory_reward * 0.9  # Conservative baseline
        advantage = trajectory_reward - baseline

        # Policy gradient: sum(log_pi * advantage)
        grpo_loss = -torch.sum(torch.stack(action_logps)) * advantage

        return grpo_loss

    def compute_adversarial_loss(self, failed_trajectory: List[Dict],
                                successful_trajectory: List[Dict]):
        """
        Penalize failure patterns through adversarial imitation.
        Learn to diverge from failure trajectories.
        """
        failed_logps = []
        success_logps = []

        for step in failed_trajectory:
            action = step['action']
            screen_features = step['screen_features']
            logp = self.actor.compute_action_logp(screen_features, action)
            failed_logps.append(logp)

        for step in successful_trajectory:
            action = step['action']
            screen_features = step['screen_features']
            logp = self.actor.compute_action_logp(screen_features, action)
            success_logps.append(logp)

        # Loss: maximize divergence between failure and success distributions
        # DKL(success || failure) = sum(log_success - log_failure)
        adversarial_loss = -torch.sum(torch.stack(success_logps)) + torch.sum(torch.stack(failed_logps))

        return adversarial_loss

    def train_step(self, successful_trajectories: List[List[Dict]],
                  failed_trajectories: List[List[Dict]]):
        """
        One training step combining GRPO and adversarial loss.
        """
        total_loss = 0.0

        # GRPO on successes
        for trajectory in successful_trajectories:
            reward = 1.0  # Successful
            loss = self.compute_grpo_loss(trajectory, reward)
            total_loss += loss

        # Adversarial on failures paired with successes
        for failed_traj, success_traj in zip(failed_trajectories, successful_trajectories):
            adv_loss = self.compute_adversarial_loss(failed_traj, success_traj)
            total_loss += 0.5 * adv_loss  # Weight adversarial less

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

**Step 4: Specialist-to-Generalist Distillation**

```python
class SpecialistToGeneralist:
    """
    Train specialists on individual software, distill to generalist.
    """

    def __init__(self):
        self.specialists = {}  # software_name -> model
        self.generalist = None

    def train_specialist(self, software_name: str, trajectories: List[List[Dict]],
                        num_epochs: int = 10):
        """Train specialist agent for one software."""
        specialist_model = create_model()
        specialist_rl = DualRL(specialist_model)

        for epoch in range(num_epochs):
            # Sample trajectories
            successful = [t for t in trajectories if t[-1]['success']]
            failed = [t for t in trajectories if not t[-1]['success']]

            loss = specialist_rl.train_step(successful, failed[:len(successful)])

            if epoch % 2 == 0:
                print(f"Specialist {software_name}, epoch {epoch}: loss={loss:.3f}")

        self.specialists[software_name] = specialist_model

    def distill_to_generalist(self, num_epochs: int = 5):
        """
        Distill knowledge from all specialists to single generalist.
        """
        # Initialize generalist
        self.generalist = create_model()

        # KL divergence loss: minimize divergence between specialist and generalist
        for epoch in range(num_epochs):
            for software_name, specialist in self.specialists.items():
                # Sample task for this software
                task = f"Interact with {software_name}"

                # Get actions from specialist
                specialist_actions = specialist.generate_actions(task, num_actions=5)

                # Train generalist to match specialist
                for action in specialist_actions:
                    generalist_logp = self.generalist.compute_action_logp(task, action)
                    specialist_logp = specialist.compute_action_logp(task, action)

                    # KL divergence (reverse)
                    kl_loss = torch.nn.functional.kl_div(
                        generalist_logp.exp(),
                        specialist_logp.exp(),
                        reduction='batchmean'
                    )

                    kl_loss.backward()

                self.generalist.optimizer.step()

    def evaluate_generalist(self, test_tasks: Dict[str, List[str]]) -> Dict:
        """Evaluate generalist on multiple software."""
        results = {}

        for software, tasks in test_tasks.items():
            successes = 0
            for task in tasks:
                success = self.generalist.attempt_task(software, task)
                if success:
                    successes += 1

            results[software] = successes / len(tasks)

        return results
```

**Step 5: Full Self-Improving Loop**

```python
def self_evolving_agent_training(initial_software_list: List[str],
                                num_rounds: int = 5):
    """
    Complete self-improving loop:
    1. Train specialist per software
    2. Distill to generalist
    3. Evaluate and expand
    """

    curriculum = CurriculumGenerator(llm_model)
    specialist_distiller = SpecialistToGeneralist()

    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num} ===")

        # Round: Train specialists on each software
        for software in initial_software_list:
            print(f"Training specialist for {software}...")

            # Generate tasks via curriculum
            tasks = [curriculum.generate_task(software, difficulty=i)
                    for i in range(1, 6)]  # Difficulties 1-5

            # Execute tasks and collect trajectories
            trajectories = []
            for task in tasks:
                trajectory = run_agent_on_task(software, task)
                trajectories.append(trajectory)

            # Train specialist
            specialist_distiller.train_specialist(software, trajectories)

            # Update curriculum guidebook
            for trajectory in trajectories:
                success = trajectory[-1].get('success', False)
                curriculum.update_guidebook_from_feedback(software, trajectory, success)

        # Distill specialists to generalist
        print("Distilling specialists to generalist...")
        specialist_distiller.distill_to_generalist()

        # Evaluate generalist on harder tasks
        print("Evaluating generalist...")
        test_tasks = {software: [curriculum.generate_task(software, difficulty=8)]
                     for software in initial_software_list}

        results = specialist_distiller.evaluate_generalist(test_tasks)

        for software, success_rate in results.items():
            print(f"  {software}: {success_rate:.1%}")

    return specialist_distiller.generalist
```

### Practical Guidance

**When to Use:**
- Computer use automation on unfamiliar software
- Scenarios with diverse interfaces requiring flexible learning
- Applications where human demonstrations are expensive
- Long-horizon tasks (multi-step interactions)

**When NOT to Use:**
- Real-time systems (RL training is slow)
- Highly specialized software with few users
- Scenarios requiring formal correctness guarantees
- Systems where mistakes have high consequences

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `specialist_epochs` | 10 | More training per specialist for deeper learning |
| `curriculum_difficulty_range` | 1-10 | Range of task difficulties generated |
| `success_weight_grpo` | 1.0 | Weight of successful trajectory rewards |
| `failure_weight_adversarial` | 0.5 | Weight of failure pattern penalties |

### Reference

**Paper**: SEAgent: Self-Evolving Computer Use Agent (2508.04700)
- 34.5% success on OSWorld professional software (from 11.3% baseline UI-TARS)
- Specialist-to-generalist approach outperforms direct multi-software training
- World State Model provides 71.6% precision trajectory evaluation
