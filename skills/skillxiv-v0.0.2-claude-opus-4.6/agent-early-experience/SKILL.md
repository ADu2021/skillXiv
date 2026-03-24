---
name: agent-early-experience
title: "Agent Learning via Early Experience: Implicit World Modeling and Self-Reflection"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.08558
keywords: [agent-learning, early-experience, world-modeling, self-reflection, imitation-learning]
description: "Bridge imitation learning and experience-driven RL by collecting state-based supervision from agents' own actions. Trigger: improve agent generalization when expert demonstrations are limited and environments lack dense rewards."
---

# Agent Learning via Early Experience

## Core Concept

Early Experience Learning addresses a fundamental challenge in agent training: expert demonstrations are limited and don't cover environment diversity, while many environments (websites, interactive systems) lack verifiable rewards. The approach collects supervision from the agent's own exploratory actions—using resulting states as training targets. This bridges imitation learning and RL by grounding policy in environmental dynamics through implicit world modeling and self-reflection.

The key insight: An agent's own experience (state transitions) is stronger supervision than limited expert demonstrations, and agent reflection on suboptimal actions drives improvement.

## Architecture Overview

- **Early Experience Collection**: Gather state transitions from agent's own exploratory actions
- **Implicit World Modeling**: Learn environmental dynamics without explicit model
- **Self-Reflection Mechanism**: Agent evaluates its own actions and learns corrections
- **Two Learning Strategies**: World modeling vs. self-reflection (complementary)
- **Foundation for RL**: Provides strong initialization for downstream RL phases

## Implementation Steps

### 1. Define the Experience Collection Protocol

Agents explore environments while recording their actions and resulting states.

```python
class EarlyExperienceCollector:
    """
    Collect state-based supervision from agent's own exploratory actions.
    """
    def __init__(self, agent_model, environment, max_steps=50):
        self.agent = agent_model
        self.env = environment
        self.max_steps = max_steps
        self.experiences = []

    def collect_trajectory(self, task_description):
        """
        Agent explores a task, recording (state, action, result_state).

        Args:
            task_description: Human-readable task prompt

        Returns:
            Trajectory with experience tuples
        """
        trajectory = {
            "task": task_description,
            "states": [],
            "actions": [],
            "result_states": [],
            "observations": []
        }

        # Initial state from environment
        state = self.env.reset(task_description)
        trajectory["states"].append(state)

        for step in range(self.max_steps):
            # Agent generates action given current state/observation
            observation = self.env.observe()
            trajectory["observations"].append(observation)

            action_output = self.agent.generate(
                f"Task: {task_description}\n"
                f"Current observation: {observation}\n"
                f"Your action: ",
                max_tokens=100
            )

            action = parse_action(action_output)
            trajectory["actions"].append(action)

            # Execute action in environment
            try:
                result_state = self.env.step(action)
                trajectory["result_states"].append(result_state)

                # Check if task completed
                if self.env.is_completed():
                    trajectory["completed"] = True
                    break
            except Exception as e:
                trajectory["result_states"].append({"error": str(e)})
                break

        trajectory["num_steps"] = len(trajectory["actions"])
        return trajectory

    def collect_batch(self, task_list, num_trajectories_per_task=3):
        """
        Collect multiple trajectories across task distribution.
        """
        all_trajectories = []

        for task in task_list:
            for _ in range(num_trajectories_per_task):
                trajectory = self.collect_trajectory(task)
                all_trajectories.append(trajectory)

        return all_trajectories
```

### 2. Implement Implicit World Modeling

Train agent to predict result states given actions, grounding it in environment dynamics.

```python
class ImplicitWorldModelTrainer:
    """
    Train world model implicitly: predict state transitions.
    """
    def __init__(self, agent_model):
        self.agent = agent_model

    def create_world_modeling_examples(self, trajectories):
        """
        Convert trajectories into (state, action, predicted_state) tuples.
        """
        training_examples = []

        for traj in trajectories:
            for step_idx in range(len(traj["actions"])):
                current_state = traj["states"][step_idx]
                action = traj["actions"][step_idx]
                result_state = traj["result_states"][step_idx]

                example = {
                    "input": f"State: {format_state(current_state)}\n"
                             f"Action: {action}\n"
                             f"Resulting state: ",
                    "target": format_state(result_state)
                }
                training_examples.append(example)

        return training_examples

    def train_world_model(self, examples, num_epochs=5):
        """
        Fine-tune agent to predict state transitions.
        This implicitly teaches environmental dynamics.
        """
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            epoch_loss = 0

            for example in examples:
                # Generate prediction
                output = self.agent.generate(
                    example["input"],
                    max_tokens=200
                )

                # Compute loss vs. ground truth state
                loss = compute_state_matching_loss(
                    output,
                    example["target"]
                )

                epoch_loss += loss.item()

                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / len(examples)
            print(f"Epoch {epoch}: world model loss = {avg_loss:.4f}")

        return self.agent
```

### 3. Implement Self-Reflection Mechanism

Agents evaluate their own actions and learn from mistakes without external feedback.

```python
class SelfReflectionTrainer:
    """
    Train agents to reflect on suboptimal actions and improve.
    """
    def __init__(self, agent_model):
        self.agent = agent_model

    def create_reflection_examples(self, trajectories, success_threshold=0.8):
        """
        Identify suboptimal actions and create reflection examples.
        """
        reflection_examples = []

        for traj in trajectories:
            # Assess if trajectory was successful
            task_success = traj.get("completed", False)
            num_steps = traj["num_steps"]
            efficiency = 1.0 / (num_steps + 1)  # Penalize long trajectories

            success_score = 1.0 if task_success else 0.0
            overall_score = 0.7 * success_score + 0.3 * efficiency

            # If not perfect, create reflection example
            if overall_score < success_threshold:
                # Identify which actions were problematic
                for step_idx in range(len(traj["actions"])):
                    action = traj["actions"][step_idx]
                    state = traj["states"][step_idx]
                    result = traj["result_states"][step_idx]

                    # Check if action led toward goal
                    progress = self.evaluate_action_progress(
                        action,
                        state,
                        result,
                        traj["task"]
                    )

                    if progress < 0.5:  # Action was suboptimal
                        # Create reflection prompt
                        reflection_example = {
                            "input": f"Task: {traj['task']}\n"
                                    f"Current state: {format_state(state)}\n"
                                    f"Action taken: {action}\n"
                                    f"Result: {format_state(result)}\n"
                                    f"Reflection (what went wrong): ",
                            "target": generate_reflection(action, result)
                        }
                        reflection_examples.append(reflection_example)

        return reflection_examples

    def train_self_reflection(self, examples, num_epochs=5):
        """
        Train agent to articulate what went wrong with its actions.
        """
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            epoch_loss = 0

            for example in examples:
                # Generate reflection
                output = self.agent.generate(
                    example["input"],
                    max_tokens=100
                )

                # Loss: how well does reflection match ground truth
                loss = compute_reflection_loss(
                    output,
                    example["target"]
                )

                epoch_loss += loss.item()

                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = epoch_loss / len(examples)
            print(f"Epoch {epoch}: reflection loss = {avg_loss:.4f}")

        return self.agent

    def evaluate_action_progress(self, action, state, result, task):
        """
        Measure if action moved agent closer to task goal.
        """
        state_embedding = encode_state(state)
        result_embedding = encode_state(result)
        task_embedding = encode_task(task)

        # Cosine similarity to goal
        similarity_improvement = (
            cosine_similarity(result_embedding, task_embedding) -
            cosine_similarity(state_embedding, task_embedding)
        )

        return max(0.0, similarity_improvement)
```

### 4. Combined Early Experience Training

Integrate world modeling and self-reflection into coherent training pipeline.

```python
def train_with_early_experience(agent, environment, task_list, config):
    """
    Full early experience training pipeline.
    """
    # Phase 1: Collect early experiences
    print("Phase 1: Collecting early experiences")
    collector = EarlyExperienceCollector(agent, environment)
    trajectories = collector.collect_batch(
        task_list,
        num_trajectories_per_task=3
    )
    print(f"Collected {len(trajectories)} trajectories")

    # Phase 2: World modeling training
    print("\nPhase 2: Implicit world modeling")
    world_trainer = ImplicitWorldModelTrainer(agent)
    wm_examples = world_trainer.create_world_modeling_examples(trajectories)
    agent = world_trainer.train_world_model(wm_examples, num_epochs=5)

    # Phase 3: Self-reflection training
    print("\nPhase 3: Self-reflection learning")
    reflection_trainer = SelfReflectionTrainer(agent)
    ref_examples = reflection_trainer.create_reflection_examples(trajectories)
    agent = reflection_trainer.train_self_reflection(
        ref_examples,
        num_epochs=5
    )

    return agent
```

### 5. Evaluation and RL Foundation

Assess early experience training and prepare for downstream RL.

```python
def evaluate_early_experience_agent(agent, benchmark_tasks):
    """
    Test agent on diverse tasks post-early-experience training.
    """
    success_rates = {}
    generalization_scores = []

    for task in benchmark_tasks:
        successes = 0
        trials = 5

        for _ in range(trials):
            trajectory = collect_single_trajectory(agent, task)
            if trajectory.get("completed"):
                successes += 1

        task_success_rate = successes / trials * 100
        success_rates[task] = task_success_rate

        # Measure generalization: novel tasks similar to training
        generalization = measure_transfer_learning(agent, task)
        generalization_scores.append(generalization)

    print(f"Average success rate: {np.mean(list(success_rates.values())):.1f}%")
    print(f"Generalization: {np.mean(generalization_scores):.2f}")

    return {
        "success_rates": success_rates,
        "generalization": np.mean(generalization_scores)
    }
```

## Practical Guidance

**Hyperparameters:**
- **Trajectories per task**: 3-5 (balance diversity vs. compute)
- **Max steps per trajectory**: 20-50 (depends on task complexity)
- **World modeling epochs**: 5 (avoid overfitting)
- **Self-reflection epochs**: 5
- **Learning rate**: 1e-5 (conservative for LLMs)

**When to Use:**
- Training agents for web environments, APIs, or interactive tasks
- Limited expert demonstrations available
- Want to improve generalization with self-supervised learning
- As initialization before downstream RL

**When NOT to Use:**
- High-reward environments (standard RLHF sufficient)
- Agents require immediate deployment (training overhead)
- Tasks with no natural state representation
- Single-turn interactions without state progression

## Reference

[Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) — arXiv:2510.08558
