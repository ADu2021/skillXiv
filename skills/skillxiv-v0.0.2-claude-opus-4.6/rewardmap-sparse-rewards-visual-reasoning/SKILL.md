---
name: rewardmap-sparse-rewards-visual-reasoning
title: "RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.02240"
keywords: [reward-design, visual-reasoning, sparse-rewards, multi-stage-RL, spatial-reasoning]
description: "Improve multimodal LLMs on fine-grained visual reasoning tasks (e.g., reading transit maps) by decomposing training into stages: basic perception (VQA) -> simple reasoning -> complex spatial reasoning. Incorporates 'detail rewards' for intermediate visual understanding, bootstrapping models from simple to complex tasks while addressing sparse reward challenges."
---

# RewardMap: Bootstrapping Visual Reasoning via Staged Reward Design

Fine-grained visual reasoning tasks—reading complex diagrams, reasoning about spatial relationships, extracting details from images—are notoriously hard for multimodal LLMs. The challenge is that end-to-end RL on these tasks produces sparse rewards: the model either completely solves a transit map puzzle or gets zero reward. This forces the model to learn everything simultaneously, which is inefficient.

RewardMap decomposes this into stages: start with simple perception tasks (answering basic questions about images), advance to spatial reasoning, and finish with complex multi-step tasks. Each stage includes intermediate "detail rewards" that provide richer supervision, turning sparse binary rewards into dense signals that guide learning.

## Core Concept

RewardMap's multi-stage curriculum uses three reward types:

1. **Perception rewards**: VQA-style questions about visual details (e.g., "What color is this box?") Score: 1 if correct.
2. **Intermediate rewards**: Partial reasoning steps (e.g., "Did you correctly identify the starting point?") Score: 0-1 based on intermediate correctness.
3. **Task rewards**: Full problem completion (e.g., "Did you find the correct transit route?") Score: 1 if task solved.

By training stage-by-stage with appropriate reward signals, the model learns perception first, then spatial reasoning, then complex multi-step logic—building capabilities progressively rather than learning all at once.

## Architecture Overview

- **Stage 1 (Perception)**: Simple VQA on image details, binary rewards
- **Stage 2 (Basic Reasoning)**: Spatial understanding (which object is left of X?), detail rewards
- **Stage 3 (Complex Reasoning)**: Full task completion (find route from A to B), combined rewards
- **Reward module**: Computes detail, intermediate, and task-level rewards
- **Data curator**: ReasonMap-Plus dataset with annotated intermediate steps

## Implementation Steps

Start by building the perception stage with VQA rewards:

```python
import torch
import torch.nn.functional as F

class VQARewardComputer:
    """
    Compute perception-level rewards from visual question answering.
    """
    def __init__(self, vqa_model):
        self.vqa_model = vqa_model  # Pretrained VQA answerer

    def compute_perception_reward(self, image, predicted_answer, question):
        """
        Score predicted answer against ground truth using VQA.

        Args:
            image: Visual input (PIL or tensor)
            predicted_answer: Model's predicted answer (text)
            question: VQA question (e.g., "What color is the top box?")

        Returns:
            reward: 1.0 if correct, 0.0 otherwise
        """
        # Get ground truth answer from VQA model
        ground_truth = self.vqa_model.get_answer(image, question)

        # Simple string match for now (could use embeddings)
        is_correct = predicted_answer.strip().lower() == ground_truth.lower()

        return float(is_correct)

    def compute_batch_perception_rewards(self, images, answers, questions):
        """
        Score multiple perception predictions.

        Args:
            images: Batch of images
            answers: Predicted answers
            questions: VQA questions

        Returns:
            rewards: Batch of binary rewards
        """
        rewards = []
        for img, ans, q in zip(images, answers, questions):
            reward = self.compute_perception_reward(img, ans, q)
            rewards.append(reward)

        return torch.tensor(rewards)
```

Now build the intermediate reasoning reward (detail rewards):

```python
class DetailRewardComputer:
    """
    Compute intermediate-step rewards for spatial reasoning.
    """
    def __init__(self, verifier_model):
        self.verifier = verifier_model  # Trained to check intermediate steps

    def compute_detail_reward(self, image, reasoning_trace, expected_step):
        """
        Score intermediate reasoning step.

        Args:
            image: Visual input
            reasoning_trace: Model's reasoning (text)
            expected_step: What should the model identify? (e.g., "identify start location")

        Returns:
            reward: 0-1 scalar for step correctness
        """
        # Extract relevant detail from reasoning
        # (e.g., did it identify the start location correctly?)
        step_accuracy = self.verifier.score_step(
            image,
            reasoning_trace,
            expected_step
        )

        return step_accuracy

    def compute_multi_step_reward(self, image, full_reasoning, step_specs):
        """
        Score multiple intermediate steps from a single reasoning.

        Args:
            image: Visual input
            full_reasoning: Complete reasoning trace
            step_specs: List of (step_description, expected_output) tuples

        Returns:
            detail_rewards: Reward for each step (average = overall detail reward)
        """
        step_rewards = []

        for step_name, expected in step_specs:
            # Check if model correctly performed this step
            step_reward = self.compute_detail_reward(image, full_reasoning, step_name)
            step_rewards.append(step_reward)

        return torch.tensor(step_rewards)
```

Implement the multi-stage training controller:

```python
class RewardMapTrainer:
    """
    Coordinate multi-stage training with appropriate rewards.
    """
    def __init__(self, model, stages=3):
        self.model = model
        self.stages = stages
        self.stage_configs = self._init_stage_configs()

    def _init_stage_configs(self):
        """Define training configuration for each stage."""
        return {
            "stage_1": {
                "name": "Perception",
                "dataset": "vqa_perception",
                "reward_type": "binary_vqa",
                "num_epochs": 2,
                "examples_per_task": 10
            },
            "stage_2": {
                "name": "Spatial Reasoning",
                "dataset": "spatial_reasoning",
                "reward_type": "detail_rewards",
                "num_epochs": 2,
                "examples_per_task": 20
            },
            "stage_3": {
                "name": "Complex Tasks",
                "dataset": "full_reasoning_tasks",
                "reward_type": "combined",  # task + detail rewards
                "num_epochs": 3,
                "examples_per_task": 30
            }
        }

    def train_stage(self, stage_num, train_data, verifier, vqa_model):
        """
        Train model on a single stage with appropriate rewards.

        Args:
            stage_num: Which stage (1, 2, or 3)
            train_data: Training examples for this stage
            verifier: Model to verify intermediate steps
            vqa_model: VQA model for perception rewards

        Returns:
            model: Updated model after training on this stage
        """
        config = self.stage_configs[f"stage_{stage_num}"]
        print(f"Training {config['name']} (Stage {stage_num})")

        reward_computer = self._get_reward_computer(
            config["reward_type"],
            vqa_model,
            verifier
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        for epoch in range(config["num_epochs"]):
            total_loss = 0
            num_examples = 0

            for example in train_data:
                image = example["image"]
                task_description = example["task"]
                ground_truth = example["answer"]
                intermediate_steps = example.get("steps", [])

                # Generate model's response
                model_output = self.model.generate(image, task_description)

                # Compute rewards based on stage
                if config["reward_type"] == "binary_vqa":
                    # Stage 1: perception rewards
                    reward = reward_computer.compute_perception_reward(
                        image,
                        model_output,
                        task_description
                    )
                    loss = -torch.log(torch.tensor(reward + 1e-6))

                elif config["reward_type"] == "detail_rewards":
                    # Stage 2: intermediate step rewards
                    detail_rewards = reward_computer.compute_multi_step_reward(
                        image,
                        model_output,
                        intermediate_steps
                    )
                    # Loss: maximize average detail reward
                    loss = -detail_rewards.mean()

                elif config["reward_type"] == "combined":
                    # Stage 3: task + detail rewards
                    task_correct = model_output.strip() == ground_truth.strip()
                    task_reward = float(task_correct)

                    detail_rewards = reward_computer.compute_multi_step_reward(
                        image,
                        model_output,
                        intermediate_steps
                    )

                    # Weighted combination
                    combined_reward = 0.7 * task_reward + 0.3 * detail_rewards.mean()
                    loss = -combined_reward

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_examples += 1

            avg_loss = total_loss / num_examples
            print(f"  Epoch {epoch + 1} loss: {avg_loss:.4f}")

        return self.model

    def _get_reward_computer(self, reward_type, vqa_model, verifier):
        """Instantiate appropriate reward computer."""
        if reward_type in ["binary_vqa", "combined"]:
            return VQARewardComputer(vqa_model)
        elif reward_type == "detail_rewards":
            return DetailRewardComputer(verifier)
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    def train_all_stages(self, stage_datasets, verifier, vqa_model):
        """
        Train through all stages sequentially.

        Args:
            stage_datasets: List of train_data for each stage
            verifier: Model to verify intermediate steps
            vqa_model: VQA model for perception rewards

        Returns:
            model: Fully trained model
        """
        for stage_num in range(1, self.stages + 1):
            train_data = stage_datasets[stage_num - 1]
            self.model = self.train_stage(
                stage_num,
                train_data,
                verifier,
                vqa_model
            )

        return self.model
```

Finally, create the ReasonMap-Plus dataset with intermediate annotations:

```python
class ReasonMapPlusDataset:
    """
    Extended dataset with intermediate reasoning steps for detail rewards.
    """
    def __init__(self):
        self.perception_examples = []  # For stage 1
        self.reasoning_examples = []   # For stage 2
        self.full_task_examples = []   # For stage 3

    def add_full_task(self, image, task_description, solution, intermediate_steps):
        """
        Add a full task with annotated intermediate steps.

        Args:
            image: Visual input
            task_description: What to do with the image
            solution: Correct answer
            intermediate_steps: List of (step_name, expected_result) tuples
                               e.g., [("identify_start", "Point A"), ("find_path", "A->B->C")]
        """
        # Create perception examples from intermediate steps
        for step_name, expected_result in intermediate_steps:
            vqa_question = f"In this image, {step_name}. What is it?"
            self.perception_examples.append({
                "image": image,
                "task": vqa_question,
                "answer": expected_result
            })

        # Create reasoning example with intermediate steps
        self.reasoning_examples.append({
            "image": image,
            "task": task_description,
            "steps": intermediate_steps,
            "answer": solution
        })

        # Create full task example
        self.full_task_examples.append({
            "image": image,
            "task": task_description,
            "steps": intermediate_steps,
            "answer": solution
        })

    def get_datasets_by_stage(self):
        """Return datasets organized by training stage."""
        return [
            self.perception_examples,    # Stage 1
            self.reasoning_examples,     # Stage 2
            self.full_task_examples      # Stage 3
        ]
```

## Practical Guidance

**When to use RewardMap:**
- Visual reasoning tasks with clear intermediate steps (transit maps, spatial reasoning, diagram understanding)
- Multimodal LLM training where perception is a bottleneck
- Tasks where you can specify intermediate milestones
- Compute budgets support multi-stage training

**When NOT to use:**
- Simple visual tasks (classification, detection) — standard supervised learning is simpler
- Tasks without clear intermediate steps
- Single-stage end-to-end optimization is acceptable
- Extreme time constraints (curriculum overhead may not pay off early)

**Training efficiency improvements:**

| Approach | Accuracy | Total Compute | Convergence |
|----------|----------|---|---|
| End-to-end RL | 65% | 100% | Slow (200 epochs) |
| RewardMap Stage 1-3 | 72% | 110% | Fast (6 epochs) |
| Improvement | +7% | +10% | 33x faster convergence |

**Reward configuration by stage:**

| Stage | Primary Reward | Secondary Reward | Typical Accuracy Gain |
|-------|---|---|---|
| 1 (Perception) | Binary VQA | None | 40-50% → 60% |
| 2 (Reasoning) | Detail rewards | None | 60% → 70% |
| 3 (Complex) | Task + detail | None | 70% → 75%+ |

**Common pitfalls:**
- **Weak intermediate annotations**: If step labels are incorrect, detail rewards train the model wrong. Validate annotations on 50 examples before full training.
- **Stage too long**: If stage 1 takes 50 epochs, students get bored and diverge. Use 2-3 epochs per stage; let learning rate decay handle convergence.
- **Reward discount mismatch**: All stages should have similar reward scales (0-1 range). If stage 3 has rewards in 0-100 range, adjust scaling.
- **Curriculum too rigid**: Some examples are inherently hard. Allow examples to skip stages if needed (e.g., hard tasks go straight to stage 3).

**Integration checklist:**
- [ ] Prepare full task dataset with annotated intermediate steps (50+ examples minimum)
- [ ] Extract or generate VQA questions for perception stage
- [ ] Train verifier model to score intermediate steps (validate on 30 examples)
- [ ] Run stage 1 for 2 epochs; validate perception accuracy improves
- [ ] Run stage 2 for 2 epochs; validate spatial reasoning improves
- [ ] Run stage 3 for 3 epochs; validate full task accuracy improves
- [ ] Compare staged training to end-to-end RL baseline

Reference: https://arxiv.org/abs/2510.02240
