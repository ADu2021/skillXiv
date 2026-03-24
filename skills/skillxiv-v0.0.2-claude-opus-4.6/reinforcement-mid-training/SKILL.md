---
name: reinforcement-mid-training
title: "Reinforcement Mid-Training: Intermediate Optimization Between Pretraining and Post-Training"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2509.24375
keywords: [intermediate-training, reinforcement-learning, token-optimization, curriculum-learning, reasoning]
description: "Add an intermediate RL stage between pretraining and post-training using dynamic token budgeting, curriculum sampling, and dual training. Trigger: reduce reasoning steps while maintaining or improving performance in post-training."
---

# Reinforcement Mid-Training (RMT): Three-Stage LLM Development

## Core Concept

Standard LLM development has two stages: pretraining (next-token prediction) and post-training (instruction-tuning + RL). RMT inserts a **critical middle stage** that applies reinforcement learning to discourage unnecessary reasoning steps while focusing training on high-value tokens. This achieves up to 64.91% performance improvement while using only 21% of reasoning tokens.

The key insight: Not all tokens contribute equally to performance. Mid-training learns to allocate compute efficiently before the model develops bad reasoning habits during post-training.

## Architecture Overview

- **Dynamic Token Budgeting**: Learns when to stop generating reasoning and produce answers
- **Curriculum-Based Sampling**: Progressive training from simple to complex tokens
- **Dual Training Strategy**: Combines RL with next-token prediction for token importance weighting
- **Three-Stage Pipeline**: Pretraining → Mid-Training (RL) → Post-Training
- **Efficiency Gains**: 21% of tokens for equivalent or better performance

## Implementation Steps

### 1. Prepare Mid-Training Data and Setup

Mid-training requires a dataset with reasoning-intensive tasks (math, code) and a clean base model from pretraining.

```python
class MidTrainingConfig:
    # Model and data
    base_model_path = "llama-7b-pretrained"
    mid_training_dataset = "math_code_reasoning"  # MATH, GSM8K, MBPP
    num_examples = 50000

    # Training schedule
    num_epochs = 3
    warmup_steps = 1000
    total_steps = 100000

    # Token budgeting
    max_reasoning_tokens = 4096  # Hard limit on thinking
    reward_for_shorter_reasoning = 0.1  # Penalize verbose thinking

    # Curriculum
    difficulty_stages = [
        {"name": "easy", "examples": 10000, "curriculum_weight": 0.5},
        {"name": "medium", "examples": 25000, "curriculum_weight": 0.3},
        {"name": "hard", "examples": 15000, "curriculum_weight": 0.2},
    ]

    # RL settings
    lr = 1e-5
    rl_weight = 0.7  # RL loss weight vs. supervised loss
    supervised_weight = 0.3
```

### 2. Implement Dynamic Token Budgeting

Create a mechanism to reward efficient reasoning and penalize excessive token generation.

```python
class TokenBudgetingReward:
    def __init__(self, max_tokens=4096, target_ratio=0.5):
        self.max_tokens = max_tokens
        self.target_ratio = target_ratio  # Ideally use 50% of budget

    def compute_reward(self, thinking_tokens, final_answer_tokens, is_correct):
        """
        Reward correctness, but penalize verbose thinking.

        Args:
            thinking_tokens: Number of reasoning tokens generated
            final_answer_tokens: Number of tokens in final answer
            is_correct: Boolean whether answer is correct

        Returns:
            Scalar reward signal
        """
        # Base correctness reward
        correctness_reward = 1.0 if is_correct else -1.0

        # Efficiency bonus: reward using <50% of budget
        efficiency_ratio = thinking_tokens / self.max_tokens
        efficiency_bonus = 0.5 if efficiency_ratio < self.target_ratio else -0.2

        # Penalize excessive reasoning
        if thinking_tokens > self.max_tokens:
            excessive_penalty = -0.5
        else:
            excessive_penalty = 0

        # Combined reward
        total_reward = (
            correctness_reward +
            0.3 * efficiency_bonus +
            excessive_penalty
        )

        return total_reward

    def record_trajectory(self, problem, thinking, answer, is_correct):
        """Log a trajectory with token counts and correctness."""
        thinking_tokens = len(thinking.split())
        answer_tokens = len(answer.split())

        reward = self.compute_reward(
            thinking_tokens,
            answer_tokens,
            is_correct
        )

        return {
            "problem": problem,
            "thinking": thinking,
            "answer": answer,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": answer_tokens,
            "is_correct": is_correct,
            "reward": reward
        }
```

### 3. Implement Curriculum-Based Sampling

Start with easy problems and progressively increase difficulty. This helps the model learn efficient reasoning on simpler tasks before tackling complex ones.

```python
class CurriculumSampler:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.current_stage = 0
        self.stage_step = 0

    def sample_batch(self, batch_size):
        """
        Sample a batch weighted by curriculum.
        """
        # Determine current difficulty stage based on training progress
        stage_config = self.config.difficulty_stages[self.current_stage]

        # Sample from current stage with high probability
        # Sample from adjacent stages with lower probability
        difficulties = ["easy", "medium", "hard"]

        if self.current_stage == 0:
            difficulty_weights = [0.8, 0.2, 0.0]
        elif self.current_stage == 1:
            difficulty_weights = [0.2, 0.6, 0.2]
        else:
            difficulty_weights = [0.0, 0.3, 0.7]

        # Weighted sampling
        batch = []
        for _ in range(batch_size):
            difficulty = np.random.choice(
                difficulties,
                p=difficulty_weights
            )
            example = self.dataset.sample_by_difficulty(difficulty)
            batch.append(example)

        # Progress through stages
        self.stage_step += 1
        if self.stage_step > stage_config.get("duration_steps", 30000):
            self.current_stage = min(self.current_stage + 1, 2)
            self.stage_step = 0

        return batch

    def get_curriculum_weight(self):
        """Return curriculum weight for loss computation."""
        return self.config.difficulty_stages[self.current_stage]["curriculum_weight"]
```

### 4. Implement Dual Training (RL + Supervised)

Combine RL losses (based on correctness) with supervised losses (predicting important tokens).

```python
def compute_mid_training_loss(
    model,
    batch,
    token_budgeter,
    curriculum_weight,
    rl_weight=0.7,
    supervised_weight=0.3
):
    """
    Compute combined RL and supervised loss for mid-training.

    Args:
        model: LLM to train
        batch: List of training examples
        token_budgeter: Reward computer
        curriculum_weight: Importance weight from curriculum

    Returns:
        Scalar loss value
    """
    total_loss = 0
    batch_size = len(batch)

    for example in batch:
        # Generate thinking and answer
        output = model.generate(
            example["problem"],
            max_thinking_tokens=4096,
            return_intermediate=True
        )

        thinking = output["thinking"]
        answer = output["answer"]

        # Evaluate correctness
        is_correct = evaluate_answer(answer, example["ground_truth"])

        # Compute reward (RL component)
        reward = token_budgeter.compute_reward(
            len(thinking.split()),
            len(answer.split()),
            is_correct
        )

        log_prob = model.compute_log_prob(thinking + answer)

        # RL loss: policy gradient
        rl_loss = -reward * log_prob

        # Supervised loss: predict important tokens
        # Identify "important" tokens (problem-solving steps, key insights)
        important_tokens = identify_important_tokens(
            thinking,
            example["solution_tokens"]
        )

        supervised_loss = compute_token_importance_loss(
            model,
            thinking,
            important_tokens
        )

        # Combine losses with curriculum weight
        combined_loss = (
            rl_weight * rl_loss +
            supervised_weight * supervised_loss
        ) * curriculum_weight

        total_loss += combined_loss

    return total_loss / batch_size
```

### 5. Full Mid-Training Loop

Orchestrate the three-component training process.

```python
def train_mid_training(
    model,
    dataset,
    config,
    evaluation_dataset=None
):
    """
    Execute reinforcement mid-training.

    Args:
        model: Pretrained base model
        dataset: Training examples with reasoning and answers
        config: MidTrainingConfig instance
        evaluation_dataset: Optional validation set

    Returns:
        Model after mid-training, loss history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.total_steps
    )

    token_budgeter = TokenBudgetingReward(
        max_tokens=config.max_reasoning_tokens
    )
    curriculum_sampler = CurriculumSampler(dataset, config)

    loss_history = []
    step = 0

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_idx in range(config.total_steps // 32):  # Batch size 32
            # Sample batch from curriculum
            batch = curriculum_sampler.sample_batch(batch_size=32)
            curriculum_weight = curriculum_sampler.get_curriculum_weight()

            # Compute loss
            loss = compute_mid_training_loss(
                model,
                batch,
                token_budgeter,
                curriculum_weight,
                rl_weight=config.rl_weight,
                supervised_weight=config.supervised_weight
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            # Logging
            if step % 100 == 0:
                avg_loss = epoch_loss / num_batches
                loss_history.append(avg_loss)
                print(f"Step {step}: loss={avg_loss:.4f}, "
                      f"curriculum_stage={curriculum_sampler.current_stage}")

            # Evaluation
            if evaluation_dataset and step % 5000 == 0:
                eval_metrics = evaluate_on_benchmark(
                    model,
                    evaluation_dataset,
                    metrics=["accuracy", "avg_reasoning_tokens"]
                )
                print(f"Eval at step {step}: {eval_metrics}")

    return model, loss_history
```

## Practical Guidance

**Hyperparameters:**
- **Max reasoning tokens**: 4096 (adapt to task complexity)
- **Target ratio for efficiency**: 0.5 (aim for 50% of budget)
- **RL loss weight**: 0.7 (emphasize correct reasoning)
- **Supervised weight**: 0.3 (maintain token prediction capability)
- **Curriculum duration**: 30K-50K steps per stage

**When to Use:**
- Developing reasoning models (math, code, logic)
- Want to reduce inference cost while maintaining quality
- Have access to reasoning datasets (MATH, GSM8K, MBPP)
- Pre-training already complete, before post-training

**When NOT to Use:**
- Single-turn generation tasks without reasoning
- Instruction-tuning directly from pretrained (skip mid-training)
- Limited compute for three-stage training pipeline
- Tasks where all tokens are equally important

## Reference

[Reinforcement Mid-Training: Intermediate Optimization Between Pretraining and Post-Training](https://arxiv.org/abs/2509.24375) — arXiv:2509.24375
