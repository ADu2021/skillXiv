---
name: parallel-r1-parallel-thinking-rl
title: "Parallel-R1: Towards Parallel Thinking via Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.07980"
keywords: [parallel reasoning, reinforcement learning, chain-of-thought, curriculum learning, mathematical reasoning, LLM training, exploration strategy, multi-perspective verification]
description: "Train language models to explore multiple reasoning paths simultaneously via reinforcement learning. Uses progressive curriculum learning to address cold-start problems, enabling 8.4% accuracy gains over sequential reasoning on complex mathematical tasks."
---

# Parallel Thinking via Reinforcement Learning

## Outcome

Enable language models to reason through complex problems by exploring multiple solution paths concurrently, achieving 8.4% accuracy improvements over sequential chain-of-thought approaches and up to 42.9% gains on high-difficulty math benchmarks through a two-stage curriculum learning framework.

## Problem Context

Large language models excel at sequential reasoning but struggle with truly exploratory problem-solving. Traditional supervised fine-tuning (SFT) produces teacher-forced imitation rather than genuine exploration. Existing parallel reasoning approaches rely entirely on synthetic data fine-tuning, leaving significant performance gains unexploited. The challenge: how can we train models to naturally explore multiple reasoning branches while maintaining computational efficiency?

## Core Concept

Parallel-R1 combines supervised fine-tuning with reinforcement learning in a progressive curriculum. Models first learn parallel reasoning on easier problems via SFT (cold-start solution), then transition to RL-based exploration on progressively harder problems. This unlocks behavioral shifts: initial parallel thinking serves as exploratory scaffolding, later evolving into multi-perspective verification for robust problem-solving.

## Architecture Overview

The framework operates in two training stages with curriculum progression:

- **Stage 1 (SFT Phase)**: Initialize parallel thinking ability on easier benchmark tasks using synthetic prompt-generated trajectories. This cold-start approach provides foundational exploration patterns before RL training.

- **Stage 2 (RL Phase)**: Transition to reinforcement learning on harder problems. Reward signals optimize for convergence toward correct solutions, efficient exploration, and early termination when high-confidence answers emerge.

- **Progressive Curriculum**: Task difficulty progression follows MATH (base) → AMC23 (medium) → AIME (challenging), allowing models to develop exploration capabilities incrementally.

- **Dual Thinking Patterns**: Early training emphasizes parallel thinking as exploration strategy. Advanced training shifts toward multi-perspective verification, using parallel paths for robustness rather than search.

- **Inference Efficiency**: By learning when to consolidate multiple reasoning chains, the model reduces computational overhead compared to exhaustive sequential reasoning.

## Implementation

### Stage 1: Supervised Fine-Tuning with Parallel Trajectories

Begin by establishing parallel reasoning foundations. Generate or collect multiple solution trajectories for easier problems, ensuring models learn to produce branching reasoning structures before RL training.

```python
# SFT Phase: Initialize parallel thinking with easier tasks
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class ParallelTrajectoryDataset(Dataset):
    """Load problem-solution pairs with multiple reasoning paths."""
    def __init__(self, problems, trajectories, tokenizer, max_length=2048):
        self.problems = problems
        self.trajectories = trajectories  # List[List[str]] - multiple paths per problem
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        paths = self.trajectories[idx]

        # Format as multi-branch reasoning: "Problem: X\nPath 1: ...\nPath 2: ..."
        combined_text = f"Problem: {problem}\n"
        for i, path in enumerate(paths, 1):
            combined_text += f"Path {i}: {path}\n"

        encoding = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Create dataset from easier tasks (MATH benchmark)
dataset = ParallelTrajectoryDataset(
    problems=easier_problems,
    trajectories=trajectory_collections,
    tokenizer=tokenizer
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Fine-tune on parallel trajectories
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

for epoch in range(3):
    for batch in dataloader:
        inputs = {k: v.to('cuda') for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save checkpoint after SFT phase
model.save_pretrained("./parallel_r1_sft_checkpoint")
```

### Stage 2: Reinforcement Learning with Progressive Curriculum

After SFT establishes parallel thinking patterns, apply RL training with reward signals that reinforce correct solutions and efficient exploration. Use progressive curriculum to scale difficulty.

```python
# RL Phase: Train with reward signals and curriculum progression
from torch.distributions import Categorical
import numpy as np

class ParallelThinkingRLTrainer:
    """RL trainer for parallel reasoning with progressive curriculum."""
    def __init__(self, model, tokenizer, reward_fn, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn  # Callable that returns reward for solution
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def generate_parallel_trajectories(self, problem, num_paths=3, max_tokens=500):
        """Generate multiple reasoning paths for a single problem."""
        trajectories = []

        for _ in range(num_paths):
            input_ids = self.tokenizer(
                f"Problem: {problem}\nReasoning: ",
                return_tensors='pt'
            )['input_ids'].to(self.device)

            # Generate with temperature sampling for diversity
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True
            )

            trajectory = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            trajectories.append(trajectory)

        return trajectories

    def compute_trajectory_reward(self, problem, trajectories):
        """Compute rewards for each trajectory based on correctness."""
        rewards = []

        for trajectory in trajectories:
            # Extract answer from trajectory
            answer = self.extract_answer(trajectory)
            # Compute reward based on correctness
            reward = self.reward_fn(problem, answer)
            rewards.append(reward)

        return torch.tensor(rewards, device=self.device)

    def extract_answer(self, trajectory):
        """Extract final answer from reasoning trajectory."""
        # Assume answer format: "Answer: X"
        if "Answer:" in trajectory:
            return trajectory.split("Answer:")[-1].strip().split()[0]
        return None

    def train_on_problem(self, problem, num_paths=3):
        """Execute one RL training step on a problem."""
        # Generate multiple trajectories
        trajectories = self.generate_parallel_trajectories(problem, num_paths)

        # Compute rewards
        rewards = self.compute_trajectory_reward(problem, trajectories)

        # Normalize rewards for stable training
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute loss: encourage high-reward trajectories
        loss = 0
        for i, trajectory in enumerate(trajectories):
            input_ids = self.tokenizer(
                f"Problem: {problem}\nReasoning: ",
                return_tensors='pt'
            )['input_ids'].to(self.device)

            # Forward pass through model
            outputs = self.model(input_ids)
            logits = outputs.logits

            # Log probability of generated trajectory (simplified)
            # In practice, compute actual log-probs of tokens in trajectory
            trajectory_logprob = torch.tensor(0.0, device=self.device)

            # Policy gradient: -E[logprob * reward]
            loss += -trajectory_logprob * rewards[i]

        loss = loss / num_paths
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train_with_curriculum(self, curriculum_stages, epochs_per_stage):
        """Execute progressive curriculum training."""
        for stage_idx, problems in enumerate(curriculum_stages):
            print(f"Training on stage {stage_idx + 1} ({len(problems)} problems)")

            for epoch in range(epochs_per_stage):
                total_loss = 0
                for problem in problems:
                    loss = self.train_on_problem(problem, num_paths=3)
                    total_loss += loss

                avg_loss = total_loss / len(problems)
                print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

            # Save checkpoint after each curriculum stage
            self.model.save_pretrained(f"./parallel_r1_rl_stage_{stage_idx + 1}")

# Setup reward function for mathematical problems
def math_reward_fn(problem, answer):
    """Reward correct answers with 1.0, incorrect with 0.0."""
    try:
        correct_answer = evaluate_math_answer(problem)  # Domain-specific evaluator
        return 1.0 if answer == correct_answer else 0.0
    except:
        return 0.0

# Execute RL training with progressive curriculum
trainer = ParallelThinkingRLTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_fn=math_reward_fn,
    device='cuda'
)

# Define curriculum: MATH (easy) → AMC23 (medium) → AIME (hard)
curriculum = [
    easy_math_problems,      # ~1000 problems, 70% solve rate
    medium_amc_problems,     # ~500 problems, 40% solve rate
    hard_aime_problems       # ~200 problems, 10% solve rate
]

trainer.train_with_curriculum(
    curriculum_stages=curriculum,
    epochs_per_stage=5
)
```

### Stage 2b: Inference with Parallel Path Consolidation

During inference, generate multiple reasoning paths in parallel and intelligently consolidate them into final answers.

```python
# Inference: Generate and consolidate parallel reasoning paths
class ParallelInference:
    def __init__(self, model, tokenizer, num_paths=5, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.num_paths = num_paths
        self.device = device

    def infer_with_parallel_thinking(self, problem, return_paths=False):
        """Generate multiple reasoning paths and consolidate results."""
        # Generate num_paths different solution attempts
        paths_data = []

        for path_idx in range(self.num_paths):
            input_text = f"Problem: {problem}\nReasoning: "
            input_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids'].to(self.device)

            # Generate with different random seeds for diversity
            torch.manual_seed(path_idx)
            output = self.model.generate(
                input_ids,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )

            trajectory = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self.extract_answer(trajectory)
            confidence = self.compute_confidence(trajectory)

            paths_data.append({
                'trajectory': trajectory,
                'answer': answer,
                'confidence': confidence,
                'path_idx': path_idx
            })

        # Consolidate answers via majority voting
        answers = [p['answer'] for p in paths_data if p['answer'] is not None]
        from collections import Counter
        answer_counts = Counter(answers)

        if answer_counts:
            final_answer = answer_counts.most_common(1)[0][0]
            agreement = answer_counts.most_common(1)[0][1] / len(answers)
        else:
            # Fallback: use highest confidence path
            best_path = max(paths_data, key=lambda x: x['confidence'])
            final_answer = best_path['answer']
            agreement = best_path['confidence']

        if return_paths:
            return final_answer, agreement, paths_data
        return final_answer, agreement

    def extract_answer(self, trajectory):
        """Extract numeric or symbolic answer from trajectory."""
        if "Answer:" in trajectory:
            answer_text = trajectory.split("Answer:")[-1].strip()
            # Extract first number or symbol
            import re
            match = re.search(r'[\d\.]+|[a-zA-Z]+', answer_text)
            return match.group(0) if match else None
        return None

    def compute_confidence(self, trajectory):
        """Estimate confidence from trajectory structure and language."""
        # Simple heuristic: presence of verification steps increases confidence
        confidence = 0.5
        if "verify" in trajectory.lower() or "check" in trajectory.lower():
            confidence += 0.2
        if trajectory.count("therefore") > 0:
            confidence += 0.15
        return min(confidence, 1.0)

# Use during inference
inference = ParallelInference(model, tokenizer, num_paths=5)
final_answer, agreement_score = inference.infer_with_parallel_thinking(
    problem="What is the value of 2^10 + 3^5?"
)
print(f"Answer: {final_answer}, Agreement: {agreement_score:.2f}")
```

## Practical Guidance

### Hyperparameters Table

| Parameter | Recommended Value | Range | Impact |
|-----------|-------------------|-------|--------|
| **SFT Learning Rate** | 2e-5 | 1e-5 to 5e-5 | Lower = more stable, slower; Higher = faster convergence but unstable |
| **RL Learning Rate** | 1e-5 | 5e-6 to 2e-5 | RL requires lower LR than SFT to prevent policy collapse |
| **Parallel Paths (Training)** | 3-4 | 2-6 | More paths = better exploration but higher compute cost |
| **Parallel Paths (Inference)** | 5-7 | 3-10 | Sweet spot balances accuracy and latency |
| **Temperature** | 0.7-0.8 | 0.5-1.0 | Controls trajectory diversity; higher = more diverse solutions |
| **Top-p (Nucleus Sampling)** | 0.9-0.95 | 0.8-0.98 | Maintains quality while allowing exploration |
| **Epochs per Curriculum Stage** | 3-5 | 1-10 | Must be sufficient to stabilize before difficulty increase |
| **Reward Normalization** | (R - mean) / (std + 1e-8) | — | Crucial for stable policy gradient updates |
| **Batch Size** | 8-16 | 4-32 | Larger = more stable gradients, higher memory |

### When to Use

- **Complex mathematical reasoning**: Multi-step algebra, geometry, combinatorics
- **Code generation with multiple valid approaches**: When problems admit diverse correct solutions
- **Risk-averse applications requiring verification**: Use parallel reasoning for cross-validation
- **Progressive task curricula available**: When you have easy→medium→hard task hierarchies
- **Sufficient compute budget**: Training cost approximately 2-3x that of standard SFT

### When NOT to Use

- **Simple retrieval or classification tasks**: Parallel reasoning adds overhead without benefit
- **Hard real-time constraints**: Multiple path generation increases latency significantly
- **Limited computational resources**: RL training requires tracking multiple trajectories simultaneously
- **Tasks with single canonical solution**: When diversity in reasoning offers no advantage
- **Insufficient training data diversity**: Cold-start SFT phase requires multiple solution trajectories per problem
- **Already-optimized sequential models**: If chain-of-thought baseline achieves 90%+ accuracy, gains diminish
- **Tasks without clear correctness evaluation**: Reward function requires objective success metrics

### Common Pitfalls

1. **Skipping SFT Phase**: Attempting RL directly on untrained models leads to divergence. Always establish foundations with SFT on easier tasks first.

2. **Insufficient Curriculum Granularity**: Jumping from MATH directly to AIME causes training instability. Use intermediate difficulty levels (AMC23, MMLU) for smooth progression.

3. **Weak Reward Functions**: Generic reward signals (e.g., simple correctness only) miss important aspects. Include path efficiency, reasoning clarity, and verification steps in reward design.

4. **Over-generating Paths**: More paths sound better but diminishing returns appear around 5-7 paths. Beyond that, inference latency dominates benefits.

5. **Inconsistent Tokenization**: Ensure trajectories and evaluation answers use identical preprocessing. Answer extraction inconsistencies corrupt reward signals.

6. **Loss of Exploration**: As models improve, reduce temperature too aggressively or paths converge prematurely. Maintain 0.7-0.8 temperature through training.

7. **Memory Exhaustion**: Generating k paths × batch_size trajectories simultaneously exceeds typical VRAM. Use gradient accumulation or sequential path generation per batch element.

## Reference

- **Paper**: Parallel-R1: Towards Parallel Thinking via Reinforcement Learning
- **ArXiv**: https://arxiv.org/abs/2509.07980
- **Authors**: Tong Zheng, Hongming Zhang, Wenhao Yu, et al. (Tencent AI Lab)
- **GitHub Repository**: https://github.com/zhengkid/Parallel-R1
- **Benchmarks**: MATH, AMC23, AIME24, AIME25
- **Performance**: 8.4% improvement over sequential baselines; 42.9% improvement on AIME25
- **License**: Check repository for license details