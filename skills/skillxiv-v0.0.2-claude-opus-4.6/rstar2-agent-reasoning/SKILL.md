---
name: rstar2-agent-reasoning
title: rStar2-Agent Agentic Reasoning
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.20722
keywords: [agentic-rl, code-execution, reinforcement-learning, python, grpo]
description: "Train efficient 14B-parameter agents via GRPO with resample-on-correct rollout strategy and Python code execution, achieving state-of-the-art reasoning through autonomous exploration and iterative refinement"
---

# rStar2-Agent: Agentic Reasoning

## Core Concept

rStar2-Agent combines agentic reinforcement learning with Python code execution to enable models to autonomously explore problem-solving strategies, validate intermediate steps, and refine solutions iteratively. The approach achieves state-of-the-art reasoning performance on mathematical, scientific, and tool-use tasks within one week of training on modest hardware resources.

## Architecture Overview

- **RL Infrastructure**: Reliable Python code environment supporting high-throughput execution with cost management
- **GRPO-RoC Algorithm**: Group Relative Policy Optimization with Resample-on-Correct rollout strategy to handle environment noise from coding tools
- **Progressive Training Recipe**: Multi-stage pipeline from supervised fine-tuning through sequential RL phases
- **Agentic Behavior**: Models learn to think carefully before tool use, reflect on execution feedback, and validate solutions

## Implementation Steps

### Stage 1: Initialize Base Model and Environment

Set up a reliable Python execution environment that sandboxes code runs and captures outputs with minimal latency.

```python
# Python execution environment setup
import subprocess
import tempfile
import signal

class PythonExecutor:
    """Safe Python code execution with timeout handling"""

    def __init__(self, timeout=30):
        self.timeout = timeout

    def execute(self, code):
        """Execute Python code and return stdout, stderr, return_code"""
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Execution timeout",
                "return_code": -1
            }
```

### Stage 2: Supervised Fine-Tuning Phase

Begin with standard supervised fine-tuning on high-quality reasoning trajectories before RL training.

```python
# SFT Phase: Prepare training data with CoT examples
sft_dataset = [
    {
        "problem": "What is the sum of 2^10 and 3^5?",
        "reasoning": "Let me calculate this step by step:\n2^10 = 1024\n3^5 = 243\nSum = 1024 + 243 = 1267",
        "code": "print(2**10 + 3**5)"
    },
    # ... more examples
]

# Train with standard cross-entropy loss on reasoning trajectories
```

### Stage 3: GRPO-RoC Training

Apply Group Relative Policy Optimization with Resample-on-Correct strategy to handle noisy code execution feedback.

```python
# GRPO-RoC training loop
import torch
from torch.nn.functional import softmax

class GRPORoCTrainer:
    """Group Relative Policy Optimization with Resample-on-Correct"""

    def __init__(self, model, executor, lr=1e-5):
        self.model = model
        self.executor = executor
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.baseline_rewards = {}

    def resample_on_correct(self, trajectories, batch_size=4):
        """Resample unsuccessful trajectories while keeping correct ones"""
        grouped = {"correct": [], "incorrect": []}
        for traj in trajectories:
            if traj["execution_correct"]:
                grouped["correct"].append(traj)
            else:
                grouped["incorrect"].append(traj)

        # Keep all correct trajectories, resample incorrect ones
        resampled = grouped["correct"]
        for _ in range(len(grouped["correct"])):
            if grouped["incorrect"]:
                resampled.append(grouped["incorrect"][0])
        return resampled

    def compute_group_advantage(self, trajectories, group_size=4):
        """Compute advantages relative to group baseline"""
        groups = [trajectories[i:i+group_size]
                  for i in range(0, len(trajectories), group_size)]
        advantages = []

        for group in groups:
            group_reward = sum(t["reward"] for t in group) / len(group)
            for traj in group:
                advantages.append(traj["reward"] - group_reward)

        return advantages

    def train_step(self, trajectories):
        """Single GRPO-RoC training step"""
        # Resample to handle execution noise
        resampled = self.resample_on_correct(trajectories)

        # Compute advantages relative to group
        advantages = self.compute_group_advantage(resampled)

        # Policy gradient update
        loss = 0
        for traj, adv in zip(resampled, advantages):
            log_probs = self.model.get_log_probs(traj["tokens"])
            loss -= (log_probs.sum() * adv)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

### Stage 4: Iterative Rollouts and Feedback

Collect trajectories by rolling out the model, executing code, and using outcomes as reward signals.

```python
# Rollout collection process
def collect_rollouts(model, executor, problems, num_rollouts=8):
    """Generate trajectories and execute code for reward computation"""
    trajectories = []

    for problem in problems:
        for _ in range(num_rollouts):
            # Model generates reasoning and code
            response = model.generate(
                problem,
                max_tokens=1024,
                temperature=0.7
            )

            # Extract and execute code
            code_blocks = extract_code(response)
            execution_results = []
            correct = False

            for code in code_blocks:
                result = executor.execute(code)
                execution_results.append(result)
                # Check if output matches expected answer
                if verify_answer(result["stdout"], problem["answer"]):
                    correct = True

            trajectories.append({
                "problem": problem,
                "response": response,
                "code": code_blocks,
                "execution": execution_results,
                "execution_correct": correct,
                "tokens": tokenize(response),
                "reward": 1.0 if correct else 0.0
            })

    return trajectories
```

## Practical Guidance

### Hyperparameters

- **Learning Rate**: Start at 1e-5 for GRPO phase, can adjust based on loss curves
- **Group Size**: 4-8 trajectories per group for relative advantage computation
- **Resample Ratio**: Keep all correct rollouts, resample incorrect ones equally with correct count
- **Rollout Temperature**: 0.7 for balance between diversity and quality
- **Training Duration**: 510 RL steps achieves state-of-the-art on 14B models

### When to Use

- Complex reasoning tasks requiring multi-step intermediate validation
- Problems where outcomes are verifiable programmatically (math, coding)
- Domains with expensive compute where sample efficiency matters
- Scenarios requiring autonomous tool use and error recovery

### When NOT to Use

- Tasks without executable feedback mechanisms or clear correctness criteria
- Domains where continuous code execution is impractical or unsafe
- Applications requiring real-time response with strict latency constraints
- Scenarios with limited computational resources (the approach requires distributed execution)

### Design Considerations

The resample-on-correct strategy is crucial because code execution introduces noise (timeouts, numerical precision issues, etc.). By keeping successful rollouts and resampling failures, the algorithm focuses learning on realistic problem-solving paths. The progressive training recipe (SFT → RL) ensures models first learn basic reasoning before attempting autonomous exploration.

## Reference

rStar2-Agent: Agentic Reasoning Technical Report. arXiv:2508.20722
- https://arxiv.org/abs/2508.20722
