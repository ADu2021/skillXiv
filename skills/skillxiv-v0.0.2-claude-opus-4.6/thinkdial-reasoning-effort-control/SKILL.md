---
name: thinkdial-reasoning-effort-control
title: "ThinkDial: Controlling Reasoning Effort in Large Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.18773
keywords: [reasoning-budget, token-compression, controllable-inference, budget-aware-training, reward-shaping]
description: "Control LLM reasoning effort through discrete modes (High/Medium/Low) using budget-aware supervised fine-tuning and adaptive reward shaping, enabling compression-performance tradeoffs."
---

# ThinkDial: Controlling Reasoning Effort in LLMs

## Core Concept

ThinkDial enables dynamic control over reasoning computation in LLMs through discrete modes: High (full capability), Medium (50% token reduction), and Low (75% token reduction). The approach combines budget-aware supervised fine-tuning (embedding budget constraints early) with two-phase reinforcement learning (offline stability + online refinement). This enables practical compression-performance tradeoffs comparable to proprietary systems while remaining open-source.

## Architecture Overview

- **Discrete Reasoning Modes**: Three effort levels with clear token budgets
- **Budget-Aware SFT**: Fine-tuning with explicit budget constraints
- **Two-Phase RL**: Offline + online for stable training
- **Adaptive Reward Shaping**: Budget-dependent rewards
- **Mode Switching**: Runtime selection of reasoning effort

## Implementation Steps

### 1. Define Reasoning Modes and Budgets

Create discrete effort levels:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict

class ReasoningMode(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class BudgetConfig:
    """Token budget for reasoning mode."""
    mode: ReasoningMode
    max_tokens: int
    thinking_proportion: float  # What fraction should be thinking vs answering
    description: str

class ThinkDialConfig:
    """Configuration for ThinkDial reasoning modes."""

    BUDGETS = {
        ReasoningMode.HIGH: BudgetConfig(
            mode=ReasoningMode.HIGH,
            max_tokens=1000,
            thinking_proportion=0.7,
            description="Full reasoning capability, highest quality"
        ),
        ReasoningMode.MEDIUM: BudgetConfig(
            mode=ReasoningMode.MEDIUM,
            max_tokens=500,  # 50% reduction
            thinking_proportion=0.5,
            description="Balanced reasoning, moderate compression"
        ),
        ReasoningMode.LOW: BudgetConfig(
            mode=ReasoningMode.LOW,
            max_tokens=250,  # 75% reduction
            thinking_proportion=0.3,
            description="Minimal reasoning, maximum compression"
        )
    }

    @staticmethod
    def get_budget(mode: ReasoningMode) -> BudgetConfig:
        return ThinkDialConfig.BUDGETS[mode]

    @staticmethod
    def get_budget_embedding(mode: ReasoningMode) -> str:
        """Textual embedding of budget constraint for prompting."""
        budget = ThinkDialConfig.get_budget(mode)
        return f"[BUDGET: {mode.value.upper()} - {budget.max_tokens} tokens max]"
```

### 2. Implement Budget-Aware SFT

Fine-tune models with explicit budget constraints:

```python
import torch
import torch.nn.functional as F

class BudgetAwareSFTTrainer:
    """Supervised fine-tuning with budget constraints."""

    def __init__(
        self,
        model: "LLM",
        learning_rate: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def compute_budget_aware_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        mode: ReasoningMode,
        current_length: int = 0
    ) -> torch.Tensor:
        """
        Compute SFT loss with budget awareness.
        """
        budget_config = ThinkDialConfig.get_budget(mode)
        remaining_budget = budget_config.max_tokens - current_length

        # Forward pass
        outputs = self.model(input_ids, labels=target_ids)
        logits = outputs.logits

        # Standard language modeling loss
        token_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_ids.view(-1),
            reduction='none'
        ).view(target_ids.shape)

        # Budget penalty: penalize generating beyond budget
        sequence_length = target_ids.shape[1]
        if sequence_length > remaining_budget:
            # Penalize tokens exceeding budget
            excess_tokens = sequence_length - remaining_budget
            budget_penalty = excess_tokens * 0.5  # Penalty weight

            token_loss = token_loss + budget_penalty / sequence_length

        # Mode-specific weighting: emphasize thinking for High mode
        if mode == ReasoningMode.HIGH:
            # Less penalty for longer reasoning
            token_loss = token_loss * 0.8

        elif mode == ReasoningMode.LOW:
            # More penalty for verbose reasoning
            token_loss = token_loss * 1.2

        return token_loss.mean()

    def train_budget_aware_sft(
        self,
        train_data: List[Dict],  # {"input": ..., "target": ..., "mode": ...}
        num_epochs: int = 3,
        batch_size: int = 8
    ) -> Dict[str, list]:
        """Train SFT with budget awareness."""

        losses = {"high": [], "medium": [], "low": []}

        for epoch in range(num_epochs):
            for batch_idx in range(0, len(train_data), batch_size):
                batch = train_data[batch_idx:batch_idx + batch_size]

                batch_loss_by_mode = {m: [] for m in ReasoningMode}

                for example in batch:
                    input_ids = torch.tensor(example["input_ids"])
                    target_ids = torch.tensor(example["target_ids"])
                    mode = example.get("mode", ReasoningMode.HIGH)

                    # Compute budget-aware loss
                    loss = self.compute_budget_aware_loss(
                        input_ids.unsqueeze(0),
                        target_ids.unsqueeze(0),
                        mode
                    )

                    batch_loss_by_mode[mode].append(loss.item())

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                # Record losses
                for mode in ReasoningMode:
                    if batch_loss_by_mode[mode]:
                        avg = sum(batch_loss_by_mode[mode]) / len(batch_loss_by_mode[mode])
                        losses[mode.value].append(avg)

        return losses
```

### 3. Implement Two-Phase RL Training

Offline + online RL with budget adaptation:

```python
class TwoPhaseRLTrainer:
    """Two-phase RL: offline stable + online refined."""

    def __init__(
        self,
        model: "LLM",
        offline_steps: int = 500,
        online_steps: int = 200,
        learning_rate: float = 1e-5
    ):
        self.model = model
        self.offline_steps = offline_steps
        self.online_steps = online_steps
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def compute_budget_aware_reward(
        self,
        response: str,
        ground_truth: str,
        mode: ReasoningMode,
        actual_tokens: int
    ) -> float:
        """
        Compute reward considering both accuracy and budget compliance.
        """
        # Accuracy component
        accuracy = self._compute_accuracy(response, ground_truth)

        # Budget compliance component
        budget_config = ThinkDialConfig.get_budget(mode)
        budget_used_ratio = actual_tokens / budget_config.max_tokens

        # Penalize exceeding budget
        if budget_used_ratio > 1.0:
            budget_compliance = 1.0 - (budget_used_ratio - 1.0) * 0.5
        else:
            # Reward efficiency: using less tokens is good
            budget_compliance = 1.0 + (1.0 - budget_used_ratio) * 0.1

        budget_compliance = max(0.0, budget_compliance)

        # Combined reward
        reward = 0.8 * accuracy + 0.2 * budget_compliance

        return reward

    def offline_rl_phase(
        self,
        dataset: List[Dict],
        mode: ReasoningMode
    ) -> Dict[str, float]:
        """
        Offline RL: learn from fixed dataset.
        Prioritize stability and budget constraint satisfaction.
        """
        print(f"Offline RL Phase for {mode.value} mode")

        metrics = {"loss": [], "reward": []}

        for step in range(self.offline_steps):
            # Sample batch
            batch = self._sample_batch(dataset, batch_size=4)

            batch_loss = 0.0
            batch_reward = 0.0

            for example in batch:
                # Generate response in specific mode
                response = self.model.generate(
                    example["input"],
                    mode=mode,
                    max_tokens=ThinkDialConfig.get_budget(mode).max_tokens
                )

                # Compute reward with budget awareness
                reward = self.compute_budget_aware_reward(
                    response,
                    example["ground_truth"],
                    mode,
                    len(response.split())
                )

                # Conservative policy loss (offline RL)
                log_prob = self.model.get_log_prob(response)
                loss = -log_prob * reward

                batch_loss += loss.item()
                batch_reward += reward

            # Optimize
            avg_loss = batch_loss / len(batch)
            avg_reward = batch_reward / len(batch)

            self.optimizer.zero_grad()
            (torch.tensor(avg_loss)).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            metrics["loss"].append(avg_loss)
            metrics["reward"].append(avg_reward)

            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")

        return metrics

    def online_rl_phase(
        self,
        tasks: List[Dict],
        mode: ReasoningMode,
        reward_model: "RewardModel" = None
    ) -> Dict[str, float]:
        """
        Online RL: interact with environment.
        Refine policy based on real feedback.
        """
        print(f"Online RL Phase for {mode.value} mode")

        metrics = {"loss": [], "reward": [], "budget_compliance": []}

        for step in range(self.online_steps):
            # Sample task
            task = tasks[step % len(tasks)]

            # Generate with current policy
            response = self.model.generate(
                task["prompt"],
                mode=mode,
                max_tokens=ThinkDialConfig.get_budget(mode).max_tokens
            )

            # Get reward
            if reward_model:
                reward = reward_model.score(response, task["expected"])
            else:
                reward = self.compute_budget_aware_reward(
                    response,
                    task["expected"],
                    mode,
                    len(response.split())
                )

            # Policy gradient update
            log_prob = self.model.get_log_prob(response)
            loss = -log_prob * reward

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Track budget compliance
            budget_config = ThinkDialConfig.get_budget(mode)
            tokens_used = len(response.split())
            compliance = 1.0 if tokens_used <= budget_config.max_tokens else 0.0

            metrics["loss"].append(loss.item())
            metrics["reward"].append(reward)
            metrics["budget_compliance"].append(compliance)

            if (step + 1) % 50 == 0:
                avg_loss = sum(metrics["loss"][-50:]) / 50
                avg_reward = sum(metrics["reward"][-50:]) / 50
                avg_compliance = sum(metrics["budget_compliance"][-50:]) / 50
                print(f"  Step {step+1}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}, Compliance={avg_compliance:.2%}")

        return metrics
```

### 4. Implement Mode Selection and Deployment

Enable runtime mode switching:

```python
class ThinkDialModel:
    """ThinkDial model with mode control."""

    def __init__(self, base_model: "LLM"):
        self.model = base_model
        self.mode_selector = ModeSelector()

    def generate_with_mode(
        self,
        prompt: str,
        mode: ReasoningMode = ReasoningMode.HIGH,
        return_mode_info: bool = False
    ) -> str:
        """Generate response in specified mode."""

        budget_config = ThinkDialConfig.get_budget(mode)
        budget_embedding = ThinkDialConfig.get_budget_embedding(mode)

        # Add mode indicator to prompt
        augmented_prompt = f"{budget_embedding}\n{prompt}"

        # Generate with budget
        response = self.model.generate(
            augmented_prompt,
            max_tokens=budget_config.max_tokens,
            temperature=0.7
        )

        if return_mode_info:
            return response, {
                "mode": mode.value,
                "tokens_used": len(response.split()),
                "budget": budget_config.max_tokens
            }

        return response

    def auto_select_mode(
        self,
        prompt: str,
        time_budget_ms: int = None,
        compute_budget: float = None
    ) -> ReasoningMode:
        """Automatically select mode based on constraints."""

        if time_budget_ms:
            # Low latency -> Low mode
            if time_budget_ms < 100:
                return ReasoningMode.LOW
            elif time_budget_ms < 500:
                return ReasoningMode.MEDIUM
            else:
                return ReasoningMode.HIGH

        if compute_budget:
            # Low compute -> Low mode
            if compute_budget < 0.3:
                return ReasoningMode.LOW
            elif compute_budget < 0.7:
                return ReasoningMode.MEDIUM
            else:
                return ReasoningMode.HIGH

        return ReasoningMode.HIGH

class ModeSelector(torch.nn.Module):
    """Learn to select appropriate mode for tasks."""

    def __init__(self, task_embedding_dim: int = 256):
        super().__init__()

        self.task_encoder = torch.nn.Linear(task_embedding_dim, 128)
        self.mode_classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(ReasoningMode))
        )

    def select_mode(self, task_embedding: torch.Tensor) -> ReasoningMode:
        """Select mode based on task."""
        encoded = self.task_encoder(task_embedding)
        logits = self.mode_classifier(encoded)
        mode_idx = torch.argmax(logits, dim=-1).item()
        return list(ReasoningMode)[mode_idx]
```

## Practical Guidance

### When to Use ThinkDial

- Deployment with variable latency constraints
- Mobile/edge devices with compute limits
- Cost-sensitive inference (pay-per-token models)
- Applications requiring quality-latency tradeoffs
- Research into compression-performance curves

### When NOT to Use

- Scenarios requiring always maximum quality
- Real-time applications (<10ms latency)
- Models without budget-aware training

### Key Hyperparameters

- **High budget**: 1000 tokens (baseline)
- **Medium budget**: 500 tokens (50% compression)
- **Low budget**: 250 tokens (75% compression)
- **accuracy_weight**: 0.8 in reward
- **budget_weight**: 0.2 in reward

### Performance Expectations

- Medium Mode: ~95% of High mode quality
- Low Mode: ~85-90% of High mode quality
- Token Reduction: 50% (Medium), 75% (Low)
- Latency Improvement: 2-3x (Medium), 4-5x (Low)
- Inference Cost: Proportional to token reduction

## Reference

Researchers. (2024). ThinkDial: Controlling Reasoning Effort in LLMs. arXiv preprint arXiv:2508.18773.
