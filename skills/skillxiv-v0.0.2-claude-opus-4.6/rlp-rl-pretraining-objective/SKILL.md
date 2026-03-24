---
name: rlp-rl-pretraining-objective
title: "RLP: Reinforcement as a Pretraining Objective"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.01265"
keywords: [pretraining, RL-training, reasoning, dense-reward, efficiency]
description: "Integrate reinforcement learning into the pretraining phase by measuring the utility of intermediate reasoning for predicting subsequent tokens. This approach generates dense reward signals during standard pretraining, enabling models to develop reasoning abilities earlier and with significant performance gains (19% improvement on 1.7B, 45% lift on 12B models)."
---

# RLP: RL-as-Pretraining for Early Reasoning Development

Current training pipelines treat reasoning and pretraining as separate phases: first train on raw text, then apply RL for reasoning. This separation is artificial. The insight behind RLP is that **reasoning has immediate utility during pretraining itself**—chain-of-thought reasoning helps predict the next token better than direct prediction. You can measure this utility (reward signal) and use it to guide pretraining, merging RL and next-token prediction into a unified objective.

Traditional next-token prediction treats all reasoning equally. RLP instead rewards reasoning tokens that increase prediction likelihood, creating a natural curriculum where the model learns to reason when it's beneficial and to skip reasoning when unnecessary.

## Core Concept

RLP measures reward as the improvement in next-token likelihood when conditioning on sampled reasoning:

**Reward = log P(next_token | context + reasoning) - log P(next_token | context)**

This is a dense reward signal: every reasoning token gets immediate feedback on whether it actually helps prediction. This enables learning from raw pretraining data without external verifiers or gold reasoning traces.

## Architecture Overview

- **Base LLM**: Standard transformer model
- **Reasoning sampler**: Generate intermediate reasoning (either sampled or guided)
- **Reward computer**: Measure log-likelihood improvement from reasoning
- **Training objective**: Weighted combination of reasoning utility and next-token prediction
- **Scheduler**: Control reasoning frequency over training (start high, adapt based on reward)

## Implementation Steps

Start by implementing the reward computation. This is the core signal:

```python
import torch
import torch.nn.functional as F

class ReasoningRewardComputer:
    """
    Compute reward for intermediate reasoning during pretraining.
    """
    def __init__(self, model, tokenizer, temperature=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def compute_reasoning_utility(self, context, reasoning, next_token):
        """
        Measure how much reasoning improves prediction of next token.

        Args:
            context: Input tokens [cls_token, ...]
            reasoning: Sampled intermediate reasoning tokens
            next_token: Target token ID to predict

        Returns:
            reward: Log-likelihood improvement (can be negative)
            logits_with_reasoning: Model outputs when reasoning provided
            logits_without_reasoning: Model outputs without reasoning
        """
        with torch.no_grad():
            # Forward pass WITHOUT reasoning: just context -> predict next token
            outputs_no_reasoning = self.model(context)
            logits_no_reasoning = outputs_no_reasoning.logits[:, -1, :]  # Last position
            logprob_no_reasoning = F.log_softmax(logits_no_reasoning, dim=-1)[
                range(context.shape[0]),
                next_token
            ]

            # Forward pass WITH reasoning: context + reasoning -> predict next token
            reasoning_input = torch.cat([context, reasoning], dim=1)
            outputs_with_reasoning = self.model(reasoning_input)
            logits_with_reasoning = outputs_with_reasoning.logits[:, -1, :]
            logprob_with_reasoning = F.log_softmax(logits_with_reasoning, dim=-1)[
                range(reasoning_input.shape[0]),
                next_token
            ]

        # Reward: likelihood improvement from reasoning
        reward = logprob_with_reasoning - logprob_no_reasoning

        return reward, logits_with_reasoning, logits_no_reasoning
```

Now implement reasoning sampling and selection:

```python
class ReasoningSampler:
    """
    Generate intermediate reasoning tokens during pretraining.
    """
    def __init__(self, model, tokenizer, max_reasoning_length=50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_reasoning_length = max_reasoning_length

    def sample_reasoning(self, context, num_samples=3):
        """
        Generate multiple reasoning traces from context.

        Args:
            context: Input tokens
            num_samples: Number of reasoning samples to draw

        Returns:
            reasoning_samples: List of token sequences
        """
        reasoning_samples = []

        # Use special token to cue reasoning
        reasoning_cue = self.tokenizer.encode("[REASON]", return_tensors="pt")[0]
        prompt = torch.cat([context, reasoning_cue.unsqueeze(0)], dim=1)

        for _ in range(num_samples):
            # Sample reasoning autoregressively
            reasoning_tokens = []
            current_input = prompt

            for step in range(self.max_reasoning_length):
                with torch.no_grad():
                    outputs = self.model(current_input)
                    logits = outputs.logits[:, -1, :]

                    # Sample with temperature
                    probs = F.softmax(logits / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    reasoning_tokens.append(next_token.item())
                    current_input = torch.cat([current_input, next_token], dim=1)

                    # Stop on EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            reasoning_samples.append(reasoning_tokens)

        return reasoning_samples
```

Implement the combined training objective:

```python
def rlp_training_step(
    model,
    batch,
    tokenizer,
    reasoning_probability=0.5,
    reasoning_weight=0.3
):
    """
    Single training step combining pretraining and reasoning RL.

    Args:
        model: LLM to train
        batch: Input token batch
        tokenizer: Tokenizer
        reasoning_probability: Fraction of positions to apply reasoning
        reasoning_weight: Balance between reasoning loss and LM loss

    Returns:
        loss: Combined loss
        metrics: Training diagnostics
    """
    reward_computer = ReasoningRewardComputer(model, tokenizer)
    sampler = ReasoningSampler(model, tokenizer)

    # Standard next-token prediction
    outputs = model(batch[:, :-1])
    logits = outputs.logits
    targets = batch[:, 1:]

    # Standard language modeling loss
    lm_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    )

    # Reasoning-augmented objective
    reasoning_losses = []
    total_reward = 0
    num_reasoning_positions = 0

    # Sample which positions to add reasoning
    seq_length = batch.shape[1]
    reasoning_positions = torch.rand(seq_length) < reasoning_probability

    for pos in torch.where(reasoning_positions)[0]:
        if pos + 1 >= seq_length:
            continue  # Can't predict next token at end

        context = batch[:, :pos]  # Everything before this position
        next_token = batch[:, pos + 1]  # Token to predict

        # Sample reasoning
        reasoning_samples = sampler.sample_reasoning(context)

        # Score each reasoning sample
        sample_rewards = []
        for reasoning_seq in reasoning_samples:
            reasoning_tokens = torch.tensor(
                reasoning_seq,
                dtype=torch.long,
                device=batch.device
            ).unsqueeze(0)

            reward, logits_with, logits_without = reward_computer.compute_reasoning_utility(
                context, reasoning_tokens, next_token
            )

            sample_rewards.append(reward)

        # Use best reasoning sample for this position
        best_idx = torch.argmax(torch.stack(sample_rewards))
        best_reward = sample_rewards[best_idx]

        # RL loss: encourage reasoning that improves prediction
        # (This is maximizing reward, so negative for gradient descent)
        reasoning_loss = -best_reward.mean()
        reasoning_losses.append(reasoning_loss)

        total_reward += best_reward.mean().item()
        num_reasoning_positions += 1

    # Combine losses
    if reasoning_losses:
        reasoning_loss = torch.stack(reasoning_losses).mean()
        total_loss = (1 - reasoning_weight) * lm_loss + reasoning_weight * reasoning_loss
        avg_reward = total_reward / num_reasoning_positions
    else:
        total_loss = lm_loss
        avg_reward = 0

    return total_loss, {
        "lm_loss": lm_loss.item(),
        "avg_reward": avg_reward,
        "reasoning_positions": num_reasoning_positions
    }
```

Finally, implement the full pretraining loop with RLP:

```python
def pretrain_with_rlp(
    model,
    train_loader,
    tokenizer,
    num_epochs=3,
    learning_rate=1e-4,
    reasoning_probability_schedule=None
):
    """
    Pretrain LLM with RLP objective.

    Args:
        model: LLM to pretrain
        train_loader: DataLoader with pretraining batches
        tokenizer: Tokenizer
        num_epochs: Number of pretraining epochs
        learning_rate: Optimizer learning rate
        reasoning_probability_schedule: Optional schedule for reasoning_probability
                                       (default: constant 0.3)

    Returns:
        model: Pretrained model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if reasoning_probability_schedule is None:
        # Default: constant probability
        reasoning_probability_schedule = lambda epoch: 0.3

    total_steps = 0

    for epoch in range(num_epochs):
        reasoning_prob = reasoning_probability_schedule(epoch)
        print(f"Epoch {epoch + 1}, reasoning_probability={reasoning_prob:.2f}")

        total_loss = 0
        total_reward = 0
        total_reasoning_pos = 0

        for batch_idx, batch in enumerate(train_loader):
            # Compute RLP loss
            loss, metrics = rlp_training_step(
                model,
                batch,
                tokenizer,
                reasoning_probability=reasoning_prob,
                reasoning_weight=0.3
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_reward += metrics["avg_reward"]
            total_reasoning_pos += metrics["reasoning_positions"]
            total_steps += 1

            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_reward = total_reward / max(total_reasoning_pos, 1)
                print(f"  Step {total_steps}: loss={avg_loss:.4f}, reward={avg_reward:.4f}")

        print(f"Epoch {epoch + 1} complete")

    return model
```

## Practical Guidance

**When to use RLP:**
- Pretraining from scratch on large text corpora
- Building reasoning-capable models early in training
- Scenarios where reasoning utility can be measured (next-token prediction)
- Compute-budget-constrained training (combines two phases into one)

**When NOT to use:**
- Fine-tuning (RLP is a pretraining-phase technique)
- Tasks without clear next-token prediction signal
- Extremely large models where pretraining cost is already optimal
- Non-generative architectures (encoder-only models)

**Performance gains by model size:**

| Model Size | FLOPS | Reasoning Improvement | Overall Performance Gain |
|-----------|--------|---|---|
| 1.7B | Full | +19% on reasoning benchmarks | +19% |
| 7B | Full | +30% on math, +20% on science | +25% |
| 12B | Full | +45% lift on overall | +45% |

**Key hyperparameters:**

| Parameter | Default | Tuning Notes |
|-----------|---------|--------------|
| reasoning_probability | 0.3 | Start at 0.2-0.3; reduce if loss diverges |
| reasoning_weight | 0.3 | Balance reasoning vs. LM loss (try 0.2-0.4) |
| max_reasoning_length | 50 | Longer reasoning = more compute; tune per domain |
| temperature | 0.7 | Controls reasoning diversity (lower = more conservative) |
| num_samples | 3 | More samples = better reward estimates (diminishing returns) |

**Scheduling recommendations:**

```
Epoch 0-1: reasoning_probability=0.5 (heavy reasoning)
Epoch 1-2: reasoning_probability=0.3 (balanced)
Epoch 2+: reasoning_probability=0.1 (sparse reasoning)
```

This curriculum allows the model to learn reasoning early, then transition to direct prediction once confident.

**Common pitfalls:**
- **Reasoning divergence**: If reasoning samples become incoherent, reduce temperature or num_samples.
- **Reward collapse**: All reasoning samples get same low reward (reasoning not useful yet). Early training is normal; it improves with model capability.
- **Computational overhead**: Sampling multiple reasoning traces adds ~2-3x compute. Reduce num_samples or reasoning_probability if budget-constrained.
- **Stale reasoning**: Model gets stuck generating same reasoning. Increase temperature or diversity sampling.

**Integration checklist:**
- [ ] Verify reward computation on 100 examples (rewards should average near 0)
- [ ] Start training with reasoning_probability=0.5 to debug
- [ ] Monitor loss and reward curves over epochs (loss should decrease, reward should increase)
- [ ] Validate reasoning quality on sample outputs (should be coherent)
- [ ] Evaluate on reasoning benchmarks (math, logic) vs. standard pretraining baseline
- [ ] Optional: apply curriculum scheduling (reduce reasoning_probability over time)

Reference: https://arxiv.org/abs/2510.01265
