---
name: rlp-reinforcement-pretraining-objective
title: "RLP: Reinforcement as Pretraining Objective"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2510.01265
keywords: [pretraining, RLVR, reasoning, information-gain, generalization]
description: "Improve reasoning during pretraining (not just post-training) by computing rewards from information gain—how much reasoning improves log-likelihood of observed tokens. Works at 1T token scale across diverse corpora."
---

# RLP: Reinforcement as Pretraining Objective

RLP introduces verifier-free reinforcement during pretraining itself. Rather than treating reasoning as post-training only, the approach computes dense reward signals from information gain: how much sampled chain-of-thought improves prediction of the next observed token. This enables reasoning to scale with pretraining data itself.

## Core Architecture

- **Information gain reward**: No external verifier; reward = ΔLogLikelihood
- **Reasoning-before-prediction**: Sample CoT tokens, then next-token prediction
- **Verifier-free**: Dense rewards from data signal itself
- **Efficient training**: Runs on standard pretraining infrastructure
- **Scale**: Tested at 1T tokens with multiple model sizes

## Implementation Steps

Setup reinforcement pretraining objective:

```python
# Initialize RLP pretraining trainer
from rlp_pretrain import ReasoningPretrainer, InformationGainReward

pretrainer = ReasoningPretrainer(
    model_config="your_transformer_config",
    reasoning_tokens_per_position=G,  # G samples per position
    reasoning_max_length=2048
)

# Define information gain reward
reward_function = InformationGainReward(
    use_ema_teacher=True,
    ema_decay=0.999,  # EMA snapshot for baseline
    importance_sampling=True
)

# Initialize with standard pretraining setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()
```

Execute reinforcement pretraining:

```python
# Training loop with reasoning and next-token prediction
for step, batch in enumerate(pretraining_dataloader):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    with torch.amp.autocast(device_type="cuda"):
        # Standard forward pass
        logits = model(input_ids, attention_mask)

        # For each position, sample reasoning before prediction
        reasoning_rewards = []

        for pos in range(input_ids.shape[1] - 1):
            # Get predicted token at position pos
            next_token_id = input_ids[:, pos + 1]

            # Sample G reasoning trajectories
            # (In practice, use conditional sampling from model)
            reasoning_samples = model.sample_reasoning(
                context=input_ids[:, :pos],
                num_samples=G,  # typically G=16
                max_length=min(2048, max_reasoning_length)
            )

            # For each reasoning trajectory, compute conditional probability
            log_probs_with_reasoning = []
            for reasoning in reasoning_samples:
                # Concatenate reasoning + input to position pos
                augmented_context = torch.cat(
                    [input_ids[:, :pos], reasoning],
                    dim=1
                )

                # Predict next token probability with reasoning
                augmented_logits = model(augmented_context, attention_mask=None)
                log_prob = F.log_softmax(augmented_logits[:, -1], dim=-1)[next_token_id]
                log_probs_with_reasoning.append(log_prob)

            # Compute information gain for each reasoning trajectory
            baseline_log_prob = F.log_softmax(logits[:, pos], dim=-1)[next_token_id]

            for log_prob_with_reasoning in log_probs_with_reasoning:
                information_gain = log_prob_with_reasoning - baseline_log_prob
                reasoning_rewards.append(information_gain)

        # Aggregate rewards for policy update
        reasoning_rewards = torch.stack(reasoning_rewards)

        # RLVR loss (policy gradient with baseline)
        log_probs_reasoning = model.compute_log_probs(reasoning_samples)
        rl_loss = -(log_probs_reasoning * (reasoning_rewards - reward_function.baseline)).mean()

        # Standard next-token prediction loss
        ntp_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1)
        )

        # Combined loss
        total_loss = ntp_loss + rl_loss_weight * rl_loss

    # Backward pass
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # Update EMA teacher for reward baseline
    reward_function.update_ema_teacher(model)
```

## Practical Guidance

**When to use RLP:**
- Building foundation models where reasoning is important
- Scenarios where pretraining data naturally contains reasoning (code, math, papers)
- Settings where post-training compute budget limited
- Models targeting reasoning-heavy downstream tasks

**When NOT to use:**
- Domains where reasoning not naturally in data (image, audio pretraining)
- Purely autoregressive models where reasoning overhead excessive
- Real-time pretraining requiring minimal computational overhead
- Models where post-training RL already sufficient

**Hyperparameters:**
- **G (num reasoning samples, 16)**: Tradeoff between quality and compute. Test 4-32 range
- **Reasoning max length (2048)**: Allow ample reasoning space; reduce to 512 for speed
- **RL loss weight**: Start at 0.1; increase to 0.3-0.5 if model relies on reasoning
- **EMA decay (0.999)**: Teacher staleness control; increase to 0.9999 for stability
- **Importance sampling**: Enable for large batch sizes; disable for memory constraints
- **Completion length (2048)**: Max length for model reasoning; match to pretraining context

## Data Composition Effects

Performance varies by corpus type:
- **Code/math corpora**: +8-12% improvements (reasoning naturally supported)
- **Academic papers**: +5-8% improvements (structured reasoning present)
- **Web crawl**: +2-4% improvements (less inherent reasoning)
- **General text**: +1-3% improvements (minimal reasoning signals)

## Computational Characteristics

- **Per-token overhead**: G forward passes for reasoning sampling (G=16 → 16x)
- **Mitigation**: Amortize over batches; use gradient accumulation
- **Effective cost**: ~2-3x pretraining cost (with optimizations)
- **Benefit**: Reasoning capability without separate post-training phase

## Generalization Properties

Trained models show strong cross-domain generalization:
- **In-distribution**: +5-8% gains on matching task types
- **Out-of-distribution**: +3-5% gains on unseen domains
- **Scale**: Benefits persist from 1.7B to 12B parameters
- **Architecture robustness**: Works with Transformers and Mamba-Transformers

## Information Gain as Reward Signal

Key insight: next-token probability improvement quantifies reasoning quality:
- High information gain: Reasoning contributes to data likelihood
- Low information gain: Reasoning irrelevant or noisy
- Dense signal: Every position provides reward feedback
- Verifier-free: No external correctness model needed

## Baseline Management

EMA teacher prevents distribution shift during training:
- **Detached targets**: No gradient flow through baseline
- **Exponential averaging**: Smooth updates prevent instability
- **Architecture matching**: Same model for baseline and policy

## References

Builds on pretraining objectives, policy gradients, and information-theoretic reward design.
