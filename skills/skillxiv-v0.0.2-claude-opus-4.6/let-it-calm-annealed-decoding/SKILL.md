---
name: let-it-calm-annealed-decoding
title: "Let it Calm: Exploratory Annealed Decoding for Verifiable Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.05251"
keywords: [Decoding Strategy, Temperature Scheduling, Reinforcement Learning, Exploration, Training Stability]
description: "Use dynamic temperature scheduling that explores early (high temperature) and exploits late (low temperature) during generation, improving sample efficiency in RL with verifiable rewards."
---

# Technique: Exploratory-then-Exploitative Temperature Scheduling

Exploration isn't equally valuable at all points during sequence generation. Early tokens define semantic direction—high uncertainty here drives diverse meaningful outputs. Later tokens fill details where exploration adds noise. Let it Calm (EAD) implements this insight through temperature scheduling that starts warm (explore early) and cools down (exploit late).

This simple but effective approach improves sample efficiency in RLVR (RL with verifiable rewards) by encouraging semantic diversity when it matters while maintaining training stability through low-entropy later generation.

## Core Concept

Exploratory Annealed Decoding operates through three mechanisms:

1. **Dynamic Temperature Schedule**: Start high, decay over sequence length τₜ = max{τ_init - e^(t/d), τ_min}

2. **Global-Step Awareness**: Adjust decay rate based on training progress (longer responses later need slower decay)

3. **Truncated Importance Sampling**: Correct off-policy issues from aggressive annealing

## Architecture Overview

- **Initial Temperature**: Start with τ > 1.0 for exploration
- **Time-Dependent Decay**: Linear-exponential schedule over token positions
- **Training-Aware Adaptation**: Slower decay as model improves
- **IS Correction**: Prevent training instability from distribution mismatch
- **RL Integration**: Plug into GRPO, DAPO, or other RLVR algorithms

## Implementation Steps

Implement the dynamic temperature schedule.

```python
def get_temperature_schedule(sequence_length, global_step,
                            tau_init=1.5, tau_min=0.5, decay_rate=0.1):
    """
    Compute temperature for each timestep in generation.

    Args:
        sequence_length: Length of sequence being generated
        global_step: Current training step
        tau_init: Initial temperature (usually > 1.0)
        tau_min: Minimum temperature
        decay_rate: Decay rate parameter

    Returns:
        temperatures: Array of temperatures for each token position
    """

    import numpy as np

    temperatures = []

    for t in range(sequence_length):
        # Main schedule: decay from tau_init to tau_min
        # Exponential decay: starts steep, then flattens
        decay = np.exp(-t / (decay_rate * sequence_length))
        tau_t = tau_min + (tau_init - tau_min) * decay

        # Adjust for training progress: longer sequences later = slower decay
        # This prevents entropy collapse as model gets better
        if global_step > 0:
            progress_factor = min(1.0, (global_step / 10000) ** 0.5)
            tau_t = tau_min + (tau_t - tau_min) * progress_factor

        temperatures.append(tau_t)

    return np.array(temperatures)
```

Implement sampling with temperature scheduling.

```python
def sample_with_annealed_temperature(model, prompt, sequence_length,
                                    global_step, return_log_probs=False):
    """
    Generate sequence using annealed temperature schedule.

    Args:
        model: Language model
        prompt: Starting prompt
        sequence_length: Number of tokens to generate
        global_step: Current training step
        return_log_probs: Return log probabilities for RL

    Returns:
        sequence: Generated token sequence
        log_probs: Log probabilities (if requested)
    """

    import torch
    import torch.nn.functional as F

    temperatures = get_temperature_schedule(sequence_length, global_step)

    tokens = []
    log_probs = []
    current_input = prompt

    for t in range(sequence_length):
        # Get logits
        with torch.no_grad():
            output = model(current_input)
            logits = output.logits[:, -1, :]  # Last token position

        # Apply temperature
        scaled_logits = logits / temperatures[t]

        # Sample from distribution
        probs = F.softmax(scaled_logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)

        # Compute log probability under original (not scaled) distribution
        log_prob = F.log_softmax(logits, dim=-1)[0, token.item()]

        tokens.append(token.item())
        log_probs.append(log_prob.item())

        # Append to input for next step
        current_input = torch.cat([current_input, token], dim=-1)

    sequence = tokens
    if return_log_probs:
        return sequence, torch.tensor(log_probs)
    return sequence
```

Implement importance sampling correction for off-policy training.

```python
def importance_sampling_correction(log_probs, target_logits, tau_schedule,
                                  clip_ratio=1.0):
    """
    Correct for distribution shift caused by temperature annealing.

    Args:
        log_probs: Log probabilities under annealed distribution
        target_logits: Logits under standard distribution (tau=1.0)
        tau_schedule: Temperature values used during sampling
        clip_ratio: Clipping ratio for IS weight

    Returns:
        corrected_log_probs: Importance-weighted log probabilities
    """

    import torch
    import torch.nn.functional as F

    # Compute log probabilities under standard distribution
    standard_log_probs = []
    for logits, tau in zip(target_logits, tau_schedule):
        standard_logp = F.log_softmax(logits, dim=-1)
        standard_log_probs.append(standard_logp)

    # Importance sampling weights
    is_weights = []
    for lp_annealed, lp_standard in zip(log_probs, standard_log_probs):
        weight = torch.exp(lp_standard - lp_annealed)
        # Clip to prevent extreme weights
        weight = torch.clamp(weight, max=clip_ratio)
        is_weights.append(weight)

    is_weights = torch.stack(is_weights)

    # Weighted log probabilities
    corrected_lp = log_probs * is_weights

    return corrected_lp
```

Integrate into RLVR training loop.

```python
def train_with_annealed_decoding(model, verifier, optimizer, prompts,
                                num_epochs=3, temperature_params=None):
    """
    Train model using EAD with verifiable reward RL.

    Args:
        model: Language model
        verifier: Reward signal (verifiable)
        optimizer: PyTorch optimizer
        prompts: Training prompts
        num_epochs: Training epochs
        temperature_params: Dict with tau_init, tau_min, decay_rate

    Returns:
        metrics: Training metrics
    """

    import torch

    if temperature_params is None:
        temperature_params = {'tau_init': 1.5, 'tau_min': 0.5, 'decay_rate': 0.1}

    metrics = {'epoch_loss': [], 'avg_reward': []}

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_rewards = []

        for prompt_idx, prompt in enumerate(prompts):
            global_step = epoch * len(prompts) + prompt_idx

            # Generate with annealed temperature
            sequence, log_probs = sample_with_annealed_temperature(
                model, prompt, sequence_length=100,
                global_step=global_step, return_log_probs=True
            )

            # Verify and get reward
            reward = verifier.verify(sequence)
            epoch_rewards.append(reward)

            # Policy gradient loss: maximize log_prob * reward
            loss = -(log_probs.mean() * reward)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        metrics['epoch_loss'].append(epoch_loss / len(prompts))
        metrics['avg_reward'].append(sum(epoch_rewards) / len(epoch_rewards))

    return metrics
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Initial temperature τ_init | 1.2-2.0 | Higher = more exploration; adjust based on task diversity |
| Minimum temperature τ_min | 0.5-0.8 | Lower = more focused; too low can collapse diversity early |
| Decay rate | 0.05-0.15 | Controls how quickly temperature drops |
| Training progression adjustment | Increase decay as training progresses | Prevents entropy collapse later in training |
| When to use | RL training with verifiable rewards (math, coding) | RLVR on reasoning tasks |
| When NOT to use | Inference where determinism is preferred | Training stages where diversity isn't valued |
| Common pitfall | Temperature too high/low throughout | Requires tuning per domain |

### When to Use EAD

- RL training on reasoning tasks with verifiable rewards (math, code, logic)
- Scenarios where semantic diversity matters (multiple valid solutions)
- Training where entropy collapse is a known problem
- Tasks combining exploration and exploitation naturally

### When NOT to Use EAD

- Deterministic inference where quality is paramount
- Tasks with single correct answer where diversity is counterproductive
- Real-time systems where variable temperature adds overhead

### Common Pitfalls

- **Temperature too aggressive**: Early tokens frozen, no exploration benefit
- **Entropy collapse**: Minimum temperature too low; increase τ_min
- **Training instability**: Importance sampling correction insufficient; increase clip_ratio
- **Poor calibration**: Temperature schedule very task-dependent; requires tuning per domain

## Reference

Paper: https://arxiv.org/abs/2510.05251
