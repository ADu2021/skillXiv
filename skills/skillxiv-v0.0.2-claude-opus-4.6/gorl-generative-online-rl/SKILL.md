---
name: gorl-generative-online-rl
title: "GoRL: Algorithm-Agnostic Online RL with Generative Policies"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.02581
keywords: [reinforcement-learning, generative-models, policy-learning, online-rl, actor-critic]
description: "Separates generative policy optimization through latent encoder (standard RL algorithms) and conditional decoder (frozen then refined), using two-timescale alternating schedule to eliminate gradient instability from direct generative policy optimization."
---

## Summary

GoRL introduces Generative Online Reinforcement Learning, which addresses instability of expressive generative policies in online RL through structural decoupling. The method separates policy optimization into a learnable latent encoder optimized via standard RL (PPO/GRPO) and a conditional generative decoder refined on improved rollouts, using a two-timescale alternating schedule.

## Core Technique

**Structural Decoupling:** Rather than treating the entire generative policy as a single learnable object, decompose it:
1. **Latent Encoder π_θ(ε|s):** Maps state to latent sample, optimized via standard RL
2. **Conditional Decoder g_φ(s,ε):** Generates actions from latent samples, refined on good trajectories

This avoids backpropagating through complex sampling chains, which causes gradient instability.

**Two-Timescale Alternating:** Use alternating optimization:
```
Phase 1: Optimize encoder via standard RL (PPO)
Phase 2: Refine decoder on trajectories with good returns
Phase 3: Reset encoder, keep improved decoder
Repeat
```

**Gaussian Prior Anchoring:** During decoder refinement, anchor the latent prior to a fixed N(0,I) to prevent distribution shift.

## Implementation

**Encoder-decoder architecture:**
```python
class GoRL_Policy:
    def __init__(self):
        self.encoder = mlp(state_dim -> latent_dim)  # π_θ
        self.decoder = mlp((state_dim + latent_dim) -> action_dim)  # g_φ

    def sample_action(self, state):
        epsilon = randn(latent_dim)  # Sample from N(0,I)
        action = self.decoder(concat(state, epsilon))
        return action
```

**Phase 1 - Encoder RL optimization:**
```python
# Standard policy gradient with encoder
def encoder_loss(states, actions, returns):
    for state, action, ret in zip(states, actions, returns):
        epsilon, log_prob = self.encoder.sample_and_log_prob(state)
        loss = -log_prob * (ret - baseline(state))
    return loss

encoder_optimizer.step(encoder_loss())
```

**Phase 2 - Decoder refinement:**
```python
# Collect good trajectories from current policy
good_trajectories = rollout(epsilon=0.1)  # Only keep good episodes

# Refine decoder to match good trajectories
for state, action in good_trajectories:
    # Sample latent to match action
    epsilon = invert_decoder(state, action)  # Approximate inverse
    decoder_loss = mse(decoder(state, epsilon), action)
    decoder_optimizer.step(decoder_loss)
```

**Two-timescale schedule:**
```python
for iteration in range(num_iterations):
    # Phase 1: RL on encoder (multiple steps)
    for _ in range(encoder_updates_per_phase):
        encoder_optimizer.step(encoder_loss())

    # Phase 2: Refine decoder (multiple steps)
    for _ in range(decoder_updates_per_phase):
        decoder_optimizer.step(decoder_loss())

    # Reset encoder, keep improved decoder
    encoder = reset_to_init(encoder)
```

## When to Use

- Online RL with complex, high-dimensional action spaces
- Scenarios where expressive generative policies improve sample efficiency
- Applications combining standard RL algorithms with generative models
- Tasks where direct gradient through sampling causes instability

## When NOT to Use

- Simple action spaces where deterministic policies suffice
- Scenarios without access to rollout trajectories for decoder refinement
- Real-time RL where alternating optimization overhead is prohibitive
- Applications preferring single unified policy architecture

## Key References

- Generative models for policy representation
- Policy gradient and actor-critic methods
- Online reinforcement learning and sample efficiency
- Two-timescale optimization and alternating algorithms
