---
name: swirl-self-improving-world-models
title: "Self-Improving World Modelling with Latent Actions"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.06130"
keywords: [World Models, Self-Improvement, Latent Actions, Inverse Dynamics, Unlabeled Data]
description: "Enable VLMs and LLMs to self-improve at world modeling by treating actions as latent variables and reciprocally optimizing forward and inverse dynamics models using only unlabeled state transitions."
---

# Self-Improving World Modelling with Latent Actions

## Problem Context

Foundation models (LLMs and VLMs) struggle with world modeling because current approaches require costly action-labeled trajectories. Collecting manual annotations for every state transition in open-world tasks is prohibitively expensive and intractable. Moreover, inverse dynamics (inferring actions from state transitions) is inherently ambiguous—purely supervised learning becomes brittle when data is sparse.

## Core Concept

SWIRL treats actions as [latent variables, reciprocal optimization, unlabeled data] to enable self-improvement without annotations. The approach alternates between optimizing the Forward World Model (FWM) to predict identifiable states and optimizing the Inverse Dynamics Model (IDM) to infer plausible actions. This mutual-information maximization enables both models to improve from unlabeled state-only sequences.

## Architecture Overview

- **Phase I**: FWM optimization maximizing state identifiability (conditional MI)
- **Phase II**: IDM optimization maximizing action inference (ELBO)
- **Reciprocal loop**: Each phase provides learning signal for the other
- **Latent action framework**: Actions inferred from transitions, not labeled
- **GRPO training**: Both models optimized via reinforcement learning with self-generated rewards

## Implementation

### Step 1: Define forward world model and inverse dynamics model

Implement both model components that cooperatively learn.

```python
# Forward World Model (FWM)
class ForwardWorldModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.encoder = torch.nn.Linear(state_dim, hidden_dim)
        self.action_encoder = torch.nn.Linear(action_dim, hidden_dim)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        """Predict next state given state and action."""
        state_emb = self.encoder(state)
        action_emb = self.action_encoder(action)
        combined = torch.cat([state_emb, action_emb], dim=-1)
        next_state = self.predictor(combined)
        return next_state

# Inverse Dynamics Model (IDM)
class InverseDynamicsModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.encoder = torch.nn.Linear(state_dim * 2, hidden_dim)  # (state, next_state)
        self.action_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward(self, state, next_state):
        """Infer action from state transition."""
        combined = torch.cat([state, next_state], dim=-1)
        encoding = self.encoder(combined)
        action_mean = self.action_decoder(encoding)
        action_log_std = self.log_std
        return action_mean, action_log_std

    def log_prob(self, action, state, next_state):
        """Compute log probability of action given transition."""
        action_mean, action_log_std = self.forward(state, next_state)
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist.log_prob(action).sum(dim=-1)
```

### Step 2: Implement Phase I - FWM optimization

Optimize FWM to generate next states that are identifiable by the IDM.

```python
# Phase I: Forward World Model optimization
def phase_i_optimize_fwm(
    fwm, idm, state_transitions,
    optimizer_fwm, num_steps=100
):
    """
    Phase I: Optimize FWM to maximize conditional MI I(Z; Ŷ | X).

    Args:
        fwm: Forward World Model
        idm: Inverse Dynamics Model (frozen)
        state_transitions: Tensor of shape (batch_size, 2, state_dim)
                          containing [state, next_state] pairs
        optimizer_fwm: Optimizer for FWM
        num_steps: Number of optimization steps
    """
    for step in range(num_steps):
        # Sample random actions (latent variables)
        batch_size = state_transitions.shape[0]
        action_dim = 4  # Assuming 4D action space
        sampled_actions = torch.randn(batch_size, action_dim)

        state = state_transitions[:, 0, :]
        next_state_real = state_transitions[:, 1, :]

        # Forward model predicts next state
        next_state_pred = fwm(state, sampled_actions)

        # Inverse model evaluates identifiability
        # High log_prob from IDM = good prediction
        idm_log_prob = idm.log_prob(sampled_actions, state, next_state_pred)

        # Loss: maximize identifiability (negative for gradient descent)
        loss = -idm_log_prob.mean()

        # Backward pass
        optimizer_fwm.zero_grad()
        loss.backward()
        optimizer_fwm.step()

        if step % 20 == 0:
            print(f"  Phase I, Step {step}: Loss={loss.item():.4f}")

    return fwm
```

### Step 3: Implement Phase II - IDM optimization

Optimize IDM to infer actions that maximize the likelihood of observed transitions.

```python
# Phase II: Inverse Dynamics Model optimization
def phase_ii_optimize_idm(
    fwm, idm, state_transitions,
    optimizer_idm, num_steps=100
):
    """
    Phase II: Optimize IDM to maximize ELBO of log p(a | s, s').

    Args:
        fwm: Forward World Model (frozen)
        idm: Inverse Dynamics Model
        state_transitions: Tensor of shape (batch_size, 2, state_dim)
        optimizer_idm: Optimizer for IDM
        num_steps: Number of optimization steps
    """
    for step in range(num_steps):
        state = state_transitions[:, 0, :]
        next_state_real = state_transitions[:, 1, :]

        # Sample actions from IDM posterior
        action_mean, action_log_std = idm(state, next_state_real)
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        sampled_actions = dist.rsample()

        # Forward model predicts next state from sampled action
        next_state_pred = fwm(state, sampled_actions)

        # Reconstruction loss: how well does the action reconstruct the transition?
        recon_loss = torch.nn.functional.mse_loss(next_state_pred, next_state_real)

        # KL regularization: encourage IDM to stay close to prior (standard normal)
        kl_loss = torch.distributions.kl_divergence(
            dist, torch.distributions.Normal(torch.zeros_like(action_mean), torch.ones_like(action_std))
        ).sum(dim=-1).mean()

        # ELBO = reconstruction + KL
        loss = recon_loss + 0.01 * kl_loss

        # Backward pass
        optimizer_idm.zero_grad()
        loss.backward()
        optimizer_idm.step()

        if step % 20 == 0:
            print(f"  Phase II, Step {step}: Loss={loss.item():.4f}, "
                  f"Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}")

    return idm
```

### Step 4: Implement reciprocal optimization loop

Alternate between phases until convergence.

```python
# Reciprocal optimization
def train_swirl(
    fwm, idm, state_transitions,
    optimizer_fwm, optimizer_idm,
    num_iterations=10, steps_per_phase=50,
    device='cuda'
):
    """
    Main SWIRL training loop: alternate between Phase I and Phase II.
    """
    state_transitions = state_transitions.to(device)
    fwm = fwm.to(device)
    idm = idm.to(device)

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")

        # Phase I: Optimize FWM
        print("  Phase I: Optimizing FWM...")
        fwm = phase_i_optimize_fwm(
            fwm, idm, state_transitions,
            optimizer_fwm, num_steps=steps_per_phase
        )

        # Phase II: Optimize IDM
        print("  Phase II: Optimizing IDM...")
        idm = phase_ii_optimize_idm(
            fwm, idm, state_transitions,
            optimizer_idm, num_steps=steps_per_phase
        )

        # Evaluate on test set
        test_loss = evaluate_world_models(fwm, idm, state_transitions)
        print(f"  Test loss: {test_loss:.4f}\n")

    return fwm, idm
```

### Step 5: Integrate with GRPO for end-to-end learning

Use RL optimization for both models with self-generated rewards.

```python
# GRPO-based SWIRL training
class SWIRLWithGRPO:
    def __init__(self, fwm, idm, verifier):
        self.fwm = fwm
        self.idm = idm
        self.verifier = verifier

    def compute_world_model_reward(self, state, next_state_pred, next_state_real):
        """
        Compute reward based on prediction accuracy.
        Could also use downstream task performance.
        """
        prediction_error = torch.norm(next_state_pred - next_state_real, dim=-1)
        reward = 1.0 / (1.0 + prediction_error)
        return reward

    def grpo_step_fwm(self, states, actions, next_states, optimizer):
        """
        GRPO step for FWM optimization.
        """
        # Generate predictions
        next_state_preds = self.fwm(states, actions)

        # Compute rewards
        rewards = self.compute_world_model_reward(
            states, next_state_preds, next_states
        )

        # GRPO loss (standard advantage estimation)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        loss = -(advantages * torch.log_softmax(next_state_preds, dim=-1).mean(dim=-1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def grpo_step_idm(self, states, next_states, optimizer):
        """
        GRPO step for IDM optimization.
        """
        # Generate action predictions
        action_mean, action_log_std = self.idm(states, next_states)

        # Compute inverse model quality (likelihood)
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(torch.zeros_like(action_mean)).sum(dim=-1)

        # GRPO advantage
        advantages = (log_prob - log_prob.mean()) / (log_prob.std() + 1e-8)
        loss = -advantages.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
```

## Practical Guidance

**When to use**: VLM/LLM-based world modeling tasks where action-labeled data is unavailable or expensive. Works for video understanding, embodied reasoning, environment simulation.

**Hyperparameters**:
- **steps_per_phase**: 50-100 (balance between convergence and computational cost)
- **num_iterations**: 5-20 (typically converges in 10-15)
- **KL weight in Phase II**: 0.01-0.1 (controls action prior strength)
- **Action dimension**: Task-dependent (4-8 typical)

**Key empirical findings**:
- Improvements across 6 benchmarks: Aurora-Bench (+16%), ByteMorph (+28%), WorldPredictionBench (+16%), StableToolBench (+14%)
- Works without action labels; reduces annotation burden
- Self-improvement continues for 5-10 iterations
- Benefits scale with data quality, not quantity

**Common pitfalls**:
- Skipping initialization with supervised fine-tuning → slow convergence
- Using highly stochastic actions → IDM becomes ill-posed
- Imbalanced optimization (one model converging much faster) → divergence
- Not validating on downstream tasks → may optimize for reconstruction without semantic meaning

**Scaling**: Linear with state/action dimensions. Tested on 3B-7B models; scaling to 70B+ untested.

## Reference

Paper: https://arxiv.org/abs/2602.06130
Code: Promised to be released
Related work: World models, inverse models, latent variable models
Benchmarks: Aurora-Bench, ByteMorph, WorldPredictionBench, StableToolBench
