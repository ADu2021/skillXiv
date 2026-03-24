---
name: ramp-adaptive-quantization
title: "RAMP: Reinforcement Adaptive Mixed Precision Quantization for On-Device LLM Inference"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.17891"
keywords: [Quantization, Mixed Precision, On-Device Inference, Reinforcement Learning]
description: "Learn optimal per-layer bit-width assignments for LLM quantization via RL, generalizing across models without retraining. Achieves superior compression under fixed bit budgets."
---

# RAMP: Reinforcement Adaptive Mixed Precision Quantization

Quantizing large language models for on-device inference requires assigning different bit-widths to different layers—a combinatorial optimization problem that standard uniform quantization handles poorly. Layers vary dramatically in their sensitivity to quantization: some can survive 2-bit precision while others need 8-bits to maintain quality.

RAMP solves this through an off-policy reinforcement learning approach that learns which layers should receive which bit-widths. The key innovation is that the learned policy *generalizes across models*—a policy trained on Llama 2 7B zero-shot transfers to Llama 2 13B and Mistral 7B without retraining. This is possible because quantization sensitivity is fundamentally architectural rather than model-specific.

## Core Concept

RAMP reframes mixed-precision quantization as a Markov Decision Process (MDP):

**State:** Layer index + activation statistics (11-dim embedding capturing weight properties, activation distributions, structural info)

**Action:** Bit-width assignment (typically 2, 4, 6, or 8 bits)

**Reward:** Quality-prioritized with asymmetric penalties: missing a quality target is costlier than exceeding a computational budget

**Policy:** Learned via off-policy Soft Actor-Critic (SAC) RL; outputs optimal bit-width for each layer given state

The state representation is model-agnostic, enabling zero-shot transfer. The reward function is shaped to handle the tradeoff between model quality (perplexity) and model size (bit budget).

## Architecture Overview

- **State Encoder**: 11-dimensional embedding of activation/weight statistics (model-agnostic)
- **Policy Network**: Maps state -> bit-width distribution via SAC
- **Value Networks**: Q-functions for off-policy learning
- **Reward Shaping**: Multi-objective: minimize perplexity loss, respect bit budget
- **Scale Folding**: Novel preconditioning that stabilizes sub-4-bit quantization
- **Zero-Shot Transfer**: Learned policy applies to different model architectures

## Implementation Steps

### Step 1: Compute Model-Agnostic State Embedding

Extract 11-dimensional state features from model layers.

```python
import torch
import torch.nn as nn
import numpy as np

def compute_layer_state_embedding(layer, activation_sample):
    """
    Compute 11-dimensional state embedding for RL policy.
    Captures layer characteristics in model-agnostic form.

    layer: torch.nn.Module (e.g., Linear layer)
    activation_sample: (batch_size, hidden_dim) tensor of activations through this layer
    """
    state_features = []

    # 1. Weight magnitude statistics (mean, std, max)
    weight = layer.weight.data
    state_features.append(weight.abs().mean().item())
    state_features.append(weight.abs().std().item())
    state_features.append(weight.abs().max().item())

    # 2. Activation magnitude statistics
    state_features.append(activation_sample.abs().mean().item())
    state_features.append(activation_sample.abs().std().item())
    state_features.append(activation_sample.abs().max().item())

    # 3. Layer structure (param count, input/output dims, layer type)
    param_count = sum(p.numel() for p in layer.parameters())
    state_features.append(np.log10(max(param_count, 1)))  # Log scale

    # 4. Input-output dimension ratio
    if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
        dim_ratio = layer.out_features / max(layer.in_features, 1)
        state_features.append(dim_ratio)
    else:
        state_features.append(1.0)

    # 5. Sparsity of activations
    sparsity = (activation_sample.abs() < 1e-6).float().mean().item()
    state_features.append(sparsity)

    # 6. Dynamic range (max / min non-zero)
    nonzero = activation_sample[activation_sample.abs() > 1e-8]
    if len(nonzero) > 0:
        dynamic_range = nonzero.abs().max() / (nonzero.abs().min() + 1e-8)
        state_features.append(np.log10(dynamic_range))
    else:
        state_features.append(0.0)

    # 7. Outlier ratio (percent beyond 2 std dev)
    outlier_ratio = (activation_sample.abs() > 2 * activation_sample.std()).float().mean().item()
    state_features.append(outlier_ratio)

    return torch.tensor(state_features, dtype=torch.float32)
```

### Step 2: Implement Scale Folding for Stability

Pre-condition the model to enable sub-4-bit quantization.

```python
class ScaleFoldingPreconditioning:
    """
    Scale folding: migrates activation outliers into weights via per-channel scaling.
    Stabilizes sub-4-bit quantization by reducing dynamic range of activations.
    """

    @staticmethod
    def compute_per_channel_scales(activations):
        """
        Compute per-channel scaling factors from activation statistics.
        activations: (batch_size, hidden_dim)
        Returns: (hidden_dim,) scaling factors
        """
        # Per-channel scaling: scale = std / target_std
        channel_stds = activations.std(dim=0, keepdim=True)
        target_std = channel_stds.median()  # Normalize to median

        scales = target_std / (channel_stds + 1e-8)
        return scales.squeeze()

    @staticmethod
    def apply_scale_folding(layer, scales):
        """
        Apply scale folding by adjusting weights and scales.
        layer: torch.nn.Module with weight and bias
        scales: (out_features,) scaling factors
        """
        # Absorb scales into weight matrix
        # For Linear: weight_new = weight / scales
        if hasattr(layer, 'weight'):
            layer.weight.data = layer.weight.data / (scales.unsqueeze(1) + 1e-8)

        # If layer has bias, adjust accordingly
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data = layer.bias.data / (scales + 1e-8)

        return scales

    @staticmethod
    def fold_activations(x, scales):
        """
        Fold scales into activations (multiply by scales).
        x: (batch_size, hidden_dim) activations
        scales: (hidden_dim,) pre-computed scales
        Returns: scaled activations
        """
        return x * scales.unsqueeze(0)
```

### Step 3: Define RL Environment for Quantization

Set up the MDP for bit-width optimization.

```python
class QuantizationEnvironment:
    """
    RL environment for mixed-precision quantization.
    State: layer embedding | Action: bit-width | Reward: perplexity - budget_penalty
    """

    def __init__(self, model, calibration_data, target_bit_budget=8.0 * 1024):
        self.model = model
        self.calibration_data = calibration_data
        self.target_bit_budget = target_bit_budget
        self.num_layers = len(list(model.parameters()))

        self.current_layer_idx = 0
        self.bit_assignments = {}
        self.layer_states = {}

        # Pre-compute layer embeddings
        self._compute_all_layer_states()

    def _compute_all_layer_states(self):
        """Pre-compute state embeddings for all layers."""
        for idx, (name, layer) in enumerate(self.model.named_modules()):
            if isinstance(layer, torch.nn.Linear):
                # Get a sample of activations
                with torch.no_grad():
                    activations = self._get_layer_activations(layer)
                state_embedding = compute_layer_state_embedding(layer, activations)
                self.layer_states[idx] = state_embedding

    def _get_layer_activations(self, target_layer):
        """Extract activations for a specific layer."""
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            for batch in self.calibration_data[:10]:  # Use first 10 batches
                self.model(batch)
        handle.remove()

        return torch.cat(activations, dim=0)

    def reset(self):
        """Start quantization from first layer."""
        self.current_layer_idx = 0
        self.bit_assignments = {}
        return self.layer_states[0]

    def step(self, bit_width_action):
        """
        Assign bit-width to current layer, move to next.
        bit_width_action: int (2, 4, 6, or 8)
        Returns: (next_state, reward, done)
        """
        self.bit_assignments[self.current_layer_idx] = bit_width_action
        self.current_layer_idx += 1

        # Get next state
        if self.current_layer_idx >= self.num_layers:
            done = True
            next_state = None
        else:
            done = False
            next_state = self.layer_states.get(self.current_layer_idx)

        # Compute reward
        reward = self._compute_reward(done)

        return next_state, reward, done

    def _compute_reward(self, done):
        """
        Compute reward balancing quality and budget.
        Quality (perplexity) is strongly prioritized over budget.
        """
        if not done:
            return 0.0  # No intermediate rewards

        # Compute total bit budget used
        total_bits = sum(
            bits * self._get_layer_param_count(layer_idx)
            for layer_idx, bits in self.bit_assignments.items()
        )

        # Compute quality loss (perplexity on calibration set)
        perplexity = self._compute_perplexity_quantized()

        # Reward: maximize quality (minimize perplexity), respect budget
        quality_reward = -perplexity  # Negative because we minimize
        budget_penalty = max(0, (total_bits - self.target_bit_budget) / self.target_bit_budget) * 100

        # Quality strongly prioritized (10x weight on quality vs budget)
        reward = quality_reward - 0.1 * budget_penalty

        return reward

    def _get_layer_param_count(self, layer_idx):
        """Get number of parameters in a layer."""
        layer = list(self.model.modules())[layer_idx]
        return sum(p.numel() for p in layer.parameters()) / 1e6  # In millions

    def _compute_perplexity_quantized(self):
        """Evaluate perplexity of quantized model on calibration set."""
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch_input, batch_target in self.calibration_data:
                logits = self.model(batch_input)
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), batch_target.view(-1))
                total_loss += loss.item() * batch_target.numel()
                total_tokens += batch_target.numel()

        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
```

### Step 4: Train RL Policy with Soft Actor-Critic

Use off-policy RL to learn optimal bit-width assignments.

```python
from torch.distributions import Categorical

class SAC_QuantizationPolicy:
    """
    Soft Actor-Critic for learning bit-width assignments.
    Off-policy allows zero-shot transfer across models.
    """

    def __init__(self, state_dim=11, num_actions=4, learning_rate=1e-4):
        self.state_dim = state_dim
        self.num_actions = num_actions  # 4 actions: 2, 4, 6, 8 bits

        # Actor network: state -> action probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        # Q-networks for value estimation
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=learning_rate
        )

    def select_action(self, state, temperature=1.0):
        """Select action with exploration."""
        logits = self.actor(state)
        probs = torch.softmax(logits / temperature, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), probs

    def train_step(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_done):
        """Single training step with SAC loss."""
        # Q-function loss
        state_action = torch.cat([batch_states, batch_actions], dim=-1)
        q1_pred = self.q1(state_action)
        q2_pred = self.q2(state_action)

        target_q = batch_rewards + 0.99 * (1 - batch_done.float()) * torch.min(
            self.q1(torch.cat([batch_next_states, self._greedy_action(batch_next_states)], dim=-1)),
            self.q2(torch.cat([batch_next_states, self._greedy_action(batch_next_states)], dim=-1))
        )

        q_loss = nn.MSELoss()(q1_pred, target_q) + nn.MSELoss()(q2_pred, target_q)

        self.optimizer_q.zero_grad()
        q_loss.backward()
        self.optimizer_q.step()

        # Actor loss (entropy regularized)
        logits = self.actor(batch_states)
        entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        actor_loss = -entropy  # Maximize entropy

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return q_loss.item(), actor_loss.item()

    def _greedy_action(self, states):
        """Select greedy action (max probability)."""
        logits = self.actor(states)
        actions = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.nn.functional.one_hot(actions.squeeze(-1), num_classes=self.num_actions).float()

    def get_bit_assignments(self, layer_states):
        """Get optimal bit-widths for all layers (greedy policy evaluation)."""
        bit_map = {0: 2, 1: 4, 2: 6, 3: 8}
        assignments = {}

        with torch.no_grad():
            for layer_idx, state in layer_states.items():
                action_idx = torch.argmax(self.actor(state.unsqueeze(0))).item()
                assignments[layer_idx] = bit_map[action_idx]

        return assignments
```

### Step 5: Zero-Shot Transfer Evaluation

Validate policy generalizes to new models without retraining.

```python
def evaluate_policy_zero_shot(policy, source_model, target_models):
    """
    Evaluate if policy trained on source_model transfers to target_models.
    """
    results = {}

    for target_name, target_model in target_models.items():
        # Compute layer states for target model
        target_states = {}
        for idx, layer in enumerate(target_model.modules()):
            if isinstance(layer, torch.nn.Linear):
                activations = get_target_activations(target_model)
                state = compute_layer_state_embedding(layer, activations)
                target_states[idx] = state

        # Use policy without retraining
        bit_assignments = policy.get_bit_assignments(target_states)

        # Evaluate perplexity with assigned bit-widths
        perplexity = evaluate_quantized_model(target_model, bit_assignments)

        results[target_name] = {
            'bit_assignments': bit_assignments,
            'perplexity': perplexity
        }

    return results
```

## Practical Guidance

**Hyperparameters:**
- State dimension: 11 (fixed, model-agnostic)
- Bit-width options: [2, 4, 6, 8] (common for on-device models)
- Target bit budget: scale with model size (e.g., 8 bits * model params in billions)
- SAC temperature: 1.0 (exploration) during training, 0.0 (greedy) during evaluation
- Scale folding scaling factor: 1.0-2.0 (controls how aggressively to fold)

**When to Use:**
- On-device LLM deployment (mobile, edge devices)
- Need for mixed-precision quantization with dynamic per-layer decisions
- Have calibration data and want to optimize bit-width allocation
- Planning to deploy to multiple model architectures (zero-shot transfer)

**When NOT to Use:**
- Server-side inference where 16-bit or FP32 is standard
- Models with highly irregular architectures (state embedding may not generalize)
- Very small models where quantization overhead dominates
- Real-time quantization (training the policy takes time upfront)

**Pitfalls:**
- Scale folding can be unstable: monitor activation ranges after folding
- Calibration data must be representative; poor calibration data leads to poor bit assignments
- Policy overfitting: if training on single model size, may not transfer; train on diverse sizes
- Reward shaping critical: asymmetric penalties prevent policy from overshooting budget; tune carefully

## Reference

Paper: [arxiv.org/abs/2603.17891](https://arxiv.org/abs/2603.17891)
