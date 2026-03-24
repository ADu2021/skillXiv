---
name: amft-meta-learning-alignment
title: "AMFT: Aligning LLM Reasoners by Meta-Learning Imitation-Exploration"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.06944
keywords: [meta-learning, sft-rl-balance, alignment, imitation-learning, reinforcement-learning]
description: "Use meta-learning to automatically balance Supervised Fine-Tuning and Reinforcement Learning signals, treating SFT and RL as complementary rewards in a unified single-stage training framework."
---

# AMFT: Aligning LLM Reasoners by Meta-Learning Imitation-Exploration

## Core Concept

Traditional LLM alignment pipelines follow two distinct stages: Supervised Fine-Tuning (SFT) for imitation, then Reinforcement Learning (RL) for exploration. This separation creates problems—catastrophic forgetting during RL and suboptimal trade-offs between following human demonstrations and discovering novel solutions.

AMFT (Adaptive Meta Fine-Tuning) reframes SFT and RL as complementary reward signals within a single training framework. Using meta-learning, the system automatically discovers the optimal balance between imitation and exploration without manual tuning of mixture weights.

## Architecture Overview

- **Implicit Rewards Framework**: Treats SFT loss and RL loss as two reward signals to be optimally balanced
- **Meta-Gradient Controller**: Uses meta-learning to compute adaptive weights that balance imitation vs. exploration
- **Single-Stage Training**: Merges SFT and RL into one unified objective with learnable interpolation
- **Entropy Regularization**: Stabilizes training by constraining policy entropy to prevent divergence
- **Dynamic Weighting**: Adapts the imitation-exploration balance based on training dynamics

## Implementation Steps

### 1. Define SFT and RL Objectives

Set up both objectives as separate loss components that will be weighted and combined.

```python
import torch
import torch.nn.functional as F
from torch.optim import Adam

def compute_sft_loss(model, batch):
    """
    Supervised Fine-Tuning loss: standard next-token prediction
    Encourages the model to match human demonstrations
    """
    input_ids = batch['input_ids']
    labels = batch['labels']

    logits = model(input_ids).logits
    sft_loss = F.cross_entropy(
        logits.view(-1, model.config.vocab_size),
        labels.view(-1),
        reduction='mean'
    )
    return sft_loss

def compute_rl_loss(model, batch, reward_model):
    """
    Reinforcement Learning loss: policy gradient with reward signal
    Encourages the model to maximize external reward (e.g., correctness)
    """
    prompts = batch['prompts']
    responses = model.generate(prompts, max_length=256, num_return_sequences=4)

    # Compute rewards for generated responses
    rewards = reward_model.score(responses)

    # Policy gradient: negative log-likelihood weighted by rewards
    log_probs = model.compute_log_probs(responses)
    rl_loss = -(log_probs * rewards).mean()

    return rl_loss
```

### 2. Compute Entropy Regularization

Add entropy regularization to prevent the policy from collapsing to a deterministic policy during RL.

```python
def compute_entropy_regularization(model, prompts, target_entropy=0.5):
    """
    Entropy regularization: encourages diverse generations
    Prevents policy collapse and stabilizes training
    """
    logits = model(prompts).logits
    probs = F.softmax(logits, dim=-1)

    # Shannon entropy
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    # Regularization term: penalize if entropy drops below target
    entropy_loss = (target_entropy - entropy).clamp(min=0)
    return entropy_loss, entropy
```

### 3. Initialize Meta-Learning Controller

Create a learnable meta-parameter that controls the balance between SFT and RL losses.

```python
class AdaptiveWeightController(torch.nn.Module):
    """
    Meta-learner that computes adaptive weights for SFT vs RL balance
    """
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size

        # Meta-network: maps training statistics to weights
        self.meta_net = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2),  # Output 2 weights: w_sft, w_rl
            torch.nn.Softmax(dim=-1)  # Normalize to sum to 1
        )

        self.optimizer = Adam(self.meta_net.parameters(), lr=1e-4)

    def forward(self, training_stats):
        """
        Compute adaptive weights based on training statistics
        training_stats: [sft_loss, rl_loss, entropy, gradient_norm]
        """
        # Normalize statistics to prevent scale issues
        normalized_stats = torch.log(training_stats + 1e-8)
        weights = self.meta_net(normalized_stats)
        return weights

    def update(self, sft_loss, rl_loss, entropy):
        """
        Meta-gradient step: optimize weights to balance objectives
        """
        # Meta-objective: minimize weighted sum, but adapt weights based on loss ratio
        loss_ratio = rl_loss / (sft_loss + 1e-8)
        meta_loss = loss_ratio.detach()  # Stop gradient for meta-optimization

        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
```

### 4. Unified Training Objective

Combine SFT and RL losses using the adaptive weights, with entropy regularization.

```python
def compute_amft_loss(model, batch, reward_model, weight_controller,
                     entropy_weight=0.1, entropy_target=0.5):
    """
    AMFT combined loss: adaptive weighted sum of SFT and RL
    """
    # Compute individual loss components
    sft_loss = compute_sft_loss(model, batch)
    rl_loss = compute_rl_loss(model, batch, reward_model)
    entropy_reg, entropy = compute_entropy_regularization(
        model, batch['prompts'], target_entropy=entropy_target
    )

    # Compute gradient norms for meta-learning
    sft_grads = torch.autograd.grad(sft_loss, model.parameters(),
                                    retain_graph=True, create_graph=False)
    sft_grad_norm = sum((g ** 2).sum() for g in sft_grads) ** 0.5

    rl_grads = torch.autograd.grad(rl_loss, model.parameters(),
                                   retain_graph=True, create_graph=False)
    rl_grad_norm = sum((g ** 2).sum() for g in rl_grads) ** 0.5

    # Get adaptive weights from meta-controller
    training_stats = torch.tensor(
        [sft_loss.item(), rl_loss.item(), entropy.item(), sft_grad_norm.item()],
        device=model.device
    )
    w_sft, w_rl = weight_controller(training_stats)

    # Combined loss
    amft_loss = w_sft * sft_loss + w_rl * rl_loss + entropy_weight * entropy_reg

    return amft_loss, {
        'sft_loss': sft_loss,
        'rl_loss': rl_loss,
        'entropy': entropy,
        'w_sft': w_sft,
        'w_rl': w_rl
    }
```

### 5. Main Training Loop

Implement the unified training with both model updates and meta-controller updates.

```python
def train_amft(model, reward_model, train_loader, num_epochs=10):
    """
    Main AMFT training loop: single-stage imitation + exploration
    """
    model_optimizer = Adam(model.parameters(), lr=5e-5)
    weight_controller = AdaptiveWeightController()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            # Forward pass: compute combined loss
            amft_loss, stats = compute_amft_loss(
                model, batch, reward_model, weight_controller,
                entropy_weight=0.1, entropy_target=0.6
            )

            # Model update
            model_optimizer.zero_grad()
            amft_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()

            # Meta-update: adjust weights based on loss dynamics
            weight_controller.update(
                sft_loss=stats['sft_loss'],
                rl_loss=stats['rl_loss'],
                entropy=stats['entropy']
            )

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}")
                print(f"  AMFT Loss: {amft_loss:.4f}")
                print(f"  SFT Loss: {stats['sft_loss']:.4f}, "
                      f"RL Loss: {stats['rl_loss']:.4f}")
                print(f"  Weights: w_sft={stats['w_sft']:.3f}, "
                      f"w_rl={stats['w_rl']:.3f}")
                print(f"  Entropy: {stats['entropy']:.4f}")

    return model, weight_controller
```

### 6. Inference and Evaluation

Use the trained model for generation and evaluate on both imitation and exploration metrics.

```python
def evaluate_amft_model(model, test_prompts, reward_model, reference_answers):
    """
    Evaluate AMFT model on both imitation and exploration quality
    """
    model.eval()

    imitation_score = 0  # How well it matches SFT training
    exploration_score = 0  # How well it maximizes reward

    with torch.no_grad():
        for prompt, reference in zip(test_prompts, reference_answers):
            response = model.generate(prompt, max_length=256)

            # Imitation quality: BLEU/exact match with reference
            imitation = compute_similarity(response, reference)

            # Exploration quality: external reward
            exploration = reward_model.score(response)

            imitation_score += imitation
            exploration_score += exploration

    avg_imitation = imitation_score / len(test_prompts)
    avg_exploration = exploration_score / len(test_prompts)

    print(f"Imitation Score: {avg_imitation:.4f}")
    print(f"Exploration Score: {avg_exploration:.4f}")
    print(f"Combined: {0.5 * avg_imitation + 0.5 * avg_exploration:.4f}")

    return avg_imitation, avg_exploration
```

## Practical Guidance

### Hyperparameters & Configuration

- **Entropy Target**: 0.5-0.8 (depends on vocabulary size; higher for larger vocabularies)
- **Entropy Weight**: 0.05-0.2 (balance between diversity and accuracy)
- **Meta Learning Rate**: 1e-4 to 1e-3 (much slower than main model learning rate)
- **Model Learning Rate**: 5e-5 to 1e-4 (typical LLM fine-tuning rates)
- **Gradient Clipping**: max_norm=1.0 recommended for stability
- **Weight Update Frequency**: Every 10-50 model steps

### When to Use AMFT

- You want unified imitation + exploration training (avoid two-stage pipeline)
- Catastrophic forgetting is an issue when transitioning from SFT to RL
- You need automatic balancing without manual mixture weight tuning
- You have strong reward signals that complement human demonstrations
- You want dynamic adaptation as training progresses

### When NOT to Use AMFT

- Your task only needs imitation (SFT is sufficient)
- You lack a good reward model (RL component would be noisy)
- You need strict control over imitation vs. exploration balance
- Training budget is very limited (meta-learning adds overhead)
- Your base model is already well-aligned (fine-tuning not needed)

### Common Pitfalls

1. **Unstable Meta-Learning**: If weight controller learns too quickly, training becomes unstable. Use conservative meta learning rates (1e-4).
2. **Insufficient Entropy Regularization**: Without entropy constraints, policy collapses to deterministic outputs. Always include regularization term.
3. **Poor Reward Signal**: If reward model is noisy or misaligned, RL component becomes harmful. Validate reward model quality first.
4. **Imbalanced Data**: If SFT batch and RL batch have different distributions, weighting becomes harder. Use balanced sampling.
5. **No Baseline Model**: Compare against pure SFT to ensure AMFT actually helps; sometimes simpler approaches suffice.

## Reference

AMFT (2508.06944): https://arxiv.org/abs/2508.06944

**Note**: This paper has been withdrawn from arXiv due to potential academic misconduct concerns. The skill documents the core technical approach but use with awareness of the paper's disputed status.

Single-stage unified training of imitation and exploration using meta-learned weight balancing achieves better performance than traditional two-stage SFT+RL pipelines.
