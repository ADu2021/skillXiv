---
name: policy-discriminative-reward-models
title: "Pre-Trained Policy Discriminators are General Reward Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.05197"
keywords: [Reward Models, Policy Learning, RLHF, Bradley-Terry Loss, Self-Supervised Pretraining]
description: "Learn generalizable reward models via unsupervised policy discrimination: pretraining models to distinguish between different policies enables efficient adaptation to human preferences and strong RLHF performance."
---

# POLAR: Policy Discriminators as Universal Reward Models

Reward models are critical for aligning language models with human values, but training them from scratch requires expensive human preference annotations. POLAR (POLicy DiscriminAtive LeaRning) proposes an alternative: pretraining reward models as policy discriminators that learn to quantify differences between policies without labeled preference data. This unsupervised pretraining creates criterion-independent reward representations that can efficiently adapt to any evaluation standard, enabling 10x smaller models to outperform much larger preference-based reward models.

The key insight is that distinguishing between different policies requires learning rich representations of behavior quality. Rather than waiting for human labels to define "good," the model learns what makes policies different. This learned representation generalizes well to downstream preference judgments because policy differences capture fundamental behavioral variation.

## Core Concept

POLAR operates in two stages. First, unsupervised pretraining learns to recognize trajectories from identical policies while discriminating between different ones using only diverse policy samples (no human labels). This creates representations sensitive to behavioral differences. Second, supervised fine-tuning adapts the pretrained model to human preference rankings via ranking tasks on trajectories, requiring far fewer examples than training from scratch.

The approach reframes reward modeling as a self-supervised representation learning problem. Policy trajectories themselves contain rich information about what distinguishes good from bad behavior—the model just needs to learn to extract it before human preferences are introduced.

## Architecture Overview

The system comprises:

- **Trajectory Encoder**: Encodes trajectory sequences into fixed-dimensional representations capturing behavioral quality
- **Policy Discriminator Head**: Learns to distinguish between policy identities during pretraining via Bradley-Terry loss
- **Preference Ranker Head**: Fine-tunes pretrained representations to rank trajectories by human preference during RLHF
- **Unsupervised Pretraining Data**: Diverse policies from multiple sources (supervised baselines, different random seeds, different model sizes) enabling policy discrimination

## Implementation

Start with the trajectory encoder:

```python
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List, Dict, Tuple

class TrajectoryEncoder(nn.Module):
    """
    Encode trajectory sequences (state-action pairs) into representations.

    Processes trajectory history and learns behavioral patterns without
    assuming anything about what makes trajectories good or bad.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512,
                 num_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Embed states and actions separately
        self.state_embedding = nn.Linear(state_dim, hidden_dim // 2)
        self.action_embedding = nn.Linear(action_dim, hidden_dim // 2)

        # Process sequence with transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to representation space
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode trajectory into representation.

        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)

        Returns:
            representation: (batch, hidden_dim)
        """
        # Embed states and actions
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)

        # Concatenate at each timestep
        trajectory = torch.cat([state_emb, action_emb], dim=-1)

        # Process with transformer
        encoded = self.transformer(trajectory)

        # Global average pooling
        representation = encoded.mean(dim=1)
        return self.output_proj(representation)
```

Implement the policy discriminator for unsupervised pretraining:

```python
import torch.nn.functional as F
from torch.optim import Adam

class PolicyDiscriminator(nn.Module):
    """
    Learn to distinguish between different policies via Bradley-Terry loss.

    Unsupervised pretraining that creates policy-aware representations
    without requiring human preference labels.
    """

    def __init__(self, encoder: TrajectoryEncoder, hidden_dim: int = 512):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim

        # Policy classification head: predict which policy generated trajectory
        self.discriminator_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Scalar discrimination score
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute policy discrimination score for trajectory.

        Returns scalar score that should be high for policy A and low for policy B.
        """
        representation = self.encoder(states, actions)
        score = self.discriminator_head(representation)
        return score

    def bradley_terry_loss(self, trajectories_a: Tuple,
                          trajectories_b: Tuple) -> torch.Tensor:
        """
        Compute Bradley-Terry loss for policy discrimination.

        Treats policy discrimination as pairwise comparison: model learns
        to assign higher scores to trajectories from policy A than B.

        Args:
            trajectories_a: (states_a, actions_a) from policy A
            trajectories_b: (states_b, actions_b) from policy B

        Returns:
            loss: Bradley-Terry loss encouraging proper ordering
        """
        states_a, actions_a = trajectories_a
        states_b, actions_b = trajectories_b

        # Compute scores
        score_a = self.forward(states_a, actions_a)
        score_b = self.forward(states_b, actions_b)

        # Bradley-Terry: log-odds that A > B should be high
        # Loss = -log(sigmoid(score_a - score_b))
        log_odds = score_a - score_b
        loss = F.softplus(-log_odds).mean()

        return loss
```

Implement the preference ranker for RLHF fine-tuning:

```python
class PreferenceRanker(nn.Module):
    """
    Fine-tune pretrained representations to human preference rankings.

    Uses learned trajectory representations and trains ranking head
    to predict human preference without retraining the encoder.
    """

    def __init__(self, encoder: TrajectoryEncoder, hidden_dim: int = 512):
        super().__init__()
        self.encoder = encoder

        # Ranking head: predict preference score
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict human preference score for trajectory."""
        representation = self.encoder(states, actions)
        return self.ranking_head(representation)

    def ranking_loss(self, preferred: Tuple, dispreferred: Tuple,
                     margin: float = 0.5) -> torch.Tensor:
        """
        Compute ranking loss for preference learning.

        Encourages higher scores for human-preferred trajectories.

        Args:
            preferred: (states_pref, actions_pref) with human preference
            dispreferred: (states_dispref, actions_dispref) without preference
            margin: margin to enforce between scores

        Returns:
            loss: ranking loss with margin
        """
        states_pref, actions_pref = preferred
        states_dispref, actions_dispref = dispreferred

        score_pref = self.forward(states_pref, actions_pref)
        score_dispref = self.forward(states_dispref, actions_dispref)

        # Margin ranking loss: score_pref should be >= score_dispref + margin
        loss = F.relu(margin - (score_pref - score_dispref)).mean()

        return loss
```

Implement the full training pipeline:

```python
class POLARTrainer:
    """
    Two-stage training: unsupervised pretraining then supervised fine-tuning.

    Stage 1: Learn to discriminate between diverse policies (no labels needed)
    Stage 2: Fine-tune to human preferences (standard RLHF)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 512, learning_rate: float = 1e-4):
        self.encoder = TrajectoryEncoder(state_dim, action_dim, hidden_dim)
        self.discriminator = PolicyDiscriminator(self.encoder, hidden_dim)
        self.ranker = PreferenceRanker(self.encoder, hidden_dim)

        self.optimizer_pretrain = Adam(self.discriminator.parameters(), lr=learning_rate)
        self.optimizer_finetune = Adam(self.ranker.parameters(), lr=learning_rate)

    def pretrain_on_policies(self, policy_trajectories: Dict[str, List[Tuple]],
                            num_epochs: int = 100) -> Dict:
        """
        Unsupervised pretraining: learn to discriminate between policies.

        Policy trajectories is dict mapping policy names to lists of (states, actions).
        """
        policy_names = list(policy_trajectories.keys())
        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Sample pairs of policies
            for i in range(len(policy_names)):
                policy_a = policy_names[i]
                policy_b = policy_names[(i + 1) % len(policy_names)]

                # Sample trajectories from each policy
                traj_a = policy_trajectories[policy_a][epoch % len(policy_trajectories[policy_a])]
                traj_b = policy_trajectories[policy_b][epoch % len(policy_trajectories[policy_b])]

                # Compute Bradley-Terry loss
                loss = self.discriminator.bradley_terry_loss(traj_a, traj_b)

                # Backward pass
                self.optimizer_pretrain.zero_grad()
                loss.backward()
                self.optimizer_pretrain.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(policy_names))
            if (epoch + 1) % 10 == 0:
                print(f"Pretraining epoch {epoch + 1}, loss: {epoch_loss / len(policy_names):.4f}")

        return {'pretraining_losses': losses}

    def finetune_on_preferences(self, preference_pairs: List[Tuple[Tuple, Tuple]],
                               num_epochs: int = 50,
                               freeze_encoder: bool = True) -> Dict:
        """
        Fine-tune to human preferences using pretrained encoder.

        preference_pairs: list of (preferred_trajectory, dispreferred_trajectory)
        """
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for preferred, dispreferred in preference_pairs:
                loss = self.ranker.ranking_loss(preferred, dispreferred)

                self.optimizer_finetune.zero_grad()
                loss.backward()
                self.optimizer_finetune.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(preference_pairs))
            if (epoch + 1) % 10 == 0:
                print(f"Fine-tuning epoch {epoch + 1}, loss: {epoch_loss / len(preference_pairs):.4f}")

        return {'finetuning_losses': losses}

    def compute_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute reward score for trajectory using fine-tuned ranker."""
        return self.ranker(states, actions)
```

## Practical Guidance

**Hyperparameter Table:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Hidden dimension | 512 | 256-1024 | Larger = more capacity; 512 sufficient for most tasks |
| Transformer layers | 2 | 1-4 | More layers improve representation but slow training |
| Margin (ranking) | 0.5 | 0.1-2.0 | Larger margin enforces stricter separation |
| Pretraining epochs | 100 | 50-500 | More epochs better but diminishing returns |
| Fine-tuning epochs | 50 | 10-100 | Usually quick convergence to preferences |
| Learning rate | 1e-4 | 1e-5 to 1e-3 | Conservative; use warmup |
| Freeze encoder | True | - | Prevents overfitting to limited preference data |

**When to Use:**
- You're building reward models for RLHF and have limited human preference annotations
- You want to leverage diverse policy trajectories without human labels
- You need a compact reward model (7B outperforms 72B baselines)
- You have access to multiple policy trajectories (supervised models, random seeds, etc.)
- You plan to deploy the same reward model across different downstream tasks

**When NOT to Use:**
- You have abundant high-quality human preference annotations
- You don't have access to diverse policy trajectories for pretraining
- Your downstream task is very niche and pretraining doesn't transfer
- You need reward estimates that exactly match human judgments (preference ranking is approximation)
- You're in a safety-critical domain requiring explicit alignment (prefer explicit human feedback)

**Common Pitfalls:**
- **Weak pretraining data**: If policy trajectories are too similar, discrimination doesn't work. Use diverse policy sources (different models, random seeds).
- **Encoder overfitting**: Without freezing the encoder during fine-tuning, it specializes to limited preference data. Always freeze for fine-tuning unless you have >10K preference pairs.
- **Bradley-Terry assumptions**: Loss assumes transitive preferences (if A > B and B > C, then A > C). Real human preferences violate this; use conservative margins.
- **Generalization failure**: Pretraining on one task set may not transfer to different tasks. Validate on held-out task distributions.
- **Limited preference data**: Fine-tuning requires at least 100-500 preference pairs for good performance. Don't try with <50 examples.
- **Margin mismatch**: Too small margin makes ranking noise insensitive; too large prevents convergence. Start at 0.5 and tune based on downstream performance.

## Reference

Authors (2025). Pre-Trained Policy Discriminators are General Reward Models. arXiv preprint arXiv:2507.05197. https://arxiv.org/abs/2507.05197
