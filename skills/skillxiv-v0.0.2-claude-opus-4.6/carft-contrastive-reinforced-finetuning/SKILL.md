---
name: carft-contrastive-reinforced-finetuning
title: "CARFT: Contrastive CoT Reinforced Fine-Tuning for LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.15868
keywords: [contrastive-learning, chain-of-thought, reinforcement-learning, representation-learning, fine-tuning]
description: "Enhance LLM reasoning by combining contrastive learning on reasoning representations with reinforced fine-tuning, leveraging both annotated chains and unsupervised signals."
---

# CARFT: Contrastive CoT Reinforced Fine-Tuning

## Core Concept

CARFT addresses limitations in both vanilla RL (ignoring annotated reasoning) and SFT (over-relying on limited examples) by combining contrastive representation learning with reinforced fine-tuning. The approach learns discriminative representations for each chain-of-thought while designing contrastive signals to guide optimization. This dual approach stabilizes training, prevents model degradation, and achieves 10.15% performance gains with up to 30.62% efficiency improvements.

## Architecture Overview

- **Chain-of-Thought Representation Learning**: Embedding space for reasoning paths
- **Contrastive Signal Design**: Positive/negative pairs for discriminative learning
- **Reinforced Fine-Tuning**: RL objectives with contrastive regularization
- **Training Stability**: Prevents both over-fitting and catastrophic forgetting
- **Efficiency Gains**: Reduced computational cost while improving performance

## Implementation Steps

### 1. Implement CoT Representation Learner

Learn embeddings for reasoning chains:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoTRepresentationLearner(nn.Module):
    """Learn representations of chain-of-thought sequences."""

    def __init__(
        self,
        hidden_size: int = 768,
        output_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()

        # Encoder for CoT sequences
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Projection heads for different loss objectives
        self.contrastive_proj = nn.Linear(output_dim, output_dim)
        self.quality_proj = nn.Linear(output_dim, 64)

    def encode_cot(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Encode chain-of-thought from hidden states.

        hidden_states: (seq_len, hidden_size) or (batch, seq_len, hidden_size)
        """
        # Use last hidden state or mean pooling
        if hidden_states.dim() == 3:
            # Batch of sequences: use mean pooling
            pooled = torch.mean(hidden_states, dim=1)
        else:
            pooled = hidden_states[-1]  # Last token

        # Encode to representation
        representation = self.encoder(pooled)
        return representation

    def compute_contrastive_projections(
        self,
        representations: torch.Tensor  # (batch, output_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute projections for contrastive learning.
        """
        contrastive = self.contrastive_proj(representations)
        quality = self.quality_proj(representations)
        return contrastive, quality
```

### 2. Design Contrastive Signals

Create positive/negative pairs for discriminative learning:

```python
class ContrastiveSignalDesigner:
    """Design contrastive objectives for CoT learning."""

    def __init__(self, margin: float = 0.5):
        self.margin = margin

    def create_contrastive_pairs(
        self,
        correct_cots: List[torch.Tensor],
        incorrect_cots: List[torch.Tensor],
        task_diversity: Optional[List[str]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create positive and negative pairs for contrastive learning.

        Strategies:
        - Correct vs incorrect (basic)
        - Similar reasoning, different answers (harder negatives)
        - Same task vs different task (task-specific)
        """
        pairs = []

        # Basic: correct vs incorrect
        for correct in correct_cots:
            for incorrect in incorrect_cots:
                pairs.append({
                    "anchor": correct,
                    "positive": correct,  # Same correct CoT
                    "negative": incorrect,
                    "pair_type": "basic"
                })

        # Hard negatives: correct answer, wrong reasoning path
        # (In practice: identify these from training data)

        return pairs

    def compute_triplet_loss(
        self,
        anchor_repr: torch.Tensor,
        positive_repr: torch.Tensor,
        negative_repr: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Triplet loss: anchor should be closer to positive than negative.
        """
        # Normalize representations
        anchor_norm = F.normalize(anchor_repr, p=2, dim=-1)
        positive_norm = F.normalize(positive_repr, p=2, dim=-1)
        negative_norm = F.normalize(negative_repr, p=2, dim=-1)

        # Compute similarities
        pos_sim = torch.mm(anchor_norm, positive_norm.t()) / temperature
        neg_sim = torch.mm(anchor_norm, negative_norm.t()) / temperature

        # Triplet objective
        loss = torch.nn.functional.softplus(neg_sim - pos_sim + self.margin).mean()
        return loss

    def compute_in_batch_negatives_loss(
        self,
        representations: torch.Tensor,  # (batch, dim)
        labels: torch.Tensor,  # (batch,) class labels
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute contrastive loss using in-batch negatives.
        """
        # Normalize
        representations = F.normalize(representations, p=2, dim=-1)

        # Compute similarity matrix
        logits = torch.mm(representations, representations.t()) / temperature

        # Create mask for positives (same label)
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        mask.fill_diagonal_(False)  # Exclude self

        # Compute loss
        pos_logits = logits[mask].view(logits.size(0), -1)
        neg_logits = logits[~mask].view(logits.size(0), -1)

        # LogSumExp trick for numerical stability
        loss = -torch.log(
            torch.exp(pos_logits).sum(dim=-1) /
            (torch.exp(pos_logits).sum(dim=-1) + torch.exp(neg_logits).sum(dim=-1) + 1e-8)
        ).mean()

        return loss
```

### 3. Implement CARFT Objective

Combine contrastive and RL losses:

```python
class CARFTLoss(nn.Module):
    """CARFT combined contrastive and RL loss."""

    def __init__(
        self,
        cot_learner: CoTRepresentationLearner,
        contrastive_weight: float = 0.5,
        rl_weight: float = 0.5,
        temperature: float = 0.07
    ):
        super().__init__()
        self.cot_learner = cot_learner
        self.contrastive_weight = contrastive_weight
        self.rl_weight = rl_weight
        self.temperature = temperature
        self.contrastive_designer = ContrastiveSignalDesigner()

    def forward(
        self,
        model_logits: torch.Tensor,  # (batch, seq_len, vocab_size)
        hidden_states: torch.Tensor,  # (batch, seq_len, hidden_size)
        labels: torch.Tensor,  # (batch, seq_len)
        rewards: torch.Tensor,  # (batch,) or (batch, seq_len)
        annotated_cots: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CARFT loss combining contrastive and RL objectives.
        """

        batch_size, seq_len, vocab_size = model_logits.shape

        # 1. RL Loss (standard policy gradient with reward)
        log_probs = F.log_softmax(model_logits, dim=-1)
        selected_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        if rewards.dim() == 1:
            # Broadcast scalar reward to sequence
            rewards_expanded = rewards.unsqueeze(-1).expand_as(selected_log_probs)
        else:
            rewards_expanded = rewards

        rl_loss = -(selected_log_probs * rewards_expanded).mean()

        # 2. Contrastive Loss (representation learning)
        representations = self.cot_learner.encode_cot(hidden_states)
        contrastive_projs, _ = self.cot_learner.compute_contrastive_projections(representations)

        # Use rewards as proxy for correctness labels
        correctness_labels = (rewards > 0.5).long()
        contrastive_loss = self.contrastive_designer.compute_in_batch_negatives_loss(
            contrastive_projs,
            correctness_labels,
            temperature=self.temperature
        )

        # 3. Annotated CoT regularization (if available)
        regularization_loss = 0.0
        if annotated_cots is not None:
            # Encourage representations to match annotated CoTs
            annotated_reprs = self.cot_learner.encode_cot(annotated_cots)
            annotated_projs, _ = self.cot_learner.compute_contrastive_projections(annotated_reprs)

            # MSE between predicted and annotated representations
            regularization_loss = F.mse_loss(contrastive_projs, annotated_projs)

        # Combine losses
        total_loss = (
            self.rl_weight * rl_loss +
            self.contrastive_weight * contrastive_loss +
            0.1 * regularization_loss
        )

        metrics = {
            "rl_loss": rl_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "regularization_loss": regularization_loss.item() if isinstance(regularization_loss, torch.Tensor) else regularization_loss,
            "total_loss": total_loss.item()
        }

        return total_loss, metrics
```

### 4. Implement CARFT Training Loop

Integrate components into training procedure:

```python
class CARFTTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cot_learner: CoTRepresentationLearner,
        carft_loss: CARFTLoss,
        optimizer: torch.optim.Optimizer,
        learning_rate: float = 1e-5
    ):
        self.model = model
        self.cot_learner = cot_learner
        self.carft_loss = carft_loss
        self.optimizer = optimizer

    def train_step(
        self,
        batch_inputs: torch.Tensor,
        batch_labels: torch.Tensor,
        batch_rewards: torch.Tensor,
        annotated_cots: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Execute single CARFT training step."""

        # Forward pass
        outputs = self.model(
            batch_inputs,
            output_hidden_states=True
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer

        # Compute CARFT loss
        loss, metrics = self.carft_loss(
            logits,
            hidden_states,
            batch_labels,
            batch_rewards,
            annotated_cots
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.cot_learner.parameters(), 1.0)
        self.optimizer.step()

        return metrics

    def train_epoch(
        self,
        train_dataloader,
        num_epochs: int = 1
    ) -> List[Dict[str, float]]:
        """Train for multiple epochs."""

        all_metrics = []

        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch_inputs = batch["input_ids"].to(self.device)
                batch_labels = batch["labels"].to(self.device)
                batch_rewards = batch["rewards"].to(self.device)
                annotated_cots = batch.get("annotated_cots")
                if annotated_cots is not None:
                    annotated_cots = annotated_cots.to(self.device)

                metrics = self.train_step(
                    batch_inputs,
                    batch_labels,
                    batch_rewards,
                    annotated_cots
                )
                all_metrics.append(metrics)

        return all_metrics
```

### 5. Validate Training with Metrics

Monitor improvement from CARFT:

```python
def evaluate_carft(
    model: torch.nn.Module,
    cot_learner: CoTRepresentationLearner,
    test_examples: List[Dict],
    baseline_model: torch.nn.Module
) -> Dict[str, float]:
    """
    Compare CARFT-trained model against baseline.
    """
    model.eval()
    baseline_model.eval()

    carft_correct = 0
    baseline_correct = 0
    carft_steps = 0
    baseline_steps = 0

    with torch.no_grad():
        for example in test_examples:
            prompt = example["prompt"]
            expected = example["expected"]

            # CARFT model
            carft_output = model.generate(prompt, max_length=500)
            carft_answer = extract_answer(carft_output)
            carft_correct += (carft_answer == expected)
            carft_steps += len(carft_output.split())

            # Baseline
            baseline_output = baseline_model.generate(prompt, max_length=500)
            baseline_answer = extract_answer(baseline_output)
            baseline_correct += (baseline_answer == expected)
            baseline_steps += len(baseline_output.split())

    carft_accuracy = carft_correct / len(test_examples)
    baseline_accuracy = baseline_correct / len(test_examples)

    return {
        "carft_accuracy": carft_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "accuracy_improvement": carft_accuracy - baseline_accuracy,
        "efficiency_gain": 1.0 - (carft_steps / baseline_steps),
        "carft_avg_steps": carft_steps / len(test_examples),
        "baseline_avg_steps": baseline_steps / len(test_examples)
    }
```

## Practical Guidance

### When to Use CARFT

- Training with both annotated reasoning chains and reward signals
- Tasks where training stability is critical
- Scenarios with limited annotation but good reward signals
- Mathematical or logical reasoning tasks
- Multi-step problem solving

### When NOT to Use

- Pure supervised learning without rewards
- Minimal annotated data available
- Tasks without clear reasoning chains
- Real-time training scenarios

### Key Hyperparameters

- **contrastive_weight**: 0.3-0.7 (balance with RL)
- **rl_weight**: 0.3-0.7
- **temperature**: 0.05-0.1 (lower = sharper contrasts)
- **margin**: 0.3-0.7 (triplet loss margin)
- **learning_rate**: 1e-5 to 1e-4

### Performance Expectations

- Accuracy Improvement: +10.15%
- Efficiency Gains: Up to 30.62% cost reduction
- Training Stability: Reduced variance vs. vanilla RL
- Convergence: Typically 1-3 epochs

## Reference

Researchers. (2024). CARFT: Boosting LLM Reasoning via Contrastive CoT Reinforced Fine-Tuning. arXiv preprint arXiv:2508.15868.
