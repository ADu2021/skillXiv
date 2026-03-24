---
name: flexible-data-mixture-of-experts
title: "FlexOlmo: Open Language Models for Flexible Data Use"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07024"
keywords: [Mixture of Experts, Data Privacy, Model Composition, Domain-specific Experts, Opt-out Learning]
description: "Train language models where each expert learns independently on closed datasets, enabling flexible inference with selective data inclusion or exclusion. 41% performance improvement while allowing users to opt out of specific data sources without retraining."
---

# Flexible Data Mixture of Experts: Composable Models with Per-Dataset Training and Inference Selection

Traditional language models pool all training data during training, making it impossible to exclude data sources later or share models with restricted data licensing—a model trained on copyrighted material cannot be shared with users who don't have rights to that content. FlexOlmo solves this through decoupled mixture-of-experts training: each expert trains independently on a single dataset (or dataset collection), enabled by a frozen public model as an anchor point that all experts learn relative to. At inference, users select which experts (data sources) to include, and the router combines only those selected experts without retraining.

When building collaborative models across organizations with different data licensing, when building systems respecting user privacy choices, or when creating composable models that can be updated incrementally, flexible mixture-of-experts enables a new paradigm: models that are modular, respectful of data constraints, and auditable about which data influences which outputs.

## Core Concept

FlexOlmo separates training and routing through three innovations. First, coordinated training: each expert trains independently on its dataset using a frozen public model as an anchor (providing consistent baseline performance), allowing experts to learn without pooling or seeing other datasets. Second, domain-informed routing: rather than joint router training (which requires mixed data), the router learns expert-specific embeddings initialized from domain representations, then finetuned independently on each dataset. These per-expert embeddings concatenate at inference. Third, inference-time flexibility: selecting which experts to include requires no retraining—the router naturally handles missing experts through concatenated embeddings. Optional lightweight tuning on small public proxy datasets can refine the router without accessing closed data.

## Architecture Overview

- **Frozen Public Model**: Baseline LLM shared across all expert training as an anchor point
- **Domain-Specific Expert Modules**: Per-dataset transformer modules learning independently
- **Expert Embeddings (Router)**: Per-expert learned representations derived from dataset characteristics
- **Router Assembly**: Concatenates selected expert embeddings at inference time
- **Gating Network**: Learns to weight and combine expert outputs conditioned on input
- **Optional Proxy Tuning**: Lightweight router refinement using public data resembling closed datasets

## Implementation

This example demonstrates the coordinated training approach where experts train independently on separate datasets using a frozen public model as anchor.

```python
# Coordinated independent expert training with frozen public anchor
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertModule(nn.Module):
    def __init__(self, hidden_dim=2048, num_layers=12, num_heads=32):
        """Expert transformer module for single dataset."""
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Expert-specific layers (independent learning)
        self.expert_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=8192,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """Process through expert layers."""
        for layer in self.expert_layers:
            x = layer(x)
        x = self.output_norm(x)
        return x


class CoordinatedMixtureOfExperts:
    def __init__(self, public_model, num_experts: int, hidden_dim=2048):
        """Initialize mixture with frozen public model and independent experts."""

        self.public_model = public_model  # Frozen anchor
        self.public_model.eval()  # Never train public model

        # Initialize experts
        self.experts = nn.ModuleList([
            ExpertModule(hidden_dim=hidden_dim)
            for _ in range(num_experts)
        ])

        self.hidden_dim = hidden_dim

    def train_expert(self, expert_id: int, dataset, optimizer, num_epochs: int):
        """Train single expert on its dataset using frozen public model as anchor."""

        expert = self.experts[expert_id]
        expert.train()

        for epoch in range(num_epochs):
            total_loss = 0.0

            for batch in dataset:
                # Get embeddings from frozen public model (anchor point)
                with torch.no_grad():
                    input_ids = batch['input_ids']
                    outputs = self.public_model(input_ids, output_hidden_states=True)
                    anchor_embeddings = outputs.hidden_states[-1]  # [batch, seq, dim]

                # Expert learns to predict next tokens better than public model
                expert_output = expert(anchor_embeddings)

                # Compute target (ground truth tokens)
                target_output = self.public_model(
                    batch['input_ids'], output_hidden_states=True
                ).last_hidden_state

                # Loss: how much better can expert do than public model?
                loss = F.mse_loss(expert_output, target_output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Expert {expert_id}, Epoch {epoch}: Loss {total_loss:.4f}")

    def forward_with_selected_experts(self, input_ids: torch.Tensor, selected_experts: list):
        """Inference with selected experts only (no retraining needed)."""

        # Get public model embeddings (anchor)
        with torch.no_grad():
            outputs = self.public_model(input_ids, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]

        # Pass through selected experts
        expert_outputs = []
        for expert_id in selected_experts:
            expert_output = self.experts[expert_id](embeddings)
            expert_outputs.append(expert_output)

        # Average selected expert outputs
        combined = torch.mean(torch.stack(expert_outputs), dim=0)

        return combined
```

This example shows the domain-informed router design where each expert gets an independent embedding learned from its dataset.

```python
class DomainInformedRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        """Router with per-expert embeddings learned independently."""

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Per-expert embeddings (learned independently, no cross-dataset mixing)
        self.expert_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim))
            for _ in range(num_experts)
        ])

        # Gating: learns to weight experts
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def initialize_embeddings_from_domain(self, domain_representations: list):
        """Initialize expert embeddings from domain characteristics.
        domain_representations: list of [hidden_dim] tensors per domain."""

        for i, domain_repr in enumerate(domain_representations):
            self.expert_embeddings[i].data = domain_repr.clone()

    def compute_expert_weights(self, input_embedding: torch.Tensor) -> torch.Tensor:
        """Compute weights for combining expert outputs.
        Only selected experts' embeddings are used."""

        # Gate takes input and produces expert weights
        weights = self.gate(input_embedding)
        return weights

    def forward_with_selected_experts(self, expert_outputs: list, input_embedding: torch.Tensor):
        """Combine outputs from selected experts.
        expert_outputs: list of [batch, seq, hidden] from selected experts only."""

        # Compute weights
        all_weights = self.compute_expert_weights(input_embedding)

        # Only use weights for selected experts
        num_selected = len(expert_outputs)

        # Renormalize weights for selected subset
        selected_weights = all_weights[:, :num_selected] / (all_weights[:, :num_selected].sum(dim=-1, keepdim=True) + 1e-10)

        # Weighted combination
        combined = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            combined += expert_output * selected_weights[:, i:i+1]

        return combined
```

This example demonstrates inference-time flexibility: dynamically selecting which experts to include without retraining.

```python
class FlexOlmoModel(nn.Module):
    def __init__(self, public_model, num_experts: int, hidden_dim=2048):
        super().__init__()

        self.moe = CoordinatedMixtureOfExperts(public_model, num_experts, hidden_dim)
        self.router = DomainInformedRouter(hidden_dim, num_experts)

    def inference_with_data_selection(self, input_ids: torch.Tensor, expert_selection: dict):
        """Perform inference with user-selected data sources.
        expert_selection: {'licenses_to_include': ['MIT', 'Apache2'], 'opt_outs': ['proprietary']}"""

        # Determine which experts to use based on selection
        selected_expert_ids = self._compute_selected_experts(expert_selection)

        # Get public model embeddings
        with torch.no_grad():
            outputs = self.public_model(input_ids, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]

        # Forward through selected experts
        selected_outputs = []
        for expert_id in selected_expert_ids:
            output = self.moe.experts[expert_id](embeddings)
            selected_outputs.append(output)

        # Combine with router (no retraining!)
        combined = self.router.forward_with_selected_experts(
            selected_outputs,
            embeddings[:, 0]  # Use [CLS] token for routing
        )

        return combined

    def _compute_selected_experts(self, selection_criteria: dict) -> list:
        """Map data selection criteria to expert indices."""
        # This would be populated with metadata about which expert trained on which data
        selected = []

        for expert_id, metadata in enumerate(self.expert_metadata):
            # Check if expert matches selection criteria
            if self._matches_criteria(metadata, selection_criteria):
                selected.append(expert_id)

        return selected if selected else list(range(len(self.moe.experts)))

    def optional_proxy_tuning(self, public_proxy_dataset, learning_rate=1e-5, num_epochs=1):
        """Lightweight router tuning using public data without accessing closed datasets."""

        optimizer = torch.optim.AdamW(self.router.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for batch in public_proxy_dataset:
                # Forward with all experts
                selected_outputs = [
                    self.moe.experts[i](batch['embeddings'])
                    for i in range(len(self.moe.experts))
                ]

                # Router learns better weighting
                combined = self.router.forward_with_selected_experts(
                    selected_outputs, batch['embeddings'][:, 0]
                )

                # Loss: matching behavior of public model
                target = self.public_model(batch['input_ids']).logits
                loss = F.mse_loss(combined, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Proxy tuning epoch {epoch}: Loss {loss:.4f}")
```

## Practical Guidance

| Hyperparameter | Recommended Value | Purpose |
|---|---|---|
| Number of experts | 4-8 | Balance diversity vs. memory |
| Expert hidden dimension | 2048 | Standard transformer scale |
| Expert depth | 12 layers | Compute-balanced with base model |
| Learning rate (expert training) | 1e-4 | Conservative independent learning |
| Learning rate (router tuning) | 1e-5 | Lightweight proxy tuning |
| Gating bottleneck dimension | hidden_dim | Avoid information loss |
| Softmax temperature (gating) | 1.0 | Standard attention temperature |
| Proxy dataset size | 10% of expert dataset | Sufficient for router refinement |

**When to use:** Apply FlexOlmo when building collaborative models across institutions with different data licensing, when offering users privacy-respecting opt-outs, or when building models that must be auditable about data influence. Use when you need to gradually add new data sources without retraining shared infrastructure.

**When NOT to use:** Skip if all data sources can be pooled freely and licensing is not a constraint. Avoid if experts must be tightly coupled (reasoning requiring cross-dataset context). Don't use if you need tight expert coordination—FlexOlmo assumes experts are somewhat independent. Skip for small-scale problems where routing overhead isn't justified.

**Common pitfalls:** Training experts jointly defeats the purpose—keep data sources separate. Using a public model that's too weak creates poor anchor; use a well-trained baseline. Not preserving domain characteristics in expert embeddings wastes the domain-informed router. Over-aggressive proxy tuning causes the router to overfit the public distribution. Forgetting that selection at inference requires metadata mapping (you must track which expert trained on which data). Not testing that expert removal actually works as expected.

## Reference

FlexOlmo Team. (2025). FlexOlmo: Open Language Models for Flexible Data Use. arXiv preprint arXiv:2507.07024. https://arxiv.org/abs/2507.07024
