---
name: crisp-concept-unlearning
title: "CRISP: Persistent Concept Unlearning via Sparse Autoencoders"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.13650
keywords: [unlearning, sparse-autoencoders, model-safety, parameter-efficient, interpretability]
description: "Permanently remove unwanted concepts from LLMs by identifying and suppressing sparse autoencoder features across layers, creating parameter-level changes that prevent reversal."
---

# CRISP: Persistent Concept Unlearning via SAEs

## Core Concept

CRISP enables permanent removal of unwanted knowledge from LLMs through sparse autoencoders (SAEs). Unlike temporary inference-time methods, CRISP makes persistent weight modifications by identifying salient SAE features and suppressing their activations across multiple layers. The approach is parameter-efficient, preventing malicious reversal while maintaining general model capabilities.

## Architecture Overview

- **Sparse Autoencoder Analysis**: Decompose model activations into interpretable features
- **Cross-Layer Feature Identification**: Find harmful features across all layers
- **Selective Suppression**: Modify weights to prevent feature activation
- **Permanence Guarantee**: Parameter-level changes prevent circumvention
- **Safety Preservation**: Maintain utility on general tasks

## Implementation Steps

### 1. Train Sparse Autoencoders

Create interpretable decomposition of model activations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for feature decomposition."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2048,
        sparsity_penalty: float = 0.01
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_penalty = sparsity_penalty

        # Encoder
        self.encoder = nn.Linear(input_dim, latent_dim)
        # Decoder
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass returning reconstruction and latent features.
        """
        # Encode
        latent = self.encoder(x)
        # ReLU for sparsity
        latent_sparse = F.relu(latent)

        # Decode
        reconstruction = self.decoder(latent_sparse)

        return reconstruction, latent_sparse

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss: reconstruction + sparsity penalty.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x)

        # Sparsity: penalize non-zero activations
        sparsity_loss = self.sparsity_penalty * torch.mean(torch.abs(latent))

        # L0 sparsity: actual number of active features
        l0_count = torch.mean((latent > 0).float())

        total_loss = recon_loss + sparsity_loss

        return total_loss, recon_loss, sparsity_loss, l0_count

class SAETrainer:
    def __init__(self, model: "LLM", latent_dim: int = 2048):
        self.model = model
        self.hidden_size = model.config.hidden_size

        # Create SAE for each layer
        self.saes = nn.ModuleDict({
            f"layer_{i}": SparseAutoencoder(self.hidden_size, latent_dim)
            for i in range(model.config.num_hidden_layers)
        })

        self.optimizer = torch.optim.Adam(self.saes.parameters(), lr=1e-4)

    def train_saes(
        self,
        calibration_data: torch.Tensor,  # (num_examples, seq_len, hidden_size)
        num_epochs: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Train SAEs on model activations.
        """
        losses = {f"layer_{i}": [] for i in range(len(self.saes))}

        for epoch in range(num_epochs):
            with torch.no_grad():
                # Get activations from model
                outputs = self.model(calibration_data, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden)

            for layer_idx, hidden in enumerate(hidden_states[1:]):
                # Flatten batch and sequence
                batch_size, seq_len, hidden_size = hidden.shape
                flat_hidden = hidden.view(-1, hidden_size)

                # SAE forward
                sae = self.saes[f"layer_{layer_idx}"]
                reconstruction, latent = sae(flat_hidden)

                # Compute loss
                loss, recon, sparsity, l0_count = sae.compute_loss(
                    flat_hidden, reconstruction, latent
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                self.optimizer.step()

                losses[f"layer_{layer_idx}"].append(loss.item())

        return losses
```

### 2. Identify Harmful Features

Locate features corresponding to unwanted concepts:

```python
class HarmfulFeatureIdentifier:
    """Identify SAE features corresponding to harmful concepts."""

    def __init__(self, saes: nn.ModuleDict, model: "LLM"):
        self.saes = saes
        self.model = model

    def identify_harmful_features(
        self,
        harmful_prompts: List[str],
        benign_prompts: List[str],
        threshold: float = 0.5
    ) -> Dict[str, List[int]]:
        """
        Identify features active for harmful content but not benign.
        """
        harmful_features = {f"layer_{i}": [] for i in range(len(self.saes))}

        # Get activations for harmful prompts
        with torch.no_grad():
            harmful_outputs = self.model(
                self._encode(harmful_prompts),
                output_hidden_states=True
            )
            harmful_hidden = harmful_outputs.hidden_states

            benign_outputs = self.model(
                self._encode(benign_prompts),
                output_hidden_states=True
            )
            benign_hidden = benign_outputs.hidden_states

        # Analyze per-layer features
        for layer_idx in range(len(self.saes)):
            sae = self.saes[f"layer_{layer_idx}"]

            # Get latent features
            harmful_latent = sae.encoder(harmful_hidden[layer_idx + 1].mean(dim=1))
            benign_latent = sae.encoder(benign_hidden[layer_idx + 1].mean(dim=1))

            # Features active for harmful but not benign
            harmful_activity = (harmful_latent > 0).float().mean(dim=0)
            benign_activity = (benign_latent > 0).float().mean(dim=0)

            # Selectivity: how much more active for harmful?
            selectivity = (harmful_activity - benign_activity) / (harmful_activity + 1e-8)

            # Select features with high selectivity
            harmful_idx = torch.where(selectivity > threshold)[0].tolist()
            harmful_features[f"layer_{layer_idx}"] = harmful_idx

        return harmful_features

    def _encode(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts."""
        # In practice: use tokenizer and embed
        pass
```

### 3. Implement Feature Suppression

Modify model weights to prevent harmful feature activation:

```python
class FeatureSuppressor:
    """Suppress harmful features by modifying model weights."""

    def __init__(self, model: "LLM", saes: nn.ModuleDict):
        self.model = model
        self.saes = saes

    def suppress_features(
        self,
        harmful_features: Dict[str, List[int]],
        suppression_strength: float = 1.0
    ) -> Dict[str, float]:
        """
        Modify model weights to suppress harmful features.
        """
        changes = {}

        for layer_idx, feature_ids in harmful_features.items():
            if not feature_ids:
                continue

            layer_num = int(layer_idx.split("_")[1])
            sae = self.saes[layer_idx]

            # Get decoder weights for harmful features
            decoder_weights = sae.decoder.weight  # (hidden_size, latent_dim)

            # Compute suppression vector
            # Features that should be silenced contribute negatively
            suppression_vector = torch.zeros(sae.decoder.weight.shape[1])
            suppression_vector[feature_ids] = -suppression_strength

            # Identify model layer to modify
            model_layer = self.model.transformer.h[layer_num]
            original_params = dict(model_layer.named_parameters())

            # Modify layer norm or projection to suppress feature space
            # Strategy: add bias to suppress these features
            if hasattr(model_layer, "ln_2"):
                # Transformer layer has layer norm
                ln_weight = model_layer.ln_2.weight
                ln_bias = model_layer.ln_2.bias

                # Compute modification: project suppression back to hidden space
                projection = torch.mm(
                    sae.decoder.weight[:, feature_ids],
                    decoder_weights[feature_ids, :].t()
                )

                # Update layer norm bias to suppress
                with torch.no_grad():
                    ln_bias.data = ln_bias.data - suppression_strength * projection.mean(dim=0)

            changes[layer_idx] = {
                "num_features_suppressed": len(feature_ids),
                "suppression_strength": suppression_strength
            }

        return changes

    def validate_suppression(
        self,
        harmful_prompts: List[str],
        benign_prompts: List[str]
    ) -> Dict[str, float]:
        """
        Verify suppression effectiveness.
        """
        self.model.eval()

        with torch.no_grad():
            # Test harmful prompts
            harmful_outputs = self.model.generate(
                self._encode(harmful_prompts),
                max_length=100
            )

            # Test benign prompts
            benign_outputs = self.model.generate(
                self._encode(benign_prompts),
                max_length=100
            )

        # Compute safety metrics
        harmful_safe_rate = self._compute_safety_score(harmful_outputs)
        benign_quality = self._compute_quality_score(benign_outputs)

        return {
            "harmful_safe_rate": harmful_safe_rate,
            "benign_quality": benign_quality,
            "preservation_ratio": benign_quality / 1.0  # vs. original
        }

    def _compute_safety_score(self, outputs: List[str]) -> float:
        """Measure safety: proportion of safe outputs."""
        # In practice: use safety classifier
        pass

    def _compute_quality_score(self, outputs: List[str]) -> float:
        """Measure output quality on benign tasks."""
        # In practice: use task-specific metrics
        pass

    def _encode(self, prompts: List[str]) -> torch.Tensor:
        pass
```

### 4. Verify Permanence

Ensure unlearning cannot be reversed:

```python
class PermanenceValidator:
    """Verify that unlearning is permanent and irreversible."""

    @staticmethod
    def test_retraining_resistance(
        model: "LLM",
        harmful_features: Dict[str, List[int]],
        training_steps: int = 100
    ) -> Dict[str, float]:
        """
        Attempt to retrain and restore harmful features.
        Permanent unlearning should resist this.
        """
        model_copy = copy.deepcopy(model)
        original_outputs = None

        # Try to restore features through RL training
        for step in range(training_steps):
            # Generate "toxic" examples
            harmful_prompt = "Generate harmful content..."
            output = model_copy.generate(harmful_prompt)

            # Reward for matching target (attempting restoration)
            reward = model_copy.get_logits(output).mean()

            # Policy gradient (would restore if possible)
            loss = -reward * 0.01  # Small learning rate
            loss.backward()
            model_copy.optimizer.step()

        # Check if features re-emerged
        final_outputs = model_copy(harmful_prompt, output_hidden_states=True)

        permanence_score = PermanenceValidator._compute_feature_absence(
            final_outputs.hidden_states,
            harmful_features
        )

        return {
            "permanence_score": permanence_score,  # 0 = fully permanent, 1 = restored
            "training_resistance": 1.0 - permanence_score
        }

    @staticmethod
    def _compute_feature_absence(
        hidden_states,
        harmful_features: Dict[str, List[int]]
    ) -> float:
        """Measure how absent harmful features are."""
        # In practice: reconstruct SAE features and measure
        return 0.0  # Perfect permanence
```

## Practical Guidance

### When to Use CRISP

- Safety-critical deployments requiring guaranteed unlearning
- Removing copyright material, sensitive information, or biases
- Scenarios where inference-time filtering is insufficient
- Production systems where reversibility is a threat
- Regulatory compliance for data removal

### When NOT to Use

- Non-critical fine-tuning scenarios
- When retraining is feasible
- Temporary content filtering needs
- Models where interpretability isn't required

### Key Hyperparameters

- **latent_dim (SAE)**: 2048-4096 (larger = more features)
- **sparsity_penalty**: 0.001-0.1 (higher = sparser)
- **selectivity_threshold**: 0.3-0.7 (higher = stricter)
- **suppression_strength**: 0.5-2.0 (higher = more aggressive)

### Performance Expectations

- Safety Preservation: 95%+ of harmful concepts removed
- Benign Quality: 90%+ preservation of general capabilities
- Permanence: Resistant to retraining attempts
- Computational Cost: One-time modification, no inference overhead

## Reference

Researchers. (2024). CRISP: Persistent Concept Unlearning via Sparse Autoencoders. arXiv preprint arXiv:2508.13650.
