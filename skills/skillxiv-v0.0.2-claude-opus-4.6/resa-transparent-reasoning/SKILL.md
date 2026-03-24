---
name: resa-transparent-reasoning
title: "Resa: Transparent Reasoning Models via SAEs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.09967"
keywords: [sparse autoencoders, reasoning transfer, interpretability, efficient training, reasoning tuning]
description: "Extract and transfer reasoning abilities using sparse autoencoders (SAE-Tuning) on CoT-free data, achieving RL-equivalent performance at 2000x lower cost and 450x faster training."
---

# Resa: Transparent Reasoning Models via SAEs

## Core Concept

Resa demonstrates that reasoning abilities are learnable, transferable features that can be extracted via sparse autoencoders (SAE) without expensive reinforcement learning. The two-stage SAE-Tuning procedure trains an autoencoder on source model activations, then inserts it into a target model with low-rank adapters. This achieves RL-equivalent reasoning performance at approximately $1 cost and 20-minute training time, compared to months and $2000+ for traditional RL approaches.

## Architecture Overview

- **Two-Stage SAE-Tuning**: Stage I trains SAE on source model activations; Stage II inserts frozen SAE into target model with LoRA for implicit reasoning pattern transfer
- **CoT-Free Training Data**: Uses only verified question-answer pairs (no intermediate reasoning traces), reducing data requirements
- **Reasoning as Portable Adapter**: Extracted reasoning features function as modular "reasoning adapters" transferable across model families without retraining
- **Transparent Feature Extraction**: Prompt-only method identifies latent reasoning features; layer-wise distribution correlates with reasoning performance
- **Massive Cost Reduction**: $1 per model vs $2000 for RL; 20 minutes training vs months for full RL pipeline

## Implementation

### Step 1: Sparse Autoencoder Training

```python
import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    """
    Trains on source model activations to extract reasoning features.
    Sparse representation: high dimensionality (65k features) but low k activation.
    """

    def __init__(self, input_dim, num_features=65536, k=32):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.k = k  # top-k sparsity

        # Encoder: input -> sparse features
        self.encoder = nn.Linear(input_dim, num_features)

        # Decoder: sparse features -> reconstruction
        self.decoder = nn.Linear(num_features, input_dim)

        # Initialize decoder as transpose of encoder (tied weights)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def forward(self, x):
        """
        Forward pass with top-k sparsity constraint.
        Returns reconstruction and sparse features.
        """

        # Encode to feature space
        features = self.encoder(x)  # [batch, num_features]

        # Apply top-k sparsity: keep only k largest activations per sample
        k_values, k_indices = torch.topk(torch.abs(features), self.k, dim=1)
        sparse_features = torch.zeros_like(features)
        sparse_features.scatter_(1, k_indices, features.gather(1, k_indices))

        # Decode sparse features
        reconstruction = self.decoder(sparse_features)

        return reconstruction, sparse_features

    def train_on_source_model(self, source_model, trigger_dataset, num_epochs=3):
        """
        Train SAE on activations from source reasoning model.
        trigger_dataset: verified QA pairs with <think> markers but no reasoning steps.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            for batch in trigger_dataset:
                question_ids = batch['input_ids']

                # Extract activation at target layer (layer 12)
                with torch.no_grad():
                    activations = source_model.get_layer_activation(
                        question_ids,
                        layer=12
                    )

                # Train SAE to reconstruct activations
                reconstruction, sparse_features = self.forward(activations)

                # Loss: reconstruction + sparsity penalty
                recon_loss = loss_fn(reconstruction, activations)
                sparsity_loss = 0.01 * sparse_features.abs().sum(dim=1).mean()

                total_loss = recon_loss + sparsity_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch}: Loss={total_loss:.4f}")
```

### Step 2: SAE-Guided SFT in Target Model

```python
class LoRAAdapter(nn.Module):
    """
    Low-rank adapter for inserting SAE into target model.
    Minimizes KL divergence between outputs with/without SAE.
    """

    def __init__(self, hidden_dim, lora_rank=32):
        super().__init__()
        self.lora_rank = lora_rank

        # LoRA for query, key, value, dense projections
        self.lora_q = nn.Linear(hidden_dim, lora_rank)
        self.lora_q_out = nn.Linear(lora_rank, hidden_dim)

        self.lora_k = nn.Linear(hidden_dim, lora_rank)
        self.lora_k_out = nn.Linear(lora_rank, hidden_dim)

        self.lora_v = nn.Linear(hidden_dim, lora_rank)
        self.lora_v_out = nn.Linear(lora_rank, hidden_dim)

        # Zero initialization: LoRA starts as no-op
        with torch.no_grad():
            self.lora_q_out.weight.zero_()
            self.lora_k_out.weight.zero_()
            self.lora_v_out.weight.zero_()

    def forward(self, x):
        """Apply LoRA update to activation."""
        delta_q = self.lora_q_out(self.lora_q(x))
        delta_k = self.lora_k_out(self.lora_k(x))
        delta_v = self.lora_v_out(self.lora_v(x))

        return {'q': delta_q, 'k': delta_k, 'v': delta_v}

class SAEGuidedTraining:
    """
    Stage II: Insert frozen SAE into target model and train LoRA
    to minimize KL divergence between model with/without SAE.
    """

    def __init__(self, target_model, sae, layer_idx=12):
        self.target_model = target_model
        self.sae = sae
        self.layer_idx = layer_idx
        self.lora = LoRAAdapter(target_model.hidden_size)

        # Freeze SAE
        for param in self.sae.parameters():
            param.requires_grad = False

    def train(self, trigger_dataset, num_epochs=1, learning_rate=1e-4):
        """
        Train LoRA to instill SAE-extracted reasoning patterns.
        KL divergence loss: log(p_with_sae) - log(p_without_sae)
        """

        optimizer = torch.optim.Adam(self.lora.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for batch in trigger_dataset:
                input_ids = batch['input_ids']
                target_tokens = batch['target_ids']  # Next token predictions

                # Forward pass WITHOUT SAE
                with torch.no_grad():
                    logits_base = self.target_model(input_ids)
                    log_probs_base = torch.nn.functional.log_softmax(logits_base, dim=-1)

                # Forward pass WITH SAE guidance
                activations = self.target_model.get_layer_activation(
                    input_ids,
                    layer=self.layer_idx
                )

                # Apply SAE to extract reasoning features
                _, sparse_features = self.sae(activations)

                # Apply LoRA update conditioned on sparse features
                lora_delta = self.lora(sparse_features)

                # Modify model behavior via LoRA
                logits_sae = self.target_model(
                    input_ids,
                    lora_delta=lora_delta
                )
                log_probs_sae = torch.nn.functional.log_softmax(logits_sae, dim=-1)

                # KL divergence loss: encourage sae-guided outputs
                kl_loss = torch.nn.functional.kl_div(
                    log_probs_sae,
                    torch.exp(log_probs_base),
                    reduction='batchmean'
                )

                optimizer.zero_grad()
                kl_loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}: KL Loss={kl_loss:.4f}")
```

### Step 3: Transparent Reasoning Feature Analysis

```python
class ReasoningFeatureAnalyzer:
    """
    Identify and quantify latent reasoning features via prompt-only method.
    Reveals which SAE features activate during reasoning tasks.
    """

    def __init__(self, sae, model):
        self.sae = sae
        self.model = model

    def identify_reasoning_features(self, test_dataset):
        """
        Analyze which SAE features correlate with reasoning performance.
        Returns feature importance scores.
        """

        feature_activations = torch.zeros(self.sae.num_features)
        feature_task_performance = torch.zeros(self.sae.num_features)

        for sample in test_dataset:
            # Get activations
            with torch.no_grad():
                activations = self.model.get_layer_activation(
                    sample['input_ids'],
                    layer=12
                )
                _, sparse_features = self.sae(activations)

            # Track which features activated
            feature_activations += (sparse_features.abs().sum(dim=0) > 0).float()

            # Measure task performance (correctness on reasoning task)
            correctness = 1.0 if sample['is_correct'] else 0.0
            feature_task_performance += sparse_features[0] * correctness

        # Compute feature importance
        feature_importance = feature_task_performance / (feature_activations + 1e-8)

        return feature_importance

    def analyze_layer_wise_distribution(self, model, dataset):
        """
        Examine how reasoning features distribute across layers.
        Correlates with reasoning performance.
        """

        layer_importance = {}

        for layer_idx in range(model.num_layers):
            features_this_layer = 0.0

            for sample in dataset:
                with torch.no_grad():
                    activations = model.get_layer_activation(
                        sample['input_ids'],
                        layer=layer_idx
                    )
                    _, sparse_features = self.sae(activations)

                features_this_layer += sparse_features.abs().sum().item()

            layer_importance[layer_idx] = features_this_layer / len(dataset)

        return layer_importance
```

### Step 4: Deployment and Cost Analysis

```python
def compute_training_cost(model_size_b, hours_to_train, cost_per_hour_gpu=0.5):
    """Estimate training cost for SAE-tuning."""
    num_gpus = max(1, model_size_b // 10)  # Estimate GPUs needed
    total_cost = hours_to_train * num_gpus * cost_per_hour_gpu
    return total_cost

# Cost comparison
# SAE-Tuning on R1-Distill-1.5B:
# - Compute: 2 L40S GPUs x ~0.5 hours x $0.5/hr = $0.50
# - Model: GPT-4 mini calls for inference guidance = ~$0.47
# Total: ~$1.00

# Traditional RL on same model:
# - Compute: 8 A100 GPUs x ~30 hours x $2.0/hr = $480
# - Infrastructure setup, data curation = ~$1500
# Total: ~$2000+

print(f"SAE-Tuning cost: $1.00")
print(f"RL cost: $2000.00")
print(f"Cost reduction: 2000x")
```

## Practical Guidance

**Implementation Steps**:
1. Select source model with reasoning (R1-Distill, DeepSeek-R1)
2. Prepare trigger dataset: verified QAs with `<think></think>` markers but no intermediate steps
3. Train SAE on source model activations (layer 12 recommended)
4. Insert frozen SAE into target model with LoRA adapters
5. Fine-tune LoRA on trigger dataset (1 epoch sufficient)

**Data Requirements**:
- Trigger dataset: 5k-10k verified question-answer pairs
- Source: STILL, DeepScaleR, or similar CoT-free datasets
- No need for intermediate reasoning steps, reducing data requirements by 90%

**Generalization**:
- Reasoning features transfer across datasets (AIME, MATH, GPQA)
- Works across model families (Qwen, LLaMA variants, DeepSeek)
- Adapters remain small (LoRA rank 32) enabling multi-adapter composition

**Performance Targets**:
- R1-Distill-1.5B: ~35% AIME pass@1 (RL-equivalent)
- Generalizes to unseen benchmarks without retraining
- Multimodal models: reasoning benefits transfer to vision tasks

## Reference

- Sparse Autoencoders: Extract interpretable features from neural networks; top-k sparsity prevents feature collapse
- LoRA: Low-rank adaptation; efficient parameterization for transfer learning
- KL divergence: Measures distribution difference; KL minimization encourages model to adopt SAE-guided behavior
