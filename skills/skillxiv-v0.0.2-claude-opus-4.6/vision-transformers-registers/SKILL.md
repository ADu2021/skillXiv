---
name: vision-transformers-registers
title: "Vision Transformers Don't Need Trained Registers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.08010"
keywords: [vision-transformers, attention-mechanisms, register-neurons, outlier-tokens, inference-optimization]
description: "Apply test-time register token injection to pre-trained Vision Transformers without retraining, eliminating high-norm outlier artifacts and improving attention map quality."
---

# Vision Transformers Don't Need Trained Registers

## Core Concept

Vision Transformers suffer from high-norm outlier tokens that create noisy attention maps and degrade downstream tasks. Previous solutions required retraining models with dedicated register tokens. This paper identifies register neurons—sparse sets of neurons that generate these outliers—and shows that by shifting activations to untrained tokens at test time, pre-trained ViTs achieve comparable or better performance without any retraining. This approach is training-free, applicable to any pre-trained ViT, and provides a simple but elegant solution to a fundamental architectural problem.

## Architecture Overview

- **Register Neuron Detection**: Identifies MLP neurons with consistently high activations at outlier patch positions
- **Attention Sinks**: High-norm tokens that attract excessive softmax probability, creating artifacts
- **Test-Time Token Injection**: Appends untrained dummy tokens that capture outlier activations
- **Neuron Activation Shifting**: Copies maximum register neuron activation to test tokens, zeros elsewhere
- **Zero Retraining**: Fully compatible with existing pre-trained checkpoints and inference pipelines

## Implementation

### Step 1: Identify Register Neurons

Analyze ViT activations to find neurons responsible for outliers:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List

class RegisterNeuronFinder:
    """Find neurons that generate high-norm outlier tokens in ViTs"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.layer_info = {}

    def extract_mlp_activations(self, images_dataloader, target_layer_idx=6):
        """
        Collect MLP activations across images to identify outlier generators.

        Typically, outliers emerge after layer 6 in OpenCLIP ViTs.
        """
        all_activations = {}
        outlier_positions = []

        for batch_idx, images in enumerate(images_dataloader):
            images = images.to(self.device)

            # Forward pass with activation capture
            activations = self.capture_layer_activations(images, target_layer_idx)
            # Shape: (batch_size, num_patches, hidden_dim)

            # Identify outlier patches (high norm tokens)
            norms = torch.norm(activations, dim=-1)  # (batch_size, num_patches)
            threshold = norms.mean() + 3 * norms.std()

            # Find positions where norms exceed threshold
            batch_outliers = torch.where(norms > threshold)
            outlier_positions.extend(batch_outliers[1].cpu().tolist())  # patch indices

            all_activations[batch_idx] = activations.detach().cpu()

        return all_activations, outlier_positions

    def capture_layer_activations(self, images, layer_idx):
        """Capture intermediate activations from specific transformer layer"""
        activations = None

        def hook_fn(module, input, output):
            nonlocal activations
            activations = output[0]  # Transformer block output

        # Register hook
        target_layer = self.model.transformer.resblocks[layer_idx]
        hook = target_layer.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            _ = self.model(images)

        hook.remove()

        return activations

    def find_register_neurons(self, all_activations, outlier_positions,
                             threshold_percentile=95):
        """
        Identify MLP neurons with high activation at outlier positions.

        Register neurons consistently activate at outlier patch positions.
        """
        register_neuron_indices = []

        for layer_idx, activations in all_activations.items():
            # Get activations at outlier positions
            outlier_activations = activations[:, outlier_positions, :]
            # (batch, num_outliers, hidden_dim)

            # Compute per-neuron activation strength at outliers
            neuron_strength = outlier_activations.abs().mean(dim=(0, 1))
            # (hidden_dim,)

            # Find neurons with high outlier-specific activation
            threshold = np.percentile(
                neuron_strength.numpy(), threshold_percentile
            )
            strong_neurons = torch.where(neuron_strength > threshold)[0]

            register_neuron_indices.append(strong_neurons.tolist())

        return register_neuron_indices
```

### Step 2: Implement Test-Time Register Token Injection

Create the core algorithm that uses dummy tokens to capture outliers:

```python
class TestTimeRegisterInjection(nn.Module):
    """Apply register token injection at test time (no training required)"""

    def __init__(self, model, register_neurons_per_layer, num_registers=1):
        super().__init__()
        self.model = model
        self.register_neurons_per_layer = register_neurons_per_layer
        self.num_registers = num_registers

        # Create test-time register tokens (trainable but only used at test time)
        self.register_tokens = nn.Parameter(
            torch.randn(1, num_registers, model.config.hidden_size) * 0.02
        )

    def forward(self, images):
        """
        Forward pass with test-time register injection.

        At inference, we append dummy tokens to capture outlier activations
        instead of letting them pollute image patch embeddings.
        """
        # Standard forward pass through ViT
        with torch.no_grad():
            embeddings = self.model.embeddings(images)
            # (batch_size, 1 + num_patches, hidden_dim)

        # Inject register tokens
        batch_size = embeddings.shape[0]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        embeddings_with_registers = torch.cat(
            [embeddings, register_tokens], dim=1
        )

        # Forward through blocks with register neuron shifting
        x = embeddings_with_registers

        for layer_idx, block in enumerate(self.model.transformer.resblocks):
            x = block(x)

            # Apply register neuron shifting
            if layer_idx in self.register_neurons_per_layer:
                x = self.shift_register_neurons(
                    x, layer_idx, register_token_start_idx=1 + embeddings.shape[1]
                )

        return x[:, 0, :]  # Return CLS token for classification

    def shift_register_neurons(self, x, layer_idx, register_token_start_idx):
        """
        Shift register neuron activations to dummy register tokens.

        This prevents outliers from appearing in image patches while
        preserving the information in dedicated register tokens.

        Args:
            x: (batch_size, 1 + num_patches + num_registers, hidden_dim)
            layer_idx: which transformer layer
            register_token_start_idx: position where register tokens start
        """
        num_registers = x.shape[1] - register_token_start_idx

        # Extract register neuron indices for this layer
        register_neurons = torch.tensor(
            self.register_neurons_per_layer[layer_idx]
        ).to(x.device)

        if len(register_neurons) == 0:
            return x

        # For each register token, collect max activation from register neurons
        for reg_idx in range(num_registers):
            token_pos = register_token_start_idx + reg_idx

            # Get activation of register neurons in image patches
            image_activations = x[:, 1:register_token_start_idx,
                                 register_neurons]
            # (batch_size, num_patches, num_register_neurons)

            # Find maximum activation
            max_activation = image_activations.max(dim=-1)[0]  # (batch, patches)
            max_activation = max_activation.max(dim=-1)[0]  # (batch,)

            # Copy max activation to register token's corresponding neurons
            x[:, token_pos, register_neurons] = max_activation.unsqueeze(-1)

            # Zero out these neurons in image patches
            x[:, 1:register_token_start_idx, register_neurons] = 0

        return x
```

### Step 3: Analyze Attention Maps Before and After

Visualize the improvement from register token injection:

```python
class AttentionAnalyzer:
    """Analyze attention map quality improvements"""

    def __init__(self, model):
        self.model = model

    def extract_attention_maps(self, images, layer_idx=11, head_idx=0):
        """Extract attention maps from specific layer and head"""
        attention_maps = []

        def hook_fn(module, input, output):
            # output is (attention_output, attention_weights)
            attention_maps.append(output[1])

        # Register hook on attention layer
        target_layer = self.model.transformer.resblocks[layer_idx].attn
        hook = target_layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(images)

        hook.remove()

        return attention_maps[0]  # (batch, num_heads, seq_len, seq_len)

    def compute_attention_quality_metrics(self, attention_maps):
        """
        Compute metrics for attention map quality.

        Good attention maps have:
        - Dispersed patterns (not concentrated)
        - Clean spatial structure
        """
        batch_size, num_heads, seq_len, _ = attention_maps.shape

        metrics = {}

        # 1. Entropy (higher = more dispersed = better)
        entropy = -torch.sum(attention_maps * torch.log(attention_maps + 1e-10),
                            dim=-1).mean()
        metrics['entropy'] = entropy.item()

        # 2. Concentration (lower = better, less "attention sinks")
        max_attention = attention_maps.max(dim=-1)[0].mean()
        metrics['max_attention'] = max_attention.item()

        # 3. Spatial coherence (should attend to nearby patches)
        spatial_coherence = self.compute_spatial_coherence(attention_maps)
        metrics['spatial_coherence'] = spatial_coherence.item()

        return metrics

    def compute_spatial_coherence(self, attention_maps):
        """Measure how much attention respects spatial locality"""
        batch_size, num_heads, seq_len, _ = attention_maps.shape

        # Compute distance between patch positions
        distances = torch.zeros(seq_len, seq_len)
        grid_size = int(np.sqrt(seq_len - 1))  # -1 for CLS token

        for i in range(1, seq_len):
            for j in range(1, seq_len):
                i_pos = i - 1
                j_pos = j - 1

                i_x, i_y = i_pos // grid_size, i_pos % grid_size
                j_x, j_y = j_pos // grid_size, j_pos % grid_size

                dist = np.sqrt((i_x - j_x)**2 + (i_y - j_y)**2)
                distances[i, j] = dist

        # Spatial coherence: correlation between distance and attention
        coherence = torch.corrcoef(
            torch.stack([distances.flatten(), attention_maps.mean(dim=(0, 1)).flatten()])
        )[0, 1]

        return coherence
```

### Step 4: Evaluation on Downstream Tasks

Compare performance with and without register injection:

```python
def evaluate_with_registers(model_original, model_with_registers, eval_dataloader):
    """Compare ViT performance with and without test-time registers"""

    results = {'original': {}, 'with_registers': {}}

    # Evaluate original model
    print("Evaluating original model...")
    original_acc = evaluate_model(model_original, eval_dataloader)
    results['original']['accuracy'] = original_acc

    # Evaluate with registers
    print("Evaluating with test-time registers...")
    registers_acc = evaluate_model(model_with_registers, eval_dataloader)
    results['with_registers']['accuracy'] = registers_acc

    # Compute improvement
    improvement = (registers_acc - original_acc) / original_acc * 100

    print(f"\nResults:")
    print(f"Original: {original_acc:.4f}")
    print(f"With Registers: {registers_acc:.4f}")
    print(f"Improvement: {improvement:+.2f}%")

    return results

def evaluate_model(model, dataloader):
    """Standard accuracy evaluation"""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(model.device)
            labels = labels.to(model.device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total
```

## Practical Guidance

- **Training-Free**: No retraining required; works with any pre-trained ViT checkpoint
- **Register Count**: Typically 1-4 registers per image; 1 often sufficient
- **Layer Selection**: Register neurons typically emerge after layer 6 in vision models
- **Threshold Tuning**: 95th percentile works well; adjust based on model/task
- **Computation Cost**: Negligible; only adds a few hundred tokens to sequence
- **Performance Boost**: Typically 1-3% improvement on vision tasks
- **Attention Quality**: Significantly cleaner attention maps (higher entropy, lower concentration)
- **Generalization**: Works across different ViT architectures (ViT-B, ViT-L, OpenCLIP variants)

## Reference

- Register neurons are sparse sets of neurons with specific geometric structure
- Test-time injection exploits the observation that outliers are not task-critical information
- Unlike trained registers, untrained dummy tokens work just as well, suggesting outlier problem is structural
- Shifting activations preserves information while improving attention map interpretability
