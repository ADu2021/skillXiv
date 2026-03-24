---
name: less-is-more-recursive-reasoning-tiny-networks
title: "Less is More: Recursive Reasoning with Tiny Networks"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.04871"
keywords: [recursive reasoning, tiny models, parameter efficiency, puzzle solving, latent recursion]
description: "Achieve complex reasoning with minimal parameters using latent recursion in 2-layer networks. A 7M-parameter Tiny Recursive Model (TRM) solves Sudoku (87% accuracy), mazes (85%), and ARC-AGI with 0.01% the parameters of large LLMs via iterative latent refinement through 6+ recursive steps without fixed-point convergence requirements."
---

# Less is More: Recursive Reasoning with Tiny Networks

## Core Concept

Complex reasoning tasks do not require billion-parameter language models. A single 2-layer network with 7M parameters can outperform much larger models through latent recursion—iteratively refining intermediate representations over 6+ steps without explicit chain-of-thought text. The key insight is that deep recursive passes combined with one gradient-enabled step per iteration enable "effective depth per supervision" rivaling multi-step transformers.

## Architecture Overview

- **Single 2-Layer Network**: Minimal architecture replacing two 4-layer networks in prior work (Hierarchical Reasoning Model)
- **Latent Recursion**: n=6 iterations where hidden state z and answer y update via z = net(x,y,z); y = net(y,z)
- **Deep Supervision Loop**: T-1 gradient-free passes followed by one backprop-enabled pass, with early stopping via confidence threshold
- **Problem-Specific Optimization**: MLP for fixed-size problems (Sudoku, mazes); self-attention for variable grids (ARC-AGI)
- **Minimal Training**: ~1000 examples sufficient; EMA smoothing (0.999) and early stopping prevent overfitting

## Implementation Steps

### 1. Architecture Design

The network maintains three key components: input embedding, iteratively refined answer, and latent reasoning state.

```python
import torch
import torch.nn as nn

class TinyRecursiveModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=81, num_layers=2, use_attention=False):
        """
        Tiny Recursive Model (TRM): 2-layer network with latent recursion.

        Args:
            input_dim: Embedded question dimension
            hidden_dim: Latent state and hidden dimension
            output_dim: Answer dimension (e.g., 81 for 9x9 Sudoku)
            num_layers: Number of stacked layers (fixed at 2)
            use_attention: Use self-attention for variable-size grids (ARC-AGI)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention

        # Shared backbone: MLP or Transformer
        if use_attention:
            # For variable-size grids (ARC-AGI)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=256, batch_first=True
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            # For fixed-size problems (Sudoku, mazes)
            self.backbone = nn.Sequential(
                nn.Linear(input_dim + output_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )

        # Output heads
        self.latent_head = nn.Linear(hidden_dim, hidden_dim)    # z update
        self.answer_head = nn.Linear(hidden_dim + hidden_dim, output_dim)  # y update
        self.halt_head = nn.Linear(hidden_dim, 1)               # Confidence for early stopping

        self.ema_momentum = 0.999

    def forward(self, x, max_iterations=16, confidence_threshold=0.95):
        """
        Forward pass with iterative latent recursion.

        Args:
            x: Embedded input question
            max_iterations: Maximum recursion steps
            confidence_threshold: Early stopping when confidence exceeds threshold
        """
        batch_size = x.size(0)

        # Initialize: y=random, z=zero
        y = torch.randn(batch_size, self.output_dim, device=x.device)
        z = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        iteration_losses = []

        for iteration in range(max_iterations):
            # Deep recursion: T-1 gradient-free passes
            for _ in range(T - 1):
                z_new = self._update_latent(x, y, z)
                y_new = self._update_answer(y, z_new)
                z = z_new
                y = y_new

            # One gradient-enabled pass
            z = self._update_latent(x, y, z)
            y_updated = self._update_answer(y, z)

            # Confidence check
            confidence = torch.sigmoid(self.halt_head(z)).mean()

            if iteration > 0:  # Supervise with loss
                target_y = compute_ground_truth(x)  # Problem-specific
                loss = nn.functional.cross_entropy(y_updated, target_y)
                iteration_losses.append(loss)

                # Backprop only on this step
                loss.backward(retain_graph=(iteration < max_iterations - 1))

            # EMA update to stabilize
            z = self.ema_momentum * z + (1 - self.ema_momentum) * z_new

            # Early stopping
            if confidence > confidence_threshold:
                break

            y = y_updated

        return y, torch.tensor(iteration_losses)

    def _update_latent(self, x, y, z):
        """z = net(x, y, z)"""
        if self.use_attention:
            combined = torch.cat([x, y], dim=1)
            features = self.backbone(combined)
        else:
            combined = torch.cat([x, y, z], dim=1)
            features = self.backbone(combined)
        return self.latent_head(features)

    def _update_answer(self, y, z):
        """y = net(y, z)"""
        combined = torch.cat([y, z], dim=1)
        return self.answer_head(combined)
```

### 2. Training Procedure with Deep Supervision

Use deep supervision loop: T-1 gradient-free refinement steps per supervised step. EMA stabilization prevents gradient explosion.

```python
def train_tiny_recursive_model(model, train_loader, num_epochs=10, T=4):
    """
    T: number of inference steps per supervised backward pass
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    best_accuracy = 0
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (x_embedded, y_target) in enumerate(train_loader):
            # Deep supervision loop
            predictions, iteration_losses = model(x_embedded, max_iterations=16)

            # Supervision loss
            loss = nn.functional.cross_entropy(predictions, y_target)
            loss = loss + 0.1 * iteration_losses.mean()  # Regularize intermediate steps

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Accuracy tracking
            preds = predictions.argmax(dim=1)
            correct += (preds == y_target).sum().item()
            total += y_target.size(0)

        scheduler.step()
        epoch_accuracy = correct / total

        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Acc={epoch_accuracy:.2%}")

        # Early stopping
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_trm.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model
```

### 3. Benchmark Configuration

Domain-specific configurations for Sudoku, mazes, and ARC-AGI puzzles.

```python
# Sudoku-Extreme: 7M parameters
sudoku_config = {
    'input_dim': 100,      # Flattened 9x9 board encoded
    'hidden_dim': 128,
    'output_dim': 81,      # 9x9 grid
    'num_layers': 2,
    'use_attention': False,  # Fixed size
    'max_iterations': 16,
    'training_data_size': 1000,
    'batch_size': 64,
}

# Expected: 87.4% accuracy (vs HRM 55%)

# Maze-Hard: 7M parameters
maze_config = {
    'input_dim': 100,
    'hidden_dim': 128,
    'output_dim': 64,      # Flattened path output
    'num_layers': 2,
    'use_attention': False,
    'max_iterations': 20,
    'training_data_size': 1000,
    'batch_size': 64,
}

# Expected: 85.3% accuracy (vs HRM 74.5%)

# ARC-AGI: Variable-size grids
arc_config = {
    'input_dim': 200,      # Variable-length encoding
    'hidden_dim': 256,
    'output_dim': 256,     # Variable output
    'num_layers': 2,
    'use_attention': True,   # Attention for variable grids
    'max_iterations': 12,
    'training_data_size': 1000,
    'batch_size': 32,
}

# Expected: 44.6% (ARC-AGI-1), 7.8% (ARC-AGI-2)
# vs LLMs with billions of parameters
```

## Practical Guidance

**Network Size**: 7M parameters is the "sweet spot" for most puzzles. Smaller networks underfit; larger networks provide marginal gains with compute overhead.

**Recursion Depth**: 6-16 iterations sufficient for most tasks. More iterations yield diminishing returns; monitor confidence scores for early stopping threshold.

**Deep Supervision**: T=4 (3 gradient-free + 1 supervised pass) balances training efficiency with solution quality. Increasing T further requires more data.

**EMA Momentum**: 0.999 provides strong gradient smoothing; prevent explosion without oscillation. Clip gradients at norm 1.0 as safeguard.

**Data Efficiency**: ~1000 examples sufficient for training due to strong inductive bias from recursive structure. Avoid data augmentation; focus on diverse problem instances.

## When to Use / When NOT to Use

**Use When**:
- Solving discrete puzzles with well-defined correct answers (Sudoku, mazes, logic grids, code synthesis)
- Parameter efficiency is critical (edge devices, embedded systems, low-latency inference)
- Problems have inherent recursive structure (iterative refinement applies)
- Training data is limited (1000 examples available)

**NOT For**:
- Open-ended generation tasks (creative writing, dialogue)
- Continuous output spaces or regression problems
- Tasks requiring broad world knowledge (not captured by puzzle structure)
- Scenarios where interpretability of chain-of-thought is essential

## Reference

This skill synthesizes techniques from "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871). The approach demonstrates that parameter count is not destiny for reasoning—recursive latent refinement enables competitive performance with 0.01% of large model parameters.
