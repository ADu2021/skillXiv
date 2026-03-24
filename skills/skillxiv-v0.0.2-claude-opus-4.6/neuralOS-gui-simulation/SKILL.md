---
name: neuralOS-gui-simulation
title: "NeuralOS: Towards Simulating Operating Systems via Neural Generative Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.08800"
keywords: [GUI Simulation, Diffusion Models, RNN State Tracking, Interactive Systems]
description: "Simulate GUI behavior by predicting screen frames in response to user inputs. NeuralOS combines hierarchical RNNs for state tracking with diffusion-based rendering, capturing mouse interactions and application state transitions. Trains on synthetic demonstrations plus random exploration; achieves 50-61% human indistinguishability on basic operations while maintaining 18 fps inference on single H100."
---

# NeuralOS: Simulate Operating System Interactions with Neural Models

Operating system GUIs involve complex state tracking and visual rendering—applications open, windows resize, text appears. Traditional automation approaches use rigid rules; NeuralOS learns to predict GUI evolution from user inputs (mouse position, clicks, keyboard) using a recurrent neural network for state tracking and diffusion models for frame generation. The system captures state transitions implicitly, enabling realistic simulation of multi-step interactions without explicit state machines.

The key insight is that GUI behavior is spatiotemporally continuous: mouse movements lead to smooth cursor motion, clicks trigger state changes with visual consequences, keyboard input fills text fields. By combining RNNs (efficient state tracking without quadratic complexity) with diffusion rendering (high-fidelity frame generation), you can simulate OS interactions at near-realistic quality.

## Core Concept

NeuralOS operates as a three-component pipeline:

1. **RNN State Tracker**: Maintains hidden system state (what windows are open, text field contents) based on accumulated user inputs, avoiding transformer quadratic complexity during inference
2. **Latent Frame Encoder**: Compresses video frames into latent codes to reduce diffusion model compute
3. **Diffusion Renderer**: Generates new frames conditioned on RNN state, user input, and spatial cursor encoding

The system learns from two data sources: agent-generated demonstrations (synthetic interactions) and random exploration (avoiding spurious agent patterns).

## Architecture Overview

- **RNN State Module**: 2-level hierarchical LSTM tracking long-term state and short-term dynamics (avoids transformer quadratic cost)
- **Latent Encoder-Decoder**: Converts 1280×720 RGB frames to 160×90 latent codes for efficient diffusion
- **Diffusion UNet Renderer**: Denoising network conditioned on RNN state via cross-attention
- **Spatial Cursor Encoding**: Gaussian spatial embeddings precisely localizing cursor (hundreds-of-pixels error without it)
- **Scheduled Sampling Trainer**: Gradually shifts from training-data frames to model-generated frames during training
- **Context Manager**: Extends from 32 to 64 frame context for capturing longer dependencies

## Implementation

The following demonstrates the hierarchical RNN state tracker and diffusion renderer:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HierarchicalRNNStateTracker(nn.Module):
    """Two-level RNN for GUI state tracking without quadratic complexity."""
    def __init__(self, input_dim: int, hidden_dim: int = 1024, context_len: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_len = context_len

        # Level 1: Per-frame dynamics (fast state changes)
        self.frame_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )

        # Level 2: Long-term context (slow state changes)
        self.context_lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Input projection (mouse position, click, keyboard events)
        self.input_proj = nn.Linear(input_dim, input_dim)

    def forward(self, input_sequence: torch.Tensor,
                frame_hidden: Optional[Tuple] = None,
                context_hidden: Optional[Tuple] = None):
        """
        Track GUI state across multiple frames.

        Args:
            input_sequence: (batch, seq_len, input_dim) mouse/keyboard inputs
            frame_hidden: LSTM hidden state for frame-level dynamics
            context_hidden: LSTM hidden state for long-term context

        Returns:
            state_embeddings: (batch, seq_len, hidden_dim)
            frame_hidden: Updated frame-level hidden state
            context_hidden: Updated context-level hidden state
        """
        # Project inputs
        projected = self.input_proj(input_sequence)

        # Frame-level processing (captures immediate response to inputs)
        frame_outputs, frame_hidden = self.frame_lstm(projected, frame_hidden)

        # Context-level processing (integrates frame outputs over time)
        context_outputs, context_hidden = self.context_lstm(frame_outputs, context_hidden)

        return context_outputs, frame_hidden, context_hidden

class SpatialCursorEncoding(nn.Module):
    """Gaussian spatial encoding for precise cursor localization."""
    def __init__(self, height: int = 90, width: int = 160, embedding_dim: int = 256):
        super().__init__()
        self.height = height
        self.width = width
        self.embedding_dim = embedding_dim

        # Learnable Gaussian parameters
        self.sigma = nn.Parameter(torch.ones(1) * 0.1)
        self.learnable_embedding = nn.Embedding(2, embedding_dim)  # 2D position

    def forward(self, cursor_x: torch.Tensor, cursor_y: torch.Tensor):
        """
        Create spatial embeddings centered at cursor position.

        Args:
            cursor_x: (batch,) normalized x coordinate [0, 1]
            cursor_y: (batch,) normalized y coordinate [0, 1]

        Returns:
            spatial_embedding: (batch, height, width, embedding_dim)
        """
        batch_size = cursor_x.shape[0]

        # Create 2D coordinate grids
        y_coords = torch.linspace(0, 1, self.height, device=cursor_x.device)
        x_coords = torch.linspace(0, 1, self.width, device=cursor_x.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Expand for batch dimension
        yy = yy.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, height, width)
        xx = xx.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute Euclidean distance to cursor per pixel
        dist = torch.sqrt(
            (xx - cursor_x.view(batch_size, 1, 1)) ** 2 +
            (yy - cursor_y.view(batch_size, 1, 1)) ** 2
        )

        # Gaussian kernel (nearby pixels get high values)
        gaussian = torch.exp(-dist ** 2 / (2 * self.sigma ** 2))  # (batch, height, width)

        # Expand to embedding dimension
        spatial_emb = gaussian.unsqueeze(-1).expand(-1, -1, -1, self.embedding_dim)
        return spatial_emb

class DiffusionFrameRenderer(nn.Module):
    """Diffusion-based UNet for rendering GUI frames conditioned on RNN state."""
    def __init__(self, latent_dim: int = 4, state_dim: int = 1024,
                 hidden_channels: int = 128, num_diffusion_steps: int = 50):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Cross-attention layer: condition on RNN state
        self.state_projection = nn.Linear(state_dim, hidden_channels)
        self.cross_attention = nn.MultiheadAttention(
            hidden_channels, num_heads=8, batch_first=True
        )

        # UNet blocks (simplified)
        self.down_blocks = nn.ModuleList([
            nn.Conv2d(latent_dim + hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1, stride=2),
        ])

        self.up_blocks = nn.ModuleList([
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=3, padding=1),
        ])

        self.time_embedding = nn.Embedding(num_diffusion_steps, hidden_channels)

    def forward(self, x_t: torch.Tensor, timestep: int, state: torch.Tensor,
                cursor_encoding: torch.Tensor):
        """
        Denoise latent frames conditioned on RNN state and cursor.

        Args:
            x_t: (batch, latent_dim, H, W) noisy latent frame
            timestep: Current diffusion step [0, num_steps]
            state: (batch, state_dim) RNN state
            cursor_encoding: (batch, H, W, embedding_dim) spatial cursor embedding

        Returns:
            denoised: (batch, latent_dim, H, W)
        """
        batch_size, _, h, w = x_t.shape

        # Project state for cross-attention
        state_proj = self.state_projection(state)  # (batch, hidden_channels)
        state_proj = state_proj.unsqueeze(1)  # (batch, 1, hidden_channels)

        # Time embedding
        time_emb = self.time_embedding(torch.tensor(timestep, device=x_t.device))
        time_emb = time_emb.unsqueeze(0).expand(batch_size, -1)

        # Reshape cursor encoding to match spatial dimensions
        cursor_flat = cursor_encoding.view(batch_size, h, w, -1).permute(0, 3, 1, 2)

        # Concatenate input with cursor encoding
        x_with_cursor = torch.cat([x_t, cursor_flat[:, :x_t.shape[1]]], dim=1)

        # Down-sampling with state conditioning
        x = x_with_cursor
        for block in self.down_blocks:
            x = block(x)
            x = F.relu(x)

        # Cross-attention: condition on RNN state
        x_flat = x.view(batch_size, -1, x.shape[1]).permute(0, 2, 1)
        attended, _ = self.cross_attention(
            query=x_flat,
            key=state_proj,
            value=state_proj
        )
        x = attended.permute(0, 2, 1).view(x.shape)

        # Up-sampling
        for block in self.up_blocks:
            x = block(x)
            x = F.relu(x)

        return x

class NeuralOSModel(nn.Module):
    """Complete NeuralOS pipeline."""
    def __init__(self, hidden_dim: int = 1024, latent_dim: int = 4):
        super().__init__()
        self.rnn_tracker = HierarchicalRNNStateTracker(
            input_dim=10,  # mouse_x, mouse_y, click, keyboard keys, etc.
            hidden_dim=hidden_dim
        )
        self.cursor_encoder = SpatialCursorEncoding(embedding_dim=hidden_dim)
        self.frame_renderer = DiffusionFrameRenderer(
            latent_dim=latent_dim, state_dim=hidden_dim
        )

    def forward(self, input_sequence: torch.Tensor, latent_frames: torch.Tensor,
                cursor_positions: torch.Tensor, timesteps: torch.Tensor):
        """
        Predict next frame given input sequence and noisy latent.

        Args:
            input_sequence: (batch, seq_len, 10) input events
            latent_frames: (batch, seq_len, latent_dim, H, W) noisy latents
            cursor_positions: (batch, seq_len, 2) normalized cursor coordinates
            timesteps: (batch, seq_len) diffusion timesteps

        Returns:
            predictions: (batch, seq_len, latent_dim, H, W)
        """
        # Track state across input sequence
        state_seq, _, _ = self.rnn_tracker(input_sequence)

        predictions = []
        for t in range(input_sequence.shape[1]):
            # Get cursor encoding for this timestep
            cursor_emb = self.cursor_encoder(
                cursor_positions[:, t, 0],
                cursor_positions[:, t, 1]
            )

            # Render frame conditioned on state
            pred = self.frame_renderer(
                latent_frames[:, t],
                timesteps[:, t].item(),
                state_seq[:, t],
                cursor_emb
            )
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

def train_neuralOS_step(model: NeuralOSModel, batch_inputs: dict,
                        optimizer: torch.optim.Optimizer,
                        diffusion_loss_fn) -> float:
    """
    Single training step with scheduled sampling (gradually use model outputs).

    Args:
        batch_inputs: Dict with keys 'input_events', 'latent_frames', 'cursor_pos', 'timesteps', 'real_frames'
        model: NeuralOS model instance
        optimizer: Training optimizer
        diffusion_loss_fn: Loss function comparing denoised frames to clean frames
    """
    optimizer.zero_grad()

    # Unpack batch
    input_events = batch_inputs['input_events']  # (batch, seq_len, 10)
    latent_frames = batch_inputs['latent_frames']  # (batch, seq_len, latent_dim, H, W)
    cursor_pos = batch_inputs['cursor_pos']  # (batch, seq_len, 2)
    timesteps = batch_inputs['timesteps']  # (batch, seq_len)
    real_frames = batch_inputs['real_frames']  # (batch, seq_len, latent_dim, H, W)

    # Forward pass
    predictions = model(input_events, latent_frames, cursor_pos, timesteps)

    # Diffusion loss: L2 distance to clean frames
    loss = diffusion_loss_fn(predictions, real_frames)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()
```

This architecture combines the efficient state tracking of RNNs (linear complexity) with diffusion-based rendering and precise spatial cursor handling.

## Practical Guidance

| Component | Configuration | Notes |
|-----------|---------------|-------|
| **RNN Hidden Dim** | 1024-2048 | Larger dim = longer context (diminishing returns after 1.5K) |
| **Diffusion Steps** | 50 (training), 20 (inference) | Fewer steps speed inference; 50-100 fine-tune optimal |
| **Context Length** | 32-64 frames | 64 frames ≈ 2 seconds at 30 fps; capture state changes |
| **Cursor Sigma** | 0.05-0.15 | Controls cursor halo size; 0.1 is balanced |
| **Batch Size** | 64-128 | Larger batches improve diffusion training stability |
| **Training Data** | 2K demos + 120K random | Ratio prevents agent-only spurious patterns |

### When to Use NeuralOS

- **GUI automation**: Recording and replaying complex multi-step user interactions
- **Testing automation**: Generating synthetic interactions for UI regression testing
- **Accessibility tools**: Predicting GUI behavior to assist users with motor impairments
- **Research on human-computer interaction**: Modeling user behavior and system response
- **Interactive AI agents**: Building agents that reason about GUI states
- **Data augmentation**: Generating additional training data for computer vision models on GUIs

### When NOT to Use

- **Real-time latency-critical systems**: 18 fps inference unacceptable for live interaction; use deterministic rule-based automation
- **Security-critical applications**: Model predictions unreliable for sensitive operations (money transfers, password entry); require explicit verification
- **Non-GUI domains**: Works only on visual interfaces; ineffective for CLI, file systems, or abstract state spaces
- **Streaming live video**: Requires offline training; not suitable for continuously changing systems
- **Models trained on different OS**: Domain gap is severe; retrain for new OS/application combinations

### Common Pitfalls

1. **Cursor Precision Ignored**: Without spatial cursor encoding, model achieves 200-500 pixel errors. Always include Gaussian spatial embeddings.
2. **Context Too Short**: 32-frame context misses windows opening. Increase to 64+ frames to capture application launch sequences.
3. **Data Imbalance**: Using only agent-generated demonstrations leads to spurious patterns (agents repeat actions). Always include random exploration data.
4. **Scheduled Sampling Rate Wrong**: If teacher-forcing ratio (real frames vs. model frames) doesn't decay gradually (e.g., 100% → 50% → 0%), error accumulation crashes training. Use exponential decay: 1.0 - 0.1^(epoch/10).
5. **Diffusion Timestep Corruption**: Ensure timesteps sampled uniformly [0, num_steps]; biased sampling toward high noise causes training instability.

## Reference

Chen, A., Li, Y., et al. (2025). NeuralOS: Towards Simulating Operating Systems via Neural Generative Models. *arXiv preprint arXiv:2507.08800*.

Available at: https://arxiv.org/abs/2507.08800
