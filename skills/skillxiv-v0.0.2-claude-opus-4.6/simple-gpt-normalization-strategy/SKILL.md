---
name: simple-gpt-normalization-strategy
title: "SimpleGPT: Improving GPT via A Simple Normalization Strategy"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.01212"
keywords: [Normalization Strategy, Training Stability, Learning Rate, Hessian Conditioning, Transformer Architecture]
description: "Place RMSNorm immediately after every linear layer to stabilize activation scales at O(sqrt(d)) and reduce Hessian spectral norm. Enables 3-10x larger learning rates and faster convergence without architectural changes; improves loss by 0.08 on 7B models."
---

# SimpleGPT: Linear-Layer Normalization for Training Stability

Standard Transformer architectures place LayerNorm only before attention and MLPs, allowing activation scales to drift during training. SimpleGPT proposes a minimal change: insert RMSNorm immediately after every linear transformation (Q/K/V projections, MLPs, output projections). This simple shift dramatically improves optimization geometry, enabling much larger learning rates and faster convergence.

The key insight is that normalization right after linear projections stabilizes the activation scale that linear layers output, preventing the representation drift that limits learning rate. This is grounded in second-order optimization: the Hessian spectral norm directly limits maximum learning rate, and SimpleNorm reduces this bound significantly.

## Core Concept

SimpleNorm applies a uniform pattern across the entire model:

**Standard Transformer**: `Linear → Activation → (Attention/MLP)`

**SimpleGPT**: `Linear → RMSNorm → Activation → (Attention/MLP)`

This seemingly small change has outsized impact because:

1. **Stabilized activation scale**: RMSNorm forces activations to remain at O(√d), preventing explosion or vanishing
2. **Reduced Hessian curvature**: Normalization smooths the loss landscape, reducing spectral norm of Hessian
3. **Weight-scale invariance**: Unlike unnormalized layers where curvature scales with ||W||₂², normalized layers maintain consistent curvature regardless of weight magnitude

## Architecture Overview

- **Linear projections** (Q, K, V in attention; feed-forward up/down): Apply RMSNorm immediately after
- **Output projections** (attention out, MLP out): Apply RMSNorm after
- **RMSNorm placement**: Before activation functions (ReLU, GELU, etc.)
- **Hyperparameter scaling**: Increase learning rate 3-10× based on model size
- **torch.compile integration**: JIT compilation adds minimal overhead

## Implementation

### Step 1: Define RMSNorm Operation

Create an efficient RMSNorm that normalizes to unit RMS value.

```python
# RMSNorm implementation
class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        """
        Root Mean Square Layer Normalization.

        Args:
            hidden_dim: Dimension to normalize over
            eps: Numerical stability epsilon
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm: x / sqrt(E[x^2] + eps) * weight

        Args:
            x: Input tensor of any shape with last dim = hidden_dim

        Returns:
            Normalized tensor, same shape as input
        """
        # Compute RMS over last dimension
        rms = torch.sqrt(
            torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps
        )
        # Normalize and scale
        normalized = x / rms
        return normalized * self.weight
```

### Step 2: Modify Linear Layers with Post-Linear Normalization

Create a composite module combining linear projection and immediate normalization.

```python
# Linear layer with post-normalization
class NormalizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 hidden_dim: Optional[int] = None):
        """
        Linear layer with RMSNorm immediately after.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            hidden_dim: Dimension for RMSNorm (default: out_features)
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.norm = RMSNorm(hidden_dim or out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear projection then normalize."""
        x = self.linear(x)
        x = self.norm(x)
        return x

# Convenience wrapper for attention projections
class NormalizedAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        """Attention with normalized Q/K/V projections."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        # Q, K, V projections with post-norm
        self.q_proj = NormalizedLinear(hidden_dim, hidden_dim, hidden_dim)
        self.k_proj = NormalizedLinear(hidden_dim, hidden_dim, hidden_dim)
        self.v_proj = NormalizedLinear(hidden_dim, hidden_dim, hidden_dim)

        # Output projection with post-norm
        self.out_proj = NormalizedLinear(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard attention with normalized projections."""
        batch_size, seq_len, _ = hidden_states.shape

        # Project and split heads
        queries = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, -1
        ).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, -1
        ).transpose(1, 2)
        values = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, -1
        ).transpose(1, 2)

        # Standard scaled dot-product attention
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.hidden_dim // self.num_heads)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection with normalization
        output = self.out_proj(attn_output)

        return output
```

### Step 3: Implement MLP with Normalized Projections

Create feed-forward network with normalized intermediate layers.

```python
# Normalized MLP
class NormalizedMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 activation: str = "gelu"):
        """
        Feed-forward network with normalized projections.

        Args:
            hidden_dim: Input/output dimension
            intermediate_dim: Hidden layer dimension (typically 4x)
            activation: Activation function name
        """
        super().__init__()
        self.up_proj = NormalizedLinear(hidden_dim, intermediate_dim, intermediate_dim)
        self.down_proj = NormalizedLinear(intermediate_dim, hidden_dim, hidden_dim)

        if activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Up-project, activate, down-project with normalization."""
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x
```

### Step 4: Assemble into Complete Transformer Layer

Integrate normalized components into standard transformer architecture.

```python
# Transformer layer with SimpleNorm
class SimpleNormTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int):
        """Transformer layer using SimpleNorm pattern."""
        super().__init__()

        # Attention with normalized projections
        self.self_attn = NormalizedAttention(hidden_dim, num_heads)

        # MLP with normalized projections
        self.mlp = NormalizedMLP(hidden_dim, intermediate_dim)

        # Pre-norm for attention and MLP (standard practice)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transformer layer: pre-norm attention + pre-norm MLP + residuals."""
        # Attention block with residual
        attn_input = self.norm1(x)
        attn_output = self.self_attn(attn_input)
        x = x + attn_output

        # MLP block with residual
        mlp_input = self.norm2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x
```

### Step 5: Training Configuration with Increased Learning Rates

Set up training with larger learning rates enabled by SimpleNorm.

```python
# Training with SimpleNorm learning rates
def train_with_simple_norm(
    model: nn.Module,
    train_dataset,
    base_learning_rate: float = 1e-3,
    scale_factor: float = 5.0,
    num_epochs: int = 10
):
    """
    Train model with SimpleNorm using increased learning rates.

    Args:
        model: Model using NormalizedLinear layers
        train_dataset: Training data
        base_learning_rate: Starting learning rate
        scale_factor: Multiplier for LR (typically 3-10x)
        num_epochs: Training epochs
    """
    # Use scaled learning rate
    learning_rate = base_learning_rate * scale_factor

    # Standard optimizer (AdamW recommended)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_dataset),
        eta_min=learning_rate * 0.1
    )

    # Optionally compile with torch.compile for speedup
    # (adds ~3% overhead but improves overall training)
    model = torch.compile(model)

    # Standard training loop
    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (input_ids, labels) in enumerate(train_dataset):
            # Forward pass
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

    return model

def validate_convergence_speedup(model_simple: nn.Module,
                                model_standard: nn.Module,
                                validation_data):
    """
    Compare convergence speed: SimpleNorm vs standard.

    Returns:
        Speedup ratio (iterations to convergence)
    """
    def measure_convergence(model, target_loss=0.1):
        steps = 0
        loss = float('inf')
        while loss > target_loss:
            # Training step...
            steps += 1
            if steps > 10000:
                break
        return steps

    steps_simple = measure_convergence(model_simple)
    steps_standard = measure_convergence(model_standard)

    speedup = steps_standard / steps_simple
    print(f"SimpleNorm converges {speedup:.2f}x faster")
    return speedup
```

## Practical Guidance

**When to use SimpleGPT normalization:**
- Model training where convergence speed is critical
- Any Transformer-based architecture (language, vision, multimodal)
- Fine-tuning scenarios where larger learning rates help
- Systems where 3-8% training time reduction is valuable

**When not to use:**
- Already-converged models (no re-training needed)
- Systems with highly optimized training pipelines where changes are risky
- Inference-only scenarios (SimpleNorm has negligible inference cost)

**Common Pitfalls:**
- Learning rate too large: Even with SimpleNorm, overly aggressive LR causes divergence; start conservative and increase
- Skipping layer normalization: Must place RMSNorm after every linear layer for stability benefits
- Not adjusting warmup: Warmup schedule may need adjustment for larger learning rates
- torch.compile incompatibility: Ensure all operations are supported by compiler

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| Learning rate scale | 3x-10x | Start at 3x; increase if stable; 10x for aggressive schedules |
| Weight decay | 0.01-0.05 | Standard value; increase if regularization needed |
| Gradient clipping | 1.0 | Keep standard clipping; SimpleNorm doesn't eliminate need |
| Warmup epochs | 0.5-2% | May increase slightly; SimpleNorm more stable earlier |

## Reference

See the full paper at: https://arxiv.org/abs/2602.01212

Key results: 0.08 point loss improvement on 7B Llama3 models; 3-10× larger learning rates enabled; validated across nanoGPT (120M), Llama2 (7B), and Llama3 (8B). Minimal training overhead with torch.compile. No architectural changes required.
