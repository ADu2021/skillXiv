---
name: modular-large-model-training
title: "AXLearn: Modular Large Model Training on Heterogeneous Infrastructure"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.05411"
keywords: [Distributed Training, Model Training Framework, Hardware Agnosticism, JAX, Modularity]
description: "Train large models efficiently across heterogeneous hardware (GPUs, TPUs, Trainium) using strict encapsulation principles, achieving constant code complexity when adding features across hundreds of modules."
---

# Modular Large Model Training: Hardware-Agnostic Architecture for Scalable ML

Training large models requires coordinating across thousands of code modules—embeddings, attention, normalization, optimizers, checkpointing, monitoring. Adding a single feature (e.g., Rotary Position Embeddings) traditionally requires modifying O(N) modules where N scales with system complexity. This technical debt makes research and deployment difficult. Strict encapsulation—forcing modules to communicate only through well-defined interfaces—enables constant O(1) complexity feature addition regardless of system size.

AXLearn demonstrates that composition-based design eliminates this scaling problem, enabling the same architecture to train efficiently on GPUs, TPUs, and AWS Trainium without framework changes. The system achieves competitive inference performance by reusing training components, reducing the overhead of maintaining separate inference stacks.

## Core Concept

Most ML frameworks use subtyping and inheritance, where features propagate through class hierarchies causing O(N) or O(N²) changes. AXLearn enforces strict encapsulation: components don't inherit from common base classes but communicate through explicit interfaces. This inverts complexity—adding features requires only changing code that uses that feature, not everything that could use it.

The key insight is that functional programming (JAX) + strict encapsulation enables both modularity and hardware agnosticism simultaneously. Rather than hardware-specific code scattered throughout, a single compiler layer (XLA) handles hardware details, allowing identical module code to run on different accelerators.

## Architecture Overview

- **Strict Encapsulation Modules**: Self-contained components with explicit input/output contracts, no subtype assumptions
- **LoC-Complexity Framework**: Metric tracking asymptotic code changes required to add features across module hierarchy
- **XLA + GSPMD Backend**: Compiler-based automatic parallelization handling data and model parallel strategies without module modifications
- **InvocationContext Abstraction**: Manages functional state (randomness, dropout, batch norm statistics) while preserving JAX's pure functional semantics
- **Hardware Abstraction Layer**: Single layer supporting GPUs (CUDA/ROCm), TPUs (XLA native), and AWS Trainium, exposed identically to all modules
- **Unified Training-Inference**: Training components reused for inference with minimal overhead
- **Checkpoint and Monitoring Framework**: Generic serialization and callback systems not requiring module-specific implementations

## Implementation

The following implements modular architecture principles for scalable model training.

**Step 1: Strict Encapsulation Module Design**

This demonstrates how to design self-contained modules with encapsulation.

```python
import jax
import jax.numpy as jnp
from typing import NamedTuple, Any, Callable

class ModuleOutput(NamedTuple):
    """Explicit interface: modules only produce outputs of known type."""
    value: jnp.ndarray
    state: dict  # Module internal state changes

class StrictModule:
    """Base pattern for strictly encapsulated modules."""

    def __init__(self, config: dict):
        """Modules are configured at construction, not modified later."""
        self.config = config
        self.params = None

    def initialize_parameters(self, key: jax.random.PRNGKey, input_shape: tuple):
        """Explicit parameter initialization with shape contract."""
        raise NotImplementedError

    def __call__(
        self,
        inputs: jnp.ndarray,
        training: bool = True,
        context: dict = None
    ) -> ModuleOutput:
        """
        Single call signature regardless of module type.
        No hidden dependencies on module type or inheritance.
        """
        raise NotImplementedError

class LinearLayer(StrictModule):
    """Example: strictly encapsulated linear layer."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.output_dim = config.get("output_dim", 768)
        self.use_bias = config.get("use_bias", True)

    def initialize_parameters(self, key, input_shape):
        """Deterministic initialization based on input shape."""
        input_dim = input_shape[-1]
        key_w, key_b = jax.random.split(key)

        # He initialization
        self.params = {
            "weight": jax.random.normal(key_w, (input_dim, self.output_dim)) * jnp.sqrt(2.0 / input_dim),
        }
        if self.use_bias:
            self.params["bias"] = jnp.zeros(self.output_dim)

    def __call__(self, inputs: jnp.ndarray, training: bool = True, context: dict = None) -> ModuleOutput:
        """Standard linear transformation with no side effects."""
        output = jnp.dot(inputs, self.params["weight"])
        if self.use_bias:
            output = output + self.params["bias"]

        return ModuleOutput(
            value=output,
            state={}  # No state changes
        )

class AttentionLayer(StrictModule):
    """Example: strictly encapsulated attention."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_heads = config.get("num_heads", 12)
        self.head_dim = config.get("hidden_dim", 768) // self.num_heads

    def initialize_parameters(self, key, input_shape):
        hidden_dim = input_shape[-1]
        key_q, key_k, key_v, key_out = jax.random.split(key, 4)

        scale = jnp.sqrt(2.0 / hidden_dim)
        self.params = {
            "q_proj": jax.random.normal(key_q, (hidden_dim, hidden_dim)) * scale,
            "k_proj": jax.random.normal(key_k, (hidden_dim, hidden_dim)) * scale,
            "v_proj": jax.random.normal(key_v, (hidden_dim, hidden_dim)) * scale,
            "out_proj": jax.random.normal(key_out, (hidden_dim, hidden_dim)) * scale,
        }

    def __call__(self, inputs: jnp.ndarray, training: bool = True, context: dict = None) -> ModuleOutput:
        batch, seq_len, hidden_dim = inputs.shape

        # Project
        Q = jnp.dot(inputs, self.params["q_proj"])
        K = jnp.dot(inputs, self.params["k_proj"])
        V = jnp.dot(inputs, self.params["v_proj"])

        # Reshape for multi-head
        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, V)

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden_dim)

        # Output projection
        output = jnp.dot(attn_output, self.params["out_proj"])

        return ModuleOutput(
            value=output,
            state={}
        )
```

**Step 2: Composable Module Hierarchy**

This shows how modules compose without breaking encapsulation.

```python
class TransformerBlock(StrictModule):
    """Composing modules: attention + FFN block."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.attention = AttentionLayer(config)
        self.ffn_layer1 = LinearLayer({"output_dim": config.get("ffn_dim", 3072)})
        self.ffn_layer2 = LinearLayer({"output_dim": config.get("hidden_dim", 768)})
        self.norm1 = config.get("norm_fn", "layer_norm")

    def initialize_parameters(self, key, input_shape):
        key_attn, key_ffn1, key_ffn2 = jax.random.split(key, 3)

        self.attention.initialize_parameters(key_attn, input_shape)
        self.ffn_layer1.initialize_parameters(key_ffn1, input_shape)

        ffn_output_shape = input_shape[:-1] + (self.config.get("ffn_dim", 3072),)
        self.ffn_layer2.initialize_parameters(key_ffn2, ffn_output_shape)

        # Params from submodules
        self.params = {
            "attention": self.attention.params,
            "ffn": {
                "layer1": self.ffn_layer1.params,
                "layer2": self.ffn_layer2.params,
            }
        }

    def __call__(self, inputs: jnp.ndarray, training: bool = True, context: dict = None) -> ModuleOutput:
        # Attention with residual
        self.attention.params = self.params["attention"]
        attn_out = self.attention(inputs, training, context)
        x = inputs + attn_out.value

        # Layer norm + FFN with residual
        x_norm = jax.nn.layer_norm(x)
        self.ffn_layer1.params = self.params["ffn"]["layer1"]
        ffn_out1 = self.ffn_layer1(x_norm, training, context)
        ffn_out1_activated = jax.nn.gelu(ffn_out1.value)

        self.ffn_layer2.params = self.params["ffn"]["layer2"]
        ffn_out2 = self.ffn_layer2(ffn_out1_activated, training, context)
        x = x + ffn_out2.value

        return ModuleOutput(
            value=x,
            state={**attn_out.state, **ffn_out2.state}
        )

class TransformerModel(StrictModule):
    """Full transformer: composition of blocks."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_layers = config.get("num_layers", 12)
        self.blocks = [TransformerBlock(config) for _ in range(self.num_layers)]

    def initialize_parameters(self, key, input_shape):
        keys = jax.random.split(key, self.num_layers)
        self.params = {}

        current_shape = input_shape
        for i, block in enumerate(self.blocks):
            block.initialize_parameters(keys[i], current_shape)
            self.params[f"block_{i}"] = block.params

    def __call__(self, inputs: jnp.ndarray, training: bool = True, context: dict = None) -> ModuleOutput:
        x = inputs
        all_states = {}

        for i, block in enumerate(self.blocks):
            block.params = self.params[f"block_{i}"]
            output = block(x, training, context)
            x = output.value
            all_states.update(output.state)

        return ModuleOutput(
            value=x,
            state=all_states
        )
```

**Step 3: Hardware-Agnostic Training Loop**

This implements training that works across GPUs, TPUs, and other hardware without modification.

```python
import optax

class HardwareAgnosticTrainer:
    """Training loop transparent to hardware backend."""

    def __init__(self, model: StrictModule, config: dict):
        self.model = model
        self.config = config
        self.optimizer = optax.adam(learning_rate=config.get("learning_rate", 1e-3))

    @jax.jit  # JIT compiles to any backend automatically
    def train_step(self, params: dict, batch: dict, opt_state: dict):
        """Single training step (JIT-compiled for any hardware)."""

        def loss_fn(params):
            self.model.params = params
            output = self.model(batch["input_ids"], training=True)
            logits = output.value

            # Compute cross-entropy loss
            targets = batch["labels"]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    def train_epoch(self, train_loader, num_epochs: int = 3):
        """Training epoch (runs on whatever hardware JAX detects)."""
        # Initialize
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 512, 768))  # Batch of sequences
        self.model.initialize_parameters(key, dummy_input.shape)

        opt_state = self.optimizer.init(self.model.params)

        for epoch in range(num_epochs):
            for batch in train_loader:
                self.model.params, opt_state, loss = self.train_step(
                    self.model.params, batch, opt_state
                )

            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss:.4f}")
```

**Step 4: Feature Addition Complexity Analysis**

This demonstrates O(1) feature addition complexity.

```python
class LoC_ComplexityAnalyzer:
    """Measure lines of code required to add features."""

    @staticmethod
    def analyze_feature_addition(feature_name: str, module_count: int) -> dict:
        """
        Hypothetical feature addition: Add RotaryPositionalEmbedding to attention layers.

        In traditional frameworks (subtyping): O(N) where N = module count
        In AXLearn (strict encapsulation): O(1)
        """

        # AXLearn approach: only modify AttentionLayer
        axlearn_changes = {
            "files_modified": 1,  # Only attention.py
            "new_classes": 1,  # RotaryEmbedding
            "modified_methods": 1,  # AttentionLayer.__call__
            "total_loc_added": 50,  # ~50 lines
            "complexity": "O(1)"
        }

        # Traditional approach: modify every attention usage
        traditional_changes = {
            "files_modified": module_count,  # Every attention use site
            "modified_methods": module_count,
            "total_loc_added": module_count * 10,  # ~10 lines per site
            "complexity": f"O(N) where N={module_count}"
        }

        return {
            "axlearn": axlearn_changes,
            "traditional": traditional_changes,
            "complexity_improvement": module_count
        }
```

## Practical Guidance

**Hyperparameters and Configuration**

| Parameter | Recommended Value | Range | Notes |
|-----------|------------------|-------|-------|
| Batch Size | 32-128 per device | 1-1024 | Adjust for memory constraints of hardware |
| Learning Rate | 1e-3 | 1e-5 to 1e-1 | Use warmup + decay schedule |
| Gradient Accumulation | 1-4 steps | 1-16 | Simulate larger batches on memory-limited hardware |
| Checkpoint Frequency | Every 1000 steps | 100-10000 | More frequent on unstable hardware |
| Max Sequence Length | 512-2048 | 128-16384 | Hardware-dependent; TPU handles longer better |
| Mixed Precision (bfloat16) | Enabled | True/False | Reduces memory 2×, better on TPU than GPU |

**When to Use**

- Training large models (7B+ parameters) requiring multiple hardware types
- Research projects adding features frequently
- Production training pipelines across heterogeneous infrastructure
- Cloud environments where hardware changes based on availability
- Scenarios where training-inference parity is critical

**When NOT to Use**

- Small models where modularity overhead outweighs benefits
- Real-time systems needing per-device optimization (JAX abstraction loses device-specific tweaks)
- Hardware requiring custom kernels (e.g., sparse tensor operations)
- Projects locked into specific frameworks (PyTorch) for ecosystem reasons
- Development where interactive debugging is essential (JIT makes debugging harder)

**Common Pitfalls**

- **Leaking implementation details**: Modules accidentally exposing internal state or parameter structures break encapsulation. Use ModuleOutput consistently.
- **Over-generalizing**: Strict encapsulation doesn't mean every detail should be generic. Keep interfaces simple and explicit.
- **JIT performance surprises**: First JIT compilation is slow. Account for warmup time in benchmarks.
- **Hardware-specific bugs surfacing late**: Test on target hardware early. CPU/GPU/TPU numerical differences emerge after weeks of training.
- **Ignoring gradient accumulation overhead**: Proper gradient handling with multiple devices requires careful state management. Test distributed training from the start.

## Reference

AXLearn: Modular Large Model Training on Heterogeneous Infrastructure. https://arxiv.org/abs/2507.05411
