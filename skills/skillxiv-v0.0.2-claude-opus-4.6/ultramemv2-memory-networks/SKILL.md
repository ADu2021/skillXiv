---
name: ultramemv2-memory-networks
title: "UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.18756
keywords: [memory-networks, sparse-models, long-context, parameter-scaling, activation-density]
description: "Scale memory networks to 120B parameters with improved long-context learning through integrated memory layers, simplified value projection, and optimized parameter ratios for superior memory-intensive tasks."
---

# UltraMemV2: Memory Networks Scaling to 120B Parameters

## Core Concept

UltraMemV2 advances memory-augmented transformer architectures to extreme scale while maintaining efficiency. The key insight is that activation density (which parameters actually compute) matters more than total parameter count. Through integrated memory layers in every transformer block, simplified value expansion, FFN-based processing, and principled initialization, UltraMemV2 achieves performance parity with 8-expert MoE while using significantly less memory. The approach enables 120B total parameters with only 2.5B activated, delivering +1.6 points on memorization and +7.9 points on in-context learning.

## Architecture Overview

- **Integrated Memory Layers**: Memory in every transformer block
- **Simplified Value Expansion**: Single linear projections
- **FFN-Based Value Processing**: Efficient computation
- **Principled Initialization**: Stability at scale
- **Sparse Activation**: 2.5B active from 120B total

## Implementation Steps

### 1. Design Memory Layer Architecture

Create efficient memory modules for each transformer block:

```python
import torch
import torch.nn as nn
from typing import Tuple, Optional

class MemoryLayer(nn.Module):
    """Memory module integrated into transformer block."""

    def __init__(
        self,
        hidden_size: int = 2048,
        memory_dim: int = 512,
        num_memory_slots: int = 64,
        ffn_expansion: int = 4
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.num_memory_slots = num_memory_slots

        # Memory storage (learnable)
        self.memory_slots = nn.Parameter(
            torch.randn(1, num_memory_slots, memory_dim) * 0.02
        )

        # Query projection for reading
        self.query_proj = nn.Linear(hidden_size, memory_dim)

        # Key/value for memory content addressing
        self.key_proj = nn.Linear(hidden_size, memory_dim)
        self.value_proj = nn.Linear(hidden_size, memory_dim)

        # Simplified value expansion: single linear projection
        self.value_expand = nn.Linear(memory_dim, hidden_size)

        # FFN-based value processing (instead of separate MLP)
        ffn_hidden = hidden_size * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden_size)
        )

        # Output mixing
        self.output_gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, hidden_size)
        memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory layer forward: read from/write to memory.
        """
        batch_size, seq_len, hidden_size = x.shape

        # Mean pooling over sequence for memory operations
        x_pooled = x.mean(dim=1)  # (batch, hidden_size)

        # Query and key for memory addressing
        query = self.query_proj(x_pooled)  # (batch, memory_dim)
        key = self.key_proj(x_pooled)      # (batch, memory_dim)

        # Content-based addressing to memory slots
        # Similarity: query @ memory.T
        mem_keys = self.memory_slots.expand(batch_size, -1, -1)  # (batch, num_slots, mem_dim)
        scores = torch.matmul(query.unsqueeze(1), mem_keys.transpose(1, 2))  # (batch, 1, num_slots)
        scores = scores / (self.memory_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)  # (batch, 1, num_slots)

        # Read from memory
        memory_read = torch.matmul(weights, mem_keys).squeeze(1)  # (batch, mem_dim)

        # Value projection: simplified single linear
        memory_value = self.value_expand(memory_read)  # (batch, hidden_size)

        # FFN-based processing of memory value
        memory_processed = self.ffn(memory_value)

        # Write back to memory (update memory slots with new key-value)
        # Gated write mechanism
        write_gate = torch.sigmoid(self.key_proj(x_pooled))
        self.memory_slots.data = (1 - 0.1) * self.memory_slots.data + 0.1 * write_gate.unsqueeze(1) * key.unsqueeze(1)

        # Combine input with memory output
        combined = torch.cat([x_pooled, memory_processed], dim=-1)
        output = self.output_gate(combined)

        # Residual connection
        output = output + x_pooled

        # Expand back to sequence dimension
        output = output.unsqueeze(1).expand(-1, seq_len, -1)

        return output, memory_state
```

### 2. Implement Sparse Activation Mechanism

Create selective activation for efficiency:

```python
class SparseActivationGate(nn.Module):
    """Gating mechanism for sparse activation."""

    def __init__(
        self,
        hidden_size: int,
        activation_ratio: float = 0.02  # 2.5B / 120B
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation_ratio = activation_ratio

        # Gate network: predict which tokens should use this layer
        self.gate = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        layer_module: nn.Module
    ) -> torch.Tensor:
        """
        Conditionally activate layer based on token importance.
        """
        batch_size, seq_len, hidden_size = x.shape

        # Compute activation logits
        gate_logits = self.gate(x)  # (batch, seq_len, 1)
        gate_logits = gate_logits.squeeze(-1)  # (batch, seq_len)

        # Hard thresholding for activation
        # Keep top activation_ratio fraction
        threshold = torch.quantile(
            gate_logits,
            1.0 - self.activation_ratio,
            dim=-1,
            keepdim=True
        )
        activation_mask = (gate_logits >= threshold).float()

        # Apply layer only to activated tokens
        active_tokens = activation_mask.sum().item()
        if active_tokens > 0:
            # Gather active tokens
            active_x = x * activation_mask.unsqueeze(-1)

            # Process active tokens
            output = layer_module(active_x)

            # Restore to original positions
            output = output * activation_mask.unsqueeze(-1)
        else:
            output = x

        return output, activation_mask, active_tokens

class UltraMemV2Block(nn.Module):
    """Transformer block with integrated memory and sparse activation."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        memory_dim: int = 512,
        activation_ratio: float = 0.02
    ):
        super().__init__()

        # Standard transformer components
        self.self_attn = nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.ln2 = nn.LayerNorm(hidden_size)

        # Memory layer (integrated)
        self.memory_layer = MemoryLayer(hidden_size, memory_dim)
        self.ln3 = nn.LayerNorm(hidden_size)

        # Sparse activation gates
        self.ffn_gate = SparseActivationGate(hidden_size, activation_ratio)
        self.memory_gate = SparseActivationGate(hidden_size, activation_ratio)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with memory and sparse activation."""

        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)

        # Sparse FFN
        ffn_out, ffn_mask, active_ffn = self.ffn_gate(x, self.ffn)
        x = x + ffn_out
        x = self.ln2(x)

        # Sparse Memory
        mem_out, mem_mask, active_mem = self.memory_gate(x, self.memory_layer)
        x = x + mem_out
        x = self.ln3(x)

        return x, memory_state
```

### 3. Implement Parameter Initialization Strategy

Ensure stability at large scale:

```python
class PrincipledInitializer:
    """Principled initialization for large-scale memory networks."""

    @staticmethod
    def init_memory_layer(module: MemoryLayer):
        """Initialize memory layer with stability in mind."""

        # Memory slots: small random initialization
        nn.init.normal_(module.memory_slots, std=0.01)

        # Linear projections: Xavier initialization
        for param in module.query_proj.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

        for param in module.value_expand.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

        # FFN: fan-in initialization
        for layer in module.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=1.0 / (layer.in_features ** 0.5))
                nn.init.zeros_(layer.bias)

    @staticmethod
    def init_model(model: torch.nn.Module):
        """Initialize entire model with scaled initialization."""

        num_layers = sum(1 for _ in model.modules() if isinstance(_, UltraMemV2Block))

        for module in model.modules():
            if isinstance(module, UltraMemV2Block):
                PrincipledInitializer.init_memory_layer(module.memory_layer)

                # Scale residual paths
                module.ln1.weight.data.uniform_(0.9, 1.1)
                module.ln2.weight.data.uniform_(0.9, 1.1)
                module.ln3.weight.data.uniform_(0.9, 1.1)

    @staticmethod
    def compute_init_scale(num_layers: int, hidden_size: int) -> float:
        """Compute initialization scale to prevent explosion/vanishing."""
        return 1.0 / (3.0 * (num_layers ** 0.5))
```

### 4. Implement Long-Context Evaluation

Measure performance on memory-intensive tasks:

```python
class LongContextEvaluator:
    """Evaluate long-context and memory-intensive performance."""

    @staticmethod
    def evaluate_memorization(
        model: "UltraMemV2",
        test_sequences: List[torch.Tensor],
        max_sequence_length: int = 32000
    ) -> Dict[str, float]:
        """
        Evaluate ability to memorize and recall long sequences.
        Metric: accuracy of predicting next state after long input.
        """
        model.eval()

        accuracies = []
        for sequence in test_sequences:
            with torch.no_grad():
                # Truncate to max length
                if sequence.shape[0] > max_sequence_length:
                    sequence = sequence[:max_sequence_length]

                # Forward pass
                output = model(sequence.unsqueeze(0))

                # Check final prediction
                predicted_next = output[0, -1, :].argmax()
                actual_next = sequence[-1]

                accuracy = (predicted_next == actual_next).float().item()
                accuracies.append(accuracy)

        return {
            "memorization_accuracy": sum(accuracies) / len(accuracies),
            "avg_sequence_length": sum(s.shape[0] for s in test_sequences) / len(test_sequences)
        }

    @staticmethod
    def evaluate_in_context_learning(
        model: "UltraMemV2",
        test_examples: List[Dict],
        context_length: int = 4096
    ) -> Dict[str, float]:
        """
        Evaluate in-context learning: ability to adapt to task from examples.
        """
        model.eval()

        successes = 0
        total = 0

        for example in test_examples:
            context = example["context"][:context_length]
            task = example["task"]
            expected_output = example["expected"]

            with torch.no_grad():
                # Process context
                _ = model(context)

                # Generate output for task
                output = model.generate(task, max_length=100)

                # Check if output matches expected
                if output.strip() == expected_output.strip():
                    successes += 1

            total += 1

        return {
            "in_context_accuracy": successes / max(total, 1),
            "num_examples": total
        }

    @staticmethod
    def benchmark_memory_efficiency(
        model: "UltraMemV2",
        batch_size: int = 4,
        sequence_length: int = 8192
    ) -> Dict[str, float]:
        """
        Measure memory usage and throughput.
        """
        import torch.cuda as cuda

        model.eval()

        # Create dummy input
        dummy_input = torch.randn(batch_size, sequence_length, model.hidden_size)

        # Memory before
        if cuda.is_available():
            cuda.reset_peak_memory_stats()
            memory_before = cuda.memory_allocated()

        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        elapsed_time = time.time() - start_time

        # Memory after
        if cuda.is_available():
            memory_after = cuda.memory_allocated()
            peak_memory = cuda.max_memory_allocated()
            memory_used = (peak_memory - memory_before) / (1024 ** 3)  # GB
        else:
            memory_used = 0.0

        throughput = (batch_size * sequence_length) / elapsed_time

        return {
            "memory_used_gb": memory_used,
            "throughput_tokens_per_second": throughput,
            "latency_ms": elapsed_time * 1000,
            "efficiency_tokens_per_gb": throughput / max(memory_used, 0.01)
        }
```

## Practical Guidance

### When to Use UltraMemV2

- Long-context tasks (>8k tokens)
- Memory-intensive applications (memorization, retrieval)
- Large-scale models with efficiency constraints
- In-context learning scenarios
- Production systems optimizing for memory/throughput

### When NOT to Use

- Short-context tasks (<1k tokens)
- Systems requiring minimal memory overhead
- Purely computational tasks without memory bottlenecks
- Scenarios demanding maximum accuracy over efficiency

### Key Hyperparameters

- **memory_dim**: 256-1024 (higher = more capacity)
- **num_memory_slots**: 32-256
- **activation_ratio**: 0.02-0.05 (2-5% activation)
- **ffn_expansion**: 4 standard
- **total_parameters**: Up to 120B feasible

### Performance Expectations

- Memorization Gain: +1.6 points vs. baseline
- In-Context Learning Gain: +7.9 points
- Memory Efficiency: 8-expert MoE parity with lower memory
- Activation Density: 2.5B / 120B total parameters
- Long-Context Support: 32k+ token sequences

## Reference

Researchers. (2024). UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning. arXiv preprint arXiv:2508.18756.
