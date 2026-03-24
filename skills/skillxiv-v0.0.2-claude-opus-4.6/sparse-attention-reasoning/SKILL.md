---
name: sparse-attention-reasoning
title: "SeerAttention-R: Sparse Attention Adaptation for Long Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.08889"
keywords: [sparse attention, reasoning decoding, efficiency, token selection]
description: "Learn sparse attention patterns for reasoning model decoding via self-distilled gating, achieving 9x speedup at 90% sparsity while maintaining reasoning quality."
---

# SeerAttention-R: Sparse Attention for Reasoning

## Core Concept

SeerAttention-R enables efficient long reasoning by learning which tokens are most important for each attention computation. A lightweight gating mechanism learns sparsity patterns during training, allowing models to focus computational resources on relevant tokens. The approach maintains near-lossless reasoning performance while achieving 9x speedup.

## Architecture Overview

- **Self-distilled gating mechanism**: Learn token importance without external labels
- **Plug-in design**: Integrates with existing models without parameter modification
- **Large sparse blocks**: 64/128 token blocks maintain coherence while enabling speedups
- **Minimal retraining**: Effective training on just 400M tokens
- **Optimized kernels**: TileLang for near-theoretical speedup on H100
- **Autoregressive compatible**: Seamlessly works with sequential token generation

## Implementation

### Step 1: Design Sparse Attention Gate

Create lightweight selector for important tokens:

```python
class SparseAttentionGate(torch.nn.Module):
    def __init__(self, hidden_dim: int = 768,
                 sparsity: float = 0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity

        # Gating network (lightweight)
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        # Temperature for sparsity control
        self.temperature = torch.nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(self, query: torch.Tensor,
               key: torch.Tensor) -> torch.Tensor:
        """Compute sparse mask indicating important keys."""

        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        # Compute importance scores
        scores = self.gate(key)  # (batch, key_len, 1)

        # Create mask based on sparsity threshold
        # Keep top (1 - sparsity) fraction of tokens
        k = max(1, int(key_len * (1 - self.sparsity)))

        # Top-k selection
        topk_scores, topk_indices = torch.topk(
            scores.squeeze(-1),
            k,
            dim=-1
        )

        # Create sparse mask
        mask = torch.zeros(batch_size, key_len,
                          device=scores.device)
        mask.scatter_(1, topk_indices, 1.0)

        return mask, topk_indices

    def forward_with_temperature(self, query: torch.Tensor,
                                key: torch.Tensor
                                ) -> tuple:
        """Sparse selection with temperature annealing."""

        # Compute importance with temperature scaling
        scores = self.gate(key) / self.temperature

        # Soften selection during early training
        mask_soft = torch.sigmoid(scores)

        # Hard selection with straight-through estimator
        mask = (mask_soft > 0.5).float()

        return mask, mask_soft
```

### Step 2: Implement Self-Distillation Training

Train gating without external labels using model self-distillation:

```python
class SelfDistillationTrainer:
    def __init__(self, model_with_gate):
        self.model = model_with_gate
        self.optimizer = torch.optim.Adam(
            self.model.gate.parameters(),
            lr=1e-4
        )

    def compute_self_distillation_loss(self,
                                      logits_full: torch.Tensor,
                                      logits_sparse: torch.Tensor,
                                      temperature: float = 4.0
                                      ) -> torch.Tensor:
        """KL divergence between full and sparse attention logits."""

        # Compute soft probabilities
        full_probs = torch.softmax(
            logits_full / temperature,
            dim=-1
        )
        sparse_probs = torch.softmax(
            logits_sparse / temperature,
            dim=-1
        )

        # KL divergence
        kl_loss = torch.nn.functional.kl_div(
            torch.log(sparse_probs),
            full_probs,
            reduction='mean'
        )

        return kl_loss

    def train_step(self, input_ids: torch.Tensor,
                  max_tokens: int = 4096) -> dict:
        """Single self-distillation training step."""

        batch_size, seq_len = input_ids.shape

        # Forward pass with full attention (teacher)
        with torch.no_grad():
            output_full = self.model(
                input_ids,
                use_sparse=False,
                output_hidden_states=True
            )
            logits_full = output_full.logits

        # Forward pass with sparse attention (student)
        output_sparse = self.model(
            input_ids,
            use_sparse=True,
            output_hidden_states=True
        )
        logits_sparse = output_sparse.logits

        # Compute distillation loss
        loss = self.compute_self_distillation_loss(
            logits_full,
            logits_sparse
        )

        # Update sparse gate only
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"distillation_loss": loss.item()}
```

### Step 3: Integrate Sparse Attention into Model

Add sparse attention as plug-in to reasoning model:

```python
class ReasoningModelWithSparseAttention(torch.nn.Module):
    def __init__(self, base_model,
                 sparsity: float = 0.9,
                 block_size: int = 128):
        super().__init__()
        self.base_model = base_model
        self.sparsity = sparsity
        self.block_size = block_size

        # Add gates for each attention head
        num_heads = base_model.config.num_attention_heads
        self.sparse_gates = torch.nn.ModuleList([
            SparseAttentionGate(
                hidden_dim=base_model.config.hidden_size,
                sparsity=sparsity
            )
            for _ in range(num_heads)
        ])

    def forward(self, input_ids: torch.Tensor,
               use_sparse: bool = True) -> dict:
        """Forward pass with optional sparse attention."""

        if not use_sparse:
            # Use standard model
            return self.base_model(input_ids)

        # Apply sparse attention via hooks
        sparse_attention_fn = self._create_sparse_attention_hook(
            use_sparse=True
        )

        # Register hooks for each attention layer
        hooks = []
        for layer in self.base_model.model.layers:
            hook = layer.self_attn.register_forward_hook(
                sparse_attention_fn
            )
            hooks.append(hook)

        # Forward pass
        output = self.base_model(input_ids)

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        return output

    def _create_sparse_attention_hook(self, use_sparse: bool):
        """Create hook for sparse attention modification."""

        def sparse_attention_hook(module, input, output):
            if not use_sparse:
                return output

            attn_output, attn_weights = output

            # Reduce attention weights based on gate
            # (This is simplified; in practice, modify QKV computation)

            return (attn_output, attn_weights)

        return sparse_attention_hook

    def forward_with_sparse_blocks(self,
                                  input_ids: torch.Tensor
                                  ) -> dict:
        """Process in large sparse blocks for efficiency."""

        seq_len = input_ids.shape[1]

        # Process in blocks
        for block_start in range(0, seq_len, self.block_size):
            block_end = min(block_start + self.block_size, seq_len)
            block_ids = input_ids[:, block_start:block_end]

            # Compute sparse attention mask for block
            with torch.no_grad():
                hidden = self.base_model.get_hidden(block_ids)

                # Use gates to identify important tokens
                masks = []
                for gate in self.sparse_gates:
                    mask, _ = gate(hidden, hidden)
                    masks.append(mask)

        output = self.base_model(input_ids)
        return output
```

### Step 4: Optimized Kernel Implementation

Implement efficient sparse attention computation:

```python
class OptimizedSparseAttentionKernel:
    """Optimized kernel for sparse attention (pseudocode)."""

    def __init__(self):
        # Would use TileLang or CUDA kernels in production
        pass

    def sparse_attention_forward(self, Q: torch.Tensor,
                                K: torch.Tensor,
                                V: torch.Tensor,
                                sparse_mask: torch.Tensor,
                                block_size: int = 128
                                ) -> torch.Tensor:
        """Compute attention only on sparse tokens."""

        # Q: (batch, num_heads, query_len, head_dim)
        # K, V: (batch, num_heads, key_len, head_dim)
        # sparse_mask: (batch, num_heads, key_len)

        batch_size, num_heads, query_len, head_dim = Q.shape

        # Compute scores only for non-masked tokens
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(head_dim))

        # Apply sparse mask
        mask_value = torch.tensor(float('-inf'))
        masked_scores = scores.masked_fill(
            sparse_mask.unsqueeze(2) == 0,
            mask_value
        )

        # Compute attention weights
        attn_weights = torch.softmax(masked_scores, dim=-1)

        # Output (only use non-masked values)
        output = torch.matmul(attn_weights, V)

        return output
```

### Step 5: Evaluate on Reasoning Tasks

Measure speedup and quality retention:

```python
def evaluate_sparse_reasoning(model,
                             benchmark_dataset: list,
                             sparsity: float = 0.9
                             ) -> dict:
    """Benchmark sparse attention on reasoning."""

    import time

    results = {
        "full_attention": {"accuracy": 0.0, "time": 0.0},
        "sparse_attention": {"accuracy": 0.0, "time": 0.0}
    }

    # Test full attention
    start = time.time()
    full_correct = 0

    for sample in benchmark_dataset:
        with torch.no_grad():
            output = model(
                sample["input_ids"],
                use_sparse=False
            )
        full_correct += (output.predictions == sample["label"]).sum()

    results["full_attention"]["accuracy"] = (
        full_correct / len(benchmark_dataset)
    )
    results["full_attention"]["time"] = time.time() - start

    # Test sparse attention
    start = time.time()
    sparse_correct = 0

    for sample in benchmark_dataset:
        with torch.no_grad():
            output = model(
                sample["input_ids"],
                use_sparse=True
            )
        sparse_correct += (output.predictions == sample["label"]).sum()

    results["sparse_attention"]["accuracy"] = (
        sparse_correct / len(benchmark_dataset)
    )
    results["sparse_attention"]["time"] = time.time() - start

    # Compute speedup
    speedup = (results["full_attention"]["time"] /
              results["sparse_attention"]["time"])

    results["speedup"] = speedup
    results["quality_retention"] = (
        results["sparse_attention"]["accuracy"] /
        results["full_attention"]["accuracy"]
    )

    return results
```

## Practical Guidance

**Sparsity Levels**: 90% sparsity achieves 9x speedup while maintaining near-lossless reasoning performance. Adjust based on latency requirements.

**Block Size**: Large blocks (64/128 tokens) maintain coherence while enabling efficient computation. Smaller blocks lose context; larger blocks reduce efficiency gains.

**Self-Distillation**: Training with full model as teacher ensures sparse model preserves reasoning quality. This eliminates need for labeled data.

**Minimal Retraining**: 400M tokens sufficient for effective sparse gate training. Don't overtrain—risk of distributional shift.

**Hardware Optimization**: Speedup depends on optimized kernels. Generic sparse operations may be slower than dense—use TileLang or custom CUDA for H100 speedups.

**When to Apply**: Use SeerAttention-R when inference latency is critical for reasoning models, or when deploying on resource-constrained hardware.

## Reference

SeerAttention-R learns to identify and focus on important tokens for reasoning via lightweight gating mechanisms. Key insight: self-distillation from the full model ensures sparse attention preserves reasoning quality while reducing computation. Achieves near-theoretical 9x speedup at 90% sparsity on modern hardware.
