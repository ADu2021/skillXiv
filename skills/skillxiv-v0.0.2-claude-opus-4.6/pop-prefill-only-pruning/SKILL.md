---
name: pop-prefill-only-pruning
title: "POP: Prefill-Only Pruning for Efficient Large Model Inference"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03295"
keywords: [Layer Pruning, Inference Optimization, Prefill Stage, Model Efficiency, No Retraining]
description: "Remove deep layers during context encoding (prefill) while keeping them for token generation (decode). Identifies layer importance asymmetry via virtual gates; achieves 1.37x prefill speedup without retraining on any pre-trained model."
---

# POP: Stage-Aware Layer Pruning for Inference

Standard inference removes entire layers, but different layers matter differently across inference stages. During prefill (context encoding), deep layers are largely redundant; during decode (token generation), they're critical. POP exploits this asymmetry by removing layers only during prefill, using virtual gates to identify which layers to prune without requiring model retraining.

The key insight is that layer importance changes dramatically between prefill and decode. This suggests a pragmatic pruning strategy that's stage-aware rather than uniform.

## Core Concept

POP operates on the observation that:

1. **Prefill stage**: Context encoding where tokens attend over full sequence; deep layers contribute minimally
2. **Decode stage**: Single-token generation where all layers are needed for quality

By identifying prunable layers via a virtual gate mechanism, POP removes them during prefill only, maintaining full model capacity for decode.

## Architecture Overview

- **Virtual Gate Estimator**: Lightweight mechanism to estimate layer importance without retraining
- **Prunable Layer Identification**: Determines which deep layers can be removed during prefill
- **Stage-Aware Execution**: Disable prunable layers only in prefill; enable in decode
- **KV Projection Handling**: Ensure cache dimensions remain consistent
- **Boundary Safeguards**: Final token always uses full model for stability

## Implementation

### Step 1: Design Virtual Gate Mechanism

Create an importance estimator using second-order information.

```python
# Virtual gate for layer importance
import torch
import torch.nn.functional as F

class VirtualGate:
    def __init__(self, model: nn.Module, eps: float = 1e-8):
        """
        Estimate layer importance via Fisher information.

        Args:
            model: Language model to analyze
            eps: Numerical stability
        """
        self.model = model
        self.eps = eps

    def estimate_layer_importance(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, hidden_dim]
        layer_idx: int,
        loss_fn
    ) -> float:
        """
        Estimate importance of layer by approximating loss change if removed.

        Uses: ΔL ≈ (1/2) g^T H^{-1} g where H is Hessian, g is gradient

        Args:
            hidden_states: Input to layer
            layer_idx: Which layer to evaluate
            loss_fn: Loss function for measurement

        Returns:
            Importance score [0, 1]
        """
        layer = self.model.layers[layer_idx]

        # Compute gradient of loss w.r.t. layer output
        hidden_states.requires_grad = True
        output = layer(hidden_states)
        loss = loss_fn(output)
        loss.backward()

        gradient = hidden_states.grad

        # Fisher approximation: diag(H) ≈ E[g^2]
        fisher_diag = torch.mean(gradient ** 2, dim=[0, 1])

        # Approximate importance: sum of |gradient| weighted by inverse Fisher
        importance = torch.sum(
            torch.abs(gradient.mean(dim=[0, 1])) / (fisher_diag + self.eps)
        )

        return min(1.0, importance.item())

    def rank_layers_by_importance(
        self,
        val_dataset: List[torch.Tensor],
        loss_fn
    ) -> List[Tuple[int, float]]:
        """
        Rank all layers by importance.

        Returns:
            List of (layer_idx, importance_score) tuples, sorted by importance
        """
        layer_scores = []

        for layer_idx in range(len(self.model.layers)):
            scores = []

            for hidden_states in val_dataset:
                score = self.estimate_layer_importance(
                    hidden_states,
                    layer_idx,
                    loss_fn
                )
                scores.append(score)

            avg_score = sum(scores) / len(scores)
            layer_scores.append((layer_idx, avg_score))

        # Sort by importance (ascending)
        sorted_layers = sorted(layer_scores, key=lambda x: x[1])
        return sorted_layers
```

### Step 2: Identify Prunable Layers

Determine which layers can be safely removed during prefill.

```python
# Prunable layer identification
class PrunableLayerAnalyzer:
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.3):
        """
        Identify which layers are prunable during prefill.

        Args:
            model: Language model
            pruning_ratio: Target fraction of layers to prune
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.num_layers = len(model.layers)
        self.target_prune_count = int(self.num_layers * pruning_ratio)

    def identify_prunable_layers(
        self,
        val_dataset: List[dict]
    ) -> List[int]:
        """
        Determine which deep layers are least important.

        Args:
            val_dataset: Validation data for importance scoring

        Returns:
            List of layer indices to prune during prefill
        """
        gate = VirtualGate(self.model)

        def loss_fn(output):
            # Simple loss: just magnitude as proxy for importance
            return torch.mean(output ** 2)

        # Get hidden states from validation set
        hidden_states_list = []
        for sample in val_dataset:
            with torch.no_grad():
                hidden = self.model.embed_tokens(sample["input_ids"])
            hidden_states_list.append(hidden)

        # Rank layers
        ranked_layers = gate.rank_layers_by_importance(
            hidden_states_list,
            loss_fn
        )

        # Select bottom N layers (least important)
        prunable = [layer_idx for layer_idx, _ in ranked_layers[:self.target_prune_count]]

        # Constraint: only consider deep layers (last 50%)
        deep_layer_threshold = self.num_layers // 2
        prunable = [l for l in prunable if l >= deep_layer_threshold]

        return sorted(prunable)
```

### Step 3: Implement Stage-Aware Execution

Create inference wrapper that applies pruning only during prefill.

```python
# Stage-aware model wrapper
class PrefillOnlyPrunedModel(nn.Module):
    def __init__(self, model: nn.Module, prunable_layers: List[int]):
        """
        Wrap model to prune layers only during prefill.

        Args:
            model: Base language model
            prunable_layers: Layers to skip during prefill
        """
        super().__init__()
        self.model = model
        self.prunable_layers = set(prunable_layers)
        self.is_prefill = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False
    ):
        """
        Forward pass with stage-aware pruning.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional mask
            past_key_values: KV cache for decode
            use_cache: Whether to return cache

        Returns:
            logits and cache if use_cache=True
        """
        # Determine stage
        # Prefill: seq_len > 1 or no past_key_values
        # Decode: seq_len == 1 and past_key_values present
        is_prefill = (input_ids.shape[1] > 1) or (past_key_values is None)
        self.is_prefill = is_prefill

        # Embedding
        hidden_states = self.model.embed_tokens(input_ids)

        new_cache = [] if use_cache else None

        # Process through layers
        for layer_idx, layer in enumerate(self.model.layers):
            # Skip prunable layers during prefill only
            if is_prefill and layer_idx in self.prunable_layers:
                # Skip layer but maintain hidden state (residual path)
                if new_cache is not None:
                    new_cache.append(None)
                continue

            # Apply layer normally
            if past_key_values is not None:
                past = past_key_values[layer_idx]
            else:
                past = None

            hidden_states, cache_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=use_cache
            )

            if use_cache:
                new_cache.append(cache_out)

        # Output projection
        logits = self.model.lm_head(hidden_states)

        if use_cache:
            return logits, new_cache
        else:
            return logits

    def generate(self, input_ids: torch.Tensor,
                max_new_tokens: int = 128) -> torch.Tensor:
        """Generate tokens using stage-aware pruning."""
        generated = input_ids.clone()
        cache = None

        for _ in range(max_new_tokens):
            # Get logits for next token
            if cache is None:
                # Prefill: process all context
                logits, cache = self.forward(
                    generated,
                    use_cache=True
                )
                next_logits = logits[:, -1, :]
            else:
                # Decode: process only last token
                logits, cache = self.forward(
                    generated[:, -1:],
                    past_key_values=cache,
                    use_cache=True
                )
                next_logits = logits[:, 0, :]

            # Sample next token
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
```

### Step 4: Handle KV Cache Consistency

Ensure cache projections remain valid despite pruned layers.

```python
# KV cache handling for pruned layers
class PrunedKVCacheManager:
    def __init__(self, model: nn.Module, prunable_layers: List[int]):
        """
        Manage KV caches when layers are pruned.

        Args:
            model: Language model
            prunable_layers: Layers that are pruned
        """
        self.prunable_layers = set(prunable_layers)
        self.num_layers = len(model.layers)

    def initialize_cache_structure(self, batch_size: int,
                                  seq_len: int) -> List:
        """Create cache structure accounting for pruned layers."""
        cache = []

        for layer_idx in range(self.num_layers):
            if layer_idx in self.prunable_layers:
                # No cache needed for pruned layers
                cache.append(None)
            else:
                # Initialize K, V caches
                cache.append({
                    "k": torch.zeros(batch_size, seq_len, dtype=torch.float32),
                    "v": torch.zeros(batch_size, seq_len, dtype=torch.float32)
                })

        return cache

    def validate_cache_consistency(self, cache: List) -> bool:
        """Check that cache matches expected structure."""
        for layer_idx, cache_entry in enumerate(cache):
            if layer_idx in self.prunable_layers:
                if cache_entry is not None:
                    return False
            else:
                if cache_entry is None:
                    return False

        return True
```

### Step 5: Complete Pruning Pipeline

Assemble full POP system.

```python
# Complete POP pipeline
def apply_pop_pruning(
    model: nn.Module,
    val_dataset: List[dict],
    pruning_ratio: float = 0.3
) -> PrefillOnlyPrunedModel:
    """
    Apply Prefill-Only Pruning to a model without retraining.

    Args:
        model: Pre-trained language model
        val_dataset: Validation data for importance estimation
        pruning_ratio: Fraction of layers to prune (typically 0.2-0.4)

    Returns:
        Model with pruning enabled
    """
    # Analyze layer importance
    analyzer = PrunableLayerAnalyzer(model, pruning_ratio)
    prunable_layers = analyzer.identify_prunable_layers(val_dataset)

    print(f"Identified {len(prunable_layers)} layers as prunable: {prunable_layers}")

    # Create pruned model
    pruned_model = PrefillOnlyPrunedModel(model, prunable_layers)

    return pruned_model

def benchmark_pop_speedup(
    original_model: nn.Module,
    pruned_model: PrefillOnlyPrunedModel,
    test_inputs: List[torch.Tensor]
):
    """Measure speedup from prefill-only pruning."""
    import time

    # Benchmark prefill
    prefill_times_original = []
    prefill_times_pruned = []

    for input_ids in test_inputs:
        # Original model
        start = time.time()
        with torch.no_grad():
            _ = original_model(input_ids)
        prefill_times_original.append(time.time() - start)

        # Pruned model
        start = time.time()
        with torch.no_grad():
            _ = pruned_model(input_ids)
        prefill_times_pruned.append(time.time() - start)

    avg_original = sum(prefill_times_original) / len(prefill_times_original)
    avg_pruned = sum(prefill_times_pruned) / len(prefill_times_pruned)

    speedup = avg_original / avg_pruned
    print(f"Prefill speedup: {speedup:.2f}x")

    return speedup
```

## Practical Guidance

**When to use POP:**
- Long-context inference where prefill dominates latency
- Scenarios accepting <1% accuracy loss for ~1.3x speedup
- Any pre-trained model without retraining budget
- Multimodal models (images/text) with expensive prefill

**When not to use:**
- Short-context inference (<512 tokens)
- Latency-critical applications needing predictable per-token timing
- Models already optimized with other compression methods
- Tasks requiring maximum model capacity

**Common Pitfalls:**
- Pruning too aggressively: Start at 20% pruning; increase if stable
- Not validating on representative data: Val set should match deployment
- Ignoring final-token safety: Always use full model for final output
- Mixing with other optimizations: Test pruning in isolation first

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| pruning_ratio | 0.2-0.4 | Lower = safer; higher = more speedup |
| layer_threshold | 50% | Only prune deep layers; conservative default |
| validation_size | 100-500 examples | More data = better importance estimates |

## Reference

See the full paper at: https://arxiv.org/abs/2602.03295

Key results: 1.37× prefill speedup on Llama-3.1, Qwen3-VL, Gemma-3 without retraining. Works on any model. Particularly effective for multimodal inference.
