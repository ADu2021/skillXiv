---
name: radial-attention-video
title: "Radial Attention: O(nlog n) Sparse Attention with Energy Decay for Long Video Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.19852"
keywords: [Sparse Attention, Video Generation, Long Sequence, Computational Efficiency, Diffusion Models]
description: "Accelerate video diffusion models using sparse radial attention that exploits energy decay patterns. Achieves 3.7× speedup on long videos while maintaining quality through O(n log n) complexity instead of O(n²)."
---

# Radial Attention: Efficient Long-Sequence Attention with Energy Decay

Video diffusion models need to attend over thousands of spatial-temporal tokens to generate coherent videos. Standard full attention requires O(n²) memory and computation—infeasible for long videos (256 or more frames). The challenge is finding a sparse pattern that preserves video quality while reducing complexity.

Radial Attention exploits a discovered property of video diffusion: attention scores naturally decay as distance between tokens increases, following an exponential pattern similar to physical signal decay. By converting this energy decay into a static sparse attention mask, you achieve O(n log n) complexity while capturing the meaningful long-range dependencies videos need.

## Core Concept

The key insight is that **dense attention weights are concentrated locally**. In a 256-frame video:

1. Tokens attend strongly to nearby frames (temporal locality)
2. Tokens attend strongly to nearby spatial positions (spatial locality)
3. Attention to distant frames decays exponentially with distance
4. You can approximate this decay pattern with a sparse mask that has logarithmic bands

Rather than computing all n² attention pairs, radial attention creates attention bands:
- Inner band (frames 0-5): full attention density
- Next band (frames 5-15): 50% density
- Next band (frames 15-50): 25% density
- Outer band (frames 50+): sparse attention

This pattern requires only O(n log n) total comparisons while preserving the exponential decay structure the model learned.

## Architecture Overview

Radial attention modifies the attention mechanism:

- **Sparse Attention Mask**: A static binary mask (computed once, reused in every forward pass) that defines which token pairs compute attention
- **Exponential Band Structure**: Attention density decreases exponentially with distance: density(distance) = (1/2)^floor(log₂(distance))
- **Spatial and Temporal Bands**: Both spatial proximity (within frame) and temporal proximity (across frames) use the same exponential decay pattern
- **LoRA Extension**: Lightweight fine-tuning adapters enable efficient adaptation to longer sequences without full retraining

## Implementation

**Step 1: Compute the radial attention mask**

Create a sparse attention pattern that encodes exponential decay.

```python
import torch
import torch.nn.functional as F
import math

def create_radial_attention_mask(seq_length, num_spatial_tokens=256,
                                 num_temporal_frames=32, device='cuda'):
    """
    Generate a sparse attention mask with radial structure.
    Attention density decays exponentially with distance.

    seq_length: total tokens (num_frames * spatial_tokens)
    num_spatial_tokens: tokens per frame (e.g., 16x16 = 256)
    num_temporal_frames: number of video frames
    """
    # Create 2D attention mask
    mask = torch.zeros(seq_length, seq_length, dtype=torch.bool, device=device)

    spatial_tokens_per_frame = num_spatial_tokens
    num_frames = num_temporal_frames

    for token_idx in range(seq_length):
        # Which frame and spatial position is this token?
        frame_i = token_idx // spatial_tokens_per_frame
        spatial_i = token_idx % spatial_tokens_per_frame

        for other_idx in range(seq_length):
            frame_j = other_idx // spatial_tokens_per_frame
            spatial_j = other_idx % spatial_tokens_per_frame

            # Compute distances
            temporal_distance = abs(frame_i - frame_j)
            spatial_distance = spatial_distance_on_grid(spatial_i, spatial_j,
                                                       grid_size=int(math.sqrt(spatial_tokens_per_frame)))

            # Radial attention rule: attend based on distance decay
            attend = should_attend_radial(temporal_distance, spatial_distance)

            if attend:
                mask[token_idx, other_idx] = True

    return mask

def spatial_distance_on_grid(pos1, pos2, grid_size):
    """
    Compute Manhattan distance on a 2D spatial grid.
    """
    x1, y1 = pos1 // grid_size, pos1 % grid_size
    x2, y2 = pos2 // grid_size, pos2 % grid_size
    return abs(x1 - x2) + abs(y1 - y2)

def should_attend_radial(temporal_distance, spatial_distance, decay_factor=0.5):
    """
    Determine if two tokens should attend using exponential decay.
    Density = (1/2)^floor(log2(distance))
    """
    # Combined distance metric: weight temporal + spatial
    # Empirically, temporal relationships matter more for video
    combined_distance = temporal_distance * 2 + spatial_distance

    if combined_distance == 0:
        # Always attend to self
        return True

    # Exponential decay: how many "doublings" of distance?
    band = math.floor(math.log2(max(combined_distance, 1)))

    # Density for this distance band
    density = decay_factor ** band

    # Stochastic threshold: attend if random() < density
    # For deterministic masking, use periodic pattern
    attend_probability = max(density, 0.05)  # Minimum density to avoid total sparsity

    # Deterministic version: attend if within threshold
    return combined_distance <= (1 / attend_probability)  # Inverted to maintain sparsity

def apply_radial_attention_mask(attn_weights, mask):
    """
    Apply the radial mask to attention weights.
    Sets masked-out positions to very negative value (they'll be ~0 after softmax).
    """
    attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
    return attn_weights
```

**Step 2: Integrate radial attention into video diffusion model**

Replace standard scaled dot-product attention with masked attention using the radial pattern.

```python
class RadialAttention(torch.nn.Module):
    """
    Multi-head attention with radial sparse masking.
    Replaces standard attention in video diffusion models.
    """

    def __init__(self, hidden_dim, num_heads=8, num_frames=32,
                 spatial_grid_size=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Pre-compute mask based on expected sequence structure
        self.num_frames = num_frames
        self.spatial_grid_size = spatial_grid_size
        self.seq_length = (spatial_grid_size ** 2) * num_frames

        # Precompute and cache the mask
        self.register_buffer(
            'radial_mask',
            create_radial_attention_mask(
                seq_length=self.seq_length,
                num_spatial_tokens=spatial_grid_size ** 2,
                num_temporal_frames=num_frames
            ),
            persistent=False
        )

        # Linear projections
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        x: [batch, seq_length, hidden_dim]
        """
        batch_size, seq_length, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, seq_length, hidden_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # [batch, seq_length, num_heads, head_dim] -> [batch, num_heads, seq_length, head_dim]
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch, num_heads, seq_length, seq_length]

        # Apply radial mask: zero out positions not in the sparse pattern
        # Adapt mask size if sequence length differs
        if seq_length != self.seq_length:
            # Create new mask for this sequence length
            mask = create_radial_attention_mask(seq_length, device=x.device)
        else:
            mask = self.radial_mask

        # Broadcast mask to batch and heads
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
        scores = apply_radial_attention_mask(scores, mask)

        # Softmax (positions masked to -inf become 0)
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch, num_heads, seq_length, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_dim)

        # Final projection
        output = self.out_proj(attn_output)

        return output
```

**Step 3: Benchmark speedup and quality**

Measure inference time and verify video generation quality remains high.

```python
def benchmark_radial_attention(model_standard, model_radial, test_prompts,
                               num_frames=256, num_inference_steps=50):
    """
    Compare standard full attention vs radial sparse attention.
    Measure latency, memory, and quality metrics.
    """
    results = {'standard': {}, 'radial': {}}

    # Warmup
    with torch.no_grad():
        _ = model_standard(test_prompts[0], num_frames=num_frames)
        _ = model_radial(test_prompts[0], num_frames=num_frames)

    # Benchmark standard attention
    latencies_standard = []
    peak_memory_standard = 0

    for prompt in test_prompts:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            output_standard = model_standard(prompt, num_frames=num_frames,
                                            num_inference_steps=num_inference_steps)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        latencies_standard.append(elapsed)
        memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        peak_memory_standard = max(peak_memory_standard, memory)

    # Benchmark radial attention
    latencies_radial = []
    peak_memory_radial = 0

    for prompt in test_prompts:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            output_radial = model_radial(prompt, num_frames=num_frames,
                                        num_inference_steps=num_inference_steps)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        latencies_radial.append(elapsed)
        memory = torch.cuda.max_memory_allocated() / (1024**2)
        peak_memory_radial = max(peak_memory_radial, memory)

    # Compute metrics
    speedup = np.mean(latencies_standard) / np.mean(latencies_radial)
    memory_reduction = (peak_memory_standard - peak_memory_radial) / peak_memory_standard

    # Quality comparison: LPIPS, FVD on video frames
    # (Details depend on your quality evaluation suite)
    quality_standard = evaluate_video_quality(output_standard)
    quality_radial = evaluate_video_quality(output_radial)
    quality_retention = quality_radial / quality_standard  # Should be > 0.95

    return {
        'speedup': speedup,
        'memory_reduction': memory_reduction,
        'quality_retention': quality_retention,
        'latency_standard_ms': np.mean(latencies_standard) * 1000,
        'latency_radial_ms': np.mean(latencies_radial) * 1000
    }
```

**Step 4: Fine-tune with LoRA for longer sequences**

Enable efficient adaptation to sequences longer than training length using Low-Rank Adapters.

```python
from peft import get_peft_model, LoraConfig

def add_lora_for_sequence_extension(model_with_radial_attention,
                                    target_seq_length=512):
    """
    Add LoRA adapters to enable efficient fine-tuning for longer sequences.
    LoRA has low rank (r=32-64) so training is cheap.
    """
    # Target attention layers for LoRA
    lora_config = LoraConfig(
        r=32,  # Low rank
        lora_alpha=64,
        target_modules=['q_proj', 'v_proj'],  # Key and value projections
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )

    # Apply LoRA to attention layers
    model_lora = get_peft_model(model_with_radial_attention, lora_config)

    # Generate a few examples with longer sequences
    # Fine-tune on these examples
    def create_long_sequence_examples(num_examples=10):
        examples = []
        for _ in range(num_examples):
            # Create synthetic long-sequence data
            video_prompt = "Generate a smooth video transition"
            examples.append(video_prompt)
        return examples

    long_examples = create_long_sequence_examples()

    # Fine-tune with LoRA
    optimizer = torch.optim.AdamW(model_lora.parameters(), lr=1e-4)

    for epoch in range(3):
        for prompt in long_examples:
            # Generate with target_seq_length
            # (requires updated mask creation for new length)
            loss = model_lora.forward_with_loss(
                prompt,
                num_frames=int(target_seq_length / 256)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model_lora
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| Decay factor | 0.5-0.7 | Controls sparsity pattern; 0.5 is standard |
| Temporal weight | 2.0-3.0 | Temporal distance matters more than spatial |
| Minimum density | 0.05 | Prevent total attention starvation |
| Band width | 1 | Logarithmic bands; change rarely |
| LoRA rank | 32-64 | Higher rank = more capacity but slower |

**When to use Radial Attention:**
- You're generating long videos (128+ frames)
- You're memory-bound (not compute-bound)
- Standard full attention causes OOM errors
- You want 2-3× speedup without retraining from scratch

**When NOT to use Radial Attention:**
- Your videos are short (< 64 frames; speedup negligible)
- You're compute-bound (GPU util already high; memory not the bottleneck)
- You need maximum quality regardless of speed (sparse attention trades some quality)
- You can't modify the attention mechanism (inference-only systems)

**Common pitfalls:**
- **Mask too sparse**: If density is too low, the model can't attend to important long-range patterns. Increase decay_factor (e.g., 0.7 instead of 0.5).
- **Flicker across frames**: If temporal bands are too tight, frames don't correlate well. Increase temporal_weight (e.g., 3.0).
- **LoRA rank too low**: If fine-tuning for longer sequences fails, increase LoRA rank (64 or 128).
- **Wrong sequence length mask**: If you change num_frames during inference, you must regenerate the mask. Cache-friendly designs should reuse masks when possible.

## Reference

Radial Attention: O(nlog n) Sparse Attention with Energy Decay for Long Video Generation
https://arxiv.org/abs/2506.19852
