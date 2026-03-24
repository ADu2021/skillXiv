---
name: deep-forcing-long-video
title: "Deep Forcing: Training-Free Long Video Generation via Deep Sink and Participative Compression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.05081
keywords: [video-generation, long-context, kv-cache-optimization, attention-sinks, temporal-modeling]
description: "Maintains half of sliding window as attention sinks with dynamic temporal RoPE alignment plus importance-aware KV cache pruning, enabling 12× extrapolation beyond training length (60+ seconds from 5-second training) without fine-tuning."
---

## Summary

Deep Forcing introduces two mechanisms enabling long video generation without fine-tuning. Deep Sink maintains approximately half of the sliding window as attention sinks while dynamically adjusting temporal RoPE to align sink tokens with current timeline. Participative Compression performs importance-aware KV cache pruning by computing attention scores between recent and candidate tokens. Together these enable 12× extrapolation with maintained visual quality.

## Core Technique

**Deep Sink Mechanism:** Attention sinks are tokens that absorb excess attention mass, preventing context collapse. The key insight is maintaining them at the current timeline:
```
Sliding window: [sink_1, sink_2, ..., recent_1, recent_2, recent_3]
                ^--- Half are sinks (fixed at current time)
```

**Dynamic Temporal RoPE:** Adjust rotary positional embeddings to align sink positions with current timeline even as the window slides:
```
rope_angle(sink_t) = current_time - training_length/2
```

**Participative Compression:** Prune KV cache by importance:
1. Compute attention scores: score[i] = query @ key[i]
2. Keep only top-k scoring tokens
3. Discard redundant tokens

## Implementation

**Sliding window with sinks:**
```python
def sliding_window_with_sinks(kv_cache, window_size=256):
    sink_ratio = 0.5
    sink_size = int(window_size * sink_ratio)

    # Maintain sinks at front
    sinks = kv_cache[:sink_size]
    recent = kv_cache[-sink_size:]

    # Concatenate sinks + recent tokens
    windowed_kv = torch.cat([sinks, recent], dim=0)

    return windowed_kv
```

**Dynamic temporal RoPE:**
```python
def dynamic_temporal_rope(positions, current_time, training_length):
    # Sink positions: earlier in sequence
    sink_positions = torch.arange(len(positions) // 2)

    # Adjust sink angles to current time
    sink_angles = current_time - (training_length / 2) + sink_positions

    # Recent positions: keep original angles
    recent_angles = current_time + torch.arange(len(positions) // 2)

    combined_angles = torch.cat([sink_angles, recent_angles])

    # Apply rotary embedding
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    angles = combined_angles.unsqueeze(-1) @ freqs.unsqueeze(0)
    rope = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    return rope
```

**Importance-aware pruning:**
```python
def participative_compression(query, kv_cache, sparsity=0.8):
    # Compute attention scores for all KV tokens
    scores = query @ kv_cache['keys'].T  # [seq_len, kv_len]

    # Keep top scoring tokens
    keep_ratio = 1.0 - sparsity
    num_keep = int(kv_cache['keys'].shape[0] * keep_ratio)

    top_k_indices = torch.topk(scores.max(dim=0).values, num_keep).indices

    # Prune KV cache
    pruned_k = kv_cache['keys'][top_k_indices]
    pruned_v = kv_cache['values'][top_k_indices]

    return {'keys': pruned_k, 'values': pruned_v}
```

**Long video generation loop:**
```python
def generate_long_video(prompt, num_frames=1440):  # 60 sec at 24fps
    # Initialize from prompt
    video = initialize_from_prompt(prompt)
    kv_cache = {}

    for frame_idx in range(num_frames):
        # Apply sliding window + sinks
        windowed_kv = sliding_window_with_sinks(kv_cache, window_size=256)

        # Apply dynamic temporal RoPE
        rope = dynamic_temporal_rope(
            positions=torch.arange(windowed_kv.shape[0]),
            current_time=frame_idx,
            training_length=120  # 5 seconds at 24fps
        )

        # Apply importance-aware pruning
        pruned_kv = participative_compression(
            query=video[-1],
            kv_cache=windowed_kv,
            sparsity=0.8
        )

        # Generate next frame
        next_frame = model(video[-1], pruned_kv, rope)
        video = torch.cat([video, next_frame.unsqueeze(0)], dim=0)

        # Update cache
        kv_cache = update_cache(pruned_kv, next_frame)

    return video
```

## When to Use

- Generating long videos (30+ seconds) from short-trained models
- Scenarios where video fine-tuning is unavailable or expensive
- Applications requiring extrapolation beyond training distribution
- Tasks balancing quality and computational efficiency

## When NOT to Use

- Short video generation where training length is sufficient
- Real-time inference where KV pruning overhead matters
- Scenarios where model fine-tuning is feasible
- Tasks requiring pixel-perfect fidelity over temporal coherence

## Key References

- Attention sinks and context collapse prevention
- Temporal rope embeddings for video transformers
- KV cache optimization and importance pruning
- Long-context generation and extrapolation
