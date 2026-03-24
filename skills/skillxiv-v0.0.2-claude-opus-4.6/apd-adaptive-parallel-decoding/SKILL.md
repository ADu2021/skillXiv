---
name: apd-adaptive-parallel-decoding
title: "Accelerating Diffusion LLMs via Adaptive Parallel Decoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.00413"
keywords: [diffusion-llm, parallel-decoding, speculative-decoding, inference-acceleration, adaptive-sampling]
description: "Accelerate diffusion language model inference by dynamically adjusting parallel tokens per step using a small auxiliary autoregressive model, achieving substantial throughput gains."
---

# Accelerating Diffusion LLMs via Adaptive Parallel Decoding

## Core Concept

Diffusion language models (dLLMs) theoretically enable parallel token generation but struggle to match autoregressive models' speed-quality tradeoffs. APD (Adaptive Parallel Decoding) bridges this gap by dynamically adjusting how many tokens are sampled in parallel based on quality assessment from a lightweight auxiliary model. The key innovation: use multiplicative mixture of dLLM marginals with autoregressive joint probabilities, accepting tokens only while both distributions agree. This converts diffusion into left-to-right generation while preserving parallelism where distributions align.

Results show substantial throughput gains (5-8 parallel tokens per step) while maintaining ~80% accuracy on mathematical reasoning tasks, surpassing autoregressive Qwen 7B in speed.

## Architecture Overview

- **Dual Distribution Mixture**: Combines dLLM marginal probabilities with autoregressive joint probabilities
- **Universal Coupling**: Gumbel-Softmax samples ensure principled token acceptance across distributions
- **Adaptive Parallelization**: Number of parallel tokens varies per step based on distribution agreement
- **KV Caching Optimization**: Efficient caching for both diffusion and autoregressive branches
- **Three Tunable Parameters**: Mixture weight R, KV cache window W, maximum masked lookahead M

## Implementation

1. **Mixture Distribution Definition**: Balance diffusion and autoregressive predictions

```python
def compute_mixture_distribution(diffusion_logits, ar_logits, mixture_weight_R):
    """
    Combine diffusion model's marginal distribution with autoregressive
    joint distribution. Higher R favors diffusion, lower favors AR.
    """
    # Diffusion: marginal probability of each token at current position
    p_diffusion = torch.softmax(diffusion_logits, dim=-1)

    # Autoregressive: joint probability given context
    p_ar = torch.softmax(ar_logits, dim=-1)

    # Multiplicative mixture: pT(x) ∝ p_D(x)^R * p_AR(x)^(1-R)
    # Take log to avoid numerical underflow
    log_mixture = R * torch.log(p_diffusion + 1e-10) + (1 - R) * torch.log(p_ar + 1e-10)

    # Normalize to valid probability distribution
    p_mixture = torch.softmax(log_mixture, dim=-1)

    return p_mixture
```

2. **Universal Coupling Token Acceptance**: Use Gumbel-Softmax for principled acceptance

```python
def universal_coupling_acceptance(mixture_dist, ar_dist, num_parallel_tokens=5):
    """
    Sample tokens from mixture distribution. Accept tokens only where
    mixture and autoregressive distributions agree (within threshold).
    Determines parallelization factor adaptively.
    """
    # Gumbel-max trick: sample from categorical using Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(mixture_dist) + 1e-10) + 1e-10)

    # Mixture samples
    mixture_samples = torch.argmax(
        torch.log(mixture_dist) + gumbel_noise, dim=-1
    )

    # AR samples (for comparison)
    ar_samples = torch.argmax(
        torch.log(ar_dist) + gumbel_noise, dim=-1
    )

    # Acceptance criterion: mixture and AR agree
    agreement = (mixture_samples == ar_samples)

    # Accept tokens while both distributions agree
    accepted_tokens = []
    disagreement_index = None

    for i in range(min(num_parallel_tokens, len(mixture_samples))):
        if agreement[i]:
            accepted_tokens.append(mixture_samples[i])
        else:
            disagreement_index = i
            break

    # Adaptive parallelization: return accepted tokens count
    return torch.stack(accepted_tokens), len(accepted_tokens)
```

3. **Efficient KV Caching**: Maintain cache for both branches

```python
def adaptive_kv_cache_update(kv_cache_diffusion, kv_cache_ar,
                             new_tokens, window_size_W, lookahead_M):
    """
    Update KV caches for both diffusion and AR models.
    Limit cache size to window_size_W for efficiency.
    Lookahead_M determines masked prediction window for diffusion.
    """
    # Append new tokens to both caches
    kv_cache_diffusion['keys'] = torch.cat([kv_cache_diffusion['keys'], new_tokens], dim=1)
    kv_cache_diffusion['values'] = torch.cat([kv_cache_diffusion['values'], new_tokens], dim=1)

    kv_cache_ar['keys'] = torch.cat([kv_cache_ar['keys'], new_tokens], dim=1)
    kv_cache_ar['values'] = torch.cat([kv_cache_ar['values'], new_tokens], dim=1)

    # Maintain sliding window for diffusion (processes in parallel)
    if len(kv_cache_diffusion['keys']) > window_size_W:
        # Keep most recent window_size_W tokens
        start_idx = len(kv_cache_diffusion['keys']) - window_size_W
        kv_cache_diffusion['keys'] = kv_cache_diffusion['keys'][:, start_idx:, :]
        kv_cache_diffusion['values'] = kv_cache_diffusion['values'][:, start_idx:, :]

    # Limit lookahead for masked diffusion prediction
    if lookahead_M > 0:
        # Only allow attending to next M positions ahead
        max_future = len(kv_cache_diffusion['keys']) + lookahead_M
        # Create attention mask for lookahead

    return kv_cache_diffusion, kv_cache_ar
```

4. **Generation Loop**: Integrate adaptive parallelization

```python
def generate_with_apd(diffusion_model, ar_model, prompt_ids, max_length,
                      mixture_weight_R=0.5, window_W=128, lookahead_M=32):
    """
    Generate tokens using adaptive parallel decoding.
    Dynamically determines parallelization per step.
    """
    generated = prompt_ids.clone()
    kv_diffusion = {'keys': None, 'values': None}
    kv_ar = {'keys': None, 'values': None}

    parallel_token_counts = []

    while len(generated) < max_length:
        # Encode current context
        context = generated[-window_W:]  # Use sliding window

        # Get diffusion predictions (multiple masked positions)
        diffusion_output = diffusion_model(
            context,
            num_steps=5,
            kv_cache=kv_diffusion,
            lookahead=lookahead_M
        )
        diffusion_logits = diffusion_output.logits[:, -1, :]

        # Get AR predictions (single position)
        ar_output = ar_model(context, kv_cache=kv_ar)
        ar_logits = ar_output.logits[:, -1, :]

        # Compute mixture and sample
        mixture_dist = compute_mixture_distribution(
            diffusion_logits, ar_logits, mixture_weight_R
        )

        # Accept tokens while distributions agree
        new_tokens, num_accepted = universal_coupling_acceptance(
            mixture_dist, torch.softmax(ar_logits, dim=-1), num_parallel_tokens=10
        )

        # Append accepted tokens
        generated = torch.cat([generated, new_tokens], dim=1)
        parallel_token_counts.append(num_accepted)

        # Update KV caches
        kv_diffusion, kv_ar = adaptive_kv_cache_update(
            kv_diffusion, kv_ar, new_tokens, window_W, lookahead_M
        )

    avg_parallel = sum(parallel_token_counts) / len(parallel_token_counts)
    return generated, avg_parallel
```

5. **Tuning Parameters**: Three key hyperparameters control tradeoff

```python
# Configuration for speed-quality balance
APD_CONFIG = {
    'mixture_weight_R': {
        'description': 'Balance between diffusion (R=1.0) and AR (R=0.0)',
        'default': 0.5,
        'range': (0.3, 0.8),
        'effect': 'Higher R → more parallelism, lower quality'
    },
    'kv_cache_window_W': {
        'description': 'Context window for KV caching',
        'default': 128,
        'range': (64, 256),
        'effect': 'Larger window → more context but slower'
    },
    'max_lookahead_M': {
        'description': 'Maximum lookahead for masked diffusion prediction',
        'default': 32,
        'range': (16, 64),
        'effect': 'Larger M → more parallelism, less accurate predictions'
    }
}
```

## Practical Guidance

**When to Apply:**
- Deploying diffusion language models where latency is critical
- Need speed improvements while maintaining reasoning accuracy
- Have auxiliary small autoregressive model available

**Implementation Prerequisites:**
- Pre-trained diffusion language model (Dream 7B or similar)
- Small autoregressive verifier model (0.5B-1B range, e.g., Qwen2.5-0.5B)
- Sufficient GPU memory for dual model inference

**Performance Expectations:**
- Parallel tokens per step: 5-8 on average
- Throughput: Substantially higher than baseline dLLM
- Accuracy: ~80% on mathematical reasoning (GSM8K, GPQA, MATH)
- Speed-quality tradeoff: Tunable via R parameter

**Key Configuration Strategies:**
- Conservative (high quality, moderate speed): R=0.3, W=128, M=16
- Balanced: R=0.5, W=128, M=32 (default)
- Aggressive (high speed, slightly lower quality): R=0.7, W=64, M=48

**Evaluation Metrics to Track:**
- Average parallel tokens per generation step
- Total latency vs. baseline autoregressive models
- Accuracy on reasoning benchmarks (maintain >90% of baseline)
- Throughput (tokens/sec) on hardware target

**Common Issues:**
- Distribution disagreement too frequent: Decrease R towards 0.5
- Quality degradation: Increase W or decrease M for better context
- OOM errors: Reduce W or use smaller AR verifier
- Slow generation: Increase R or decrease lookahead M

## Reference

Demonstrated on Dream 7B Instruct (diffusion) with Qwen2.5 0.5B (auxiliary AR) on NVIDIA A5000. Achieves 5-8 parallel tokens with negligible performance impact. Evaluated on GSM8K, GPQA, MATH, HumanEval benchmarks. Shows competitive or superior throughput vs. autoregressive Qwen 7B.
