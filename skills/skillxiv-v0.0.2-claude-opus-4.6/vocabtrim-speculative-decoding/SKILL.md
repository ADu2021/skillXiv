---
name: vocabtrim-speculative-decoding
title: "VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.22694"
keywords: [Speculative Decoding, Vocabulary Pruning, Inference Optimization, Memory Efficiency, Token Prediction]
description: "Accelerate speculative decoding by pruning drafter vocabulary to high-frequency tokens. Achieves 16% speedup in memory-bound settings by eliminating unused vocabulary entries without retraining."
---

# VocabTrim: Memory-Efficient Vocabulary Pruning for Speculative Decoding

Speculative decoding uses a small drafter model to propose multiple tokens per inference step, which the target verifier model accepts or rejects. This approach can 2-3× speedup inference when the drafter is fast. However, the drafter's language modeling head (the final layer outputting logits over all vocabulary tokens) becomes a memory bottleneck. For Llama-3 with 128K vocabulary tokens, computing logits over all 128K tokens at every step wastes memory and computation even though the drafter only samples from a tiny subset of frequently-occurring tokens.

VocabTrim solves this by reconstructing the drafter's vocabulary to contain only the high-frequency tokens it actually samples during inference. The insight is that drafters are biased toward "easy-to-predict" tokens (common words, punctuation) and rarely sample rare tokens. By trimming the vocabulary to the most frequent 25-50K tokens, you eliminate 60-75% of the LM head computation with negligible impact on acceptance rates.

## Core Concept

VocabTrim works on a simple principle: **replace the drafter's full vocabulary with a smaller one containing only tokens it frequently generates**. The key insights are:

1. Drafters naturally focus on high-frequency, predictable tokens because they're trying to be fast and accurate
2. Target models have access to the full vocabulary during verification, so drafter coverage doesn't need to be complete
3. Token remapping can translate drafter outputs back to full vocabulary indices during generation
4. The method is training-free: measure token frequencies from any calibration data and reconstruct the LM head

The result is a smaller drafter that produces fewer logits per step (reduced memory transfers) and faster matrix multiplication (fewer output dimensions), with only minimal increase in rejection rate (1-3%).

## Architecture Overview

VocabTrim modifies only the drafter's language modeling head:

- **Token Frequency Measurement**: Run the drafter on calibration data and count which tokens are sampled during generation
- **Vocabulary Selection**: Extract the top-K most frequently sampled tokens across the calibration set
- **LM Head Reconstruction**: Create a new smaller LM head matrix containing only weights for selected token indices
- **Token Remapping**: During inference, map drafter predictions (indices in small vocabulary) back to full vocabulary indices for the verifier
- **Verifier Unchanged**: The target model receives the full vocabulary indices and operates without modification

## Implementation

**Step 1: Measure token frequencies on calibration data**

Sample from the drafter on representative data and count which tokens appear most frequently in the output distribution.

```python
def measure_drafter_token_frequencies(drafter, tokenizer, calibration_data,
                                     num_samples=100000, sample_length=256):
    """
    Run the drafter on calibration data and measure which tokens
    are sampled most frequently. This identifies which tokens matter for inference speed.
    """
    token_counts = {}
    total_tokens_sampled = 0

    for batch_idx, batch in enumerate(calibration_data):
        input_ids = tokenizer.encode(batch, return_tensors='pt').to(device)

        # Generate from drafter, collecting all sampled tokens
        with torch.no_grad():
            for step in range(sample_length):
                outputs = drafter(input_ids)
                logits = outputs.logits[:, -1, :]  # Last position logits

                # Sample from the distribution (how the drafter would generate in practice)
                probs = torch.softmax(logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1)

                # Count token occurrences
                for token_id in sampled_tokens.flatten().tolist():
                    token_counts[token_id] = token_counts.get(token_id, 0) + 1
                    total_tokens_sampled += 1

                input_ids = torch.cat([input_ids, sampled_tokens], dim=1)

        if (batch_idx + 1) * sample_length >= num_samples:
            break

    # Sort by frequency
    sorted_tokens = sorted(token_counts.items(),
                          key=lambda x: x[1], reverse=True)

    return dict(sorted_tokens)
```

**Step 2: Select vocabulary and create remapping**

Choose the top-K tokens and create an index mapping from small vocabulary to full vocabulary.

```python
def create_vocab_mapping(token_frequencies, target_vocab_size=32000):
    """
    Select the most frequent tokens and create a mapping.
    This tells us which full vocabulary indices to keep.
    """
    # Select top-K tokens by frequency
    top_tokens = sorted(token_frequencies.items(),
                       key=lambda x: x[1], reverse=True)[:target_vocab_size]

    # Create two mappings:
    # 1. full_to_trimmed: maps full vocabulary indices to trimmed indices
    # 2. trimmed_to_full: maps trimmed indices back to full vocabulary
    full_to_trimmed = {}
    trimmed_to_full = {}

    for trimmed_idx, (full_token_id, count) in enumerate(top_tokens):
        full_to_trimmed[full_token_id] = trimmed_idx
        trimmed_to_full[trimmed_idx] = full_token_id

    # Calculate coverage: what percentage of real samples are covered
    total_count = sum(token_frequencies.values())
    selected_count = sum(count for _, count in top_tokens)
    coverage = selected_count / total_count

    return {
        'full_to_trimmed': full_to_trimmed,
        'trimmed_to_full': trimmed_to_full,
        'vocab_size': target_vocab_size,
        'coverage': coverage
    }
```

**Step 3: Reconstruct the drafter's LM head**

Create a new, smaller language modeling head by extracting only the necessary weight rows.

```python
def reconstruct_lm_head(drafter, vocab_mapping):
    """
    Extract the drafter's LM head weights for selected tokens only.
    This creates a new head with fewer output dimensions.
    """
    original_head = drafter.lm_head
    original_weight = original_head.weight  # Shape: [vocab_size, hidden_dim]

    # Select only rows corresponding to kept tokens
    selected_indices = torch.tensor(
        [vocab_mapping['trimmed_to_full'][i]
         for i in range(vocab_mapping['vocab_size'])],
        device=original_weight.device
    )

    # Extract selected rows
    trimmed_weight = original_weight[selected_indices, :]

    # Create new head with smaller output vocabulary
    trimmed_head = torch.nn.Linear(
        original_head.in_features,
        vocab_mapping['vocab_size'],
        bias=(original_head.bias is not None)
    )

    # Copy weights
    with torch.no_grad():
        trimmed_head.weight.copy_(trimmed_weight)
        if original_head.bias is not None:
            original_bias = original_head.bias
            selected_bias = original_bias[selected_indices]
            trimmed_head.bias.copy_(selected_bias)

    return trimmed_head
```

**Step 4: Implement token remapping in the speculative decoding loop**

During generation, convert drafter indices (from trimmed vocabulary) back to full vocabulary indices.

```python
def speculative_decode_with_trimmed_vocab(target_model, drafter,
                                         input_ids, vocab_mapping,
                                         num_steps=256, draft_length=4):
    """
    Run speculative decoding with a trimmed-vocabulary drafter.
    The drafter outputs indices in the small vocabulary,
    which we remap to the full vocabulary before verification.
    """
    full_to_trimmed = vocab_mapping['full_to_trimmed']
    trimmed_to_full = vocab_mapping['trimmed_to_full']

    current_ids = input_ids

    for step in range(num_steps):
        # Drafter generates draft tokens
        draft_tokens = []
        drafter_ids = current_ids

        for draft_step in range(draft_length):
            with torch.no_grad():
                # Forward pass through drafter with TRIMMED vocabulary
                drafter_outputs = drafter(drafter_ids)
                drafter_logits = drafter_outputs.logits[:, -1, :]

                # Sample from trimmed vocabulary
                trimmed_probs = torch.softmax(drafter_logits, dim=-1)
                trimmed_sample = torch.multinomial(trimmed_probs, num_samples=1)

                # REMAP: convert trimmed index to full vocabulary index
                full_sample = torch.tensor(
                    [[trimmed_to_full[int(trimmed_sample[0, 0])]]],
                    device=trimmed_sample.device
                )

                draft_tokens.append(full_sample)
                drafter_ids = torch.cat([drafter_ids, full_sample], dim=1)

        # Concatenate draft tokens for verification
        draft_ids = torch.cat(draft_tokens, dim=1)
        candidate_ids = torch.cat([current_ids, draft_ids], dim=1)

        # Target model verifies candidate tokens (with FULL vocabulary)
        with torch.no_grad():
            target_outputs = target_model(candidate_ids)
            target_logits = target_outputs.logits

        # Verification: accept tokens while probabilities remain high
        for accept_idx in range(draft_length):
            position = current_ids.shape[1] + accept_idx
            candidate_token = draft_ids[0, accept_idx].item()

            # Get target model's probability for this token
            position_logits = target_logits[0, position - 1, :]
            target_probs = torch.softmax(position_logits, dim=-1)
            target_prob = target_probs[candidate_token].item()

            # Accept or reject
            if torch.rand(1).item() < target_prob:
                # Accept: token is added permanently
                current_ids = torch.cat([current_ids, draft_ids[:, accept_idx:accept_idx+1]], dim=1)
            else:
                # Reject: resample from target model
                resampled = torch.multinomial(target_probs, num_samples=1)
                current_ids = torch.cat([current_ids, resampled], dim=1)
                break

    return current_ids
```

**Step 5: Evaluate speedup and coverage trade-off**

Measure the actual inference speedup and how often the drafter misses tokens outside its vocabulary.

```python
def evaluate_vocabtrim_performance(target_model, drafter, vocab_mapping,
                                  eval_prompts, vocab_mapping_baseline):
    """
    Compare trimmed drafter (VocabTrim) against baseline speculative decoding.
    Key metrics: speedup, acceptance rate, coverage.
    """
    results = {
        'trimmed': [],
        'baseline': []
    }

    for prompt in eval_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Baseline: full vocabulary drafter
        start_time = time.time()
        output_baseline = speculative_decode_baseline(target_model, drafter,
                                                     input_ids, max_length=256)
        time_baseline = time.time() - start_time

        # Trimmed: VocabTrim drafter
        start_time = time.time()
        output_trimmed = speculative_decode_with_trimmed_vocab(
            target_model, drafter, input_ids, vocab_mapping, num_steps=256
        )
        time_trimmed = time.time() - start_time

        # Compute acceptance rate
        # (tokens where drafter was right and target accepted them)
        acceptance_baseline = count_accepted_tokens(output_baseline)
        acceptance_trimmed = count_accepted_tokens(output_trimmed)

        results['baseline'].append({
            'time': time_baseline,
            'acceptance': acceptance_baseline
        })
        results['trimmed'].append({
            'time': time_trimmed,
            'acceptance': acceptance_trimmed
        })

    speedup = np.mean([r['time'] for r in results['baseline']]) / \
              np.mean([r['time'] for r in results['trimmed']])

    acceptance_degradation = (np.mean([r['acceptance'] for r in results['baseline']]) -
                             np.mean([r['acceptance'] for r in results['trimmed']])) / \
                            np.mean([r['acceptance'] for r in results['baseline']])

    return {
        'speedup': speedup,
        'acceptance_degradation': acceptance_degradation,
        'vocab_coverage': vocab_mapping['coverage']
    }
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| Target vocab size | 25K-50K | Higher = more speedup but diminishing returns |
| Coverage threshold | 85-95% | Ensure you're covering most sampled tokens |
| Calibration data size | 10K-50K examples | Measure frequencies on representative data |
| Acceptance rate tolerance | 97-99% | Can afford 1-3% degradation for 16% speedup |
| Remapping overhead | <1% | Negligible compared to inference speedup |

**When to use VocabTrim:**
- You're using speculative decoding with draft-verify architecture
- Your drafter has a large vocabulary (Llama-3 with 128K tokens)
- You're memory-bound (not compute-bound) during inference
- You need 10-20% inference speedup without retraining

**When NOT to use VocabTrim:**
- Your drafter already has a small vocabulary (less than 50K)
- You're compute-bound (vocabulary pruning won't help)
- You need zero acceptance rate degradation
- Your workload requires coverage of very rare tokens

**Common pitfalls:**
- **Selecting wrong calibration data**: Use in-domain data (same domain as inference). Measuring frequencies on web text won't work for code generation.
- **Vocabulary too small**: If you select only 10K tokens, acceptance rate drops sharply. Usually 30-50K is the sweet spot.
- **Forgetting rare tokens**: Some tasks (like code generation) need rare tokens. Gradually increase vocabulary size if acceptance degrades.
- **Remapping overhead**: Use a hash table or tensor lookup for fast remapping; don't iterate through the mapping.

## Reference

VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs
https://arxiv.org/abs/2506.22694
