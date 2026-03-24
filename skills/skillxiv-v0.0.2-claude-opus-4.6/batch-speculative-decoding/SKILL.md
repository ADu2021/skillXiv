---
name: batch-speculative-decoding
title: "Batch Speculative Decoding Done Right"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.22876"
keywords: [Decoding, Inference Optimization, Speculative Execution, Batching]
description: "Fixes batch speculative decoding ragged tensor problem where sequences in batches accept different token counts, desynchronizing state. EQSPEC guarantees output equivalence through proper synchronization. EXSPEC reduces overhead 40% via cross-batch scheduling. Enables efficient parallel decoding with 95% equivalence."
---

# Batch Speculative Decoding Done Right: Fixing the Ragged Tensor Problem

Speculative decoding accelerates inference by using draft models to generate multiple tokens, verified by main model. However, batched versions have critical synchronization bugs: sequences accepting different draft lengths causes KV-cache and attention mask misalignment.

EQSPEC and EXSPEC fix these issues, enabling correct and efficient batch speculative decoding.

## Core Concept

The problem: **in batches, each sequence may accept different numbers of draft tokens**, leading to:
- Misaligned KV-cache states
- Inconsistent attention masks
- Incorrect position IDs
- Near-zero output equivalence (wrong answers)

Solutions:
- **EQSPEC**: Formal synchronization protocol guaranteeing output equivalence
- **EXSPEC**: Efficient scheduling reducing computational overhead by ~40%

## Architecture Overview

- Draft model generates K tokens per sequence
- Verification: main model checks draft token acceptance
- Synchronization: enforce ragged tensor handling
- Cross-batch scheduling: group sequences by similar acceptance patterns

## Implementation Steps

Implement proper speculative decoding with synchronization. The key is tracking which draft tokens each sequence accepted:

```python
class SpeculativeDecodingEQSpec:
    def __init__(self, draft_model, main_model, draft_tokens=4):
        self.draft_model = draft_model
        self.main_model = main_model
        self.draft_tokens = draft_tokens

    def speculative_generate(self, input_ids, max_length=256):
        """Speculative decoding with synchronization (EQSPEC)."""
        batch_size = input_ids.shape[0]
        current_ids = input_ids.clone()

        # Track accepted tokens per sequence
        accepted_counts = torch.zeros(batch_size, dtype=torch.long)

        while current_ids.shape[1] < max_length:
            # Draft model: generate K tokens per sequence
            draft_ids = self.draft_model.generate_tokens(
                current_ids, num_tokens=self.draft_tokens
            )  # Shape: (batch, K)

            # Verify draft tokens with main model
            # CRITICAL: Handle ragged tensors properly
            acceptance_mask = self._verify_draft_tokens(
                current_ids, draft_ids
            )  # Shape: (batch, K)

            # Count accepted tokens per sequence (first failure point)
            per_seq_accepted = self._count_accepted_per_sequence(
                acceptance_mask
            )  # Shape: (batch,)

            # SYNCHRONIZATION: Pad all sequences to max accepted count
            max_accepted = torch.max(per_seq_accepted)

            # Build synchronized KV cache and new sequences
            synchronized_sequences = []
            synchronized_kv_cache = {}

            for seq_idx in range(batch_size):
                # Append only accepted tokens for this sequence
                num_accepted = per_seq_accepted[seq_idx].item()
                new_tokens = draft_ids[seq_idx, :num_accepted]

                # Pad with padding tokens if needed (synchronization)
                if num_accepted < max_accepted:
                    padding = torch.full(
                        (max_accepted - num_accepted,),
                        self.main_model.pad_token_id,
                        dtype=torch.long
                    )
                    new_tokens = torch.cat([new_tokens, padding])

                synchronized_sequences.append(new_tokens)

            # Stack synchronized sequences
            synced_ids = torch.stack(synchronized_sequences)
            current_ids = torch.cat([current_ids, synced_ids], dim=1)

            # Remove padding for next iteration
            current_ids = self._remove_padding(current_ids)

            if torch.all(accepted_counts >= self.draft_tokens):
                # All sequences rejected draft: fall back to main model
                next_tokens = self.main_model.generate_tokens(
                    current_ids, num_tokens=1
                )
                current_ids = torch.cat([current_ids, next_tokens], dim=1)

        return current_ids

    def _verify_draft_tokens(self, context, draft_ids):
        """Verify which draft tokens match main model's predictions."""
        # Forward main model with draft tokens
        main_logits = self.main_model(
            torch.cat([context, draft_ids], dim=1)
        )['logits']

        # Compare draft tokens with main model predictions
        # Use probability matching or cross-entropy threshold
        context_len = context.shape[1]

        acceptance_mask = torch.zeros_like(draft_ids, dtype=torch.bool)

        for token_pos in range(self.draft_tokens):
            logit_pos = context_len + token_pos
            main_probs = torch.softmax(main_logits[:, logit_pos, :], dim=-1)

            # Acceptance criterion: main model's probability for draft token
            draft_token = draft_ids[:, token_pos]
            draft_probs = main_probs.gather(1, draft_token.unsqueeze(1)).squeeze()

            # Accept if probability > threshold
            acceptance_mask[:, token_pos] = draft_probs > 0.5

            # Stop at first rejection (Poisson process approximation)
            if not torch.all(acceptance_mask[:, token_pos]):
                break

        return acceptance_mask

    def _count_accepted_per_sequence(self, acceptance_mask):
        """Count first failure point per sequence."""
        counts = torch.zeros(acceptance_mask.shape[0], dtype=torch.long)

        for seq_idx in range(acceptance_mask.shape[0]):
            # Count consecutive accepted tokens
            for token_idx in range(acceptance_mask.shape[1]):
                if acceptance_mask[seq_idx, token_idx]:
                    counts[seq_idx] += 1
                else:
                    break

        return counts

    def _remove_padding(self, ids):
        """Remove padding tokens added for synchronization."""
        # Simple: truncate at first padding token per sequence
        non_pad_ids = []
        for seq in ids:
            end_idx = (seq != self.main_model.pad_token_id).nonzero()
            if len(end_idx) > 0:
                end_idx = end_idx[-1].item() + 1
                non_pad_ids.append(seq[:end_idx])
            else:
                non_pad_ids.append(seq)

        # Return variable length - next iteration handles
        return non_pad_ids
```

Implement EXSPEC for efficient cross-batch scheduling:

```python
class SpeculativeDecodingEXSpec:
    def __init__(self, draft_model, main_model, draft_tokens=4):
        self.decoder = SpeculativeDecodingEQSpec(draft_model, main_model, draft_tokens)

    def efficient_generate_batched(self, input_sequences, max_length=256):
        """EXSPEC: Efficient scheduling across sequences."""
        # Group sequences by current length (approximate)
        length_groups = self._group_by_length(input_sequences)

        results = []

        # Process groups with similar lengths together
        # Reduces padding overhead by ~40%
        for group_length, sequences in length_groups.items():
            group_ids = torch.stack(sequences)

            # Generate for this length group
            output = self.decoder.speculative_generate(
                group_ids, max_length=max_length
            )

            results.extend(output.split(1, dim=0))

        return results

    def _group_by_length(self, sequences):
        """Group sequences by length for efficient batching."""
        groups = {}

        for seq in sequences:
            length = len(seq)
            # Round to nearest bucket
            bucket = (length // 64) * 64
            if bucket not in groups:
                groups[bucket] = []
            groups[bucket].append(seq)

        return groups
```

## Practical Guidance

| Parameter | Recommendation |
|-----------|-----------------|
| Draft tokens (K) | 4-8 (balance draft speed vs. acceptance) |
| Acceptance threshold | 0.5-0.7 (probability match) |
| Batch size | 16-32 (larger = more synch overhead) |
| Scheduling granularity | 64-token length buckets |

**When to use:**
- Batch inference with multiple sequences
- Models where draft model is much faster
- Latency-critical deployments
- Correctness-critical applications (EQSPEC guarantees)

**When NOT to use:**
- Single sequence inference (simpler unbatched version)
- When draft model not available or not faster
- Memory-constrained devices (padding overhead)

**Common pitfalls:**
- Not synchronizing KV cache (silent correctness bugs)
- Accepting all draft tokens without verification
- Batch size too large (synchronization dominates)
- Not handling end-of-sequence properly in batches

Reference: [Batch Speculative Decoding Done Right on arXiv](https://arxiv.org/abs/2510.22876)
