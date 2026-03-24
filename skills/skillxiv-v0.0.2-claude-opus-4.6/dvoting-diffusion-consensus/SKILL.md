---
name: dvoting-diffusion-consensus
title: "dVoting: Fast Voting for dLLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.12153"
keywords: [Diffusion Language Models, Test-Time Scaling, Voting, Inference Optimization, Ensemble Methods]
description: "Accelerate test-time scaling for diffusion language models by identifying inconsistent tokens, selectively remask and regenerate only uncertain tokens, and aggregate across samples via voting. Achieve 5.5-22× speedup over standard iterative sampling with 6-8% accuracy gains on reasoning tasks."
---

# dVoting: Fast Voting for dLLMs

## Problem Context

Diffusion language models can generate tokens in any order and refine them through masking and redenoising. Standard test-time scaling generates multiple samples and averages predictions. However, most tokens remain consistent across samples—only a small subset vary. dVoting exploits this by identifying variable tokens, regenerating only those selectively, and using voting to aggregate.

## Core Concept

dVoting operates in three phases: (1) identify tokens with cross-sample disagreement, (2) selectively remask inconsistent tokens, (3) regenerate and aggregate via voting. This reduces computation by focusing computation on the small disagreement set.

## Architecture Overview

- **Consistency detection**: Compare tokens across multiple samples
- **Uncertainty identification**: Flag tokens with disagreement
- **Selective remask**: Only remask uncertain tokens
- **Regeneration**: Run denoising only on uncertainty regions
- **Voting aggregation**: Combine candidate answers via vote

## Implementation

### Step 1: Generate samples and identify inconsistencies

```python
import torch
from typing import List, Dict, Tuple
from collections import Counter

class DiffusionConsistencyAnalyzer:
    """Identify tokens showing cross-sample disagreement."""

    def __init__(self, agreement_threshold: float = 0.8):
        self.agreement_threshold = agreement_threshold

    def detect_inconsistent_tokens(
        self,
        samples: List[torch.Tensor],  # [num_samples, seq_len]
        seq_len: int
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Identify tokens with low agreement across samples.

        Args:
            samples: List of generated token sequences
            seq_len: Sequence length

        Returns:
            agreement_scores: Agreement ratio per token [seq_len]
            inconsistent_positions: Indices of inconsistent tokens
        """
        num_samples = len(samples)
        agreement_scores = torch.ones(seq_len)
        inconsistent_positions = []

        for pos in range(seq_len):
            tokens_at_pos = [sample[pos].item() for sample in samples]
            token_counts = Counter(tokens_at_pos)
            max_count = max(token_counts.values())
            agreement = max_count / num_samples

            agreement_scores[pos] = agreement

            if agreement < self.agreement_threshold:
                inconsistent_positions.append(pos)

        return agreement_scores, inconsistent_positions
```

### Step 2: Selective remask and regenerate

```python
class SelectiveRemaskingStrategy:
    """Remask only inconsistent tokens for efficient regeneration."""

    def __init__(self, model, num_denoising_steps: int = 10):
        self.model = model
        self.num_denoising_steps = num_denoising_steps

    def remask_inconsistent_tokens(
        self,
        sample: torch.Tensor,      # [seq_len]
        inconsistent_pos: List[int]
    ) -> torch.Tensor:
        """Create mask for inconsistent tokens."""
        masked_sample = sample.clone()
        for pos in inconsistent_pos:
            masked_sample[pos] = self.model.mask_token_id
        return masked_sample

    def regenerate_masked_region(
        self,
        masked_sample: torch.Tensor,
        context: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Run diffusion denoising on masked region.

        Leverages dLLM's ability to refine specific positions.
        """
        refined = self.model.denoise(
            masked_sample,
            num_steps=self.num_denoising_steps,
            context=context
        )
        return refined

    def iterative_refinement(
        self,
        sample: torch.Tensor,
        inconsistent_pos: List[int],
        num_iterations: int = 2
    ) -> torch.Tensor:
        """Multiple passes of selective remask-regenerate."""
        current = sample.clone()

        for iteration in range(num_iterations):
            masked = self.remask_inconsistent_tokens(current, inconsistent_pos)
            current = self.regenerate_masked_region(masked)

        return current
```

### Step 3: Aggregate predictions via voting

```python
class TokenVotingAggregator:
    """Aggregate multiple samples via majority voting."""

    @staticmethod
    def majority_vote(
        samples: List[torch.Tensor],  # [num_samples, seq_len]
        positions: List[int] = None
    ) -> torch.Tensor:
        """
        Compute majority vote for each position.

        Args:
            samples: Generated samples
            positions: Positions to vote on (default: all)

        Returns:
            voted_sequence: Final tokens from voting [seq_len]
        """
        num_samples = len(samples)
        seq_len = samples[0].shape[0]

        if positions is None:
            positions = list(range(seq_len))

        voted_sequence = samples[0].clone()

        for pos in positions:
            tokens_at_pos = [sample[pos].item() for sample in samples]
            token_counts = Counter(tokens_at_pos)
            majority_token = token_counts.most_common(1)[0][0]
            voted_sequence[pos] = majority_token

        return voted_sequence

    @staticmethod
    def confidence_weighted_voting(
        samples: List[torch.Tensor],
        log_probs: List[torch.Tensor],  # [num_samples, seq_len]
        positions: List[int] = None
    ) -> torch.Tensor:
        """
        Weight voting by model confidence scores.

        Higher likelihood tokens weighted more heavily.
        """
        num_samples = len(samples)
        seq_len = samples[0].shape[0]

        if positions is None:
            positions = list(range(seq_len))

        voted_sequence = samples[0].clone()

        for pos in positions:
            # Weighted vote by log probability
            vote_weights = {}
            for sample_idx, sample in enumerate(samples):
                token = sample[pos].item()
                weight = log_probs[sample_idx][pos].exp().item()

                if token not in vote_weights:
                    vote_weights[token] = 0.0
                vote_weights[token] += weight

            best_token = max(vote_weights.items(), key=lambda x: x[1])[0]
            voted_sequence[pos] = best_token

        return voted_sequence
```

### Step 4: Full dVoting pipeline

```python
class DVotingPipeline:
    """Complete dVoting strategy for fast test-time scaling."""

    def __init__(
        self,
        model,
        agreement_threshold: float = 0.8,
        num_regeneration_steps: int = 10,
        num_refinement_passes: int = 1
    ):
        self.model = model
        self.analyzer = DiffusionConsistencyAnalyzer(agreement_threshold)
        self.remask_strategy = SelectiveRemaskingStrategy(model, num_regeneration_steps)
        self.aggregator = TokenVotingAggregator()
        self.num_refinement_passes = num_refinement_passes

    def dvoting_forward(
        self,
        prompt: str,
        num_samples: int = 8,
        compute_log_probs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Execute dVoting for fast ensemble inference.

        Args:
            prompt: Input prompt
            num_samples: Initial samples to generate
            compute_log_probs: Return confidence scores

        Returns:
            final_sequence: Voted final prediction
            inconsistent_count: Number of inconsistent tokens
            speedup_factor: Estimated speedup vs. standard voting
        """
        # Step 1: Generate initial samples (full diffusion)
        samples = []
        log_probs_list = []

        for _ in range(num_samples):
            sample, log_probs = self.model.generate_with_logprobs(
                prompt, num_steps=50
            )
            samples.append(sample)
            log_probs_list.append(log_probs)

        # Step 2: Identify inconsistencies
        agreement_scores, inconsistent_pos = self.analyzer.detect_inconsistent_tokens(
            samples, samples[0].shape[0]
        )

        inconsistent_count = len(inconsistent_pos)
        total_tokens = samples[0].shape[0]

        # Step 3: Selective regeneration
        regenerated_samples = []
        for sample in samples:
            refined = self.remask_strategy.iterative_refinement(
                sample, inconsistent_pos,
                num_iterations=self.num_refinement_passes
            )
            regenerated_samples.append(refined)

        # Step 4: Vote on inconsistent positions
        if compute_log_probs:
            final_sequence = self.aggregator.confidence_weighted_voting(
                regenerated_samples, log_probs_list, inconsistent_pos
            )
        else:
            # On consistent positions, use original agreement
            final_sequence = regenerated_samples[0].clone()
            final_sequence = self.aggregator.majority_vote(
                regenerated_samples, inconsistent_pos
            )

        # Estimate speedup
        regeneration_fraction = inconsistent_count / total_tokens
        speedup_vs_full = 1.0 / max(regeneration_fraction, 0.1)

        return {
            'final_sequence': final_sequence,
            'agreement_scores': agreement_scores,
            'inconsistent_positions': inconsistent_pos,
            'inconsistent_count': inconsistent_count,
            'speedup_factor': speedup_vs_full
        }
```

### Step 5: Training and evaluation

```python
def evaluate_dvoting_speedup(
    model,
    test_prompts: List[str],
    verifier,
    num_samples: int = 8,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark dVoting speedup and accuracy.
    """
    dvoting = DVotingPipeline(model, num_samples)

    total_correct = 0
    total_speedup = 0.0
    num_tests = 0

    for prompt in test_prompts:
        result = dvoting.dvoting_forward(prompt, num_samples=num_samples)

        # Check correctness
        prediction = model.decode(result['final_sequence'])
        is_correct = verifier(prediction)
        total_correct += is_correct

        # Track speedup
        total_speedup += result['speedup_factor']
        num_tests += 1

    return {
        'accuracy': total_correct / num_tests,
        'avg_speedup': total_speedup / num_tests
    }
```

## Practical Guidance

**When to use**: Diffusion language models needing test-time scaling; ensemble prediction with limited compute budget

**Hyperparameters**:
- **agreement_threshold**: 0.75-0.85 (higher = more selective)
- **num_regeneration_steps**: 5-20 (tradeoff: quality vs speed)
- **num_samples**: 4-8 (initial sample count)
- **num_refinement_passes**: 1-2 (iterations on inconsistent tokens)

**Key advantages**:
- 5-22× speedup vs. standard iterative sampling
- 6-8% accuracy improvements via voting
- Focuses computation on uncertainty
- Works with any diffusion LM

**Common pitfalls**:
- agreement_threshold too high → misses real disagreements
- Too few refinement passes → suboptimal regeneration
- Not weighting by confidence → loses signal quality

**Scaling**: Speedup scales with inconsistency sparsity; best on diverse generation tasks.

## Reference

Paper: https://arxiv.org/abs/2602.12153
Related work: Diffusion models, test-time scaling, ensemble methods, majority voting
Benchmarks: GSM8K, MATH, reasoning tasks
