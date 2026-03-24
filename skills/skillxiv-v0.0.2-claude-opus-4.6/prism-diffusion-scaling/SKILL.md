---
name: prism-diffusion-scaling
title: "Prism: Efficient Test-Time Scaling via Hierarchical Search for Diffusion Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.01842"
keywords: [Test-Time Scaling, Discrete Diffusion, Inference Optimization, Adaptive Search, Self-Verification]
description: "Scale inference efficiency for discrete diffusion language models through hierarchical trajectory search with adaptive pruning and self-verified feedback. Achieve 3-4× speedup versus best-of-N with equal quality."
---

# Prism: Efficient Test-Time Scaling for Discrete Diffusion Language Models

Most inference-scaling methods optimize for autoregressive decoding, but discrete diffusion models generate sequences through parallel iterative denoising—fundamentally different. Naive best-of-N approaches require O(NT) function evaluations (N trajectories, T denoising steps) which is computationally prohibitive. Prism introduces three innovations: hierarchical trajectory search that progressively prunes low-quality trajectories during mid-denoising, self-verified feedback reusing the model as verifier, and partial remasking preserving high-confidence tokens while exploring alternatives.

The key insight is that diffusion's bidirectional context enables effective mid-generation pruning unavailable to autoregressive models.

## Core Concept

Prism uses three complementary mechanisms:

1. **Hierarchical Trajectory Search (HTS)**: Divide inference into stages with geometric decay of active trajectories from N → K during early denoising when "logic skeletons" stabilize, then final refinement with K survivors

2. **Self-Verified Feedback (SVF)**: Reuse the dLLM itself as verifier through dedicated Yes/No prompts on intermediate completions, eliminating overhead of separate models

3. **Local Branching via Partial Remasking**: Preserve high-confidence tokens as fixed "logic skeleton" while selectively re-masking low-confidence positions, enabling diverse exploration within fixed budget

These reduce complexity to approximately O(N + KT), achieving significant speedups.

## Architecture Overview

- **Trajectory Initializer**: Launch N parallel trajectories
- **Denoising Step Executor**: Standard diffusion denoising step on all active trajectories
- **Confidence Scorer**: Estimate token confidence from model logits
- **Pruning Engine**: Geometric decay schedule reducing trajectories from N → K
- **Self-Verification Module**: Query dLLM for Yes/No feedback on intermediates
- **Partial Remasking**: Selectively remask low-confidence positions
- **Skeleton Preservation**: Keep high-confidence tokens fixed across variants

## Implementation

The method involves trajectory management, confidence scoring, and adaptive pruning.

Initialize and manage parallel diffusion trajectories:

```python
import torch
from typing import List, Dict

class TrajectoryBatch:
    """Manage multiple parallel diffusion trajectories."""

    def __init__(self, initial_noise, num_trajectories=8):
        self.trajectories = [initial_noise.clone() for _ in range(num_trajectories)]
        self.active_mask = torch.ones(num_trajectories, dtype=torch.bool)
        self.confidence_scores = []
        self.step_count = 0

    def update_active(self, new_active_indices):
        """Update which trajectories remain active."""
        self.active_mask = torch.zeros(len(self.trajectories), dtype=torch.bool)
        self.active_mask[new_active_indices] = True

        # Keep only active trajectories
        self.trajectories = [self.trajectories[i] for i in new_active_indices]

    def get_active_trajectories(self):
        """Return currently active trajectories."""
        return [traj for traj, active in zip(self.trajectories, self.active_mask) if active]

    def record_step(self, step_idx):
        """Record trajectory state at denoising step."""
        self.step_count = step_idx

batch = TrajectoryBatch(initial_noise, num_trajectories=8)
```

Implement hierarchical trajectory search with pruning schedule:

```python
class HierarchicalSearchScheduler:
    """Manage trajectory pruning schedule across denoising steps."""

    def __init__(self, num_trajectories=8, num_steps=50, final_survivors=2):
        self.num_trajectories = num_trajectories
        self.num_steps = num_steps
        self.final_survivors = final_survivors

        # Geometric decay schedule
        decay_rate = (final_survivors / num_trajectories) ** (1.0 / (num_steps * 0.7))
        self.decay_rate = decay_rate

    def get_num_active(self, step):
        """Get number of active trajectories at denoising step."""
        if step < self.num_steps * 0.3:
            # Stage I: Exploration - keep all trajectories
            return self.num_trajectories
        else:
            # Stage II/III: Progressive thinning
            current_active = int(
                self.num_trajectories * (self.decay_rate ** (step - self.num_steps * 0.3))
            )
            return max(self.final_survivors, current_active)

    def should_prune(self, step):
        """Check if this step should perform trajectory pruning."""
        if step < self.num_steps * 0.3:
            return False  # Don't prune early

        # Prune when number of active trajectories decreases
        current = self.get_num_active(step)
        prev = self.get_num_active(step - 1)
        return current < prev

scheduler = HierarchicalSearchScheduler(num_trajectories=8, num_steps=50)
```

Implement self-verified feedback using the model:

```python
class SelfVerifier:
    """Use language model for self-verification of intermediate completions."""

    def __init__(self, model, query):
        self.model = model
        self.query = query
        self.cache = {}

    def verify_intermediate(self, completion, step_idx):
        """Query model: is this on track to solve the problem?"""

        # Check cache first
        cache_key = (completion, step_idx)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Construct verification prompt
        verify_prompt = f"""
Original query: {self.query}

Current partial solution (step {step_idx}):
{completion}

Does this look like it's on track to correctly solve the query?
Answer yes or no only."""

        # Get model's confidence in yes/no
        yes_logits = self.model.get_token_logits(verify_prompt, "yes")
        no_logits = self.model.get_token_logits(verify_prompt, "no")

        confidence = torch.softmax(torch.tensor([yes_logits, no_logits]), dim=0)[0]

        self.cache[cache_key] = confidence.item()
        return confidence.item()

verifier = SelfVerifier(model, query)
```

Implement partial remasking for local branching:

```python
def compute_token_confidence(logits, temperature=1.0):
    """Estimate token confidence from model logits."""
    # Confidence = max probability
    probs = torch.softmax(logits / temperature, dim=-1)
    confidence, _ = torch.max(probs, dim=-1)
    return confidence

def apply_partial_remasking(latent, confidence_scores, mask_ratio=0.3):
    """Remask low-confidence positions, keep high-confidence tokens fixed."""

    # Identify low-confidence positions
    num_to_mask = int(latent.shape[-1] * mask_ratio)
    low_conf_indices = torch.argsort(confidence_scores)[:num_to_mask]

    # Create new noise for low-confidence positions
    noise = torch.randn_like(latent)

    # Preserve high-confidence positions
    mask = torch.ones_like(latent)
    mask[low_conf_indices] = 0  # Will be remasked

    # Apply partial remask
    remasked_latent = latent * mask + noise * (1 - mask)

    return remasked_latent

def hierarchical_search_inference(model, query, num_initial=8, num_final=2):
    """Run Prism hierarchical search inference."""

    scheduler = HierarchicalSearchScheduler(
        num_trajectories=num_initial,
        final_survivors=num_final
    )
    verifier = SelfVerifier(model, query)

    # Initialize trajectories
    batch = TrajectoryBatch(
        torch.randn(1, seq_len, hidden_dim),
        num_trajectories=num_initial
    )

    # Denoising loop
    for step in range(num_steps):
        # Denoise all active trajectories
        for i, traj in enumerate(batch.get_active_trajectories()):
            traj = model.denoise_step(traj, step, query)
            batch.trajectories[i] = traj

        # Prune trajectories if scheduled
        if scheduler.should_prune(step):
            num_active = scheduler.get_num_active(step)

            # Score trajectories with self-verification
            scores = []
            for traj in batch.get_active_trajectories():
                partial_completion = model.decode_partial(traj)
                score = verifier.verify_intermediate(partial_completion, step)
                scores.append(score)

            # Keep top-scoring trajectories
            top_indices = torch.argsort(torch.tensor(scores), descending=True)[:num_active]
            batch.update_active(top_indices.tolist())

        # Optional: partial remasking for exploration (Stage III)
        if step > num_steps * 0.6:
            for i, traj in enumerate(batch.get_active_trajectories()):
                confidence = compute_token_confidence(
                    model.get_logits(traj, query)
                )
                traj = apply_partial_remasking(traj, confidence, mask_ratio=0.2)
                batch.trajectories[i] = traj

    # Decode final trajectories and select best
    final_completions = []
    for traj in batch.get_active_trajectories():
        completion = model.decode(traj, query)
        final_completions.append(completion)

    return final_completions

results = hierarchical_search_inference(model, query)
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|-----------------|-------|
| Initial Trajectories (N) | 8-16 | Exploration breadth |
| Final Survivors (K) | 2-4 | Refinement depth |
| Pruning Start | 30% through steps | When structure stabilizes |
| Decay Schedule | Geometric (smooth pruning) | Avoid sharp transitions |
| Verification Overhead | <10% of total | Keep verifier efficient |
| Mask Ratio | 20-30% low-confidence | Preserve skeleton |

**When to use**: For discrete diffusion language models needing test-time scaling. When you need 3-4× speedup without quality loss. For generation quality not achievable in single pass.

**When NOT to use**: For autoregressive models (different decoding paradigm). When compute budget is unconstrained.

**Common pitfalls**:
- Pruning too aggressive causes mode collapse—validate on dev set
- Verification overhead can dominate compute—use efficient Yes/No queries
- Self-verification can become overconfident—monitor verification accuracy on known cases
- Geometric decay schedule requires tuning—experiment with decay_rate parameter
- Partial remasking can cause incoherence—keep skeleton strong (70%+ preserved)

## Reference

Prism: Efficient Test-Time Scaling via Hierarchical Search for Diffusion Language Models
https://arxiv.org/abs/2602.01842
