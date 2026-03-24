---
name: unmasking-diffusion-policies
title: "Learning Unmasking Policies for Diffusion Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.09106
keywords: [diffusion language models, unmasking policies, reinforcement learning, semi-autoregressive, block-based generation]
description: "Learn which tokens to unmask during diffusion sampling via reinforcement learning instead of heuristics. Policies eliminate manual tuning and scale across block sizes—crucial when semi-autoregressive generation needs dynamic, learned unmasking strategies."
---

## Overview

Rather than relying on manual confidence thresholds for token unmasking, this approach casts diffusion language model sampling as an MDP where a lightweight transformer policy learns optimal unmasking decisions based on token confidences.

## When to Use

- Diffusion language model inference optimization
- Semi-autoregressive generation with block-based sampling
- Need for dynamic unmasking beyond confidence thresholds
- Scaling across different block sizes
- Learning unmasking strategies instead of manual tuning

## When NOT to Use

- Scenarios where heuristic thresholds work adequately
- Autoregressive decoding without masking
- Tasks not benefiting from RL optimization

## Core Technique

RL-based policy for unmasking token selection:

```python
# Learned unmasking policies for dLLMs
class UnmaskingPolicy:
    def __init__(self):
        self.policy = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)  # Unmasking logits
        )

    def learn_unmasking_policy(self, dllm, dataset):
        """Train policy via RL on dLLM sampling."""
        for batch in dataset:
            # Generate with dLLM
            confidences = dllm.get_token_confidences(batch)

            # Policy decides which tokens to unmask
            unmasking_logits = self.policy(confidences)
            unmasking_probs = torch.softmax(unmasking_logits, dim=-1)

            # Sample unmasking decisions
            decisions = torch.multinomial(unmasking_probs, num_samples=1)

            # Apply unmasking
            sampled_tokens = dllm.sample_with_unmasking(decisions)

            # Reward: similarity to full diffusion output
            full_output = dllm.full_diffusion_sampling(batch)
            reward = compute_similarity(sampled_tokens, full_output)

            # Policy gradient
            loss = -reward * torch.log(unmasking_probs[decisions])
            loss.backward()

        self.optimizer.step()

    def unmask_during_inference(self, confidences):
        """Apply learned policy for token selection."""
        unmasking_logits = self.policy(confidences)
        decisions = torch.argmax(unmasking_logits, dim=-1)
        return decisions
```

## Key Results

- Learned policies match/exceed heuristic thresholds
- Scalability across block sizes
- Eliminates hyperparameter tuning

## References

- Original paper: https://arxiv.org/abs/2512.09106
- Focus: Optimal unmasking in diffusion LLMs
- Domain: Diffusion models, sampling strategies
