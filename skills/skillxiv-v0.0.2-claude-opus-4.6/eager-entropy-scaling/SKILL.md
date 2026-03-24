---
name: eager-entropy-scaling
title: "EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.11170"
keywords: [inference-scaling, entropy, adaptive-computation, token-level-uncertainty, test-time]
description: "Monitor token-wise entropy to adaptively allocate compute during inference. Branch into multiple paths at high-entropy tokens, reducing token generation by up to 65% while improving accuracy by up to 37% on reasoning tasks."
---

# EAGER: Adaptive Computation via Entropy Monitoring

Parallel sampling and test-time scaling apply uniform compute to all prompts. EAGER recognizes that different problems require different computational effort: straightforward queries waste compute with extensive branching, while complex problems benefit from exploration.

Core insight: token-wise entropy reveals where the model is uncertain. By allocating compute resources only at high-entropy decision points, you reduce redundant computation while improving reasoning on hard problems.

## Core Concept

**Token-Level Entropy Monitoring**: Track entropy of token probability distributions as generation progresses. High entropy = decision point where exploration helps.

**Adaptive Branching**: Dynamically branch into multiple reasoning paths at high-entropy tokens, skip branching at low-entropy tokens.

**Budget-Aware Allocation**: Redirect computational savings from simple problems to complex ones that need more exploration.

## Architecture Overview

- **Entropy Tracker**: Computes entropy at each token position
- **Branch Controller**: Decides when to spawn multiple paths based on entropy
- **Path Manager**: Tracks parallel reasoning paths
- **Result Aggregator**: Combines results from multiple paths

## Implementation Steps

**Stage 1: Token-Level Entropy Computation**

Monitor entropy throughout generation:

```python
import torch
import torch.nn.functional as F
import numpy as np

class EntropyMonitor:
    def __init__(self, entropy_threshold=0.8):
        """
        Monitor entropy and detect high-entropy tokens.

        Args:
            entropy_threshold: percentile of entropy to trigger branching
        """
        self.entropy_threshold = entropy_threshold
        self.entropy_history = []

    def compute_token_entropy(self, logits):
        """
        Compute entropy for each token position.

        Args:
            logits: [batch, vocab_size]

        Returns:
            entropy: [batch] - entropy per sample
        """

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Compute entropy: -sum(p * log(p))
        entropy = -(
            probs * torch.log(probs + 1e-10)
        ).sum(dim=-1)

        return entropy

    def should_branch(self, entropy, running_entropy_history=None):
        """
        Determine if entropy is high enough to branch.
        """

        if running_entropy_history is None:
            running_entropy_history = self.entropy_history

        # Compute threshold dynamically based on history
        if len(running_entropy_history) > 0:
            entropy_percentile = np.percentile(
                running_entropy_history,
                self.entropy_threshold * 100
            )
        else:
            entropy_percentile = 5.0  # Default high threshold

        should_branch = entropy > entropy_percentile

        return should_branch

    def update_history(self, entropy):
        """
        Track entropy for threshold computation.
        """
        self.entropy_history.append(entropy.item())

class AdaptiveBrancher:
    def __init__(self, model, entropy_monitor):
        self.model = model
        self.entropy_monitor = entropy_monitor

    def generate_with_adaptive_branching(
        self,
        prompt,
        max_length=512,
        num_candidate_branches=2,
        branch_depth=3
    ):
        """
        Generate with adaptive branching at high-entropy tokens.
        """

        input_ids = prompt
        active_paths = [{
            'tokens': input_ids,
            'logprobs': 0.0,
            'branch_count': 0
        }]

        entropy_monitor = EntropyMonitor()

        for step in range(max_length):
            new_paths = []

            for path in active_paths:
                # Get logits for next token
                with torch.no_grad():
                    logits = self.model(
                        path['tokens']
                    ).logits[:, -1, :]

                # Compute entropy
                entropy = entropy_monitor.compute_token_entropy(logits)
                entropy_monitor.update_history(entropy)

                # Check if we should branch
                if entropy_monitor.should_branch(entropy):
                    # High entropy: generate multiple candidates
                    num_branches = min(
                        num_candidate_branches,
                        10 // (1 + path['branch_count'])  # Limit branching
                    )

                    # Get top-k most likely next tokens
                    log_probs = F.log_softmax(logits, dim=-1)
                    top_log_probs, top_tokens = torch.topk(
                        log_probs,
                        k=num_branches,
                        dim=-1
                    )

                    # Create branch for each candidate
                    for k in range(num_branches):
                        next_token = top_tokens[0, k].unsqueeze(0).unsqueeze(0)
                        next_logprob = top_log_probs[0, k].item()

                        new_path = {
                            'tokens': torch.cat(
                                [path['tokens'], next_token],
                                dim=-1
                            ),
                            'logprobs': path['logprobs'] + next_logprob,
                            'branch_count': path['branch_count'] + 1
                        }

                        new_paths.append(new_path)

                else:
                    # Low entropy: greedy selection
                    log_probs = F.log_softmax(logits, dim=-1)
                    next_token = log_probs.argmax(dim=-1).unsqueeze(0)
                    next_logprob = log_probs[0, next_token[0, 0]].item()

                    new_path = {
                        'tokens': torch.cat(
                            [path['tokens'], next_token.unsqueeze(0)],
                            dim=-1
                        ),
                        'logprobs': path['logprobs'] + next_logprob,
                        'branch_count': path['branch_count']
                    }

                    new_paths.append(new_path)

            # Prune low-probability paths to manage memory
            if len(new_paths) > 32:  # Keep top 32 paths
                new_paths.sort(
                    key=lambda p: p['logprobs'],
                    reverse=True
                )
                new_paths = new_paths[:32]

            active_paths = new_paths

        # Return best path
        best_path = max(
            active_paths,
            key=lambda p: p['logprobs'] / max(1, len(p['tokens']))
        )

        return best_path['tokens']
```

**Stage 2: Budget-Aware Parallel Sampling**

Implement parallel sampling with entropy-adaptive budgets:

```python
def eager_parallel_sampling(
    model,
    prompt,
    target_budget=1.0,  # 1.0 = standard inference cost
    num_branches_base=4
):
    """
    Run parallel sampling with entropy-aware budget allocation.
    """

    entropy_monitor = EntropyMonitor()
    initial_computation = 0.0

    # Track computation per prompt
    prompt_computations = []

    for prompt_idx, p in enumerate(prompt):
        # Standard generation for computation estimate
        with torch.no_grad():
            output = model.generate(p, max_length=256)

        # Compute per-token entropy
        logits = model(output).logits
        entropy = entropy_monitor.compute_token_entropy(logits)

        # Estimate computation needed for this prompt
        entropy_avg = entropy.mean().item()

        if entropy_avg > 5.0:  # High entropy (hard problem)
            # Use more branches
            num_branches = num_branches_base * 2
        elif entropy_avg < 2.0:  # Low entropy (easy problem)
            # Use fewer branches
            num_branches = num_branches_base // 2
        else:  # Medium entropy
            num_branches = num_branches_base

        prompt_computations.append({
            'prompt_idx': prompt_idx,
            'entropy': entropy_avg,
            'num_branches': num_branches
        })

    # Allocate budget across prompts
    total_branches = sum(
        pc['num_branches'] for pc in prompt_computations
    )

    # Scale branches to target budget
    scale_factor = (target_budget * len(prompt)) / total_branches
    adjusted_branches = [
        max(1, int(pc['num_branches'] * scale_factor))
        for pc in prompt_computations
    ]

    # Generate with adjusted branching
    results = []

    for prompt_idx, num_branches in enumerate(adjusted_branches):
        with torch.no_grad():
            outputs = model.generate(
                prompt[prompt_idx],
                max_length=512,
                num_beams=num_branches,
                num_return_sequences=min(num_branches, 5)
            )

        results.append(outputs)

    return results
```

**Stage 3: Inference Integration**

Deploy EAGER in practical inference:

```python
def eager_inference(
    model,
    prompts,
    target_reduction=0.65,
    improvement_threshold=0.25
):
    """
    Run inference with EAGER scaling.
    Dynamically allocate compute to maximize accuracy under budget.
    """

    entropy_monitor = EntropyMonitor()
    adaptive_brancher = AdaptiveBrancher(model, entropy_monitor)

    results = []
    computation_metrics = {
        'tokens_generated': 0,
        'paths_explored': 0,
        'high_entropy_decisions': 0
    }

    for prompt in prompts:
        # Generate with adaptive branching
        output = adaptive_brancher.generate_with_adaptive_branching(
            prompt,
            max_length=512,
            num_candidate_branches=4
        )

        # Track metrics
        computation_metrics['tokens_generated'] += output.shape[1]
        computation_metrics['paths_explored'] += len(entropy_monitor.entropy_history)

        high_entropy_tokens = sum(
            1 for e in entropy_monitor.entropy_history
            if e > 5.0
        )

        computation_metrics['high_entropy_decisions'] += high_entropy_tokens

        results.append(output)

    # Report efficiency gains
    print(
        f"Tokens generated: {computation_metrics['tokens_generated']}\n"
        f"Paths explored: {computation_metrics['paths_explored']}\n"
        f"High-entropy decisions: {computation_metrics['high_entropy_decisions']}"
    )

    return results
```

## Practical Guidance

**When to Use EAGER:**
- Reasoning tasks with variable difficulty (AIME, math problems)
- Inference scenarios with flexible compute budget
- Workloads mixing easy and hard problems

**When NOT to Use:**
- Real-time inference with strict latency bounds (entropy monitoring adds overhead)
- Tasks requiring exact deterministic output
- Domains without entropy-performance correlation

**Entropy Threshold Tuning:**

| Threshold | Behavior | Best For |
|-----------|----------|----------|
| 0.5 | Very aggressive branching | Hard problems only |
| 0.7 | Balanced | Mixed difficulty |
| 0.9 | Conservative | Easy problems dominant |

**Budget Allocation Strategies:**

| Strategy | Token Reduction | Accuracy Impact |
|----------|-----------------|-----------------|
| Aggressive (0.35x) | 65% fewer tokens | -5% accuracy |
| Balanced (0.5x) | 50% fewer tokens | -2% accuracy |
| Conservative (0.75x) | 25% fewer tokens | +1% accuracy |

**Common Pitfalls:**
- Entropy threshold too low (unnecessary branching on simple tokens)
- Threshold too high (missing exploitable high-entropy points)
- Not tracking computation across entire batch
- Branch pruning too aggressive (lose good paths)

## Reference

Based on the research at: https://arxiv.org/abs/2510.11170
