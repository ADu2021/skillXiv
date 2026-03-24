---
name: modomodо-multimodal-rl
title: "MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24871"
keywords: [Multimodal RL, Data Mixture, Vision-Language, RLVR, Multi-Task]
description: "Optimize data mixtures across diverse vision-language domains when applying RL with verifiable rewards to multimodal LLMs, balancing task-specific performance with generalization."
---

# Optimize Multi-Domain Data Mixtures for Multimodal RL Training

Multimodal large language models (MLLMs) must handle heterogeneous tasks: visual QA, image captioning, spatial reasoning, scene understanding. When applying reinforcement learning with verifiable rewards (RLVR), a critical challenge emerges: how should you balance training data across these diverse domains?

MoDoMoDo addresses this through systematic data mixture optimization: finding the right proportion of training from each domain so the model learns task-specific skills without forgetting others. Unlike single-domain RL, multimodal RL requires careful orchestration of data flows to prevent task interference and maximize transfer.

## Core Concept

MoDoMoDo optimizes task mixture ratios for multimodal RL:

- **Domain-specific capability**: Each domain requires unique visual, logical, and spatial skills
- **Interference avoidance**: Training hard on one domain shouldn't degrade others
- **Curriculum design**: Sequence domains in ways that enable positive transfer
- **Metric tracking**: Monitor per-domain performance during training
- **Adaptive mixture**: Adjust ratios based on observed performance gaps
- **Generalization**: Find mixtures that improve broader capability

The key insight is that heterogeneous task distributions demand explicit mixture management—uniform random sampling isn't optimal.

## Architecture Overview

- **Task taxonomy**: Categorize multimodal tasks by required capabilities
- **Capacity tracking**: Monitor model capacity utilization per task
- **Performance monitoring**: Track metrics separately for each domain
- **Interference detection**: Identify when training on one domain hurts others
- **Mixture scheduler**: Dynamically adjust domain sampling probabilities
- **Transfer measurement**: Quantify positive/negative transfer between domains
- **Consolidation strategy**: Ensure all domains reach target performance

## Implementation

Build a data mixture optimizer for multimodal RL:

```python
# MoDoMoDo: Multi-Domain Data Mixture Optimization
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple
import numpy as np

class MultiDomainDataMixer:
    """
    Optimize data mixture ratios for multimodal RL training across domains.
    """
    def __init__(self, domain_names: List[str], initial_weights: Dict[str, float] = None):
        self.domains = domain_names
        self.num_domains = len(domain_names)

        # Initialize mixture weights (uniform by default)
        if initial_weights is None:
            initial_weights = {d: 1.0 / self.num_domains for d in domain_names}

        self.mixture_weights = initial_weights
        self.performance_history = {d: [] for d in domain_names}
        self.interference_matrix = np.zeros((self.num_domains, self.num_domains))

    def create_mixed_dataloader(self, domain_datasets: Dict[str, torch.utils.data.Dataset],
                                batch_size: int = 32, training: bool = True):
        """
        Create a dataloader that samples from domains according to mixture weights.
        """
        if training:
            # Weighted sampling favors domains with lower performance
            adjusted_weights = self._adjust_weights_by_performance()

            # Create sampler that respects mixture ratios
            all_indices = []
            all_domain_labels = []

            # Concatenate datasets
            total_samples = 0
            dataset_offsets = {}
            concatenated_data = []

            offset = 0
            for domain in self.domains:
                dataset_offsets[domain] = offset
                concatenated_data.extend(domain_datasets[domain])
                offset += len(domain_datasets[domain])
                total_samples += len(domain_datasets[domain])

            # Create weighted sampler
            sample_weights = [adjusted_weights.get(d, 1.0 / self.num_domains)
                             for d in self.domains]

            # Oversample domains with high mixture weight
            samples_per_domain = {
                d: int(adjusted_weights[d] * total_samples)
                for d in self.domains
            }

            # Stratified sampling
            indices = []
            for domain in self.domains:
                domain_size = len(domain_datasets[domain])
                domain_indices = np.random.choice(
                    domain_size,
                    size=min(samples_per_domain[domain], domain_size),
                    replace=True
                )
                indices.extend(domain_indices + dataset_offsets[domain])

            # Shuffle
            np.random.shuffle(indices)

            # Create dataloader
            from torch.utils.data import Subset
            subset = Subset(concatenated_data, indices)
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        else:
            # Test: no mixture, equal distribution
            concatenated_data = []
            for domain in self.domains:
                concatenated_data.extend(domain_datasets[domain])

            dataloader = DataLoader(concatenated_data, batch_size=batch_size, shuffle=False)

        return dataloader

    def update_mixture_from_performance(self, per_domain_metrics: Dict[str, float]):
        """
        Adjust mixture weights based on observed per-domain performance.
        """
        # Store history
        for domain, metric in per_domain_metrics.items():
            self.performance_history[domain].append(metric)

        # Compute performance improvement rate per domain
        improvement_rates = {}
        for domain in self.domains:
            hist = self.performance_history[domain]
            if len(hist) >= 2:
                recent = np.mean(hist[-5:]) if len(hist) >= 5 else hist[-1]
                older = np.mean(hist[:5]) if len(hist) >= 5 else hist[0]
                improvement = recent - older
                improvement_rates[domain] = improvement
            else:
                improvement_rates[domain] = 0.0

        # Find slowest-improving domains and increase their mixture weight
        mean_improvement = np.mean(list(improvement_rates.values()))

        new_weights = {}
        for domain in self.domains:
            if improvement_rates[domain] < mean_improvement:
                # Increase weight for slow-improving domains
                new_weights[domain] = self.mixture_weights[domain] * 1.1
            else:
                # Decrease weight for fast-improving domains
                new_weights[domain] = self.mixture_weights[domain] * 0.9

        # Normalize to sum to 1
        total_weight = sum(new_weights.values())
        self.mixture_weights = {d: w / total_weight for d, w in new_weights.items()}

        print(f"Updated mixture weights: {self.mixture_weights}")

    def _adjust_weights_by_performance(self) -> Dict[str, float]:
        """
        Adjust mixture weights based on current performance gaps.
        Higher weight for lower-performance domains to prevent catastrophic forgetting.
        """
        # Get latest metrics for each domain
        latest_metrics = {}
        for domain in self.domains:
            if self.performance_history[domain]:
                latest_metrics[domain] = self.performance_history[domain][-1]
            else:
                latest_metrics[domain] = 0.5  # Neutral

        # Compute inverse performance (lower performance = higher weight)
        adjusted_weights = {}
        for domain in self.domains:
            # Use inverse of performance gap from target
            target_performance = 0.9
            gap = max(0, target_performance - latest_metrics[domain])
            adjusted_weights[domain] = (1.0 + gap) * self.mixture_weights[domain]

        # Normalize
        total = sum(adjusted_weights.values())
        adjusted_weights = {d: w / total for d, w in adjusted_weights.items()}

        return adjusted_weights

    def detect_interference(self, before_metrics: Dict[str, float],
                          after_metrics: Dict[str, float]):
        """
        Detect negative transfer: when training on one domain hurts another.
        """
        for d1 in self.domains:
            for d2 in self.domains:
                if d1 != d2:
                    # Did training emphasize d1 hurt d2?
                    d2_degradation = before_metrics[d2] - after_metrics[d2]
                    if d2_degradation > 0.02:  # >2% drop is interference
                        self.interference_matrix[self.domains.index(d1),
                                              self.domains.index(d2)] += 1

    def get_curriculum_order(self) -> List[str]:
        """
        Determine order of domains to minimize negative transfer.
        Uses interference matrix to schedule domains intelligently.
        """
        # Start with lowest-interference domains
        # This enables positive transfer before tackling harder domains
        interference_sums = self.interference_matrix.sum(axis=1)
        order_indices = np.argsort(interference_sums)
        return [self.domains[i] for i in order_indices]
```

Implement RLVR training with mixture management:

```python
def train_mllm_with_mixed_domains(model, domain_datasets: Dict[str, torch.utils.data.Dataset],
                                  num_epochs=10, target_domains: Dict[str, float] = None):
    """
    Train multimodal LLM using RL with verifiable rewards across multiple domains.
    """
    mixer = MultiDomainDataMixer(list(domain_datasets.keys()))

    # Initialize target performance per domain
    if target_domains is None:
        target_domains = {d: 0.8 for d in domain_datasets.keys()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        # Create mixed dataloader
        train_loader = mixer.create_mixed_dataloader(domain_datasets, batch_size=32)

        epoch_metrics = {d: [] for d in domain_datasets.keys()}
        before_epoch_metrics = {d: 0.5 for d in domain_datasets.keys()}  # Placeholder

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images']
            questions = batch['questions']
            ground_truths = batch['ground_truths']
            domain_labels = batch['domain']  # Which domain this batch is from

            # Forward pass: get model predictions
            outputs = model(images, questions)

            # Compute per-domain rewards (verifiable)
            per_domain_rewards = {}
            for domain in domain_datasets.keys():
                domain_mask = domain_labels == domain
                if domain_mask.sum() > 0:
                    domain_outputs = outputs[domain_mask]
                    domain_gts = ground_truths[domain_mask]

                    # Task-specific verification
                    if domain == 'vqa':
                        reward = verify_vqa_answer(domain_outputs, domain_gts)
                    elif domain == 'caption':
                        reward = verify_caption_quality(domain_outputs, domain_gts)
                    elif domain == 'spatial':
                        reward = verify_spatial_reasoning(domain_outputs, domain_gts)
                    else:
                        reward = verify_generic_task(domain_outputs, domain_gts)

                    per_domain_rewards[domain] = reward.mean().item()
                    epoch_metrics[domain].extend(reward.cpu().numpy())

            # RL update with domain-aware weighting
            mixture_weights = mixer.mixture_weights
            total_loss = 0

            for domain, reward in per_domain_rewards.items():
                domain_loss = -reward  # Negative because we maximize reward
                weighted_loss = domain_loss * mixture_weights[domain]
                total_loss += weighted_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Update mixture based on domain performance
        avg_domain_metrics = {d: np.mean(epoch_metrics[d]) if epoch_metrics[d] else 0.5
                             for d in domain_datasets.keys()}
        mixer.update_mixture_from_performance(avg_domain_metrics)

        # Detect interference
        mixer.detect_interference(before_epoch_metrics, avg_domain_metrics)

        # Log performance
        print(f"Epoch {epoch}:")
        for domain, metric in avg_domain_metrics.items():
            target = target_domains[domain]
            print(f"  {domain}: {metric:.3f} (target: {target:.3f})")

        before_epoch_metrics = avg_domain_metrics

    return mixer
```

## Practical Guidance

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Initial mixture ratio | Uniform or inverse performance | Start balanced, then adapt |
| Weight adjustment speed | 1.05 - 1.2 per epoch | Too fast = instability; too slow = slow adaptation |
| Target performance gap threshold | 0.02 - 0.05 | When to increase domain weight |
| Interference detection threshold | 0.02 drop | What counts as negative transfer |
| Num domains | 2 - 8 | More domains = more complex optimization |

**When to use MoDoMoDo:**
- Training MLLMs on multiple heterogeneous vision-language tasks
- RL with verifiable rewards across diverse domains
- Need to prevent catastrophic forgetting of some tasks
- Want to measure and understand task interference
- Building robust multimodal foundation models

**When NOT to use:**
- Single task training (mixture optimization not relevant)
- Data is naturally uniform in distribution
- Tasks are already well-balanced in importance
- Computational budget for tracking per-domain metrics is tight
- Using pure supervised learning without RL

**Common pitfalls:**
- Not tracking per-domain metrics separately (can't detect problems)
- Mixture weights diverge too far (one domain gets almost nothing)
- Ignoring negative transfer signals (interference goes unaddressed)
- Adjusting weights too aggressively (training becomes unstable)
- Using same reward for all domains (domain-specific verification critical)
- Not considering curriculum order (some domains should be learned first)
- Interference detection threshold too loose (false positives)

## Reference

**MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning**
https://arxiv.org/abs/2505.24871
