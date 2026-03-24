---
name: multi-task-grpo-robust
title: "Multi-Task GRPO: Reliable LLM Reasoning Across Tasks"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.05547"
keywords: [Multi-Task Learning, Robust Optimization, GRPO, Task Reweighting, Reinforcement Learning]
description: "Enable balanced multi-task GRPO training via robustness-aware optimization and improvement-aware task reweighting, dynamically adjusting task weights based on both reward and loss trajectory improvement, achieving 6-28% worst-task improvements while maintaining competitive average accuracy."
---

# Multi-Task GRPO: Robustness-Aware Optimization Across Tasks

Applying GRPO independently to multiple tasks leads to performance imbalance where some tasks dominate training while others stagnate. Multi-Task GRPO introduces robustness-aware optimization formulated as a minimax problem, balancing average performance against task-performance disparities. Improvement-aware task reweighting combines task-level rewards with learning progress signals, preventing weight collapse while ensuring all tasks improve.

## Core Concept

The key insight is that task importance depends not only on reward signals but also on whether a task is making progress. A task with zero reward but improving loss is more important than a plateau task with high reward. By jointly optimizing for average performance and worst-case performance while tracking improvement, MT-GRPO achieves both robustness and efficiency.

## Architecture Overview

- **Robustness-Aware Objective**: Constrained optimization minimizing performance disparities while maintaining competitive average accuracy
- **Improvement-Aware Task Reweighting**: Combines task reward with loss trajectory to detect stagnation and allocate more training budget
- **Ratio-Preserving Sampler**: Addresses GRPO's unique challenge where zero-gradient tasks (identical rewards) require special handling
- **Lagrangian Relaxation**: Converts constrained problem into unconstrained minimax form for practical optimization
- **Convergence Guarantees**: Provably converges to balanced solution

## Implementation

### Step 1: Define Robustness-Aware Objective

Formulate multi-task optimization as constrained problem balancing robustness and average performance.

```python
import torch
import torch.nn.functional as F

def compute_robustness_objective(task_rewards, task_weights, lambda_robustness=1.0):
    """
    Robustness-aware objective combining:
    1. Average performance across tasks
    2. Minimum performance across tasks (worst-case)
    """
    # Weighted average reward
    avg_reward = (task_rewards * task_weights).sum()
    
    # Worst-case (minimum) reward
    min_reward = task_rewards.min()
    
    # Robustness objective: maximize avg while minimizing disparity
    disparity = avg_reward - min_reward
    
    # Combined objective (higher is better)
    robustness_loss = avg_reward - lambda_robustness * disparity
    
    return robustness_loss, avg_reward, min_reward, disparity
```

### Step 2: Implement Improvement-Aware Task Reweighting

Track task progress and adjust weights dynamically.

```python
class ImprovementAwareTaskWeighter:
    """
    Dynamically reweight tasks based on reward and loss improvement.
    """
    def __init__(self, num_tasks, initial_weight=1.0):
        self.num_tasks = num_tasks
        self.task_weights = torch.ones(num_tasks) * initial_weight
        self.task_reward_history = [[] for _ in range(num_tasks)]
        self.task_loss_history = [[] for _ in range(num_tasks)]
    
    def update_weights(self, task_id, reward, loss, window_size=10):
        """
        Update weights for a task based on recent performance trend.
        Improvement: positive change in reward or negative change in loss.
        """
        self.task_reward_history[task_id].append(reward)
        self.task_loss_history[task_id].append(loss)
        
        # Compute improvement signals
        if len(self.task_reward_history[task_id]) >= window_size:
            recent_rewards = self.task_reward_history[task_id][-window_size:]
            recent_losses = self.task_loss_history[task_id][-window_size:]
            
            # Improvement metrics
            reward_improvement = (recent_rewards[-1] - recent_rewards[0]) / (recent_rewards[0] + 1e-6)
            loss_improvement = (recent_losses[0] - recent_losses[-1]) / (recent_losses[0] + 1e-6)
            
            # Combined improvement signal
            improvement = max(reward_improvement, loss_improvement)
        else:
            improvement = 1.0  # Default high weight for new tasks
        
        # Update weight: boost tasks showing improvement
        self.task_weights[task_id] *= (1 + improvement)
        
        # Normalize weights
        self.task_weights = self.task_weights / self.task_weights.sum() * self.num_tasks
        
        return self.task_weights[task_id].item()
    
    def get_weights(self):
        """Return current task weights."""
        return self.task_weights / self.task_weights.sum() * self.num_tasks
```

### Step 3: Implement Ratio-Preserving Sampler

Handle GRPO's challenge where multiple tasks have identical rewards (zero-gradient cases).

```python
class RatioPreservingSampler:
    """
    Ensures task proportions are preserved in post-filtered batches.
    
    Problem: In GRPO, trajectories with identical rewards contribute zero gradient.
    Tasks with high zero-gradient rates get underrepresented in processed batches.
    Solution: Oversample and acceptance-aware resample.
    """
    def __init__(self, num_tasks, target_ratios):
        self.num_tasks = num_tasks
        self.target_ratios = target_ratios  # Desired task proportions in batch
    
    def sample_with_ratio_preservation(self, trajectories_per_task, group_size=4):
        """
        Sample from trajectories while maintaining target ratios.
        """
        batch = []
        task_counts = {t: 0 for t in range(self.num_tasks)}
        
        for task_id in range(self.num_tasks):
            trajectories = trajectories_per_task[task_id]
            target_count = int(group_size * self.target_ratios[task_id])
            
            # Oversample to account for zero-gradient filtering
            # Estimate acceptance rate (trajectories that contribute to gradient)
            acceptance_rates = [
                1.0 if len(set(t['reward'] for t in group)) > 1 else 0.0
                for group in [trajectories[i:i+4] for i in range(0, len(trajectories), 4)]
            ]
            avg_acceptance = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.5
            
            # Oversample to compensate
            oversample_count = int(target_count / (avg_acceptance + 1e-6))
            
            # Sample with replacement
            sampled = torch.multinomial(
                torch.ones(len(trajectories)),
                min(oversample_count, len(trajectories)),
                replacement=True
            )
            
            selected = [trajectories[i] for i in sampled]
            batch.extend(selected)
            task_counts[task_id] = len(selected)
        
        return batch, task_counts
```

### Step 4: Multi-Task GRPO Training Loop

Integrate robustness objective and improvement-aware reweighting.

```python
def multi_task_grpo_training(
    model,
    task_dataloaders,
    num_tasks,
    num_steps=1000,
    lambda_robustness=0.5,
    group_size=4
):
    """
    Multi-task GRPO with robustness and improvement awareness.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    weighter = ImprovementAwareTaskWeighter(num_tasks)
    sampler = RatioPreservingSampler(
        num_tasks,
        target_ratios=torch.ones(num_tasks) / num_tasks
    )
    
    for step in range(num_steps):
        # Collect trajectories from all tasks
        all_trajectories = []
        task_rewards = []
        
        for task_id in range(num_tasks):
            dataloader = task_dataloaders[task_id]
            batch = next(iter(dataloader))
            
            # Generate rollouts
            trajectories = model.generate_rollouts(batch, num_rollouts=group_size)
            rewards = [compute_reward(t) for t in trajectories]
            
            all_trajectories.append(trajectories)
            task_rewards.append(torch.tensor(rewards).mean())
            
            # Update weights based on task performance
            loss = model.compute_loss(batch)
            weighter.update_weights(task_id, rewards[0], loss.item())
        
        # Get current task weights
        task_weights = weighter.get_weights()
        task_weights = task_weights / task_weights.sum()
        
        # Compute robustness objective
        task_rewards = torch.stack(task_rewards)
        robustness_loss, avg_reward, min_reward, disparity = compute_robustness_objective(
            task_rewards, task_weights, lambda_robustness
        )
        
        # Sample with ratio preservation
        batch, task_counts = sampler.sample_with_ratio_preservation(all_trajectories, group_size)
        
        # GRPO loss with robustness weighting
        total_loss = 0.0
        for task_id, trajectories in enumerate(all_trajectories):
            for trajectory in trajectories:
                # Get log probabilities
                log_prob = model.get_log_prob(trajectory['tokens'])
                
                # Advantage within group
                group_rewards = [t['reward'] for t in trajectories[:group_size]]
                advantage = trajectory['reward'] - torch.tensor(group_rewards).mean()
                
                # Weighted by task importance
                task_loss = -log_prob * advantage * task_weights[task_id]
                total_loss += task_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step}: avg_reward={avg_reward:.3f}, "
                  f"min_reward={min_reward:.3f}, disparity={disparity:.3f}")
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|----------------|-------|
| Lambda Robustness | 0.3-1.0 | Higher = more emphasis on worst-case performance; 0.5 is good balance |
| Improvement Window | 5-15 steps | Track recent progress to detect stagnation |
| Group Size | 4-8 trajectories | Affects GRPO stability and variance reduction |
| Task Count | 3-20 tasks | Scalability tested up to 9 tasks; behavior TBD for larger sets |
| Oversample Factor | 1.5-2x | Compensate for zero-gradient cases via resampling |

**When to Use:**
- Multi-task LLM training where task balance matters
- Scenarios with diverse task difficulties (some fail, some saturate)
- Applications where worst-case performance is critical (safety, robustness)

**When Not to Use:**
- Single-task training (use standard GRPO)
- Tasks with naturally balanced reward distributions
- Systems requiring strict per-task isolation

## Reference

Achieves 6-28% improvement on worst-task performance while maintaining competitive average accuracy, requiring 50% fewer training steps to reach robustness milestones compared to standard GRPO and other multi-task baselines.
