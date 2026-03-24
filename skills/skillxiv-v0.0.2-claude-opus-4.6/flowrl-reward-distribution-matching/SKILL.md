---
name: flowrl-reward-distribution-matching
title: "FlowRL: Matching Reward Distributions for LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.15207"
keywords: [LLM reasoning, reward distribution matching, flow-balanced optimization, policy learning, chain-of-thought]
description: "Train LLMs with distribution-matching rewards instead of reward maximization to achieve 10% improvement on math reasoning while improving solution diversity by matching the full reward distribution via flow balancing, addressing mode collapse in long chain-of-thought tasks."
---

# Flow-Balanced Reward Distribution Matching: Improving LLM Reasoning Diversity

Traditional reinforcement learning approaches for large language models maximize expected rewards, which causes premature convergence to a single dominant solution mode. This severely limits diversity in reasoning paths, especially in long chain-of-thought (CoT) tasks where multiple valid solutions exist. FlowRL solves this by treating the problem as distributional learning—instead of maximizing rewards, it matches the full distribution of rewards across all generated trajectories using flow-based optimization.

This approach is grounded in GFlowNets theory, which proves that matching reward distributions via flow balance mathematically encourages exploration and diverse solution discovery. By normalizing rewards through a learnable partition function and minimizing KL divergence between the policy and target distribution, FlowRL recovers diverse reasoning strategies while maintaining or improving task performance.

## Core Concept

FlowRL replaces the typical RL objective of maximizing expected reward with a distribution-matching objective: instead of pushing toward high-reward trajectories, it shapes the entire probability distribution to follow a target reward distribution. This subtle shift has profound effects:

1. **Mode coverage**: The policy explores multiple solution paths proportional to their rewards, preventing collapse to a single strategy
2. **Length stability**: By normalizing gradients by sequence length, long CoT reasoning doesn't dominate training signals
3. **Trajectory balance**: The flow-balanced formulation ensures information conservation across the generation process

The key insight: maximization creates peaks; distribution matching creates landscapes. An LLM that understands multiple reasoning strategies is more robust and generalizable than one trained solely on the highest-reward trajectory.

## Architecture Overview

- **Partition function**: A 3-layer MLP that normalizes scalar rewards into a probability distribution, taking hidden state features as input
- **Policy network**: The base LLM, updated to match the target reward distribution instead of maximizing rewards
- **Loss function**: Squared form of flow-balance objective, incorporating length normalization and importance sampling corrections
- **Reference policy**: Maintained for stability during off-policy updates, preventing excessive deviation from the foundation model
- **Importance sampling**: Applies PPO-style clipping to handle distribution mismatch between trajectory samples and current policy

## Implementation

### Step 1: Set Up Reward Normalization with Partition Function

The partition function Z_φ(x) converts scalar rewards into a proper probability distribution. This learnable component receives contextual features and outputs normalized reward weights. The function is crucial because raw rewards have different scales and ranges depending on the task.

```python
import torch
import torch.nn as nn

class PartitionFunction(nn.Module):
    """Learnable partition function that normalizes rewards into a distribution."""

    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, hidden_dim) contextual features
        Returns:
            log_Z: (batch_size, 1) log partition values
        """
        log_Z = self.net(hidden_states)
        return log_Z
```

The partition function takes the mean of the LLM's hidden states and produces a scalar normalization factor. During training, its parameters adjust to balance reward distributions across the batch.

### Step 2: Compute Flow-Balanced Loss with Length Normalization

The core FlowRL loss matches distributions by minimizing KL divergence with length-normalized gradients. Length normalization prevents long sequences from dominating gradients—critical for stable training on variable-length CoT outputs.

```python
def compute_flowrl_loss(
    log_Z: torch.Tensor,  # (batch_size,)
    log_probs: torch.Tensor,  # (batch_size,) log π_θ(y|x)
    rewards: torch.Tensor,  # (batch_size,)
    sequence_lengths: torch.Tensor,  # (batch_size,)
    log_ref_probs: torch.Tensor,  # (batch_size,) log π_ref(y|x)
    beta: float = 1.0,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute the flow-balanced loss.

    Args:
        log_Z: Log partition function values from the partition network
        log_probs: Log probabilities of generated trajectories under current policy
        rewards: Task rewards (0/1 for correctness or continuous scores)
        sequence_lengths: Length of each generated sequence
        log_ref_probs: Reference policy log probabilities
        beta: Temperature parameter controlling reward emphasis
        eps: Small epsilon for numerical stability

    Returns:
        loss: Scalar tensor representing flow-balanced objective
    """
    # Normalize by sequence length to prevent long sequences from dominating
    length_factor = 1.0 / (sequence_lengths.float() + eps)

    # Main flow-balance equation:
    # log Z_φ(x) + (1/|y|) log π_θ(y|x) = β r̂(x,y) + (1/|y|) log π_ref(y|x)
    lhs = log_Z + length_factor * log_probs
    rhs = beta * rewards + length_factor * log_ref_probs

    # Squared loss for distribution matching
    flow_loss = torch.mean((lhs - rhs) ** 2)

    return flow_loss
```

The squared formulation of the flow-balance equation enables distribution matching. The length_factor term is critical: it divides the log-probability contribution by sequence length, ensuring that generating a 100-token vs. 10-token response doesn't create 10x gradient magnitude differences.

### Step 3: Integrate Importance Sampling for Off-Policy Updates

Since trajectories come from the previous policy iteration, FlowRL applies PPO-style importance sampling to prevent off-policy divergence. This stabilizes training when the current policy drifts from the data collection policy.

```python
def apply_importance_sampling_correction(
    log_probs: torch.Tensor,  # (batch_size,)
    log_old_probs: torch.Tensor,  # (batch_size,) from trajectory collection
    flow_loss: torch.Tensor,  # computed flow loss
    clip_ratio: float = 0.2,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Apply PPO-style importance sampling to correct for off-policy data.

    Args:
        log_probs: Current policy log probabilities
        log_old_probs: Log probabilities when trajectories were sampled
        flow_loss: Base flow-balanced loss
        clip_ratio: Clipping threshold for importance weights
        eps: Numerical stability epsilon

    Returns:
        corrected_loss: Flow loss adjusted for policy divergence
    """
    # Importance weight: π_θ / π_old
    log_importance = log_probs - log_old_probs
    importance_weights = torch.exp(torch.clamp(log_importance, min=-5.0, max=5.0))

    # Clip importance weights to [1-clip, 1+clip]
    clipped_weights = torch.clamp(
        importance_weights,
        min=1.0 - clip_ratio,
        max=1.0 + clip_ratio
    )

    # Reweight loss by clipped importance weights
    corrected_loss = torch.mean(clipped_weights * flow_loss)

    return corrected_loss
```

Importance sampling correction prevents the loss from exploding when the current policy becomes too different from the data collection policy. The clipping mechanism bounds the maximum influence of any single example, similar to PPO's clip function.

### Step 4: Training Loop Integration

Wire FlowRL into your LLM training pipeline. This shows how to orchestrate trajectory sampling, reward computation, and the three loss components.

```python
def train_flowrl_step(
    model: nn.Module,  # LLM to train
    partition_fn: PartitionFunction,  # Reward normalizer
    batch: dict,  # Contains: input_ids, labels, sequence_lengths
    reward_model: callable,  # Function that scores (input, output) -> reward
    reference_model: nn.Module,  # Base model (frozen)
    optimizer: torch.optim.Optimizer,
    beta: float = 1.0,
    clip_ratio: float = 0.2
) -> float:
    """
    Single FlowRL training step.

    Args:
        model: LLM being trained
        partition_fn: Partition function network
        batch: Training data batch
        reward_model: Evaluates correctness of completions
        reference_model: Frozen reference policy
        optimizer: Optimizer for both model and partition_fn
        beta: Reward emphasis temperature
        clip_ratio: Importance sampling clipping threshold

    Returns:
        loss_value: Scalar loss for monitoring
    """
    # 1. Forward pass through LLM
    logits = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    ).logits

    # 2. Get log probabilities and hidden states
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    batch_log_probs = extract_trajectory_log_probs(
        log_probs,
        batch['labels']
    )  # (batch_size,)

    hidden_states = model.get_hidden_states()  # Access final hidden layer
    hidden_mean = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)

    # 3. Compute partition function
    log_Z = partition_fn(hidden_mean).squeeze(-1)  # (batch_size,)

    # 4. Compute rewards from reward model
    rewards = batch.get('rewards', torch.zeros(batch['input_ids'].size(0)))

    # 5. Get reference policy log probs
    with torch.no_grad():
        ref_logits = reference_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        ).logits
        ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        batch_log_ref_probs = extract_trajectory_log_probs(
            ref_log_probs,
            batch['labels']
        )  # (batch_size,)

    # 6. Compute flow-balanced loss with length normalization
    flow_loss = compute_flowrl_loss(
        log_Z=log_Z,
        log_probs=batch_log_probs,
        rewards=rewards,
        sequence_lengths=batch['sequence_lengths'],
        log_ref_probs=batch_log_ref_probs,
        beta=beta
    )

    # 7. Apply importance sampling correction
    with torch.no_grad():
        old_log_probs = batch.get('old_log_probs', batch_log_probs.detach())

    corrected_loss = apply_importance_sampling_correction(
        log_probs=batch_log_probs,
        log_old_probs=old_log_probs,
        flow_loss=flow_loss,
        clip_ratio=clip_ratio
    )

    # 8. Backward pass and optimization
    optimizer.zero_grad()
    corrected_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return corrected_loss.item()

def extract_trajectory_log_probs(log_probs, labels):
    """Helper: extract log probability of the chosen label at each position."""
    # Shift: predict token i+1 from token i
    shift_logits = log_probs[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Gather log probs for selected labels
    log_probs_gathered = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1))

    # Sum across sequence dimension, normalize by length
    traj_log_probs = log_probs_gathered.squeeze(-1).sum(dim=-1)
    return traj_log_probs
```

The training loop orchestrates five key steps: forward pass through the LLM, partition function evaluation, reward computation, reference policy log-prob collection, and the combined FlowRL + importance sampling loss. The reference model is frozen to provide a stable baseline for KL divergence.

## Practical Guidance

### Hyperparameter Configuration

| Hyperparameter | Range | Recommended | Notes |
|---|---|---|---|
| **beta (reward scale)** | 0.5-2.0 | 1.0 | Controls reward emphasis in distribution matching. Higher values make distribution sharper around high-reward solutions. |
| **Learning rate (LLM)** | 1e-5 to 1e-4 | 5e-5 | Smaller than standard supervised tuning; distribution matching is more sensitive to step size. |
| **Learning rate (partition_fn)** | 1e-4 to 1e-3 | 5e-4 | Can be higher than LLM since partition function has fewer parameters. |
| **Batch size** | 32-256 | 64 | Larger batches improve distribution estimate stability; smaller batches add regularization. |
| **Sequence length (train)** | 256-2048 | 512-1024 | Must accommodate CoT; longer sequences benefit more from length normalization. |
| **Clip ratio (importance)** | 0.15-0.3 | 0.2 | Standard PPO clipping; prevents outlier examples from dominating. |
| **Gradient clip norm** | 0.5-2.0 | 1.0 | Prevents gradient explosion from length normalization interactions. |

### When to Use FlowRL

- You have multiple valid reasoning paths for a task (math problems, code generation)
- Solution diversity is important for robustness and generalization
- You're training on chain-of-thought reasoning where intermediate steps vary
- Your reward model provides meaningful gradations (not just binary pass/fail)
- Your baseline RL approach exhibits mode collapse or repetitive outputs

### When NOT to Use FlowRL

- Task has a single dominant optimal solution (e.g., translation with fixed answer)
- Reward signal is extremely sparse or noisy (partition function won't find stable distributions)
- Training stability is critical and you need maximum simplicity (basic PPO may be safer)
- Your LLM is very large (>100B) without infrastructure for stable distributed training
- You lack a reliable reward model; FlowRL amplifies reward signal quality issues

### Common Pitfalls

**Gradient explosion from length normalization**: When sequences vary wildly in length (10 tokens to 1000+ tokens), the 1/length factor can create unstable gradient magnitudes. Solution: bucketing sequences by length or using length-conditional beta scheduling.

**Partition function overfitting**: If the partition function capacity is too high relative to batch size, it may memorize rewards rather than learning stable distributions. Solution: keep the MLP small (3 layers, 128-256 hidden dim) and monitor validation curves.

**Off-policy divergence**: If you use very old trajectory data with importance sampling, the clipped weights become mostly 1s and corrections fail. Solution: refresh trajectory buffer frequently (every 1-2 epochs) and monitor importance weight magnitudes.

**Reference policy mismatch**: If the reference model diverges from the training model too far, log-prob differences explode. Solution: periodically update reference model checkpoint and monitor KL divergence.

**Reward scale sensitivity**: Unlike reward maximization which naturally normalizes through expectation, distribution matching is sensitive to reward magnitude. Solution: normalize rewards to [0, 1] or use adaptive reward scaling based on running statistics.

## Reference

Paper: https://arxiv.org/abs/2509.15207

Implementation: https://github.com/Xuekai-Zhu/FlowRL
