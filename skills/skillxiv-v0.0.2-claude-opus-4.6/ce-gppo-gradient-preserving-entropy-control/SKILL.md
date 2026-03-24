---
name: ce-gppo-gradient-preserving-entropy-control
title: "CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.20712"
keywords: [reinforcement learning, policy optimization, entropy control, gradient clipping, gradient preserving, LLM training, PPO variants]
description: "Control policy entropy dynamics in RL by reweighting gradients from clipped tokens. CE-GPPO preserves out-of-clip gradients with beta parameters to stabilize exploration-exploitation balance, preventing entropy collapse while maintaining training stability in LLM fine-tuning."
---

# Outcome: Stable Entropy Control in Policy Optimization

CE-GPPO enables fine-grained control over policy entropy dynamics during reinforcement learning training, preventing entropy collapse while maintaining training stability. This is critical for LLM fine-tuning where premature convergence to deterministic policies undermines exploration and model reasoning capabilities.

## Problem Context

Standard policy gradient methods like PPO use gradient clipping to constrain policy updates, but this creates a hidden cost: gradients from tokens outside the clipping interval (low-probability and high-probability tokens) are discarded. This asymmetric gradient loss accelerates entropy collapse.

In LLM training, entropy collapse manifests as the model converging too quickly to high-probability tokens, losing exploration capability needed for complex reasoning tasks. Analysis shows that:

1. **Positive Advantage, Low Probability (PA&LP) tokens**: Their clipped gradients would encourage exploration but are discarded
2. **Negative Advantage, Low Probability (NA&LP) tokens**: Their clipped gradients accelerate convergence but dominate the update

The covariance between log-probabilities and advantages drives entropy change. When clipped gradients are dropped, this covariance increases unnaturally, forcing rapid policy convergence.

## Core Concept

CE-GPPO reframes entropy control as a gradient reweighting problem. Instead of discarding clipped-token gradients, the method incorporates them with tunable weights beta1 and beta2:

- **beta1**: Controls weight on NA&LP gradients (negative advantage, low probability)
- **beta2**: Controls weight on PA&LP gradients (positive advantage, low probability)

The key insight: policy entropy change is governed by the covariance between log-probabilities and advantages. By reweighting these specific gradient sources, CE-GPPO directly controls entropy dynamics without separate entropy regularization terms. This preserves gradient information while maintaining the pessimistic clipping behavior of standard PPO, ensuring stable optimization.

## Architecture Overview

- **Gradient Classification**: Tokens are classified into three groups based on importance sampling ratio and clipping bounds
- **Weighted Gradient Preservation**: Out-of-clip gradients are multiplied by importance ratios, then weighted by beta parameters
- **Stable Integration**: Non-clipped tokens follow standard PPO gradients; clipped tokens contribute weighted gradients proportional to their importance
- **Hyperparameter Control**: Beta values directly tune entropy dynamics without requiring separate loss terms or complex tuning

The method maintains PPO's pessimistic update mechanism: when advantages are positive but ratios are too high, updates are clipped; when advantages are negative but ratios are too low, updates are clipped. CE-GPPO adds controlled gradient flow from these clipped regions.

## Implementation Overview

### Step 1: Token Classification and Ratio Computation

For each token in a batch, compute the importance sampling ratio (new policy probability / old policy probability) and classify it relative to clipping bounds [1-epsilon, 1+epsilon].

```python
# Token-level importance sampling and classification
import torch
import torch.nn.functional as F

def compute_token_importance_ratios(
    log_probs_new: torch.Tensor,  # [batch, seq_len]
    log_probs_old: torch.Tensor,  # [batch, seq_len]
    advantages: torch.Tensor,     # [batch, seq_len]
    epsilon: float = 0.2
) -> tuple:
    """
    Compute importance ratios and token classifications.

    Returns:
        ratios: importance sampling ratios
        is_in_clip: boolean mask for tokens inside clipping interval
        is_pa_lp: boolean mask for PA&LP tokens (pos advantage, clipped low)
        is_na_lp: boolean mask for NA&LP tokens (neg advantage, clipped low)
    """
    # Importance sampling: exp(log_new - log_old)
    ratios = torch.exp(log_probs_new - log_probs_old)

    # Clipping bounds
    lower_bound = 1.0 - epsilon
    upper_bound = 1.0 + epsilon

    # Determine which tokens are inside the clipping interval
    is_in_clip = (ratios >= lower_bound) & (ratios <= upper_bound)

    # PA&LP: positive advantage but ratio too low (below lower bound)
    is_pa_lp = (advantages > 0) & (ratios < lower_bound)

    # NA&LP: negative advantage but ratio too high (above upper bound)
    is_na_lp = (advantages < 0) & (ratios > upper_bound)

    return ratios, is_in_clip, is_pa_lp, is_na_lp
```

### Step 2: Compute CE-GPPO Objective Loss

The CE-GPPO loss combines standard clipped losses with weighted out-of-clip gradients.

```python
def ce_gppo_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    beta_1: float = 0.5,
    beta_2: float = 1.0
) -> torch.Tensor:
    """
    CE-GPPO loss with gradient-preserving clipping.

    Args:
        log_probs_new: log probabilities under new policy [batch, seq_len]
        log_probs_old: log probabilities under old policy [batch, seq_len]
        advantages: advantage estimates [batch, seq_len]
        epsilon: PPO clipping parameter
        beta_1: weight for NA&LP token gradients
        beta_2: weight for PA&LP token gradients

    Returns:
        loss: scalar CE-GPPO loss (negate for gradient ascent)
    """
    ratios, is_in_clip, is_pa_lp, is_na_lp = compute_token_importance_ratios(
        log_probs_new, log_probs_old, advantages, epsilon
    )

    # Initialize loss accumulator
    loss = torch.zeros(1, device=log_probs_new.device, dtype=log_probs_new.dtype)

    # Standard PPO clipped loss for tokens inside interval
    clipped_ratio = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon)
    standard_loss = -torch.min(
        ratios * advantages,
        clipped_ratio * advantages
    )
    loss = loss + standard_loss[is_in_clip].mean()

    # Weighted gradients for PA&LP tokens (encourage exploration)
    # Stop gradient on clipped ratio to preserve original probability gradients
    if is_pa_lp.any():
        pa_lp_loss = -beta_2 * (ratios * advantages)
        loss = loss + pa_lp_loss[is_pa_lp].mean()

    # Weighted gradients for NA&LP tokens (accelerate convergence)
    if is_na_lp.any():
        na_lp_loss = -beta_1 * (ratios * advantages)
        loss = loss + na_lp_loss[is_na_lp].mean()

    return loss / (is_in_clip.sum().float() + is_pa_lp.sum().float() + is_na_lp.sum().float()).clamp(min=1.0)
```

### Step 3: Batch-Level Training Loop Integration

Integrate CE-GPPO into a standard RL training loop with value function updates.

```python
def ce_gppo_train_step(
    model: torch.nn.Module,
    batch_log_probs_new: torch.Tensor,
    batch_log_probs_old: torch.Tensor,
    batch_advantages: torch.Tensor,
    batch_values_new: torch.Tensor,
    batch_returns: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epsilon: float = 0.2,
    beta_1: float = 0.5,
    beta_2: float = 1.0,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01
) -> dict:
    """
    Single CE-GPPO training step combining policy and value updates.

    Args:
        model: policy and value network
        batch_log_probs_new: new policy log probs [num_samples, seq_len]
        batch_log_probs_old: old policy log probs [num_samples, seq_len]
        batch_advantages: advantage estimates [num_samples, seq_len]
        batch_values_new: value predictions [num_samples]
        batch_returns: cumulative returns [num_samples]
        optimizer: torch optimizer
        epsilon: PPO clipping parameter
        beta_1: weight for NA&LP gradients
        beta_2: weight for PA&LP gradients
        value_coeff: coefficient for value loss
        entropy_coeff: coefficient for entropy regularization

    Returns:
        metrics: dict with loss components
    """
    optimizer.zero_grad()

    # Compute policy loss
    policy_loss = ce_gppo_loss(
        batch_log_probs_new,
        batch_log_probs_old,
        batch_advantages,
        epsilon,
        beta_1,
        beta_2
    )

    # Compute value loss (standard MSE)
    value_loss = F.mse_loss(batch_values_new, batch_returns)

    # Optional entropy regularization (usually disabled with CE-GPPO)
    # Entropy from the new policy
    log_probs_dist = torch.exp(batch_log_probs_new)
    entropy = -(log_probs_dist * batch_log_probs_new).sum(dim=-1).mean()

    # Combined loss
    total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

    # Backward pass and optimization
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "total_loss": total_loss.item()
    }
```

### Step 4: LLM-Specific Integration with Transformers

Adapt CE-GPPO for transformer-based LLMs, handling token sequences and attention masks.

```python
def compute_ce_gppo_gradients_for_lm(
    model: torch.nn.Module,
    input_ids: torch.Tensor,           # [batch, seq_len]
    attention_mask: torch.Tensor,      # [batch, seq_len]
    advantage_scores: torch.Tensor,    # [batch, seq_len-1]
    old_logits: torch.Tensor,          # [batch, seq_len-1, vocab_size]
    epsilon: float = 0.2,
    beta_1: float = 0.5,
    beta_2: float = 1.0
) -> torch.Tensor:
    """
    Compute CE-GPPO loss for language model fine-tuning.

    Args:
        model: transformer language model
        input_ids: token indices
        attention_mask: padding mask
        advantage_scores: token-level advantages from reward model
        old_logits: cached logits from old policy
        epsilon, beta_1, beta_2: CE-GPPO hyperparameters

    Returns:
        loss: scalar loss for backpropagation (negate for gradient ascent)
    """
    # Forward pass to get new logits
    outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=False)
    new_logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]

    # Get target token indices (next tokens in sequence)
    target_tokens = input_ids[:, 1:]  # [batch, seq_len-1]

    # Compute log probabilities
    log_probs_new = F.log_softmax(new_logits, dim=-1)
    log_probs_old = F.log_softmax(old_logits, dim=-1)

    # Extract log probabilities for target tokens
    log_probs_new = torch.gather(log_probs_new, -1, target_tokens.unsqueeze(-1)).squeeze(-1)
    log_probs_old = torch.gather(log_probs_old, -1, target_tokens.unsqueeze(-1)).squeeze(-1)

    # Mask out padding tokens
    advantage_scores = advantage_scores * attention_mask[:, 1:]

    # Compute CE-GPPO loss
    loss = ce_gppo_loss(
        log_probs_new,
        log_probs_old,
        advantage_scores,
        epsilon=epsilon,
        beta_1=beta_1,
        beta_2=beta_2
    )

    return loss
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| beta_1 | 0.5 | [0.0, 1.0] | Weight for NA&LP (negative advantage, low probability) gradients. Higher values accelerate convergence but may increase entropy decay rate. |
| beta_2 | 1.0 | [0.0, 1.0] | Weight for PA&LP (positive advantage, low probability) gradients. Higher values encourage exploration by preserving encouraging-but-clipped gradients. |
| epsilon | 0.2 | [0.15, 0.3] | PPO clipping range. Larger epsilon increases clipping region, affecting how many gradients are reweighted. |
| entropy_coeff | 0.0 | [0.0, 0.01] | Coefficient for entropy regularization. CE-GPPO typically sets this to 0 since entropy is controlled via betas. |
| value_coeff | 0.5 | [0.1, 1.0] | Coefficient for value function loss in combined objective. |

### When to Use CE-GPPO

CE-GPPO is recommended for:

- **LLM fine-tuning with RL**: When training language models with GRPO, DPO variants, or custom reward signals
- **Tasks requiring sustained exploration**: Mathematical reasoning, code generation, complex multi-step reasoning
- **Entropy collapse symptoms**: When observing rapid policy convergence and degraded downstream performance in mid-training
- **Fine-grained entropy control**: When you need to tune exploration-exploitation balance without separate entropy regularization
- **Training stability**: When other entropy mitigation methods (plain entropy regularization) cause gradient instability

Example scenarios: training models on AIME, mathematical problem-solving, reasoning-heavy instruction following.

### When NOT to Use CE-GPPO

Do not use CE-GPPO when:

- **Simple supervised fine-tuning**: If your task doesn't require sustained exploration (e.g., basic instruction-following from a strong teacher policy), standard PPO or simpler methods are sufficient
- **Already converged policies needed**: If your goal is rapid convergence to the best policy and exploration is explicitly undesired
- **Reward signal is noisy**: CE-GPPO amplifies gradients from edge cases (clipped tokens). Noisy rewards will create unstable training signals in these regions
- **Very small model sizes**: Overhead of tracking multiple gradient components may not justify benefits for tiny models
- **Single-turn generation**: Tasks without complex sequential decision-making don't benefit from entropy control
- **Hardware-constrained training**: The method requires tracking additional token-level metadata (clipped token masks, beta coefficients)

### Tuning Strategy

1. **Start with defaults** (beta_1=0.5, beta_2=1.0): These settings balance entropy preservation with convergence
2. **Monitor entropy curves**: Track policy entropy throughout training. CE-GPPO should show stable, gradually declining entropy
3. **Adjust beta_1 if entropy decays too slowly**: Increase beta_1 to accelerate convergence (weight NA&LP gradients more)
4. **Adjust beta_2 if entropy decays too fast**: Increase beta_2 to preserve exploration (weight PA&LP gradients more)
5. **Validate on held-out benchmarks**: Ensure entropy dynamics correlate with improved downstream performance
6. **Test robustness**: CE-GPPO shows robustness to moderate hyperparameter changes; small adjustments (±0.25) typically don't degrade performance

### Common Pitfalls

1. **Setting both betas to 0**: Degenerates to standard PPO with entropy collapse. At least one beta should be non-zero
2. **Excessive entropy regularization alongside CE-GPPO**: Double-counting entropy control leads to divergence. Set entropy_coeff=0 when using CE-GPPO
3. **Ignoring base policy stability**: CE-GPPO assumes the old policy (policy_old in ratios) is stable. If old policy changes rapidly between updates, reweighting becomes unstable
4. **Over-weighting PA&LP gradients**: Very high beta_2 (>1.5) can cause gradient explosion in early training. Keep beta_2 ≤ 1.0
5. **Misaligned advantage signals**: If advantages are incorrectly computed or scaled, clipped-token reweighting will amplify errors. Validate advantage computation independently

### Training Dynamics Insights

- **Early training**: High entropy is generally beneficial. Maintain stable, high entropy to allow broad exploration
- **Mid training**: Entropy should gradually decline as the policy discovers good solutions. CE-GPPO controls the rate
- **Late training**: Entropy stabilizes at a non-zero level. This prevents mode-collapse and allows sampling of diverse solutions

Research shows that CE-GPPO achieves the best balance when maintaining "relatively high and stable entropy" throughout training, with greater weight on PA&LP gradients (higher beta_2) than NA&LP gradients (lower beta_1).

## Reference

Paper: CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning

Authors: Zhenpeng Su, Leiyu Pan, Minxuan Lv, Yuntao Li, Wenping Hu, Fuzheng Zhang, Kun Gai, Guorui Zhou

arXiv: https://arxiv.org/abs/2509.20712

Citation:
```
@article{su2025cegppo,
  title={CE-GPPO: Controlling Entropy via Gradient-Preserving Clipping Policy Optimization in Reinforcement Learning},
  author={Su, Zhenpeng and Pan, Leiyu and Lv, Minxuan and Li, Yuntao and Hu, Wenping and Zhang, Fuzheng and Gai, Kun and Zhou, Guorui},
  journal={arXiv preprint arXiv:2509.20712},
  year={2025}
}
```

### Key Experimental Results

On mathematical reasoning benchmarks (AIME24, AIME25, HMMT25, MATH500, AMC23):

- **1.5B model**: 45.2% → 54.9% average benchmark accuracy (+9.7 points)
- **7B model**: 60.8% → 67.5% average benchmark accuracy (+6.7 points)

Compared to:
- GRPO baseline: Suffers from entropy collapse
- DAPO: Overly high entropy in early training, slower performance gains
- CE-GPPO: Stable entropy curves with consistent performance improvements across benchmarks

Implementation tested on DeepSeek-R1-Distill models with mathematical reasoning datasets (30k samples). Gradients remain stable with KL divergence and gradient norms within expected ranges relative to standard PPO.
