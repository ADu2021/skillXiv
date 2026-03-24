---
name: soft-grpo-soft-thinking-rl-lms
title: "SofT-GRPO: Surpassing Discrete-Token LLM RL via Gumbel-Reparameterized Soft-Thinking"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.06411"
keywords: [Reinforcement Learning, Soft Tokens, Policy Gradients, LLM Training, Gumbel Reparameterization]
description: "Enable policy gradient optimization on soft LLM tokens by injecting Gumbel noise and applying Gumbel-Softmax reparameterization—allowing soft-thinking patterns to match discrete-token RL performance while maintaining continuous optimization advantages."
---

# Optimize Soft-Thinking LLMs with Gumbel-Reparameterized Policy Gradients

Soft-thinking enables LLMs to decompose reasoning through continuous internal tokens rather than discrete outputs. However, applying reinforcement learning to soft tokens has been problematic—naive approaches produce out-of-distribution tokens or fail to optimize effectively. SofT-GRPO bridges this gap through three innovations: Gumbel noise injection, Gumbel-Softmax reparameterization, and policy gradient adaptation.

The result is that soft-thinking patterns can now match or exceed discrete-token RL performance. On reasoning tasks, soft-thinking achieves +0.13% improvement on single-attempt metrics and substantial +2.19% gains on multi-attempt evaluation, closing a previously intractable performance gap.

## Core Concept

SofT-GRPO treats soft token generation as a continuous optimization problem solvable via policy gradients. The key insight is that standard RL fails because soft tokens drift out of the pre-trained embedding space. By injecting Gumbel noise and applying the Gumbel-Softmax trick, we maintain tokens within the learned embedding space while enabling full differentiability for policy optimization.

The architecture combines soft-thinking internal reasoning with group relative policy optimization (GRPO), enabling agents to develop deeper reasoning patterns while maintaining the benefits of token-level optimization.

## Architecture Overview

- **Soft Token Generator**: Produces continuous representations instead of discrete tokens
- **Gumbel Noise Injection**: Adds Gumbel-distributed noise to logits for stochasticity
- **Gumbel-Softmax Layer**: Maps noisy logits to embedded space while preserving differentiability
- **Reparameterization Trick**: Enables gradients to flow through the sampling process
- **GRPO Loss Module**: Computes policy gradients using relative rewards across rollouts
- **Embedding Space Validation**: Ensures generated tokens remain in pre-trained vocabulary space

## Implementation Steps

**Step 1: Gumbel Noise Injection**

Add Gumbel-distributed noise to logits to introduce stochasticity required for exploration in policy learning.

```python
import torch
import torch.nn.functional as F

def gumbel_noise(shape, device, eps=1e-20):
    """
    Sample Gumbel(0, 1) noise.

    Args:
        shape: Tensor shape for noise
        device: torch device
        eps: Small value for numerical stability

    Returns:
        gumbel_samples: Gumbel-distributed noise
    """
    # Sample uniform [0, 1)
    uniform = torch.rand(shape, device=device)

    # Transform to Gumbel(0, 1): -log(-log(u))
    gumbel = -torch.log(-torch.log(uniform + eps) + eps)
    return gumbel

def inject_gumbel_noise(logits, temperature=1.0):
    """
    Add Gumbel noise to logits for stochastic sampling.

    Args:
        logits: Raw model logits [batch_size, vocab_size]
        temperature: Softmax temperature (higher = more uniform)

    Returns:
        noisy_logits: Logits with Gumbel noise injected
    """
    gumbel = gumbel_noise(logits.shape, logits.device)
    noisy_logits = logits + gumbel
    return noisy_logits / temperature
```

**Step 2: Gumbel-Softmax Reparameterization**

Map noisy logits to differentiable categorical samples that stay within the pre-trained embedding space.

```python
def gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Gumbel-Softmax: differentiable approximation of categorical sampling.

    Args:
        logits: Raw model logits [batch_size, vocab_size]
        tau: Temperature parameter (lower = sharper, closer to discrete)
        hard: If True, return one-hot; if False, return soft probabilities

    Returns:
        samples: Differentiable samples in [0, 1]^vocab_size
    """
    # Inject Gumbel noise
    noisy_logits = inject_gumbel_noise(logits, temperature=tau)

    # Softmax: converts to probability distribution
    soft_samples = F.softmax(noisy_logits, dim=-1)

    if hard:
        # Straight-through estimator: one-hot in forward, softmax in backward
        hard_samples = torch.zeros_like(soft_samples)
        hard_samples.scatter_(-1, soft_samples.argmax(dim=-1, keepdim=True), 1)
        # Gradient flows through soft_samples; output is hard
        return hard_samples - soft_samples.detach() + soft_samples
    else:
        return soft_samples
```

**Step 3: Soft Token Embedding Projection**

Map soft categorical distributions to continuous embeddings while ensuring in-distribution tokens.

```python
class SoftTokenEmbedding(torch.nn.Module):
    """Projects soft token distributions to embedding space."""

    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings):
        """
        Args:
            vocab_size: Number of tokens in vocabulary
            embedding_dim: Dimension of embedding space
            pretrained_embeddings: Pre-trained token embeddings [vocab_size, embedding_dim]
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Store pre-trained embeddings (frozen or fine-tunable)
        self.register_buffer('embeddings', pretrained_embeddings)

    def forward(self, soft_probs):
        """
        Project soft categorical distribution to embedding space.

        Args:
            soft_probs: Soft samples [batch_size, seq_len, vocab_size] from Gumbel-Softmax

        Returns:
            embedded: Continuous embeddings [batch_size, seq_len, embedding_dim]
        """
        # Weighted sum: E[embedding] = sum_i p_i * embedding_i
        # This keeps tokens within convex hull of pre-trained embeddings
        embedded = torch.einsum('bsv,ve->bse', soft_probs, self.embeddings)
        return embedded
```

**Step 4: Policy Gradient Computation with GRPO**

Compute group-relative policy optimization loss enabling RL on soft tokens.

```python
def compute_grpo_loss(soft_logits, rewards, baseline_logits=None):
    """
    Group Relative Policy Optimization loss for soft tokens.

    Args:
        soft_logits: Model output logits [batch_size, seq_len, vocab_size]
        rewards: Task rewards [batch_size] (correctness scores)
        baseline_logits: Logits from previous policy (for baseline)

    Returns:
        loss: GRPO loss scalar
    """
    batch_size = rewards.shape[0]

    # Compute log probabilities under Gumbel-Softmax
    soft_probs = F.softmax(soft_logits, dim=-1)
    log_probs = torch.log(soft_probs + 1e-10)  # [batch_size, seq_len, vocab_size]

    # Group relative advantage: compare within batch
    mean_reward = rewards.mean()
    advantages = rewards - mean_reward  # [batch_size]

    # Policy loss: maximize log_prob * advantage
    # Average log_prob across tokens for each example
    mean_log_probs = log_probs.mean(dim=(1, 2))  # [batch_size]

    policy_loss = -(mean_log_probs * advantages).mean()

    # KL penalty: diverge from baseline gradually
    if baseline_logits is not None:
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        kl_div = F.kl_div(
            log_probs.reshape(-1, log_probs.shape[-1]),
            baseline_probs.reshape(-1, baseline_probs.shape[-1]),
            reduction='batchmean'
        )
        policy_loss += 0.01 * kl_div  # Weight KL penalty

    return policy_loss
```

**Step 5: End-to-End Soft-Thinking Training Loop**

Integrate Gumbel-Softmax projection and GRPO loss into training.

```python
class SoftThinkingAgent(torch.nn.Module):
    """LLM agent using soft-thinking with Gumbel-GRPO optimization."""

    def __init__(self, model, embedding_layer, vocab_size):
        super().__init__()
        self.model = model
        self.soft_embedding = SoftTokenEmbedding(vocab_size, model.hidden_size,
                                                   model.get_input_embeddings().weight)

    def forward(self, input_ids, attention_mask=None):
        """Generate soft logits."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        return outputs.logits

    def generate_soft_tokens(self, prompt, max_steps=100, temperature=0.5):
        """
        Generate reasoning through soft tokens.

        Args:
            prompt: Initial prompt text
            max_steps: Maximum soft token generation steps
            temperature: Gumbel temperature (lower = sharper)

        Returns:
            soft_trajectory: Soft tokens across reasoning steps
        """
        input_ids = self.tokenize(prompt)
        soft_trajectory = []

        for step in range(max_steps):
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :]  # Last position

            # Gumbel-Softmax sampling
            soft_sample = gumbel_softmax(next_logits, tau=temperature, hard=False)
            soft_trajectory.append(soft_sample)

            # Project to embedding space and append
            embedded = self.soft_embedding(soft_sample.unsqueeze(1))
            input_ids = self.append_soft_token(input_ids, embedded)

        return soft_trajectory

def train_soft_thinking(agent, tasks, num_epochs=3):
    """
    Training loop combining soft token generation with GRPO.

    Args:
        agent: SoftThinkingAgent instance
        tasks: List of reasoning tasks with ground truth
        num_epochs: Training epochs
    """
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        for batch_tasks in batch_iterator(tasks, batch_size=32):
            # Generate soft tokens for each task
            soft_trajectories = []
            logits_list = []

            for task in batch_tasks:
                input_ids = agent.tokenize(task.prompt)
                logits = agent.forward(input_ids)
                logits_list.append(logits)

                trajectory = agent.generate_soft_tokens(task.prompt)
                soft_trajectories.append(trajectory)

            # Evaluate trajectories: compute task rewards
            rewards = []
            for trajectory, task in zip(soft_trajectories, batch_tasks):
                # Decode soft tokens to text (or evaluate directly)
                response = decode_soft_trajectory(trajectory)
                correctness = evaluate_response(response, task.ground_truth)
                rewards.append(correctness)

            rewards = torch.tensor(rewards, dtype=torch.float32)

            # Compute GRPO loss
            stacked_logits = torch.stack(logits_list)
            loss = compute_grpo_loss(stacked_logits, rewards)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Reward {rewards.mean():.2f}")
```

## Practical Guidance

**When to Use SofT-GRPO:**
- Internal reasoning and chain-of-thought generation (soft tokens are hidden)
- Tasks where continuous optimization benefits model training
- Scenarios with verifiable rewards for RL signal

**When NOT to Use:**
- Tasks requiring discrete token outputs (use discrete GRPO instead)
- Real-time inference requiring low latency (soft tokens add computational overhead)
- Models where discrete outputs are already optimal

**Hyperparameters and Configuration:**
- Temperature (tau): Start at 0.5; decrease to 0.1 during training (sharper tokens)
- Gumbel temperature: 1.0 for initial exploration, decay to 0.5 during training
- KL penalty: 0.01 (keep policy close to pre-training; adjust if divergence is too constrained)
- Batch size: 32+ for stable advantage estimation

**Pitfalls to Avoid:**
1. **Temperature too low** - Gumbel-Softmax becomes too discrete; lose differentiation benefits
2. **Ignoring embedding space** - Soft tokens must project into pre-trained embedding convex hull; validate token distributions
3. **Insufficient baseline** - Without baseline logits for KL penalty, policy can diverge from pre-training
4. **Gradient instability** - Use small learning rates (1e-5 or lower); soft token optimization is more sensitive

---

Reference: https://arxiv.org/abs/2511.06411
