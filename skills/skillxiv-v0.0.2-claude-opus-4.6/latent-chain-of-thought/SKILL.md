---
name: latent-chain-of-thought
title: "Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.21358"
keywords: [Latent Reasoning, Planning, Decoupled Verbalization, EMA Aggregation]
description: "Improve reasoning quality by decoupling reasoning from verbalization. Planner generates deterministic latent trajectories while Decoder grounds them to text. Enables dynamic termination, better Pass@k scaling, and interpretable intermediate states."
---

# PLaT: Planning with Latent Thoughts

## Problem
Standard chain-of-thought requires discrete token commitments at each step, limiting solution space exploration. Models must balance reasoning coherence with early commitments that prune future options.

Fixed-length reasoning doesn't adapt to problem difficulty. Interpretability of intermediate reasoning is limited.

## Core Concept
PLaT decouples planning (latent trajectory) from verbalization (decoder). The Planner evolves latent state representations without committing to tokens. The Decoder optionally converts latent states to text for inspection or final answers.

This maintains superposition of multiple reasoning paths longer than token-level approaches, enables dynamic termination, and supports Pass@k exploration.

## Architecture Overview

- **Planner**: Generates deterministic latent trajectories in continuous space
- **Decoder**: Grounds latent plans into text when needed
- **EMA Aggregation**: Stabilizes planning states across reasoning steps
- **Lazy Decoding**: Efficient inference checking only first token for termination
- **Supervised Fine-Tuning**: Aligns latent representations with reasoning steps
- **Decoupled GRPO**: Refines Decoder verbalization while freezing Planner

## Implementation

### Step 1: Build Latent Planner
Create deterministic planning trajectory in continuous embedding space.

```python
class LatentPlanner(nn.Module):
    def __init__(self, hidden_dim=1024, num_latent_steps=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_latent_steps = num_latent_steps
        self.planning_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, input_embedding):
        """Generate planning trajectory from input."""
        latent_states = []
        current_state = input_embedding

        for step in range(self.num_latent_steps):
            # Deterministic evolution of latent state
            next_state = self.planning_mlp(current_state)
            latent_states.append(next_state)
            current_state = next_state

        return latent_states
```

### Step 2: Build Latent-Grounded Decoder
Decode latent states into text when needed.

```python
class LatentDecoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, hidden_dim=1024):
        super().__init__()
        self.projection = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048
        )
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

    def decode_step(self, latent_state):
        """Decode single latent state to next token."""
        projected = self.projection(latent_state)
        hidden = self.decoder(projected)
        logits = self.output_linear(hidden)
        return logits
```

### Step 3: SFT Alignment to Reasoning Steps
Train latent representations to correspond with ground-truth reasoning.

```python
def supervised_latent_training(planner, decoder, reasoning_data, num_epochs=3):
    """Align latent states with supervision."""
    optimizer = torch.optim.AdamW(list(planner.parameters()) + list(decoder.parameters()), lr=1e-4)

    for epoch in range(num_epochs):
        for problem, reasoning_steps, final_answer in reasoning_data:
            # Generate latent trajectory
            input_emb = embed_problem(problem)
            latent_trajectory = planner(input_emb)

            # For each latent state, predict next reasoning token
            for step_idx, latent in enumerate(latent_trajectory):
                if step_idx < len(reasoning_steps):
                    target_tokens = tokenize(reasoning_steps[step_idx])

                    # Decode latent and compute cross-entropy loss
                    logits = decoder.decode_step(latent)
                    loss = F.cross_entropy(logits, target_tokens)
                    loss.backward()

            optimizer.step()
            optimizer.zero_grad()
```

### Step 4: Lazy Decoding and Dynamic Termination
Efficiently determine when reasoning is sufficient.

```python
def lazy_decode_with_termination(planner, decoder, problem, max_latent_steps=32):
    """Generate reasoning with dynamic termination."""
    input_emb = embed_problem(problem)
    latent_states = planner(input_emb)

    response_tokens = []

    for latent_state in latent_states:
        # Lazy decoding: check only first token
        logits = decoder.decode_step(latent_state)
        next_token_probs = F.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(next_token_probs, 1)
        response_tokens.append(next_token)

        # Check termination: end-of-sequence token
        if next_token == END_TOKEN_ID:
            break

    return detokenize(response_tokens)
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Latent dimension | 1024-2048 | Sufficient reasoning capacity |
| Num latent steps | 32-64 | Reasoning horizon |
| EMA decay rate | 0.99 | State aggregation smoothness |
| Supervised training epochs | 2-4 | Alignment convergence |
| GRPO learning rate | 1e-5 to 5e-5 | Decoder refinement rate |

### When to Use

- Mathematical reasoning benchmarks with multiple solution strategies
- Tasks where intermediate steps help exploration (Pass@k optimization)
- Interpretability needs (decode intermediate latents for inspection)
- Models where token-level decisions create dead-ends
- Long-horizon reasoning problems

### When Not to Use

- Fast inference requirements (additional latent computation overhead)
- Tasks requiring immediate text output (only one verbalization needed)
- Simple single-path problems
- Architectures without efficient latent representation support

### Common Pitfalls

1. **Latent-text misalignment**: Lazy decoding may not capture latent evolution. Monitor alignment on validation set.
2. **Early termination biasing**: Dynamic termination benefits from well-calibrated thresholds. Validate distribution.
3. **Insufficient latent capacity**: Hidden dimension too small limits reasoning diversity. Start with 1024+.
4. **Loss imbalance**: EMA aggregation may de-prioritize early states. Track gradient flow across steps.

## Reference
Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization
https://arxiv.org/abs/2601.21358
