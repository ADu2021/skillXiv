---
name: tiadar-diffusion-autoregressive-hybrid
title: "TiDAR: Think in Diffusion, Talk in Autoregression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.08923"
keywords: [Hybrid Architecture, Diffusion Models, Autoregressive Language, Parallel Generation, Token Throughput]
description: "Combine diffusion-based parallel drafting with autoregressive sampling in a single forward pass using structured attention masks—achieving 5x+ token throughput while maintaining autoregressive-level output quality through hybrid generation."
---

# Hybrid Diffusion-Autoregressive Generation for High-Throughput Language Models

Autoregressive language models are high-quality but slow (one token per forward pass). Diffusion models generate tokens in parallel but with lower quality. TiDAR merges both paradigms: diffusion generates candidate tokens in parallel (drafting phase), then autoregression selects final outputs sequentially (refinement phase)—all in a single forward pass using structured attention.

The approach achieves 4.71x to 5.91x tokens per second compared to pure autoregression while maintaining comparable quality, solving a fundamental speed-quality tradeoff.

## Core Concept

TiDAR operates in two phases within one neural forward pass:

1. **Thinking (Diffusion)** - Parallel iterative refinement generates k candidate tokens for each position
2. **Talking (Autoregression)** - Sequential sampling selects final tokens using context and diffusion candidates

Structured attention masks enable this hybrid within a single transformer: diffusion layers have all-to-all connectivity (parallel thinking), while autoregressive layers have causal masks (sequential talking). The architecture transitions smoothly between thinking and talking phases.

## Architecture Overview

- **Diffusion Thinking Layers**: Parallel token generation with iterative refinement
- **Structured Attention Masks**: All-to-all for diffusion; causal for autoregression
- **Candidate Representation**: Stores k candidate tokens per position for AR selection
- **Autoregressive Refinement**: Sequential sampling from diffusion candidates
- **Hybrid Router**: Decides when to transition from diffusion to autoregressive phase
- **Efficient Masking**: Single forward pass enables gradient flow through both paradigms

## Implementation Steps

**Step 1: Diffusion-Based Candidate Generation**

Generate multiple token candidates per position through iterative diffusion.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionTokenGenerator(nn.Module):
    """
    Generates token candidates through diffusion-style iterative refinement.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_candidates: int = 8,
                 num_diffusion_steps: int = 4):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_candidates: Number of candidate tokens per position
            num_diffusion_steps: Refinement iterations
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_candidates = num_candidates
        self.num_diffusion_steps = num_diffusion_steps

        # Learnable noise scheduler
        self.noise_schedule = nn.Parameter(
            torch.linspace(1.0, 0.0, num_diffusion_steps)
        )

        # Refinement layers
        self.refinement_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_diffusion_steps)
        ])

    def generate_candidates(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Generate candidate tokens through diffusion.

        Args:
            hidden_states: Model hidden states [batch_size, seq_len, embed_dim]

        Returns:
            candidates: Candidate logits [batch_size, seq_len, num_candidates, vocab_size]
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        device = hidden_states.device

        # Initialize candidates with noise
        # Start from uniform random; refine toward true distribution
        candidates = torch.randn(
            batch_size, seq_len, self.num_candidates, embed_dim,
            device=device
        )

        # Iterative refinement (diffusion steps)
        for step in range(self.num_diffusion_steps):
            noise_level = self.noise_schedule[step]

            # Refine candidates with context
            # Attend to hidden states to bias candidates toward relevant tokens
            refined = candidates + hidden_states.unsqueeze(2)  # Add context bias
            refined = self.refinement_layers[step](refined)

            # Gradually remove noise
            candidates = refined * (1 - noise_level) + \
                        torch.randn_like(refined) * noise_level

        # Project to logits
        # Map embedding space to vocabulary
        logits = torch.matmul(
            candidates,
            torch.randn(embed_dim, self.vocab_size, device=device)
        )

        return logits
```

**Step 2: Structured Attention Masks**

Create attention patterns enabling diffusion (all-to-all) and autoregression (causal) in one forward pass.

```python
def create_hybrid_attention_mask(batch_size: int, seq_len: int, num_candidates: int,
                                 diffusion_layers: int, ar_layers: int,
                                 device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Create structured attention masks for hybrid architecture.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_candidates: Number of candidates per position
        diffusion_layers: Number of diffusion (all-to-all) layers
        ar_layers: Number of autoregressive (causal) layers
        device: torch device

    Returns:
        masks: {diffusion_mask, ar_mask, candidate_mask}
    """
    # Diffusion mask: all-to-all connectivity (thinking phase)
    # Every position can attend to every other position
    diffusion_mask = torch.ones(
        batch_size, seq_len, seq_len,
        device=device, dtype=torch.bool
    )

    # Autoregressive mask: causal (talking phase)
    # Each position attends to itself and previous positions only
    ar_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    ).unsqueeze(0).expand(batch_size, -1, -1)

    # Candidate mask: connections between AR positions and diffusion candidates
    # AR layer can attend to all candidate positions from previous step
    candidate_mask = torch.ones(
        batch_size, seq_len, num_candidates, seq_len,
        device=device, dtype=torch.bool
    )

    return {
        'diffusion_mask': diffusion_mask,
        'ar_mask': ar_mask,
        'candidate_mask': candidate_mask
    }

class HybridAttentionLayer(nn.Module):
    """
    Single attention layer supporting both diffusion and AR patterns.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                phase: str = 'diffusion') -> torch.Tensor:
        """
        Apply hybrid attention.

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            attention_mask: Attention mask (diffusion or AR)
            phase: 'diffusion' or 'autoregressive'

        Returns:
            output: Attended states [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = hidden_states.shape

        # Compute Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask
        if attention_mask is not None:
            # Mask shape: [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(~attention_mask.unsqueeze(1), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.output(attn_output)

        return output
```

**Step 3: Autoregressive Refinement from Candidates**

Select final tokens from diffusion candidates using autoregressive sampling.

```python
class AutoregressiveRefinement(nn.Module):
    """
    Refines diffusion candidates through autoregressive sampling.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Selection network: learns to pick best candidate
        self.selector = nn.Sequential(
            nn.Linear(embed_dim + vocab_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def select_tokens(self, candidates_logits: torch.Tensor,
                     context_hidden: torch.Tensor) -> torch.Tensor:
        """
        Select best candidate tokens given context.

        Args:
            candidates_logits: [batch, seq_len, num_candidates, vocab_size]
            context_hidden: [batch, seq_len, embed_dim]

        Returns:
            selected_tokens: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len, num_candidates, vocab_size = candidates_logits.shape

        # Compute candidate probabilities
        candidate_probs = F.softmax(candidates_logits, dim=-1)

        # Score each candidate position
        scores = []
        for c in range(num_candidates):
            # Get probabilities for this candidate set
            cand_probs = candidate_probs[:, :, c, :]

            # Combine with context
            combined = torch.cat([
                context_hidden,
                cand_probs
            ], dim=-1)

            # Compute selection score
            score = self.selector(combined).squeeze(-1)
            scores.append(score)

        scores = torch.stack(scores, dim=-1)  # [batch, seq_len, num_candidates]

        # Select highest-scoring candidate per position
        selected_idx = torch.argmax(scores, dim=-1)  # [batch, seq_len]

        # Gather selected logits
        selected_logits = torch.gather(
            candidates_logits,
            2,
            selected_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, vocab_size)
        ).squeeze(2)

        return selected_logits
```

**Step 4: Unified Forward Pass**

Combine diffusion thinking and AR talking into single forward pass.

```python
class TiDARModel(nn.Module):
    """
    Unified model combining diffusion thinking with autoregressive talking.
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_candidates: int = 8,
                 num_diffusion_layers: int = 6, num_ar_layers: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_candidates = num_candidates

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Diffusion thinking layers
        self.diffusion_layers = nn.ModuleList([
            HybridAttentionLayer(embed_dim) for _ in range(num_diffusion_layers)
        ])
        self.diffusion_generator = DiffusionTokenGenerator(
            vocab_size, embed_dim, num_candidates
        )

        # Autoregressive talking layers
        self.ar_layers = nn.ModuleList([
            HybridAttentionLayer(embed_dim) for _ in range(num_ar_layers)
        ])
        self.ar_refinement = AutoregressiveRefinement(vocab_size, embed_dim)

        # Output projection
        self.to_logits = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate tokens through hybrid diffusion-autoregressive process.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed input
        hidden = self.embedding(input_ids)

        # PHASE 1: Diffusion thinking (all-to-all connectivity)
        diffusion_mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

        for layer in self.diffusion_layers:
            hidden = layer(hidden, diffusion_mask, phase='diffusion')
            hidden = F.relu(hidden)

        # Generate candidate tokens
        candidates_logits = self.diffusion_generator.generate_candidates(hidden)

        # PHASE 2: Autoregressive talking (causal mask)
        ar_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

        for layer in self.ar_layers:
            hidden = layer(hidden, ar_mask, phase='autoregressive')
            hidden = F.relu(hidden)

        # Refine candidates through AR
        final_logits = self.ar_refinement.select_tokens(candidates_logits, hidden)

        return final_logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        """
        Generate text token-by-token using hybrid approach.

        Args:
            input_ids: [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate

        Returns:
            output_ids: [batch_size, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Forward pass through hybrid model
            logits = self.forward(input_ids)

            # Sample next token from last position
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
```

## Practical Guidance

**When to Use TiDAR:**
- High-throughput inference scenarios (5x+ speedup valuable)
- Scenarios tolerating slight quality reduction for speed
- Long-context generation (diffusion thinking handles global context)

**When NOT to Use:**
- Tasks requiring maximum output quality (pure AR still slightly better)
- Real-time low-latency requirements (diffusion layers add latency)
- Fine-grained token-level control (hybrid approach less interpretable)

**Hyperparameters and Configuration:**
- Number of candidates: 4-8 (tradeoff between diversity and memory)
- Diffusion steps: 2-4 (more steps = better candidates but slower)
- Diffusion vs AR layer ratio: Equal works well (6D+6AR); adjust if needed
- Temperature for candidate selection: 1.0 (deterministic); increase for diversity

**Pitfalls to Avoid:**
1. **Candidate redundancy** - If candidates are too similar, AR selection provides no benefit; increase diffusion steps
2. **Attention mask errors** - Mask shapes must align with attention dimensions; test carefully
3. **Gradient flow issues** - Ensure gradients flow through diffusion candidates into refinement
4. **Memory overhead** - num_candidates increases memory; monitor for OOM

---

Reference: https://arxiv.org/abs/2511.08923
