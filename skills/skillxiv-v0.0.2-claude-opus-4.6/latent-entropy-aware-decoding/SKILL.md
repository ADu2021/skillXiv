---
name: latent-entropy-aware-decoding
title: "Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.13366"
keywords: [Hallucination Mitigation, Entropy-Aware Decoding, Multimodal Reasoning, Uncertainty Quantification, Continuous Embeddings]
description: "Reduce hallucinations in multimodal reasoning by detecting high-entropy (uncertain) states and switching to continuous latent embeddings instead of discrete tokens. Use prior-guided visual anchoring during uncertain phases to maintain grounding."
---

# Latent Entropy-Aware Decoding: Uncertainty-Guided Multimodal Reasoning

Multimodal large reasoning models (MLRMs) frequently hallucinate during high-uncertainty reasoning phases, particularly at transition points where the model is uncertain about causal structure (words like "because," "however," "wait"). LEAD (Latent Entropy-Aware Decoding) mitigates this by monitoring the entropy of the model's latent representations. During high-uncertainty phases, rather than committing to discrete token selections, the model maintains superposed probability-weighted embeddings capturing multiple competing hypotheses. This allows the model to extract contextual information more reliably and reduces unfounded claims during uncertain reasoning.

The technique is plug-and-play and works with existing MLRMs by modifying the decoding strategy, not the underlying architecture.

## Core Concept

LEAD operates through entropy-aware mode switching:

1. **Entropy Monitoring** — Track entropy of model's hidden states during generation
2. **Mode Detection** — Identify high-uncertainty phases (entropy > threshold)
3. **Representation Switching** — During uncertainty, use continuous superposed embeddings; during certainty, use discrete tokens
4. **Visual Grounding** — Apply prior-guided visual anchoring to reinforce factual grounding during uncertain phases

The key insight: at high-entropy states, the model is uncertain about which discrete token to emit, but the continuous probability distribution contains useful information. By leveraging that distribution, you avoid premature commitment to hallucinated tokens.

## Architecture Overview

- **Entropy Calculator** — Computes per-layer entropy over logits during generation
- **Uncertainty Detector** — Identifies states where entropy exceeds task-specific threshold
- **Continuous Embedding Generator** — Creates weighted sum of token embeddings proportional to token probabilities
- **Visual Anchor Injector** — Reinforces visual-semantic alignment during uncertain phases
- **Context Aggregator** — Maintains latent trajectory across multiple candidate semantics
- **Decoding Pipeline** — Routes between discrete (low entropy) and continuous (high entropy) paths

## Implementation Steps

Begin by computing entropy over model logits and detecting uncertainty phases during generation.

```python
import torch
import torch.nn.functional as F
import numpy as np

class EntropyMonitor:
    """Track entropy in model predictions to detect uncertainty."""

    def __init__(self, entropy_threshold=1.5):
        self.entropy_threshold = entropy_threshold
        self.entropy_history = []

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of logits."""
        # logits: [batch_size, vocab_size]
        probs = F.softmax(logits, dim=-1)

        # Shannon entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy

    def is_uncertain(self, logits: torch.Tensor) -> bool:
        """Check if prediction entropy exceeds threshold."""
        entropy = self.compute_entropy(logits)
        is_high_entropy = entropy > self.entropy_threshold

        self.entropy_history.append(entropy.item())
        return is_high_entropy.item()

    def get_uncertainty_regions(self, logits_sequence: list) -> list:
        """Identify contiguous uncertainty regions in generation."""
        uncertainties = [self.is_uncertain(logits) for logits in logits_sequence]

        regions = []
        start = None
        for i, is_unc in enumerate(uncertainties):
            if is_unc and start is None:
                start = i
            elif not is_unc and start is not None:
                regions.append((start, i))
                start = None
        if start is not None:
            regions.append((start, len(uncertainties)))

        return regions
```

Next, implement continuous embedding generation during high-entropy states.

```python
class ContinuousEmbeddingDecoder:
    """Generate superposed embeddings instead of discrete tokens during uncertainty."""

    def __init__(self, embedding_matrix: torch.Tensor, temperature=0.7):
        self.embedding_matrix = embedding_matrix  # [vocab_size, embed_dim]
        self.temperature = temperature

    def discrete_decode(self, logits: torch.Tensor) -> int:
        """Standard: select token with highest probability."""
        token_id = torch.argmax(logits, dim=-1)
        return token_id.item()

    def continuous_decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Uncertain: create superposed embedding from probability distribution."""
        # Temperature-scaled probabilities
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)

        # Weighted combination of embeddings
        # probs: [vocab_size], embedding_matrix: [vocab_size, embed_dim]
        continuous_embedding = torch.matmul(probs, self.embedding_matrix)
        # [embed_dim]

        return continuous_embedding

    def hybrid_decode(self, logits: torch.Tensor, entropy: float,
                      threshold: float = 1.5) -> torch.Tensor:
        """Choose decoding based on entropy."""
        if entropy > threshold:
            return self.continuous_decode(logits)
        else:
            # For discrete, return embedding of selected token
            token_id = self.discrete_decode(logits)
            return self.embedding_matrix[token_id]
```

Now implement visual grounding that reinforces factual anchoring during uncertain phases.

```python
class VisualAnchor:
    """Inject visual information to ground reasoning during uncertainty."""

    def __init__(self, visual_embed_dim=768, text_embed_dim=768):
        self.visual_embed_dim = visual_embed_dim
        self.text_embed_dim = text_embed_dim

        # Alignment projection
        self.text_to_visual = torch.nn.Linear(text_embed_dim, visual_embed_dim)
        self.visual_to_text = torch.nn.Linear(visual_embed_dim, text_embed_dim)

    def compute_visual_prior(self, visual_embeddings: torch.Tensor,
                            question_embedding: torch.Tensor) -> torch.Tensor:
        """Compute prior distribution over visual regions relevant to question."""
        # visual_embeddings: [num_regions, visual_embed_dim]
        # question_embedding: [text_embed_dim]

        # Project question to visual space
        question_in_visual = self.text_to_visual(question_embedding)

        # Compute relevance scores
        relevance = torch.matmul(visual_embeddings, question_in_visual)
        relevance = F.softmax(relevance, dim=0)

        return relevance

    def inject_visual_anchor(self, text_embedding: torch.Tensor,
                            visual_embeddings: torch.Tensor,
                            visual_prior: torch.Tensor,
                            injection_strength: float = 0.3) -> torch.Tensor:
        """Blend text embedding with weighted visual information."""
        # Aggregate visual embeddings using prior as weights
        aggregated_visual = torch.matmul(visual_prior, visual_embeddings)

        # Project to text space
        visual_in_text = self.visual_to_text(aggregated_visual)

        # Blend: higher injection_strength = more visual influence
        anchored = (1.0 - injection_strength) * text_embedding + \
                   injection_strength * visual_in_text

        return anchored
```

Finally, integrate entropy-aware decoding into the generation loop.

```python
class LatentEntropyAwareDecoder:
    """Full decoding pipeline with entropy-aware mode switching."""

    def __init__(self, model, tokenizer, embedding_matrix, visual_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.entropy_monitor = EntropyMonitor(entropy_threshold=1.5)
        self.continuous_decoder = ContinuousEmbeddingDecoder(embedding_matrix)
        self.visual_anchor = VisualAnchor() if visual_model else None
        self.visual_model = visual_model

    def generate(self, prompt: str, visual_input=None, max_length=256):
        """Generate with entropy-aware mode switching."""
        token_ids = self.tokenizer.encode(prompt)
        generated = []
        embeddings_history = []

        for step in range(max_length):
            # Get model logits
            with torch.no_grad():
                outputs = self.model(torch.tensor([token_ids]))
                logits = outputs.logits[0, -1, :]  # Last token logits

            # Compute entropy
            entropy = self.entropy_monitor.compute_entropy(logits)

            # Choose decoding strategy based on entropy
            if entropy > self.entropy_monitor.entropy_threshold:
                # High entropy: use continuous embedding
                embedding = self.continuous_decoder.continuous_decode(logits)

                # Apply visual grounding if available
                if self.visual_anchor and visual_input is not None:
                    question_embedding = self.model.encode_text(prompt)
                    visual_embeddings = self.visual_model(visual_input)
                    visual_prior = self.visual_anchor.compute_visual_prior(
                        visual_embeddings, question_embedding)

                    embedding = self.visual_anchor.inject_visual_anchor(
                        embedding, visual_embeddings, visual_prior,
                        injection_strength=0.4)

                embeddings_history.append(embedding)

                # For output, select token closest to continuous embedding
                distances = torch.norm(
                    self.continuous_decoder.embedding_matrix - embedding,
                    dim=1)
                next_token = torch.argmin(distances).item()

            else:
                # Low entropy: standard discrete decoding
                next_token = torch.argmax(logits, dim=-1).item()
                embeddings_history.append(
                    self.continuous_decoder.embedding_matrix[next_token])

            generated.append(next_token)
            token_ids.append(next_token)

            # Stopping criteria
            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated)


def evaluate_hallucination_mitigation(model, test_cases, visual_inputs=None):
    """Measure hallucination reduction with LEAD."""
    decoder = LatentEntropyAwareDecoder(model, tokenizer, embedding_matrix,
                                       visual_model=vision_model)

    hallucination_scores = []
    for i, test in enumerate(test_cases):
        visual = visual_inputs[i] if visual_inputs else None

        output = decoder.generate(test['prompt'], visual_input=visual)

        # Score hallucination (factuality vs. reference)
        hallucination_score = compute_factuality(output, test['reference'])
        hallucination_scores.append(hallucination_score)

    avg_factuality = np.mean(hallucination_scores)
    print(f"Average Factuality Score: {avg_factuality:.3f}")
    return hallucination_scores
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Entropy threshold typically 1.0-2.0; lower values trigger continuous mode more frequently, reducing hallucinations but potentially adding noise
- Visual injection strength 0.2-0.5; stronger injection provides better grounding but may reduce model flexibility
- Temperature for continuous embeddings 0.5-1.0; lower temperatures sharpen probabilities, higher values smooth them
- Apply when visual information is available and can ground reasoning
- Most effective for visual QA, multimodal reasoning, and tasks with clear visual-semantic alignment

**When NOT to use:**
- For text-only models without visual grounding; benefits diminish
- When computational budget is tight; continuous embedding generation adds overhead
- For tasks where entropy of correct reasoning is naturally high (open-ended generation)

**Common Pitfalls:**
- Using fixed entropy threshold across all tasks; calibrate per task or use adaptive thresholds
- Visual anchoring without proper alignment learning; train projection matrices on paired visual-semantic data
- Switching modes too aggressively, causing incoherent outputs; use gradual mode transitions with blending
- Embedding matrix misalignment if using quantized or custom vocabularies; ensure consistency with model tokenizer

## Reference

Paper: [Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding](https://arxiv.org/abs/2603.13366)
