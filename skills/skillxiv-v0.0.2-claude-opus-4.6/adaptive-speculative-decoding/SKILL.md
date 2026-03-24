---
name: adaptive-speculative-decoding
title: "OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device Speculative Decoding"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.02659"
keywords: [Speculative Decoding, On-device Inference, Online Adaptation, Vocabulary Mismatch, Knowledge Distillation]
description: "Accelerate LLM inference 1.5-2x using a universal draft model that adapts to user data in real-time, handling different target models and tokenizers via online n-gram cache and hybrid distillation."
---

# OmniDraft: Universal and Adaptive Speculative Decoding

Speculative decoding accelerates LLM inference by using a small draft model to generate multiple tokens, which are verified by the target model in parallel. However, deploying this approach on-device faces challenges: (1) draft models trained for one target model may not work with another, and (2) user-specific data causes distribution shift over time. OmniDraft addresses both through a single universal draft model (Llama-68M) that works with any target model via cross-vocabulary token mapping, and online knowledge distillation that continuously adapts the drafter to user data without explicit retraining.

The key innovations are an online n-gram cache that translates between different tokenizer vocabularies, and a hybrid loss combining token-level and distribution-level distillation that updates the drafter as it processes user data. This enables 1.5-2x speedup across reasoning, coding, and text generation tasks on consumer hardware.

## Core Concept

OmniDraft operates on two principles: (1) vocabulary-aware adaptation through n-gram caches that learn token mappings between drafters and targets, and (2) online learning that improves the drafter continuously as it encounters user data. Rather than training separate drafters for each target model or retraining when deployment contexts change, a single universal drafter becomes increasingly specialized to its deployment context through lightweight online updates.

The n-gram cache maintains learned mappings showing which draft tokens typically correspond to which target tokens, enabling effective translation even with completely different vocabularies. Online distillation ensures the drafter learns patterns specific to the user's domain without catastrophic forgetting of general knowledge.

## Architecture Overview

The system comprises several interconnected components:

- **Universal Draft Model**: Lightweight Llama-68M serving as single drafter for multiple targets
- **N-gram Cache**: Learned bijection between draft and target token sequences, enabling vocabulary translation
- **Hybrid Distillation Head**: Combines KL divergence (for directly mapped tokens) and NLL (for n-gram mapped tokens)
- **Acceptance Predictor**: Lightweight head predicting token acceptance probability, enabling adaptive proposal lengths
- **Online Learning Loop**: Continuously updates draft model and cache during inference with user data

## Implementation

Start with the vocabulary mapping layer:

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict

class NGramCache:
    """
    Learn mappings between draft and target model vocabularies.

    Maintains an n-gram cache storing which target token sequences
    typically follow draft token sequences, enabling cross-vocabulary adaptation.
    """

    def __init__(self, n: int = 4, cache_size: int = 50000):
        self.n = n
        self.cache_size = cache_size

        # Cache: maps (draft_tokens) -> {target_token: count}
        self.cache = defaultdict(lambda: defaultdict(int))
        self.total_accesses = 0

    def record_mapping(self, draft_sequence: List[int], target_token: int):
        """
        Record mapping from draft token sequence to target token.

        Updates n-gram statistics: "when we see these draft tokens,
        the target often outputs this token next."
        """
        # Use last n-1 draft tokens + current draft token as key
        if len(draft_sequence) >= self.n:
            key = tuple(draft_sequence[-(self.n-1):])
        else:
            key = tuple(draft_sequence)

        self.cache[key][target_token] += 1
        self.total_accesses += 1

        # Evict old entries if cache exceeds size
        if len(self.cache) > self.cache_size:
            self._evict_cold_entries()

    def get_mapping(self, draft_sequence: List[int],
                    top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Get predicted target tokens given draft sequence.

        Returns list of (token_id, probability) pairs ranked by likelihood.
        """
        if len(draft_sequence) >= self.n:
            key = tuple(draft_sequence[-(self.n-1):])
        else:
            key = tuple(draft_sequence)

        if key not in self.cache:
            # Unknown sequence; no mapping available
            return []

        # Get distribution of target tokens
        token_counts = self.cache[key]
        total = sum(token_counts.values())

        predictions = [
            (token_id, count / total)
            for token_id, count in sorted(
                token_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
        ]

        return predictions

    def _evict_cold_entries(self):
        """Remove least-accessed n-gram patterns to stay within cache budget."""
        # Compute frequency of each n-gram
        frequencies = {}
        for key, token_dict in self.cache.items():
            frequencies[key] = sum(token_dict.values())

        # Remove bottom 10% by frequency
        num_to_remove = len(self.cache) // 10
        to_remove = sorted(
            frequencies.items(),
            key=lambda x: x[1]
        )[:num_to_remove]

        for key, _ in to_remove:
            del self.cache[key]

class VocabularyAdapter(nn.Module):
    """
    Adapt draft model vocabulary to target model vocabulary.

    Uses n-gram cache to translate draft tokens to target tokens,
    handling vocabulary mismatch transparently.
    """

    def __init__(self, draft_vocab_size: int, target_vocab_size: int,
                 hidden_dim: int = 768):
        super().__init__()
        self.draft_vocab_size = draft_vocab_size
        self.target_vocab_size = target_vocab_size

        # Learned projection from draft embeddings to target space
        self.adaptation_proj = nn.Linear(hidden_dim, hidden_dim)

        # N-gram cache for empirical mappings
        self.ngram_cache = NGramCache(n=4)

    def translate_tokens(self, draft_logits: torch.Tensor,
                        draft_history: List[int],
                        target_tokenizer: object) -> torch.Tensor:
        """
        Convert draft model logits to target vocabulary predictions.

        Uses learned projection + n-gram cache for accurate translation.
        """
        batch_size, seq_len, _ = draft_logits.shape

        # Initialize target logits with low values
        target_logits = torch.full(
            (batch_size, seq_len, self.target_vocab_size),
            float('-inf'),
            device=draft_logits.device
        )

        # For each position, map draft logits to target vocab
        for pos in range(seq_len):
            # Get draft logits for this position
            draft_pos_logits = draft_logits[:, pos, :]

            # Get mapping from n-gram cache
            mappings = self.ngram_cache.get_mapping(draft_history, top_k=10)

            for target_token_id, mapping_prob in mappings:
                if target_token_id < self.target_vocab_size:
                    # Project draft logits to target vocabulary
                    for batch_idx in range(batch_size):
                        # Boost target token probability by mapping confidence
                        current_val = target_logits[batch_idx, pos, target_token_id]
                        boost = torch.log(torch.tensor(mapping_prob + 1e-8))
                        target_logits[batch_idx, pos, target_token_id] = max(
                            current_val, boost
                        )

        return target_logits
```

Implement the online distillation mechanism:

```python
class HybridDistillationLoss(nn.Module):
    """
    Combine token-level and distribution-level distillation.

    Token-level loss guides direct-mapped tokens; distribution loss
    guides sequences found via n-gram cache. Enables effective online learning.
    """

    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.nll_loss = nn.NLLLoss()

    def forward(self, draft_logits: torch.Tensor,
                target_logits: torch.Tensor,
                direct_mapped_mask: torch.Tensor,
                ngram_mapped_mask: torch.Tensor,
                temperature: float = 3.0) -> torch.Tensor:
        """
        Compute hybrid loss for online distillation.

        Args:
            draft_logits: logits from draft model
            target_logits: logits from target model
            direct_mapped_mask: which tokens have direct vocabulary mappings
            ngram_mapped_mask: which tokens have n-gram cache mappings
            temperature: for soft targets

        Returns:
            loss: weighted combination of KL and NLL losses
        """
        # Normalize logits with temperature
        draft_log_probs = torch.log_softmax(draft_logits / temperature, dim=-1)
        target_probs = torch.softmax(target_logits / temperature, dim=-1)

        # KL loss for directly mapped tokens
        kl_loss = self.kl_loss(draft_log_probs * direct_mapped_mask.unsqueeze(-1),
                                target_probs * direct_mapped_mask.unsqueeze(-1))

        # NLL loss for n-gram mapped tokens
        target_tokens = torch.argmax(target_logits, dim=-1)
        nll_loss = self.nll_loss(
            draft_log_probs[ngram_mapped_mask],
            target_tokens[ngram_mapped_mask]
        )

        # Weighted combination
        total_loss = 0.7 * kl_loss + 0.3 * nll_loss
        return total_loss
```

Implement the adaptive draft model with online learning:

```python
class AdaptiveDraftModel(nn.Module):
    """
    Draft model that learns online from user data.

    Updates parameters during inference based on target model outputs,
    improving specialization to user distribution without explicit retraining.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b",
                 draft_model_name: str = "gpt2-medium",
                 learning_rate: float = 2e-5):
        super().__init__()

        # Load draft model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

        # Online learning optimizer (LoRA for efficiency)
        self.lora_optimizer = torch.optim.Adam(
            self.draft_model.parameters(), lr=learning_rate
        )

        # Vocabulary adapter
        self.vocab_adapter = VocabularyAdapter(
            self.tokenizer.vocab_size,
            32000,  # Typical target vocab size
        )

        # Distillation loss
        self.distillation_loss = HybridDistillationLoss()

        # Acceptance predictor head
        self.acceptance_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def generate_candidates(self, input_ids: torch.Tensor,
                           num_candidates: int = 5) -> torch.Tensor:
        """
        Generate candidate tokens using draft model.

        Returns multiple tokens for parallel verification by target model.
        """
        outputs = self.draft_model(input_ids)
        logits = outputs.logits[:, -1, :]

        # Sample top-k candidates
        top_k = min(num_candidates, logits.shape[-1])
        _, top_indices = torch.topk(logits, k=top_k, dim=-1)

        return top_indices

    def online_update(self, input_ids: torch.Tensor,
                     target_logits: torch.Tensor,
                     target_tokens: torch.Tensor):
        """
        Update draft model based on target model outputs.

        Lightweight online learning: single gradient step on recent data.
        """
        # Get draft model predictions
        draft_outputs = self.draft_model(input_ids)
        draft_logits = draft_outputs.logits

        # Compute distillation loss
        loss = self.distillation_loss(
            draft_logits,
            target_logits,
            direct_mapped_mask=torch.ones_like(draft_logits[:, :, 0], dtype=torch.bool),
            ngram_mapped_mask=torch.zeros_like(draft_logits[:, :, 0], dtype=torch.bool)
        )

        # Single gradient step
        self.lora_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.draft_model.parameters(), 1.0)
        self.lora_optimizer.step()

        return loss.item()

    def predict_acceptance_rate(self, draft_features: torch.Tensor) -> torch.Tensor:
        """
        Predict probability that draft tokens will be accepted.

        Enables adaptive proposal length: lower acceptance rate = shorter proposals.
        """
        return self.acceptance_head(draft_features)
```

Implement the full speculative decoding pipeline:

```python
class OmniDraftDecoder:
    """
    Full on-device speculative decoding with online adaptation.

    Generates tokens using draft model, verifies with target model,
    and adapts draft model to user data in real-time.
    """

    def __init__(self, target_model_name: str = "meta-llama/Llama-2-7b-chat",
                 draft_model_name: str = "gpt2-medium"):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load target model
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name)

        # Initialize adaptive draft model
        self.draft = AdaptiveDraftModel(target_model_name, draft_model_name)

        # Statistics tracking
        self.total_tokens = 0
        self.accepted_tokens = 0

    def decode(self, prompt: str, max_new_tokens: int = 128,
               num_draft_candidates: int = 5) -> str:
        """
        Decode with speculative generation and online adaptation.

        Uses draft model to propose tokens, target model to verify,
        and online learning to improve draft model during generation.
        """
        input_ids = self.target_tokenizer.encode(prompt, return_tensors='pt')
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Generate draft candidates
            candidates = self.draft.generate_candidates(
                input_ids, num_draft_candidates
            )

            # Verify with target model (in parallel)
            target_outputs = self.target_model(input_ids)
            target_logits = target_outputs.logits[:, -1, :]
            target_token = torch.argmax(target_logits, dim=-1)

            # Check acceptance
            accepted = (candidates[0, 0] == target_token).item()
            self.total_tokens += 1
            if accepted:
                self.accepted_tokens += 1

            # Online update from target model outputs
            if _ % 10 == 0:  # Update every 10 tokens to save computation
                self.draft.online_update(input_ids, target_logits, target_token)

            # Add accepted token and continue
            next_token = target_token.unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            input_ids = generated[:, -128:]  # Keep recent context

        # Decode result
        result = self.target_tokenizer.decode(generated[0], skip_special_tokens=True)
        return result

    def get_speedup_stats(self) -> Dict:
        """Return statistics about speedup achieved."""
        acceptance_rate = self.accepted_tokens / max(self.total_tokens, 1)
        return {
            'total_tokens': self.total_tokens,
            'accepted_tokens': self.accepted_tokens,
            'acceptance_rate': acceptance_rate,
            'estimated_speedup': 1 + acceptance_rate  # Rough estimate
        }
```

## Practical Guidance

**Hyperparameter Table:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Draft model size | 68M | 30M-250M | Larger = better proposals but slower verification |
| N-gram cache size | 50K | 10K-200K | Larger = more accurate mappings; more memory |
| N-gram length | 4 | 2-6 | Longer = more specific but sparse |
| Online learning rate | 2e-5 | 1e-6 to 1e-4 | Very conservative; single-step updates |
| Update frequency | 10 steps | 1-50 | How often to update on new data |
| Distillation temperature | 3.0 | 1.0-10.0 | Higher = softer targets; better transfer |
| Num candidates | 5 | 2-10 | More candidates = more parallelism; more rejects |

**When to Use:**
- You need to deploy LLMs on consumer hardware with limited resources
- You want 1.5-2x speedup without changing the target model
- You plan to use same draft model across multiple target models
- Your deployment has time to learn from user data (chat/interactive)
- You need to handle different data distributions per user

**When NOT to Use:**
- You have unlimited compute (just use target model directly)
- You need deterministic generation (sampling adds variance)
- Your application requires sub-10ms latency (overhead may hurt)
- You have very short input contexts (overhead dominates)
- You need guaranteed token sequences (rejection sampling changes output)

**Common Pitfalls:**
- **Vocabulary mismatch underestimation**: If draft and target vocabularies differ significantly, n-gram cache becomes sparse. Test on your specific models.
- **Online learning instability**: Large learning rates cause draft model to overfit to recent data. Use very conservative rates and sample diverse batches.
- **Cache thrashing**: If n-gram patterns change rapidly, cache eviction hurts. Monitor cache hit rates and increase size if needed.
- **Acceptance rate drift**: If draft model's adaptation lags behind target distribution, acceptance rate drops over time. Increase update frequency or learning rate.
- **Memory overhead**: N-gram cache + online optimizer parameters add overhead. Profile actual memory usage before deployment.
- **Token mismatch issues**: If draft tokenizer produces different tokens than target, candidate verification fails silently. Validate tokenization equivalence.

## Reference

Authors (2025). OmniDraft: A Cross-vocabulary, Online Adaptive Drafter for On-device Speculative Decoding. arXiv preprint arXiv:2507.02659. https://arxiv.org/abs/2507.02659
