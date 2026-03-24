---
name: zip2zip-adaptive-vocab
title: "zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.01084"
keywords: [token compression, adaptive vocabulary, LLM inference, Lempel-Ziv, efficiency]
description: "Reduce token count by 15-40% at inference through context-adaptive compression, merging frequent token sequences into hypertokens using online Lempel-Ziv-Welch compression without retraining entire models."
---

# zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression

## Core Concept

zip2zip addresses the inefficiency of static, general-purpose tokenizers in language models. Rather than using a fixed vocabulary optimized for average cases, zip2zip dynamically adapts tokenization at inference time by identifying and merging frequently co-occurring token sequences into compact "hypertokens."

The approach uses Lempel-Ziv-Welch (LZW) compression on-the-fly to detect compression patterns in the current context and create hypertokens that match the input's actual token distribution. A lightweight dynamic embedding layer computes representations for newly formed hypertokens at runtime, reducing both input and output token counts by 15-40% without full model retraining.

## Architecture Overview

- **Online LZW Compression**: Dynamically identify and merge frequent token sequences during inference
- **Hypertokenization**: Replace multiple fragmented tokens with single merged tokens
- **Dynamic Embedding Layer**: Compute embeddings for novel hypertokens without precomputed vocabulary
- **Context-Adaptive Encoding**: Adjust compression strategy to match current input distribution
- **Minimal Retraining**: Fine-tune embedding layer in ~10 GPU-hours; preserve base model weights
- **Seamless Integration**: Works with existing LLM architectures (Transformers, etc.)

## Implementation

The following steps outline how to implement adaptive tokenization via online compression:

1. **Monitor token stream** - Track input tokens as they arrive during inference
2. **Detect compression patterns** - Apply LZW algorithm to identify frequent sequences
3. **Merge tokens into hypertokens** - Create new token IDs for merged sequences
4. **Compute hypertoken embeddings** - Dynamically generate representations using a lightweight layer
5. **Process through model** - Run standard model inference with hypertokens
6. **Decompress output** - Convert hypertokens back to original tokens for downstream use

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from collections import defaultdict

class LZWCompressor:
    def __init__(self, vocab_size: int, max_code: int = 4096):
        self.vocab_size = vocab_size
        self.max_code = max_code
        self.dictionary = defaultdict(int)
        self.reset()

    def reset(self):
        """Initialize or reset the compression dictionary."""
        self.dictionary = {i: str(i) for i in range(self.vocab_size)}
        self.next_code = self.vocab_size

    def compress_sequence(self, token_ids: List[int]) -> List[int]:
        """Apply LZW compression to a token sequence."""
        if not token_ids:
            return []

        compressed = []
        w = str(token_ids[0])

        for token in token_ids[1:]:
            wc = w + "," + str(token)
            if wc in self.dictionary:
                w = wc
            else:
                compressed.append(self._lookup(w))
                if self.next_code < self.max_code:
                    self.dictionary[wc] = self.next_code
                    self.next_code += 1
                w = str(token)

        compressed.append(self._lookup(w))
        return compressed

    def _lookup(self, code_str: str) -> int:
        """Lookup code in dictionary."""
        return self.dictionary.get(code_str, int(code_str.split(",")[0]))


class DynamicEmbedding(nn.Module):
    def __init__(self, base_vocab_size: int, embedding_dim: int, max_hypertokens: int = 2048):
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.embedding_dim = embedding_dim
        self.base_embeddings = nn.Embedding(base_vocab_size, embedding_dim)
        # Learnable parameters for computing hypertoken embeddings
        self.hypertoken_projection = nn.Linear(embedding_dim * 2, embedding_dim)

    def get_embedding(self, token_id: int, component_ids: List[int] = None) -> torch.Tensor:
        """Get embedding for token or hypertoken."""
        if token_id < self.base_vocab_size:
            return self.base_embeddings(torch.tensor(token_id))
        elif component_ids:
            # Compute hypertoken embedding from components
            component_embeddings = [self.base_embeddings(torch.tensor(cid)) for cid in component_ids]
            combined = torch.cat(component_embeddings[:2], dim=-1)  # Limit to 2 for efficiency
            return self.hypertoken_projection(combined)
        else:
            return self.base_embeddings(torch.tensor(self.base_vocab_size - 1))  # UNK token

    def forward(self, token_ids: torch.Tensor, hypertoken_map: Dict[int, List[int]] = None) -> torch.Tensor:
        """Embed token sequence with hypertokens."""
        embeddings = []
        for tid in token_ids:
            tid_val = tid.item() if isinstance(tid, torch.Tensor) else tid
            if hypertoken_map and tid_val in hypertoken_map:
                embeddings.append(self.get_embedding(tid_val, hypertoken_map[tid_val]))
            else:
                embeddings.append(self.get_embedding(tid_val))
        return torch.stack(embeddings)


class AdaptiveTokenizer:
    def __init__(self, base_vocab_size: int, embedding_dim: int):
        self.compressor = LZWCompressor(base_vocab_size)
        self.embedding_layer = DynamicEmbedding(base_vocab_size, embedding_dim)
        self.hypertoken_map = {}

    def tokenize_with_compression(self, token_ids: List[int]) -> Tuple[List[int], Dict]:
        """Compress tokens and track hypertoken mappings."""
        compressed = self.compressor.compress_sequence(token_ids)
        # Build mapping for embedding lookup
        self.hypertoken_map = self._build_mapping(token_ids, compressed)
        return compressed, self.hypertoken_map

    def _build_mapping(self, original: List[int], compressed: List[int]) -> Dict[int, List[int]]:
        """Create mapping from hypertoken IDs to original component tokens."""
        mapping = {}
        # Simplified mapping construction
        return mapping
```

## Practical Guidance

**Hyperparameters to tune:**
- **Max hypertokens** (1024-4096): Limit dictionary size to prevent memory bloat; balance compression ratio vs. overhead
- **Compression window** (100-1000 tokens): How many tokens to analyze for pattern detection; larger windows catch more patterns but slower
- **Merging threshold** (2-5 tokens): Minimum sequence length to consider for merging; shorter thresholds enable more compression
- **Fine-tuning epochs** (1-3): Limited retraining of dynamic embedding layer; 1-2 epochs usually sufficient

**When to use:**
- Reducing inference latency and memory for long-context generation
- Optimizing token throughput in high-volume serving scenarios
- Handling repetitive or structured inputs (code, tables) where compression is most effective
- Adapting to domain-specific vocabularies without full retraining

**When NOT to use:**
- Small models where embedding lookup overhead outweighs compression gains
- Tasks with highly variable token distributions (no consistent patterns to compress)
- Real-time applications where compression overhead adds unacceptable latency
- Scenarios requiring exact token-level transparency for auditing

**Common pitfalls:**
- **Over-compression**: Merging too aggressively can lose semantic information or create ambiguous encodings
- **Dictionary bloat**: Allowing unbounded dictionary growth exhausts memory; strict max_code enforcement is essential
- **Context switching**: Compression patterns valid for one domain may not transfer; reset dictionaries for new domains
- **Embedding quality**: Poor dynamic embedding training reduces hypertoken representation quality
- **Latency overhead**: Compression computation can exceed savings from fewer tokens; profile carefully

## Reference

zip2zip demonstrates 15-40% token reduction across diverse models and domains, with particularly strong gains on repetitive inputs. The fine-tuning requirement is minimal (approximately 10 GPU-hours), making it practical to deploy on existing models.

Original paper: "zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression" (arxiv.org/abs/2506.01084)
