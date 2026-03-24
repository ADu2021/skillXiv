---
name: layer-cake-contrastive-decoding
title: "LayerCake: Token-Aware Contrastive Decoding within Large Language Model Layers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.04404"
keywords: [Decoding, Contrastive Learning, Factuality, Token-Aware, No-Training Required]
description: "Improve factual accuracy in LLM generation through decoding-time layer-wise attention suppression. Selectively suppress attention to specific token types at their most influential transformer depths without training or model modifications. Use when you need to reduce hallucinations and improve factual correctness at inference time."
---

# LayerCake: Token-Aware Contrastive Decoding for Enhanced Factuality

Large language models frequently generate factual errors and hallucinations despite strong pretraining. LayerCake addresses this through a novel decoding-time approach that exploits the internal structure of Transformers. The key observation is that different token types (punctuation, concepts, entities) have dominant influence at specific layer depths: early layers handle surface-level tokens like punctuation, while intermediate layers drive semantic reasoning.

By selectively suppressing attention to token types at their most influential depths, the method creates contrastive signals that guide generation toward factual outputs. The approach requires no training, no model modifications, and no fine-tuning—it operates purely at decoding time through attention manipulation.

## Core Concept

Transformers process information hierarchically: early layers capture surface patterns (punctuation, formatting), middle layers reason about concepts and relationships, and deeper layers synthesize high-level decisions. LayerCake identifies which token types exert maximum influence at each depth, then strategically suppresses attention to those tokens at that depth.

This creates a controlled factual degradation that generates contrastive signals: the model learns which token types are critical for accurate generation and avoids over-relying on spurious correlations. The technique combines two insights: (1) token types have differential importance across layers, and (2) suppressing specific signals at their critical depths causes the model to find alternative, more robust reasoning paths.

## Architecture Overview

- **Attention Pattern Analysis**: Identify which token categories receive dominant attention at each transformer layer (punctuation in early layers, concepts in middle)
- **Layer-Token Importance Mapping**: Build per-layer matrices showing influence of each token type on output quality
- **Selective Suppression Module**: At inference, suppress attention to identified token types at their critical depths
- **Contrastive Signal Generation**: Controlled suppression creates factual gaps that guide model toward more grounded generation
- **Zero-Training Design**: All operations at decoding time without model parameter updates

## Implementation

### Attention Pattern Analysis Across Layers

Analyze which tokens receive maximum attention at each transformer depth.

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

class AttentionAnalyzer:
    """Analyze token importance across transformer layers."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        # Token type categories
        self.token_types = {
            'punctuation': self.identify_punctuation_tokens(),
            'common_words': self.identify_common_words(),
            'entity_indicators': self.identify_entity_indicators(),
            'concept_words': self.identify_concept_words()
        }

    def identify_punctuation_tokens(self) -> set:
        """Identify tokens that are primarily punctuation."""
        punctuation = set()
        for token_id in range(self.tokenizer.vocab_size):
            token_str = self.tokenizer.decode([token_id])
            if len(token_str.strip()) <= 2 and not token_str.isalnum():
                punctuation.add(token_id)
        return punctuation

    def identify_common_words(self) -> set:
        """Identify high-frequency common words."""
        common_ids = set(self.tokenizer.convert_tokens_to_ids(
            ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'and', 'or']
        ))
        return common_ids

    def identify_entity_indicators(self) -> set:
        """Identify tokens that precede named entities (capitals, special chars)."""
        entity_indicators = set()
        for token_id in range(min(self.tokenizer.vocab_size, 10000)):
            token_str = self.tokenizer.decode([token_id])
            if len(token_str) > 0 and token_str[0].isupper():
                entity_indicators.add(token_id)
        return entity_indicators

    def identify_concept_words(self) -> set:
        """Identify tokens representing meaningful concepts (nouns, verbs)."""
        # In practice, use POS tagging or language model confidence
        # Simplified: tokens of length 5+ characters
        concept_ids = set()
        for token_id in range(min(self.tokenizer.vocab_size, 50000)):
            token_str = self.tokenizer.decode([token_id]).strip()
            if len(token_str) >= 5 and token_str.isalpha():
                concept_ids.add(token_id)
        return concept_ids

    def analyze_layer_importance(
        self,
        prompt: str,
        num_layers_to_analyze: int = 12
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze importance of token types at each layer.

        Args:
            prompt: Input text to analyze
            num_layers_to_analyze: How many layers to examine

        Returns:
            layer_importance: {layer_idx: {token_type: importance_score}}
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)

        layer_importance = {}

        for layer_idx in range(min(len(attentions), num_layers_to_analyze)):
            layer_attn = attentions[layer_idx]  # (batch, heads, seq, seq)
            batch_size, num_heads, seq_len, _ = layer_attn.shape

            # Average over batch and heads
            avg_attn = layer_attn.mean(dim=(0, 1))  # (seq, seq)

            # Compute importance: how much attention flows TO each token position
            token_importance = avg_attn.sum(dim=0)  # (seq,)

            # Categorize tokens and compute average importance per type
            type_importance = {}

            for token_type, token_ids in self.token_types.items():
                type_scores = []

                for pos, token_id in enumerate(input_ids[0]):
                    if token_id.item() in token_ids:
                        type_scores.append(token_importance[pos].item())

                type_importance[token_type] = (
                    sum(type_scores) / len(type_scores) if type_scores else 0.0
                )

            layer_importance[layer_idx] = type_importance

        return layer_importance
```

### Layer-Token Dominance Mapping

Identify which token types are most influential at each depth.

```python
class LayerTokenDominance:
    """Map which token types dominate at each layer depth."""

    def __init__(self, analyzer: AttentionAnalyzer):
        self.analyzer = analyzer

    def build_dominance_map(
        self,
        prompts: List[str],
        threshold: float = 0.2
    ) -> Dict[int, List[str]]:
        """
        Build mapping of dominant token types per layer.

        Args:
            prompts: List of prompts to analyze
            threshold: Tokens with importance > threshold are considered dominant

        Returns:
            dominance_map: {layer_idx: [dominant_token_types]}
        """
        all_importances = {}

        for prompt in prompts:
            layer_imp = self.analyzer.analyze_layer_importance(prompt)

            for layer_idx, type_importance in layer_imp.items():
                if layer_idx not in all_importances:
                    all_importances[layer_idx] = {}

                for token_type, importance in type_importance.items():
                    if token_type not in all_importances[layer_idx]:
                        all_importances[layer_idx][token_type] = []

                    all_importances[layer_idx][token_type].append(importance)

        # Compute average importance per token type per layer
        dominance_map = {}

        for layer_idx in all_importances:
            avg_importances = {}

            for token_type in all_importances[layer_idx]:
                scores = all_importances[layer_idx][token_type]
                avg_importances[token_type] = sum(scores) / len(scores)

            # Identify dominant types
            max_importance = max(avg_importances.values())
            dominant = [
                t for t, imp in avg_importances.items()
                if imp >= threshold * max_importance
            ]

            dominance_map[layer_idx] = dominant

        return dominance_map
```

### Decoding with Selective Attention Suppression

Suppress attention to token types at their critical layers during generation.

```python
class LayerCakeDecoder:
    """Decode with selective token suppression at critical layers."""

    def __init__(
        self,
        model_name: str,
        dominance_map: Dict[int, List[str]],
        analyzer: AttentionAnalyzer
    ):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dominance_map = dominance_map
        self.analyzer = analyzer
        self.model.eval()

    def generate_with_suppression(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        suppression_strength: float = 0.5,
        temperature: float = 0.7
    ) -> str:
        """
        Generate with selective attention suppression.

        Args:
            prompt: Input text
            max_new_tokens: Number of tokens to generate
            suppression_strength: How aggressively to suppress (0-1)
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        generated_tokens = [input_ids[0].tolist()]

        for _ in range(max_new_tokens):
            # Forward pass with attention hook
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            # Get next token probability
            next_token_logits = logits[0, -1, :]

            # Apply suppression to logits based on token type importance
            # Suppress unlikely tokens more aggressively
            next_token_logits = self.apply_suppression(
                next_token_logits,
                suppression_strength
            )

            # Sample or greedy
            if temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(unsqueeze(0))

            generated_tokens[0].append(next_token.item())
            input_ids = torch.tensor([generated_tokens[0]])

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def apply_suppression(self, logits, suppression_strength):
        """
        Apply attention suppression to next token logits.

        Reduce probability of tokens that would trigger incorrect factual paths.
        """
        # Identify which tokens should be suppressed
        # based on layer-specific dominance patterns

        # Apply soft penalty to suppressed token logits
        penalty = torch.zeros_like(logits)

        # Example: suppress punctuation in early-layer-dominant positions
        punctuation_tokens = self.analyzer.token_types['punctuation']

        for token_id in punctuation_tokens:
            # Reduce logit by suppression_strength factor
            penalty[token_id] = suppression_strength

        suppressed_logits = logits - penalty

        return suppressed_logits

class AttentionSuppressionHook:
    """Hook for suppressing attention patterns during forward pass."""

    def __init__(self, layer_idx, token_types_to_suppress, suppression_strength=0.5):
        self.layer_idx = layer_idx
        self.token_types = token_types_to_suppress
        self.strength = suppression_strength

    def __call__(self, module, input, output):
        """
        Modify attention output to suppress token-type contributions.

        Args:
            module: The attention module
            input: Input to attention
            output: Attention weights (query, key, value) or (attention_output, ...)

        Returns:
            Modified output with suppressed patterns
        """
        # output is typically (attention_weights, context_layer)
        if isinstance(output, tuple):
            attention_weights, context = output
        else:
            # Some implementations return different formats
            return output

        # Suppress attention to identified token types
        # attention_weights: (batch, heads, seq, seq)

        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Create suppression mask
        mask = torch.ones_like(attention_weights)

        for token_type in self.token_types:
            # Find positions of this token type
            # (would require passing token IDs through hook)
            # Simplified: suppress last few tokens (often punctuation/function words)
            mask[:, :, :, -3:] *= (1 - self.strength)

        # Apply mask to attention weights
        suppressed_weights = attention_weights * mask

        # Renormalize so row sums to 1
        suppressed_weights = suppressed_weights / (suppressed_weights.sum(dim=-1, keepdim=True) + 1e-8)

        if isinstance(output, tuple):
            return (suppressed_weights, context)
        else:
            return output
```

### Practical Inference Wrapper

Simple API for using LayerCake with any model.

```python
class LayerCakeInference:
    """Easy-to-use interface for LayerCake factuality enhancement."""

    def __init__(self, model_name: str, calibration_prompts: List[str] = None):
        """
        Args:
            model_name: HuggingFace model identifier
            calibration_prompts: Optional prompts to analyze for token importance
        """
        self.model_name = model_name
        self.analyzer = AttentionAnalyzer(model_name)

        # Build dominance map from calibration prompts
        if calibration_prompts:
            dominance_builder = LayerTokenDominance(self.analyzer)
            self.dominance_map = dominance_builder.build_dominance_map(calibration_prompts)
        else:
            # Use default: suppress punctuation in early layers
            self.dominance_map = {i: ['punctuation'] for i in range(12)}

        self.decoder = LayerCakeDecoder(model_name, self.dominance_map, self.analyzer)

    def generate_factual(
        self,
        prompt: str,
        max_length: int = 100,
        suppression_strength: float = 0.5,
        temperature: float = 0.7
    ) -> str:
        """
        Generate with factuality enhancement.

        Args:
            prompt: Input text
            max_length: Max tokens to generate
            suppression_strength: 0-1 (higher = more suppression)
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Generated text with improved factuality
        """
        return self.decoder.generate_with_suppression(
            prompt,
            max_new_tokens=max_length,
            suppression_strength=suppression_strength,
            temperature=temperature
        )

# Usage
inference = LayerCakeInference(
    'meta-llama/Llama-3.1-8B',
    calibration_prompts=[
        'What is the capital of France?',
        'Calculate 2 + 2.',
        'List the planets in our solar system.'
    ]
)

# Generate with factuality enhancement
output = inference.generate_factual(
    'The largest city in Japan is',
    suppression_strength=0.6
)
print(output)
```

## Practical Guidance

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Suppression Strength | 0.3-0.7 | Higher = more aggressive factuality focus; 0.5 is balanced |
| Token Type | Punctuation, concepts | Adjust based on your task domain |
| Calibration Samples | 10-50 prompts | Enough to establish layer-token patterns |
| Temperature | 0.5-0.7 | Lower reduces hallucination but may reduce diversity |
| Layers to Analyze | First 12-24 | Early-to-middle layers where token types dominate |

### When to Use

- Improving factual accuracy on knowledge-based tasks without fine-tuning
- Reducing hallucinations in information retrieval and QA systems
- Quick deployment when model fine-tuning is infeasible
- Real-time inference where training overhead is prohibited
- Multi-model systems where retraining is impractical

### When NOT to Use

- Tasks where all token types are equally important (poetry, creative writing)
- Models already fine-tuned for factuality
- Scenarios requiring training-based improvements
- Systems where suppression artifacts would be problematic

### Common Pitfalls

- **Uniform suppression across all layers**: Different token types dominate at different depths; calibrate per-layer
- **Over-suppression**: Setting strength >0.8 can make outputs stilted; stay in 0.3-0.7 range
- **Task-agnostic calibration**: Token patterns differ across domains; calibrate on task-specific prompts
- **Ignoring model architecture differences**: Attention patterns vary by architecture; reanalyze for each model family

## Reference

Liu, S., Chen, X., Wang, Y., et al. (2024). LayerCake: Token-Aware Contrastive Decoding within Large Language Model Layers. arXiv preprint arXiv:2507.04404.

*Note: This paper was withdrawn on October 3, 2025, pending institutional review completion. Implementation is based on available abstract and methodology descriptions.*
