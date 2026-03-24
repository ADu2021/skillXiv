---
name: relay-gen-model-switching
title: "RelayGen: Intra-Generation Model Switching for Efficient Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.06454"
keywords: [Model Composition, Inference Efficiency, Reasoning Decomposition, Speculative Decoding, Cost-Quality Tradeoff]
description: "Reduce inference cost by dynamically switching from large to small LLMs during reasoning generation. Large model handles demanding reasoning phases; small model completes consolidation and answer stages triggered by discourse cues. Achieves 2.2× speedup with minimal accuracy loss."
---

# RelayGen: Discourse-Cued Model Switching for Efficient Generation

Reasoning outputs naturally decompose into phases of varying difficulty. Early reasoning requires substantial computational power to break down problems; later consolidation and answer generation are mechanically simpler. RelayGen exploits this by switching models mid-generation: a large model handles complex reasoning, then hands off to a smaller model when discourse-level cues (Thus, Similarly, Therefore) indicate transition to easier content.

This training-free approach requires only empirically-derived switching heuristics, making it immediately applicable to any large-small model pair without retraining or fine-tuning.

## Core Concept

Standard approach: use large model for entire generation, wasting compute on easy phases.

RelayGen: monitor output discourse markers for phase transitions. When the model generates cues indicating easier content (answer formatting, consolidation), switch to small model. Large model resumes for complex sections if needed.

Key insight: discourse-level cues (words like "Thus," "Therefore," "Here's the answer:") reliably signal difficulty drops, making token-level routing unnecessary.

## Architecture Overview

- **Large Model Path**: Handles demanding reasoning steps requiring full model capacity
- **Small Model Path**: Completes lower-difficulty consolidation and answer phases
- **Discourse Monitoring**: Track output tokens for phase-transition cues
- **Training-Free Switching**: Use empirically-derived heuristic rules (no learned router)
- **Composability**: Works naturally with speculative decoding and other acceleration techniques

## Implementation

Define discourse cues and switching logic:

```python
PHASE_TRANSITION_CUES = {
    'answer': ['answer:', 'solution:', 'result:', 'therefore,', 'thus,', 'finally,'],
    'consolidation': ['in conclusion', 'to summarize', 'simplifying', 'therefore'],
    'calculation': ['{', '}', '=']
}

def should_switch_to_small_model(generated_text, current_phase='reasoning'):
    """Determine if output indicates transition to easier phase."""
    lower_text = generated_text.lower()

    # Check for answer-phase cues
    for cue in PHASE_TRANSITION_CUES['answer']:
        if cue in lower_text:
            return True, 'answer'

    # Check for consolidation cues
    for cue in PHASE_TRANSITION_CUES['consolidation']:
        if cue in lower_text:
            return True, 'consolidation'

    return False, current_phase

def get_last_token_window(generated_text, window_size=50):
    """Get last N characters to check for recent cues."""
    return generated_text[-window_size:]

class RelayGenDecoder:
    """Implements relay switching between large and small models."""

    def __init__(self, large_model, small_model, switch_threshold=0.8, max_tokens=512):
        self.large_model = large_model
        self.small_model = small_model
        self.switch_threshold = switch_threshold
        self.max_tokens = max_tokens

    def generate_with_relay(self, prompt, temperature=0.7):
        """Generate text with dynamic model switching."""
        generated = ""
        current_model = 'large'
        tokens_generated = 0

        while tokens_generated < self.max_tokens:
            # Select model
            model = self.large_model if current_model == 'large' else self.small_model

            # Generate next token
            next_token = model.generate_token(
                prompt + generated,
                temperature=temperature
            )

            if next_token == '<|endoftext|>':  # End token
                break

            generated += next_token
            tokens_generated += 1

            # Check for phase transition every N tokens
            if tokens_generated % 10 == 0:
                recent = get_last_token_window(generated, window_size=100)
                should_switch, new_phase = should_switch_to_small_model(
                    recent,
                    current_phase='answer' if current_model == 'small' else 'reasoning'
                )

                if should_switch and current_model == 'large':
                    print(f"Switching to small model at phase: {new_phase}")
                    current_model = 'small'

        return generated

    def generate_with_speculative_decoding(self, prompt, temperature=0.7, speculation_length=5):
        """RelayGen combined with speculative decoding for further speedup."""
        generated = ""
        current_model = 'large'
        tokens_generated = 0

        while tokens_generated < self.max_tokens:
            if current_model == 'large':
                model = self.large_model
            else:
                # Small model can use speculative decoding for additional acceleration
                model = self.small_model

            # Generate multiple tokens speculatively with small model
            speculative_tokens = model.generate_tokens(
                prompt + generated,
                num_tokens=speculation_length if current_model == 'small' else 1,
                temperature=temperature
            )

            for spec_token in speculative_tokens:
                generated += spec_token
                tokens_generated += 1

                # Check switching condition every 5 tokens
                if tokens_generated % 5 == 0:
                    recent = get_last_token_window(generated, window_size=100)
                    should_switch, new_phase = should_switch_to_small_model(recent)

                    if should_switch and current_model == 'large':
                        current_model = 'small'
                        break  # Switch immediately

                if spec_token == '<|endoftext|>':
                    return generated

        return generated
```

Evaluate efficiency gains:

```python
import time

def benchmark_relay_generation(large_model, small_model, prompts, num_runs=10):
    """Measure speedup and accuracy trade-offs."""
    relay_decoder = RelayGenDecoder(large_model, small_model)

    # Baseline: large model only
    large_start = time.time()
    for _ in range(num_runs):
        large_output = large_model.generate(prompts[0], max_tokens=200)
    large_time = time.time() - large_start

    # RelayGen: with switching
    relay_start = time.time()
    for _ in range(num_runs):
        relay_output = relay_decoder.generate_with_relay(prompts[0], max_tokens=200)
    relay_time = time.time() - relay_start

    speedup = large_time / relay_time
    accuracy_match = evaluate_semantic_similarity(large_output, relay_output)

    print(f"Speedup: {speedup:.2f}×")
    print(f"Accuracy match: {accuracy_match:.1%}")
    print(f"Time: Large={large_time:.2f}s, RelayGen={relay_time:.2f}s")

    return speedup, accuracy_match
```

## Practical Guidance

| Parameter | Recommendation | Notes |
|-----------|-----------------|-------|
| Switch timing | Every 10-20 tokens checked | Balance responsiveness with overhead. |
| Small/Large ratio | 1:5 to 1:10 (params) | Larger gap increases speedup; may hurt quality. |
| Phase transition window | 50-100 tokens | Look back this far for phase-change cues. |
| Initial phase | Always 'reasoning' | Start with large model for robustness. |
| Temperature | 0.7-0.9 | Match generation hyperparameters. |

**When to Use**
- Inference cost is primary constraint
- You have pairs of models with known size/capability ratios
- Reasoning outputs follow structured phase decomposition
- Combining with other speedup techniques (speculative decoding, quantization)

**When NOT to Use**
- Model quality must be consistent throughout (small model could hurt coherence)
- Domains without clear phase structure
- Single-turn simple queries (switching overhead not worth it)

**Common Pitfalls**
- Switching cues too strict (never trigger); use broader keyword sets
- Not testing model pair compatibility; ensure small model handles answer phases well
- Assuming all reasoning has same structure; customize cues per domain
- Switching multiple times; usually one switch per generation is optimal

## Reference

See https://arxiv.org/abs/2602.06454 for empirical validation on reasoning benchmarks, detailed discourse cue analysis, and compatibility with speculative decoding and other acceleration methods.
