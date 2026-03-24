---
name: gnosis-llm-self-awareness
title: "Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2512.20578"
keywords: [LLM Reliability, Self-Awareness, Uncertainty, Internal States, Interpretability]
description: "Enable frozen LLMs to predict their own correctness by decoding signals from internal hidden states and attention patterns, achieving reliable self-verification without external judges—adding only 5M parameters while reducing inference cost and improving calibration."
---

## Overview

Gnosis is a lightweight self-awareness mechanism enabling frozen LLMs to perform intrinsic self-verification by inspecting internal computation states. It addresses a critical problem: LLMs generate confident but incorrect outputs, and detecting failures typically requires expensive external judges or multiple inference passes.

**Core Innovation:** Rather than relying on external verification, extract correctness prediction signals directly from the LLM's internal states—the hidden representations and attention patterns that encode the model's confidence and reasoning process.

## Architecture

### State Extraction Layer

Capture internal states at each layer of the transformer:

**Hidden States:**
- Extract final token's hidden representations from each transformer layer
- Retain information about accumulated computation and reasoning

**Attention Patterns:**
- Track attention weights across heads and layers
- Reveal what model attends to when reasoning
- Indicate confidence through attention concentration

**Multi-Layer Analysis:**
- Early layers encode syntactic/surface information
- Middle layers perform semantic reasoning
- Later layers prepare for output generation
- Different layers provide complementary correctness signals

### State Compression

Compress extracted states into fixed-budget descriptors:

```python
def compress_internal_states(hidden_states, attention_patterns):
    """Compress multi-layer states to fixed-size descriptor."""

    # Extract key statistics from each layer
    descriptors = []
    for layer_idx, (hidden, attn) in enumerate(
        zip(hidden_states, attention_patterns)
    ):
        # Hidden state statistics
        mean_activation = hidden.mean(dim=-1)
        activation_variance = hidden.var(dim=-1)

        # Attention statistics
        attention_entropy = calculate_entropy(attn)
        attention_concentration = attn.max()

        # Temporal statistics (change across tokens)
        activation_change = (hidden[:, -1] - hidden[:, -2]).norm()

        layer_descriptor = {
            "layer": layer_idx,
            "mean_activation": mean_activation,
            "activation_variance": activation_variance,
            "attention_entropy": attention_entropy,
            "attention_concentration": attention_concentration,
            "activation_change": activation_change,
        }
        descriptors.append(layer_descriptor)

    return descriptors
```

### Correctness Prediction Head

Lightweight neural network predicting correctness from compressed states:

```python
class CorrectnessPredictor(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        # Small linear layers for prediction
        self.layers = nn.Sequential(
            nn.Linear(descriptor_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),  # Output: correctness probability [0, 1]
        )

    def forward(self, descriptors):
        """Predict correctness probability."""
        return self.layers(descriptors)
```

**Parameter Count:** ~5M parameters (negligible vs. LLM size)

## Performance Across Domains

**Mathematical Reasoning (MATH, AMC datasets):**
- Accurately predicts correctness on challenging math problems
- Outperforms external judges in both accuracy and calibration
- Works across problem types (algebra, geometry, combinatorics)

**Open-Domain Question Answering (Natural Questions, TriviaQA):**
- Detects hallucinations and factual errors
- Better calibration than text-based confidence measures
- Generalizes to diverse question types

**Academic Knowledge (CollegeBench, STEM Q&A):**
- Identifies knowledge gaps and uncertain reasoning
- Useful for educational applications
- Robust across knowledge domains

**Backbone Model Range:** 1.7B to 20B parameters
- Consistent effectiveness across model scales
- No need for model-specific training

## Key Advantages

**vs. External Judges:**
- No additional inference passes needed
- Negligible computational overhead
- Works with API-only models (observe states, pass to Gnosis)
- Better calibration on most benchmarks

**vs. Text-Based Self-Critique:**
- Observes actual computation, not post-hoc reasoning
- Lower token cost (no generating explanations)
- More reliable on tasks where reasoning is implicit

**vs. Ensemble Methods:**
- Single inference pass (just access internal states)
- Consistent performance across ensemble sizes
- Deterministic verification (not sampling-based)

## Zero-Shot Generalization

A powerful capability: Gnosis trained on one task type generalizes zero-shot to partial generations:

**Partial Generation Detection:**
- Predict correctness before full response is generated
- Enable early stopping when model is failing
- Reduce wasted computation on doomed attempts

**Implementation:**
```python
def early_stopping_inference(prompt, model, gnosis, threshold=0.3):
    """Generate tokens with early stopping on low correctness signal."""

    generated = []
    for step in range(max_steps):
        # Generate next token
        next_token, hidden_states = model.forward_with_states(
            generated, return_internal_states=True
        )
        generated.append(next_token)

        # Predict correctness after K tokens
        if len(generated) > min_generation_length:
            descriptor = compress_states(hidden_states)
            correctness = gnosis(descriptor)

            if correctness < threshold:
                # Restart with different strategy
                return "RESTART"

    return "".join(generated)
```

**Benefit:** Compute-aware control—allocate more computation to promising solution attempts.

## When to Use Gnosis

**Use when:**
- Building reliable LLM applications requiring confidence estimates
- Verifying LLM outputs without external services
- Need to detect hallucinations and failures in closed-book settings
- Optimizing inference compute through early stopping
- Fine-tuning models and needing in-domain correctness prediction

**When NOT to use:**
- Tasks with very simple, deterministic outputs (direct evaluation sufficient)
- Scenarios where external judge cost is acceptable
- Applications where inference overhead is critical (5-10% latency increase)
- Domains with extremely different task distributions than training

## Implementation Considerations

**Training Data Requirements:**
- Pairs of (generation, correctness) labels
- Domain-specific training for best accuracy
- Data from same task distribution as deployment
- Binary correctness labels sufficient (yes/no, correct/incorrect)

**Inference Setup:**
- Requires access to model's internal states
- Straightforward with HuggingFace models (register hooks)
- Works with frozen model weights (no fine-tuning needed)
- ~5-10% inference latency overhead

**Calibration:**
- Probability outputs well-calibrated across domains
- Can adjust threshold for precision/recall tradeoff
- Temperature scaling if additional calibration needed

## Research Contributions

- **Internal State Interpretation:** Correctness signals are intrinsic to generation
- **Efficient Detection:** Lightweight mechanism (5M parameters) for self-awareness
- **Zero-Shot Generalization:** Transfer to partial generations and new domains
- **Comprehensive Evaluation:** Testing across 7+ benchmarks and 4 model scales

## Related Work

**External Verification:**
- Self-Critique, Reflexion (external judges)
- Multi-sample consistency (expensive inference)

**Internal Inspection:**
- Activation analysis and circuit discovery
- Probe-based interpretation (Gnosis is lightweight alternative)

## Code Availability

Full implementation and models available at repository.

**Evaluation Benchmarks:**
- MATH, AMC (mathematical reasoning)
- Natural Questions, TriviaQA (open-domain QA)
- CollegeBench, STEM datasets (academic knowledge)

## References

- Gnosis achieves consistent improvements across 7+ benchmarks
- Works with frozen LLMs (1.7B-20B parameters)
- 5M-parameter overhead with reliable correctness prediction
- Superior calibration vs. external judges and self-critique methods
