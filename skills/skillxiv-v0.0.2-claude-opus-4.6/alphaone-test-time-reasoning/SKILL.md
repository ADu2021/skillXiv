---
name: alphaone-test-time-reasoning
title: "AlphaOne: Reasoning Models Thinking Slow and Fast at Test Time"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24863"
keywords: [Test-Time Computation, Reasoning, Token Prediction, Inference Optimization]
description: "Dynamically modulate reasoning depth at test time using alpha moments and Bernoulli scheduling to optimize inference speed-quality tradeoffs without retraining."
---

# Optimize Reasoning Compute Dynamically at Test Time

AlphaOne introduces a universal framework for controlling how much a reasoning model thinks before generating answers. Rather than forcing a fixed amount of reasoning computation, it learns to schedule thinking tokens based on problem difficulty—generating more internal reasoning for hard problems and fewer for easy ones. This adaptive approach maintains quality while reducing computational waste.

The core innovation is the alpha moment: a unified parameter that scales the entire internal thinking phase. By modeling reasoning token insertion as a stochastic process, AlphaOne can dynamically interpolate between "fast thinking" (minimal internal computation) and "slow thinking" (deep reasoning) at inference time, without requiring model retraining.

## Core Concept

AlphaOne decouples thinking from generation through three key mechanisms:

- **Alpha moment (α)**: A universal scalar parameter controlling pre-response reasoning intensity
- **Bernoulli scheduling**: Models token insertion as a stochastic process rather than fixed sequences
- **Post-α deterministic generation**: Fast, confident generation once reasoning is complete
- **Test-time control**: Adjust α at inference without model changes

The insight is that reasoning models contain both "slow thinking" (internal reasoning tokens) and "fast thinking" (direct generation). By controlling when to transition from thinking to generation, you optimize the speed-quality frontier dynamically.

## Architecture Overview

- **Reasoning token mechanism**: Model emits special tokens during internal reasoning phase
- **Scheduling policy**: Bernoulli process determines whether each reasoning step continues or transitions to generation
- **Alpha parameter**: Scales probability of continuing reasoning (higher α = more thinking)
- **Pre-α phase**: Scaled-down reasoning with uniform probability control
- **Post-α phase**: Deterministic, high-confidence generation with no internal reasoning
- **Inference controller**: Runtime system to select α value per input

## Implementation

This implementation shows how to add dynamic reasoning scheduling to a standard language model:

```python
# AlphaOne: Dynamic test-time reasoning scheduling
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class AlphaOneReasoner:
    def __init__(self, model_name, reasoning_token_id=None):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Typically a special token like <reasoning> or <think>
        self.reasoning_token_id = reasoning_token_id or self.tokenizer.eos_token_id

    def generate_with_alpha(self, prompt, alpha=0.5, max_reasoning_tokens=200):
        """
        Generate response with controlled reasoning depth via alpha parameter.
        alpha: 0.0 = fast thinking only, 1.0 = maximum reasoning
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        device = input_ids.device

        # Phase 1: Pre-alpha reasoning with scaled probability
        reasoning_phase = True
        reasoning_tokens = []
        current_ids = input_ids.clone()
        step = 0

        while reasoning_phase and step < max_reasoning_tokens:
            # Get model logits for next token
            with torch.no_grad():
                outputs = self.model(current_ids)
                next_logits = outputs.logits[:, -1, :]

            # Sample from model distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Bernoulli decision: continue reasoning or transition?
            # P(continue_reasoning) = alpha
            if next_token.item() == self.reasoning_token_id:
                # Explicit reasoning token
                continue_prob = alpha
            else:
                # Content token during reasoning phase
                continue_prob = alpha * 0.8  # Slightly reduced for content tokens

            should_continue = torch.bernoulli(torch.tensor(continue_prob)).item()

            if should_continue and step < max_reasoning_tokens:
                reasoning_tokens.append(next_token.item())
                current_ids = torch.cat([current_ids, next_token], dim=1)
                step += 1
            else:
                reasoning_phase = False

        # Phase 2: Post-alpha deterministic generation
        # Generate final response with greedy decoding (high confidence)
        max_new_tokens = 256
        response_ids = current_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(response_ids)
                next_logits = outputs.logits[:, -1, :]

            # Greedy selection (deterministic, high-confidence)
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            response_ids = torch.cat([response_ids, next_token], dim=1)

        # Decode full sequence
        full_response = self.tokenizer.decode(response_ids[0])
        reasoning_text = self.tokenizer.decode(reasoning_tokens) if reasoning_tokens else "[No reasoning]"

        return {
            'full_response': full_response,
            'reasoning_tokens_count': len(reasoning_tokens),
            'alpha_used': alpha
        }
```

Implement adaptive alpha selection based on input difficulty:

```python
def estimate_difficulty_and_select_alpha(prompt, difficulty_classifier):
    """
    Estimate problem difficulty and select appropriate alpha.
    Easy problems use low alpha (fast), hard problems use high alpha (slow).
    """
    # Use a lightweight classifier to estimate difficulty
    difficulty_score = difficulty_classifier(prompt)  # Returns 0.0 to 1.0

    # Map difficulty to alpha: easy (0.2) to hard (0.8)
    alpha = 0.2 + (difficulty_score * 0.6)

    return alpha

# Usage in generation pipeline
for problem in test_problems:
    alpha = estimate_difficulty_and_select_alpha(problem, classifier)
    result = reasoner.generate_with_alpha(problem, alpha=alpha)
    print(f"Problem: {problem}")
    print(f"Thinking depth (α={alpha}): {result['reasoning_tokens_count']} tokens")
    print(f"Response: {result['full_response']}")
```

Create a utility to sweep alpha values and measure the speed-quality frontier:

```python
def measure_speed_quality_frontier(model, test_set, alpha_values=[0.0, 0.3, 0.5, 0.7, 1.0]):
    """
    Evaluate accuracy and latency across different alpha values.
    Helps find optimal operating point for your use case.
    """
    results = []
    for alpha in alpha_values:
        total_time = 0
        correct = 0

        for problem, correct_answer in test_set:
            start = time.time()
            response = model.generate_with_alpha(problem, alpha=alpha)
            elapsed = time.time() - start

            is_correct = check_correctness(response['full_response'], correct_answer)
            correct += is_correct
            total_time += elapsed

        accuracy = correct / len(test_set)
        avg_latency = total_time / len(test_set)
        results.append({
            'alpha': alpha,
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency * 1000
        })
        print(f"α={alpha}: Accuracy={accuracy:.3f}, Latency={avg_latency*1000:.1f}ms")

    return results
```

## Practical Guidance

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Alpha | 0.1 - 0.9 | Lower = fast inference, higher = better accuracy |
| Max reasoning tokens | 100 - 500 | Caps internal thinking length |
| Temperature (reasoning phase) | 0.7 - 1.0 | Higher for diverse reasoning paths |
| Temperature (generation phase) | 0.0 - 0.3 | Lower for confident, coherent responses |
| Difficulty classification threshold | 0.3 - 0.7 | Dividing point between easy/hard problems |

**When to use AlphaOne:**
- You have inference time constraints but want to preserve accuracy
- Problems vary significantly in difficulty (some easy, some hard)
- Your model has internal reasoning capability (like o1, Gemini 2.0 Thinking)
- You want inference optimization without retraining
- User experience matters (faster responses for easy questions)

**When NOT to use AlphaOne:**
- Your model doesn't produce internal reasoning tokens naturally
- All problems have similar difficulty levels (uniform alpha better)
- Inference latency isn't a constraint
- You need maximum accuracy regardless of compute cost
- Model doesn't support test-time scaling of reasoning

**Common pitfalls:**
- Setting fixed alpha for all inputs (defeats adaptive benefit)
- Alpha too low (1/3 of capacity) causes quality degradation
- Alpha too high (>0.85) wastes compute without accuracy gains
- Not measuring both latency and accuracy during tuning
- Difficulty classifier correlates poorly with actual problem hardness
- Applying AlphaOne to models without internal reasoning mechanisms

## Reference

**AlphaOne: Reasoning Models Thinking Slow and Fast at Test Time**
https://arxiv.org/abs/2505.24863
