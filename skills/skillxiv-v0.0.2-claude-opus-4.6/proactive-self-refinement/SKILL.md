---
name: proactive-self-refinement
title: "A Stitch in Time: Proactive Self-Refinement for Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.12903
keywords: [self-refinement, dynamic-revision, in-generation-improvement, token-efficiency]
description: "Enable models to refine outputs dynamically during generation based on internal signals, reducing token consumption by 41.6% while improving accuracy by 8.2%."
---

# A Stitch in Time: Proactive Self-Refinement for Language Models

## Core Concept

Traditional self-refinement works in fixed cycles: generate → evaluate → regenerate. This is inefficient because the model can't start refinement until generation completes.

Proactive Active Self-Refinement (PASR) lets models decide dynamically during generation whether, when, and how to refine. The model learns to detect when its reasoning is going wrong and self-correct mid-generation, like humans revising thoughts while speaking.

## Architecture Overview

- **Dynamic Refinement Trigger**: Learn to detect when refinement is needed
- **In-Generation Revision**: Backtrack and revise during generation
- **Internal Quality Signals**: Use model's own uncertainty/confidence
- **Learned Refinement Strategy**: Decide how aggressively to refine
- **Token Efficiency**: Avoid wasteful regeneration of correct portions
- **Adaptive Refinement**: Different tasks get different refinement patterns

## Implementation Steps

### 1. Define Refinement Points and Signals

```python
import torch
import torch.nn as nn
from typing import List, Tuple

class RefinementSignal:
    """Detect when refinement is needed"""
    def __init__(self, model):
        self.model = model

    def compute_confidence(self, logits: torch.Tensor) -> float:
        """Compute model's confidence in current prediction"""
        # Softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        # Maximum probability = confidence
        confidence = probs.max().item()
        return confidence

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute entropy of output distribution"""
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        return entropy

    def compute_consistency(self, logits_list: List[torch.Tensor]) -> float:
        """Compute consistency across multiple forward passes"""
        if len(logits_list) < 2:
            return 1.0

        # Compare top predictions
        predictions = [logits.argmax().item() for logits in logits_list]
        consistency = predictions.count(predictions[0]) / len(predictions)
        return consistency

    def should_refine(self, confidence: float, entropy: float,
                     token_count: int, max_tokens: int) -> bool:
        """Determine if refinement is needed"""
        # Refine if: low confidence + high entropy + not too many tokens
        low_confidence = confidence < 0.5
        high_entropy = entropy > 2.0
        reasonable_length = token_count < max_tokens * 0.8

        return low_confidence and high_entropy and reasonable_length
```

### 2. Implement Dynamic Backtracking

```python
class DynamicBacktracker:
    """Backtrack and revise during generation"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.token_history = []

    def find_refinement_point(self, current_tokens: List[int],
                            quality_scores: List[float]) -> int:
        """Find best point to backtrack to"""
        # Backtrack to where quality dropped
        min_quality_idx = 0
        min_quality = quality_scores[0]

        for i, score in enumerate(quality_scores):
            if score < min_quality:
                min_quality = score
                min_quality_idx = i

        # Don't backtrack too far (keep at least 20% of generated text)
        min_backtrack = int(len(current_tokens) * 0.2)
        backtrack_point = max(min_backtrack_idx, min_backtrack)

        return backtrack_point

    def revise_from_point(self, context_tokens: List[int],
                         backtrack_point: int, num_alternatives: int = 3):
        """Generate alternatives from backtrack point"""
        # Truncate to backtrack point
        revised_tokens = context_tokens[:backtrack_point]

        alternatives = []
        for temp in [0.7, 0.8, 0.9]:
            # Generate alternative continuation
            output = self.model.generate(
                torch.tensor([revised_tokens]),
                max_new_tokens=50,
                temperature=temp,
                do_sample=True
            )
            alternatives.append(output[0].tolist())

        return alternatives

    def select_best_alternative(self, alternatives: List[List[int]],
                               quality_fn) -> Tuple[List[int], float]:
        """Select best alternative based on quality"""
        best_seq = None
        best_quality = -1

        for alt in alternatives:
            # Decode and evaluate quality
            text = self.tokenizer.decode(alt)
            quality = quality_fn(text)

            if quality > best_quality:
                best_quality = quality
                best_seq = alt

        return best_seq, best_quality
```

### 3. Train Refinement Policy

```python
class RefinementPolicy(nn.Module):
    """Learn when and how to refine"""
    def __init__(self, hidden_size=768):
        super().__init__()

        # Input: embeddings of generated so far
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )

        # Output heads
        self.refinement_trigger = nn.Linear(hidden_size, 1)  # Binary: refine or not
        self.backtrack_distance = nn.Linear(hidden_size, 100)  # How far to backtrack
        self.refinement_intensity = nn.Linear(hidden_size, 1)  # How aggressive

    def forward(self, token_embeddings: torch.Tensor) -> dict:
        """
        Decide refinement strategy

        Args:
            token_embeddings: [seq_len, hidden_size] embeddings of generated tokens

        Returns:
            refinement_decision: dict with trigger, backtrack_distance, intensity
        """
        # Encode sequence
        context = self.encoder(token_embeddings.unsqueeze(0))
        context = context[0, -1, :]  # Take last token

        # Predict refinement strategy
        refine_logit = self.refinement_trigger(context)  # [1]
        refine_prob = torch.sigmoid(refine_logit)  # 0-1

        backtrack_logits = self.backtrack_distance(context)  # [100]
        backtrack_dist = torch.softmax(backtrack_logits, dim=0)

        intensity = torch.sigmoid(self.refinement_intensity(context))  # [1]

        return {
            'should_refine': refine_prob.item() > 0.5,
            'refine_probability': refine_prob.item(),
            'backtrack_distance': backtrack_dist.argmax().item(),
            'refinement_intensity': intensity.item()
        }

def train_refinement_policy(policy, model, tokenizer, train_data, num_epochs=10):
    """Train policy with RL"""
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in train_data:
            prompts = batch['prompts']
            target_outputs = batch['outputs']

            losses = []

            for prompt, target in zip(prompts, target_outputs):
                # Generate with refinement
                generated_tokens = []
                token_embeddings = []
                refinements_applied = 0

                # Autoregressive generation
                tokens = tokenizer.encode(prompt)
                for step in range(256):  # Max generation length
                    # Get embeddings
                    embeddings = model.get_embeddings(torch.tensor([tokens]))

                    # Check if refinement needed
                    decision = policy(embeddings[0])

                    # Collect embeddings for policy
                    token_embeddings.append(embeddings[0, -1, :])

                    if decision['should_refine'] and step > 10:
                        # Apply refinement
                        backtrack_dist = decision['backtrack_distance']
                        tokens = tokens[:-backtrack_dist]
                        refinements_applied += 1

                    # Generate next token
                    logits = model(torch.tensor([tokens])).logits[0, -1, :]
                    next_token = logits.argmax().item()
                    tokens.append(next_token)

                    if next_token == tokenizer.eos_token_id:
                        break

                # Compute loss: reward based on correctness vs efficiency
                generated_text = tokenizer.decode(tokens)
                similarity = compute_similarity(generated_text, target)
                efficiency_bonus = (1.0 - len(tokens) / 256)  # Shorter is better

                reward = 0.8 * similarity + 0.2 * efficiency_bonus
                loss = -torch.tensor(reward)

                losses.append(loss)

            # Update policy
            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### 4. Inference with PASR

```python
def generate_with_pasr(model, policy, tokenizer, prompt: str,
                      max_length: int = 256, refinement_threshold: float = 0.5):
    """Generate with dynamic proactive self-refinement"""
    tokens = tokenizer.encode(prompt)
    generated = []
    refinement_count = 0
    quality_scores = []

    for step in range(max_length):
        # Get current embeddings
        embeddings = model.get_embeddings(torch.tensor([tokens]))
        curr_embedding = embeddings[0, -1, :]

        # Get refinement decision
        decision = policy(curr_embedding.unsqueeze(0))

        # Compute current quality
        confidence = decision['refine_probability']
        quality_scores.append(confidence)

        # Check if refinement triggered
        if (decision['should_refine'] and
            confidence < refinement_threshold and
            step > 10 and len(generated) > 5):

            # Backtrack
            backtrack = decision['backtrack_distance']
            tokens = tokens[:-min(backtrack, len(generated))]
            generated = generated[:-min(backtrack, len(generated))]
            refinement_count += 1

            # Regenerate with higher temperature for diversity
            logits = model(torch.tensor([tokens])).logits[0, -1, :]
            logits = logits / 0.8  # Increase temperature
            next_token = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).item()
        else:
            # Normal generation
            logits = model(torch.tensor([tokens])).logits[0, -1, :]
            next_token = logits.argmax().item()

        tokens.append(next_token)
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    # Decode result
    result_text = tokenizer.decode(generated)

    return {
        'text': result_text,
        'tokens_generated': len(generated),
        'refinements_applied': refinement_count,
        'efficiency': 1.0 - (refinement_count / len(generated))
    }
```

### 5. Evaluation

```python
def evaluate_pasr(model, policy, tokenizer, benchmark_tasks):
    """Evaluate PASR on accuracy and efficiency"""
    accuracy = 0.0
    token_efficiency = 0.0
    num_tasks = 0

    for task in benchmark_tasks:
        prompt = task['prompt']
        target = task['target']

        # Generate with PASR
        result = generate_with_pasr(model, policy, tokenizer, prompt)
        generated_text = result['text']

        # Check accuracy
        is_correct = check_correctness(generated_text, target)
        accuracy += 1.0 if is_correct else 0.0

        # Track token efficiency
        baseline_tokens = 100  # Estimated
        token_efficiency += 1.0 - (result['tokens_generated'] / baseline_tokens)

        num_tasks += 1

    avg_accuracy = accuracy / num_tasks if num_tasks > 0 else 0.0
    avg_efficiency = token_efficiency / num_tasks if num_tasks > 0 else 0.0

    print(f"Accuracy: {avg_accuracy * 100:.1f}%")
    print(f"Token Efficiency: {avg_efficiency * 100:.1f}%")

    return avg_accuracy, avg_efficiency
```

## Practical Guidance

- **Refinement Threshold**: 0.4-0.6 (lower = more aggressive refinement)
- **Backtrack Distance**: 5-20 tokens (avoid over-revision)
- **Temperature**: 0.7-0.9 for alternatives (higher = more diversity)
- **Policy Training**: Mix supervised + RL (80% supervised, 20% RL)
- **Quality Function**: Use task-specific metrics (BLEU, exact match, etc.)

## Reference

A Stitch in Time (2508.12903): https://arxiv.org/abs/2508.12903

Enable dynamic, in-generation self-refinement where models decide when to backtrack and revise, achieving 41.6% token reduction and 8.2% accuracy improvement over baseline generation.
