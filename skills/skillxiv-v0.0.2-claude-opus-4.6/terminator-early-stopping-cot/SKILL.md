---
name: terminator-early-stopping-cot
title: "TERMINATOR: Learning Optimal Exit Points for Early Stopping in CoT Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.12529"
keywords: [Early Stopping, Chain-of-Thought, Reasoning Efficiency, Token Reduction, Exit Prediction]
description: "Learn when to terminate chain-of-thought reasoning by predicting answer arrival patterns. Train an early-exit mechanism on empirical first-answer positions to reduce reasoning length by 14-55% while maintaining or improving accuracy."
---

# TERMINATOR: Learning Optimal Exit Points for Early Stopping in CoT Reasoning

Large Reasoning Models (LRMs) generate excessive intermediate tokens during chain-of-thought reasoning, continuing to compute and output reasoning steps long after arriving at the correct answer. TERMINATOR solves this by treating the problem not as hand-tuned stopping rules, but as a learnable prediction task. By observing where answers first appear in successful reasoning traces, you can train an early-exit mechanism that terminates generation at those optimal points—reducing reasoning token length by 14-55% across benchmarks while maintaining or improving accuracy.

The core insight: answer arrival patterns are predictable from problem characteristics and early reasoning steps. Rather than arbitrary length limits, you can learn task-specific stopping points empirically from data.

## Core Concept

TERMINATOR operates through three key phases:

1. **Data Collection** — Generate reasoning traces with full CoT output, identify first answer occurrence position
2. **Pattern Learning** — Train predictor on early reasoning context to forecast optimal exit point
3. **Inference** — Use predictor to signal early termination, preventing wasteful over-generation

The technique treats reasoning length not as a fixed budget, but as a learned function of problem difficulty and generation progress.

## Architecture Overview

- **Reference Dataset Generation** — Run full CoT on target tasks, mark first answer arrival position
- **Early-Context Encoder** — Processes partial reasoning (first N% of generated tokens) to predict optimal stopping point
- **Exit Signal Predictor** — Outputs confidence that answer has arrived; triggers stopping when confidence exceeds threshold
- **Length Estimation Model** — Predicts task-specific optimal reasoning length distribution
- **Inference Pipeline** — Monitor generation stream, apply early exit when signal triggers
- **Validation Framework** — Test on held-out problems, measure accuracy preservation and token reduction

## Implementation Steps

Begin by collecting reference data showing where answers first appear in successful reasoning traces.

```python
import json
from collections import defaultdict

def collect_answer_arrival_data(model, tasks, output_dir="arrivals"):
    """Generate full CoT traces and mark answer first-arrival positions."""
    arrivals = []

    for task_id, task in enumerate(tasks):
        # Generate full CoT trace
        prompt = f"Reason step-by-step:\n{task['question']}"
        full_trace = model.generate(prompt, max_tokens=2000, return_token_log=True)

        # Parse full trace to find answer
        answer_token_start = None
        answer_text = task.get('answer', '')

        # Search for answer in generated tokens
        for token_idx, token_data in enumerate(full_trace['tokens']):
            if answer_text.lower() in token_data['text'].lower():
                answer_token_start = token_idx
                break

        if answer_token_start is None:
            continue  # Skip if answer not found

        arrivals.append({
            'task_id': task_id,
            'task_domain': task['domain'],
            'total_tokens': len(full_trace['tokens']),
            'answer_token_position': answer_token_start,
            'answer_arrival_ratio': answer_token_start / len(full_trace['tokens']),
            'full_trace': full_trace['text'],
            'partial_trace_at_answer': full_trace['text'][:answer_token_start]
        })

    # Analyze distribution by domain
    by_domain = defaultdict(list)
    for arrival in arrivals:
        by_domain[arrival['task_domain']].append(
            arrival['answer_arrival_ratio'])

    print("Answer arrival ratios by domain:")
    for domain, ratios in by_domain.items():
        avg_ratio = sum(ratios) / len(ratios)
        print(f"  {domain}: {avg_ratio:.2%} (n={len(ratios)})")

    return arrivals
```

Next, train the exit predictor on early reasoning context. The key is learning from partial traces where the model hasn't yet completed reasoning.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ExitPredictor(nn.Module):
    """Predicts whether answer has arrived based on partial trace."""

    def __init__(self, embed_dim=768, hidden_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(50257, embed_dim)  # GPT vocab size
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Confidence 0-1
        )

    def forward(self, token_ids, partial_lengths):
        """
        Args:
            token_ids: [batch, max_seq_len] partial reasoning token IDs
            partial_lengths: [batch] actual length of each partial trace
        """
        embeddings = self.embedding(token_ids)

        # Mask padding
        mask = torch.arange(token_ids.size(1)).unsqueeze(0) < partial_lengths.unsqueeze(1)

        # Encode with attention
        encoded = self.encoder(embeddings, src_key_padding_mask=~mask)

        # Pool using last non-padding token
        last_token_indices = (partial_lengths - 1).clamp(min=0)
        last_embeddings = encoded[torch.arange(encoded.size(0)),
                                   last_token_indices]

        # Predict exit confidence
        exit_confidence = self.confidence_head(last_embeddings)
        return exit_confidence.squeeze(-1)


def train_exit_predictor(arrivals, model, tokenizer, epochs=10, batch_size=32):
    """Train predictor on answer-arrival data."""
    # Prepare training data
    X = []  # Partial traces at different completion percentages
    y = []  # Binary labels: has answer arrived?

    for arrival in arrivals:
        full_tokens = tokenizer.encode(arrival['full_trace'])
        answer_pos = arrival['answer_token_position']

        # Create samples at 20%, 40%, 60%, 80% completion
        for completion_pct in [0.2, 0.4, 0.6, 0.8]:
            current_pos = int(len(full_tokens) * completion_pct)
            partial_tokens = full_tokens[:current_pos]

            X.append(torch.tensor(partial_tokens))
            # Label: 1 if we've reached answer, 0 otherwise
            y.append(1.0 if current_pos >= answer_pos else 0.0)

    # Pad sequences
    max_len = max(len(x) for x in X)
    X_padded = torch.zeros(len(X), max_len, dtype=torch.long)
    lengths = []
    for i, x in enumerate(X):
        X_padded[i, :len(x)] = x
        lengths.append(len(x))

    y_tensor = torch.tensor(y, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    # Create dataset and loader
    dataset = TensorDataset(X_padded, lengths_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train
    predictor = ExitPredictor()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, lengths_batch, y_batch in loader:
            logits = predictor(X_batch, lengths_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    return predictor
```

Finally, integrate the exit predictor into your inference pipeline to terminate early.

```python
# Inference with early exit using the predictor
def generate_with_early_exit(model, prompt, predictor, tokenizer,
                             exit_threshold=0.7, max_tokens=2000):
    """Generate with learned early stopping."""
    generated_tokens = []
    token_ids = tokenizer.encode(prompt)

    for step in range(max_tokens):
        # Generate next token
        with torch.no_grad():
            logits = model(torch.tensor([token_ids]))
            next_token = torch.argmax(logits[-1, -1, :])
            generated_tokens.append(next_token.item())

        # Check exit signal periodically (every 10 tokens)
        if step > 50 and step % 10 == 0:
            # Tokenize partial trace
            partial_tokens = torch.tensor([token_ids + generated_tokens])
            partial_length = torch.tensor([len(token_ids) + len(generated_tokens)])

            # Get exit confidence
            with torch.no_grad():
                exit_conf = predictor(partial_tokens, partial_length)

            print(f"Step {step}: Exit confidence = {exit_conf:.2%}")

            # Exit if confident that answer has arrived
            if exit_conf > exit_threshold:
                print(f"Early exit at step {step} with confidence {exit_conf:.2%}")
                break

        token_ids.append(next_token.item())

        # Standard stopping: end-of-sequence token
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_tokens)


def benchmark_early_exit(model, predictor, tokenizer, test_tasks,
                         thresholds=[0.5, 0.6, 0.7, 0.8]):
    """Measure accuracy vs token reduction across thresholds."""
    results = []

    for threshold in thresholds:
        total_tokens = 0
        correct = 0

        for task in test_tasks:
            output = generate_with_early_exit(model, task['prompt'],
                                             predictor, tokenizer,
                                             exit_threshold=threshold)
            total_tokens += len(tokenizer.encode(output))

            # Check correctness
            if task['answer'].lower() in output.lower():
                correct += 1

        avg_tokens = total_tokens / len(test_tasks)
        accuracy = correct / len(test_tasks)

        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'avg_tokens': avg_tokens
        })

        print(f"Threshold {threshold}: Accuracy={accuracy:.1%}, "
              f"Avg Tokens={avg_tokens:.0f}")

    return results
```

## Practical Guidance

**Hyperparameters and When to Use:**
- Exit threshold typically ranges 0.6-0.8; lower thresholds reduce tokens more aggressively but risk premature stopping
- Collect reference data on at least 1000 examples per task domain; smaller datasets overfit exit patterns
- Use when reasoning outputs often contain extended justification after the answer is already provided
- Particularly effective for math, QA, and code generation tasks with clear answer formats

**When NOT to use:**
- For open-ended generation where there is no clear "answer" (creative writing, summarization)
- When accuracy is paramount and cannot tolerate even small degradation
- For tasks where reasoning steps after the first answer contain critical corrections or refinements

**Common Pitfalls:**
- Training on only successful traces ignores failure patterns; collect data from both correct and incorrect outputs
- Using fixed threshold across all task types; train separate predictors or dynamic thresholds per domain
- Not accounting for answer format variations (e.g., numerical vs. verbal answers); normalize answer representations
- Excessive early exit reducing reasoning quality for complex problems; use confidence thresholding, not fixed position thresholds

## Reference

Paper: [TERMINATOR: Learning Optimal Exit Points for Early Stopping in CoT Reasoning](https://arxiv.org/abs/2603.12529)
