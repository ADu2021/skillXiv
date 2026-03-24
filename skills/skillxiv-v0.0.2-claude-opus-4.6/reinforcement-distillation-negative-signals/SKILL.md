---
name: reinforcement-distillation-negative-signals
title: "Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24850"
keywords: [Distillation, Reinforcement Learning, Negative Signals, Chain-of-Thought]
description: "Extract maximum value from limited reasoning traces by leveraging both successful and failed examples through REINFORCE-style distillation."
---

# REDI: Learn from Both Right and Wrong Reasoning

Standard distillation in reasoning tasks typically discards failed reasoning traces, treating them as waste. REDI (Reinforcement Distillation) reverses this by treating incorrect reasoning paths as valuable negative signals. A two-stage pipeline—supervised fine-tuning on successful examples, then reinforcement learning that explicitly contrasts success and failure—achieves better data efficiency than preference-based methods like DPO, reaching 83.1% accuracy on MATH-500 with 131k traces versus competitors needing 800k examples.

The key insight is that failed reasoning contains information: understanding *why* a path was wrong is as instructive as copying correct paths. By jointly training on both signal types with a carefully designed REINFORCE objective, student models learn more efficiently from limited teacher data.

## Core Concept

Traditional distillation uses only positive examples (teacher's correct outputs). REDI extends this to a two-phase learning process: first absorb the structure and patterns from successful traces, then use reinforcement learning to learn what to avoid. The REINFORCE objective directly optimizes the probability difference between correct and incorrect traces, avoiding the implicit assumptions built into preference-based losses like DPO that can work poorly when distilling from limited data.

## Architecture Overview

- **Two-Stage Pipeline**: SFT phase on positive traces, then REINFORCE-style RL phase that incorporates both success and failure examples
- **Positive Trace Collection**: Curated set of teacher-generated correct reasoning chains for initial supervised training
- **Negative Trace Utilization**: Parallel collection of incorrect teacher traces used as contrastive pairs during RL
- **REDI Objective**: REINFORCE-based loss that increases likelihood of correct traces while decreasing incorrect ones
- **Data Efficiency**: Achieves competitive results with 5-10x fewer total examples than approaches requiring large-scale distillation datasets

## Implementation

This implementation demonstrates the two-stage REDI pipeline for distilling reasoning from teacher models.

First, prepare and balance positive and negative traces:

```python
import json
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ReasoningTrace:
    problem: str
    reasoning: str
    answer: str
    is_correct: bool

def prepare_distillation_dataset(teacher_generations: List[Dict]) -> Dict[str, List]:
    """
    Organize teacher traces into positive and negative examples.
    Assumes teacher_generations contains (problem, reasoning, answer, correctness) tuples.
    """
    positive_traces = []
    negative_traces = []

    for item in teacher_generations:
        trace = ReasoningTrace(
            problem=item["problem"],
            reasoning=item["reasoning"],
            answer=item["answer"],
            is_correct=item["is_correct"]
        )
        if trace.is_correct:
            positive_traces.append(trace)
        else:
            negative_traces.append(trace)

    # Balance dataset: ensure negative examples don't overwhelm
    min_count = min(len(positive_traces), len(negative_traces))
    balanced_negatives = negative_traces[:min_count]

    return {
        "positive": positive_traces,
        "negative": balanced_negatives
    }

# Example usage
teacher_data = [
    {
        "problem": "Solve 2x + 5 = 13",
        "reasoning": "Subtract 5 from both sides: 2x = 8. Divide by 2: x = 4.",
        "answer": "4",
        "is_correct": True
    },
    {
        "problem": "Solve 2x + 5 = 13",
        "reasoning": "Add 5 to both sides: 2x = 18. x = 18.",
        "answer": "18",
        "is_correct": False
    }
]
distillation_data = prepare_distillation_dataset(teacher_data)
```

Implement the SFT stage using positive traces only:

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

class PositiveTraceDataset(Dataset):
    """Dataset for supervised fine-tuning on correct reasoning traces."""

    def __init__(self, traces: List[ReasoningTrace], tokenizer, max_length=512):
        self.traces = traces
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        trace = self.traces[idx]
        # Format: problem + reasoning
        text = f"Problem: {trace.problem}\nReasoning: {trace.reasoning}\nAnswer: {trace.answer}"
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }

    def __len__(self):
        return len(self.traces)

def run_sft_phase(positive_traces: List, model_name="gpt2-medium", epochs=3):
    """Stage 1: Supervised fine-tuning on positive traces."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = PositiveTraceDataset(positive_traces, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    return model, tokenizer

# Run SFT
sft_model, tokenizer = run_sft_phase(distillation_data["positive"], epochs=2)
```

Implement the REINFORCE-based RL phase that leverages negative signals:

```python
class REDIReinforceOptimizer:
    """REINFORCE-style distillation that contrasts positive and negative traces."""

    def __init__(self, model, tokenizer, learning_rate=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def compute_reinforce_loss(self, problem, positive_trace, negative_trace):
        """
        REINFORCE objective: maximize log P(positive) - log P(negative).
        This directly increases probability of correct reasoning while decreasing incorrect.
        """
        # Encode positive example
        pos_text = f"Problem: {problem}\nReasoning: {positive_trace}"
        pos_tokens = self.tokenizer(pos_text, return_tensors="pt", truncation=True)

        # Encode negative example
        neg_text = f"Problem: {problem}\nReasoning: {negative_trace}"
        neg_tokens = self.tokenizer(neg_text, return_tensors="pt", truncation=True)

        # Forward pass to get log probabilities
        with torch.no_grad():
            pos_outputs = self.model(**pos_tokens, output_hidden_states=True)
            neg_outputs = self.model(**neg_tokens, output_hidden_states=True)

        # Compute log probabilities for each token sequence
        pos_logits = pos_outputs.logits
        neg_logits = neg_outputs.logits

        # Calculate log probs using cross-entropy over sequences
        pos_log_prob = -F.cross_entropy(
            pos_logits.view(-1, pos_logits.size(-1)),
            pos_tokens["input_ids"].view(-1)
        )
        neg_log_prob = -F.cross_entropy(
            neg_logits.view(-1, neg_logits.size(-1)),
            neg_tokens["input_ids"].view(-1)
        )

        # REINFORCE loss: push apart the two distributions
        reinforce_loss = -(pos_log_prob - neg_log_prob)
        return reinforce_loss

    def train_step(self, problem, positive_trace, negative_trace):
        """Single RL training step."""
        loss = self.compute_reinforce_loss(problem, positive_trace, negative_trace)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

# Run RL phase
rl_optimizer = REDIReinforceOptimizer(sft_model, tokenizer)
print("Starting RL phase with negative signal learning...")

for epoch in range(3):
    epoch_loss = 0
    for pos_trace, neg_trace in zip(
        distillation_data["positive"],
        distillation_data["negative"]
    ):
        loss = rl_optimizer.train_step(
            pos_trace.problem,
            pos_trace.reasoning,
            neg_trace.reasoning
        )
        epoch_loss += loss

    avg_loss = epoch_loss / len(distillation_data["positive"])
    print(f"RL Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```

Evaluate the distilled model on reasoning tasks:

```python
def evaluate_reasoning_model(model, tokenizer, test_problems, correct_answers):
    """Evaluate model accuracy on reasoning tasks."""
    model.eval()
    correct = 0

    for problem, expected_answer in zip(test_problems, correct_answers):
        prompt = f"Problem: {problem}\nReasoning:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=256,
                temperature=0.7,
                do_sample=False
            )

        generated_text = tokenizer.decode(outputs[0])
        # Simple check: does the answer appear in output?
        if str(expected_answer) in generated_text:
            correct += 1

    accuracy = correct / len(test_problems)
    return accuracy

# Evaluation
test_problems = ["Solve 3x - 7 = 5", "What is 15% of 200?"]
test_answers = ["4", "30"]
accuracy = evaluate_reasoning_model(sft_model, tokenizer, test_problems, test_answers)
print(f"Reasoning Accuracy: {accuracy*100:.1f}%")
```

## Practical Guidance

| Aspect | Details |
|--------|---------|
| **Positive/Negative Ratio** | Aim for 1:1 pairing; imbalanced ratios reduce RL signal strength |
| **Trace Length** | 50-300 tokens typical; longer traces provide more reasoning context but increase compute |
| **SFT Epochs** | 2-5 epochs sufficient; monitor validation loss to avoid overfitting to positive traces |
| **RL Learning Rate** | Start at 1e-5, reduce if divergence occurs; REINFORCE can be unstable with large LR |
| **Batch Strategy** | Process positive-negative pairs together; mini-batches of 4-8 pairs work well |

**When to Use:**
- Limited teacher data available (< 200k examples for target task)
- Both correct and incorrect reasoning traces are easy to collect or generate
- Maximizing data efficiency is critical (mobile/edge deployment, expensive annotation)
- Preference-based methods (DPO, SimPO) underperform on your domain
- Student model scale is significantly smaller than teacher

**When NOT to Use:**
- Massive distillation datasets already available (plain SFT sufficient and simpler)
- Only correct outputs available (no negative examples can be collected)
- Teacher has very low accuracy (negative signals become uninformative)
- Real-time distillation where longer training pipelines are impractical
- Downstream task requires diversity rather than high accuracy

**Common Pitfalls:**
- Mislabeled negative examples: carefully verify that negatives are truly incorrect, else RL learns wrong contrasts
- REINFORCE instability: loss can oscillate; use gradient clipping and conservative learning rates
- Overfitting to negative traces: if negative examples are too similar to positive ones, model may fail to generalize
- Ignoring positive trace quality: garbage in, garbage out; poor initial SFT hurts RL phase significantly

## Reference

Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning
https://arxiv.org/abs/2505.24850
