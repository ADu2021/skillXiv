---
name: toolrm-outcome-reward-tool-calling
title: "ToolRM: Outcome Reward Models for Tool-Calling Large Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.11963"
keywords: [reward-models, tool-calling, LLMs, outcome-evaluation, function-calling, reinforcement-learning, preference-optimization, evaluation-benchmarks]
description: "Specialized outcome reward models (1.7B-14B) for evaluating tool-calling performance in LLMs. Addresses the critical gap in reward modeling where general-purpose models miss key signals of effective tool use. Enables better Best-of-N sampling, data filtering, and RL-based policy training through FC-RewardBench evaluation framework."
---

# ToolRM: Evaluating Tool Use Outcomes in LLMs

## The Outcome: Accurate Tool-Calling Reward Signals

As LLMs increasingly orchestrate complex workflows through tool calling (APIs, databases, calculators), the ability to score whether a tool invocation achieved the user's goal has become essential. ToolRM delivers outcome reward models (1.7B-14B parameters) that accurately evaluate tool-calling success—enabling 25% performance improvements in Best-of-N sampling, robust data filtering for RL training, and effective policy optimization for autonomous agent systems.

## Problem Context

Large language models now serve as orchestrators of external tools and services. Yet existing reward models trained on natural language outputs frequently miss the key signals of effective tool use: Did the model invoke the right tools? With correct arguments? In the right sequence? General-purpose reward models like those on RewardBench show poor correlation with actual tool-calling performance, leaving practitioners without reliable signals for preference optimization or reinforcement learning.

The gap is acute because tool-calling scenarios differ fundamentally from open-ended generation:
- **Binary outcome clarity**: Tool calls either invoke the correct function or they don't—grounding is explicit.
- **Structured reasoning**: Tool selection and argument filling follow deterministic patterns, not creative prose.
- **Multi-turn complexity**: Tool calls often require sequential reasoning with dependencies (search result → next query → database lookup).
- **Robustness requirements**: Models need signals that degrade gracefully under input noise, missing APIs, or partial information.

Prior work on reward modeling addresses code verification, math problem solving, and process-level step supervision, but no systematic framework existed for evaluating tool-calling reward models specifically.

## Core Concept

ToolRM tackles outcome reward modeling for tool-calling tasks through three aligned contributions:

**1. FC-RewardBench**: The first benchmark (1,500 data points) specifically designed for evaluating reward models on function-calling tasks. Each example contains a user query, tool catalog, correct tool calls, and incorrect candidates generated from a diverse pool of 25 LLMs (0.5B–685B parameters).

**2. ToolRM Suite**: Outcome reward models trained end-to-end on tool-calling data, available in four scales (1.7B, 7B, 14B parameters). These models score the entire tool call sequence in context, capturing correctness signals that general-purpose models miss.

**3. Empirical Validation**: Demonstration that ToolRM outperforms general-purpose baselines across diverse scenarios (Best-of-N sampling, data filtering, RL-guided fine-tuning), with correlation analysis showing ToolRM scores predict downstream task success.

The key insight: outcome reward models work better than process-level scoring for tool-calling because the correctness of a tool invocation is fundamentally tied to the final outcome—whether the query was answered correctly. A single wrong parameter ruins the entire call.

## Architecture Overview

ToolRM adopts a lightweight, practical design for integrating into LLM training pipelines:

- **Base Architecture**: Transformer-based, decoder-only models (similar to Llama, Mistral) as foundational encoders
- **Input Representation**: Concatenates user query, tool catalog (function names, descriptions, parameter schemas), and the proposed tool call sequence into a single prompt context
- **Scoring Head**: Trained classification or regression head that outputs a scalar reward or preference logit
- **Training Signal**: Synthetic preference pairs (correct vs. incorrect tool calls) from diverse LLMs, augmented with permissive-license data sources
- **Scales**: 1.7B, 7B, 14B parameters, allowing trade-offs between latency and accuracy
- **No Tool Execution**: The model evaluates tool calls statically (syntactic correctness, semantic alignment) without actually executing functions
- **Robustness**: Evaluation protocol includes noise injection (missing APIs, malformed arguments) to assess graceful degradation

The architecture intentionally avoids process-level supervision (rewarding intermediate steps) in favor of outcome evaluation, because tool-calling correctness is largely deterministic—the presence of a single error invalidates the entire sequence.

## Implementation

### Step 1: Prepare FC-RewardBench Evaluation Data

Create a dataset with 1,500 function-calling examples following the BFCL-v3 format: user query → tool catalog → correct calls → incorrect candidates.

```python
import json
import random

# Load BFCL-v3 single-turn data
with open("bfcl-v3-single-turn.jsonl") as f:
    examples = [json.loads(line) for line in f]

# Structure for FC-RewardBench
def create_eval_example(query, tools, correct_calls, incorrect_candidates):
    return {
        "user_query": query,
        "tool_catalog": tools,  # List of tool definitions with name, desc, params
        "correct_calls": correct_calls,  # Reference tool call sequences
        "incorrect": incorrect_candidates,  # Negative examples from diverse LLMs
        "difficulty": assess_complexity(query, tools)
    }

# Generate negatives using 25 LLMs (0.5B to 685B parameters)
lm_pool = ["gpt2", "pythia-1b", "llama-7b", "gpt-3.5", "llama-70b"]
incorrect_calls = {}

for lm in lm_pool:
    try:
        # Prompt each LM with: "Query: {query}\nTools: {tools}\nCall:"
        response = call_llm_with_prompt(query, tools, lm_id=lm)
        incorrect_calls[lm] = parse_tool_call(response)
    except Exception as e:
        pass  # Skip if LM call fails

eval_data = {
    "examples": [
        create_eval_example(ex["query"], ex["tools"], ex["calls"], incorrect_calls)
        for ex in examples[:1500]
    ]
}

with open("fc_rewardbench.json", "w") as f:
    json.dump(eval_data, f)
```

### Step 2: Synthesize Training Data for ToolRM

Generate preference pairs (correct call wins, incorrect call loses) using permissively licensed LLMs. This avoids licensing concerns with proprietary models and ensures reproducibility.

```python
import torch
from datasets import Dataset
from transformers import AutoTokenizer

# Source: permissive LLMs (Llama, Mistral, MPT)
training_queries = [
    {
        "query": "Find Python libraries for web scraping",
        "tools": [
            {"name": "web_search", "description": "Search the web", "params": {"q": "string"}},
            {"name": "code_search", "description": "Search code repositories", "params": {"lang": "string", "query": "string"}}
        ]
    },
    # ... 100K+ examples
]

def create_training_pair(query, tools, correct_call, incorrect_call):
    # Concatenate context: query + tool definitions + call
    context = f"Query: {query}\nTools:\n"
    for t in tools:
        context += f"- {t['name']}: {t['description']}\n"

    pair = {
        "context": context,
        "correct": correct_call,
        "incorrect": incorrect_call,
        "label": 1  # Correct call is preferred
    }
    return pair

training_pairs = []
for ex in training_queries:
    # Generate correct and incorrect calls
    correct = synthesize_correct_call(ex["query"], ex["tools"])
    incorrect = synthesize_incorrect_call(ex["query"], ex["tools"])

    pair = create_training_pair(ex["query"], ex["tools"], correct, incorrect)
    training_pairs.append(pair)

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(training_pairs)
dataset.save_to_disk("toolrm_training_data")

print(f"Created {len(training_pairs)} training pairs")
```

### Step 3: Train ToolRM with Outcome Reward Objective

Fine-tune a base LLM (Llama 7B, Mistral 7B, etc.) on preference pairs using a binary classification loss. The model learns to output high scores for correct tool calls and low scores for incorrect ones.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Initialize base model
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# Add scalar reward head
class ToolRMModel(torch.nn.Module):
    def __init__(self, base_model, hidden_size=4096):
        super().__init__()
        self.base_model = base_model
        self.reward_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        # Get base model output (last hidden state)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # Score at end-of-sequence token
        eos_mask = (input_ids == tokenizer.eos_token_id).float()
        eos_idx = eos_mask.argmax(dim=1)

        scores = torch.zeros(input_ids.shape[0], device=input_ids.device)
        for i, idx in enumerate(eos_idx):
            if idx > 0:
                scores[i] = self.reward_head(last_hidden[i, idx])

        return scores

model = ToolRMModel(model)

# Define training function
def compute_preference_loss(correct_scores, incorrect_scores, margin=0.5):
    # Bradley-Terry ranking loss: higher score for correct > incorrect
    loss = -torch.log(torch.sigmoid(correct_scores - incorrect_scores - margin))
    return loss.mean()

# Tokenize training data
def preprocess_function(examples):
    # Tokenize context + correct call
    correct_text = examples["context"] + examples["correct"]
    correct_ids = tokenizer(correct_text, truncation=True, max_length=512)

    # Tokenize context + incorrect call
    incorrect_text = examples["context"] + examples["incorrect"]
    incorrect_ids = tokenizer(incorrect_text, truncation=True, max_length=512)

    return {
        "correct_input_ids": correct_ids["input_ids"],
        "correct_attention_mask": correct_ids["attention_mask"],
        "incorrect_input_ids": incorrect_ids["input_ids"],
        "incorrect_attention_mask": incorrect_ids["attention_mask"]
    }

dataset = dataset.map(preprocess_function, batched=True, remove_columns=["context", "correct", "incorrect", "label"])

# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

batch_size = 8
num_epochs = 3
warmup_steps = 500

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]

        # Convert to tensors
        correct_input_ids = torch.tensor(batch["correct_input_ids"]).to(device)
        correct_attention = torch.tensor(batch["correct_attention_mask"]).to(device)
        incorrect_input_ids = torch.tensor(batch["incorrect_input_ids"]).to(device)
        incorrect_attention = torch.tensor(batch["incorrect_attention_mask"]).to(device)

        # Forward pass
        correct_scores = model(correct_input_ids, correct_attention)
        incorrect_scores = model(incorrect_input_ids, incorrect_attention)

        # Compute loss
        loss = compute_preference_loss(correct_scores, incorrect_scores)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i // batch_size) % 100 == 0:
            print(f"Epoch {epoch}, Step {i//batch_size}, Loss: {loss.item():.4f}")

model.save_pretrained("toolrm-7b")
print("Training complete. Model saved.")
```

### Step 4: Evaluate ToolRM on FC-RewardBench

Benchmark the trained model against baselines and measure correlation with downstream performance.

```python
from sklearn.metrics import accuracy_score, roc_auc_score, spearmanr
import numpy as np

def evaluate_on_fc_rewardbench(model, tokenizer, eval_data, device="cuda"):
    model.eval()

    accuracies = []
    auc_scores = []

    with torch.no_grad():
        for example in eval_data["examples"]:
            query = example["user_query"]
            tools = example["tool_catalog"]
            correct = example["correct_calls"][0]

            # Create context
            context = f"Query: {query}\nTools:\n"
            for t in tools:
                context += f"- {t['name']}: {t['description']}\n"

            # Score correct call
            correct_text = context + correct
            correct_ids = tokenizer(correct_text, truncation=True, max_length=512, return_tensors="pt").to(device)
            correct_score = model(correct_ids["input_ids"], correct_ids["attention_mask"]).item()

            # Score incorrect candidates
            incorrect_scores = []
            for incorrect in example["incorrect"]:
                incorrect_text = context + incorrect
                incorrect_ids = tokenizer(incorrect_text, truncation=True, max_length=512, return_tensors="pt").to(device)
                score = model(incorrect_ids["input_ids"], incorrect_ids["attention_mask"]).item()
                incorrect_scores.append(score)

            # Accuracy: does model rank correct higher than all incorrect?
            correct_wins = sum(1 for s in incorrect_scores if correct_score > s)
            acc = correct_wins / len(incorrect_scores)
            accuracies.append(acc)

            # AUC: ranking quality
            labels = [1] * 1 + [0] * len(incorrect_scores)
            scores = [correct_score] + incorrect_scores
            auc = roc_auc_score(labels, scores)
            auc_scores.append(auc)

    return {
        "accuracy": np.mean(accuracies),
        "auc": np.mean(auc_scores),
        "std_accuracy": np.std(accuracies)
    }

# Run evaluation
eval_results = evaluate_on_fc_rewardbench(model, tokenizer, eval_data)
print(f"Accuracy: {eval_results['accuracy']:.3f}")
print(f"AUC: {eval_results['auc']:.3f}")
print(f"Std Dev: {eval_results['std_accuracy']:.3f}")
```

### Step 5: Apply ToolRM for Best-of-N Sampling

Use the trained reward model to select the best tool-calling trajectory from N candidates.

```python
def best_of_n_sampling(query, tools, candidate_calls, model, tokenizer, device="cuda", n=5):
    """
    Given N candidate tool-calling sequences, use ToolRM to select the best one.
    """
    model.eval()

    # Create context
    context = f"Query: {query}\nTools:\n"
    for t in tools:
        context += f"- {t['name']}: {t['description']}\nParams: {t['params']}\n"

    scores = []
    with torch.no_grad():
        for call_seq in candidate_calls:
            text = context + f"Tool calls:\n{call_seq}"
            input_ids = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
            score = model(input_ids["input_ids"], input_ids["attention_mask"]).item()
            scores.append(score)

    # Return highest-scoring trajectory
    best_idx = np.argmax(scores)
    return candidate_calls[best_idx], scores[best_idx]

# Example usage
query = "What are the latest papers on multimodal AI?"
tools = [
    {"name": "web_search", "description": "Search the web", "params": {"q": "string"}},
    {"name": "fetch_arxiv", "description": "Fetch papers from arXiv", "params": {"query": "string", "max_results": "int"}}
]

candidate_calls = [
    "web_search(q='multimodal AI papers 2025')",
    "fetch_arxiv(query='multimodal', max_results=10)",
    "web_search(q='vision language models')\nfetch_arxiv(query='multimodal transformers')"
]

best_call, score = best_of_n_sampling(query, tools, candidate_calls, model, tokenizer)
print(f"Selected: {best_call}\nScore: {score:.3f}")
```

### Step 6: Integrate ToolRM for RL-Based Fine-Tuning

Use ToolRM as the reward signal in GRPO (Group Relative Policy Optimization) or similar RL algorithms to optimize a policy model for tool calling.

```python
import torch
from trl import GRPOTrainer, GRPOConfig

# ToolRM as reward function
def reward_function(query, tool_calls, model, tokenizer, device="cuda"):
    """
    Evaluate tool-calling trajectory using ToolRM.
    Returns scalar reward (can be batched).
    """
    context = f"Query: {query}\nTools: ...\n"
    text = context + tool_calls

    input_ids = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        reward = model(input_ids["input_ids"], input_ids["attention_mask"])

    return reward

# Configure GRPO trainer
grpo_config = GRPOConfig(
    output_dir="./toolrm_grpo_checkpoint",
    learning_rate=1e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_seq_length=512,
    temperature=0.7
)

# Initialize trainer
grpo_trainer = GRPOTrainer(
    model=policy_model,  # Policy model to optimize (Llama, Mistral, etc.)
    args=grpo_config,
    train_dataset=tool_calling_dataset,
    reward_model=model,  # ToolRM
    tokenizer=tokenizer
)

# Fine-tune policy
grpo_trainer.train()
grpo_trainer.save_model("policy_model_grpo_optimized")

print("RL fine-tuning complete.")
```

## Practical Guidance

### Hyperparameters and Configuration

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| Model size | 7B | Best balance; 1.7B faster, 14B more accurate |
| Max context length | 512–1024 tokens | Covers most single-turn tool-calling queries |
| Training batch size | 8–16 | Depends on GPU memory; bfloat16 reduces footprint |
| Learning rate | 1e-5 to 5e-5 | Standard for adapter-based fine-tuning |
| Margin (preference loss) | 0.5 | Separation threshold between correct/incorrect |
| Training data size | 10K–50K pairs | Synthetic data from diverse LLMs works well |
| Evaluation frequency | Every 500 steps | Monitor overfitting to specific tool distributions |
| Inference temperature | 0.0 (greedy) | Scoring is deterministic; no sampling needed |

### When to Use ToolRM

- **Best-of-N sampling**: Generate N candidate tool-calling sequences and use ToolRM to rank them (optimal for small N < 10)
- **Data filtering**: Identify high-quality tool-calling examples in synthetic data before RL training
- **RL reward signal**: Integrate ToolRM as the outcome reward in GRPO or PPO for policy optimization
- **Benchmark evaluation**: Systematically compare tool-calling LLMs across dimensions (tool diversity, complexity, noise robustness)
- **Preference optimization**: Use ToolRM preference signals in DPO or IPO to align LLMs toward better tool use
- **Deployment monitoring**: Track tool-calling quality in production by scoring call sequences real-time

### When NOT to Use ToolRM

- **Open-ended generation tasks**: ToolRM is specialized for structured tool calls; use general-purpose reward models (RewardBench top models) for natural language preference
- **Process-level feedback**: If your task requires step-by-step reasoning guidance (intermediate steps matter), use process reward models instead; ToolRM gives final-outcome signals only
- **Real-time tool execution feedback**: ToolRM provides static evaluation; if you need rewards from actually executing tools (e.g., database queries), integrate tool execution directly into the loop
- **Extreme domain drift**: ToolRM trained on BFCL tools (web search, code, APIs) may not generalize to proprietary or highly specialized tool sets; fine-tune on your domain
- **Multi-agent coordination**: ToolRM scores individual tool calls; for multi-agent orchestration, model inter-agent dependencies explicitly
- **Low-latency inference**: 7B/14B models require GPU; use 1.7B version if latency is critical, or distill further

### Common Pitfalls and Mitigations

**Pitfall 1: Training on same LLM errors repeatedly**
- The negative examples matter. If you only use errors from 2-3 LMs, the model learns narrow failure patterns.
- *Mitigation*: Synthesize negatives from 25+ diverse models as in the paper; rotate models during training.

**Pitfall 2: Overfitting to tool catalog format**
- If all training examples use the same tool description style, ToolRM fails on differently-formatted catalogs.
- *Mitigation*: Vary tool schemas, description lengths, and parameter naming during data generation; test on out-of-distribution catalogs.

**Pitfall 3: Missing low-signal negatives**
- Generating negatives only from weak models (0.5B params) means they're too obvious to distinguish. Strong negatives train better discernment.
- *Mitigation*: Include negatives from models at or above your target policy quality (7B–70B range).

**Pitfall 4: Margin too large or too small**
- Margin 2.0 in preference loss makes the training too strict; margin 0.1 allows incorrect calls to remain competitive.
- *Mitigation*: Start at 0.5, measure AUC on held-out eval data, adjust if AUC plateaus or diverges.

**Pitfall 5: Ignoring robustness under noise**
- ToolRM trained on clean tool catalogs may collapse on missing APIs, malformed arguments, or incomplete context.
- *Mitigation*: Explicitly corrupt training data (drop 10% of tools, mangle 5% of arguments) to teach graceful degradation.

## Reference

**Paper**: ToolRM: Outcome Reward Models for Tool-Calling Large Language Models

**Authors**: Mayank Agarwal, Ibrahim Abdelaziz, Kinjal Basu, Merve Unuvar, Luis A. Lastras, Yara Rizk, Pavan Kapanipathi (IBM Research)

**Published**: September 2025 | **Version**: v2 (January 2026)

**arXiv**: https://arxiv.org/abs/2509.11963

**Key Contributions**:
1. **FC-RewardBench**: First benchmark (1,500 examples) for evaluating reward models on function-calling tasks
2. **ToolRM Models**: Outcome reward models (1.7B–14B params) trained on synthetic tool-calling data from diverse LLMs
3. **Empirical Results**: Up to 25% improvement with Best-of-N sampling; strong correlation with downstream task success; robustness to input noise

**Related Work**: RewardBench (general-purpose reward evaluation), BFCL (tool-calling benchmarks), GRPO/PPO (RL for LLM alignment)
