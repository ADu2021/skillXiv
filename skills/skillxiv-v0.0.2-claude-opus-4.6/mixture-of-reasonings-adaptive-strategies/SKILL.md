---
name: mixture-of-reasonings-adaptive-strategies
title: "Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.00606"
keywords: [Reasoning, LLM Training, Chain-of-Thought, Prompt Engineering, Adaptive Strategies]
description: "Enable LLMs to autonomously select and apply diverse reasoning strategies without prompt engineering. Trains models with diverse thought templates covering 50-500 distinct reasoning approaches, achieving 2-13% improvements over baseline prompting methods."
---

# Mixture of Reasonings: Teaching LLMs to Reason Independently

Current large language models depend heavily on prompt engineering. Chain-of-Thought prompting improves reasoning by 2-3%, but it requires task-specific design and must be retuned for different problem types. What if models could learn to recognize when to use step-by-step reasoning, when to use analogies, when to decompose problems, and when to use direct inference—all without explicit instructions?

MoR (Mixture of Reasonings) trains this capability directly into model parameters. Instead of relying on external prompts, it creates diverse reasoning templates covering dozens of strategies, pairs them with actual problems and correct answers, and fine-tunes the model to embed reasoning flexibility. The result: models that autonomously apply effective reasoning without external guidance, generalizing across diverse problem types.

## Core Concept

The key insight is that effective reasoning isn't about finding the one best strategy—it's about having multiple strategies available and knowing when to apply each. MoR implements this through:

1. **Diverse Reasoning Template Generation**: Using GPT-4o to create 50-500 distinct reasoning chain templates that cover different cognitive approaches (step-by-step decomposition, analogy-based reasoning, constraint satisfaction, backward chaining, etc.)

2. **Dataset Construction**: Pairing each template with benchmark problems and filtering for correct responses, creating a supervised learning corpus that teaches when different strategies work

3. **Embedded Reasoning**: Fine-tuning models via SFT (Supervised Fine-Tuning) so reasoning patterns become part of the model's learned parameters rather than external prompts

This approach eliminates the gap between prompting (fast but task-specific) and in-context few-shot examples (general but expensive). Models learn reasoning at training time and apply it flexibly at inference.

## Architecture Overview

The MoR training pipeline consists of these components:

- **Template Generation Engine**: GPT-4o generating diverse reasoning chains covering 10+ strategy classes (decomposition, analogy, simulation, constraint-based, etc.)
- **Template Filtering**: Validation that templates produce correct answers on benchmark datasets
- **SFT Dataset Construction**: Pairing problem instances with diverse templates and correct outputs
- **Fine-tuning Pipeline**: LoRA or full fine-tuning on the template-augmented corpus
- **Inference Mechanism**: Model generates reasoning automatically; no external prompt required
- **Multi-task Evaluation**: Testing across math, QA, reasoning tasks to verify generalization

## Implementation

This section demonstrates how to implement Mixture of Reasonings for your models.

**Step 1: Generate diverse reasoning templates using GPT-4o**

This code creates multiple reasoning strategies for a problem type:

```python
import openai
import json
from typing import List

def generate_reasoning_templates(problem_type: str, num_templates: int = 100) -> List[str]:
    """
    Use GPT-4o to generate diverse reasoning templates for a problem type.
    Each template represents a different cognitive strategy.
    """

    template_prompt = f"""You are an expert in reasoning strategies. For {problem_type} problems, generate {num_templates} diverse reasoning approaches that could solve such problems. Each approach should:
1. Use a different cognitive strategy (e.g., step-by-step decomposition, analogical reasoning, backward chaining, constraint satisfaction, simulation-based reasoning, etc.)
2. Be distinctive from the others
3. Be explicit enough to guide a language model
4. Cover both simple and complex problem-solving strategies

Format each as a reasoning template that starts with a prompt fragment like:
- "Let's break this into steps..."
- "I can think of a similar problem..."
- "Let me work backwards from the answer..."
- "The constraints are..."
- "I can visualize this situation..."

Output only the templates, one per line, starting with the strategy name in brackets."""

    templates = []

    # Use GPT-4o with longer context for diverse output
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": template_prompt}],
        temperature=0.9,  # Higher temperature for diversity
        max_tokens=3000
    )

    raw_templates = response.choices[0].message.content.split('\n')
    templates = [t.strip() for t in raw_templates if t.strip() and len(t) > 10]

    return templates[:num_templates]  # Return requested count

# Generate templates for math reasoning
math_templates = generate_reasoning_templates("arithmetic and algebra problems", num_templates=150)
print(f"Generated {len(math_templates)} math reasoning templates")
for template in math_templates[:5]:
    print(f"  - {template[:80]}...")
```

This generates diverse reasoning strategies using GPT-4o's creative capabilities.

**Step 2: Create SFT dataset by pairing templates with problems and answers**

This code constructs training data with diverse reasoning paths:

```python
import random
from datasets import load_dataset

def create_mixture_sft_dataset(template_strategies: List[str], benchmark_dataset, max_samples=10000):
    """
    Create SFT training data by pairing reasoning templates with problems.
    Each problem is solved using multiple templates; we keep only correct solutions.
    """

    sft_examples = []

    for idx, problem in enumerate(benchmark_dataset):
        if idx >= max_samples:
            break

        question = problem.get('question', problem.get('text', ''))
        answer = problem.get('answer', problem.get('label', ''))

        # Try multiple reasoning templates for this problem
        for template in random.sample(template_strategies, min(5, len(template_strategies))):
            # Use GPT-4o to execute this reasoning template on this problem
            reasoning_prompt = f"""{template}

Problem: {question}

Think step by step and provide the answer."""

            reasoning_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": reasoning_prompt}],
                temperature=0.7,
                max_tokens=500
            )

            predicted = reasoning_response.choices[0].message.content

            # Validate: is this response correct?
            validation_prompt = f"""Is this response correct for the problem?
Problem: {question}
Expected answer: {answer}
Predicted response: {predicted}

Respond with "CORRECT" or "INCORRECT"."""

            validation = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": validation_prompt}],
                max_tokens=20
            )

            is_correct = "CORRECT" in validation.choices[0].message.content.upper()

            # Add to dataset only if correct (supervised learning of good reasoning)
            if is_correct:
                sft_examples.append({
                    'instruction': question,
                    'reasoning_template': template,
                    'reasoning_chain': predicted,
                    'answer': answer
                })

    return sft_examples

# Create dataset
benchmark = load_dataset("math_qa", split="train")
sft_data = create_mixture_sft_dataset(math_templates, benchmark, max_samples=5000)
print(f"Created {len(sft_data)} SFT examples with diverse reasoning")

# Save for fine-tuning
import json
with open("mixture_of_reasonings_sft.json", 'w') as f:
    json.dump(sft_data, f)
```

This creates a training dataset where each problem has multiple correct reasoning paths.

**Step 3: Fine-tune model with LoRA on reasoning diversity**

This code adapts a base model to learn diverse reasoning strategies:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json

# Load base model
model_name = "meta-llama/Llama-2-7b"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

class MixtureReasoningSFTDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Construct training text: problem + reasoning_chain + answer
        # This teaches the model to generate reasoning followed by answer
        text = f"""Question: {example['instruction']}

Reasoning: {example['reasoning_chain']}

Answer: {example['answer']}"""

        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze()
        }

# Create dataset and dataloader
dataset = MixtureReasoningSFTDataset("mixture_of_reasonings_sft.json", tokenizer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Fine-tuning loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()

num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} complete. Avg loss: {total_loss / len(dataloader):.4f}")

print("Fine-tuning complete!")
```

This adapts the model to learn diverse reasoning strategies through supervised fine-tuning.

**Step 4: Evaluate reasoning capability across diverse tasks**

This code tests whether the model generalizes reasoning across problem types:

```python
def evaluate_reasoning_generalization(model, tokenizer, test_datasets=['math_qa', 'arc', 'commonsenseqa']):
    """
    Evaluate whether MoR-trained model generalizes reasoning across different task types.
    """

    results = {}

    for dataset_name in test_datasets:
        if dataset_name == 'math_qa':
            test_data = load_dataset("math_qa", split="test")
        elif dataset_name == 'arc':
            test_data = load_dataset("arc", split="test")
        elif dataset_name == 'commonsenseqa':
            test_data = load_dataset("commonsenseqa", split="test")
        else:
            continue

        correct = 0
        total = min(len(test_data), 200)  # Test on first 200 examples

        for idx, example in enumerate(test_data):
            if idx >= total:
                break

            question = example.get('question', example.get('text', ''))

            # Model generates reasoning and answer autonomously (no external prompt)
            prompt = f"Question: {question}\n\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'].to(model.device),
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.95
                )

            predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer from generated text (simplified; real implementation parses)
            expected_answer = example.get('answer', example.get('answerKey', ''))

            if expected_answer.lower() in predicted.lower():
                correct += 1

        accuracy = correct / total
        results[dataset_name] = accuracy
        print(f"{dataset_name}: {accuracy:.2%} accuracy ({correct}/{total})")

    return results

# Evaluate on diverse tasks
results = evaluate_reasoning_generalization(model, tokenizer)
print(f"\nAverage accuracy across tasks: {sum(results.values()) / len(results):.2%}")
```

This evaluates whether reasoning learned on one task transfers to others.

## Practical Guidance

**When to use Mixture of Reasonings:**
- Fine-tuning language models where diverse problem-solving is needed
- Tasks where chain-of-thought prompting helps but requires constant tuning
- Applications across multiple problem types where generalization matters
- Scenarios where inference-time prompt engineering is impractical
- Models where embedding reasoning patterns is feasible (not resource-constrained inference)

**When NOT to use:**
- Simple classification tasks without reasoning requirements
- Real-time systems with strict latency constraints (fine-tuning adds overhead)
- Tasks where single, deterministic strategies are optimal
- Very large models where LoRA fine-tuning becomes expensive
- Domains with limited labeled reasoning data (need sufficient training examples)

**Hyperparameters and Configuration:**

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Number of Templates | 150-500 | More templates → more diversity; diminishing returns after 300 |
| LoRA Rank | 16-32 | Higher for more capacity; 16 sufficient for reasoning specialization |
| Learning Rate | 2e-4 | Standard for LLM fine-tuning; reduce if diverging |
| Epochs | 3-5 | Reasoning requires fewer epochs than general language modeling |
| Batch Size | 8-16 | Diversity of reasoning paths benefits from smaller batches |
| Max Sequence Length | 512-1024 | Sufficient for problem + reasoning chain + answer |
| Template per Problem | 3-5 | Balances diversity with dataset size |

**Common Pitfalls:**
- Using too few templates—don't cover diverse strategies
- Not filtering incorrect reasoning chains—teaches bad patterns
- Fine-tuning on single task only—doesn't generalize to new domains
- Using generic templates—must be specific enough to guide reasoning
- Forgetting to test on out-of-distribution tasks—evaluate true generalization
- Overtraining on templates—causes overfitting to specific phrasings

**Key Design Decisions:**
MoR separates template generation (done via GPT-4o, captures human reasoning diversity) from fine-tuning (adapts base model to leverage templates). This hybrid approach combines the creativity of larger models with the efficiency of smaller fine-tuned models. Templates are created once, reused across many problems, and validated before training—ensuring that models learn genuine reasoning patterns, not artifacts. By embedding reasoning into parameters rather than prompts, inference becomes parameter-efficient and generalizable.

## Reference

Li, X., Yu, Z., Wang, J., Zhang, Z., Zhang, Y., & Sun, Y. (2025). Mixture of Reasonings: Teach Large Language Models to Reason with Adaptive Strategies. arXiv preprint arXiv:2507.00606. https://arxiv.org/abs/2507.00606
