---
name: reasoning-vectors-cot-task-arithmetic
title: "Reasoning Vectors: Transferring Chain-of-Thought Capabilities via Task Arithmetic"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.01363"
keywords: [task vectors, chain-of-thought, reasoning transfer, model merging, weight arithmetic, fine-tuning, reinforcement learning, capability composition]
description: "Extract and transfer reasoning capabilities between language models using task vectors derived from supervised fine-tuning and reinforcement learning weight differences. Apply reasoning vectors via simple arithmetic to enhance any compatible instruction-tuned model without retraining."
---

# Transferring Chain-of-Thought Reasoning via Task Vectors

## Outcome

Enhance language model reasoning capabilities on mathematical and logical tasks by adding a pre-computed reasoning vector to any compatible instruction-tuned model. Achieve +4.9% on GSM8K, +4.3% on HumanEval, and +12.3% on BigBenchHard without retraining or fine-tuning.

## Problem Context

Reasoning capabilities in language models typically require expensive reinforcement learning optimization. Once learned, these capabilities remain locked inside trained models and cannot be easily transferred or reused. Organizations must choose between expensive full-model retraining or accepting degraded reasoning performance on smaller or different models. The computational cost to develop reasoning-capable models limits adoption and prevents efficient capability recycling across the model zoo.

## Core Concept

Reasoning vectors are compact weight perturbations that capture the difference between reinforcement-learning-optimized and supervised-fine-tuned versions of the same model trained on identical datasets. By computing the weight difference, you isolate reasoning patterns—the mathematical insight is that reasoning improvements introduced by RL remain additive and can be extracted as: `reasoning_vector = RL_weights - SFT_weights`.

This vector operates in weight space, not activation space, enabling direct arithmetic with any compatible model. The key innovation is recognizing that reasoning knowledge occupies learnable subspaces within neural networks that generalize across model architectures when expressed as weight perturbations. Rather than memorizing dataset-specific outputs, the vector captures transferable optimization patterns that improve reasoning on held-out tasks.

## Architecture Overview

- **Vector Extraction**: Two identically initialized models (SFT and RL) trained on the same reasoning dataset produce a reasoning vector through subtraction
- **Vector Application**: Add the reasoning vector to base instruction-tuned models using a scaling factor to control transfer intensity
- **Composability**: Multiple reasoning vectors can be combined additively for specialized reasoning domains
- **Model Compatibility**: Works across different base model families (LLaMA, Mistral, Qwen) provided they share similar architectures and parameter counts

The approach factors out task-specific knowledge (captured in SFT) from reasoning optimization (captured in RL delta), creating a pure reasoning vector. This isolation enables reliable transfer without contaminating target models with dataset artifacts.

## Implementation

### Step 1: Prepare Models and Datasets

Start with two identically initialized base models and a curated reasoning dataset. The dataset should contain chain-of-thought examples (mathematical problems with step-by-step solutions, logical inference tasks, etc.). Use standard CoT datasets like GSM8K (math word problems), MATH (competition problems), or HumanEval (code generation).

Initialize both models with identical seeds and parameters to ensure the only difference is the training procedure.

```python
import torch
from transformers import AutoModel, AutoTokenizer
import copy

# Load base model twice with identical seeds
torch.manual_seed(42)
base_model_state = torch.load("qwen2.5-base.pt")

# Model 1: Will be fine-tuned with SFT
sft_model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B")
sft_model.load_state_dict(copy.deepcopy(base_model_state))

# Model 2: Will be fine-tuned with RL
rl_model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B")
rl_model.load_state_dict(copy.deepcopy(base_model_state))

# Load reasoning dataset
reasoning_dataset = load_dataset("gsm8k", "main")
train_data = reasoning_dataset["train"]

print(f"Base model parameters: {sum(p.numel() for p in sft_model.parameters())}")
print(f"Training examples: {len(train_data)}")
```

### Step 2: Apply Supervised Fine-Tuning

Fine-tune the first model using standard supervised learning on the CoT dataset. Each example pairs a problem with its correct solution showing all reasoning steps. Use cross-entropy loss between predicted tokens and ground-truth solution tokens.

```python
from torch.optim import AdamW
from torch.utils.data import DataLoader

def fine_tune_sft(model, dataset, epochs=3, batch_size=8):
    """Supervised fine-tuning on reasoning dataset"""
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Tokenize problem + solution
            inputs = tokenizer(
                batch["question"] + " " + batch["answer"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Compute cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model

sft_model = fine_tune_sft(sft_model, train_data)
sft_weights = {name: param.clone().detach()
               for name, param in sft_model.named_parameters()}
```

### Step 3: Apply Reinforcement Learning Optimization

Fine-tune the second model using group relative policy optimization (GRPO), a reward-guided RL algorithm. GRPO groups completions, computes relative rewards within groups, and optimizes the policy to maximize reasoning accuracy. Start from the same initialization and use the same dataset for direct comparison.

```python
def group_relative_policy_optimization(
    model, dataset, reward_model, num_epochs=3, batch_size=8, beta=0.01
):
    """GRPO: optimize reasoning via RL with relative rewards"""
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        for batch in DataLoader(dataset, batch_size=batch_size):
            # Generate completions (step-by-step reasoning)
            with torch.no_grad():
                completions = []
                for problem in batch["question"]:
                    completion = model.generate(
                        tokenizer.encode(problem, return_tensors="pt"),
                        max_length=512,
                        num_return_sequences=4,  # Group size
                        temperature=0.7
                    )
                    completions.append(completion)

            # Compute rewards (correctness of reasoning steps)
            rewards = []
            for completion_group in completions:
                group_rewards = []
                for completion in completion_group:
                    text = tokenizer.decode(completion)
                    # Check if final answer matches ground truth
                    is_correct = check_correctness(text, batch["answer"])
                    group_rewards.append(float(is_correct))
                rewards.append(group_rewards)

            # Relative reward: shift within group to compare solutions
            relative_rewards = []
            for group_rewards in rewards:
                group_mean = sum(group_rewards) / len(group_rewards)
                relative = [r - group_mean for r in group_rewards]
                relative_rewards.append(relative)

            # Policy gradient optimization
            log_probs_list = []
            for i, completion_group in enumerate(completions):
                for j, completion in enumerate(completion_group):
                    input_ids = completion.unsqueeze(0)
                    outputs = model(input_ids, output_hidden_states=False)
                    log_probs = torch.nn.functional.log_softmax(
                        outputs.logits, dim=-1
                    )
                    action_log_prob = log_probs[0, -1, completion[-1]].unsqueeze(0)
                    log_probs_list.append(
                        action_log_prob * relative_rewards[i][j]
                    )

            # Compute policy loss with entropy regularization
            policy_loss = -sum(log_probs_list) / len(log_probs_list)
            entropy = -torch.mean(torch.sum(
                torch.nn.functional.softmax(outputs.logits, dim=-1) *
                torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                dim=-1
            ))
            loss = policy_loss - beta * entropy

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

rl_model = group_relative_policy_optimization(
    rl_model, train_data, reward_model=None
)
rl_weights = {name: param.clone().detach()
              for name, param in rl_model.named_parameters()}
```

### Step 4: Extract Reasoning Vector

Compute the weight difference between the RL-optimized and SFT-optimized models. This vector isolates reasoning improvements from shared knowledge. Store the vector for later application to other models.

```python
def extract_reasoning_vector(sft_weights, rl_weights):
    """Extract reasoning vector: RL - SFT"""
    reasoning_vector = {}

    for name in sft_weights.keys():
        if name in rl_weights:
            reasoning_vector[name] = rl_weights[name] - sft_weights[name]
        else:
            print(f"Warning: {name} not in RL weights")

    return reasoning_vector

reasoning_vector = extract_reasoning_vector(sft_weights, rl_weights)

# Calculate vector statistics
total_params = sum(v.numel() for v in reasoning_vector.values())
avg_magnitude = sum(torch.abs(v).mean() for v in reasoning_vector.values()) / len(reasoning_vector)
max_magnitude = max(torch.abs(v).max() for v in reasoning_vector.values())

print(f"Reasoning vector: {total_params} parameters")
print(f"Average magnitude: {avg_magnitude:.6f}")
print(f"Max magnitude: {max_magnitude:.6f}")

# Save vector for reuse
torch.save(reasoning_vector, "reasoning_vector.pt")
```

### Step 5: Apply Reasoning Vector to Target Models

Load any instruction-tuned model and add the reasoning vector using a scaling factor alpha. Start with alpha=1.0 and adjust based on the target model size and task. Smaller alpha (0.5-0.7) works better for very different architectures or smaller models.

```python
def apply_reasoning_vector(target_model, reasoning_vector, alpha=1.0):
    """Add reasoning vector to target model weights"""
    target_weights = dict(target_model.named_parameters())

    for name, vector in reasoning_vector.items():
        if name in target_weights:
            # Add scaled vector: target + alpha * reasoning_vector
            with torch.no_grad():
                target_weights[name].add_(vector * alpha)
        else:
            print(f"Warning: {name} not in target model")

    return target_model

# Load target instruction-tuned model
target_model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Apply reasoning vector
alpha = 0.8  # Adjust based on model size
reasoning_vector = torch.load("reasoning_vector.pt")
enhanced_model = apply_reasoning_vector(target_model, reasoning_vector, alpha=alpha)

print("Reasoning vector applied. Model ready for evaluation.")
```

### Step 6: Evaluate Enhanced Model

Test the enhanced model on held-out reasoning benchmarks. Compare against the baseline model and direct fine-tuning approaches. Measure accuracy, reasoning quality, and generalization to unseen task variants.

```python
def evaluate_reasoning(model, test_dataset, num_samples=100):
    """Evaluate model on reasoning tasks"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, example in enumerate(test_dataset[:num_samples]):
            problem = example["question"]
            ground_truth = example["answer"]

            # Generate reasoning response
            input_ids = tokenizer.encode(problem, return_tensors="pt")
            output = model.generate(
                input_ids,
                max_length=512,
                temperature=0.0,
                do_sample=False
            )
            prediction = tokenizer.decode(output[0])

            # Check if answer is correct
            is_correct = check_correctness(prediction, ground_truth)
            if is_correct:
                correct += 1
            total += 1

            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{num_samples}, Accuracy: {correct/total:.2%}")

    final_accuracy = correct / total
    print(f"Final Accuracy: {final_accuracy:.2%}")
    return final_accuracy

# Load test set
test_data = reasoning_dataset["test"]

# Evaluate baseline
baseline_model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
baseline_acc = evaluate_reasoning(baseline_model, test_data)

# Evaluate enhanced model
enhanced_acc = evaluate_reasoning(enhanced_model, test_data)

print(f"Baseline: {baseline_acc:.2%}")
print(f"Enhanced: {enhanced_acc:.2%}")
print(f"Improvement: +{(enhanced_acc - baseline_acc):.2%}")
```

### Step 7: Compose Multiple Reasoning Vectors (Optional)

Combine reasoning vectors for different domains (math, code, logic) additively. This enables multi-domain reasoning enhancement without retraining.

```python
def compose_reasoning_vectors(vectors_dict, weights=None):
    """Combine multiple reasoning vectors with optional weighting"""
    if weights is None:
        weights = {name: 1.0 for name in vectors_dict.keys()}

    composed = None
    for domain, vector in vectors_dict.items():
        alpha = weights.get(domain, 1.0)
        if composed is None:
            composed = {k: v * alpha for k, v in vector.items()}
        else:
            for name in composed.keys():
                if name in vector:
                    composed[name] += vector[name] * alpha

    return composed

# Load multiple reasoning vectors
math_vector = torch.load("reasoning_vector_math.pt")
code_vector = torch.load("reasoning_vector_code.pt")
logic_vector = torch.load("reasoning_vector_logic.pt")

# Compose with equal weights
composed_vector = compose_reasoning_vectors({
    "math": math_vector,
    "code": code_vector,
    "logic": logic_vector
}, weights={"math": 1.0, "code": 0.8, "logic": 0.6})

# Apply composed vector
enhanced_multi = apply_reasoning_vector(target_model, composed_vector, alpha=0.9)
```

## Practical Guidance

### Hyperparameters and Configuration

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| alpha (scaling factor) | 0.3–1.5 | 0.8 | Controls reasoning intensity; lower values safer for dissimilar models |
| SFT learning rate | 1e-5 to 5e-5 | 2e-5 | Balance between convergence and stability |
| RL learning rate | 5e-6 to 2e-5 | 1e-5 | Typically lower than SFT to stabilize policy optimization |
| GRPO beta (entropy) | 0.001–0.1 | 0.01 | Higher values encourage exploration in reasoning paths |
| Group size | 2–8 | 4 | More completions per problem improve relative reward signal |
| Epochs | 1–5 | 3 | More epochs on small curated datasets; 1-2 on large datasets |
| Batch size | 4–32 | 8 | Smaller on limited VRAM; affects gradient stability |

### When to Use

- Enhancing reasoning on mathematical word problems (GSM8K, MATH)
- Improving code generation (HumanEval, MBPP)
- Boosting performance on logical reasoning benchmarks
- Creating lightweight reasoning-aware models without full RL optimization
- Transferring reasoning capabilities across the model family (different sizes, architectures)
- Composing specialized reasoning for multi-domain tasks
- Recycling investment in previously trained models

### When NOT to Use

- Reasoning vector from one model family (e.g., Qwen) will not transfer reliably to fundamentally different architectures (e.g., Transformer vs. SSM-based models)
- Models with significantly different sizes (e.g., 1.5B source → 70B target) show diminished transfer; alpha adjustment alone may not recover performance
- Non-reasoning tasks (summarization, translation, general QA) do not benefit; the vector captures specialized CoT optimization
- If source and target models have different initializations or training procedures, the vector may introduce harmful interference
- Real-time or latency-critical applications where the vector composition overhead matters
- Instruction-tuned models that have already undergone extensive post-training RL may show minimal gains or negative interference
- When source reasoning data distribution differs significantly from target task distribution (vector overfits to source domain)

### Common Pitfalls

**Alpha Scaling**: Too high (alpha > 1.2) risks catastrophic forgetting or instability in downstream capabilities. Too low (alpha < 0.3) provides negligible reasoning gains. Empirically tune on a validation set.

**Architecture Mismatch**: Vectors are parameter-count-aware. Do not apply a vector from a 7B model directly to a 13B model without interpolation or re-scaling. Parameter alignment is critical.

**Contaminated Reasoning Vectors**: If the SFT model already contains implicit reasoning knowledge (e.g., from massive web-scale pretraining), the extracted vector is contaminated. Use identical base models to minimize this.

**Accumulation in Composition**: Combining many reasoning vectors (>3) additively can lead to magnitude explosion. Use weighted composition with careful alpha tuning for multi-vector scenarios.

**Dataset Overfitting**: If the reasoning dataset is very small (<1000 examples) or unrepresentative, the vector captures dataset artifacts rather than transferable reasoning patterns. Use diverse, well-curated CoT data.

**Verification Before Deployment**: Always evaluate on held-out benchmarks (different from training tasks) to confirm the vector improves reasoning generally, not just on the source task.

## Reference

Paper: "Reasoning Vectors: Transferring Chain-of-Thought Capabilities via Task Arithmetic"
Authors: Mohammad Zbeeb, Hasan Abed Al Kader Hammoud, Bernard Ghanem
arXiv: https://arxiv.org/abs/2509.01363
Status: Under Review (submitted September 1, 2025)

See also: [Qwen2.5-Math Technical Report](https://arxiv.org/html/2409.12122v1) for details on the RL optimization pipeline used in the paper.
