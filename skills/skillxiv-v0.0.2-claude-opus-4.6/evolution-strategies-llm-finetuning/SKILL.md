---
name: evolution-strategies-llm-finetuning
title: "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.24372"
keywords: [evolution-strategies, llm-finetuning, parameter-optimization, backpropagation-free, reward-modeling, population-based-search, training-stability]
description: "Scale Evolution Strategies to billion-parameter LLMs without backpropagation for superior robustness and stability across diverse models, reward horizons, and evaluation tasks. Outperforms RL methods while eliminating gradient computation overhead."
---

## Evolution Strategies Fine-Tuning: Direct Parameter Optimization at Billion Scale

### Outcome

Fine-tune large language models through population-based direct parameter search, achieving robust model improvements across diverse architectures with 15.5× lower training variance than gradient-based RL methods and resistance to reward hacking without explicit penalties.

### Problem Context

Current LLM fine-tuning relies on backpropagation through gradient-based reinforcement learning (PPO, GRPO), which struggles with:

- **Sparse, long-horizon rewards**: Intermediate supervision often unavailable for reasoning tasks; gradients through long sequences become unstable
- **Reward hacking**: Gradient-based optimization exploits loopholes (short-but-nonsensical outputs) without explicit KL constraints
- **Cross-model brittleness**: Fine-tuning success varies dramatically across base model architectures; GRPO failed entirely on certain models
- **Training instability**: High variance across runs (15.5× higher than ES) makes expensive fine-tuning unreliable for large deployments
- **Computational overhead**: Backpropagation and KL penalty computation add substantial memory and compute burden

Evolution Strategies offer an alternative: direct parameter space search using only reward signals, no gradients required.

### Core Concept

Evolution Strategies treat model parameters as a genome subject to evolutionary pressure. The algorithm repeatedly:

1. Sample parameter perturbations from a normal distribution
2. Evaluate perturbed models on the target task to obtain rewards
3. Update parameters in the direction of high-reward perturbations (natural gradient)

Key insight: ES needs only reward values, not gradients, enabling response-level supervision (did the model solve the problem?) rather than loss gradients. This decouples optimization from model architecture and enables effective search in sparse reward regimes.

At billion-parameter scale, seven engineering optimizations make ES tractable: noise reproducibility via random seeds, parallel GPU evaluation, in-place perturbation, reward normalization, greedy decoding, decomposed updates, and simplified learning rates.

### Architecture Overview

**Population-Based Search**
- Small fixed population (30 members vs. 10,000+ in prior work) evaluates perturbations in parallel
- Each member: base weights + scaled Gaussian noise sampled from seed
- Parallel evaluation across GPUs; single machines or distributed clusters via Hugging Face Accelerate

**Reward-Driven Parameter Updates**
- Collect reward signal (scalar, delayed OK) from each population member
- Normalize rewards to zero-mean unit-variance
- Compute utility-weighted average of perturbations: Δθ ∝ Σ(utility_i × noise_i)
- Apply learning rate: θ_new = θ_old + α × Δθ

**Memory & Compute Efficiency**
- Noise retrieval: reconstruct perturbations from random seeds on-the-fly (no storage overhead)
- Layer-level in-place perturbation: modify weights sequentially, evaluate, restore (single copy in memory)
- Batch GPU evaluation: evaluate multiple perturbed models per GPU via threading
- No backpropagation: ~50% memory reduction vs. gradient methods

**Stability Properties**
- ES update is rank-based utility weighting (robust to reward outliers and scale)
- No explicit KL penalties; ES naturally avoids reward hacking through population diversity
- Variance reduction: 15.5× lower than GRPO across runs on identical problems

### Implementation

#### 1. Environment Setup

Prepare the Python environment and install dependencies for distributed GPU evaluation.

```python
# Create and activate virtual environment
python3.10 -m venv es_env
source es_env/bin/activate

# Install dependencies (from repository)
pip install -r requirements.txt

# Key packages:
# - torch>=2.0.0
# - transformers>=4.40.0
# - accelerate>=0.27.0 (distributed training)
# - datasets>=2.18.0 (data loading)
# - numpy, pandas (utilities)
```

#### 2. Define the Reward Function

The reward function takes a model and returns a scalar score. ES optimizes this directly—no gradients needed.

```python
def compute_reward(model, tokenizer, examples):
    """
    Evaluate model on a task and return scalar reward.

    Args:
        model: LLM instance (already loaded)
        tokenizer: Tokenizer for the model
        examples: List of {input, expected_output} dicts

    Returns:
        float: Aggregated reward (0-1 range recommended)
    """
    correct = 0
    for example in examples:
        # Generate response with greedy decoding
        inputs = tokenizer(example["input"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Check correctness (task-specific)
        if is_correct(response, example["expected_output"]):
            correct += 1

    # Return fraction correct
    return correct / len(examples)


def is_correct(response, expected):
    """Task-specific correctness check."""
    # Example: exact match
    return response.strip() == expected.strip()
```

#### 3. Initialize Population and State

Set up the ES state: mean parameters, step size, and population utilities.

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Flatten parameters into a single vector (for ES state)
params_init = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
num_params = params_init.numel()

# ES hyperparameters
population_size = 30  # Small population due to engineering optimizations
learning_rate = 0.001
sigma = 0.017  # Standard deviation of perturbations (tune per task)

print(f"Model parameters: {num_params:,} | Population: {population_size}")

# Initialize utilities (per-member weighting)
utilities = np.array([max(0, np.log(population_size/2 + 1) - np.log(i+1))
                       for i in range(population_size)])
utilities /= np.sum(utilities)  # Normalize
```

#### 4. Main ES Loop: Mutation, Evaluation, and Update

Run ES for multiple generations, accumulating rewards and updating parameters.

```python
def es_train_loop(
    model, tokenizer, params_init, reward_fn,
    generations=100, population_size=30, sigma=0.017, lr=0.001,
    seed_base=42, device="cuda"
):
    """
    Main Evolution Strategies training loop.

    Args:
        model: LLM to fine-tune
        tokenizer: Model tokenizer
        params_init: Initial parameter vector
        reward_fn: Function(model, tokenizer) -> float
        generations: Number of ES iterations
        population_size: Population members per iteration
        sigma: Perturbation std dev (controls exploration)
        lr: Natural gradient step size
        seed_base: RNG seed for reproducibility
        device: "cuda" or "cpu"
    """
    params_current = params_init.clone()
    rewards_history = []

    for gen in range(generations):
        gen_rewards = []
        param_updates = np.zeros(params_init.numel())

        # Generate and evaluate population
        for member_id in range(population_size):
            # Deterministic noise from seed (no storage overhead)
            seed = seed_base + gen * population_size + member_id
            np.random.seed(seed)
            noise = torch.tensor(
                np.random.randn(params_init.numel()),
                dtype=params_init.dtype,
                device=device
            )

            # Perturbed parameters
            params_perturbed = params_current + sigma * noise

            # Update model weights in-place (layer by layer)
            offset = 0
            for param in model.parameters():
                param_size = param.numel()
                param.data = params_perturbed[offset:offset+param_size].reshape(param.shape)
                offset += param_size

            # Evaluate (reward only, no gradients)
            reward = reward_fn(model, tokenizer)
            gen_rewards.append(reward)

            # Accumulate utility-weighted noise for update
            param_updates += utilities[member_id] * noise.cpu().numpy()

        # Normalize rewards and update parameters
        rewards_array = np.array(gen_rewards)
        rewards_normalized = (rewards_array - np.mean(rewards_array)) / (np.std(rewards_array) + 1e-8)

        # Natural gradient update: θ ← θ + α * (1/σ) * Σ util_i * noise_i * (r_i - mean_r)
        param_updates_weighted = np.zeros_like(param_updates)
        for member_id in range(population_size):
            seed = seed_base + gen * population_size + member_id
            np.random.seed(seed)
            noise_update = np.random.randn(params_init.numel())
            param_updates_weighted += utilities[member_id] * noise_update * rewards_normalized[member_id]

        params_current = params_current.cpu() + (lr / sigma) * torch.tensor(param_updates_weighted, dtype=params_current.dtype)
        params_current = params_current.to(device)

        # Log progress
        best_reward = np.max(gen_rewards)
        mean_reward = np.mean(gen_rewards)
        rewards_history.append(best_reward)

        if (gen + 1) % 10 == 0:
            print(f"Gen {gen+1:3d} | Best: {best_reward:.4f} | Mean: {mean_reward:.4f} | Std: {np.std(gen_rewards):.4f}")

    return params_current, rewards_history
```

#### 5. Save and Evaluate Fine-Tuned Model

After training, restore final parameters and test performance.

```python
def save_finetuned_model(model, params_final, output_path):
    """
    Write final parameters back to model and save to disk.

    Args:
        model: LLM with architecture to save
        params_final: Final parameter vector from ES
        output_path: Directory to save (will create via model.save_pretrained)
    """
    # Restore final parameters
    offset = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data = params_final[offset:offset+param_size].reshape(param.shape)
        offset += param_size

    # Save to disk
    model.save_pretrained(output_path)
    print(f"Fine-tuned model saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Load data (example: math reasoning)
    train_examples = [
        {"input": "Solve: 2x + 3 = 7", "expected_output": "x = 2"},
        # ... more examples
    ]

    # Define reward function
    def reward_fn(m, t):
        return compute_reward(m, t, train_examples[:20])  # Subset for speed

    # Run ES fine-tuning
    params_final, history = es_train_loop(
        model, tokenizer, params_init, reward_fn,
        generations=100,
        population_size=30,
        sigma=0.017,
        lr=0.001
    )

    # Save and evaluate
    save_finetuned_model(model, params_final, "./model_finetuned")
```

#### 6. Distributed Multi-GPU Setup (via Accelerate)

For large models, distribute population evaluation across multiple GPUs or machines.

```python
from accelerate import Accelerator

def es_train_distributed(
    model_name, reward_fn,
    generations=100, population_size=30,
    num_processes=2, gpu_threads=15
):
    """
    Multi-GPU ES training using Hugging Face Accelerate.
    Total parallel evaluations = num_processes * gpu_threads.

    Args:
        model_name: HuggingFace model ID
        reward_fn: Reward function (called per process)
        num_processes: Number of GPUs (or machines)
        gpu_threads: Threads per GPU (model copies per GPU)
    """
    accelerator = Accelerator()

    # Each process loads model independently
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = accelerator.prepare(model)

    # Each process evaluates a subset of population
    local_pop_size = population_size // num_processes

    # Main ES loop (same as single-GPU, but rewards aggregated)
    # ...

    print(f"Rank {accelerator.process_index}: evaluating {local_pop_size} members")
```

### Practical Guidance

#### Hyperparameter Recommendations

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| `population_size` | 20–50 | Smaller than RL batch sizes; 30 is default. Increase for harder tasks. |
| `sigma` (noise std) | 0.01–0.05 | Controls exploration vs. exploitation. Start at 0.017; lower for final refinement. |
| `learning_rate` | 0.0001–0.01 | Step size for parameter updates. 0.001 is standard; reduce if oscillating. |
| `generations` | 50–500 | Task-dependent; monitor reward curve to detect plateau. |
| `seed_base` | any | Ensures reproducibility; increment per run if multiple trials needed. |

#### When to Use ES Fine-Tuning

- **Reasoning tasks** with sparse, delayed rewards (math, logic, puzzle solving)
- **Heterogeneous base models**: Need a method that works across Qwen, Llama, Mistral, etc.
- **Robustness critical**: Training stability matters more than marginal reward gains
- **Reward specification difficult**: You have outcome labels but not intermediate supervision
- **Small datasets**: ES is sample-efficient (often < 20% of RL data needed)
- **Long-horizon tasks**: Few intermediate steps; only final answer is evaluable

#### When NOT to Use ES Fine-Tuning

- **Dense reward signals**: If you have loss gradients or detailed intermediate supervision, gradient-based RL (PPO, DPO) will be faster
- **Continuous action spaces**: ES excels at large discrete parameter spaces; for action fine-tuning, RL is more direct
- **Extreme speed required**: ES requires multiple forward passes per update; if latency is critical, SFT or single-pass methods preferred
- **Highly model-specific optimization**: If you're tuning for a single model and have unlimited compute for gradient tuning, RL may squeeze out extra performance
- **Limited evaluation budget**: Each generation requires `population_size` full model evaluations; if evaluation is expensive (e.g., human-in-the-loop), use smaller populations or RL with importance weighting

#### Common Pitfalls

1. **Σ too high or low**: If noise is too large, updates become random. If too small, stuck in local optima. Adapt σ per task (start 0.017, halve if rewards plateau).

2. **Ignoring reward scale**: Normalizing rewards per generation is critical for stable updates. If rewards are 0–1 vs. 0–1000, learning rate must adjust; the algorithm handles this via z-score normalization.

3. **Small population on large tasks**: With population_size < 15, gradient estimates become noisy. For complex reasoning, use 30+.

4. **Not greedy decoding**: ES assumes deterministic reward (same input → same output). Sampling during generation adds noise; use greedy decoding or fix seed.

5. **Starting from mid-training checkpoint**: ES searches from the current parameter point; if base model is undertrained, ES may optimize for weak behaviors. Fine-tune strong base models.

6. **Incorrect utility weights**: The utility vector ranks population members by reward. Ensure it's recalculated per generation (don't reuse across different tasks).

### Reference

**Paper**: Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning
**Authors**: Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Elliot Meyerson, Babak Hodjat, Risto Miikkulainen
**ArXiv**: [2509.24372](https://arxiv.org/abs/2509.24372)
**Code**: [GitHub – Cognizant AI Lab](https://github.com/cognizant-ai-lab/es-fine-tuning)

**Cited Baselines**: PPO (Schulman et al., 2017), GRPO (Xu et al., 2024), DPO (Rafailov et al., 2023)
