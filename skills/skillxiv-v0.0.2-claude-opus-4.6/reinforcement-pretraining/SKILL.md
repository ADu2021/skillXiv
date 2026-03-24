---
name: reinforcement-pretraining
title: "Reinforcement Pre-Training: RL-Based Scaling for LLM Foundation Development"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.08007"
keywords: [reinforcement-learning, pretraining, scaling-laws, next-token-prediction, reasoning]
description: "Apply reinforcement learning to pre-training by framing next-token prediction as a reasoning task with verifiable rewards, achieving superior scaling compared to standard language modeling."
---

# Reinforcement Pre-Training: RL-Based Scaling for LLM Foundation Development

## Core Concept

Reinforcement Pre-Training (RPT) reframes next-token prediction as a reasoning task trained with reinforcement learning rather than supervised learning. Instead of matching target tokens directly, models generate reasoning chains and receive binary rewards for correctly predicting the next token. This approach leverages vast text corpora as a naturally verifiable reward signal, enabling general-purpose RL during pre-training without external annotations. RPT achieves stronger scaling properties and matches larger supervised baselines using significantly fewer training tokens.

## Architecture Overview

- **Reasoning-Capable Foundation**: Built on models like Deepseek-R1-Distill-Qwen-14B with chain-of-thought capability
- **Verifiable Reward System**: Binary rewards based on exact token prediction, eliminating reward hacking
- **On-Policy GRPO Training**: Group Relative Policy Optimization with typically 8 parallel rollouts
- **Entropy-Based Data Filtering**: Prioritizes challenging token positions where reasoning matters most
- **Scaling Law Alignment**: Consistent power-law improvement with compute, following predictable patterns

## Implementation

### Step 1: Set Up Reward Function and Filtering

Implement the core verifiable reward mechanism:

```python
import torch
import numpy as np
from transformers import AutoTokenizer

class NextTokenRewardFunction:
    """Verifiable reward based on next-token prediction accuracy"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_reward(self, generated_text, ground_truth_text, prediction_mode='prefix'):
        """
        Compute binary reward for next-token prediction.

        Args:
            generated_text: model's generated reasoning + prediction
            ground_truth_text: ground truth completion
            prediction_mode: 'prefix' for exact prefix matching

        Returns:
            reward: 1.0 if generated is exact prefix, 0.0 otherwise
        """
        if prediction_mode == 'prefix':
            # Exact byte-sequence prefix matching
            is_prefix = ground_truth_text.startswith(generated_text)
            return float(is_prefix)

        return 0.0

def entropy_based_filtering(texts, tokenizer, entropy_threshold=2.0):
    """
    Filter to include only challenging token positions.

    Uses Shannon entropy to identify positions where prediction is non-trivial.
    """
    filtered_texts = []

    for text in texts:
        tokens = tokenizer.encode(text)

        # Compute entropy for each position
        entropies = []
        for t in range(1, len(tokens)):
            prev_tokens = tokens[:t]
            next_token = tokens[t]

            # Compute entropy based on token frequency in corpus
            # (simplified; real implementation would use language model)
            entropy = estimate_entropy(prev_tokens, next_token)
            entropies.append(entropy)

        # Keep positions with high entropy (challenging to predict)
        challenging_positions = [i for i, e in enumerate(entropies) if e > entropy_threshold]

        if len(challenging_positions) > 0:
            filtered_texts.append({
                'text': text,
                'challenging_positions': challenging_positions,
                'entropy_mean': np.mean(entropies)
            })

    return filtered_texts

def estimate_entropy(prev_tokens, next_token, vocab_size=32000):
    """Simplified entropy estimation (real version uses full language model)"""
    # In practice, use a pre-trained model to estimate token probabilities
    import math
    return math.log(vocab_size) * 0.8  # Placeholder: most tokens have moderate entropy
```

### Step 2: Implement RPT Training Loop with GRPO

Create the reinforcement pre-training pipeline:

```python
import torch.nn.functional as F
from torch.optim import AdamW

def rpt_training_step(model, prefix_ids, ground_truth_ids, reward_fn, group_size=8,
                      learning_rate=1e-6):
    """
    Single RPT training step using GRPO.

    Args:
        model: Language model
        prefix_ids: Input context
        ground_truth_ids: Ground truth next tokens
        reward_fn: Reward function instance
        group_size: Number of parallel rollouts (G=8 typical)
    """
    batch_size = prefix_ids.shape[0]
    device = prefix_ids.device

    # Generate multiple rollouts from the same prefix
    rollout_ids = []
    rollout_log_probs = []
    rewards = torch.zeros(batch_size, group_size, device=device)

    model.eval()
    with torch.no_grad():
        for g in range(group_size):
            # Generate reasoning chain + next token
            generated = generate_with_reasoning(model, prefix_ids, max_new_tokens=128)
            rollout_ids.append(generated)

            # Compute reward
            for b in range(batch_size):
                generated_text = model.tokenizer.decode(generated[b])
                ground_truth_text = model.tokenizer.decode(ground_truth_ids[b])

                reward = reward_fn.compute_reward(generated_text, ground_truth_text)
                rewards[b, g] = reward

    # Now train with GRPO
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Compute log probabilities for generated sequences
    for g in range(group_size):
        optimizer.zero_grad()

        generated = rollout_ids[g]
        outputs = model(input_ids=generated)
        logits = outputs.logits

        # Compute log probability of generated sequence
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        action_log_probs = log_probs.gather(2, generated[:, 1:].unsqueeze(2)).squeeze(2)
        sequence_log_prob = action_log_probs.sum(dim=1)  # (batch_size,)

        # GRPO advantage estimation
        group_reward = rewards[:, g]
        baseline = rewards.mean(dim=1)  # Value baseline
        advantage = group_reward - baseline

        # Policy gradient loss
        loss = -(sequence_log_prob * advantage.detach()).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {
        'loss': loss.item(),
        'reward_mean': rewards.mean().item(),
        'reward_std': rewards.std().item()
    }

def generate_with_reasoning(model, prefix_ids, max_new_tokens=128, temperature=0.8):
    """Generate reasoning chain before predicting next token"""
    device = prefix_ids.device

    # Phase 1: Generate reasoning (chain-of-thought)
    reasoning_ids = model.generate(
        input_ids=prefix_ids,
        max_new_tokens=min(max_new_tokens, 64),  # Limit reasoning length
        temperature=temperature,
        do_sample=True,
        pad_token_id=model.config.eos_token_id
    )

    # Phase 2: Generate next token(s) after reasoning
    output_ids = model.generate(
        input_ids=reasoning_ids,
        max_new_tokens=max(1, max_new_tokens - 64),
        temperature=temperature,
        do_sample=True,
        pad_token_id=model.config.eos_token_id
    )

    return output_ids
```

### Step 3: Implement Scaling Study Infrastructure

Track scaling laws across different training scales:

```python
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class ScalingStudy:
    def __init__(self, model_size=14):
        self.model_size = model_size
        self.results = []

    def power_law(self, x, a, b):
        """Power law: f(x) = a * x^(-b)"""
        return a * (x ** (-b))

    def run_scaling_experiment(self, token_budgets=[1e8, 5e8, 1e9, 5e9, 1e10]):
        """Run RPT with different compute budgets"""

        for tokens in token_budgets:
            print(f"Training with {tokens:.1e} tokens...")

            # Train RPT model
            model = load_base_model()
            train_loss = train_rpt_model(model, num_tokens=int(tokens))

            # Evaluate on benchmarks
            superglue_score = evaluate_superglue(model)
            mmlu_pro_score = evaluate_mmlu_pro(model)

            self.results.append({
                'tokens': tokens,
                'loss': train_loss,
                'superglue': superglue_score,
                'mmlu_pro': mmlu_pro_score
            })

        return self.results

    def plot_scaling_laws(self):
        """Visualize scaling curves"""
        tokens = [r['tokens'] for r in self.results]
        mmlu_scores = [r['mmlu_pro'] for r in self.results]

        # Fit power law
        popt, _ = curve_fit(self.power_law, tokens, mmlu_scores, p0=[100, 0.3])

        plt.figure(figsize=(10, 6))
        plt.loglog(tokens, mmlu_scores, 'o-', label='RPT-14B')
        plt.loglog(tokens, self.power_law(np.array(tokens), *popt), '--',
                  label=f'Fit: {popt[0]:.2f} * N^(-{popt[1]:.2f})')
        plt.xlabel('Training Tokens')
        plt.ylabel('MMLU-Pro Score')
        plt.legend()
        plt.savefig('scaling_laws.png')
        return popt
```

### Step 4: Comparative Evaluation

Compare RPT against supervised baselines:

```python
def comparative_evaluation(rpt_model, supervised_baseline, test_datasets):
    """
    Compare RPT vs. supervised baselines.

    RPT typically matches 2x+ larger supervised models using fewer tokens.
    """
    results = {
        'rpt': {},
        'supervised': {}
    }

    for dataset_name, dataset in test_datasets.items():
        # Evaluate RPT
        rpt_score = evaluate_dataset(rpt_model, dataset)
        results['rpt'][dataset_name] = rpt_score

        # Evaluate supervised baseline
        supervised_score = evaluate_dataset(supervised_baseline, dataset)
        results['supervised'][dataset_name] = supervised_score

        # Compute improvement
        improvement = ((rpt_score - supervised_score) / supervised_score) * 100
        print(f"{dataset_name}: RPT={rpt_score:.2%}, Supervised={supervised_score:.2%}, "
              f"Improvement={improvement:+.1f}%")

    return results
```

## Practical Guidance

- **Base Model Choice**: Use reasoning-capable models (Deepseek-R1, DeepSeek-V3) as foundation
- **Sequence Length**: Use 8k token sequences; longer contexts reduce FLOPs per token but increase memory
- **Learning Rate**: Start with 1×10⁻⁶; scale down for larger models to prevent instability
- **Group Size**: G=8 provides good balance between gradient quality and computational cost
- **Entropy Filtering**: Prioritize hard examples (high entropy positions) for stable training
- **Reward Design**: Binary prefix matching is stable; avoid continuous rewards that encourage reward hacking
- **Data Selection**: Works best on well-curated, diverse text with challenging predictions
- **Downstream RL**: RPT models serve as excellent foundation for further RL fine-tuning

## Reference

- RPT achieves better scaling by converting supervised loss to RL problem with natural verifiability
- Next-token prediction provides abundant, cost-free reward signal without external annotation
- Reasoning chains enable models to solve harder predictions before committing to token choice
- Scaling laws consistently follow power-law patterns with good fit, enabling predictability
