---
name: math-reasoning-transfer
title: "Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.00432"
keywords: [Math Reasoning, Transfer Learning, Reinforcement Learning, Representation Drift, Generalization]
description: "Understand why math reasoning improvements don't always transfer to general capabilities. Use RL-based training instead of SFT to preserve representation structure and enable broader generalization."
---

# Math Reasoning Transfer: When Specialization Preserves Generality

Training a language model to excel at mathematics—adding chain-of-thought prompting, reinforcement learning on mathematical reasoning, instruction tuning on problem sets—seems like an obvious way to improve general reasoning. Yet most models trained this way experience "catastrophic forgetting": they become better at math but worse at reading comprehension, commonsense reasoning, and other non-mathematical tasks. This creates a dilemma: improve specialized math performance at the cost of general capabilities, or maintain general abilities while sacrificing math performance.

The core problem is not math training itself, but *how* the model is trained. Supervised fine-tuning (SFT) on math examples causes the model's internal representations to drift away from their original structure—the patterns that made the model generally useful in the first place. Reinforcement learning (RL), by contrast, adjusts the model's behavior while preserving its representation structure through KL regularization, allowing math improvements to coexist with general capabilities.

## Core Concept

The key insight is that **representation structure matters more than task data**. When you SFT on math data:

1. The model's hidden layers shift to prioritize mathematical patterns
2. Non-mathematical reasoning patterns (needed for other tasks) degrade
3. Transfer is blocked because the general reasoning capability is damaged

When you RL-train on math with KL regularization:

1. The model learns to prefer better math outputs
2. But KL keeps hidden representations close to the original
3. General reasoning patterns remain intact, enabling transfer

The surprising finding: RL-trained models generalize better than SFT-trained models regardless of model size. A 3B RL-trained model transfers better than a 14B SFT-trained model on the same math reasoning task.

## Architecture Overview

The approach uses diagnostic analysis to understand representation drift:

- **On-Policy RL Training**: Use GRPO, PPO, or DPO with KL regularization to optimize math reasoning while constraining representation drift
- **Representation Analysis**: Track how model hidden states evolve during training using PCA and dimensionality reduction
- **Token Distribution Analysis**: Monitor token probability distributions (entropy, rank changes) to detect catastrophic forgetting
- **Ablation Framework**: Isolate which RL components (on-policy sampling, credit assignment, negative gradients) drive generalization

## Implementation

**Step 1: Measure baseline transferability before training**

Establish baseline performance on both math and general tasks, computing a transferability index.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def evaluate_baseline_capabilities(model, tokenizer,
                                  math_eval_data, general_eval_data):
    """
    Evaluate the model on both math and general tasks before any training.
    Establish baseline for comparison.
    """
    math_scores = []
    general_scores = []

    # Evaluate on math
    for prompt, correct_answer in math_eval_data:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, max_length=256)
        output_text = tokenizer.decode(output_ids[0])

        is_correct = evaluate_mathematical_correctness(output_text, correct_answer)
        math_scores.append(float(is_correct))

    # Evaluate on general tasks
    for prompt, expected_answer in general_eval_data:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids, max_length=256)
        output_text = tokenizer.decode(output_ids[0])

        is_correct = evaluate_general_correctness(output_text, expected_answer)
        general_scores.append(float(is_correct))

    return {
        'math_baseline': np.mean(math_scores),
        'general_baseline': np.mean(general_scores)
    }

def compute_transferability_index(math_improvement, general_change):
    """
    Compute a metric that normalizes transfer across benchmarks.
    Positive = good transfer; negative = catastrophic forgetting.
    """
    # Normalize math improvement to 0-1 range (assume max possible is 50% improvement)
    math_gain = min(math_improvement / 0.5, 1.0)

    # Normalize general change to -1 to 0 (should be non-negative)
    general_delta = min(max(general_change, -1.0), 0.0)  # Capped at -1 for catastrophic failure

    # Transferability: combination of math gains and preserved general capabilities
    # RL should have positive general_delta, SFT often has negative
    transferability = math_gain + max(0, general_delta)

    return transferability
```

**Step 2: Track representation drift during training**

Use PCA and KL-divergence to monitor how model representations change.

```python
def track_representation_drift(model, reference_model, eval_prompts, layer_idx=-2):
    """
    Measure how much the model's internal representations have shifted
    compared to the reference (pre-training) model.
    Uses PCA dimensionality reduction to visualize drift.
    """
    current_activations = []
    reference_activations = []

    for prompt in eval_prompts[:100]:  # Use subset for efficiency
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Extract hidden states from current model
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            current_hidden = outputs.hidden_states[layer_idx]  # [batch, seq, hidden]
            current_activations.append(current_hidden.mean(dim=1).cpu().numpy())  # Average over sequence

        # Extract from reference model
        with torch.no_grad():
            ref_outputs = reference_model(input_ids, output_hidden_states=True)
            ref_hidden = ref_outputs.hidden_states[layer_idx]
            reference_activations.append(ref_hidden.mean(dim=1).cpu().numpy())

    # Stack activations
    current_activations = np.vstack(current_activations)  # [num_prompts, hidden_dim]
    reference_activations = np.vstack(reference_activations)

    # Compute PCA distance: how much do the principal components differ?
    from sklearn.decomposition import PCA

    pca_current = PCA(n_components=10)
    pca_reference = PCA(n_components=10)

    current_components = pca_current.fit_transform(current_activations)
    reference_components = pca_reference.fit_transform(reference_activations)

    # Procrustes distance: best alignment between two matrices
    drift_distance = procrustes_distance(current_components, reference_components)

    # Also compute KL divergence of token distributions
    # (does the model still generate similar token probabilities?)
    kl_divergences = []

    for prompt in eval_prompts[:50]:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            current_logits = model(input_ids).logits[:, -1, :]  # Last position
            current_probs = torch.softmax(current_logits, dim=-1)

            ref_logits = reference_model(input_ids).logits[:, -1, :]
            ref_probs = torch.softmax(ref_logits, dim=-1)

            # KL divergence between distributions
            kl = torch.sum(ref_probs * (torch.log(ref_probs) - torch.log(current_probs)))
            kl_divergences.append(kl.item())

    return {
        'representation_drift': drift_distance,
        'kl_divergence': np.mean(kl_divergences),
        'is_drifting': drift_distance > 0.3
    }
```

**Step 3: Train with RL using KL-regularized objectives**

Use GRPO with KL penalty to optimize math performance while preserving general capabilities.

```python
def train_math_with_rl_kl_regularization(model, tokenizer,
                                        math_train_data, general_train_data,
                                        math_reward_fn, learning_rate=1e-5,
                                        kl_weight=0.1, num_steps=1000):
    """
    Train on math using reinforcement learning with KL regularization.
    KL weight controls how much the model can deviate from the reference.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    reference_model = copy.deepcopy(model)  # Frozen reference for KL

    for step in range(num_steps):
        # Gather math training batch
        math_batch = random.sample(math_train_data, 16)
        general_batch = random.sample(general_train_data, 8)

        # Process math examples for RL
        math_losses = []

        for prompt, correct_answer in math_batch:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

            # Generate multiple samples
            samples = []
            for _ in range(4):
                output_ids = model.generate(
                    input_ids,
                    max_length=256,
                    temperature=1.0,  # Higher temperature for on-policy sampling
                    num_return_sequences=1
                )
                samples.append(output_ids[0])

            # Score samples with reward function
            rewards = [math_reward_fn(tokenizer.decode(s), correct_answer)
                      for s in samples]
            rewards = torch.tensor(rewards, device=device)

            # GRPO loss: optimize probability of high-reward samples
            # Log probability of each sample
            log_probs = []
            for sample in samples:
                log_prob = compute_log_probability(model, input_ids, sample)
                log_probs.append(log_prob)

            log_probs = torch.stack(log_probs)

            # Advantage: rewards relative to mean
            advantages = rewards - rewards.mean()
            grpo_loss = -(log_probs * advantages).mean()

            # KL penalty: divergence from reference model
            with torch.no_grad():
                ref_log_probs = [compute_log_probability(reference_model, input_ids, s)
                                for s in samples]
                ref_log_probs = torch.stack(ref_log_probs)

            kl_penalty = (log_probs - ref_log_probs).mean()
            total_loss = grpo_loss + kl_weight * kl_penalty

            math_losses.append(total_loss)

        # Add general task examples to constrain drift
        general_losses = []

        for prompt, expected in general_batch:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            labels = tokenizer.encode(expected, return_tensors='pt')[0].to(device)

            outputs = model(input_ids, labels=labels)
            general_losses.append(outputs.loss)

        # Combine losses: math RL + general SFT
        total_loss = sum(math_losses) / len(math_losses) + \
                    0.5 * sum(general_losses) / len(general_losses)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Math loss {sum(math_losses)/len(math_losses):.4f}, "
                  f"General loss {sum(general_losses)/len(general_losses):.4f}")

    return model

def compute_log_probability(model, input_ids, sample_ids, device='cuda'):
    """
    Compute the log probability of generating sample_ids given input_ids.
    """
    # Concatenate input and sample
    full_ids = torch.cat([input_ids, sample_ids.unsqueeze(0)], dim=1)

    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=False)
        logits = outputs.logits

    # Get log probabilities for the sample part
    sample_start = input_ids.shape[1]
    sample_logits = logits[:, sample_start-1:-1, :]  # Shift by 1 for next-token prediction

    log_probs = torch.nn.functional.log_softmax(sample_logits, dim=-1)

    # Extract log probs for the actual samples
    log_prob_sum = 0.0
    for i, token_id in enumerate(sample_ids):
        log_prob = log_probs[0, i, token_id]
        log_prob_sum += log_prob.item()

    return log_prob_sum / len(sample_ids)
```

**Step 4: Measure and report transferability results**

Compare RL-trained and SFT-trained models on transfer metrics.

```python
def compare_training_methods(model_sft, model_rl, math_eval_data,
                            general_eval_data):
    """
    Compare how well SFT-trained vs RL-trained models transfer.
    """
    results = {}

    # Evaluate both models
    for model_name, model in [('SFT', model_sft), ('RL', model_rl)]:
        math_scores = []
        general_scores = []

        for prompt, answer in math_eval_data:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            output_ids = model.generate(input_ids, max_length=256)
            output = tokenizer.decode(output_ids[0])
            score = float(evaluate_mathematical_correctness(output, answer))
            math_scores.append(score)

        for prompt, answer in general_eval_data:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            output_ids = model.generate(input_ids, max_length=256)
            output = tokenizer.decode(output_ids[0])
            score = float(evaluate_general_correctness(output, answer))
            general_scores.append(score)

        results[model_name] = {
            'math_accuracy': np.mean(math_scores),
            'general_accuracy': np.mean(general_scores),
            'transferability': compute_transferability_index(
                np.mean(math_scores) - baseline_math,
                np.mean(general_scores) - baseline_general
            )
        }

    return results
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| KL weight | 0.05-0.15 | Higher = preserve representations; lower = focus on task |
| Ratio of math to general data | 0.5-0.7 | 50-70% math examples, rest general |
| RL temperature | 0.8-1.2 | Controls diversity during sampling |
| Training duration | 2-5K steps | Diminishing returns after 3K |
| Reference model update | Never | Use frozen initial model for KL |

**When to use RL-based reasoning training:**
- You want to improve math reasoning without hurting general capabilities
- You have multiple task domains and need balanced performance
- Representation drift matters more than raw task performance
- You're willing to spend more compute on RL training

**When NOT to use RL-based training:**
- You only care about a single specialized task (use SFT)
- Compute budget is critical (RL is 30-50% more expensive than SFT)
- Your task requires very high specialization and general forgetting is acceptable
- You don't have multiple evaluation domains to assess transfer

**Common pitfalls:**
- **KL weight too high**: Model won't improve much. Start with 0.1 and decrease if needed.
- **Reference model drift**: Never update the reference model; it's the anchor for KL.
- **Insufficient general data**: If the model still forgets general tasks, increase general task ratio to 50%.
- **Reward signal too noisy**: If RL training is unstable, use a simpler reward function or increase batch size for more stable advantage estimation.

## Reference

Does Math Reasoning Improve General LLM Capabilities? Understanding Transferability of LLM Reasoning
https://arxiv.org/abs/2507.00432
