---
name: tower-plus-multilingual
title: "Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.17080"
keywords: [Multilingual, Machine Translation, Post-training Pipeline, Preference Optimization, Reinforcement Learning]
description: "Build multilingual LLMs that excel at machine translation while preserving general-purpose capabilities through a staged training pipeline combining continued pretraining, SFT, preference optimization, and RL with verifiable rewards."
---

# Tower+: Balanced Translation and General Capabilities in Multilingual Models

The core challenge in multilingual model development is that specialist translation models often sacrifice general-purpose reasoning and chat abilities, while general-purpose models underperform on translation tasks. This creates a false choice: either have a translation powerhouse that struggles with general tasks, or a generalist model that performs mediocrely at translation. Tower+ demonstrates this choice is unnecessary by systematically balancing both capabilities through an integrated four-stage post-training pipeline.

Current open-weight multilingual models typically excel in one domain at the expense of others. Tower+ achieves a Pareto frontier between specialization and generality by carefully composing multilingual signals during training while preserving chat abilities through reward-based filtering and RL optimization. The result is a model family (2B, 9B, 72B parameters) that matches frontier proprietary models in translation while exceeding open-weight competitors on general benchmarks.

## Core Concept

Tower+ uses a staged post-training approach that treats translation capability and general understanding as complementary goals rather than competing objectives. The key insight is that you can simultaneously improve specialized performance and maintain general capabilities by:

1. Loading the model with diverse multilingual signals during continued pretraining
2. Selecting training examples through reward-based filtering rather than naive mixing
3. Using preference optimization to balance multiple objectives
4. Applying reinforcement learning with specialized rewards that preserve instruction-following accuracy

This approach avoids the common pitfall where translation-focused training causes "catastrophic forgetting" of general capabilities—a problem documented in math reasoning transfer studies where SFT-based approaches often degrade on non-specialized tasks.

## Architecture Overview

The Tower+ post-training pipeline consists of four sequential stages:

- **Continued Pretraining (CPT)**: 66% monolingual data, 33% parallel translation pairs, plus 1% instruction-following data across 27 languages. This stage grounds the model in diverse linguistic patterns before specialization.
- **Supervised Fine-tuning (SFT)**: 1.3M curated samples where only 22% are translation tasks; the remainder are general instruction-following examples. Responses are selected from multiple teacher models using reward-based filtering to ensure high quality.
- **Preference Optimization**: Weighted Preference Optimization (WPO) combining SFT prompt data with UltraFeedback data, using trained reward models to guide preference selection.
- **Reinforcement Learning**: GRPO (Group Relative Policy Optimization) with dual verifiable rewards—one for translation quality and one for instruction-following precision—to maintain both capabilities.

## Implementation

The following steps outline how to implement Tower+-style training:

**Step 1: Prepare multilingual continued pretraining data**

This phase mixes different data types with careful proportions. You need to assemble monolingual text, parallel sentence pairs, and instruction data across your target languages.

```python
def prepare_cpt_data(languages=['en', 'es', 'zh']):
    """
    Prepare mixed-language data for continued pretraining.
    Monolingual data teaches linguistic patterns;
    parallel data enables cross-lingual transfer.
    """
    data = {
        'monolingual': load_monolingual_text(languages, proportion=0.66),
        'parallel': load_parallel_pairs(languages, proportion=0.33),
        'instruction': load_instruction_data(languages, proportion=0.01)
    }

    # Interleave data to create balanced batches
    combined = interleave_datasets(data, weights=[0.66, 0.33, 0.01])
    return combined
```

**Step 2: Create reward-based filtering for SFT data**

Rather than randomly mixing translation and general tasks, use reward models to select high-quality examples from multiple teacher models.

```python
def filter_sft_data_by_reward(candidates, reward_model, threshold=0.7):
    """
    Select SFT examples using a trained reward model.
    This ensures balanced task distribution while filtering low-quality responses
    from teacher models.
    """
    filtered = []
    task_counts = {'translation': 0, 'instruction': 0}

    for prompt, candidate_responses in candidates:
        # Score all candidate responses
        scores = reward_model.score(prompt, candidate_responses)
        best_idx = scores.argmax()
        best_response = candidate_responses[best_idx]

        # Accept only high-quality examples
        if scores[best_idx] > threshold:
            task_type = identify_task_type(prompt)
            if should_add(task_type, task_counts):
                filtered.append((prompt, best_response))
                task_counts[task_type] += 1

    return filtered
```

**Step 3: Apply Weighted Preference Optimization**

WPO combines multiple data sources (SFT, feedback data) and uses reward models to weight preferences, avoiding mode collapse where the model learns only one behavior.

```python
def weighted_preference_optimization(model, sft_data, feedback_data,
                                    reward_model, wpo_alpha=0.5):
    """
    Optimize preferences using weighted combination of SFT and feedback data.
    The reward model guides which preferences matter for each example.
    """
    losses = []

    for batch_idx, batch in enumerate(sft_data):
        prompt, response = batch

        # Compute reward for this response
        reward = reward_model.score(prompt, response)

        # Preference loss: model should prefer high-reward responses
        loss = -torch.log(torch.sigmoid(reward))
        losses.append(loss)

    # Mix in UltraFeedback preferences with weighting
    for batch in feedback_data:
        prompt, preferred, dispreferred = batch
        pref_reward = reward_model.score(prompt, preferred)
        dis_reward = reward_model.score(prompt, dispreferred)

        # Bradley-Terry preference loss, weighted by alpha
        pref_loss = -torch.log(torch.sigmoid(pref_reward - dis_reward))
        losses.append(wpo_alpha * pref_loss)

    return torch.stack(losses).mean()
```

**Step 4: Apply GRPO with dual verifiable rewards**

GRPO (Group Relative Policy Optimization) optimizes both translation and instruction-following using separate reward signals, with KL regularization to prevent diverging from the SFT checkpoint.

```python
def grpo_dual_reward_training(model, prompt_pool,
                             translation_reward_fn,
                             instruction_reward_fn,
                             kl_weight=0.01):
    """
    Train with two complementary reward signals:
    - Translation reward measures translation quality and accuracy
    - Instruction reward measures whether the model follows instructions correctly
    This prevents catastrophic forgetting by explicitly optimizing both objectives.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for batch in prompt_pool:
        # Generate rollouts
        responses = model.generate(batch['prompt'], num_samples=4)

        # Score with both reward functions
        trans_rewards = [translation_reward_fn(batch['source'], r)
                        for r in responses]
        instr_rewards = [instruction_reward_fn(batch['instruction'], r)
                        for r in responses]

        # Combine rewards: prioritize instruction-following during translation
        # This is the key to maintaining general capabilities
        combined_rewards = torch.tensor(trans_rewards) * 0.6 + \
                          torch.tensor(instr_rewards) * 0.4

        # GRPO: optimize relative rankings within the group
        baseline = combined_rewards.mean()
        advantages = combined_rewards - baseline

        # Compute policy gradient with KL regularization
        log_probs = model.get_log_probs(batch['prompt'], responses)
        policy_loss = -(log_probs * advantages).mean()

        # KL divergence from SFT checkpoint prevents divergence
        kl_div = model.kl_divergence(batch['prompt'], responses)
        total_loss = policy_loss + kl_weight * kl_div

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return total_loss.item()
```

## Practical Guidance

| Hyperparameter | Recommended Range | Notes |
|---|---|---|
| CPT duration | 10-20K steps | Typically 5-10% of original pretraining |
| SFT proportion | 22% translation, 78% general | Maintains 3:1 general-to-specialized ratio |
| WPO alpha | 0.3-0.7 | Higher alpha = more feedback data weight |
| GRPO translation weight | 0.5-0.7 | Higher = prioritize translation; lower = preserve general abilities |
| KL penalty | 0.01-0.05 | Prevents drift from SFT baseline |
| Sampling temperature | 0.8-1.2 | For preference optimization phase |

**When to use Tower+:**
- You need strong machine translation performance AND general-purpose capabilities
- Your target languages are under-resourced in open-weight models
- You have access to parallel data and instruction data for your languages
- You want to extend existing multilingual models rather than train from scratch

**When NOT to use Tower+:**
- Your primary goal is maximum translation performance regardless of general capabilities
- You have limited parallel sentence data
- You only need general-purpose performance (use standard instruction-tuning instead)
- Your timeline doesn't allow for four sequential post-training stages

**Common pitfalls:**
- **Forgetting general capabilities**: This happens when translation-focused SFT dominates. Mitigate by keeping translation at 22% or lower in SFT data and using explicit instruction rewards in GRPO.
- **Plateau in preference optimization**: If performance stalls, reduce WPO alpha or increase the number of preference pairs to provide stronger training signal.
- **KL divergence explosion**: If KL penalty is too high, the model can't move far enough from the SFT checkpoint. Start with 0.01 and increase gradually.
- **Reward model misalignment**: If your reward models don't correlate with actual quality, filtering becomes noise. Use multiple teacher models and cross-validate reward scoring.

## Reference

Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs
https://arxiv.org/abs/2506.17080
