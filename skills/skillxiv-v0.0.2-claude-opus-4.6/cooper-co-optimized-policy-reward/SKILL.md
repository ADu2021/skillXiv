---
name: cooper-co-optimized-policy-reward
title: "Cooper: Co-Optimizing Policy and Reward Models in RL for LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.05613
keywords: [reinforcement-learning, reward-modeling, policy-optimization, llm-training, reward-hacking]
description: "Joint optimization of policy and reward models in LLM reinforcement learning by leveraging rule-based reward precision and dynamically constructing training pairs to prevent reward hacking and improve performance."
---

# Cooper: Co-Optimizing Policy and Reward Models

## Core Concept

Cooper addresses a fundamental challenge in RLHF (Reinforcement Learning from Human Feedback) for large language models: the tension between rule-based and model-based reward systems. Rule-based rewards lack robustness to distribution shift, while learned reward models are vulnerable to reward hacking—where the policy learns to exploit the reward model rather than optimize genuine task performance.

The core insight is that jointly optimizing both the policy and reward model, while dynamically updating the reward model during training, can mitigate reward hacking and improve end-to-end performance.

## Architecture Overview

- **Hybrid Reward System**: Combines high-precision rule-based rewards (for detecting correct responses) with learned model-based rewards (for generalization)
- **Reference-Based Reward Modeling**: Introduces a VerifyRM model that takes both the response and a reference correct answer as input, improving reward signal quality
- **Dynamic Reward Model Updates**: Continuously constructs and selects positive-negative sample pairs to retrain the reward model rather than freezing it after initial training
- **Joint Optimization Loop**: Policy and reward model are optimized together, where improved reward modeling guides better policy learning and vice versa

## Implementation Steps

### 1. Initialize Base Models

Set up your base language model (policy) and initialize a reward model architecture. The reward model should accept concatenated inputs: [response, reference_answer, prompt].

```python
# PyTorch/Hugging Face style pseudocode
policy_model = AutoModelForCausalLM.from_pretrained("llm-base")
reward_model = RewardModel(hidden_size=768, num_labels=1)
optimizer_policy = AdamW(policy_model.parameters(), lr=5e-6)
optimizer_reward = AdamW(reward_model.parameters(), lr=1e-5)
```

### 2. Collect Initial Training Data

Generate policy rollouts and obtain gold-standard reference answers. Annotate with rule-based rewards where available (exact match, constraint satisfaction, etc.).

```python
# Generate rollouts from current policy
rollouts = []
for prompt in prompt_batch:
    response = policy_model.generate(prompt, max_length=256)
    reference = gold_standard_answers[prompt_id]
    rule_reward = compute_rule_based_reward(response, reference)
    rollouts.append({
        'prompt': prompt,
        'response': response,
        'reference': reference,
        'rule_reward': rule_reward
    })
```

### 3. Construct Positive-Negative Pairs for Reward Model

Dynamically select pairs where rule-based rewards provide high-confidence signals. Create pairs by matching high-reward responses (positive) with low-reward responses (negative) from the same or similar prompts.

```python
# Construct training pairs using rule-based signal
pairs = []
for prompt_group in group_by_prompt(rollouts):
    high_reward_samples = [r for r in prompt_group if r['rule_reward'] > threshold]
    low_reward_samples = [r for r in prompt_group if r['rule_reward'] < threshold]

    for pos in high_reward_samples:
        for neg in low_reward_samples[:2]:  # Limit negatives per positive
            pairs.append((pos, neg))
```

### 4. Train Reward Model with Reference Inputs

Train the VerifyRM reward model using the constructed pairs. The model should learn to prefer responses that match the reference answer and follow task constraints.

```python
# Train reward model with reference answers as additional context
for pos_sample, neg_sample in pairs:
    pos_input = tokenize([pos_sample['prompt'], pos_sample['response'],
                          pos_sample['reference']])
    neg_input = tokenize([neg_sample['prompt'], neg_sample['response'],
                          neg_sample['reference']])

    pos_score = reward_model(pos_input)
    neg_score = reward_model(neg_input)

    # Preference loss (margin-based or ranking)
    loss = max(0, margin - (pos_score - neg_score))
    optimizer_reward.zero_grad()
    loss.backward()
    optimizer_reward.step()
```

### 5. Use Updated Reward Model for Policy Optimization

Run PPO or similar RL algorithm using the newly trained reward model to score policy rollouts. The key difference from standard RLHF is that the reward model itself improves over time.

```python
# Policy optimization with updated reward model
for epoch in range(num_ppo_epochs):
    policy_rollouts = policy_model.generate(prompts, max_length=256)

    # Score with updated reward model
    with torch.no_grad():
        policy_rollout_inputs = tokenize(prompts, policy_rollouts, references)
        rewards = reward_model(policy_rollout_inputs)

    # PPO update
    advantages = compute_advantages(rewards, value_baseline)
    policy_loss = -torch.mean(log_probs * advantages)

    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()
```

### 6. Iterate: Update Both Models

Repeat steps 2-5. The system converges as the reward model becomes more accurate and the policy learns to optimize genuine task performance rather than exploiting reward signal artifacts.

```python
# Main training loop
for iteration in range(num_iterations):
    # Collect new rollouts with current policy
    rollouts = generate_rollouts(policy_model, prompts, references)

    # Construct pairs using rule-based signals
    pairs = construct_pairs(rollouts, rule_reward_fn)

    # Update reward model
    train_reward_model(reward_model, pairs, num_epochs=3)

    # Update policy with new reward model
    train_policy(policy_model, reward_model, rollouts, num_ppo_steps=100)

    # Evaluate on held-out test set
    eval_score = evaluate(policy_model, test_prompts, gold_answers)
    print(f"Iteration {iteration}: Eval score = {eval_score}")
```

## Practical Guidance

### Hyperparameters & Configuration

- **Reward Model Learning Rate**: 1e-5 to 5e-5 (typically higher than policy LR due to smaller model)
- **Policy Learning Rate**: 5e-6 to 1e-5 (conservative to avoid divergence)
- **Rule-Reward Threshold**: Set to 0.7-0.9 to identify high-confidence positive samples
- **Update Frequency**: Retrain reward model every 10-50 policy update steps
- **Margin for Preference Loss**: 0.1-0.5 depending on reward scale

### When to Use Cooper

- Your task has access to clear rule-based signals (exact match, format validation, constraint satisfaction)
- You observe reward hacking behavior in standard RLHF
- You have computational budget for joint training of two models
- Task correctness can be partially verified with reference answers
- You want to improve both safety and performance of RL-trained policies

### When NOT to Use Cooper

- Task has no clear rule-based reward signal (pure open-ended generation)
- Computational constraints require single-model training
- Reference answers are unavailable or too expensive to obtain
- Standard RLHF already achieves target performance without hacking
- Task requires black-box reward models without access to training loop

### Common Pitfalls

1. **Over-Weighting Rule-Based Rewards**: If rule-based signals are too brittle, they'll constrain learning. Use them only for pair construction, not direct policy training.
2. **Stale Reward Models**: Updating too infrequently causes policy-reward divergence. Update at least every 20 policy steps.
3. **Insufficient Negative Samples**: Reward models need diverse negative examples to avoid degenerate solutions. Collect from off-policy rollouts.
4. **Reference Answer Bias**: If references are suboptimal, the system will converge to suboptimal solutions. Ensure reference quality.

## Reference

Cooper (2508.05613): https://arxiv.org/abs/2508.05613

Joint optimization of policy and reward models prevents reward hacking and improves end-to-end RL performance for LLM alignment, achieving 0.54% accuracy gains on instruction-following benchmarks.
