---
name: prorl-reasoning-expansion
title: "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24864"
keywords: [Reinforcement Learning, Reasoning, Language Models, GRPO, Test-Time Computation]
description: "Train LLMs to discover novel reasoning strategies beyond base model capabilities using prolonged RL with KL control, reference policy resets, and diverse task suites."
---

# Unlock Novel Reasoning Strategies Through Prolonged RL Training

Recent advances have shown that reinforcement learning can align language models with verifiable rewards, but the fundamental question remains: does RL truly expand reasoning capabilities, or merely amplify outputs already latent in the base model? ProRL demonstrates that extended RL training with proper regularization can uncover entirely new reasoning strategies inaccessible through sampling alone, fundamentally expanding the model's solution space.

The key insight is that reasoning boundaries grow when three conditions align: sufficient base model competence on the task, extended training duration, and controlled KL divergence to prevent distribution collapse. Unlike training approaches that plateau quickly, prolonged RL explores new regions of the solution space over time, enabling models to solve problems they previously could not.

## Core Concept

ProRL treats extended RL training as an exploration process that uncovers new reasoning pathways. Instead of assuming all possible solutions exist in the base model's distribution (the lottery ticket hypothesis), ProRL shows that RL training genuinely discovers novel strategies through:

- **Extended training horizons** that allow gradual discovery of new patterns
- **Reference policy resets** that prevent mode collapse into a single high-reward trajectory
- **KL divergence control** that maintains exploration while preventing catastrophic forgetting
- **Diverse task distributions** that exercise different reasoning capabilities

The mechanism is distinct from mere sampling: even with unlimited attempts, the base model cannot find solutions that RL discovers, indicating genuine capability expansion rather than amplification.

## Architecture Overview

- **Reward signal**: Use verifiable, binary or scalar rewards that align with task correctness
- **KL coefficient scheduling**: Gradually adjust KL penalty (typically 0.1-0.5) to balance reward maximization and distribution stability
- **Reference policy updates**: Reset reference policy periodically (every 500-1000 training steps) to prevent optimal policy drift
- **Task diversity**: Implement a multi-task curriculum covering different reasoning domains
- **Training duration**: Run RL training for extended periods (10k-100k+ steps per task domain)
- **Checkpoint selection**: Track pass@k improvements across different sampling counts to validate genuine capability growth

## Implementation

The following pseudocode outlines the ProRL training loop with KL control and reference policy resets:

```python
# ProRL training loop with capability expansion monitoring
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("base-model")
tokenizer = AutoTokenizer.from_pretrained("base-model")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
kl_coeff = 0.1
reference_policy = None
best_pass_at_k = {}
training_step = 0
reset_interval = 500

for epoch in range(num_epochs):
    for batch in data_loader:
        # Generate completions using current policy
        prompts = batch['prompts']
        rewards = batch['rewards']  # Binary or scalar reward signals

        # Compute log probabilities with current and reference policy
        with torch.no_grad():
            # Use reference policy (initially same as model)
            if reference_policy is None:
                reference_policy = model
            ref_logits = reference_policy(prompts)
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

        logits = model(prompts)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Calculate policy gradient with KL regularization
        policy_loss = -(rewards * log_probs).mean()
        kl_div = torch.nn.functional.kl_div(log_probs, ref_log_probs, reduction='batchmean')
        total_loss = policy_loss + kl_coeff * kl_div

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Periodically reset reference policy to enable exploration
        training_step += 1
        if training_step % reset_interval == 0:
            reference_policy = model  # Reset reference to current policy

        # Track pass@k to detect capability expansion
        if training_step % 100 == 0:
            test_results = evaluate_pass_at_k(model, test_set, k_values=[1, 5, 10])
            for k, pass_k in test_results.items():
                if k not in best_pass_at_k or pass_k > best_pass_at_k[k]:
                    best_pass_at_k[k] = pass_k
                    print(f"New best pass@{k}: {pass_k:.3f}")
```

Monitor training progress by evaluating pass@k across different values to confirm genuine capability growth:

```python
def evaluate_pass_at_k(model, test_set, k_values=[1, 5, 10, 100]):
    """Evaluate whether model can solve problems with k attempts"""
    results = {}
    for k in k_values:
        solved = 0
        for prompt, correct_output in test_set:
            # Generate k different solutions
            solutions = []
            for _ in range(k):
                output = model.generate(prompt, max_length=512, temperature=0.8)
                solutions.append(output)
            # Check if any solution is correct
            if any(is_correct(sol, correct_output) for sol in solutions):
                solved += 1
        results[k] = solved / len(test_set)
    return results
```

For task diversity, implement a curriculum that exposes the model to different reasoning domains:

```python
# Multi-domain task curriculum for ProRL
task_domains = {
    'math': load_math_reasoning_tasks(),
    'logic': load_logic_reasoning_tasks(),
    'coding': load_code_generation_tasks(),
    'search': load_search_planning_tasks(),
}

for domain, tasks in task_domains.items():
    print(f"Training on {domain} domain...")
    domain_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    # Domain-specific RL training
    for epoch in range(domain_epochs):
        for task in tasks:
            prompt = task['prompt']
            # Generate trajectories
            completions = model.generate(prompt, num_return_sequences=8, temperature=0.9)
            # Get rewards (must be verifiable)
            task_rewards = [verify_solution(comp, task['ground_truth']) for comp in completions]
            # RL update as above...
```

## Practical Guidance

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| KL coefficient | 0.05 - 0.5 | Higher values prevent distribution shift; too high limits exploration |
| Reference reset interval | 500 - 2000 steps | Frequent resets enable broader exploration; affects convergence speed |
| Training steps per domain | 5000 - 50000 | Longer training discovers more novel strategies |
| Temperature for generation | 0.7 - 1.0 | Higher temperature encourages exploration |
| Batch size | 8 - 64 | Balance compute efficiency with gradient stability |

**When to use ProRL:**
- Your base model solves <50% of target tasks but has some foundational capability
- Rewards are binary or easily verifiable (correctness, test passing)
- You can afford extended training runs (days to weeks)
- Novel reasoning strategies are more valuable than marginal performance gains
- You have diverse task distributions to enable generalization

**When NOT to use ProRL:**
- Base model accuracy is already >80% (diminishing returns)
- Reward signal is sparse, noisy, or difficult to verify automatically
- Training compute is severely limited (<1000 steps per task)
- You only have a single, narrow task domain
- You need interpretability (RL-discovered strategies may be opaque)
- Alignment concerns dominate performance (KL control needed for safety)

**Common pitfalls:**
- Starting with too-high KL coefficient, preventing exploration
- Resetting reference policy too frequently, causing oscillation
- Using weak reward signals that don't correlate with actual capability
- Training on too-narrow task distributions, limiting generalization
- Stopping training too early before novel strategies emerge
- Not tracking pass@k across multiple k values to confirm expansion

## Reference

**ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models**
https://arxiv.org/abs/2505.24864
