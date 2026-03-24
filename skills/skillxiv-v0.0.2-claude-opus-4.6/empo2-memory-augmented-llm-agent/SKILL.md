---
name: empo2-memory-augmented-llm-agent
title: "EMPO2: Exploratory Memory-Augmented LLM Agent via Hybrid On/Off-Policy"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.23008"
keywords: [reinforcement learning, memory augmentation, exploration, LLM agents, meta-learning]
description: "Improve exploration in LLM-based agents through external memory-augmented RL with hybrid on/off-policy training. Agents generate exploration 'tips' (self-reflections) after trajectories, storing them in memory. During rollouts, policy samples between standard execution and memory-conditioned execution. Off-policy updates distill memory-guided behaviors into base policy via reward-guided knowledge distillation. Achieves 128.6% improvement on ScienceWorld and 11.3% on WebShop vs. GRPO."
---

# EMPO2: Structured Exploration via Memory and Hybrid Policy Updates

Large language model agents often struggle with exploration in complex environments: they become trapped in local optima, repeatedly executing unsuccessful strategies. Standard RL algorithms treat each trajectory independently, missing opportunities to learn from past mistakes and structured exploration patterns.

The challenge is encoding and reusing exploration insights. Agents need mechanisms to: (1) retrospectively analyze trajectories, (2) distill insights as reusable hints, (3) leverage these hints during future rollouts, and (4) eventually internalize their benefits without explicit memory access.

## Core Concept

EMPO2 combines parametric (policy parameters) and non-parametric (external memory) exploration mechanisms:

**Memory-Based Exploration**: After trajectories complete, agents self-reflect and generate "tips"—natural language guidance for avoiding mistakes and finding promising directions. Tips are stored in a memory buffer indexed by state/goal.

**Hybrid Policy Modes**: During rollouts, the policy samples between standard execution and memory-conditioned execution (using retrieved tips). This forces exploration of memory-guided paths.

**Hybrid Update Mechanism**:
- On-policy updates: Improve policy while conditioning on memory
- Off-policy updates: Distill memory-conditioned behaviors into base policy without memory, enabling inference-time execution without memory access

This hybrid approach both enables exploration (via memory) and distills benefits into the policy itself.

## Architecture Overview

- **Memory Buffer**: Stores (state, tip) pairs, indexed for fast retrieval
- **Self-Reflection Module**: After trajectory, generate summary and tips via LLM
- **Tip Retrieval**: Given current state, retrieve relevant tips from memory
- **Dual-Mode Policy**: Standard rollout (no memory) vs. memory-conditioned rollout (with retrieved tips)
- **On-Policy Optimizer**: Improve policy while using memory guidance
- **Off-Policy Distiller**: Reward-guided knowledge distillation from memory-conditioned to base policy
- **Memory Manager**: Update tips based on outcome; age out low-value tips

## Implementation

Implement self-reflection to generate exploration tips:

```python
def generate_exploration_tips(trajectory, reward, model):
    """
    After trajectory, generate tips for future exploration.
    Returns list of natural language hints.
    """
    prompt = f"""
Trajectory:
{format_trajectory(trajectory)}

Reward: {reward}

Generate 2-3 tips for improving future trajectories on similar problems.
Focus on what went wrong and what should be tried next.

Tips:
"""

    tips_text = model.generate(prompt, temperature=0.3, max_tokens=150)
    tips = parse_tips(tips_text)  # Return as list of strings

    return tips

def add_tips_to_memory(memory_buffer, state, tips, trajectory_reward):
    """
    Add tips to memory buffer, indexed by state.
    """
    state_key = hash_state(state)

    if state_key not in memory_buffer:
        memory_buffer[state_key] = []

    for tip in tips:
        memory_buffer[state_key].append({
            'tip': tip,
            'reward': trajectory_reward,
            'access_count': 0,
            'success_count': 0
        })

def retrieve_tips(memory_buffer, current_state, max_tips=3):
    """
    Retrieve relevant tips from memory for current state.
    """
    state_key = hash_state(current_state)

    if state_key not in memory_buffer:
        return []

    # Sort by success rate (success_count / (access_count + 1))
    tips_list = memory_buffer[state_key]
    tips_list.sort(
        key=lambda x: x['success_count'] / max(1, x['access_count']),
        reverse=True
    )

    return [tip['tip'] for tip in tips_list[:max_tips]]
```

Implement dual-mode policy with memory conditioning:

```python
def execute_rollout_dual_mode(
    policy, initial_state, goal, memory_buffer, use_memory_ratio=0.5
):
    """
    Execute rollout with dual modes: standard and memory-conditioned.
    use_memory_ratio: probability of using memory-conditioned mode
    """
    state = initial_state
    trajectory = []
    rewards = []
    modes = []  # Track which mode was used at each step

    for step in range(100):  # Max steps
        # Sample mode
        use_memory = np.random.uniform() < use_memory_ratio

        if use_memory:
            # Memory-conditioned execution
            tips = retrieve_tips(memory_buffer, state)

            prompt = f"""
State: {state}
Goal: {goal}
Helpful tips: {tips}

Next action:
"""
            action = policy.generate(prompt, temperature=0.7, max_tokens=50)
            modes.append('memory')
        else:
            # Standard execution
            prompt = f"""
State: {state}
Goal: {goal}

Next action:
"""
            action = policy.generate(prompt, temperature=0.7, max_tokens=50)
            modes.append('standard')

        # Execute action
        next_state, reward = execute_action(action, state, goal)

        trajectory.append((state, action, next_state))
        rewards.append(reward)

        if reward > 0.9:  # Goal reached
            break

        state = next_state

    return trajectory, rewards, modes
```

Implement hybrid optimization (on-policy + off-policy):

```python
def empo2_update(
    trajectories, memory_buffer, policy, alpha_on=0.5, alpha_off=0.5
):
    """
    Hybrid update: on-policy (with memory) + off-policy (distillation).
    """
    # Separate trajectories by mode
    memory_trajectories = [
        (traj, rew, modes) for traj, rew, modes in trajectories
        if any(m == 'memory' for m in modes)
    ]

    standard_trajectories = [
        (traj, rew, modes) for traj, rew, modes in trajectories
        if all(m == 'standard' for m in modes)
    ]

    # Phase 1: On-policy update (improve policy using memory)
    if memory_trajectories:
        on_policy_loss = 0

        for trajectory, rewards, modes in memory_trajectories:
            for step_idx, (state, action, next_state) in enumerate(trajectory):
                # Retrieve tips that were used
                tips = retrieve_tips(memory_buffer, state)

                # Predict action with memory
                prompt_with_memory = f"""
State: {state}
Tips: {tips}

Action:
"""
                predicted_action = policy.generate(
                    prompt_with_memory, temperature=0.3
                )

                # Loss: likelihood of taken action
                log_prob = policy.log_prob(action, prompt_with_memory)
                advantage = compute_advantage(rewards[step_idx:])

                on_policy_loss += -log_prob * advantage

        on_policy_loss = on_policy_loss / max(1, len(memory_trajectories))
        policy.optimize(on_policy_loss)

    # Phase 2: Off-policy distillation (distill memory-guided behavior into base policy)
    if memory_trajectories:
        off_policy_loss = 0

        for trajectory, rewards, modes in memory_trajectories:
            for step_idx, (state, action, next_state) in enumerate(trajectory):
                # Predict action WITHOUT memory (base policy)
                prompt_no_memory = f"""
State: {state}

Action:
"""
                predicted_action_no_memory = policy.generate(
                    prompt_no_memory, temperature=0.3
                )

                # KL divergence: encourage base policy to match memory-guided action
                advantage = compute_advantage(rewards[step_idx:])

                if advantage > 0:  # Only distill high-reward trajectories
                    log_prob_no_memory = policy.log_prob(action, prompt_no_memory)
                    off_policy_loss += -log_prob_no_memory * advantage

        off_policy_loss = off_policy_loss / max(1, len(memory_trajectories))
        policy.optimize(off_policy_loss)

    # Update memory tips with success information
    for trajectory, rewards, modes in trajectories:
        final_reward = sum(rewards)
        state = trajectory[0][0]

        tips = retrieve_tips(memory_buffer, state)
        if tips:
            state_key = hash_state(state)
            for tip_entry in memory_buffer[state_key]:
                if tip_entry['tip'] in tips:
                    tip_entry['access_count'] += 1
                    if final_reward > 0.5:
                        tip_entry['success_count'] += 1
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Memory ratio (use memory) | 0.5 | Higher (0.7) for more memory-guided exploration; lower (0.3) for more standard |
| Alpha on-policy | 0.5 | Weight for on-policy update; higher emphasizes memory guidance |
| Alpha off-policy | 0.5 | Weight for distillation; higher emphasizes internalization |
| Max tips per state | 3 | Balance relevance with memory lookup cost |
| Memory size | 10,000 entries | Increase for longer episodes; implement LRU eviction |

**When to use**: For complex RL environments (web navigation, code generation, scientific simulation) where exploration patterns are reusable across similar states.

**When not to use**: For simple tasks with dense rewards or unique solutions; memory overhead unjustified.

**Common pitfalls**:
- Tips too generic ("try harder"); enforce specific, actionable tips during generation
- Off-policy distillation forgetting that memory-conditioned mode was used; only distill high-reward trajectories
- Memory bloat from poor retrieval; implement periodic memory pruning (remove low-success tips)

## Reference

EMPO2 achieves 128.6% improvement on ScienceWorld (exploration-heavy benchmark) and 11.3% on WebShop over GRPO baseline. The hybrid approach enables strong in-distribution performance while maintaining generalization to out-of-distribution queries without explicit memory access.
