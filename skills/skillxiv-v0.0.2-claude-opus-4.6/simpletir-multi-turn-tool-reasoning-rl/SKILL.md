---
name: simpletir-multi-turn-tool-reasoning-rl
title: "SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.02479"
keywords: [reinforcement learning, tool integration, multi-turn reasoning, GRPO, trajectory filtering, distributional drift, gradient stability, mathematical reasoning]
description: "Train LLMs for multi-turn tool-integrated reasoning end-to-end using RL without supervised pretraining. SimpleTIR stabilizes training by filtering void turns (responses lacking code blocks or final answers) to prevent gradient explosion from distributional drift, enabling discovery of emergent reasoning patterns like self-correction and cross-validation on mathematical benchmarks."
---

# SimpleTIR: Stable End-to-End RL for Multi-Turn Tool-Integrated Reasoning

## Outcome

Enable base language models to learn multi-turn tool-integrated reasoning (code generation, execution, reflection, correction) end-to-end through reinforcement learning without requiring supervised fine-tuning or domain-specific pretraining, achieving state-of-the-art performance on mathematical reasoning benchmarks while maintaining stable training dynamics.

## Problem Context

When training language models for multi-turn tool-integrated reasoning—where models generate code, execute it, observe results, and iterate—the training process becomes fundamentally unstable. The root cause lies in distributional drift: external tool outputs (execution results, error messages) exist far outside the model's pretraining distribution. When the model generates a response that leads to unexpected tool feedback, subsequent token generations fall into extremely low-probability regions of the learned distribution.

These low-probability tokens compound across turns. After three or four incorrect generations followed by tool feedback, the model assigns vanishingly small probabilities to valid continuations. During backpropagation, this produces catastrophic gradient norm explosions. Importance ratio terms explode, gradient magnitudes spike orders of magnitude above normal RL training, and the policy diverges into nonsensical behavior. The naive approach crashes entirely—gradient explosion destroys the learned policy.

Existing solutions impose external structure: supervised fine-tuning on human examples, careful curriculum learning, or complex reward shaping. These sacrifice the core promise of end-to-end learning: discovering reasoning strategies naturally from task feedback alone.

## Core Concept

SimpleTIR introduces a minimal but surgically effective filtering mechanism targeting the symptom of distributional drift: "void turns"—individual model responses that contain neither a complete code block nor a final answer. These represent generation failures caused by the model assigning high probability to out-of-distribution token sequences.

By filtering entire trajectories containing void turns before computing policy losses, SimpleTIR prevents the high-magnitude gradients associated with problematic sequences while preserving credit assignment for successful reasoning chains. The mechanism is algorithm-agnostic, works as a plug-and-play wrapper around existing policy optimization methods, and requires no threshold tuning.

The theoretical grounding reveals that gradient norm depends inversely on token probabilities and is exacerbated by two compounding factors: unbounded importance ratios for low-probability tokens and sustained high gradient magnitudes when the policy assigns low probability to sampled sequences. Filtering void turns blocks the worst offenders without requiring complex annealing or uncertainty estimation.

## Architecture Overview

SimpleTIR operates within a hierarchical MDP framework:

- **Prompt Level**: Input text describing the task (e.g., "Solve this math problem")
- **Turn Level**: Each turn represents one round of agent action + environment feedback (model generates response, tool executes code, returns result)
- **Token Level**: Within each turn, the model samples individual tokens from its policy distribution

The training loop uses Group Relative Policy Optimization (GRPO), a recent variance-reduction technique that normalizes rewards within groups of trajectories sampled from the same prompt:

- Sample batch of prompts
- For each prompt, collect G trajectories (different reasoning paths)
- Compute rewards for each trajectory (final answer correctness determines signal)
- Normalize advantage estimates within each group using mean and standard deviation
- Compute policy loss using clipped surrogate objective (PPO-style)
- Filter trajectories containing void turns before backpropagation
- Update model parameters

The key architectural distinction: void turn detection and filtering happens at the trajectory level, not token level. This preserves full credit assignment for trajectories that successfully navigate distributional drift, only excluding those that fail catastrophically.

## Implementation

### Step 1: Void Turn Detection

Detect void turns by parsing model output. A void turn contains neither a complete code block nor a final answer.

```python
# Void turn detection logic
def is_void_turn(response_text: str) -> bool:
    """Check if model response is a void turn."""
    # Check for complete code blocks (backtick-delimited)
    has_complete_code = (
        response_text.count("```python") > 0 and
        response_text.count("```") >= 2  # Opening and closing
    )

    # Check for final answer marker
    has_final_answer = "final_answer(" in response_text.lower()

    # Void if neither code block nor final answer
    return not (has_complete_code or has_final_answer)

def filter_void_trajectories(trajectories: list[dict]) -> list[dict]:
    """Remove trajectories containing any void turn."""
    filtered = []
    for trajectory in trajectories:
        contains_void = any(
            is_void_turn(turn['response'])
            for turn in trajectory['turns']
        )
        if not contains_void:
            filtered.append(trajectory)
    return filtered
```

### Step 2: Reward Computation

Evaluate trajectory correctness. For mathematical reasoning, reward is binary (correct final answer = 1, incorrect = 0) or scaled by partial credit.

```python
def compute_trajectory_reward(trajectory: dict, expected_answer: str) -> float:
    """Compute reward for trajectory based on final answer correctness."""
    turns = trajectory['turns']
    if not turns:
        return 0.0

    last_response = turns[-1]['response']

    # Extract final answer from final_answer() call
    answer_match = None
    if 'final_answer(' in last_response:
        start = last_response.find('final_answer(')
        if start != -1:
            # Simple extraction; in practice use proper parsing
            content = last_response[start + len('final_answer('):]
            end = content.find(')')
            if end != -1:
                answer_match = content[:end].strip().strip('"').strip("'")

    # Normalize both answers (remove whitespace, lowercase for strings)
    if answer_match and expected_answer:
        answer_match = str(answer_match).strip().lower()
        expected = str(expected_answer).strip().lower()
        if answer_match == expected:
            return 1.0

    return 0.0

def batch_compute_rewards(trajectories: list[dict],
                          expected_answers: list[str]) -> list[float]:
    """Compute rewards for a batch of trajectories."""
    return [
        compute_trajectory_reward(traj, expected)
        for traj, expected in zip(trajectories, expected_answers)
    ]
```

### Step 3: GRPO Advantage Computation with Feedback Masking

Normalize advantages within groups and apply masking to exclude tool feedback tokens from loss.

```python
import numpy as np

def compute_group_advantages(rewards: np.ndarray,
                            gamma: float = 1.0) -> np.ndarray:
    """
    Compute advantages normalized within trajectory group (GRPO).

    Args:
        rewards: shape (G,) where G is number of trajectories per group
        gamma: discount factor (typically 1.0 for episodic tasks)

    Returns:
        advantages: shape (G,) normalized within group
    """
    # Normalize advantages using group statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Avoid division by zero
    if std_reward < 1e-8:
        std_reward = 1.0

    advantages = (rewards - mean_reward) / std_reward
    return advantages

def create_feedback_mask(turn: dict, max_length: int) -> np.ndarray:
    """
    Create binary mask for tokens. Mask=1 for agent response, 0 for tool feedback.

    Args:
        turn: dict with keys 'response' (agent text) and 'feedback' (tool output)
        max_length: maximum sequence length for padding

    Returns:
        mask: shape (max_length,) where 1=count, 0=ignore in loss
    """
    mask = np.zeros(max_length, dtype=np.float32)

    # Agent response tokens are counted
    response_length = len(turn['response'].split())  # Approximate tokenization
    mask[:min(response_length, max_length)] = 1.0

    # Tool feedback tokens are masked out (remain 0)
    # This ensures credit assignment targets policy actions only

    return mask

def apply_masking_to_loss(token_losses: np.ndarray,
                         masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply token masks to loss and return masked loss + normalization factors.

    Args:
        token_losses: shape (batch_size, max_seq_len)
        masks: list of masks, each shape (max_seq_len,)

    Returns:
        masked_losses: (batch_size, max_seq_len) with feedback tokens zeroed
        normalizers: (batch_size,) denominator for averaging
    """
    batch_size = token_losses.shape[0]
    masked_losses = token_losses.copy()
    normalizers = np.zeros(batch_size)

    for i, mask in enumerate(masks):
        masked_losses[i] *= mask
        normalizers[i] = np.sum(mask) + 1e-8  # Avoid division by zero

    return masked_losses, normalizers
```

### Step 4: PPO Loss Computation with Clipping

Standard PPO clipped surrogate loss applied only to non-masked tokens.

```python
def compute_ppo_loss(log_probs_new: np.ndarray,
                     log_probs_old: np.ndarray,
                     advantages: np.ndarray,
                     epsilon: float = 0.2,
                     masks: list[np.ndarray] = None) -> float:
    """
    Compute PPO clipped surrogate loss.

    Args:
        log_probs_new: log probabilities under current policy, shape (batch,)
        log_probs_old: log probabilities under old policy, shape (batch,)
        advantages: normalized advantage estimates, shape (batch,)
        epsilon: PPO clip threshold (0.2 or 0.28 common values)
        masks: optional list of token masks for masking feedback

    Returns:
        loss: scalar loss value
    """
    # Compute importance ratios
    ratio = np.exp(log_probs_new - log_probs_old)

    # Clipped surrogate objective
    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
    loss = -np.minimum(unclipped, clipped)

    # Apply masking if provided
    if masks is not None:
        normalizers = np.array([np.sum(m) + 1e-8 for m in masks])
        loss_per_sample = loss / normalizers
    else:
        loss_per_sample = loss

    return np.mean(loss_per_sample)

def compute_tir_training_objective(trajectories: list[dict],
                                   log_probs_new: list[np.ndarray],
                                   log_probs_old: list[np.ndarray],
                                   rewards: np.ndarray,
                                   epsilon: float = 0.2,
                                   masks: list[list[np.ndarray]] = None) -> float:
    """
    Full SimpleTIR training objective: GRPO + feedback masking + trajectory filtering.

    Args:
        trajectories: list of trajectories (already filtered for void turns)
        log_probs_new: list of log-probability arrays per trajectory
        log_probs_old: list of old log-probability arrays
        rewards: array of rewards for trajectories
        epsilon: PPO clip parameter
        masks: nested list of masks (per-trajectory, per-turn)

    Returns:
        loss: scalar training loss
    """
    # Compute group advantages using GRPO normalization
    advantages = compute_group_advantages(rewards)

    # Aggregate loss across all tokens in all trajectories
    total_loss = 0.0
    total_count = 0

    for i, trajectory in enumerate(trajectories):
        adv = advantages[i]
        log_new = log_probs_new[i]  # shape (num_tokens,)
        log_old = log_probs_old[i]  # shape (num_tokens,)

        # Per-token PPO loss
        ratio = np.exp(log_new - log_old)
        unclipped = ratio * adv
        clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon) * adv
        token_loss = -np.minimum(unclipped, clipped)

        # Apply feedback masking
        if masks is not None:
            traj_masks = masks[i]  # list of masks per turn
            masked_loss = token_loss * np.concatenate(traj_masks)
            normalizer = np.sum(np.concatenate(traj_masks)) + 1e-8
        else:
            masked_loss = token_loss
            normalizer = len(token_loss)

        total_loss += np.sum(masked_loss) / normalizer
        total_count += 1

    return total_loss / total_count
```

### Step 5: Training Loop with Void Turn Filtering

Main training loop integrating all components.

```python
import torch
from torch.optim import Adam

class SimpleTIRTrainer:
    """End-to-end RL trainer for multi-turn tool-integrated reasoning."""

    def __init__(self, model, learning_rate: float = 1e-6,
                 ppo_epsilon: float = 0.2, max_turns: int = 5):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.ppo_epsilon = ppo_epsilon
        self.max_turns = max_turns

    def rollout(self, prompts: list[str], env) -> list[dict]:
        """
        Generate trajectories by rolling out policy.

        Args:
            prompts: list of problem descriptions
            env: environment with code execution capability

        Returns:
            trajectories: list of {turns, reward, void_flagged}
        """
        trajectories = []

        for prompt in prompts:
            turns = []
            context = prompt
            trajectory_done = False
            turn_count = 0

            while not trajectory_done and turn_count < self.max_turns:
                # Generate response from policy
                with torch.no_grad():
                    response = self.model.generate(
                        context,
                        max_length=16384,
                        temperature=1.0,
                        top_p=0.95
                    )

                turns.append({
                    'response': response,
                    'context': context
                })

                # Check for final answer (trajectory termination)
                if 'final_answer(' in response:
                    trajectory_done = True
                else:
                    # Extract and execute code
                    code = self._extract_code_block(response)
                    if code:
                        feedback = env.execute_code(code)
                        context += f"\n\nCode Execution Result:\n{feedback}"
                        turns[-1]['feedback'] = feedback
                    else:
                        # Void turn: no code block
                        trajectory_done = True

                turn_count += 1

            trajectories.append({
                'turns': turns,
                'prompt': prompt,
                'void_flagged': any(
                    is_void_turn(turn['response']) for turn in turns
                )
            })

        return trajectories

    def train_step(self, trajectories: list[dict],
                   expected_answers: list[str]) -> float:
        """
        Single training step: filter voids, compute loss, update parameters.
        """
        # Filter trajectories containing void turns
        valid_trajectories = filter_void_trajectories(trajectories)

        if len(valid_trajectories) == 0:
            print("Warning: all trajectories filtered as void")
            return 0.0

        # Compute rewards only for valid trajectories
        expected_valid = [
            expected_answers[i] for i, traj in enumerate(trajectories)
            if not traj['void_flagged']
        ]
        rewards = batch_compute_rewards(valid_trajectories, expected_valid)

        # Forward pass: compute log probabilities under current policy
        log_probs_new = []
        log_probs_old = []

        for trajectory in valid_trajectories:
            # Recompute log probs (new policy)
            trajectory_log_probs = self._compute_log_probs(
                trajectory,
                require_grad=True
            )
            log_probs_new.append(trajectory_log_probs)

            # Use cached log probs (old policy)
            trajectory_log_probs_old = self._compute_log_probs(
                trajectory,
                require_grad=False
            )
            log_probs_old.append(trajectory_log_probs_old)

        # Create feedback masks
        masks = []
        for trajectory in valid_trajectories:
            traj_masks = [
                create_feedback_mask(turn, max_length=16384)
                for turn in trajectory['turns']
            ]
            masks.append(traj_masks)

        # Compute loss with all components
        advantages = compute_group_advantages(np.array(rewards))

        loss = 0.0
        for i in range(len(valid_trajectories)):
            # PPO loss with masking
            ratio = torch.exp(
                log_probs_new[i] - log_probs_old[i]
            )
            adv = advantages[i]
            unclipped = ratio * adv
            clipped = torch.clamp(
                ratio,
                1 - self.ppo_epsilon,
                1 + self.ppo_epsilon
            ) * adv

            trajectory_loss = -torch.mean(
                torch.minimum(unclipped, clipped)
            )
            loss += trajectory_loss

        loss = loss / len(valid_trajectories)

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )
        self.optimizer.step()

        return loss.item()

    def _extract_code_block(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        start = text.find("```python")
        if start == -1:
            return ""
        start += len("```python")
        end = text.find("```", start)
        if end == -1:
            return ""
        return text[start:end].strip()

    def _compute_log_probs(self, trajectory: dict,
                          require_grad: bool) -> torch.Tensor:
        """Compute log probabilities of trajectory tokens."""
        # In practice: use model.forward() to get logits,
        # compute log softmax, extract log probs for sampled tokens
        # This is pseudocode; real implementation requires tokenization details
        pass
```

## Practical Guidance

### Hyperparameters Reference

| Parameter | SimpleTIR Value | Notes |
|-----------|-----------------|-------|
| Learning Rate (Actor) | 1e-6 | Conservative; use Adam optimizer |
| PPO Clip Epsilon | 0.2–0.28 | 0.2 for stability, 0.28 for flexibility |
| Batch Size (Training) | 512 | Tokens per update; adjust for GPU memory |
| Batch Size (Sampling) | 1,280 | Trajectories collected per step |
| Max Response Length | 16,384 tokens | Large to allow multi-step reasoning |
| Max Interaction Turns | 5–10 | Depends on task complexity |
| Temperature | 1.0 | Keep high for exploration |
| Discount Factor (γ) | 1.0 | Episodic tasks (no bootstrapping) |
| GAE Lambda (λ) | 1.0 | Use MC return (full trajectory) |
| Gradient Clipping | 1.0 (global norm) | Critical for training stability |
| PPO Epochs | 4 | Reuse per batch before sampling new |
| Entropy Coefficient | 0 | Not used in SimpleTIR |
| KL Coefficient (β) | 0 | Not used (no reference model) |

### When to Use SimpleTIR

**Good fit:**
- Training LLMs for math problem solving (AIME, competition math)
- Multi-step reasoning with tool feedback (code execution, symbolic math systems)
- Base models without supervised pretraining (cost-effective)
- Scenarios where you want emergent reasoning patterns (self-correction, cross-validation)
- Projects with sufficient compute for end-to-end RL (~16–32 GPUs for 7B models)
- Tasks with clear correctness signals (mathematical reasoning, coding)

**Start with SimpleTIR if:** You want to avoid supervised fine-tuning cold starts and have well-defined reward signals. The method works best on benchmarks with binary or easily scored outcomes.

### When NOT to Use SimpleTIR

**Poor fit:**
- Tasks with ambiguous, subjective, or sparse rewards (creative writing, open-ended chat)
- Training on smaller models (<1B parameters) where distributional drift is less severe
- Scenarios requiring immediate deployment (RL training is inherently slow)
- Tasks lacking clear turn structure (e.g., single-turn generation)
- Problems where supervised fine-tuning is already working well and cost is not a concern
- Domains without reliable execution environments (no clear "tool feedback")
- Real-time systems needing low latency (expensive generation lengths)

**Avoid SimpleTIR if:** Your reward signal is noisy, sparse, or ill-defined. Distributional drift filtering only works when you can reliably detect generation failures (void turns). If your task doesn't have discrete "success/failure" outcomes, the method will either filter too much or too little.

### Common Pitfalls

1. **Insufficient void turn detection:** If your void turn detection is too permissive (allows partial code), filtering will not prevent gradient explosion. Be strict: require complete code blocks with balanced backticks and explicit final_answer() calls.

2. **Entropy decay:** The method produces zero entropy coefficient by design. If exploration collapses early, add small entropy bonus (0.01–0.05) or increase temperature during rollout.

3. **Reward signal design:** Binary rewards (correct/incorrect) work best. Scaled rewards (partial credit) can work but require careful normalization. Avoid continuous rewards without clear scale.

4. **Masking misalignment:** If feedback tokens are tokenized differently than expected, the mask will misalign with the loss computation, defeating the purpose. Verify that feedback masks match actual token boundaries.

5. **Turn count limits:** Five turns is typical for math but may be insufficient for harder problems. Monitor the proportion of trajectories hitting max turns; if >30%, increase max_turns.

6. **Memory scaling:** 16,384 token max length can consume significant GPU memory. If OOM occurs, reduce response length or batch size, but be aware this limits complex reasoning chains.

7. **Importance ratio explosion:** Even with void filtering, if old policy is very stale (many PPO epochs without refreshing), importance ratios can explode. Keep PPO epochs low (4) and refresh policy frequently.

### Practical Implementation Checklist

- [ ] Implement strict void turn detection (both code block and final_answer required)
- [ ] Verify reward signal is clear and well-defined
- [ ] Set up execution environment for tool feedback (code execution, symbolic math, etc.)
- [ ] Initialize model from base checkpoint (no SFT pretraining)
- [ ] Start with conservative learning rate (1e-6) and increase only if training stalls
- [ ] Monitor gradient norms during training; should be ~1-10, not 100+
- [ ] Log proportion of trajectories filtered each step (expect 10–40% for early training)
- [ ] Validate on held-out examples every N steps
- [ ] Use gradient clipping always (max norm = 1.0)
- [ ] Run small-scale experiments first (100 prompts) before full-scale training

## Reference

Paper: SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning

Authors: Zhenghai Xue, Longtao Zheng, Qian Liu, et al.

arXiv: https://arxiv.org/abs/2509.02479

Preprint released September 2025. License: CC-BY 4.0.

### Key Citation

```
@article{xue2025simpletir,
  title={SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning},
  author={Xue, Zhenghai and Zheng, Longtao and Liu, Qian and others},
  journal={arXiv preprint arXiv:2509.02479},
  year={2025}
}
```

### Related Work

- Group Relative Policy Optimization (GRPO): Baseline RL algorithm
- Tool-Integrated Reasoning (TIR): Multi-turn reasoning with external tools
- Distributional Shift in RL: Problem addressed by SimpleTIR's filtering approach
- Mathematical Reasoning Benchmarks: AIME24, Math500 used for evaluation
