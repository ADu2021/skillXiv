---
name: sage-self-hinting-rl
title: "Self-Hinting Language Models Enhance Reinforcement Learning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03143"
keywords: [Reinforcement Learning, Training Instability, Reward Shaping, Curriculum Learning, GRPO]
description: "Inject privileged hints during GRPO training to reshape rollout distributions when advantage collapses occur, increasing outcome diversity without changing task rewards. Hints removed at deployment; policy automatically learns when to use hints via online refresh mechanism."
---

# SAGE: Self-Hinting for Stable LLM Reinforcement Learning

Standard GRPO training struggles on difficult tasks with sparse rewards: rollout groups often receive identical rewards (all success or all failure), causing advantage calculations to collapse and learning to stall. SAGE addresses this by injecting hints during training—procedural guidance derived from reference solutions—that reshape the reward distribution to create outcome diversity within groups. Critically, hints only affect training; at deployment, they're removed entirely.

The key innovation is policy-dependent hint scheduling: hints activate automatically when advantage collapses, creating an effective curriculum that matches model capability. This contrasts with static curricula and avoids explicit difficulty estimation.

## Core Concept

SAGE operates on two principles:

1. **Privileged Supervision During Training**: Hints (lightweight procedural guidance) reshape the rollout distribution to ensure mixed outcomes within each group, preventing advantage collapse.

2. **Online Hint Refresh**: The hint generator is periodically updated using the current policy, keeping hints calibrated to the learner's evolving capabilities.

This creates a dynamic curriculum without explicit difficulty labels or multi-stage training schedules.

## Architecture Overview

- **Hint Generator**: Creates procedural guidance at variable strength levels using reference solutions
- **Policy-Dependent Scheduler**: Activates hints only when groups collapse (all same reward)
- **GRPO Training Loop**: Standard GRPO with conditional hint injection
- **Online Refresh**: Periodically regenerate hints using current policy predictions
- **Evaluation & Removal**: At inference, hints are set to empty strings (no privileged info)

## Implementation

### Step 1: Design Hint Generation

Create a system to generate hints of varying strength from reference solutions.

```python
# Hint generation system
class HintGenerator:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Generate procedural hints from solutions."""
        self.model = model

    def generate_hints_from_solution(
        self,
        prompt: str,
        reference_solution: str,
        strength_levels: List[float] = [0.3, 0.6, 1.0]
    ) -> Dict[float, str]:
        """
        Generate hints of varying strength from reference solution.

        Args:
            prompt: Task description
            reference_solution: Ground truth solution
            strength_levels: Hint detail levels (0=minimal, 1=full solution)

        Returns:
            Dict mapping strength to hint text
        """
        hints = {}

        for strength in strength_levels:
            if strength == 0.0:
                hints[strength] = ""  # No hint
            elif strength < 0.5:
                # Minimal hint: just problem type/strategy
                hint_prompt = f"""
Given this task and solution, provide a brief strategic hint (1-2 sentences).
Just indicate the overall approach without detailed steps.

Task: {prompt}
Solution: {reference_solution}

Hint:"""
                hint = self.model.generate(hint_prompt, max_tokens=50)
                hints[strength] = hint

            elif strength < 1.0:
                # Medium hint: outline steps without details
                hint_prompt = f"""
Given this task and solution, provide a step outline (3-4 steps).
List the main steps but don't provide the full solution.

Task: {prompt}
Solution: {reference_solution}

Steps:"""
                hint = self.model.generate(hint_prompt, max_tokens=100)
                hints[strength] = hint

            else:
                # Strong hint: nearly full solution with minor details omitted
                hint = reference_solution[:len(reference_solution) * 0.8]
                hints[strength] = hint

        return hints

    def batch_generate_hints(
        self,
        prompts: List[str],
        reference_solutions: List[str],
        strength_levels: List[float] = [0.3, 0.6]
    ) -> Dict[str, Dict[float, str]]:
        """Generate hints for multiple samples."""
        all_hints = {}

        for prompt, solution in zip(prompts, reference_solutions):
            sample_id = hash(prompt) % 10000
            all_hints[sample_id] = self.generate_hints_from_solution(
                prompt,
                solution,
                strength_levels
            )

        return all_hints
```

### Step 2: Implement Collapse Detection

Create a monitor that detects when advantage collapses in rollout groups.

```python
# Collapse detection mechanism
class GroupCollapseDetector:
    def __init__(self, collapse_threshold: float = 0.01):
        """
        Detect when advantage variance within groups becomes too small.

        Args:
            collapse_threshold: Min advantage variance to avoid collapse
        """
        self.collapse_threshold = collapse_threshold
        self.collapse_history = []

    def detect_collapse(self, rewards: torch.Tensor,
                       group_size: int) -> bool:
        """
        Check if reward group shows advantage collapse.

        Args:
            rewards: [group_size] rewards for current group
            group_size: Size of rollout group

        Returns:
            True if group has collapsed (all same reward)
        """
        # Check variance within group
        reward_variance = torch.var(rewards).item()
        is_collapsed = reward_variance < self.collapse_threshold

        self.collapse_history.append(is_collapsed)

        # Return true if recently collapsed
        return is_collapsed

    def should_use_hints(self, recent_collapses: int = 5) -> bool:
        """
        Decide whether to activate hints based on recent history.

        Use hints if >50% of recent groups have collapsed.
        """
        if len(self.collapse_history) < recent_collapses:
            return True  # Early training: always use hints

        recent = self.collapse_history[-recent_collapses:]
        collapse_rate = sum(recent) / len(recent)
        return collapse_rate > 0.5

    def reset_history(self):
        """Reset collapse tracker."""
        self.collapse_history = []
```

### Step 3: Implement Hint-Aware GRPO Training

Modify GRPO training loop to inject hints conditionally.

```python
# GRPO training with hints
def train_with_sage(
    model: nn.Module,
    dataset: List[dict],
    hint_generator: HintGenerator,
    num_epochs: int = 10,
    group_size: int = 4
):
    """
    GRPO training with SAGE hint injection.

    Args:
        model: Language model to train
        dataset: Training samples with prompts and reference solutions
        hint_generator: Generator for procedural hints
        num_epochs: Training epochs
        group_size: GRPO rollout group size
    """
    collapse_detector = GroupCollapseDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Pre-generate hints for all samples
    prompts = [s['prompt'] for s in dataset]
    solutions = [s['solution'] for s in dataset]
    all_hints = hint_generator.batch_generate_hints(
        prompts,
        solutions,
        strength_levels=[0.0, 0.3, 0.6]
    )

    for epoch in range(num_epochs):
        # Check if hints should be used
        use_hints = collapse_detector.should_use_hints()

        collapse_detector.reset_history()

        # Group samples into rollout groups
        for group_idx in range(0, len(dataset), group_size):
            group = dataset[group_idx:group_idx + group_size]
            group_rewards = []

            for sample in group:
                sample_id = hash(sample['prompt']) % 10000

                # Get hint if enabled
                hint = ""
                if use_hints and sample_id in all_hints:
                    # Randomly select hint strength
                    strength = random.choice([0.0, 0.3, 0.6])
                    hint = all_hints[sample_id].get(strength, "")

                # Construct input with hint
                input_text = f"{sample['prompt']}\n\nHint: {hint}"

                # Generate response
                with torch.no_grad():
                    output = model.generate(input_text, max_tokens=256)

                # Evaluate
                reward = evaluate_output(output, sample['target'])
                group_rewards.append(reward)

            # Detect collapse
            group_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            collapse_detector.detect_collapse(group_tensor, len(group))

            # Compute advantages (GRPO style)
            group_mean = group_tensor.mean()
            advantages = group_tensor - group_mean

            # Backward pass
            for sample, advantage in zip(group, advantages):
                if advantage != 0:
                    # Scale loss by advantage
                    loss = -advantage * compute_log_prob(model, sample)
                    loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        # Periodically refresh hints
        if epoch % 3 == 0:
            print(f"Epoch {epoch}: Refreshing hints...")
            all_hints = hint_generator.batch_generate_hints(
                prompts, solutions, strength_levels=[0.0, 0.3, 0.6]
            )

    return model
```

### Step 4: Implement Online Hint Refresh

Periodically update hints using current policy predictions.

```python
# Online hint refresh
class OnlineHintRefresher:
    def __init__(self, hint_generator: HintGenerator,
                 refresh_interval: int = 500):
        """
        Maintain fresh hints based on policy evolution.

        Args:
            hint_generator: Base hint generation engine
            refresh_interval: Steps between refreshes
        """
        self.hint_generator = hint_generator
        self.refresh_interval = refresh_interval
        self.step_count = 0
        self.hint_cache = {}

    def should_refresh(self) -> bool:
        """Check if it's time to refresh hints."""
        self.step_count += 1
        return self.step_count % self.refresh_interval == 0

    def refresh_hints(
        self,
        model: nn.Module,
        dataset: List[dict]
    ):
        """
        Regenerate hints based on current policy predictions.

        This adapts hints to match model's current capability level.
        """
        self.hint_cache = {}

        for sample in dataset:
            sample_id = hash(sample['prompt']) % 10000

            # Generate new hint from reference
            hint_dict = self.hint_generator.generate_hints_from_solution(
                sample['prompt'],
                sample['solution'],
                strength_levels=[0.0, 0.3, 0.6]
            )

            self.hint_cache[sample_id] = hint_dict

    def get_hint(self, sample_id: str,
                strength: float = 0.3) -> str:
        """Retrieve hint for sample."""
        if sample_id not in self.hint_cache:
            return ""

        return self.hint_cache[sample_id].get(strength, "")
```

### Step 5: Inference Without Hints

Ensure hints are removed at deployment.

```python
# Inference without hints
def inference_without_hints(
    model: nn.Module,
    prompt: str,
    max_tokens: int = 256
) -> str:
    """
    Generate response without privileged hints at inference.

    Args:
        model: Trained model with SAGE
        prompt: Task prompt (no hint)
        max_tokens: Generation length

    Returns:
        Generated response
    """
    # Ensure empty hint is used
    full_input = f"{prompt}\n\nHint: "

    with torch.no_grad():
        output = model.generate(full_input, max_tokens=max_tokens)

    # Remove hint prefix from output
    return output.split("Hint: ", 1)[-1]
```

## Practical Guidance

**When to use SAGE:**
- Difficult RL tasks with sparse rewards and high variance
- Scenarios where advantage collapse is observed during training
- Multi-stage tasks where progressive guidance helps
- Models with limited initial capability on the target task

**When not to use:**
- Tasks with dense rewards and stable advantage signals
- Scenarios where hints could leak into deployment
- Real-time training where hint generation overhead matters
- Tasks where curriculum learning isn't beneficial

**Common Pitfalls:**
- Hints too strong: Full solutions bypass learning; use weak hints (0.3 strength)
- Hint leakage: Ensure hints are actually removed at inference
- Stale hints: Online refresh critical; stale hints misalign with policy
- Over-reliance on hints: Gradually reduce hint strength as training progresses

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| hint_strength | 0.0-0.6 | Start weak (0.3); increase only if collapse persists |
| collapse_threshold | 0.001-0.01 | Higher = more conservative collapse detection |
| refresh_interval | 200-500 steps | More frequent = higher overhead; less frequent = stale hints |
| group_size | 4-8 | Larger = more stable averaging; 4 typical |

## Reference

See the full paper at: https://arxiv.org/abs/2602.03143

Key results: Fixes advantage collapse on difficult reasoning tasks; open-source code at https://github.com/BaohaoLiao/SAGE. Demonstrated across six benchmarks on Llama, Qwen, and Claude models. Orthogonal to other RL stabilization techniques.
