---
name: curriculum-efficient-reasoning
title: Train Long Think Short - Curriculum Learning for Efficient Reasoning
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.08940
keywords: [curriculum-learning, efficient-reasoning, token-budget, policy-optimization]
description: "Improves reasoning efficiency through curriculum learning that progressively constrains token budgets, enabling models to first discover solution strategies then distill them into concise traces."
---

## Train Long Think Short: Curriculum Learning for Efficient Reasoning

### Core Concept

Train Long Think Short addresses the challenge of training efficient reasoning models by using curriculum learning to progressively tighten token budgets during training. Rather than using fixed-length constraints from the start, models begin with generous budgets to discover effective solution strategies, then gradually reduce budgets to compress reasoning into more efficient traces. This provides a powerful inductive bias for learning length-controlled reasoning.

### Architecture Overview

- **Progressive Budget Constraints**: Start loose, gradually tighten token limits
- **Multi-Signal Reward Function**: Balance correctness, efficiency, and formatting
- **Group Relative Policy Optimization**: RL algorithm for constrained training
- **Curriculum Phases**: Exploration phase, compression phase, optimization phase
- **Adaptive Difficulty**: Adjust constraint schedule based on model performance

### Implementation Steps

**Step 1: Design Progressive Token Budget Schedule**

Create curriculum for token constraints:

```python
# Pseudocode for curriculum budget scheduling
class CurriculumBudgetScheduler:
    def __init__(self, initial_budget=2000, final_budget=200, num_phases=10):
        super().__init__()
        self.initial_budget = initial_budget
        self.final_budget = final_budget
        self.num_phases = num_phases
        self.current_phase = 0

    def get_budget_for_phase(self, phase):
        """
        Get token budget for training phase.

        Args:
            phase: Current training phase (0 to num_phases-1)

        Returns:
            budget: Token limit for this phase
        """
        # Exponential decay schedule
        decay_rate = (self.final_budget / self.initial_budget) ** (1.0 / self.num_phases)
        budget = self.initial_budget * (decay_rate ** phase)

        return int(budget)

    def get_current_budget(self):
        """
        Get current phase budget.
        """
        return self.get_budget_for_phase(self.current_phase)

    def advance_phase(self):
        """
        Move to next curriculum phase.
        """
        if self.current_phase < self.num_phases - 1:
            self.current_phase += 1
            return True
        return False

    def visualize_schedule(self):
        """
        Show budget progression.
        """
        budgets = [self.get_budget_for_phase(p) for p in range(self.num_phases)]
        return {
            'phase_budgets': budgets,
            'total_phases': self.num_phases,
            'initial': self.initial_budget,
            'final': self.final_budget
        }

    def adaptive_schedule(self, performance_history):
        """
        Adapt schedule based on performance.

        Args:
            performance_history: Recent accuracy scores

        Returns:
            should_advance: Whether to move to next phase
        """
        if len(performance_history) < 5:
            return False

        recent_perf = performance_history[-5:]
        stability = np.std(recent_perf)

        # Advance if performance is stable
        if stability < 0.05 and np.mean(recent_perf) > 0.85:
            return True

        return False
```

**Step 2: Implement Multi-Signal Reward Function**

Design comprehensive reward for constrained optimization:

```python
# Pseudocode for multi-signal rewards
class MultiSignalRewardFunction:
    def __init__(self, verifier_model):
        super().__init__()
        self.verifier = verifier_model

    def compute_reward(self, generated_reasoning, target_answer, token_budget, current_tokens):
        """
        Compute multi-component reward signal.

        Args:
            generated_reasoning: Model-generated reasoning trace
            target_answer: Ground truth answer
            token_budget: Maximum allowed tokens for this phase
            current_tokens: Tokens used in generation

        Returns:
            reward: Combined reward signal
        """
        # Component 1: Task correctness via verifier
        correctness_reward = self._compute_correctness(generated_reasoning, target_answer)

        # Component 2: Length efficiency
        efficiency_reward = self._compute_efficiency(current_tokens, token_budget)

        # Component 3: Formatting/structure adherence
        format_reward = self._compute_format_score(generated_reasoning)

        # Component 4: Reasoning quality (semantic coherence)
        quality_reward = self._compute_reasoning_quality(generated_reasoning)

        # Combine with dynamic weights (shift toward efficiency as phases progress)
        correctness_weight = 0.6
        efficiency_weight = 0.2
        format_weight = 0.1
        quality_weight = 0.1

        total_reward = (
            correctness_weight * correctness_reward +
            efficiency_weight * efficiency_reward +
            format_weight * format_reward +
            quality_weight * quality_reward
        )

        return total_reward

    def _compute_correctness(self, reasoning, target):
        """
        Verify answer correctness.
        """
        extracted_answer = self._extract_answer(reasoning)

        with torch.no_grad():
            is_correct = self.verifier.verify(extracted_answer, target)

        return 1.0 if is_correct else 0.0

    def _compute_efficiency(self, used_tokens, budget):
        """
        Reward for staying within token budget.
        """
        if used_tokens > budget:
            # Penalize budget overrun
            return -0.5

        # Bonus for using less than budget
        utilization = used_tokens / budget
        return 1.0 - (0.5 * utilization)  # Max reward at 0% budget, still positive at 100%

    def _compute_format_score(self, text):
        """
        Check formatting (steps, structure, etc).
        """
        # Count step markers (1., 2., etc)
        steps = len([l for l in text.split('\n') if l and l[0].isdigit()])

        # Count [SKIP] tokens (bonus for compression awareness)
        skip_count = text.count('[SKIP]')

        format_score = min(steps / 5.0, 1.0) + 0.1 * skip_count
        return min(format_score, 1.0)

    def _compute_reasoning_quality(self, reasoning):
        """
        Assess semantic coherence of reasoning.
        """
        sentences = [s.strip() for s in reasoning.split('.') if s.strip()]

        if len(sentences) < 2:
            return 0.3

        # Simple coherence: adjacent sentences should share terms
        shared_term_count = 0
        for i in range(len(sentences) - 1):
            terms1 = set(sentences[i].lower().split())
            terms2 = set(sentences[i + 1].lower().split())
            if terms1 & terms2:
                shared_term_count += 1

        coherence = shared_term_count / (len(sentences) - 1)
        return min(coherence, 1.0)

    def _extract_answer(self, reasoning):
        """
        Extract final answer from reasoning.
        """
        # Look for answer section
        import re
        match = re.search(r'answer[:\s]*(.+?)(?:\n|$)', reasoning, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return reasoning.split('\n')[-1]
```

**Step 3: Implement Curriculum-Based GRPO Training**

Train with group relative policy optimization:

```python
# Pseudocode for curriculum GRPO
class CurriculumGRPOTrainer:
    def __init__(self, model, reward_fn, scheduler):
        super().__init__()
        self.model = model
        self.reward_fn = reward_fn
        self.scheduler = scheduler

    def train_phase(self, training_data, phase_steps=1000):
        """
        Train for single curriculum phase.

        Args:
            training_data: Training examples
            phase_steps: Steps for this phase

        Returns:
            phase_stats: Training statistics
        """
        current_budget = self.scheduler.get_current_budget()
        optimizer = AdamW(self.model.parameters(), lr=5e-6)

        phase_stats = {
            'budget': current_budget,
            'step_losses': [],
            'step_rewards': [],
            'step_lengths': []
        }

        for step in range(phase_steps):
            batch = self._sample_batch(training_data)

            # Generate multiple reasoning traces (group)
            group_size = 4
            reasoning_group = []
            reward_group = []

            for question, target in batch:
                traces = []
                rewards = []

                for _ in range(group_size):
                    trace = self.model.generate(
                        question,
                        max_tokens=current_budget,
                        temperature=0.8
                    )

                    reward = self.reward_fn.compute_reward(
                        trace,
                        target,
                        current_budget,
                        len(trace.split())
                    )

                    traces.append(trace)
                    rewards.append(reward)

                reasoning_group.append(traces)
                reward_group.append(rewards)

            # GRPO: group relative policy optimization
            loss = self._compute_grpo_loss(reasoning_group, reward_group)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            phase_stats['step_losses'].append(loss.item())
            phase_stats['step_rewards'].append(np.mean([r for rg in reward_group for r in rg]))
            phase_stats['step_lengths'].append(np.mean([len(t.split()) for tg in reasoning_group for t in tg]))

        return phase_stats

    def _compute_grpo_loss(self, reasoning_group, reward_group):
        """
        Compute GRPO loss (group-relative).
        """
        total_loss = 0

        for traces, rewards in zip(reasoning_group, reward_group):
            # Rank rewards within group
            reward_ranks = np.argsort(rewards)

            # Compute log probs
            for idx, (trace, rank) in enumerate(zip(traces, reward_ranks)):
                input_ids = self.model.tokenizer(trace, return_tensors='pt')['input_ids']
                outputs = self.model(input_ids)
                log_prob = -outputs.loss  # Approximate

                # GRPO: optimize based on relative rank
                relative_reward = (rank - (len(traces) - 1) / 2) / len(traces)
                loss_step = -log_prob * relative_reward

                total_loss = total_loss + loss_step

        return total_loss / (len(reasoning_group) * len(reasoning_group[0]))

    def _sample_batch(self, training_data, batch_size=8):
        """
        Sample batch of training examples.
        """
        indices = np.random.choice(len(training_data), batch_size)
        return [training_data[i] for i in indices]

    def full_curriculum_training(self, training_data, steps_per_phase=1000):
        """
        Run full curriculum training.

        Args:
            training_data: All training examples
            steps_per_phase: Steps per curriculum phase

        Returns:
            all_stats: Statistics for all phases
        """
        all_stats = []

        for phase in range(self.scheduler.num_phases):
            print(f"Phase {phase+1}: Budget = {self.scheduler.get_current_budget()}")

            stats = self.train_phase(training_data, steps_per_phase)
            all_stats.append(stats)

            # Check if ready to advance
            if self.scheduler.adaptive_schedule(stats['step_rewards']):
                self.scheduler.advance_phase()
            else:
                # Fixed schedule
                self.scheduler.advance_phase()

        return all_stats
```

**Step 4: Evaluate Length-Controlled Reasoning**

Test efficiency and accuracy tradeoff:

```python
# Pseudocode for evaluation
class LengthControlledReasoningEvaluator:
    def __init__(self, model):
        super().__init__()
        self.model = model

    def evaluate_at_budget(self, test_examples, budget):
        """
        Evaluate model at specific token budget.

        Args:
            test_examples: Test questions with answers
            budget: Token limit

        Returns:
            metrics: Accuracy and efficiency metrics
        """
        correct = 0
        total_tokens = 0

        for question, target_answer in test_examples:
            generated = self.model.generate(
                question,
                max_tokens=budget,
                temperature=0.1
            )

            extracted = self._extract_answer(generated)
            is_correct = self._verify_answer(extracted, target_answer)

            if is_correct:
                correct += 1

            total_tokens += len(generated.split())

        accuracy = correct / len(test_examples)
        avg_length = total_tokens / len(test_examples)

        return {
            'accuracy': accuracy,
            'avg_length': avg_length,
            'budget': budget,
            'efficiency': (1 - avg_length / budget) * 100  # Efficiency percentage
        }

    def evaluate_curriculum_progression(self, test_examples, budgets):
        """
        Evaluate model at different budget levels.

        Returns:
            curves: Accuracy vs efficiency curve
        """
        results = []

        for budget in budgets:
            metrics = self.evaluate_at_budget(test_examples, budget)
            results.append(metrics)

        return results

    def _extract_answer(self, text):
        """
        Extract final answer from text.
        """
        lines = text.strip().split('\n')
        return lines[-1] if lines else ''

    def _verify_answer(self, generated, target):
        """
        Check if answer is correct.
        """
        return generated.lower().strip() == target.lower().strip()
```

### Practical Guidance

**Hyperparameters and Configuration**:
- Initial budget: 2000 tokens (generous discovery)
- Final budget: 200-400 tokens (efficient reasoning)
- Number of curriculum phases: 8-12
- Group size for GRPO: 4-8 traces
- Learning rate: 5e-6 to 1e-5
- Phase steps: 500-2000 depending on data size

**When to Use Curriculum Learning for Reasoning**:
- Training models for length-controlled reasoning tasks
- Scenarios where both accuracy and efficiency matter
- Mathematical or algorithmic reasoning requiring exploration then compression
- Systems with variable computational budgets

**When NOT to Use**:
- Single-budget inference scenarios (fixed token limits)
- Tasks where reasoning naturally short
- Very large models (training overhead significant)
- When maximum accuracy is only concern

**Implementation Notes**:
- Progressive constraint provides powerful inductive bias
- Multi-signal rewards crucial for balancing competing objectives
- GRPO's group-relative optimization prevents distribution collapse
- Adaptive scheduling helps models learn faster
- Monitor both accuracy curves and efficiency gains per phase

### Reference

Paper: Train Long Think Short: Curriculum Learning for Efficient Reasoning
ArXiv: 2508.08940
Performance: Curriculum-based training consistently outperforms fixed-budget baselines on mathematical reasoning datasets (GSM8K, MATH500, SVAMP)
